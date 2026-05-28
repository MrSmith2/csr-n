"""
CSR-N: prefix-tree structure for sparse N-dimensional tensors.

Storage layout - flat arrays, generalization of scipy CSR to N dims:

    indptr        : int64, shape (n_nodes + 1,)
        indptr[i]:indptr[i+1] = slice in idx for node i children.
    base          : int64, shape (n_nodes,)
        For non-leaf node i: global id of its first child.
        For leaf node i   : position in values of its first leaf value.
    idx           : int64, shape (total degree,)
        Sorted child coordinate values, laid out level by level using BFS.
    values        : float64, shape (nnz,)
        Non-zero values in lexicographic order
    level_offsets : int64, shape (N + 1,)
        level_offsets[k] = global id of the first node at level k.
"""

import numpy as np
import scipy.sparse as _sp_sparse


class CSRN:

    def __init__(self, shape, indptr, base, idx, values, level_offsets):
        self.shape = tuple(shape)
        self.indptr = indptr
        self.base = base
        self.idx = idx
        self.values = values
        self.level_offsets = level_offsets
        self._coords_cache = None
        self._use_cache = True

    @classmethod
    def from_coo(cls, coords, values, shape):
        """Build a CSR-N from a coordinate list (COO)."""
        coords = np.asarray(coords, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        N = len(shape)
        nnz = values.shape[0]

        if coords.shape != (N, nnz):
            raise ValueError(
                f"coords shape {coords.shape} doesn't match (N={N}, nnz={nnz})"
            )

        if nnz == 0:
            indptr = np.zeros(2, dtype=np.int64)
            base = np.zeros(1, dtype=np.int64)
            idx = np.zeros(0, dtype=np.int64)
            level_offsets = np.ones(N + 1, dtype=np.int64)
            level_offsets[0] = 0
            return cls(shape, indptr, base, idx, values, level_offsets)

        order = np.lexsort(coords[::-1])
        coords = coords[:, order]
        values = values[order]

        if nnz == 1:
            first_diff = np.zeros(0, dtype=np.int64)
        else:
            diffs = coords[:, 1:] != coords[:, :-1]
            if not diffs.any(axis=0).all():
                raise ValueError("Duplicate coordinates detected.")
            first_diff = diffs.argmax(axis=0).astype(np.int64)

        sizes = np.empty(N, dtype=np.int64)
        sizes[0] = 1
        for k in range(1, N):
            sizes[k] = 1 + int(np.sum(first_diff < k))

        level_offsets = np.zeros(N + 1, dtype=np.int64)
        level_offsets[1:] = np.cumsum(sizes)
        n_nodes = int(level_offsets[-1])

        idx_chunks, degree_chunks, base_chunks = [], [], []

        for k in range(N):
            mask = np.empty(nnz, dtype=bool)
            mask[0] = True
            if nnz > 1:
                mask[1:] = first_diff <= k
            idx_chunk = coords[k, mask]

            if k == 0:
                parent_id_at_mask = np.zeros(int(mask.sum()), dtype=np.int64)
            else:
                parent_id = np.empty(nnz, dtype=np.int64)
                parent_id[0] = 0
                if nnz > 1:
                    parent_id[1:] = np.cumsum((first_diff < k).astype(np.int64))
                parent_id_at_mask = parent_id[mask]

            degree = np.bincount(parent_id_at_mask, minlength=int(sizes[k]))

            base_k = np.empty(int(sizes[k]), dtype=np.int64)
            base_k[0] = 0
            if int(sizes[k]) > 1:
                base_k[1:] = np.cumsum(degree[:-1])
            if k < N - 1:
                base_k = base_k + int(level_offsets[k + 1])

            idx_chunks.append(idx_chunk)
            degree_chunks.append(degree)
            base_chunks.append(base_k)

        idx_out = np.concatenate(idx_chunks)
        all_degrees = np.concatenate(degree_chunks)

        indptr = np.empty(n_nodes + 1, dtype=np.int64)
        indptr[0] = 0
        indptr[1:] = np.cumsum(all_degrees)
        base_out = np.concatenate(base_chunks)

        return cls(shape, indptr, base_out, idx_out, values.copy(), level_offsets)

    @property
    def n_nodes(self):
        return int(self.level_offsets[-1])

    @property
    def nnz(self):
        return int(self.values.shape[0])

    def memory_bytes(self):
        """Total memory footprint of storage arrays in bytes."""
        return int(
            self.indptr.nbytes + self.base.nbytes + self.idx.nbytes
            + self.values.nbytes + self.level_offsets.nbytes
        )

    def __getitem__(self, index):
        """Find a value by multi-index. Returns 0.0 if none."""
        if not isinstance(index, tuple):
            index = (index,)
        N = len(self.shape)
        if len(index) != N:
            raise IndexError(f"expected {N} indices, got {len(index)}")

        node_id = 0
        for d in range(N - 1):
            first = int(self.indptr[node_id])
            last = int(self.indptr[node_id + 1])
            if first == last:
                return 0.0
            arr = self.idx[first:last]
            pos = int(np.searchsorted(arr, index[d]))
            if pos >= len(arr) or arr[pos] != index[d]:
                return 0.0
            node_id = int(self.base[node_id]) + pos

        first = int(self.indptr[node_id])
        last = int(self.indptr[node_id + 1])
        if first == last:
            return 0.0
        arr = self.idx[first:last]
        pos = int(np.searchsorted(arr, index[-1]))
        if pos >= len(arr) or arr[pos] != index[-1]:
            return 0.0
        return float(self.values[int(self.base[node_id]) + pos])

    def _compute_subtree_sizes(self):
        """For every node: number of leaf values reachable."""
        N = len(self.shape)
        subtree = np.empty(self.n_nodes, dtype=np.int64)

        leaf_start = int(self.level_offsets[N - 1])
        subtree[leaf_start:] = np.diff(self.indptr[leaf_start:])

        for k in range(N - 2, -1, -1):
            p_start = int(self.level_offsets[k])
            p_end = int(self.level_offsets[k + 1])
            c_start = int(self.level_offsets[k + 1])
            c_end = int(self.level_offsets[k + 2])

            bounds = (self.base[p_start:p_end] - c_start).astype(np.intp)
            subtree[p_start:p_end] = np.add.reduceat(
                subtree[c_start:c_end], bounds
            )

        return subtree

    def _reconstruct_coords(self):
        """Reconstruct the (N, nnz) sorted coordinate matrix."""
        if self._use_cache and self._coords_cache is not None:
            return self._coords_cache

        N = len(self.shape)
        nnz = self.nnz
        subtree = self._compute_subtree_sizes()

        coords = np.empty((N, nnz), dtype=np.int64)
        for k in range(N):
            level_start = int(self.level_offsets[k])
            level_end = int(self.level_offsets[k + 1])
            chunk_start = int(self.indptr[level_start])
            chunk_end = int(self.indptr[level_end])
            chunk = self.idx[chunk_start:chunk_end]

            if k < N - 1:
                child_start = level_end
                weights = subtree[child_start:child_start + len(chunk)]
                coords[k] = np.repeat(chunk, weights)
            else:
                coords[k] = chunk

        if self._use_cache:
            self._coords_cache = coords
        return coords

    def to_dense(self):
        """Materialize the dense (shape) tensor."""
        out = np.zeros(self.shape, dtype=np.float64)
        if self.nnz > 0:
            coords = self._reconstruct_coords()
            out[tuple(coords)] = self.values
        return out

    def add(self, Y):
        """Element-wise addition: Z[i] = X[i] + Y[i]."""
        if not isinstance(Y, CSRN):
            raise TypeError(f"Y must be CSRN, got {type(Y).__name__}")
        if self.shape != Y.shape:
            raise ValueError(f"shape mismatch: {self.shape} != {Y.shape}")

        if self.nnz == 0 and Y.nnz == 0:
            return CSRN.from_coo(
                np.zeros((len(self.shape), 0), dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                self.shape,
            )

        if self.nnz == 0:
            return CSRN.from_coo(
                Y._reconstruct_coords(), Y.values.copy(), self.shape
            )
        if Y.nnz == 0:
            return CSRN.from_coo(
                self._reconstruct_coords(), self.values.copy(), self.shape
            )

        coords_x = self._reconstruct_coords()
        coords_y = Y._reconstruct_coords()
        combined_coords = np.hstack([coords_x, coords_y])
        combined_values = np.concatenate([self.values, Y.values])

        flat = np.ravel_multi_index(combined_coords, self.shape)
        order = np.argsort(flat, kind='stable')
        flat_sorted = flat[order]
        values_sorted = combined_values[order]

        total = len(flat_sorted)
        new_group = np.empty(total, dtype=bool)
        new_group[0] = True
        new_group[1:] = flat_sorted[1:] != flat_sorted[:-1]
        group_starts = np.where(new_group)[0]
        agg = np.add.reduceat(values_sorted, group_starts)
        unique_flat = flat_sorted[group_starts]

        nz_mask = agg != 0.0
        if not nz_mask.any():
            return CSRN.from_coo(
                np.zeros((len(self.shape), 0), dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                self.shape,
            )
        final_coords = np.array(np.unravel_index(unique_flat[nz_mask], self.shape))
        final_values = agg[nz_mask]
        return CSRN.from_coo(final_coords, final_values, self.shape)

    def __add__(self, other):
        return self.add(other)

    def scale(self, scalar):
        """Element-wise scalar multiplication: X * c."""
        scalar = float(scalar)
        if scalar == 0.0:
            return CSRN.from_coo(
                np.zeros((len(self.shape), 0), dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                self.shape,
            )
        result = CSRN(
            self.shape, self.indptr, self.base, self.idx,
            self.values * scalar, self.level_offsets,
        )

        result._coords_cache = self._coords_cache
        result._use_cache = self._use_cache
        return result

    def ttv(self, v, mode=0):
        """Tensor times vector along  given mode."""
        v = np.asarray(v, dtype=np.float64)
        N = len(self.shape)
        if not (0 <= mode < N):
            raise ValueError(f"mode {mode} out of range [0, {N})")
        if v.shape[0] != self.shape[mode]:
            raise ValueError(
                f"v.shape[0]={v.shape[0]} != tensor.shape[{mode}]={self.shape[mode]}"
            )

        if N == 1:
            if self.nnz == 0:
                return 0.0
            coords = self._reconstruct_coords()
            return float(np.sum(v[coords[0]] * self.values))

        out_shape = tuple(s for k, s in enumerate(self.shape) if k != mode)
        if self.nnz == 0:
            return np.zeros(out_shape, dtype=np.float64)

        coords = self._reconstruct_coords()
        weighted = v[coords[mode]] * self.values
        other_coords = tuple(coords[k] for k in range(N) if k != mode)
        flat_idx = np.ravel_multi_index(other_coords, out_shape)
        out_flat = np.bincount(
            flat_idx, weights=weighted, minlength=int(np.prod(out_shape))
        )
        return out_flat.reshape(out_shape)

    def ttv_last_native(self, v, return_sparse=False): 
        # TODO: docstring, ValuError, filter explicit zeros
        v = np.asarray(v, dtype=np.float64)
        N = len(self.shape)
        out_shape = self.shape[:-1]
        leaf_start = int(self.level_offsets[N - 1])
        leaf_end = int(self.level_offsets[N])

        if self.nnz == 0:
            if N == 1:
                return 0.0
            out = CSRN.from_coo(np.zeros((N - 1, 0), dtype=np.int64),
                                np.zeros(0, dtype=np.float64), out_shape)
            return out if return_sparse else np.zeros(out_shape, dtype=np.float64)

        last_axis_coords = self.idx[int(self.indptr[leaf_start]):int(self.indptr[leaf_end])]
        weighted = v[last_axis_coords] * self.values
        starts = self.indptr[leaf_start:leaf_end].astype(np.intp) - int(self.indptr[leaf_start])
        reduced = np.add.reduceat(weighted, starts)

        if N == 1:
            return float(reduced.sum())

        out_indptr = self.indptr[:leaf_start + 1].copy()
        out_idx = self.idx[:int(self.indptr[leaf_start])].copy()
        out_base = self.base[:leaf_start].copy()
        out_base[int(self.level_offsets[N - 2]):leaf_start] -= leaf_start
        out = CSRN(out_shape, out_indptr, out_base, out_idx, reduced, self.level_offsets[:N].copy())

        return out if return_sparse else out.to_dense()

    def ttm(self, A, mode=0):
        """Tensor times matrix along  given mode with dense or sparse A."""
        N = len(self.shape)
        if not (0 <= mode < N):
            raise ValueError(f"mode {mode} out of range [0, {N})")

        if _sp_sparse.issparse(A):
            return self._ttm_sparse_a(A.tocsc(), mode)

        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        J, d_mode = A.shape
        if d_mode != self.shape[mode]:
            raise ValueError(
                f"A.shape[1]={d_mode} != tensor.shape[{mode}]={self.shape[mode]}"
            )

        out_shape = list(self.shape)
        out_shape[mode] = J
        out_shape = tuple(out_shape)

        if self.nnz == 0:
            return np.zeros(out_shape, dtype=np.float64)

        coords = self._reconstruct_coords()
        nnz = self.nnz

        weighted = A[:, coords[mode]] * self.values

        j_axis = np.arange(J, dtype=np.int64)
        ravel_arrays = []
        for k in range(N):
            if k == mode:
                ravel_arrays.append(np.broadcast_to(j_axis[:, None], (J, nnz)))
            else:
                ravel_arrays.append(np.broadcast_to(coords[k][None, :], (J, nnz)))

        flat_idx = np.ravel_multi_index(ravel_arrays, out_shape)

        out_flat = np.bincount(
            flat_idx.ravel(),
            weights=weighted.ravel(),
            minlength=int(np.prod(out_shape)),
        )
        return out_flat.reshape(out_shape)

    def _ttm_sparse_a(self, A_csc, mode):
        """TTM where A is a scipy.sparse CSC matrix, iterates only over nnz_A."""
        N = len(self.shape)
        J = A_csc.shape[0]
        I = A_csc.shape[1]
        if I != self.shape[mode]:
            raise ValueError(
                f"A.shape[1]={I} != tensor.shape[{mode}]={self.shape[mode]}"
            )

        out_shape = list(self.shape)
        out_shape[mode] = J
        out_shape = tuple(out_shape)

        if self.nnz == 0 or A_csc.nnz == 0:
            return np.zeros(out_shape, dtype=np.float64)

        coords = self._reconstruct_coords()
        x_mode = coords[mode]

        col_starts = A_csc.indptr.astype(np.int64)
        row_idx = A_csc.indices.astype(np.int64)
        A_data = A_csc.data.astype(np.float64)

        nnz_per_x = col_starts[x_mode + 1] - col_starts[x_mode]
        total_pairs = int(nnz_per_x.sum())
        if total_pairs == 0:
            return np.zeros(out_shape, dtype=np.float64)

        x_idx = np.repeat(np.arange(self.nnz, dtype=np.int64), nnz_per_x)

        starts_per_x = col_starts[x_mode]
        starts_cum = np.empty(self.nnz, dtype=np.int64)
        starts_cum[0] = 0
        if self.nnz > 1:
            starts_cum[1:] = np.cumsum(nnz_per_x[:-1])
        within = (np.arange(total_pairs, dtype=np.int64)
                  - np.repeat(starts_cum, nnz_per_x))
        a_positions = np.repeat(starts_per_x, nnz_per_x) + within

        j_array = row_idx[a_positions]
        contributions = self.values[x_idx] * A_data[a_positions]

        out_index_arrays = []
        for k in range(N):
            if k == mode:
                out_index_arrays.append(j_array)
            else:
                out_index_arrays.append(coords[k, x_idx])
        flat_idx = np.ravel_multi_index(out_index_arrays, out_shape)

        out_flat = np.bincount(
            flat_idx, weights=contributions,
            minlength=int(np.prod(out_shape)),
        )
        return out_flat.reshape(out_shape)

    def ttt(self, Y, mode_x=0, mode_y=0):
        """Sparse-sparse tensor-times-tensor with single-axis contraction. """
        if not isinstance(Y, CSRN):
            raise TypeError(f"Y must be CSRN, got {type(Y).__name__}")
        N_x = len(self.shape)
        N_y = len(Y.shape)
        if not (0 <= mode_x < N_x):
            raise ValueError(f"mode_x={mode_x} out of range [0, {N_x})")
        if not (0 <= mode_y < N_y):
            raise ValueError(f"mode_y={mode_y} out of range [0, {N_y})")
        if self.shape[mode_x] != Y.shape[mode_y]:
            raise ValueError(
                f"Contraction shape mismatch: {self.shape[mode_x]} != {Y.shape[mode_y]}"
            )

        keep_x = [k for k in range(N_x) if k != mode_x]
        keep_y = [k for k in range(N_y) if k != mode_y]
        out_shape = tuple([self.shape[k] for k in keep_x]
                          + [Y.shape[k] for k in keep_y])

        if self.nnz == 0 or Y.nnz == 0:
            if len(out_shape) == 0:
                return 0.0
            return CSRN.from_coo(
                np.zeros((len(out_shape), 0), dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                out_shape,
            )

        coords_x = self._reconstruct_coords()
        coords_y = Y._reconstruct_coords()

        order_x = np.argsort(coords_x[mode_x], kind='stable')
        coords_x = coords_x[:, order_x]
        values_x = self.values[order_x]
        order_y = np.argsort(coords_y[mode_y], kind='stable')
        coords_y = coords_y[:, order_y]
        values_y = Y.values[order_y]

        I = self.shape[mode_x]
        cnt_x = np.bincount(coords_x[mode_x], minlength=I)
        cnt_y = np.bincount(coords_y[mode_y], minlength=I)
        pairs_per_i = cnt_x * cnt_y
        total_pairs = int(pairs_per_i.sum())

        if total_pairs == 0:
            if len(out_shape) == 0:
                return 0.0
            return CSRN.from_coo(
                np.zeros((len(out_shape), 0), dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                out_shape,
            )

        start_pair = np.empty(I + 1, dtype=np.int64)
        start_pair[0] = 0
        start_pair[1:] = np.cumsum(pairs_per_i)

        p_arr = np.arange(total_pairs, dtype=np.int64)
        i_of_pair = np.searchsorted(start_pair, p_arr, side='right') - 1
        within = p_arr - start_pair[i_of_pair]
        cnt_y_pp = cnt_y[i_of_pair]
        x_off = within // cnt_y_pp
        y_off = within % cnt_y_pp

        x_starts = np.empty(I + 1, dtype=np.int64)
        x_starts[0] = 0
        x_starts[1:] = np.cumsum(cnt_x)
        y_starts = np.empty(I + 1, dtype=np.int64)
        y_starts[0] = 0
        y_starts[1:] = np.cumsum(cnt_y)

        x_pos = x_starts[i_of_pair] + x_off
        y_pos = y_starts[i_of_pair] + y_off

        pair_values = values_x[x_pos] * values_y[y_pos]

        if len(out_shape) == 0:
            return float(pair_values.sum())

        out_coord_arrays = []
        for k in keep_x:
            out_coord_arrays.append(coords_x[k, x_pos])
        for k in keep_y:
            out_coord_arrays.append(coords_y[k, y_pos])

        flat_out = np.ravel_multi_index(out_coord_arrays, out_shape)
        order = np.argsort(flat_out, kind='stable')
        flat_sorted = flat_out[order]
        values_sorted = pair_values[order]

        if total_pairs == 1:
            agg = values_sorted
            unique_flat = flat_sorted
        else:
            new_group = np.empty(total_pairs, dtype=bool)
            new_group[0] = True
            new_group[1:] = flat_sorted[1:] != flat_sorted[:-1]
            group_starts = np.where(new_group)[0]
            agg = np.add.reduceat(values_sorted, group_starts)
            unique_flat = flat_sorted[group_starts]

        nz_mask = agg != 0.0
        if not nz_mask.any():
            return CSRN.from_coo(
                np.zeros((len(out_shape), 0), dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                out_shape,
            )
        final_coords = np.array(np.unravel_index(unique_flat[nz_mask], out_shape))
        final_values = agg[nz_mask]
        return CSRN.from_coo(final_coords, final_values, out_shape)


