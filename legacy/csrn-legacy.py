import numpy as np

NODE_DTYPE = np.dtype([
    ('first',  np.int64),
    ('degree', np.int64),
    ('base',   np.int64),
])


class CSRN:

    def __init__(self, shape, node, idx, values):
        self.shape = tuple(shape)
        self.node = node
        self.idx = idx
        self.values = values

    @classmethod
    def from_coo(cls, coords, data, shape):
        coords = np.asarray(coords, dtype=np.int64)
        data = np.asarray(data, dtype=np.float64)

        N = coords.shape[0]
        nnz = coords.shape[1]

        if nnz == 0:
            node = np.zeros(1, dtype=NODE_DTYPE)
            return cls(shape, node, np.array([], dtype=np.int64), np.array([], dtype=np.float64))

        order = np.lexsort(coords[::-1])
        coords = coords[:, order]
        data = data[order]

        prefix_map = [dict() for _ in range(N)]
        children = [[] for _ in range(N)]

        prefix_map[0][()] = 0
        children[0].append([])

        for j in range(nnz):
            for k in range(N):
                prefix = tuple(coords[:k, j])
                parent = prefix_map[k][prefix]
                val = int(coords[k, j])

                if k < N - 1:
                    new_pref = prefix + (val,)
                    if new_pref not in prefix_map[k + 1]:
                        idx_new = len(prefix_map[k + 1])
                        prefix_map[k + 1][new_pref] = idx_new
                        children[k + 1].append([])
                        children[k][parent].append(val)
                else:
                    children[k][parent].append(val)

        sizes = [len(prefix_map[k]) for k in range(N)]
        offsets = np.cumsum([0] + sizes[:-1])
        total_nodes = sum(sizes)

        total_idx = sum(len(c) for lvl in children for c in lvl)
        idx_arr = np.empty(total_idx, dtype=np.int64)

        node = np.zeros(total_nodes, dtype=NODE_DTYPE)

        ptr = 0
        for k in range(N):
            for i in range(sizes[k]):
                gid = offsets[k] + i
                ch = children[k][i]

                node['first'][gid] = ptr
                node['degree'][gid] = len(ch)

                if ch:
                    idx_arr[ptr:ptr+len(ch)] = ch

                ptr += len(ch)

        for k in range(N):
            cum = 0
            for i in range(sizes[k]):
                gid = offsets[k] + i

                if k < N - 1:
                    node['base'][gid] = offsets[k+1] + cum
                else:
                    node['base'][gid] = cum

                cum += len(children[k][i])

        return cls(shape, node, idx_arr, data.copy())

    def to_dense(self):
        out = np.zeros(self.shape, dtype=np.float64)
        if len(self.values) == 0:
            return out

        self._fill(0, 0, (), out)
        return out

    def _fill(self, node_id, depth, prefix, out):
        n = self.node[node_id]
        first = n['first']
        deg = n['degree']
        base = n['base']

        if depth == len(self.shape) - 1:
            for i in range(deg):
                idx = self.idx[first + i]
                out[prefix + (idx,)] = self.values[base + i]
        else:
            for i in range(deg):
                idx = self.idx[first + i]
                self._fill(base + i, depth + 1, prefix + (idx,), out)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        node_id = 0

        for d in range(len(index) - 1):
            n = self.node[node_id]
            first, deg, base = n

            if deg == 0:
                return 0.0

            arr = self.idx[first:first+deg]
            pos = np.searchsorted(arr, index[d])

            if pos >= deg or arr[pos] != index[d]:
                return 0.0

            node_id = base + pos

        n = self.node[node_id]
        first, deg, base = n

        arr = self.idx[first:first+deg]
        pos = np.searchsorted(arr, index[-1])

        if pos >= deg or arr[pos] != index[-1]:
            return 0.0

        return self.values[base + pos]

    def ttm_mode0(self, A):
        A = np.asarray(A, dtype=np.float64)
        J, I0 = A.shape

        out = np.zeros((J,) + self.shape[1:], dtype=np.float64)

        if len(self.values) == 0:
            return out

        root = self.node[0]
        first, deg, base = root

        for i in range(deg):
            i0 = self.idx[first + i]
            sub = self._subtree_dense(base + i, 1)
            out += np.multiply.outer(A[:, i0], sub)

        return out

    def _subtree_dense(self, node_id, depth):
        shape = self.shape[depth:]
        out = np.zeros(shape, dtype=np.float64)
        self._fill(node_id, depth, (), out)
        return out
