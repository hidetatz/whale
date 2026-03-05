# TensorGraph
# ↓  Type Inference, Constant Fold, Dead Node Elimination, Operator Fusion
# AlgorithmIR
# ↓  Auto Loop Scheduling (Vectorization, Unroll, Tile)
# ScheduledIR
# ↓  Hardware Optimization
# BackendIR
# ↓  Compile and Data Transfer
# Execute

# def compile_and_exec(t):
#     # lower tensor tree to GraphIR
#     tensors = []
#     seen = set()

#     def dfs(_t: Tensor):
#         if _t in seen:
#             return

#         seen.add(_t)

#         if _t.inputs is not None:
#             for i in _t.inputs:
#                 dfs(i)

#         tensors.append(_t)

#     dfs(t)
#     return tensors

