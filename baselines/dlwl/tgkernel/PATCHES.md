# Vendored TGKernel

Reference implementation of Oettershagen, Kriege, Morris & Mutzel, *Temporal Graph Kernels for
Classifying Dissemination Processes* (SDM 2020), cloned from https://gitlab.com/tgpublic/tgkernel.

Two changes were needed to compile on macOS (clang, Eigen 3.4 from Homebrew):

1. `CMakeLists.txt`: `CMAKE_CXX_STANDARD` 11 → 17 (Eigen 3.4 uses `std::conditional_t`).
2. `src/Timer.{h,cpp}`: `time_point<system_clock>` → `time_point<high_resolution_clock>`
   (on libc++ `high_resolution_clock` is `steady_clock`, so the original does not type-check).

One functional fix:

3. `src/DataSetLoader.cpp`, `split_string`: `stoi` → `stoul`. Time stamps are `unsigned long`
   everywhere else, but the parser went through `int`, so it threw `out_of_range` on graphs spanning
   more than 2^31 (the class 3/4 graphs of `eu_email_dept2` slice 9 span ~2.2e9).

One addition for the delta-restricted variant of the DL-WL baseline:

4. `src/TGRepresentations.cpp`, `temporalGraphStreamToDirectedLineGraph`: if the environment variable
   `TGKERNEL_MAX_WAITING_TIME` is set to delta, two events (u, v; t) and (v, w; t') are only connected if
   t < t' <= t + delta (Definition 4 of our paper). Without the variable, the expansion is unbounded as in the
   original code.

Known limitations of the upstream code, hit by this data and *not* patched (only the directed line
graph expansion, action 7, is used here):

- `lib/wlkernel/ColorRefinementKernel.cpp` dereferences the result of `edge_labels.find(...)` without
  comparing against `end()`, so the undirected transformations (actions 1 and 4) segfault unless every
  event is present in both directions.
- `TGRepresentations::temporalGraphStreamToStaticExpansion` calls `nodemap.at((u, t_prev+1))` for a
  time vertex that only exists if `u` was a target at `t_prev`, so the static expansion (action 10)
  throws `out_of_range` on directed input.

Build:

    export EIGEN3_INCLUDE_DIR=/opt/homebrew/include/eigen3
    mkdir -p release && cd release
    cmake -DCMAKE_BUILD_TYPE=Release .. && make

This produces `release/tgkernel`, which `../dlwl.py` calls.
