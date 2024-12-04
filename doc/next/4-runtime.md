## Runtime {#runtime}

<!-- go.dev/issue/54766 -->
<!-- go.dev/cl/614795 -->
<!-- go.dev/issue/68578 -->

Several performance improvements to the runtime have decreased CPU overheads by
2—3% on average across a suite of representative benchmarks.
Results may vary by application.
These improvements include a new builtin `map` implementation based on
[Swiss Tables](https://abseil.io/about/design/swisstables), more efficient
memory allocation of small objects, and a new runtime-internal mutex
implementation.

The new builtin `map` implementation and new runtime-internal mutex may be
disabled by setting `GOEXPERIMENT=noswissmap` and `GOEXPERIMENT=nospinbitmutex`
at build time respectively.
