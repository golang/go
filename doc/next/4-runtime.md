## Runtime {#runtime}

### New garbage collector

The Green Tea garbage collector, previously available as an experiment in
Go 1.25, is now enabled by default after incorporating feedback.

This garbage collector’s design improves the performance of marking and
scanning small objects through better locality and CPU scalability.
Benchmark result vary, but we expect somewhere between a 10—40% reduction
in garbage collection overhead in real-world programs that heavily use the
garbage collector.
Further improvements, on the order of 10% in garbage collection overhead,
are expected when running on newer amd64-based CPU platforms (Intel Ice
Lake or AMD Zen 4 and newer), as the garbage collector now leverages
vector instructions for scanning small objects when possible.

The new garbage collector may be disabled by setting
`GOEXPERIMENT=nogreenteagc` at build time.
This opt-out setting is expected to be removed in Go 1.27.
If you disable the new garbage collector for any reason related to its
performance or behavior, please [file an issue](/issue/new).
