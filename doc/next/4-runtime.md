## Runtime {#runtime}

<!-- go.dev/issue/71517 -->

The message printed when a program exits due to an unhandled panic
that was recovered and repanicked no longer repeats the text of
the panic value.

Previously, a program which panicked with `panic("PANIC")`,
recovered the panic, and then repanicked with the original
value would print:

    panic: PANIC [recovered]
      panic: PANIC

This program will now print:

    panic: PANIC [recovered, repanicked]

<!-- go.dev/issue/71546 -->

On Linux systems with kernel support for anonymous VMA names
(`CONFIG_ANON_VMA_NAME`), the Go runtime will annotate anonymous memory
mappings with context about their purpose. e.g., `[anon: Go: heap]` for heap
memory. This can be disabled with the [GODEBUG setting](/doc/godebug)
`decoratemappings=0`.

<!-- go.dev/issue/73581 -->

A new experimental garbage collector is now available as an experiment. The
new design aims to improve the efficiency of garbage collection through better
locality and CPU scalability in the mark algorithm. Benchmark result vary, but
we expect somewhere between a 10—40% reduction in garbage collection overhead
in real-world programs that heavily use the garbage collector.

The new garbage collector may be enabled by setting `GOEXPERIMENT=greenteagc`
at build time. See the [GitHub issue](/issue/73581) for more details on the design
and instructions on how to report feedback.
