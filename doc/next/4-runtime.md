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

<!-- go.dev/issue/73193 -->

The default behavior of the `GOMAXPROCS` has changed. In prior versions of Go,
`GOMAXPROCS` defaults to the number of logical CPUs available at startup
([runtime.NumCPU]). Go 1.25 introduces two changes:

1. On Linux, the runtime considers the CPU bandwidth limit of the cgroup
   containing the process, if any. If the CPU bandwidth limit is lower than the
   number of logical CPUs available, `GOMAXPROCS` will default to the lower
   limit. In container runtime systems like Kubernetes, cgroup CPU bandwidth
   limits generally correspond to the "CPU limit" option. The Go runtime does
   not consider the "CPU requests" option.

2. On all OSes, the runtime periodically updates `GOMAXPROCS` if the number
   of logical CPUs available or the cgroup CPU bandwidth limit change.

Both of these behaviors are automatically disabled if `GOMAXPROCS` is set
manually via the `GOMAXPROCS` environment variable or a call to
[runtime.GOMAXPROCS]. They can also be disabled explicitly with the [GODEBUG
settings](/doc/godebug) `containermaxprocs=0` and `updatemaxprocs=0`,
respectively.

In order to support reading updated cgroup limits, the runtime will keep cached
file descriptors for the cgroup files for the duration of the process lifetime.

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
