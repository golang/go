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

### Faster cgo calls

<!-- CL 646198 -->

The baseline runtime overhead of cgo calls has been reduced by ~30%.

### Goroutine leak profiles {#goroutineleak-profiles}

A new profile type that reports leaked goroutines is now available as an
experiment. The new profile type, named `goroutineleak` in the [runtime/pprof]
package, may be enabled by setting `GOEXPERIMENT=goroutineleakprofile`
at build time. Enabling the experiment also makes the profile available
as a [net/http/pprof] endpoint, `debug/pprof/goroutineleak`.

The following example showcases a real-world goroutine leak that
can be revealed by the new profile:
```go
type result struct{
		res workResult
		err error
}

func processWorkItems(ws []workItem) ([]workResult, error) {
	// Process work items in parallel, aggregating results in ch.
	ch := make(chan result)
	for _, w := range ws {
		go func() {
			res, err := processWorkItem(w)
			ch <- result{res, err}
		}()
	}

	// Collect the results from ch, or return an error if one is found.
	var results []workResult
	for range len(ws) {
		r := <-ch
		if r.err != nil {
			// This premature return may cause goroutine leaks
			return nil, r.err
		}
		results = append(results, r.res)
	}
	return results, nil
}
```
Because `ch` is unbuffered, if `processWorkItems` returns early due to
an error, all remaining `processWorkItem` goroutines will leak.
However, `ch` also becomes unreachable to all other goroutines
not involved in the leak soon after the leak itself occurs.
In general, the runtime is now equipped to identify as leaked
any goroutines blocked on operations over concurrency primitives
(e.g., channels, [sync.Mutex]) that are not reachable from runnable goroutines.

Note, however, that the runtime may fail to identify leaks caused by
blocking on operations over concurrency primitives reachable
through global variables or the local variables of runnable goroutines.

Special thanks to Vlad Saioc at Uber for contributing this work.
The underlying theory is presented in detail by Saioc et al. in [this publication](https://dl.acm.org/doi/pdf/10.1145/3676641.3715990).

We encourage users to try out the new feature with examples derived from known patterns in [the Go playground](https://go.dev/play/p/3C71z4Dpav-?v=gotip),
as well as experiment with different environments (tests, CI, production).
We welcome feedback on the [proposal issue](https://github.com/golang/go/issues/74609).
