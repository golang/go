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

We introduce a new profile type for goroutine leaks. With the experimental
flag set to `GOEXPERIMENT=goroutineleakprofile`, it becomes accessible
through `pprof` under the name `"goroutineleak"`.

The following snippet showcases a common, but erroneous pattern
that leads to goroutine leaks:
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
Because `ch` is unbuffered, if `processWorkItems` returns early due to an error,
all remaining work item goroutines will leak.
However, also note that, soon after the leak occurs, `ch` is inaccessible
to any other goroutine, except those involved in the leak.

To generalize, a goroutine is leaked if it is blocked by concurrency
primitives (specifically channels, and `sync` primitives such as mutex) that
are only referenced by the blocked goroutine itself, or other leaked goroutines.
The Go runtime is now equipped to reveal leaked goroutines by recording their stacks in
goroutine leak profiles.
In the example above, the stacks of work item goroutines point to the culprit channel send
operation.

Note that, while goroutine leak profiles only include true positives, goroutine leaks may be
missed when caused by concurrency primitives that are accessible globally, or referenced
by runnable goroutines.

Special thanks to Vlad Saioc at Uber for contributing this work.
The underlying theory is presented in detail by Saioc et al. in [this publication](https://dl.acm.org/doi/pdf/10.1145/3676641.3715990).

<!-- More details about the implementation are presented in the [design document](https://github.com/golang/proposal/blob/master/design/74609-goroutine-leak-detection-gc.md). -->
We encourage users to try out the new feature with examples derived from known patterns in [the Go playground](https://go.dev/play/p/aZc-HJiSH-R?v=gotip),
as well as experiment with different environments (tests, CI, production).
We welcome feedback on the [proposal issue](https://github.com/golang/go/issues/74609).
