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

We introduce a new profile type for goroutine leaks. With the experimental flag set to `GOEXPERIMENT=goroutineleakprofile`, it becomes accessible through `pprof` under the name `"goroutineleak"`.

The following snippet showcases a common anti-pattern that leads to goroutine leaks:
```go
// AggregateResults concurrently processes each request and aggregates the results.
// If one of the requests returns an error, the function returns immediately with the error.
func (s *Server[T, R]) AggregateResults(reqs []T) ([]R, error) {
	ch := make(chan wrap[R])
	for _, req := range reqs {
		go func(req T) {
      res, err := s.processRequest(req)
      ch <- wrap[R]{
        res: res,
        err: err,
      }
		}(req)
	}

	var results []R
	for range len(reqs) {
		x := <-ch
		if x.err != nil {
			return nil, x.err
		}
		results = append(results, x.res)
	}
	return results, nil
}
```
Channel `ch` is used to synchronize when concurrently processing each request in the slice `reqs`.
The responses are aggregated in a slice if all requests succeed.
Conversely, if any request produces an error, `AggregateResults` is shortcircuited to
return the error.
However, because `ch` is unbuffered, all pending request goroutines beyond the first to produce
the error will leak.

The key insight is that `ch` is inaccessible outside the scope of `AggregateResults`.
The Go runtime is now equipped to detect such patterns as they occur at execution time,
and record them in the goroutine leak profile.
For the case above, the goroutine leak profile would appear as:
```
Samples:
goroutineleak/count
          6: 1 2 3 4
Locations
     1: 0x104235daf M=1 runtime.gopark src/runtime/proc.go:464:0 s=447
     2: 0x1041c1ce7 M=1 runtime.chansend src/runtime/chan.go:283:0 s=176
     3: 0x1041c18f7 M=1 runtime.chansend1 src/runtime/chan.go:161:0 s=160
     4: 0x10428dd6b M=1 app.(*Server[go.shape.int,go.shape.int]).AggregateResults.func1 app/server.go:37:0 s=35
```
The leaked goroutines' stack precisely pinpoints the leaking operation in the source code.

The main advantage of goroutine leak profiles is that they have **no false positives**, but, for theoretical reasons, they may nevertheless
miss some goroutine leaks, e.g., when caused by global channels.
The underlying theory is presented in detail in [this publication by Saioc et al.](https://dl.acm.org/doi/pdf/10.1145/3676641.3715990).

More details about the implementation are presented in the [design document](https://github.com/golang/proposal/blob/master/design/74609-goroutine-leak-detection-gc.md).
We encourage users to experiment with the new feature in different environments (tests, CI, production), and welcome feedback on the [proposal issue](https://github.com/golang/go/issues/74609).
