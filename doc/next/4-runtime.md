## Runtime {#runtime}

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
We encourage users to experiment with the new feature in different environments
(tests, CI, production), and welcome feedback on the [proposal issue](https://github.com/golang/go/issues/74609).