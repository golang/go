// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"
)

var matchBenchmarks = flag.String("test.bench", "", "regular expression to select benchmarks to run")
var benchTime = flag.Float64("test.benchtime", 1, "approximate run time for each benchmark, in seconds")

// An internal type but exported because it is cross-package; part of the implementation
// of gotest.
type InternalBenchmark struct {
	Name string
	F    func(b *B)
}

// B is a type passed to Benchmark functions to manage benchmark
// timing and to specify the number of iterations to run.
type B struct {
	common
	N         int
	benchmark InternalBenchmark
	bytes     int64
	timerOn   bool
	result    BenchmarkResult
}

// StartTimer starts timing a test.  This function is called automatically
// before a benchmark starts, but it can also used to resume timing after
// a call to StopTimer.
func (b *B) StartTimer() {
	if !b.timerOn {
		b.start = time.Now()
		b.timerOn = true
	}
}

// StopTimer stops timing a test.  This can be used to pause the timer
// while performing complex initialization that you don't
// want to measure.
func (b *B) StopTimer() {
	if b.timerOn {
		b.duration += time.Now().Sub(b.start)
		b.timerOn = false
	}
}

// ResetTimer sets the elapsed benchmark time to zero.
// It does not affect whether the timer is running.
func (b *B) ResetTimer() {
	if b.timerOn {
		b.start = time.Now()
	}
	b.duration = 0
}

// SetBytes records the number of bytes processed in a single operation.
// If this is called, the benchmark will report ns/op and MB/s.
func (b *B) SetBytes(n int64) { b.bytes = n }

func (b *B) nsPerOp() int64 {
	if b.N <= 0 {
		return 0
	}
	return b.duration.Nanoseconds() / int64(b.N)
}

// runN runs a single benchmark for the specified number of iterations.
func (b *B) runN(n int) {
	// Try to get a comparable environment for each run
	// by clearing garbage from previous runs.
	runtime.GC()
	b.N = n
	b.ResetTimer()
	b.StartTimer()
	b.benchmark.F(b)
	b.StopTimer()
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

// roundDown10 rounds a number down to the nearest power of 10.
func roundDown10(n int) int {
	var tens = 0
	// tens = floor(log_10(n))
	for n > 10 {
		n = n / 10
		tens++
	}
	// result = 10^tens
	result := 1
	for i := 0; i < tens; i++ {
		result *= 10
	}
	return result
}

// roundUp rounds x up to a number of the form [1eX, 2eX, 5eX].
func roundUp(n int) int {
	base := roundDown10(n)
	if n < (2 * base) {
		return 2 * base
	}
	if n < (5 * base) {
		return 5 * base
	}
	return 10 * base
}

// run times the benchmark function in a separate goroutine.
func (b *B) run() BenchmarkResult {
	go b.launch()
	<-b.signal
	return b.result
}

// launch launches the benchmark function.  It gradually increases the number
// of benchmark iterations until the benchmark runs for a second in order
// to get a reasonable measurement.  It prints timing information in this form
//		testing.BenchmarkHello	100000		19 ns/op
// launch is run by the fun function as a separate goroutine.
func (b *B) launch() {
	// Run the benchmark for a single iteration in case it's expensive.
	n := 1

	// Signal that we're done whether we return normally
	// or by FailNow's runtime.Goexit.
	defer func() {
		b.signal <- b
	}()

	b.runN(n)
	// Run the benchmark for at least the specified amount of time.
	d := time.Duration(*benchTime * float64(time.Second))
	for !b.failed && b.duration < d && n < 1e9 {
		last := n
		// Predict iterations/sec.
		if b.nsPerOp() == 0 {
			n = 1e9
		} else {
			n = int(d.Nanoseconds() / b.nsPerOp())
		}
		// Run more iterations than we think we'll need for a second (1.5x).
		// Don't grow too fast in case we had timing errors previously.
		// Be sure to run at least one more than last time.
		n = max(min(n+n/2, 100*last), last+1)
		// Round up to something easy to read.
		n = roundUp(n)
		b.runN(n)
	}
	b.result = BenchmarkResult{b.N, b.duration, b.bytes}
}

// The results of a benchmark run.
type BenchmarkResult struct {
	N     int           // The number of iterations.
	T     time.Duration // The total time taken.
	Bytes int64         // Bytes processed in one iteration.
}

func (r BenchmarkResult) NsPerOp() int64 {
	if r.N <= 0 {
		return 0
	}
	return r.T.Nanoseconds() / int64(r.N)
}

func (r BenchmarkResult) mbPerSec() float64 {
	if r.Bytes <= 0 || r.T <= 0 || r.N <= 0 {
		return 0
	}
	return (float64(r.Bytes) * float64(r.N) / 1e6) / r.T.Seconds()
}

func (r BenchmarkResult) String() string {
	mbs := r.mbPerSec()
	mb := ""
	if mbs != 0 {
		mb = fmt.Sprintf("\t%7.2f MB/s", mbs)
	}
	nsop := r.NsPerOp()
	ns := fmt.Sprintf("%10d ns/op", nsop)
	if r.N > 0 && nsop < 100 {
		// The format specifiers here make sure that
		// the ones digits line up for all three possible formats.
		if nsop < 10 {
			ns = fmt.Sprintf("%13.2f ns/op", float64(r.T.Nanoseconds())/float64(r.N))
		} else {
			ns = fmt.Sprintf("%12.1f ns/op", float64(r.T.Nanoseconds())/float64(r.N))
		}
	}
	return fmt.Sprintf("%8d\t%s%s", r.N, ns, mb)
}

// An internal function but exported because it is cross-package; part of the implementation
// of gotest.
func RunBenchmarks(matchString func(pat, str string) (bool, error), benchmarks []InternalBenchmark) {
	// If no flag was specified, don't run benchmarks.
	if len(*matchBenchmarks) == 0 {
		return
	}
	for _, Benchmark := range benchmarks {
		matched, err := matchString(*matchBenchmarks, Benchmark.Name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for -test.bench: %s\n", err)
			os.Exit(1)
		}
		if !matched {
			continue
		}
		for _, procs := range cpuList {
			runtime.GOMAXPROCS(procs)
			b := &B{
				common: common{
					signal: make(chan interface{}),
				},
				benchmark: Benchmark,
			}
			benchName := Benchmark.Name
			if procs != 1 {
				benchName = fmt.Sprintf("%s-%d", Benchmark.Name, procs)
			}
			fmt.Printf("%s\t", benchName)
			r := b.run()
			if b.failed {
				// The output could be very long here, but probably isn't.
				// We print it all, regardless, because we don't want to trim the reason
				// the benchmark failed.
				fmt.Printf("--- FAIL: %s\n%s", benchName, b.output)
				continue
			}
			fmt.Printf("%v\n", r)
			// Unlike with tests, we ignore the -chatty flag and always print output for
			// benchmarks since the output generation time will skew the results.
			if len(b.output) > 0 {
				b.trimOutput()
				fmt.Printf("--- BENCH: %s\n%s", benchName, b.output)
			}
			if p := runtime.GOMAXPROCS(-1); p != procs {
				fmt.Fprintf(os.Stderr, "testing: %s left GOMAXPROCS set to %d\n", benchName, p)
			}
		}
	}
}

// trimOutput shortens the output from a benchmark, which can be very long.
func (b *B) trimOutput() {
	// The output is likely to appear multiple times because the benchmark
	// is run multiple times, but at least it will be seen. This is not a big deal
	// because benchmarks rarely print, but just in case, we trim it if it's too long.
	const maxNewlines = 10
	for nlCount, j := 0, 0; j < len(b.output); j++ {
		if b.output[j] == '\n' {
			nlCount++
			if nlCount >= maxNewlines {
				b.output = append(b.output[:j], "\n\t... [output truncated]\n"...)
				break
			}
		}
	}
}

// Benchmark benchmarks a single function. Useful for creating
// custom benchmarks that do not use gotest.
func Benchmark(f func(b *B)) BenchmarkResult {
	b := &B{
		common: common{
			signal: make(chan interface{}),
		},
		benchmark: InternalBenchmark{"", f},
	}
	return b.run()
}
