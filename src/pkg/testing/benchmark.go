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
	N         int
	benchmark InternalBenchmark
	ns        int64
	bytes     int64
	start     int64
}

// StartTimer starts timing a test.  This function is called automatically
// before a benchmark starts, but it can also used to resume timing after
// a call to StopTimer.
func (b *B) StartTimer() {
	if b.start == 0 {
		b.start = time.Nanoseconds()
	}
}

// StopTimer stops timing a test.  This can be used to pause the timer
// while performing complex initialization that you don't
// want to measure.
func (b *B) StopTimer() {
	if b.start > 0 {
		b.ns += time.Nanoseconds() - b.start
	}
	b.start = 0
}

// ResetTimer sets the elapsed benchmark time to zero.
// It does not affect whether the timer is running.
func (b *B) ResetTimer() {
	if b.start > 0 {
		b.start = time.Nanoseconds()
	}
	b.ns = 0
}

// SetBytes records the number of bytes processed in a single operation.
// If this is called, the benchmark will report ns/op and MB/s.
func (b *B) SetBytes(n int64) { b.bytes = n }

func (b *B) nsPerOp() int64 {
	if b.N <= 0 {
		return 0
	}
	return b.ns / int64(b.N)
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

// run times the benchmark function.  It gradually increases the number
// of benchmark iterations until the benchmark runs for a second in order
// to get a reasonable measurement.  It prints timing information in this form
//		testing.BenchmarkHello	100000		19 ns/op
func (b *B) run() BenchmarkResult {
	// Run the benchmark for a single iteration in case it's expensive.
	n := 1
	b.runN(n)
	// Run the benchmark for at least the specified amount of time.
	time := int64(*benchTime * 1e9)
	for b.ns < time && n < 1e9 {
		last := n
		// Predict iterations/sec.
		if b.nsPerOp() == 0 {
			n = 1e9
		} else {
			n = int(time / b.nsPerOp())
		}
		// Run more iterations than we think we'll need for a second (1.5x).
		// Don't grow too fast in case we had timing errors previously.
		// Be sure to run at least one more than last time.
		n = max(min(n+n/2, 100*last), last+1)
		// Round up to something easy to read.
		n = roundUp(n)
		b.runN(n)
	}
	return BenchmarkResult{b.N, b.ns, b.bytes}
}

// The results of a benchmark run.
type BenchmarkResult struct {
	N     int   // The number of iterations.
	Ns    int64 // The total time taken.
	Bytes int64 // Bytes processed in one iteration.
}

func (r BenchmarkResult) NsPerOp() int64 {
	if r.N <= 0 {
		return 0
	}
	return r.Ns / int64(r.N)
}

func (r BenchmarkResult) mbPerSec() float64 {
	if r.Bytes <= 0 || r.Ns <= 0 || r.N <= 0 {
		return 0
	}
	return float64(r.Bytes) * float64(r.N) / float64(r.Ns) * 1e3
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
			ns = fmt.Sprintf("%13.2f ns/op", float64(r.Ns)/float64(r.N))
		} else {
			ns = fmt.Sprintf("%12.1f ns/op", float64(r.Ns)/float64(r.N))
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
			b := &B{benchmark: Benchmark}
			benchName := Benchmark.Name
			if procs != 1 {
				benchName = fmt.Sprintf("%s-%d", Benchmark.Name, procs)
			}
			fmt.Printf("%s\t", benchName)
			r := b.run()
			fmt.Printf("%v\n", r)
			if p := runtime.GOMAXPROCS(-1); p != procs {
				fmt.Fprintf(os.Stderr, "testing: %s left GOMAXPROCS set to %d\n", benchName, p)
			}
		}
	}
}

// Benchmark benchmarks a single function. Useful for creating
// custom benchmarks that do not use gotest.
func Benchmark(f func(b *B)) BenchmarkResult {
	b := &B{benchmark: InternalBenchmark{"", f}}
	return b.run()
}
