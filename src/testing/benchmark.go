// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

var matchBenchmarks = flag.String("test.bench", "", "regular expression to select benchmarks to run")
var benchTime = flag.Duration("test.benchtime", 1*time.Second, "approximate run time for each benchmark")
var benchmarkMemory = flag.Bool("test.benchmem", false, "print memory allocations for benchmarks")

// Global lock to ensure only one benchmark runs at a time.
var benchmarkLock sync.Mutex

// Used for every benchmark for measuring memory.
var memStats runtime.MemStats

// An internal type but exported because it is cross-package; part of the implementation
// of the "go test" command.
type InternalBenchmark struct {
	Name string
	F    func(b *B)
}

// B is a type passed to Benchmark functions to manage benchmark
// timing and to specify the number of iterations to run.
type B struct {
	common
	N                int
	previousN        int           // number of iterations in the previous run
	previousDuration time.Duration // total duration of the previous run
	benchmark        InternalBenchmark
	bytes            int64
	timerOn          bool
	showAllocResult  bool
	result           BenchmarkResult
	parallelism      int // RunParallel creates parallelism*GOMAXPROCS goroutines
	// The initial states of memStats.Mallocs and memStats.TotalAlloc.
	startAllocs uint64
	startBytes  uint64
	// The net total of this test after being run.
	netAllocs uint64
	netBytes  uint64
}

// StartTimer starts timing a test.  This function is called automatically
// before a benchmark starts, but it can also used to resume timing after
// a call to StopTimer.
func (b *B) StartTimer() {
	if !b.timerOn {
		runtime.ReadMemStats(&memStats)
		b.startAllocs = memStats.Mallocs
		b.startBytes = memStats.TotalAlloc
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
		runtime.ReadMemStats(&memStats)
		b.netAllocs += memStats.Mallocs - b.startAllocs
		b.netBytes += memStats.TotalAlloc - b.startBytes
		b.timerOn = false
	}
}

// ResetTimer zeros the elapsed benchmark time and memory allocation counters.
// It does not affect whether the timer is running.
func (b *B) ResetTimer() {
	if b.timerOn {
		runtime.ReadMemStats(&memStats)
		b.startAllocs = memStats.Mallocs
		b.startBytes = memStats.TotalAlloc
		b.start = time.Now()
	}
	b.duration = 0
	b.netAllocs = 0
	b.netBytes = 0
}

// SetBytes records the number of bytes processed in a single operation.
// If this is called, the benchmark will report ns/op and MB/s.
func (b *B) SetBytes(n int64) { b.bytes = n }

// ReportAllocs enables malloc statistics for this benchmark.
// It is equivalent to setting -test.benchmem, but it only affects the
// benchmark function that calls ReportAllocs.
func (b *B) ReportAllocs() {
	b.showAllocResult = true
}

func (b *B) nsPerOp() int64 {
	if b.N <= 0 {
		return 0
	}
	return b.duration.Nanoseconds() / int64(b.N)
}

// runN runs a single benchmark for the specified number of iterations.
func (b *B) runN(n int) {
	benchmarkLock.Lock()
	defer benchmarkLock.Unlock()
	// Try to get a comparable environment for each run
	// by clearing garbage from previous runs.
	runtime.GC()
	b.N = n
	b.parallelism = 1
	b.ResetTimer()
	b.StartTimer()
	b.benchmark.F(b)
	b.StopTimer()
	b.previousN = n
	b.previousDuration = b.duration
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
	for n >= 10 {
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

// roundUp rounds x up to a number of the form [1eX, 2eX, 3eX, 5eX].
func roundUp(n int) int {
	base := roundDown10(n)
	switch {
	case n <= base:
		return base
	case n <= (2 * base):
		return 2 * base
	case n <= (3 * base):
		return 3 * base
	case n <= (5 * base):
		return 5 * base
	default:
		return 10 * base
	}
}

// run times the benchmark function in a separate goroutine.
func (b *B) run() BenchmarkResult {
	go b.launch()
	<-b.signal
	return b.result
}

// launch launches the benchmark function.  It gradually increases the number
// of benchmark iterations until the benchmark runs for the requested benchtime.
// It prints timing information in this form
//		testing.BenchmarkHello	100000		19 ns/op
// launch is run by the run function as a separate goroutine.
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
	d := *benchTime
	for !b.failed && b.duration < d && n < 1e9 {
		last := n
		// Predict required iterations.
		if b.nsPerOp() == 0 {
			n = 1e9
		} else {
			n = int(d.Nanoseconds() / b.nsPerOp())
		}
		// Run more iterations than we think we'll need (1.2x).
		// Don't grow too fast in case we had timing errors previously.
		// Be sure to run at least one more than last time.
		n = max(min(n+n/5, 100*last), last+1)
		// Round up to something easy to read.
		n = roundUp(n)
		b.runN(n)
	}
	b.result = BenchmarkResult{b.N, b.duration, b.bytes, b.netAllocs, b.netBytes}
}

// The results of a benchmark run.
type BenchmarkResult struct {
	N         int           // The number of iterations.
	T         time.Duration // The total time taken.
	Bytes     int64         // Bytes processed in one iteration.
	MemAllocs uint64        // The total number of memory allocations.
	MemBytes  uint64        // The total number of bytes allocated.
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

func (r BenchmarkResult) AllocsPerOp() int64 {
	if r.N <= 0 {
		return 0
	}
	return int64(r.MemAllocs) / int64(r.N)
}

func (r BenchmarkResult) AllocedBytesPerOp() int64 {
	if r.N <= 0 {
		return 0
	}
	return int64(r.MemBytes) / int64(r.N)
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

func (r BenchmarkResult) MemString() string {
	return fmt.Sprintf("%8d B/op\t%8d allocs/op",
		r.AllocedBytesPerOp(), r.AllocsPerOp())
}

// benchmarkName returns full name of benchmark including procs suffix.
func benchmarkName(name string, n int) string {
	if n != 1 {
		return fmt.Sprintf("%s-%d", name, n)
	}
	return name
}

// An internal function but exported because it is cross-package; part of the implementation
// of the "go test" command.
func RunBenchmarks(matchString func(pat, str string) (bool, error), benchmarks []InternalBenchmark) {
	// If no flag was specified, don't run benchmarks.
	if len(*matchBenchmarks) == 0 {
		return
	}
	// Collect matching benchmarks and determine longest name.
	maxprocs := 1
	for _, procs := range cpuList {
		if procs > maxprocs {
			maxprocs = procs
		}
	}
	maxlen := 0
	var bs []InternalBenchmark
	for _, Benchmark := range benchmarks {
		matched, err := matchString(*matchBenchmarks, Benchmark.Name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for -test.bench: %s\n", err)
			os.Exit(1)
		}
		if matched {
			bs = append(bs, Benchmark)
			benchName := benchmarkName(Benchmark.Name, maxprocs)
			if l := len(benchName); l > maxlen {
				maxlen = l
			}
		}
	}
	for _, Benchmark := range bs {
		for _, procs := range cpuList {
			runtime.GOMAXPROCS(procs)
			b := &B{
				common: common{
					signal: make(chan interface{}),
				},
				benchmark: Benchmark,
			}
			benchName := benchmarkName(Benchmark.Name, procs)
			fmt.Printf("%-*s\t", maxlen, benchName)
			r := b.run()
			if b.failed {
				// The output could be very long here, but probably isn't.
				// We print it all, regardless, because we don't want to trim the reason
				// the benchmark failed.
				fmt.Printf("--- FAIL: %s\n%s", benchName, b.output)
				continue
			}
			results := r.String()
			if *benchmarkMemory || b.showAllocResult {
				results += "\t" + r.MemString()
			}
			fmt.Println(results)
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

// A PB is used by RunParallel for running parallel benchmarks.
type PB struct {
	globalN *uint64 // shared between all worker goroutines iteration counter
	grain   uint64  // acquire that many iterations from globalN at once
	cache   uint64  // local cache of acquired iterations
	bN      uint64  // total number of iterations to execute (b.N)
}

// Next reports whether there are more iterations to execute.
func (pb *PB) Next() bool {
	if pb.cache == 0 {
		n := atomic.AddUint64(pb.globalN, pb.grain)
		if n <= pb.bN {
			pb.cache = pb.grain
		} else if n < pb.bN+pb.grain {
			pb.cache = pb.bN + pb.grain - n
		} else {
			return false
		}
	}
	pb.cache--
	return true
}

// RunParallel runs a benchmark in parallel.
// It creates multiple goroutines and distributes b.N iterations among them.
// The number of goroutines defaults to GOMAXPROCS. To increase parallelism for
// non-CPU-bound benchmarks, call SetParallelism before RunParallel.
// RunParallel is usually used with the go test -cpu flag.
//
// The body function will be run in each goroutine. It should set up any
// goroutine-local state and then iterate until pb.Next returns false.
// It should not use the StartTimer, StopTimer, or ResetTimer functions,
// because they have global effect.
func (b *B) RunParallel(body func(*PB)) {
	// Calculate grain size as number of iterations that take ~100µs.
	// 100µs is enough to amortize the overhead and provide sufficient
	// dynamic load balancing.
	grain := uint64(0)
	if b.previousN > 0 && b.previousDuration > 0 {
		grain = 1e5 * uint64(b.previousN) / uint64(b.previousDuration)
	}
	if grain < 1 {
		grain = 1
	}
	// We expect the inner loop and function call to take at least 10ns,
	// so do not do more than 100µs/10ns=1e4 iterations.
	if grain > 1e4 {
		grain = 1e4
	}

	n := uint64(0)
	numProcs := b.parallelism * runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	wg.Add(numProcs)
	for p := 0; p < numProcs; p++ {
		go func() {
			defer wg.Done()
			pb := &PB{
				globalN: &n,
				grain:   grain,
				bN:      uint64(b.N),
			}
			body(pb)
		}()
	}
	wg.Wait()
	if n <= uint64(b.N) && !b.Failed() {
		b.Fatal("RunParallel: body exited without pb.Next() == false")
	}
}

// SetParallelism sets the number of goroutines used by RunParallel to p*GOMAXPROCS.
// There is usually no need to call SetParallelism for CPU-bound benchmarks.
// If p is less than 1, this call will have no effect.
func (b *B) SetParallelism(p int) {
	if p >= 1 {
		b.parallelism = p
	}
}

// Benchmark benchmarks a single function. Useful for creating
// custom benchmarks that do not use the "go test" command.
func Benchmark(f func(b *B)) BenchmarkResult {
	b := &B{
		common: common{
			signal: make(chan interface{}),
		},
		benchmark: InternalBenchmark{"", f},
	}
	return b.run()
}
