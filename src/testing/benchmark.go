// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"context"
	"flag"
	"fmt"
	"internal/sysinfo"
	"io"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

func initBenchmarkFlags() {
	matchBenchmarks = flag.String("test.bench", "", "run only benchmarks matching `regexp`")
	benchmarkMemory = flag.Bool("test.benchmem", false, "print memory allocations for benchmarks")
	flag.Var(&benchTime, "test.benchtime", "run each benchmark for duration `d` or N times if `d` is of the form Nx")
}

var (
	matchBenchmarks *string
	benchmarkMemory *bool

	benchTime = durationOrCountFlag{d: 1 * time.Second} // changed during test of testing package
)

type durationOrCountFlag struct {
	d         time.Duration
	n         int
	allowZero bool
}

func (f *durationOrCountFlag) String() string {
	if f.n > 0 {
		return fmt.Sprintf("%dx", f.n)
	}
	return f.d.String()
}

func (f *durationOrCountFlag) Set(s string) error {
	if strings.HasSuffix(s, "x") {
		n, err := strconv.ParseInt(s[:len(s)-1], 10, 0)
		if err != nil || n < 0 || (!f.allowZero && n == 0) {
			return fmt.Errorf("invalid count")
		}
		*f = durationOrCountFlag{n: int(n)}
		return nil
	}
	d, err := time.ParseDuration(s)
	if err != nil || d < 0 || (!f.allowZero && d == 0) {
		return fmt.Errorf("invalid duration")
	}
	*f = durationOrCountFlag{d: d}
	return nil
}

// Global lock to ensure only one benchmark runs at a time.
var benchmarkLock sync.Mutex

// Used for every benchmark for measuring memory.
var memStats runtime.MemStats

// InternalBenchmark is an internal type but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
type InternalBenchmark struct {
	Name string
	F    func(b *B)
}

// B is a type passed to [Benchmark] functions to manage benchmark
// timing and control the number of iterations.
//
// A benchmark ends when its Benchmark function returns or calls any of the methods
// FailNow, Fatal, Fatalf, SkipNow, Skip, or Skipf. Those methods must be called
// only from the goroutine running the Benchmark function.
// The other reporting methods, such as the variations of Log and Error,
// may be called simultaneously from multiple goroutines.
//
// Like in tests, benchmark logs are accumulated during execution
// and dumped to standard output when done. Unlike in tests, benchmark logs
// are always printed, so as not to hide output whose existence may be
// affecting benchmark results.
type B struct {
	common
	importPath       string // import path of the package containing the benchmark
	bstate           *benchState
	N                int
	previousN        int           // number of iterations in the previous run
	previousDuration time.Duration // total duration of the previous run
	benchFunc        func(b *B)
	benchTime        durationOrCountFlag
	bytes            int64
	missingBytes     bool // one of the subbenchmarks does not have bytes set.
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
	// Extra metrics collected by ReportMetric.
	extra map[string]float64
	// For Loop() to be executed in benchFunc.
	// Loop() has its own control logic that skips the loop scaling.
	// See issue #61515.
	loopN int
}

// StartTimer starts timing a test. This function is called automatically
// before a benchmark starts, but it can also be used to resume timing after
// a call to [B.StopTimer].
func (b *B) StartTimer() {
	if !b.timerOn {
		runtime.ReadMemStats(&memStats)
		b.startAllocs = memStats.Mallocs
		b.startBytes = memStats.TotalAlloc
		b.start = highPrecisionTimeNow()
		b.timerOn = true
	}
}

// StopTimer stops timing a test. This can be used to pause the timer
// while performing steps that you don't want to measure.
func (b *B) StopTimer() {
	if b.timerOn {
		b.duration += highPrecisionTimeSince(b.start)
		runtime.ReadMemStats(&memStats)
		b.netAllocs += memStats.Mallocs - b.startAllocs
		b.netBytes += memStats.TotalAlloc - b.startBytes
		b.timerOn = false
	}
}

// ResetTimer zeroes the elapsed benchmark time and memory allocation counters
// and deletes user-reported metrics.
// It does not affect whether the timer is running.
func (b *B) ResetTimer() {
	if b.extra == nil {
		// Allocate the extra map before reading memory stats.
		// Pre-size it to make more allocation unlikely.
		b.extra = make(map[string]float64, 16)
	} else {
		clear(b.extra)
	}
	if b.timerOn {
		runtime.ReadMemStats(&memStats)
		b.startAllocs = memStats.Mallocs
		b.startBytes = memStats.TotalAlloc
		b.start = highPrecisionTimeNow()
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

// runN runs a single benchmark for the specified number of iterations.
func (b *B) runN(n int) {
	benchmarkLock.Lock()
	defer benchmarkLock.Unlock()
	ctx, cancelCtx := context.WithCancel(context.Background())
	defer func() {
		b.runCleanup(normalPanic)
		b.checkRaces()
	}()
	// Try to get a comparable environment for each run
	// by clearing garbage from previous runs.
	runtime.GC()
	b.resetRaces()
	b.N = n
	b.loopN = 0
	b.ctx = ctx
	b.cancelCtx = cancelCtx

	b.parallelism = 1
	b.ResetTimer()
	b.StartTimer()
	b.benchFunc(b)
	b.StopTimer()
	b.previousN = n
	b.previousDuration = b.duration
}

// run1 runs the first iteration of benchFunc. It reports whether more
// iterations of this benchmarks should be run.
func (b *B) run1() bool {
	if bstate := b.bstate; bstate != nil {
		// Extend maxLen, if needed.
		if n := len(b.name) + bstate.extLen + 1; n > bstate.maxLen {
			bstate.maxLen = n + 8 // Add additional slack to avoid too many jumps in size.
		}
	}
	go func() {
		// Signal that we're done whether we return normally
		// or by FailNow's runtime.Goexit.
		defer func() {
			b.signal <- true
		}()

		b.runN(1)
	}()
	<-b.signal
	if b.failed {
		fmt.Fprintf(b.w, "%s--- FAIL: %s\n%s", b.chatty.prefix(), b.name, b.output)
		return false
	}
	// Only print the output if we know we are not going to proceed.
	// Otherwise it is printed in processBench.
	b.mu.RLock()
	finished := b.finished
	b.mu.RUnlock()
	if b.hasSub.Load() || finished {
		tag := "BENCH"
		if b.skipped {
			tag = "SKIP"
		}
		if b.chatty != nil && (len(b.output) > 0 || finished) {
			b.trimOutput()
			fmt.Fprintf(b.w, "%s--- %s: %s\n%s", b.chatty.prefix(), tag, b.name, b.output)
		}
		return false
	}
	return true
}

var labelsOnce sync.Once

// run executes the benchmark in a separate goroutine, including all of its
// subbenchmarks. b must not have subbenchmarks.
func (b *B) run() {
	labelsOnce.Do(func() {
		fmt.Fprintf(b.w, "goos: %s\n", runtime.GOOS)
		fmt.Fprintf(b.w, "goarch: %s\n", runtime.GOARCH)
		if b.importPath != "" {
			fmt.Fprintf(b.w, "pkg: %s\n", b.importPath)
		}
		if cpu := sysinfo.CPUName(); cpu != "" {
			fmt.Fprintf(b.w, "cpu: %s\n", cpu)
		}
	})
	if b.bstate != nil {
		// Running go test --test.bench
		b.bstate.processBench(b) // Must call doBench.
	} else {
		// Running func Benchmark.
		b.doBench()
	}
}

func (b *B) doBench() BenchmarkResult {
	go b.launch()
	<-b.signal
	return b.result
}

func predictN(goalns int64, prevIters int64, prevns int64, last int64) int {
	if prevns == 0 {
		// Round up to dodge divide by zero. See https://go.dev/issue/70709.
		prevns = 1
	}

	// Order of operations matters.
	// For very fast benchmarks, prevIters ~= prevns.
	// If you divide first, you get 0 or 1,
	// which can hide an order of magnitude in execution time.
	// So multiply first, then divide.
	n := goalns * prevIters / prevns
	// Run more iterations than we think we'll need (1.2x).
	n += n / 5
	// Don't grow too fast in case we had timing errors previously.
	n = min(n, 100*last)
	// Be sure to run at least one more than last time.
	n = max(n, last+1)
	// Don't run more than 1e9 times. (This also keeps n in int range on 32 bit platforms.)
	n = min(n, 1e9)
	return int(n)
}

// launch launches the benchmark function. It gradually increases the number
// of benchmark iterations until the benchmark runs for the requested benchtime.
// launch is run by the doBench function as a separate goroutine.
// run1 must have been called on b.
func (b *B) launch() {
	// Signal that we're done whether we return normally
	// or by FailNow's runtime.Goexit.
	defer func() {
		b.signal <- true
	}()

	// b.Loop does its own ramp-up logic so we just need to run it once.
	// If b.loopN is non zero, it means b.Loop has already run.
	if b.loopN == 0 {
		// Run the benchmark for at least the specified amount of time.
		if b.benchTime.n > 0 {
			// We already ran a single iteration in run1.
			// If -benchtime=1x was requested, use that result.
			// See https://golang.org/issue/32051.
			if b.benchTime.n > 1 {
				b.runN(b.benchTime.n)
			}
		} else {
			d := b.benchTime.d
			for n := int64(1); !b.failed && b.duration < d && n < 1e9; {
				last := n
				// Predict required iterations.
				goalns := d.Nanoseconds()
				prevIters := int64(b.N)
				n = int64(predictN(goalns, prevIters, b.duration.Nanoseconds(), last))
				b.runN(int(n))
			}
		}
	}
	b.result = BenchmarkResult{b.N, b.duration, b.bytes, b.netAllocs, b.netBytes, b.extra}
}

// Elapsed returns the measured elapsed time of the benchmark.
// The duration reported by Elapsed matches the one measured by
// [B.StartTimer], [B.StopTimer], and [B.ResetTimer].
func (b *B) Elapsed() time.Duration {
	d := b.duration
	if b.timerOn {
		d += highPrecisionTimeSince(b.start)
	}
	return d
}

// ReportMetric adds "n unit" to the reported benchmark results.
// If the metric is per-iteration, the caller should divide by b.N,
// and by convention units should end in "/op".
// ReportMetric overrides any previously reported value for the same unit.
// ReportMetric panics if unit is the empty string or if unit contains
// any whitespace.
// If unit is a unit normally reported by the benchmark framework itself
// (such as "allocs/op"), ReportMetric will override that metric.
// Setting "ns/op" to 0 will suppress that built-in metric.
func (b *B) ReportMetric(n float64, unit string) {
	if unit == "" {
		panic("metric unit must not be empty")
	}
	if strings.IndexFunc(unit, unicode.IsSpace) >= 0 {
		panic("metric unit must not contain whitespace")
	}
	b.extra[unit] = n
}

func (b *B) stopOrScaleBLoop() bool {
	timeElapsed := highPrecisionTimeSince(b.start)
	if timeElapsed >= b.benchTime.d {
		// Stop the timer so we don't count cleanup time
		b.StopTimer()
		return false
	}
	// Loop scaling
	goalns := b.benchTime.d.Nanoseconds()
	prevIters := int64(b.N)
	b.N = predictN(goalns, prevIters, timeElapsed.Nanoseconds(), prevIters)
	b.loopN++
	return true
}

func (b *B) loopSlowPath() bool {
	if b.loopN == 0 {
		// If it's the first call to b.Loop() in the benchmark function.
		// Allows more precise measurement of benchmark loop cost counts.
		// Also initialize b.N to 1 to kick start loop scaling.
		b.N = 1
		b.loopN = 1
		b.ResetTimer()
		return true
	}
	// Handles fixed iterations case
	if b.benchTime.n > 0 {
		if b.N < b.benchTime.n {
			b.N = b.benchTime.n
			b.loopN++
			return true
		}
		b.StopTimer()
		return false
	}
	// Handles fixed time case
	return b.stopOrScaleBLoop()
}

// Loop returns true as long as the benchmark should continue running.
//
// A typical benchmark is structured like:
//
//	func Benchmark(b *testing.B) {
//		... setup ...
//		for b.Loop() {
//			... code to measure ...
//		}
//		... cleanup ...
//	}
//
// Loop resets the benchmark timer the first time it is called in a benchmark,
// so any setup performed prior to starting the benchmark loop does not count
// toward the benchmark measurement. Likewise, when it returns false, it stops
// the timer so cleanup code is not measured.
//
// The compiler never optimizes away calls to functions within the body of a
// "for b.Loop() { ... }" loop. This prevents surprises that can otherwise occur
// if the compiler determines that the result of a benchmarked function is
// unused. The loop must be written in exactly this form, and this only applies
// to calls syntactically between the curly braces of the loop. Optimizations
// are performed as usual in any functions called by the loop.
//
// After Loop returns false, b.N contains the total number of iterations that
// ran, so the benchmark may use b.N to compute other average metrics.
//
// Prior to the introduction of Loop, benchmarks were expected to contain an
// explicit loop from 0 to b.N. Benchmarks should either use Loop or contain a
// loop to b.N, but not both. Loop offers more automatic management of the
// benchmark timer, and runs each benchmark function only once per measurement,
// whereas b.N-based benchmarks must run the benchmark function (and any
// associated setup and cleanup) several times.
func (b *B) Loop() bool {
	if b.loopN != 0 && b.loopN < b.N {
		b.loopN++
		return true
	}
	return b.loopSlowPath()
}

// BenchmarkResult contains the results of a benchmark run.
type BenchmarkResult struct {
	N         int           // The number of iterations.
	T         time.Duration // The total time taken.
	Bytes     int64         // Bytes processed in one iteration.
	MemAllocs uint64        // The total number of memory allocations.
	MemBytes  uint64        // The total number of bytes allocated.

	// Extra records additional metrics reported by ReportMetric.
	Extra map[string]float64
}

// NsPerOp returns the "ns/op" metric.
func (r BenchmarkResult) NsPerOp() int64 {
	if v, ok := r.Extra["ns/op"]; ok {
		return int64(v)
	}
	if r.N <= 0 {
		return 0
	}
	return r.T.Nanoseconds() / int64(r.N)
}

// mbPerSec returns the "MB/s" metric.
func (r BenchmarkResult) mbPerSec() float64 {
	if v, ok := r.Extra["MB/s"]; ok {
		return v
	}
	if r.Bytes <= 0 || r.T <= 0 || r.N <= 0 {
		return 0
	}
	return (float64(r.Bytes) * float64(r.N) / 1e6) / r.T.Seconds()
}

// AllocsPerOp returns the "allocs/op" metric,
// which is calculated as r.MemAllocs / r.N.
func (r BenchmarkResult) AllocsPerOp() int64 {
	if v, ok := r.Extra["allocs/op"]; ok {
		return int64(v)
	}
	if r.N <= 0 {
		return 0
	}
	return int64(r.MemAllocs) / int64(r.N)
}

// AllocedBytesPerOp returns the "B/op" metric,
// which is calculated as r.MemBytes / r.N.
func (r BenchmarkResult) AllocedBytesPerOp() int64 {
	if v, ok := r.Extra["B/op"]; ok {
		return int64(v)
	}
	if r.N <= 0 {
		return 0
	}
	return int64(r.MemBytes) / int64(r.N)
}

// String returns a summary of the benchmark results.
// It follows the benchmark result line format from
// https://golang.org/design/14313-benchmark-format, not including the
// benchmark name.
// Extra metrics override built-in metrics of the same name.
// String does not include allocs/op or B/op, since those are reported
// by [BenchmarkResult.MemString].
func (r BenchmarkResult) String() string {
	buf := new(strings.Builder)
	fmt.Fprintf(buf, "%8d", r.N)

	// Get ns/op as a float.
	ns, ok := r.Extra["ns/op"]
	if !ok {
		ns = float64(r.T.Nanoseconds()) / float64(r.N)
	}
	if ns != 0 {
		buf.WriteByte('\t')
		prettyPrint(buf, ns, "ns/op")
	}

	if mbs := r.mbPerSec(); mbs != 0 {
		fmt.Fprintf(buf, "\t%7.2f MB/s", mbs)
	}

	// Print extra metrics that aren't represented in the standard
	// metrics.
	var extraKeys []string
	for k := range r.Extra {
		switch k {
		case "ns/op", "MB/s", "B/op", "allocs/op":
			// Built-in metrics reported elsewhere.
			continue
		}
		extraKeys = append(extraKeys, k)
	}
	slices.Sort(extraKeys)
	for _, k := range extraKeys {
		buf.WriteByte('\t')
		prettyPrint(buf, r.Extra[k], k)
	}
	return buf.String()
}

func prettyPrint(w io.Writer, x float64, unit string) {
	// Print all numbers with 10 places before the decimal point
	// and small numbers with four sig figs. Field widths are
	// chosen to fit the whole part in 10 places while aligning
	// the decimal point of all fractional formats.
	var format string
	switch y := math.Abs(x); {
	case y == 0 || y >= 999.95:
		format = "%10.0f %s"
	case y >= 99.995:
		format = "%12.1f %s"
	case y >= 9.9995:
		format = "%13.2f %s"
	case y >= 0.99995:
		format = "%14.3f %s"
	case y >= 0.099995:
		format = "%15.4f %s"
	case y >= 0.0099995:
		format = "%16.5f %s"
	case y >= 0.00099995:
		format = "%17.6f %s"
	default:
		format = "%18.7f %s"
	}
	fmt.Fprintf(w, format, x, unit)
}

// MemString returns r.AllocedBytesPerOp and r.AllocsPerOp in the same format as 'go test'.
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

type benchState struct {
	match *matcher

	maxLen int // The largest recorded benchmark name.
	extLen int // Maximum extension length.
}

// RunBenchmarks is an internal function but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
func RunBenchmarks(matchString func(pat, str string) (bool, error), benchmarks []InternalBenchmark) {
	runBenchmarks("", matchString, benchmarks)
}

func runBenchmarks(importPath string, matchString func(pat, str string) (bool, error), benchmarks []InternalBenchmark) bool {
	// If no flag was specified, don't run benchmarks.
	if len(*matchBenchmarks) == 0 {
		return true
	}
	// Collect matching benchmarks and determine longest name.
	maxprocs := 1
	for _, procs := range cpuList {
		if procs > maxprocs {
			maxprocs = procs
		}
	}
	bstate := &benchState{
		match:  newMatcher(matchString, *matchBenchmarks, "-test.bench", *skip),
		extLen: len(benchmarkName("", maxprocs)),
	}
	var bs []InternalBenchmark
	for _, Benchmark := range benchmarks {
		if _, matched, _ := bstate.match.fullName(nil, Benchmark.Name); matched {
			bs = append(bs, Benchmark)
			benchName := benchmarkName(Benchmark.Name, maxprocs)
			if l := len(benchName) + bstate.extLen + 1; l > bstate.maxLen {
				bstate.maxLen = l
			}
		}
	}
	main := &B{
		common: common{
			name:  "Main",
			w:     os.Stdout,
			bench: true,
		},
		importPath: importPath,
		benchFunc: func(b *B) {
			for _, Benchmark := range bs {
				b.Run(Benchmark.Name, Benchmark.F)
			}
		},
		benchTime: benchTime,
		bstate:    bstate,
	}
	if Verbose() {
		main.chatty = newChattyPrinter(main.w)
	}
	main.runN(1)
	return !main.failed
}

// processBench runs bench b for the configured CPU counts and prints the results.
func (s *benchState) processBench(b *B) {
	for i, procs := range cpuList {
		for j := uint(0); j < *count; j++ {
			runtime.GOMAXPROCS(procs)
			benchName := benchmarkName(b.name, procs)

			// If it's chatty, we've already printed this information.
			if b.chatty == nil {
				fmt.Fprintf(b.w, "%-*s\t", s.maxLen, benchName)
			}
			// Recompute the running time for all but the first iteration.
			if i > 0 || j > 0 {
				b = &B{
					common: common{
						signal: make(chan bool),
						name:   b.name,
						w:      b.w,
						chatty: b.chatty,
						bench:  true,
					},
					benchFunc: b.benchFunc,
					benchTime: b.benchTime,
				}
				b.run1()
			}
			r := b.doBench()
			if b.failed {
				// The output could be very long here, but probably isn't.
				// We print it all, regardless, because we don't want to trim the reason
				// the benchmark failed.
				fmt.Fprintf(b.w, "%s--- FAIL: %s\n%s", b.chatty.prefix(), benchName, b.output)
				continue
			}
			results := r.String()
			if b.chatty != nil {
				fmt.Fprintf(b.w, "%-*s\t", s.maxLen, benchName)
			}
			if *benchmarkMemory || b.showAllocResult {
				results += "\t" + r.MemString()
			}
			fmt.Fprintln(b.w, results)
			// Unlike with tests, we ignore the -chatty flag and always print output for
			// benchmarks since the output generation time will skew the results.
			if len(b.output) > 0 {
				b.trimOutput()
				fmt.Fprintf(b.w, "%s--- BENCH: %s\n%s", b.chatty.prefix(), benchName, b.output)
			}
			if p := runtime.GOMAXPROCS(-1); p != procs {
				fmt.Fprintf(os.Stderr, "testing: %s left GOMAXPROCS set to %d\n", benchName, p)
			}
			if b.chatty != nil && b.chatty.json {
				b.chatty.Updatef("", "=== NAME  %s\n", "")
			}
		}
	}
}

// If hideStdoutForTesting is true, Run does not print the benchName.
// This avoids a spurious print during 'go test' on package testing itself,
// which invokes b.Run in its own tests (see sub_test.go).
var hideStdoutForTesting = false

// Run benchmarks f as a subbenchmark with the given name. It reports
// whether there were any failures.
//
// A subbenchmark is like any other benchmark. A benchmark that calls Run at
// least once will not be measured itself and will be called once with N=1.
func (b *B) Run(name string, f func(b *B)) bool {
	// Since b has subbenchmarks, we will no longer run it as a benchmark itself.
	// Release the lock and acquire it on exit to ensure locks stay paired.
	b.hasSub.Store(true)
	benchmarkLock.Unlock()
	defer benchmarkLock.Lock()

	benchName, ok, partial := b.name, true, false
	if b.bstate != nil {
		benchName, ok, partial = b.bstate.match.fullName(&b.common, name)
	}
	if !ok {
		return true
	}
	var pc [maxStackLen]uintptr
	n := runtime.Callers(2, pc[:])
	sub := &B{
		common: common{
			signal:  make(chan bool),
			name:    benchName,
			parent:  &b.common,
			level:   b.level + 1,
			creator: pc[:n],
			w:       b.w,
			chatty:  b.chatty,
			bench:   true,
		},
		importPath: b.importPath,
		benchFunc:  f,
		benchTime:  b.benchTime,
		bstate:     b.bstate,
	}
	if partial {
		// Partial name match, like -bench=X/Y matching BenchmarkX.
		// Only process sub-benchmarks, if any.
		sub.hasSub.Store(true)
	}

	if b.chatty != nil {
		labelsOnce.Do(func() {
			fmt.Printf("goos: %s\n", runtime.GOOS)
			fmt.Printf("goarch: %s\n", runtime.GOARCH)
			if b.importPath != "" {
				fmt.Printf("pkg: %s\n", b.importPath)
			}
			if cpu := sysinfo.CPUName(); cpu != "" {
				fmt.Printf("cpu: %s\n", cpu)
			}
		})

		if !hideStdoutForTesting {
			if b.chatty.json {
				b.chatty.Updatef(benchName, "=== RUN   %s\n", benchName)
			}
			fmt.Println(benchName)
		}
	}

	if sub.run1() {
		sub.run()
	}
	b.add(sub.result)
	return !sub.failed
}

// add simulates running benchmarks in sequence in a single iteration. It is
// used to give some meaningful results in case func Benchmark is used in
// combination with Run.
func (b *B) add(other BenchmarkResult) {
	r := &b.result
	// The aggregated BenchmarkResults resemble running all subbenchmarks as
	// in sequence in a single benchmark.
	r.N = 1
	r.T += time.Duration(other.NsPerOp())
	if other.Bytes == 0 {
		// Summing Bytes is meaningless in aggregate if not all subbenchmarks
		// set it.
		b.missingBytes = true
		r.Bytes = 0
	}
	if !b.missingBytes {
		r.Bytes += other.Bytes
	}
	r.MemAllocs += uint64(other.AllocsPerOp())
	r.MemBytes += uint64(other.AllocedBytesPerOp())
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
	globalN *atomic.Uint64 // shared between all worker goroutines iteration counter
	grain   uint64         // acquire that many iterations from globalN at once
	cache   uint64         // local cache of acquired iterations
	bN      uint64         // total number of iterations to execute (b.N)
}

// Next reports whether there are more iterations to execute.
func (pb *PB) Next() bool {
	if pb.cache == 0 {
		n := pb.globalN.Add(pb.grain)
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
// non-CPU-bound benchmarks, call [B.SetParallelism] before RunParallel.
// RunParallel is usually used with the go test -cpu flag.
//
// The body function will be run in each goroutine. It should set up any
// goroutine-local state and then iterate until pb.Next returns false.
// It should not use the [B.StartTimer], [B.StopTimer], or [B.ResetTimer] functions,
// because they have global effect. It should also not call [B.Run].
//
// RunParallel reports ns/op values as wall time for the benchmark as a whole,
// not the sum of wall time or CPU time over each parallel goroutine.
func (b *B) RunParallel(body func(*PB)) {
	if b.N == 0 {
		return // Nothing to do when probing.
	}
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

	var n atomic.Uint64
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
	if n.Load() <= uint64(b.N) && !b.Failed() {
		b.Fatal("RunParallel: body exited without pb.Next() == false")
	}
}

// SetParallelism sets the number of goroutines used by [B.RunParallel] to p*GOMAXPROCS.
// There is usually no need to call SetParallelism for CPU-bound benchmarks.
// If p is less than 1, this call will have no effect.
func (b *B) SetParallelism(p int) {
	if p >= 1 {
		b.parallelism = p
	}
}

// Benchmark benchmarks a single function. It is useful for creating
// custom benchmarks that do not use the "go test" command.
//
// If f depends on testing flags, then [Init] must be used to register
// those flags before calling Benchmark and before calling [flag.Parse].
//
// If f calls Run, the result will be an estimate of running all its
// subbenchmarks that don't call Run in sequence in a single benchmark.
func Benchmark(f func(b *B)) BenchmarkResult {
	b := &B{
		common: common{
			signal: make(chan bool),
			w:      discard{},
		},
		benchFunc: f,
		benchTime: benchTime,
	}
	if b.run1() {
		b.run()
	}
	return b.result
}

type discard struct{}

func (discard) Write(b []byte) (n int, err error) { return len(b), nil }
