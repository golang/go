// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testing provides support for automated testing of Go packages.
// It is intended to be used in concert with the ``go test'' command, which automates
// execution of any function of the form
//     func TestXxx(*testing.T)
// where Xxx can be any alphanumeric string (but the first letter must not be in
// [a-z]) and serves to identify the test routine.
//
// Within these functions, use the Error, Fail or related methods to signal failure.
//
// To write a new test suite, create a file whose name ends _test.go that
// contains the TestXxx functions as described here. Put the file in the same
// package as the one being tested. The file will be excluded from regular
// package builds but will be included when the ``go test'' command is run.
// For more detail, run ``go help test'' and ``go help testflag''.
//
// Tests and benchmarks may be skipped if not applicable with a call to
// the Skip method of *T and *B:
//     func TestTimeConsuming(t *testing.T) {
//         if testing.Short() {
//             t.Skip("skipping test in short mode.")
//         }
//         ...
//     }
//
// Benchmarks
//
// Functions of the form
//     func BenchmarkXxx(*testing.B)
// are considered benchmarks, and are executed by the "go test" command when
// its -bench flag is provided. Benchmarks are run sequentially.
//
// For a description of the testing flags, see
// https://golang.org/cmd/go/#hdr-Description_of_testing_flags.
//
// A sample benchmark function looks like this:
//     func BenchmarkHello(b *testing.B) {
//         for i := 0; i < b.N; i++ {
//             fmt.Sprintf("hello")
//         }
//     }
//
// The benchmark function must run the target code b.N times.
// During benchmark execution, b.N is adjusted until the benchmark function lasts
// long enough to be timed reliably.  The output
//     BenchmarkHello    10000000    282 ns/op
// means that the loop ran 10000000 times at a speed of 282 ns per loop.
//
// If a benchmark needs some expensive setup before running, the timer
// may be reset:
//
//     func BenchmarkBigLen(b *testing.B) {
//         big := NewBig()
//         b.ResetTimer()
//         for i := 0; i < b.N; i++ {
//             big.Len()
//         }
//     }
//
// If a benchmark needs to test performance in a parallel setting, it may use
// the RunParallel helper function; such benchmarks are intended to be used with
// the go test -cpu flag:
//
//     func BenchmarkTemplateParallel(b *testing.B) {
//         templ := template.Must(template.New("test").Parse("Hello, {{.}}!"))
//         b.RunParallel(func(pb *testing.PB) {
//             var buf bytes.Buffer
//             for pb.Next() {
//                 buf.Reset()
//                 templ.Execute(&buf, "World")
//             }
//         })
//     }
//
// Examples
//
// The package also runs and verifies example code. Example functions may
// include a concluding line comment that begins with "Output:" and is compared with
// the standard output of the function when the tests are run. (The comparison
// ignores leading and trailing space.) These are examples of an example:
//
//     func ExampleHello() {
//             fmt.Println("hello")
//             // Output: hello
//     }
//
//     func ExampleSalutations() {
//             fmt.Println("hello, and")
//             fmt.Println("goodbye")
//             // Output:
//             // hello, and
//             // goodbye
//     }
//
// Example functions without output comments are compiled but not executed.
//
// The naming convention to declare examples for the package, a function F, a type T and
// method M on type T are:
//
//     func Example() { ... }
//     func ExampleF() { ... }
//     func ExampleT() { ... }
//     func ExampleT_M() { ... }
//
// Multiple example functions for a package/type/function/method may be provided by
// appending a distinct suffix to the name. The suffix must start with a
// lower-case letter.
//
//     func Example_suffix() { ... }
//     func ExampleF_suffix() { ... }
//     func ExampleT_suffix() { ... }
//     func ExampleT_M_suffix() { ... }
//
// The entire test file is presented as the example when it contains a single
// example function, at least one other function, type, variable, or constant
// declaration, and no test or benchmark functions.
//
// Main
//
// It is sometimes necessary for a test program to do extra setup or teardown
// before or after testing. It is also sometimes necessary for a test to control
// which code runs on the main thread. To support these and other cases,
// if a test file contains a function:
//
//	func TestMain(m *testing.M)
//
// then the generated test will call TestMain(m) instead of running the tests
// directly. TestMain runs in the main goroutine and can do whatever setup
// and teardown is necessary around a call to m.Run. It should then call
// os.Exit with the result of m.Run. When TestMain is called, flag.Parse has
// not been run. If TestMain depends on command-line flags, including those
// of the testing package, it should call flag.Parse explicitly.
//
// A simple implementation of TestMain is:
//
//	func TestMain(m *testing.M) {
//		flag.Parse()
//		os.Exit(m.Run())
//	}
//
package testing

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"runtime/trace"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	// The short flag requests that tests run more quickly, but its functionality
	// is provided by test writers themselves.  The testing package is just its
	// home.  The all.bash installation script sets it to make installation more
	// efficient, but by default the flag is off so a plain "go test" will do a
	// full test of the package.
	short = flag.Bool("test.short", false, "run smaller test suite to save time")

	// The directory in which to create profile files and the like. When run from
	// "go test", the binary always runs in the source directory for the package;
	// this flag lets "go test" tell the binary to write the files in the directory where
	// the "go test" command is run.
	outputDir = flag.String("test.outputdir", "", "directory in which to write profiles")

	// Report as tests are run; default is silent for success.
	chatty           = flag.Bool("test.v", false, "verbose: print additional output")
	count            = flag.Uint("test.count", 1, "run tests and benchmarks `n` times")
	coverProfile     = flag.String("test.coverprofile", "", "write a coverage profile to the named file after execution")
	match            = flag.String("test.run", "", "regular expression to select tests and examples to run")
	memProfile       = flag.String("test.memprofile", "", "write a memory profile to the named file after execution")
	memProfileRate   = flag.Int("test.memprofilerate", 0, "if >=0, sets runtime.MemProfileRate")
	cpuProfile       = flag.String("test.cpuprofile", "", "write a cpu profile to the named file during execution")
	blockProfile     = flag.String("test.blockprofile", "", "write a goroutine blocking profile to the named file after execution")
	blockProfileRate = flag.Int("test.blockprofilerate", 1, "if >= 0, calls runtime.SetBlockProfileRate()")
	traceFile        = flag.String("test.trace", "", "write an execution trace to the named file after execution")
	timeout          = flag.Duration("test.timeout", 0, "if positive, sets an aggregate time limit for all tests")
	cpuListStr       = flag.String("test.cpu", "", "comma-separated list of number of CPUs to use for each test")
	parallel         = flag.Int("test.parallel", runtime.GOMAXPROCS(0), "maximum test parallelism")

	haveExamples bool // are there examples?

	cpuList []int
)

// common holds the elements common between T and B and
// captures common methods such as Errorf.
type common struct {
	mu       sync.RWMutex // guards output and failed
	output   []byte       // Output generated by test or benchmark.
	failed   bool         // Test or benchmark has failed.
	skipped  bool         // Test of benchmark has been skipped.
	finished bool

	start    time.Time // Time test or benchmark started
	duration time.Duration
	self     interface{}      // To be sent on signal channel when done.
	signal   chan interface{} // Output for serial tests.
}

// Short reports whether the -test.short flag is set.
func Short() bool {
	return *short
}

// Verbose reports whether the -test.v flag is set.
func Verbose() bool {
	return *chatty
}

// decorate prefixes the string with the file and line of the call site
// and inserts the final newline if needed and indentation tabs for formatting.
func decorate(s string) string {
	_, file, line, ok := runtime.Caller(3) // decorate + log + public function.
	if ok {
		// Truncate file name at last file name separator.
		if index := strings.LastIndex(file, "/"); index >= 0 {
			file = file[index+1:]
		} else if index = strings.LastIndex(file, "\\"); index >= 0 {
			file = file[index+1:]
		}
	} else {
		file = "???"
		line = 1
	}
	buf := new(bytes.Buffer)
	// Every line is indented at least one tab.
	buf.WriteByte('\t')
	fmt.Fprintf(buf, "%s:%d: ", file, line)
	lines := strings.Split(s, "\n")
	if l := len(lines); l > 1 && lines[l-1] == "" {
		lines = lines[:l-1]
	}
	for i, line := range lines {
		if i > 0 {
			// Second and subsequent lines are indented an extra tab.
			buf.WriteString("\n\t\t")
		}
		buf.WriteString(line)
	}
	buf.WriteByte('\n')
	return buf.String()
}

// fmtDuration returns a string representing d in the form "87.00s".
func fmtDuration(d time.Duration) string {
	return fmt.Sprintf("%.2fs", d.Seconds())
}

// TB is the interface common to T and B.
type TB interface {
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Skip(args ...interface{})
	SkipNow()
	Skipf(format string, args ...interface{})
	Skipped() bool

	// A private method to prevent users implementing the
	// interface and so future additions to it will not
	// violate Go 1 compatibility.
	private()
}

var _ TB = (*T)(nil)
var _ TB = (*B)(nil)

// T is a type passed to Test functions to manage test state and support formatted test logs.
// Logs are accumulated during execution and dumped to standard error when done.
//
// A test ends when its Test function returns or calls any of the methods
// FailNow, Fatal, Fatalf, SkipNow, Skip, or Skipf. Those methods, as well as
// the Parallel method, must be called only from the goroutine running the
// Test function.
//
// The other reporting methods, such as the variations of Log and Error,
// may be called simultaneously from multiple goroutines.
type T struct {
	common
	name          string // Name of test.
	isParallel    bool
	startParallel chan bool // Parallel tests will wait on this.
}

func (c *common) private() {}

// Fail marks the function as having failed but continues execution.
func (c *common) Fail() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failed = true
}

// Failed reports whether the function has failed.
func (c *common) Failed() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.failed
}

// FailNow marks the function as having failed and stops its execution.
// Execution will continue at the next test or benchmark.
// FailNow must be called from the goroutine running the
// test or benchmark function, not from other goroutines
// created during the test. Calling FailNow does not stop
// those other goroutines.
func (c *common) FailNow() {
	c.Fail()

	// Calling runtime.Goexit will exit the goroutine, which
	// will run the deferred functions in this goroutine,
	// which will eventually run the deferred lines in tRunner,
	// which will signal to the test loop that this test is done.
	//
	// A previous version of this code said:
	//
	//	c.duration = ...
	//	c.signal <- c.self
	//	runtime.Goexit()
	//
	// This previous version duplicated code (those lines are in
	// tRunner no matter what), but worse the goroutine teardown
	// implicit in runtime.Goexit was not guaranteed to complete
	// before the test exited.  If a test deferred an important cleanup
	// function (like removing temporary files), there was no guarantee
	// it would run on a test failure.  Because we send on c.signal during
	// a top-of-stack deferred function now, we know that the send
	// only happens after any other stacked defers have completed.
	c.finished = true
	runtime.Goexit()
}

// log generates the output. It's always at the same stack depth.
func (c *common) log(s string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.output = append(c.output, decorate(s)...)
}

// Log formats its arguments using default formatting, analogous to Println,
// and records the text in the error log. For tests, the text will be printed only if
// the test fails or the -test.v flag is set. For benchmarks, the text is always
// printed to avoid having performance depend on the value of the -test.v flag.
func (c *common) Log(args ...interface{}) { c.log(fmt.Sprintln(args...)) }

// Logf formats its arguments according to the format, analogous to Printf,
// and records the text in the error log. For tests, the text will be printed only if
// the test fails or the -test.v flag is set. For benchmarks, the text is always
// printed to avoid having performance depend on the value of the -test.v flag.
func (c *common) Logf(format string, args ...interface{}) { c.log(fmt.Sprintf(format, args...)) }

// Error is equivalent to Log followed by Fail.
func (c *common) Error(args ...interface{}) {
	c.log(fmt.Sprintln(args...))
	c.Fail()
}

// Errorf is equivalent to Logf followed by Fail.
func (c *common) Errorf(format string, args ...interface{}) {
	c.log(fmt.Sprintf(format, args...))
	c.Fail()
}

// Fatal is equivalent to Log followed by FailNow.
func (c *common) Fatal(args ...interface{}) {
	c.log(fmt.Sprintln(args...))
	c.FailNow()
}

// Fatalf is equivalent to Logf followed by FailNow.
func (c *common) Fatalf(format string, args ...interface{}) {
	c.log(fmt.Sprintf(format, args...))
	c.FailNow()
}

// Skip is equivalent to Log followed by SkipNow.
func (c *common) Skip(args ...interface{}) {
	c.log(fmt.Sprintln(args...))
	c.SkipNow()
}

// Skipf is equivalent to Logf followed by SkipNow.
func (c *common) Skipf(format string, args ...interface{}) {
	c.log(fmt.Sprintf(format, args...))
	c.SkipNow()
}

// SkipNow marks the test as having been skipped and stops its execution.
// Execution will continue at the next test or benchmark. See also FailNow.
// SkipNow must be called from the goroutine running the test, not from
// other goroutines created during the test. Calling SkipNow does not stop
// those other goroutines.
func (c *common) SkipNow() {
	c.skip()
	c.finished = true
	runtime.Goexit()
}

func (c *common) skip() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.skipped = true
}

// Skipped reports whether the test was skipped.
func (c *common) Skipped() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.skipped
}

// Parallel signals that this test is to be run in parallel with (and only with)
// other parallel tests.
func (t *T) Parallel() {
	if t.isParallel {
		panic("testing: t.Parallel called multiple times")
	}
	t.isParallel = true

	// We don't want to include the time we spend waiting for serial tests
	// in the test duration. Record the elapsed time thus far and reset the
	// timer afterwards.
	t.duration += time.Since(t.start)
	t.signal <- (*T)(nil) // Release main testing loop
	<-t.startParallel     // Wait for serial tests to finish
	t.start = time.Now()
}

// An internal type but exported because it is cross-package; part of the implementation
// of the "go test" command.
type InternalTest struct {
	Name string
	F    func(*T)
}

func tRunner(t *T, test *InternalTest) {
	// When this goroutine is done, either because test.F(t)
	// returned normally or because a test failure triggered
	// a call to runtime.Goexit, record the duration and send
	// a signal saying that the test is done.
	defer func() {
		t.duration += time.Now().Sub(t.start)
		// If the test panicked, print any test output before dying.
		err := recover()
		if !t.finished && err == nil {
			err = fmt.Errorf("test executed panic(nil) or runtime.Goexit")
		}
		if err != nil {
			t.Fail()
			t.report()
			panic(err)
		}
		t.signal <- t
	}()

	t.start = time.Now()
	test.F(t)
	t.finished = true
}

// An internal function but exported because it is cross-package; part of the implementation
// of the "go test" command.
func Main(matchString func(pat, str string) (bool, error), tests []InternalTest, benchmarks []InternalBenchmark, examples []InternalExample) {
	os.Exit(MainStart(matchString, tests, benchmarks, examples).Run())
}

// M is a type passed to a TestMain function to run the actual tests.
type M struct {
	matchString func(pat, str string) (bool, error)
	tests       []InternalTest
	benchmarks  []InternalBenchmark
	examples    []InternalExample
}

// MainStart is meant for use by tests generated by 'go test'.
// It is not meant to be called directly and is not subject to the Go 1 compatibility document.
// It may change signature from release to release.
func MainStart(matchString func(pat, str string) (bool, error), tests []InternalTest, benchmarks []InternalBenchmark, examples []InternalExample) *M {
	return &M{
		matchString: matchString,
		tests:       tests,
		benchmarks:  benchmarks,
		examples:    examples,
	}
}

// Run runs the tests. It returns an exit code to pass to os.Exit.
func (m *M) Run() int {
	// TestMain may have already called flag.Parse.
	if !flag.Parsed() {
		flag.Parse()
	}

	parseCpuList()

	before()
	startAlarm()
	haveExamples = len(m.examples) > 0
	testOk := RunTests(m.matchString, m.tests)
	exampleOk := RunExamples(m.matchString, m.examples)
	stopAlarm()
	if !testOk || !exampleOk {
		fmt.Println("FAIL")
		after()
		return 1
	}
	fmt.Println("PASS")
	RunBenchmarks(m.matchString, m.benchmarks)
	after()
	return 0
}

func (t *T) report() {
	dstr := fmtDuration(t.duration)
	format := "--- %s: %s (%s)\n%s"
	if t.Failed() {
		fmt.Printf(format, "FAIL", t.name, dstr, t.output)
	} else if *chatty {
		if t.Skipped() {
			fmt.Printf(format, "SKIP", t.name, dstr, t.output)
		} else {
			fmt.Printf(format, "PASS", t.name, dstr, t.output)
		}
	}
}

func RunTests(matchString func(pat, str string) (bool, error), tests []InternalTest) (ok bool) {
	ok = true
	if len(tests) == 0 && !haveExamples {
		fmt.Fprintln(os.Stderr, "testing: warning: no tests to run")
		return
	}
	for _, procs := range cpuList {
		runtime.GOMAXPROCS(procs)
		// We build a new channel tree for each run of the loop.
		// collector merges in one channel all the upstream signals from parallel tests.
		// If all tests pump to the same channel, a bug can occur where a test
		// kicks off a goroutine that Fails, yet the test still delivers a completion signal,
		// which skews the counting.
		var collector = make(chan interface{})

		numParallel := 0
		startParallel := make(chan bool)

		for i := 0; i < len(tests); i++ {
			matched, err := matchString(*match, tests[i].Name)
			if err != nil {
				fmt.Fprintf(os.Stderr, "testing: invalid regexp for -test.run: %s\n", err)
				os.Exit(1)
			}
			if !matched {
				continue
			}
			testName := tests[i].Name
			t := &T{
				common: common{
					signal: make(chan interface{}),
				},
				name:          testName,
				startParallel: startParallel,
			}
			t.self = t
			if *chatty {
				fmt.Printf("=== RUN   %s\n", t.name)
			}
			go tRunner(t, &tests[i])
			out := (<-t.signal).(*T)
			if out == nil { // Parallel run.
				go func() {
					collector <- <-t.signal
				}()
				numParallel++
				continue
			}
			t.report()
			ok = ok && !out.Failed()
		}

		running := 0
		for numParallel+running > 0 {
			if running < *parallel && numParallel > 0 {
				startParallel <- true
				running++
				numParallel--
				continue
			}
			t := (<-collector).(*T)
			t.report()
			ok = ok && !t.Failed()
			running--
		}
	}
	return
}

// before runs before all testing.
func before() {
	if *memProfileRate > 0 {
		runtime.MemProfileRate = *memProfileRate
	}
	if *cpuProfile != "" {
		f, err := os.Create(toOutputDir(*cpuProfile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s", err)
			return
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't start cpu profile: %s", err)
			f.Close()
			return
		}
		// Could save f so after can call f.Close; not worth the effort.
	}
	if *traceFile != "" {
		f, err := os.Create(toOutputDir(*traceFile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s", err)
			return
		}
		if err := trace.Start(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't start tracing: %s", err)
			f.Close()
			return
		}
		// Could save f so after can call f.Close; not worth the effort.
	}
	if *blockProfile != "" && *blockProfileRate >= 0 {
		runtime.SetBlockProfileRate(*blockProfileRate)
	}
	if *coverProfile != "" && cover.Mode == "" {
		fmt.Fprintf(os.Stderr, "testing: cannot use -test.coverprofile because test binary was not built with coverage enabled\n")
		os.Exit(2)
	}
}

// after runs after all testing.
func after() {
	if *cpuProfile != "" {
		pprof.StopCPUProfile() // flushes profile to disk
	}
	if *traceFile != "" {
		trace.Stop() // flushes trace to disk
	}
	if *memProfile != "" {
		f, err := os.Create(toOutputDir(*memProfile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			os.Exit(2)
		}
		runtime.GC() // materialize all statistics
		if err = pprof.WriteHeapProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write %s: %s\n", *memProfile, err)
			os.Exit(2)
		}
		f.Close()
	}
	if *blockProfile != "" && *blockProfileRate >= 0 {
		f, err := os.Create(toOutputDir(*blockProfile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			os.Exit(2)
		}
		if err = pprof.Lookup("block").WriteTo(f, 0); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write %s: %s\n", *blockProfile, err)
			os.Exit(2)
		}
		f.Close()
	}
	if cover.Mode != "" {
		coverReport()
	}
}

// toOutputDir returns the file name relocated, if required, to outputDir.
// Simple implementation to avoid pulling in path/filepath.
func toOutputDir(path string) string {
	if *outputDir == "" || path == "" {
		return path
	}
	if runtime.GOOS == "windows" {
		// On Windows, it's clumsy, but we can be almost always correct
		// by just looking for a drive letter and a colon.
		// Absolute paths always have a drive letter (ignoring UNC).
		// Problem: if path == "C:A" and outputdir == "C:\Go" it's unclear
		// what to do, but even then path/filepath doesn't help.
		// TODO: Worth doing better? Probably not, because we're here only
		// under the management of go test.
		if len(path) >= 2 {
			letter, colon := path[0], path[1]
			if ('a' <= letter && letter <= 'z' || 'A' <= letter && letter <= 'Z') && colon == ':' {
				// If path starts with a drive letter we're stuck with it regardless.
				return path
			}
		}
	}
	if os.IsPathSeparator(path[0]) {
		return path
	}
	return fmt.Sprintf("%s%c%s", *outputDir, os.PathSeparator, path)
}

var timer *time.Timer

// startAlarm starts an alarm if requested.
func startAlarm() {
	if *timeout > 0 {
		timer = time.AfterFunc(*timeout, func() {
			debug.SetTraceback("all")
			panic(fmt.Sprintf("test timed out after %v", *timeout))
		})
	}
}

// stopAlarm turns off the alarm.
func stopAlarm() {
	if *timeout > 0 {
		timer.Stop()
	}
}

func parseCpuList() {
	for _, val := range strings.Split(*cpuListStr, ",") {
		val = strings.TrimSpace(val)
		if val == "" {
			continue
		}
		cpu, err := strconv.Atoi(val)
		if err != nil || cpu <= 0 {
			fmt.Fprintf(os.Stderr, "testing: invalid value %q for -test.cpu\n", val)
			os.Exit(1)
		}
		for i := uint(0); i < *count; i++ {
			cpuList = append(cpuList, cpu)
		}
	}
	if cpuList == nil {
		for i := uint(0); i < *count; i++ {
			cpuList = append(cpuList, runtime.GOMAXPROCS(-1))
		}
	}
}
