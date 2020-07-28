// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testing provides support for automated testing of Go packages.
// It is intended to be used in concert with the "go test" command, which automates
// execution of any function of the form
//     func TestXxx(*testing.T)
// where Xxx does not start with a lowercase letter. The function name
// serves to identify the test routine.
//
// Within these functions, use the Error, Fail or related methods to signal failure.
//
// To write a new test suite, create a file whose name ends _test.go that
// contains the TestXxx functions as described here. Put the file in the same
// package as the one being tested. The file will be excluded from regular
// package builds but will be included when the "go test" command is run.
// For more detail, run "go help test" and "go help testflag".
//
// A simple test function looks like this:
//
//     func TestAbs(t *testing.T) {
//         got := Abs(-1)
//         if got != 1 {
//             t.Errorf("Abs(-1) = %d; want 1", got)
//         }
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
// https://golang.org/cmd/go/#hdr-Testing_flags
//
// A sample benchmark function looks like this:
//     func BenchmarkRandInt(b *testing.B) {
//         for i := 0; i < b.N; i++ {
//             rand.Int()
//         }
//     }
//
// The benchmark function must run the target code b.N times.
// During benchmark execution, b.N is adjusted until the benchmark function lasts
// long enough to be timed reliably. The output
//     BenchmarkRandInt-8   	68453040	        17.8 ns/op
// means that the loop ran 68453040 times at a speed of 17.8 ns per loop.
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
//         fmt.Println("hello")
//         // Output: hello
//     }
//
//     func ExampleSalutations() {
//         fmt.Println("hello, and")
//         fmt.Println("goodbye")
//         // Output:
//         // hello, and
//         // goodbye
//     }
//
// The comment prefix "Unordered output:" is like "Output:", but matches any
// line order:
//
//     func ExamplePerm() {
//         for _, value := range Perm(5) {
//             fmt.Println(value)
//         }
//         // Unordered output: 4
//         // 2
//         // 1
//         // 3
//         // 0
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
// Skipping
//
// Tests or benchmarks may be skipped at run time with a call to
// the Skip method of *T or *B:
//
//     func TestTimeConsuming(t *testing.T) {
//         if testing.Short() {
//             t.Skip("skipping test in short mode.")
//         }
//         ...
//     }
//
// Subtests and Sub-benchmarks
//
// The Run methods of T and B allow defining subtests and sub-benchmarks,
// without having to define separate functions for each. This enables uses
// like table-driven benchmarks and creating hierarchical tests.
// It also provides a way to share common setup and tear-down code:
//
//     func TestFoo(t *testing.T) {
//         // <setup code>
//         t.Run("A=1", func(t *testing.T) { ... })
//         t.Run("A=2", func(t *testing.T) { ... })
//         t.Run("B=1", func(t *testing.T) { ... })
//         // <tear-down code>
//     }
//
// Each subtest and sub-benchmark has a unique name: the combination of the name
// of the top-level test and the sequence of names passed to Run, separated by
// slashes, with an optional trailing sequence number for disambiguation.
//
// The argument to the -run and -bench command-line flags is an unanchored regular
// expression that matches the test's name. For tests with multiple slash-separated
// elements, such as subtests, the argument is itself slash-separated, with
// expressions matching each name element in turn. Because it is unanchored, an
// empty expression matches any string.
// For example, using "matching" to mean "whose name contains":
//
//     go test -run ''      # Run all tests.
//     go test -run Foo     # Run top-level tests matching "Foo", such as "TestFooBar".
//     go test -run Foo/A=  # For top-level tests matching "Foo", run subtests matching "A=".
//     go test -run /A=1    # For all top-level tests, run subtests matching "A=1".
//
// Subtests can also be used to control parallelism. A parent test will only
// complete once all of its subtests complete. In this example, all tests are
// run in parallel with each other, and only with each other, regardless of
// other top-level tests that may be defined:
//
//     func TestGroupedParallel(t *testing.T) {
//         for _, tc := range tests {
//             tc := tc // capture range variable
//             t.Run(tc.Name, func(t *testing.T) {
//                 t.Parallel()
//                 ...
//             })
//         }
//     }
//
// The race detector kills the program if it exceeds 8192 concurrent goroutines,
// so use care when running parallel tests with the -race flag set.
//
// Run does not return until parallel subtests have completed, providing a way
// to clean up after a group of parallel tests:
//
//     func TestTeardownParallel(t *testing.T) {
//         // This Run will not return until the parallel tests finish.
//         t.Run("group", func(t *testing.T) {
//             t.Run("Test1", parallelTest1)
//             t.Run("Test2", parallelTest2)
//             t.Run("Test3", parallelTest3)
//         })
//         // <tear-down code>
//     }
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
// and teardown is necessary around a call to m.Run. m.Run will return an exit
// code that may be passed to os.Exit. If TestMain returns, the test wrapper
// will pass the result of m.Run to os.Exit itself.
//
// When TestMain is called, flag.Parse has not been run. If TestMain depends on
// command-line flags, including those of the testing package, it should call
// flag.Parse explicitly. Command line flags are always parsed by the time test
// or benchmark functions run.
//
// A simple implementation of TestMain is:
//
//	func TestMain(m *testing.M) {
//		// call flag.Parse() here if TestMain uses flags
//		os.Exit(m.Run())
//	}
//
package testing

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"internal/race"
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/trace"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var initRan bool

// Init registers testing flags. These flags are automatically registered by
// the "go test" command before running test functions, so Init is only needed
// when calling functions such as Benchmark without using "go test".
//
// Init has no effect if it was already called.
func Init() {
	if initRan {
		return
	}
	initRan = true
	// The short flag requests that tests run more quickly, but its functionality
	// is provided by test writers themselves. The testing package is just its
	// home. The all.bash installation script sets it to make installation more
	// efficient, but by default the flag is off so a plain "go test" will do a
	// full test of the package.
	short = flag.Bool("test.short", false, "run smaller test suite to save time")

	// The failfast flag requests that test execution stop after the first test failure.
	failFast = flag.Bool("test.failfast", false, "do not start new tests after the first test failure")

	// The directory in which to create profile files and the like. When run from
	// "go test", the binary always runs in the source directory for the package;
	// this flag lets "go test" tell the binary to write the files in the directory where
	// the "go test" command is run.
	outputDir = flag.String("test.outputdir", "", "write profiles to `dir`")
	// Report as tests are run; default is silent for success.
	chatty = flag.Bool("test.v", false, "verbose: print additional output")
	count = flag.Uint("test.count", 1, "run tests and benchmarks `n` times")
	coverProfile = flag.String("test.coverprofile", "", "write a coverage profile to `file`")
	matchList = flag.String("test.list", "", "list tests, examples, and benchmarks matching `regexp` then exit")
	match = flag.String("test.run", "", "run only tests and examples matching `regexp`")
	memProfile = flag.String("test.memprofile", "", "write an allocation profile to `file`")
	memProfileRate = flag.Int("test.memprofilerate", 0, "set memory allocation profiling `rate` (see runtime.MemProfileRate)")
	cpuProfile = flag.String("test.cpuprofile", "", "write a cpu profile to `file`")
	blockProfile = flag.String("test.blockprofile", "", "write a goroutine blocking profile to `file`")
	blockProfileRate = flag.Int("test.blockprofilerate", 1, "set blocking profile `rate` (see runtime.SetBlockProfileRate)")
	mutexProfile = flag.String("test.mutexprofile", "", "write a mutex contention profile to the named file after execution")
	mutexProfileFraction = flag.Int("test.mutexprofilefraction", 1, "if >= 0, calls runtime.SetMutexProfileFraction()")
	traceFile = flag.String("test.trace", "", "write an execution trace to `file`")
	timeout = flag.Duration("test.timeout", 0, "panic test binary after duration `d` (default 0, timeout disabled)")
	cpuListStr = flag.String("test.cpu", "", "comma-separated `list` of cpu counts to run each test with")
	parallel = flag.Int("test.parallel", runtime.GOMAXPROCS(0), "run at most `n` tests in parallel")
	testlog = flag.String("test.testlogfile", "", "write test action log to `file` (for use only by cmd/go)")

	initBenchmarkFlags()
}

var (
	// Flags, registered during Init.
	short                *bool
	failFast             *bool
	outputDir            *string
	chatty               *bool
	count                *uint
	coverProfile         *string
	matchList            *string
	match                *string
	memProfile           *string
	memProfileRate       *int
	cpuProfile           *string
	blockProfile         *string
	blockProfileRate     *int
	mutexProfile         *string
	mutexProfileFraction *int
	traceFile            *string
	timeout              *time.Duration
	cpuListStr           *string
	parallel             *int
	testlog              *string
	printer              *testPrinter

	haveExamples bool // are there examples?

	cpuList     []int
	testlogFile *os.File

	numFailed uint32 // number of test failures
)

type testPrinter struct {
	chatty bool

	lastNameMu sync.Mutex // guards lastName
	lastName   string     // last printed test name in chatty mode
}

func newTestPrinter(chatty bool) *testPrinter {
	return &testPrinter{
		chatty: chatty,
	}
}

func (p *testPrinter) Print(testName, out string) {
	p.Fprint(os.Stdout, testName, out)
}

func (p *testPrinter) Fprint(w io.Writer, testName, out string) {
	p.lastNameMu.Lock()
	defer p.lastNameMu.Unlock()

	if !p.chatty ||
		strings.HasPrefix(out, "--- PASS") ||
		strings.HasPrefix(out, "--- FAIL") ||
		strings.HasPrefix(out, "=== CONT") ||
		strings.HasPrefix(out, "=== RUN") {
		p.lastName = testName
		fmt.Fprint(w, out)
		return
	}

	if p.lastName == "" {
		p.lastName = testName
	} else if p.lastName != testName {
		// Always printed as-is, with 0 decoration or indentation. So, we skip
		// printing to w.
		fmt.Printf("=== CONT  %s\n", testName)
		p.lastName = testName
	}
	fmt.Fprint(w, out)
}

// The maximum number of stack frames to go through when skipping helper functions for
// the purpose of decorating log messages.
const maxStackLen = 50

// common holds the elements common between T and B and
// captures common methods such as Errorf.
type common struct {
	mu          sync.RWMutex        // guards this group of fields
	output      []byte              // Output generated by test or benchmark.
	w           io.Writer           // For flushToParent.
	ran         bool                // Test or benchmark (or one of its subtests) was executed.
	failed      bool                // Test or benchmark has failed.
	skipped     bool                // Test of benchmark has been skipped.
	done        bool                // Test is finished and all subtests have completed.
	helpers     map[string]struct{} // functions to be skipped when writing file/line info
	cleanup     func()              // optional function to be called at the end of the test
	cleanupName string              // Name of the cleanup function.
	cleanupPc   []uintptr           // The stack trace at the point where Cleanup was called.

	chatty     bool   // A copy of the chatty flag.
	bench      bool   // Whether the current test is a benchmark.
	finished   bool   // Test function has completed.
	hasSub     int32  // Written atomically.
	raceErrors int    // Number of races detected during test.
	runner     string // Function name of tRunner running the test.

	parent   *common
	level    int       // Nesting depth of test or benchmark.
	creator  []uintptr // If level > 0, the stack trace at the point where the parent called t.Run.
	name     string    // Name of test or benchmark.
	start    time.Time // Time test or benchmark started
	duration time.Duration
	barrier  chan bool // To signal parallel subtests they may start.
	signal   chan bool // To signal a test is done.
	sub      []*T      // Queue of subtests to be run in parallel.

	tempDirOnce sync.Once
	tempDir     string
	tempDirErr  error
	tempDirSeq  int32
}

// Short reports whether the -test.short flag is set.
func Short() bool {
	if short == nil {
		panic("testing: Short called before Init")
	}
	// Catch code that calls this from TestMain without first calling flag.Parse.
	if !flag.Parsed() {
		panic("testing: Short called before Parse")
	}

	return *short
}

// CoverMode reports what the test coverage mode is set to. The
// values are "set", "count", or "atomic". The return value will be
// empty if test coverage is not enabled.
func CoverMode() string {
	return cover.Mode
}

// Verbose reports whether the -test.v flag is set.
func Verbose() bool {
	// Same as in Short.
	if chatty == nil {
		panic("testing: Verbose called before Init")
	}
	if !flag.Parsed() {
		panic("testing: Verbose called before Parse")
	}
	return *chatty
}

// frameSkip searches, starting after skip frames, for the first caller frame
// in a function not marked as a helper and returns that frame.
// The search stops if it finds a tRunner function that
// was the entry point into the test and the test is not a subtest.
// This function must be called with c.mu held.
func (c *common) frameSkip(skip int) runtime.Frame {
	// If the search continues into the parent test, we'll have to hold
	// its mu temporarily. If we then return, we need to unlock it.
	shouldUnlock := false
	defer func() {
		if shouldUnlock {
			c.mu.Unlock()
		}
	}()
	var pc [maxStackLen]uintptr
	// Skip two extra frames to account for this function
	// and runtime.Callers itself.
	n := runtime.Callers(skip+2, pc[:])
	if n == 0 {
		panic("testing: zero callers found")
	}
	frames := runtime.CallersFrames(pc[:n])
	var firstFrame, prevFrame, frame runtime.Frame
	for more := true; more; prevFrame = frame {
		frame, more = frames.Next()
		if frame.Function == c.cleanupName {
			frames = runtime.CallersFrames(c.cleanupPc)
			continue
		}
		if firstFrame.PC == 0 {
			firstFrame = frame
		}
		if frame.Function == c.runner {
			// We've gone up all the way to the tRunner calling
			// the test function (so the user must have
			// called tb.Helper from inside that test function).
			// If this is a top-level test, only skip up to the test function itself.
			// If we're in a subtest, continue searching in the parent test,
			// starting from the point of the call to Run which created this subtest.
			if c.level > 1 {
				frames = runtime.CallersFrames(c.creator)
				parent := c.parent
				// We're no longer looking at the current c after this point,
				// so we should unlock its mu, unless it's the original receiver,
				// in which case our caller doesn't expect us to do that.
				if shouldUnlock {
					c.mu.Unlock()
				}
				c = parent
				// Remember to unlock c.mu when we no longer need it, either
				// because we went up another nesting level, or because we
				// returned.
				shouldUnlock = true
				c.mu.Lock()
				continue
			}
			return prevFrame
		}
		if _, ok := c.helpers[frame.Function]; !ok {
			// Found a frame that wasn't inside a helper function.
			return frame
		}
	}
	return firstFrame
}

// decorate prefixes the string with the file and line of the call site
// and inserts the final newline if needed and indentation spaces for formatting.
// This function must be called with c.mu held.
func (c *common) decorate(s string, skip int) string {
	frame := c.frameSkip(skip)
	file := frame.File
	line := frame.Line
	if file != "" {
		// Truncate file name at last file name separator.
		if index := strings.LastIndex(file, "/"); index >= 0 {
			file = file[index+1:]
		} else if index = strings.LastIndex(file, "\\"); index >= 0 {
			file = file[index+1:]
		}
	} else {
		file = "???"
	}
	if line == 0 {
		line = 1
	}
	buf := new(strings.Builder)
	// Every line is indented at least 4 spaces.
	buf.WriteString("    ")
	fmt.Fprintf(buf, "%s:%d: ", file, line)
	lines := strings.Split(s, "\n")
	if l := len(lines); l > 1 && lines[l-1] == "" {
		lines = lines[:l-1]
	}
	for i, line := range lines {
		if i > 0 {
			// Second and subsequent lines are indented an additional 4 spaces.
			buf.WriteString("\n        ")
		}
		buf.WriteString(line)
	}
	buf.WriteByte('\n')
	return buf.String()
}

// flushToParent writes c.output to the parent after first writing the header
// with the given format and arguments.
func (c *common) flushToParent(testName, format string, args ...interface{}) {
	p := c.parent
	p.mu.Lock()
	defer p.mu.Unlock()

	printer.Fprint(p.w, testName, fmt.Sprintf(format, args...))

	c.mu.Lock()
	defer c.mu.Unlock()
	io.Copy(p.w, bytes.NewReader(c.output))
	c.output = c.output[:0]
}

type indenter struct {
	c *common
}

func (w indenter) Write(b []byte) (n int, err error) {
	n = len(b)
	for len(b) > 0 {
		end := bytes.IndexByte(b, '\n')
		if end == -1 {
			end = len(b)
		} else {
			end++
		}
		// An indent of 4 spaces will neatly align the dashes with the status
		// indicator of the parent.
		const indent = "    "
		w.c.output = append(w.c.output, indent...)
		w.c.output = append(w.c.output, b[:end]...)
		b = b[end:]
	}
	return
}

// fmtDuration returns a string representing d in the form "87.00s".
func fmtDuration(d time.Duration) string {
	return fmt.Sprintf("%.2fs", d.Seconds())
}

// TB is the interface common to T and B.
type TB interface {
	Cleanup(func())
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Helper()
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Name() string
	Skip(args ...interface{})
	SkipNow()
	Skipf(format string, args ...interface{})
	Skipped() bool
	TempDir() string

	// A private method to prevent users implementing the
	// interface and so future additions to it will not
	// violate Go 1 compatibility.
	private()
}

var _ TB = (*T)(nil)
var _ TB = (*B)(nil)

// T is a type passed to Test functions to manage test state and support formatted test logs.
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
	isParallel bool
	context    *testContext // For running tests and subtests.
}

func (c *common) private() {}

// Name returns the name of the running test or benchmark.
func (c *common) Name() string {
	return c.name
}

func (c *common) setRan() {
	if c.parent != nil {
		c.parent.setRan()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ran = true
}

// Fail marks the function as having failed but continues execution.
func (c *common) Fail() {
	if c.parent != nil {
		c.parent.Fail()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// c.done needs to be locked to synchronize checks to c.done in parent tests.
	if c.done {
		panic("Fail in goroutine after " + c.name + " has completed")
	}
	c.failed = true
}

// Failed reports whether the function has failed.
func (c *common) Failed() bool {
	c.mu.RLock()
	failed := c.failed
	c.mu.RUnlock()
	return failed || c.raceErrors+race.Errors() > 0
}

// FailNow marks the function as having failed and stops its execution
// by calling runtime.Goexit (which then runs all deferred calls in the
// current goroutine).
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
	// before the test exited. If a test deferred an important cleanup
	// function (like removing temporary files), there was no guarantee
	// it would run on a test failure. Because we send on c.signal during
	// a top-of-stack deferred function now, we know that the send
	// only happens after any other stacked defers have completed.
	c.finished = true
	runtime.Goexit()
}

// log generates the output. It's always at the same stack depth.
func (c *common) log(s string) {
	c.logDepth(s, 3) // logDepth + log + public function
}

// logDepth generates the output at an arbitrary stack depth.
func (c *common) logDepth(s string, depth int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.done {
		// This test has already finished. Try and log this message
		// with our parent. If we don't have a parent, panic.
		for parent := c.parent; parent != nil; parent = parent.parent {
			parent.mu.Lock()
			defer parent.mu.Unlock()
			if !parent.done {
				parent.output = append(parent.output, parent.decorate(s, depth+1)...)
				return
			}
		}
		panic("Log in goroutine after " + c.name + " has completed")
	} else {
		if c.chatty {
			if c.bench {
				// Benchmarks don't print === CONT, so we should skip the test
				// printer and just print straight to stdout.
				fmt.Print(c.decorate(s, depth+1))
			} else {
				printer.Print(c.name, c.decorate(s, depth+1))
			}

			return
		}
		c.output = append(c.output, c.decorate(s, depth+1)...)
	}
}

// Log formats its arguments using default formatting, analogous to Println,
// and records the text in the error log. For tests, the text will be printed only if
// the test fails or the -test.v flag is set. For benchmarks, the text is always
// printed to avoid having performance depend on the value of the -test.v flag.
func (c *common) Log(args ...interface{}) { c.log(fmt.Sprintln(args...)) }

// Logf formats its arguments according to the format, analogous to Printf, and
// records the text in the error log. A final newline is added if not provided. For
// tests, the text will be printed only if the test fails or the -test.v flag is
// set. For benchmarks, the text is always printed to avoid having performance
// depend on the value of the -test.v flag.
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

// SkipNow marks the test as having been skipped and stops its execution
// by calling runtime.Goexit.
// If a test fails (see Error, Errorf, Fail) and is then skipped,
// it is still considered to have failed.
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

// Helper marks the calling function as a test helper function.
// When printing file and line information, that function will be skipped.
// Helper may be called simultaneously from multiple goroutines.
func (c *common) Helper() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.helpers == nil {
		c.helpers = make(map[string]struct{})
	}
	c.helpers[callerName(1)] = struct{}{}
}

// Cleanup registers a function to be called when the test and all its
// subtests complete. Cleanup functions will be called in last added,
// first called order.
func (c *common) Cleanup(f func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	oldCleanup := c.cleanup
	oldCleanupPc := c.cleanupPc
	c.cleanup = func() {
		if oldCleanup != nil {
			defer func() {
				c.cleanupPc = oldCleanupPc
				oldCleanup()
			}()
		}
		c.cleanupName = callerName(0)
		f()
	}
	var pc [maxStackLen]uintptr
	// Skip two extra frames to account for this function and runtime.Callers itself.
	n := runtime.Callers(2, pc[:])
	c.cleanupPc = pc[:n]
}

var tempDirReplacer struct {
	sync.Once
	r *strings.Replacer
}

// TempDir returns a temporary directory for the test to use.
// The directory is automatically removed by Cleanup when the test and
// all its subtests complete.
// Each subsequent call to t.TempDir returns a unique directory;
// if the directory creation fails, TempDir terminates the test by calling Fatal.
func (c *common) TempDir() string {
	// Use a single parent directory for all the temporary directories
	// created by a test, each numbered sequentially.
	c.tempDirOnce.Do(func() {
		c.Helper()

		// ioutil.TempDir doesn't like path separators in its pattern,
		// so mangle the name to accommodate subtests.
		tempDirReplacer.Do(func() {
			tempDirReplacer.r = strings.NewReplacer("/", "_", "\\", "_", ":", "_")
		})
		pattern := tempDirReplacer.r.Replace(c.Name())

		c.tempDir, c.tempDirErr = ioutil.TempDir("", pattern)
		if c.tempDirErr == nil {
			c.Cleanup(func() {
				if err := os.RemoveAll(c.tempDir); err != nil {
					c.Errorf("TempDir RemoveAll cleanup: %v", err)
				}
			})
		}
	})
	if c.tempDirErr != nil {
		c.Fatalf("TempDir: %v", c.tempDirErr)
	}
	seq := atomic.AddInt32(&c.tempDirSeq, 1)
	dir := fmt.Sprintf("%s%c%03d", c.tempDir, os.PathSeparator, seq)
	if err := os.Mkdir(dir, 0777); err != nil {
		c.Fatalf("TempDir: %v", err)
	}
	return dir
}

// panicHanding is an argument to runCleanup.
type panicHandling int

const (
	normalPanic panicHandling = iota
	recoverAndReturnPanic
)

// runCleanup is called at the end of the test.
// If catchPanic is true, this will catch panics, and return the recovered
// value if any.
func (c *common) runCleanup(ph panicHandling) (panicVal interface{}) {
	c.mu.Lock()
	cleanup := c.cleanup
	c.cleanup = nil
	c.mu.Unlock()
	if cleanup == nil {
		return nil
	}

	if ph == recoverAndReturnPanic {
		defer func() {
			panicVal = recover()
		}()
	}

	cleanup()
	return nil
}

// callerName gives the function name (qualified with a package path)
// for the caller after skip frames (where 0 means the current function).
func callerName(skip int) string {
	// Make room for the skip PC.
	var pc [1]uintptr
	n := runtime.Callers(skip+2, pc[:]) // skip + runtime.Callers + callerName
	if n == 0 {
		panic("testing: zero callers found")
	}
	frames := runtime.CallersFrames(pc[:n])
	frame, _ := frames.Next()
	return frame.Function
}

// Parallel signals that this test is to be run in parallel with (and only with)
// other parallel tests. When a test is run multiple times due to use of
// -test.count or -test.cpu, multiple instances of a single test never run in
// parallel with each other.
func (t *T) Parallel() {
	if t.isParallel {
		panic("testing: t.Parallel called multiple times")
	}
	t.isParallel = true

	// We don't want to include the time we spend waiting for serial tests
	// in the test duration. Record the elapsed time thus far and reset the
	// timer afterwards.
	t.duration += time.Since(t.start)

	// Add to the list of tests to be released by the parent.
	t.parent.sub = append(t.parent.sub, t)
	t.raceErrors += race.Errors()

	if t.chatty {
		// Print directly to root's io.Writer so there is no delay.
		root := t.parent
		for ; root.parent != nil; root = root.parent {
		}
		root.mu.Lock()
		fmt.Fprintf(root.w, "=== PAUSE %s\n", t.name)
		root.mu.Unlock()
	}

	t.signal <- true   // Release calling test.
	<-t.parent.barrier // Wait for the parent test to complete.
	t.context.waitParallel()

	if t.chatty {
		// Print directly to root's io.Writer so there is no delay.
		root := t.parent
		for ; root.parent != nil; root = root.parent {
		}
		root.mu.Lock()
		printer.Fprint(root.w, t.name, fmt.Sprintf("=== CONT  %s\n", t.name))
		root.mu.Unlock()
	}

	t.start = time.Now()
	t.raceErrors += -race.Errors()
}

// InternalTest is an internal type but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
type InternalTest struct {
	Name string
	F    func(*T)
}

var errNilPanicOrGoexit = errors.New("test executed panic(nil) or runtime.Goexit")

func tRunner(t *T, fn func(t *T)) {
	t.runner = callerName(0)

	// When this goroutine is done, either because fn(t)
	// returned normally or because a test failure triggered
	// a call to runtime.Goexit, record the duration and send
	// a signal saying that the test is done.
	defer func() {
		if t.Failed() {
			atomic.AddUint32(&numFailed, 1)
		}

		if t.raceErrors+race.Errors() > 0 {
			t.Errorf("race detected during execution of test")
		}

		// If the test panicked, print any test output before dying.
		err := recover()
		signal := true
		if !t.finished && err == nil {
			err = errNilPanicOrGoexit
			for p := t.parent; p != nil; p = p.parent {
				if p.finished {
					t.Errorf("%v: subtest may have called FailNow on a parent test", err)
					err = nil
					signal = false
					break
				}
			}
		}

		doPanic := func(err interface{}) {
			t.Fail()
			if r := t.runCleanup(recoverAndReturnPanic); r != nil {
				t.Logf("cleanup panicked with %v", r)
			}
			// Flush the output log up to the root before dying.
			for root := &t.common; root.parent != nil; root = root.parent {
				root.mu.Lock()
				root.duration += time.Since(root.start)
				d := root.duration
				root.mu.Unlock()
				root.flushToParent(root.name, "--- FAIL: %s (%s)\n", root.name, fmtDuration(d))
				if r := root.parent.runCleanup(recoverAndReturnPanic); r != nil {
					fmt.Fprintf(root.parent.w, "cleanup panicked with %v", r)
				}
			}
			panic(err)
		}
		if err != nil {
			doPanic(err)
		}

		t.duration += time.Since(t.start)

		if len(t.sub) > 0 {
			// Run parallel subtests.
			// Decrease the running count for this test.
			t.context.release()
			// Release the parallel subtests.
			close(t.barrier)
			// Wait for subtests to complete.
			for _, sub := range t.sub {
				<-sub.signal
			}
			cleanupStart := time.Now()
			err := t.runCleanup(recoverAndReturnPanic)
			t.duration += time.Since(cleanupStart)
			if err != nil {
				doPanic(err)
			}
			if !t.isParallel {
				// Reacquire the count for sequential tests. See comment in Run.
				t.context.waitParallel()
			}
		} else if t.isParallel {
			// Only release the count for this test if it was run as a parallel
			// test. See comment in Run method.
			t.context.release()
		}
		t.report() // Report after all subtests have finished.

		// Do not lock t.done to allow race detector to detect race in case
		// the user does not appropriately synchronizes a goroutine.
		t.done = true
		if t.parent != nil && atomic.LoadInt32(&t.hasSub) == 0 {
			t.setRan()
		}
		t.signal <- signal
	}()
	defer func() {
		if len(t.sub) == 0 {
			t.runCleanup(normalPanic)
		}
	}()

	t.start = time.Now()
	t.raceErrors = -race.Errors()
	fn(t)

	// code beyond here will not be executed when FailNow is invoked
	t.finished = true
}

// Run runs f as a subtest of t called name. It runs f in a separate goroutine
// and blocks until f returns or calls t.Parallel to become a parallel test.
// Run reports whether f succeeded (or at least did not fail before calling t.Parallel).
//
// Run may be called simultaneously from multiple goroutines, but all such calls
// must return before the outer test function for t returns.
func (t *T) Run(name string, f func(t *T)) bool {
	atomic.StoreInt32(&t.hasSub, 1)
	testName, ok, _ := t.context.match.fullName(&t.common, name)
	if !ok || shouldFailFast() {
		return true
	}
	// Record the stack trace at the point of this call so that if the subtest
	// function - which runs in a separate stack - is marked as a helper, we can
	// continue walking the stack into the parent test.
	var pc [maxStackLen]uintptr
	n := runtime.Callers(2, pc[:])
	t = &T{
		common: common{
			barrier: make(chan bool),
			signal:  make(chan bool),
			name:    testName,
			parent:  &t.common,
			level:   t.level + 1,
			creator: pc[:n],
			chatty:  t.chatty,
		},
		context: t.context,
	}
	t.w = indenter{&t.common}

	if t.chatty {
		// Print directly to root's io.Writer so there is no delay.
		root := t.parent
		for ; root.parent != nil; root = root.parent {
		}
		root.mu.Lock()
		printer.Fprint(root.w, t.name, fmt.Sprintf("=== RUN   %s\n", t.name))
		root.mu.Unlock()
	}
	// Instead of reducing the running count of this test before calling the
	// tRunner and increasing it afterwards, we rely on tRunner keeping the
	// count correct. This ensures that a sequence of sequential tests runs
	// without being preempted, even when their parent is a parallel test. This
	// may especially reduce surprises if *parallel == 1.
	go tRunner(t, f)
	if !<-t.signal {
		// At this point, it is likely that FailNow was called on one of the
		// parent tests by one of the subtests. Continue aborting up the chain.
		runtime.Goexit()
	}
	return !t.failed
}

// Deadline reports the time at which the test binary will have
// exceeded the timeout specified by the -timeout flag.
//
// The ok result is false if the -timeout flag indicates “no timeout” (0).
func (t *T) Deadline() (deadline time.Time, ok bool) {
	deadline = t.context.deadline
	return deadline, !deadline.IsZero()
}

// testContext holds all fields that are common to all tests. This includes
// synchronization primitives to run at most *parallel tests.
type testContext struct {
	match    *matcher
	deadline time.Time

	mu sync.Mutex

	// Channel used to signal tests that are ready to be run in parallel.
	startParallel chan bool

	// running is the number of tests currently running in parallel.
	// This does not include tests that are waiting for subtests to complete.
	running int

	// numWaiting is the number tests waiting to be run in parallel.
	numWaiting int

	// maxParallel is a copy of the parallel flag.
	maxParallel int
}

func newTestContext(maxParallel int, m *matcher) *testContext {
	return &testContext{
		match:         m,
		startParallel: make(chan bool),
		maxParallel:   maxParallel,
		running:       1, // Set the count to 1 for the main (sequential) test.
	}
}

func (c *testContext) waitParallel() {
	c.mu.Lock()
	if c.running < c.maxParallel {
		c.running++
		c.mu.Unlock()
		return
	}
	c.numWaiting++
	c.mu.Unlock()
	<-c.startParallel
}

func (c *testContext) release() {
	c.mu.Lock()
	if c.numWaiting == 0 {
		c.running--
		c.mu.Unlock()
		return
	}
	c.numWaiting--
	c.mu.Unlock()
	c.startParallel <- true // Pick a waiting test to be run.
}

// No one should be using func Main anymore.
// See the doc comment on func Main and use MainStart instead.
var errMain = errors.New("testing: unexpected use of func Main")

type matchStringOnly func(pat, str string) (bool, error)

func (f matchStringOnly) MatchString(pat, str string) (bool, error)   { return f(pat, str) }
func (f matchStringOnly) StartCPUProfile(w io.Writer) error           { return errMain }
func (f matchStringOnly) StopCPUProfile()                             {}
func (f matchStringOnly) WriteProfileTo(string, io.Writer, int) error { return errMain }
func (f matchStringOnly) ImportPath() string                          { return "" }
func (f matchStringOnly) StartTestLog(io.Writer)                      {}
func (f matchStringOnly) StopTestLog() error                          { return errMain }

// Main is an internal function, part of the implementation of the "go test" command.
// It was exported because it is cross-package and predates "internal" packages.
// It is no longer used by "go test" but preserved, as much as possible, for other
// systems that simulate "go test" using Main, but Main sometimes cannot be updated as
// new functionality is added to the testing package.
// Systems simulating "go test" should be updated to use MainStart.
func Main(matchString func(pat, str string) (bool, error), tests []InternalTest, benchmarks []InternalBenchmark, examples []InternalExample) {
	os.Exit(MainStart(matchStringOnly(matchString), tests, benchmarks, examples).Run())
}

// M is a type passed to a TestMain function to run the actual tests.
type M struct {
	deps       testDeps
	tests      []InternalTest
	benchmarks []InternalBenchmark
	examples   []InternalExample

	timer     *time.Timer
	afterOnce sync.Once

	numRun int

	// value to pass to os.Exit, the outer test func main
	// harness calls os.Exit with this code. See #34129.
	exitCode int
}

// testDeps is an internal interface of functionality that is
// passed into this package by a test's generated main package.
// The canonical implementation of this interface is
// testing/internal/testdeps's TestDeps.
type testDeps interface {
	ImportPath() string
	MatchString(pat, str string) (bool, error)
	StartCPUProfile(io.Writer) error
	StopCPUProfile()
	StartTestLog(io.Writer)
	StopTestLog() error
	WriteProfileTo(string, io.Writer, int) error
}

// MainStart is meant for use by tests generated by 'go test'.
// It is not meant to be called directly and is not subject to the Go 1 compatibility document.
// It may change signature from release to release.
func MainStart(deps testDeps, tests []InternalTest, benchmarks []InternalBenchmark, examples []InternalExample) *M {
	Init()
	return &M{
		deps:       deps,
		tests:      tests,
		benchmarks: benchmarks,
		examples:   examples,
	}
}

// Run runs the tests. It returns an exit code to pass to os.Exit.
func (m *M) Run() (code int) {
	defer func() {
		code = m.exitCode
	}()

	// Count the number of calls to m.Run.
	// We only ever expected 1, but we didn't enforce that,
	// and now there are tests in the wild that call m.Run multiple times.
	// Sigh. golang.org/issue/23129.
	m.numRun++

	// TestMain may have already called flag.Parse.
	if !flag.Parsed() {
		flag.Parse()
	}

	printer = newTestPrinter(Verbose())

	if *parallel < 1 {
		fmt.Fprintln(os.Stderr, "testing: -parallel can only be given a positive integer")
		flag.Usage()
		m.exitCode = 2
		return
	}

	if len(*matchList) != 0 {
		listTests(m.deps.MatchString, m.tests, m.benchmarks, m.examples)
		m.exitCode = 0
		return
	}

	parseCpuList()

	m.before()
	defer m.after()
	deadline := m.startAlarm()
	haveExamples = len(m.examples) > 0
	testRan, testOk := runTests(m.deps.MatchString, m.tests, deadline)
	exampleRan, exampleOk := runExamples(m.deps.MatchString, m.examples)
	m.stopAlarm()
	if !testRan && !exampleRan && *matchBenchmarks == "" {
		fmt.Fprintln(os.Stderr, "testing: warning: no tests to run")
	}
	if !testOk || !exampleOk || !runBenchmarks(m.deps.ImportPath(), m.deps.MatchString, m.benchmarks) || race.Errors() > 0 {
		fmt.Println("FAIL")
		m.exitCode = 1
		return
	}

	fmt.Println("PASS")
	m.exitCode = 0
	return
}

func (t *T) report() {
	if t.parent == nil {
		return
	}
	dstr := fmtDuration(t.duration)
	format := "--- %s: %s (%s)\n"
	if t.Failed() {
		t.flushToParent(t.name, format, "FAIL", t.name, dstr)
	} else if t.chatty {
		if t.Skipped() {
			t.flushToParent(t.name, format, "SKIP", t.name, dstr)
		} else {
			t.flushToParent(t.name, format, "PASS", t.name, dstr)
		}
	}
}

func listTests(matchString func(pat, str string) (bool, error), tests []InternalTest, benchmarks []InternalBenchmark, examples []InternalExample) {
	if _, err := matchString(*matchList, "non-empty"); err != nil {
		fmt.Fprintf(os.Stderr, "testing: invalid regexp in -test.list (%q): %s\n", *matchList, err)
		os.Exit(1)
	}

	for _, test := range tests {
		if ok, _ := matchString(*matchList, test.Name); ok {
			fmt.Println(test.Name)
		}
	}
	for _, bench := range benchmarks {
		if ok, _ := matchString(*matchList, bench.Name); ok {
			fmt.Println(bench.Name)
		}
	}
	for _, example := range examples {
		if ok, _ := matchString(*matchList, example.Name); ok {
			fmt.Println(example.Name)
		}
	}
}

// RunTests is an internal function but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
func RunTests(matchString func(pat, str string) (bool, error), tests []InternalTest) (ok bool) {
	var deadline time.Time
	if *timeout > 0 {
		deadline = time.Now().Add(*timeout)
	}
	ran, ok := runTests(matchString, tests, deadline)
	if !ran && !haveExamples {
		fmt.Fprintln(os.Stderr, "testing: warning: no tests to run")
	}
	return ok
}

func runTests(matchString func(pat, str string) (bool, error), tests []InternalTest, deadline time.Time) (ran, ok bool) {
	ok = true
	for _, procs := range cpuList {
		runtime.GOMAXPROCS(procs)
		for i := uint(0); i < *count; i++ {
			if shouldFailFast() {
				break
			}
			ctx := newTestContext(*parallel, newMatcher(matchString, *match, "-test.run"))
			ctx.deadline = deadline
			t := &T{
				common: common{
					signal:  make(chan bool),
					barrier: make(chan bool),
					w:       os.Stdout,
					chatty:  *chatty,
				},
				context: ctx,
			}
			tRunner(t, func(t *T) {
				for _, test := range tests {
					t.Run(test.Name, test.F)
				}
				// Run catching the signal rather than the tRunner as a separate
				// goroutine to avoid adding a goroutine during the sequential
				// phase as this pollutes the stacktrace output when aborting.
				go func() { <-t.signal }()
			})
			ok = ok && !t.Failed()
			ran = ran || t.ran
		}
	}
	return ran, ok
}

// before runs before all testing.
func (m *M) before() {
	if *memProfileRate > 0 {
		runtime.MemProfileRate = *memProfileRate
	}
	if *cpuProfile != "" {
		f, err := os.Create(toOutputDir(*cpuProfile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			return
		}
		if err := m.deps.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't start cpu profile: %s\n", err)
			f.Close()
			return
		}
		// Could save f so after can call f.Close; not worth the effort.
	}
	if *traceFile != "" {
		f, err := os.Create(toOutputDir(*traceFile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			return
		}
		if err := trace.Start(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't start tracing: %s\n", err)
			f.Close()
			return
		}
		// Could save f so after can call f.Close; not worth the effort.
	}
	if *blockProfile != "" && *blockProfileRate >= 0 {
		runtime.SetBlockProfileRate(*blockProfileRate)
	}
	if *mutexProfile != "" && *mutexProfileFraction >= 0 {
		runtime.SetMutexProfileFraction(*mutexProfileFraction)
	}
	if *coverProfile != "" && cover.Mode == "" {
		fmt.Fprintf(os.Stderr, "testing: cannot use -test.coverprofile because test binary was not built with coverage enabled\n")
		os.Exit(2)
	}
	if *testlog != "" {
		// Note: Not using toOutputDir.
		// This file is for use by cmd/go, not users.
		var f *os.File
		var err error
		if m.numRun == 1 {
			f, err = os.Create(*testlog)
		} else {
			f, err = os.OpenFile(*testlog, os.O_WRONLY, 0)
			if err == nil {
				f.Seek(0, io.SeekEnd)
			}
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			os.Exit(2)
		}
		m.deps.StartTestLog(f)
		testlogFile = f
	}
}

// after runs after all testing.
func (m *M) after() {
	m.afterOnce.Do(func() {
		m.writeProfiles()
	})
}

func (m *M) writeProfiles() {
	if *testlog != "" {
		if err := m.deps.StopTestLog(); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write %s: %s\n", *testlog, err)
			os.Exit(2)
		}
		if err := testlogFile.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write %s: %s\n", *testlog, err)
			os.Exit(2)
		}
	}
	if *cpuProfile != "" {
		m.deps.StopCPUProfile() // flushes profile to disk
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
		if err = m.deps.WriteProfileTo("allocs", f, 0); err != nil {
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
		if err = m.deps.WriteProfileTo("block", f, 0); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write %s: %s\n", *blockProfile, err)
			os.Exit(2)
		}
		f.Close()
	}
	if *mutexProfile != "" && *mutexProfileFraction >= 0 {
		f, err := os.Create(toOutputDir(*mutexProfile))
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			os.Exit(2)
		}
		if err = m.deps.WriteProfileTo("mutex", f, 0); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write %s: %s\n", *mutexProfile, err)
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
	// On Windows, it's clumsy, but we can be almost always correct
	// by just looking for a drive letter and a colon.
	// Absolute paths always have a drive letter (ignoring UNC).
	// Problem: if path == "C:A" and outputdir == "C:\Go" it's unclear
	// what to do, but even then path/filepath doesn't help.
	// TODO: Worth doing better? Probably not, because we're here only
	// under the management of go test.
	if runtime.GOOS == "windows" && len(path) >= 2 {
		letter, colon := path[0], path[1]
		if ('a' <= letter && letter <= 'z' || 'A' <= letter && letter <= 'Z') && colon == ':' {
			// If path starts with a drive letter we're stuck with it regardless.
			return path
		}
	}
	if os.IsPathSeparator(path[0]) {
		return path
	}
	return fmt.Sprintf("%s%c%s", *outputDir, os.PathSeparator, path)
}

// startAlarm starts an alarm if requested.
func (m *M) startAlarm() time.Time {
	if *timeout <= 0 {
		return time.Time{}
	}

	deadline := time.Now().Add(*timeout)
	m.timer = time.AfterFunc(*timeout, func() {
		m.after()
		debug.SetTraceback("all")
		panic(fmt.Sprintf("test timed out after %v", *timeout))
	})
	return deadline
}

// stopAlarm turns off the alarm.
func (m *M) stopAlarm() {
	if *timeout > 0 {
		m.timer.Stop()
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
		cpuList = append(cpuList, cpu)
	}
	if cpuList == nil {
		cpuList = append(cpuList, runtime.GOMAXPROCS(-1))
	}
}

func shouldFailFast() bool {
	return *failFast && atomic.LoadUint32(&numFailed) > 0
}
