// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"
)

func initFuzzFlags() {
	matchFuzz = flag.String("test.fuzz", "", "run the fuzz test matching `regexp`")
	flag.Var(&fuzzDuration, "test.fuzztime", "time to spend fuzzing; default is to run indefinitely")
	flag.Var(&minimizeDuration, "test.fuzzminimizetime", "time to spend minimizing a value after finding a failing input")

	fuzzCacheDir = flag.String("test.fuzzcachedir", "", "directory where interesting fuzzing inputs are stored (for use only by cmd/go)")
	isFuzzWorker = flag.Bool("test.fuzzworker", false, "coordinate with the parent process to fuzz random values (for use only by cmd/go)")
}

var (
	matchFuzz        *string
	fuzzDuration     durationOrCountFlag
	minimizeDuration = durationOrCountFlag{d: 60 * time.Second, allowZero: true}
	fuzzCacheDir     *string
	isFuzzWorker     *bool

	// corpusDir is the parent directory of the fuzz test's seed corpus within
	// the package.
	corpusDir = "testdata/fuzz"
)

// fuzzWorkerExitCode is used as an exit code by fuzz worker processes after an
// internal error. This distinguishes internal errors from uncontrolled panics
// and other failures. Keep in sync with internal/fuzz.workerExitCode.
const fuzzWorkerExitCode = 70

// InternalFuzzTarget is an internal type but exported because it is
// cross-package; it is part of the implementation of the "go test" command.
type InternalFuzzTarget struct {
	Name string
	Fn   func(f *F)
}

// F is a type passed to fuzz tests.
//
// Fuzz tests run generated inputs against a provided fuzz target, which can
// find and report potential bugs in the code being tested.
//
// A fuzz test runs the seed corpus by default, which includes entries provided
// by [F.Add] and entries in the testdata/fuzz/<FuzzTestName> directory. After
// any necessary setup and calls to [F.Add], the fuzz test must then call
// [F.Fuzz] to provide the fuzz target. See the testing package documentation
// for an example, and see the [F.Fuzz] and [F.Add] method documentation for
// details.
//
// *F methods can only be called before [F.Fuzz]. Once the test is
// executing the fuzz target, only [*T] methods can be used. The only *F methods
// that are allowed in the [F.Fuzz] function are [F.Failed] and [F.Name].
type F struct {
	common
	fstate *fuzzState
	tstate *testState

	// inFuzzFn is true when the fuzz function is running. Most F methods cannot
	// be called when inFuzzFn is true.
	inFuzzFn bool

	// corpus is a set of seed corpus entries, added with F.Add and loaded
	// from testdata.
	corpus []corpusEntry

	result     fuzzResult
	fuzzCalled bool
}

var _ TB = (*F)(nil)

// corpusEntry is an alias to the same type as internal/fuzz.CorpusEntry.
// We use a type alias because we don't want to export this type, and we can't
// import internal/fuzz from testing.
type corpusEntry = struct {
	Parent     string
	Path       string
	Data       []byte
	Values     []any
	Generation int
	IsSeed     bool
}

// Helper marks the calling function as a test helper function.
// When printing file and line information, that function will be skipped.
// Helper may be called simultaneously from multiple goroutines.
func (f *F) Helper() {
	if f.inFuzzFn {
		panic("testing: f.Helper was called inside the fuzz target, use t.Helper instead")
	}

	// common.Helper is inlined here.
	// If we called it, it would mark F.Helper as the helper
	// instead of the caller.
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.helperPCs == nil {
		f.helperPCs = make(map[uintptr]struct{})
	}
	// repeating code from callerName here to save walking a stack frame
	var pc [1]uintptr
	n := runtime.Callers(2, pc[:]) // skip runtime.Callers + Helper
	if n == 0 {
		panic("testing: zero callers found")
	}
	if _, found := f.helperPCs[pc[0]]; !found {
		f.helperPCs[pc[0]] = struct{}{}
		f.helperNames = nil // map will be recreated next time it is needed
	}
}

// Fail marks the function as having failed but continues execution.
func (f *F) Fail() {
	// (*F).Fail may be called by (*T).Fail, which we should allow. However, we
	// shouldn't allow direct (*F).Fail calls from inside the (*F).Fuzz function.
	if f.inFuzzFn {
		panic("testing: f.Fail was called inside the fuzz target, use t.Fail instead")
	}
	f.common.Helper()
	f.common.Fail()
}

// Skipped reports whether the test was skipped.
func (f *F) Skipped() bool {
	// (*F).Skipped may be called by tRunner, which we should allow. However, we
	// shouldn't allow direct (*F).Skipped calls from inside the (*F).Fuzz function.
	if f.inFuzzFn {
		panic("testing: f.Skipped was called inside the fuzz target, use t.Skipped instead")
	}
	f.common.Helper()
	return f.common.Skipped()
}

// Add will add the arguments to the seed corpus for the fuzz test. This will be
// a no-op if called after or within the fuzz target, and args must match the
// arguments for the fuzz target.
func (f *F) Add(args ...any) {
	var values []any
	for i := range args {
		if t := reflect.TypeOf(args[i]); !supportedTypes[t] {
			panic(fmt.Sprintf("testing: unsupported type to Add %v", t))
		}
		values = append(values, args[i])
	}
	f.corpus = append(f.corpus, corpusEntry{Values: values, IsSeed: true, Path: fmt.Sprintf("seed#%d", len(f.corpus))})
}

// supportedTypes represents all of the supported types which can be fuzzed.
var supportedTypes = map[reflect.Type]bool{
	reflect.TypeOf(([]byte)("")):  true,
	reflect.TypeOf((string)("")):  true,
	reflect.TypeOf((bool)(false)): true,
	reflect.TypeOf((byte)(0)):     true,
	reflect.TypeOf((rune)(0)):     true,
	reflect.TypeOf((float32)(0)):  true,
	reflect.TypeOf((float64)(0)):  true,
	reflect.TypeOf((int)(0)):      true,
	reflect.TypeOf((int8)(0)):     true,
	reflect.TypeOf((int16)(0)):    true,
	reflect.TypeOf((int32)(0)):    true,
	reflect.TypeOf((int64)(0)):    true,
	reflect.TypeOf((uint)(0)):     true,
	reflect.TypeOf((uint8)(0)):    true,
	reflect.TypeOf((uint16)(0)):   true,
	reflect.TypeOf((uint32)(0)):   true,
	reflect.TypeOf((uint64)(0)):   true,
}

// Fuzz runs the fuzz function, ff, for fuzz testing. If ff fails for a set of
// arguments, those arguments will be added to the seed corpus.
//
// ff must be a function with no return value whose first argument is [*T] and
// whose remaining arguments are the types to be fuzzed.
// For example:
//
//	f.Fuzz(func(t *testing.T, b []byte, i int) { ... })
//
// The following types are allowed: []byte, string, bool, byte, rune, float32,
// float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64.
// More types may be supported in the future.
//
// ff must not call any [*F] methods, e.g. [F.Log], [F.Error], [F.Skip]. Use
// the corresponding [*T] method instead. The only [*F] methods that are allowed in
// the F.Fuzz function are [F.Failed] and [F.Name].
//
// This function should be fast and deterministic, and its behavior should not
// depend on shared state. No mutable input arguments, or pointers to them,
// should be retained between executions of the fuzz function, as the memory
// backing them may be mutated during a subsequent invocation. ff must not
// modify the underlying data of the arguments provided by the fuzzing engine.
//
// When fuzzing, F.Fuzz does not return until a problem is found, time runs out
// (set with -fuzztime), or the test process is interrupted by a signal. F.Fuzz
// should be called exactly once, unless [F.Skip] or [F.Fail] is called beforehand.
func (f *F) Fuzz(ff any) {
	if f.fuzzCalled {
		panic("testing: F.Fuzz called more than once")
	}
	f.fuzzCalled = true
	if f.failed {
		return
	}
	f.Helper()

	// ff should be in the form func(*testing.T, ...interface{})
	fn := reflect.ValueOf(ff)
	fnType := fn.Type()
	if fnType.Kind() != reflect.Func {
		panic("testing: F.Fuzz must receive a function")
	}
	if fnType.NumIn() < 2 || fnType.In(0) != reflect.TypeOf((*T)(nil)) {
		panic("testing: fuzz target must receive at least two arguments, where the first argument is a *T")
	}
	if fnType.NumOut() != 0 {
		panic("testing: fuzz target must not return a value")
	}

	// Save the types of the function to compare against the corpus.
	var types []reflect.Type
	for i := 1; i < fnType.NumIn(); i++ {
		t := fnType.In(i)
		if !supportedTypes[t] {
			panic(fmt.Sprintf("testing: unsupported type for fuzzing %v", t))
		}
		types = append(types, t)
	}

	// Load the testdata seed corpus. Check types of entries in the testdata
	// corpus and entries declared with F.Add.
	//
	// Don't load the seed corpus if this is a worker process; we won't use it.
	if f.fstate.mode != fuzzWorker {
		for _, c := range f.corpus {
			if err := f.fstate.deps.CheckCorpus(c.Values, types); err != nil {
				// TODO(#48302): Report the source location of the F.Add call.
				f.Fatal(err)
			}
		}

		// Load seed corpus
		c, err := f.fstate.deps.ReadCorpus(filepath.Join(corpusDir, f.name), types)
		if err != nil {
			f.Fatal(err)
		}
		for i := range c {
			c[i].IsSeed = true // these are all seed corpus values
			if f.fstate.mode == fuzzCoordinator {
				// If this is the coordinator process, zero the values, since we don't need
				// to hold onto them.
				c[i].Values = nil
			}
		}

		f.corpus = append(f.corpus, c...)
	}

	// run calls fn on a given input, as a subtest with its own T.
	// run is analogous to T.Run. The test filtering and cleanup works similarly.
	// fn is called in its own goroutine.
	run := func(captureOut io.Writer, e corpusEntry) (ok bool) {
		if e.Values == nil {
			// The corpusEntry must have non-nil Values in order to run the
			// test. If Values is nil, it is a bug in our code.
			panic(fmt.Sprintf("corpus file %q was not unmarshaled", e.Path))
		}
		if shouldFailFast() {
			return true
		}
		testName := f.name
		if e.Path != "" {
			testName = fmt.Sprintf("%s/%s", testName, filepath.Base(e.Path))
		}
		if f.tstate.isFuzzing {
			// Don't preserve subtest names while fuzzing. If fn calls T.Run,
			// there will be a very large number of subtests with duplicate names,
			// which will use a large amount of memory. The subtest names aren't
			// useful since there's no way to re-run them deterministically.
			f.tstate.match.clearSubNames()
		}

		ctx, cancelCtx := context.WithCancel(f.ctx)

		// Record the stack trace at the point of this call so that if the subtest
		// function - which runs in a separate stack - is marked as a helper, we can
		// continue walking the stack into the parent test.
		var pc [maxStackLen]uintptr
		n := runtime.Callers(2, pc[:])
		t := &T{
			common: common{
				barrier:   make(chan bool),
				signal:    make(chan bool),
				name:      testName,
				parent:    &f.common,
				level:     f.level + 1,
				creator:   pc[:n],
				chatty:    f.chatty,
				ctx:       ctx,
				cancelCtx: cancelCtx,
			},
			tstate: f.tstate,
		}
		if captureOut != nil {
			// t.parent aliases f.common.
			t.parent.w = captureOut
		}
		t.w = indenter{&t.common}
		if t.chatty != nil {
			t.chatty.Updatef(t.name, "=== RUN   %s\n", t.name)
		}
		f.common.inFuzzFn, f.inFuzzFn = true, true
		go tRunner(t, func(t *T) {
			args := []reflect.Value{reflect.ValueOf(t)}
			for _, v := range e.Values {
				args = append(args, reflect.ValueOf(v))
			}
			// Before resetting the current coverage, defer the snapshot so that
			// we make sure it is called right before the tRunner function
			// exits, regardless of whether it was executed cleanly, panicked,
			// or if the fuzzFn called t.Fatal.
			if f.tstate.isFuzzing {
				defer f.fstate.deps.SnapshotCoverage()
				f.fstate.deps.ResetCoverage()
			}
			fn.Call(args)
		})
		<-t.signal
		if t.chatty != nil && t.chatty.json {
			t.chatty.Updatef(t.parent.name, "=== NAME  %s\n", t.parent.name)
		}
		f.common.inFuzzFn, f.inFuzzFn = false, false
		return !t.Failed()
	}

	switch f.fstate.mode {
	case fuzzCoordinator:
		// Fuzzing is enabled, and this is the test process started by 'go test'.
		// Act as the coordinator process, and coordinate workers to perform the
		// actual fuzzing.
		corpusTargetDir := filepath.Join(corpusDir, f.name)
		cacheTargetDir := filepath.Join(*fuzzCacheDir, f.name)
		err := f.fstate.deps.CoordinateFuzzing(
			fuzzDuration.d,
			int64(fuzzDuration.n),
			minimizeDuration.d,
			int64(minimizeDuration.n),
			*parallel,
			f.corpus,
			types,
			corpusTargetDir,
			cacheTargetDir)
		if err != nil {
			f.result = fuzzResult{Error: err}
			f.Fail()
			fmt.Fprintf(f.w, "%v\n", err)
			if crashErr, ok := err.(fuzzCrashError); ok {
				crashPath := crashErr.CrashPath()
				fmt.Fprintf(f.w, "Failing input written to %s\n", crashPath)
				testName := filepath.Base(crashPath)
				fmt.Fprintf(f.w, "To re-run:\ngo test -run=%s/%s\n", f.name, testName)
			}
		}
		// TODO(jayconrod,katiehockman): Aggregate statistics across workers
		// and add to FuzzResult (ie. time taken, num iterations)

	case fuzzWorker:
		// Fuzzing is enabled, and this is a worker process. Follow instructions
		// from the coordinator.
		if err := f.fstate.deps.RunFuzzWorker(func(e corpusEntry) error {
			// Don't write to f.w (which points to Stdout) if running from a
			// fuzz worker. This would become very verbose, particularly during
			// minimization. Return the error instead, and let the caller deal
			// with the output.
			var buf strings.Builder
			if ok := run(&buf, e); !ok {
				return errors.New(buf.String())
			}
			return nil
		}); err != nil {
			// Internal errors are marked with f.Fail; user code may call this too, before F.Fuzz.
			// The worker will exit with fuzzWorkerExitCode, indicating this is a failure
			// (and 'go test' should exit non-zero) but a failing input should not be recorded.
			f.Errorf("communicating with fuzzing coordinator: %v", err)
		}

	default:
		// Fuzzing is not enabled, or will be done later. Only run the seed
		// corpus now.
		for _, e := range f.corpus {
			name := fmt.Sprintf("%s/%s", f.name, filepath.Base(e.Path))
			if _, ok, _ := f.tstate.match.fullName(nil, name); ok {
				run(f.w, e)
			}
		}
	}
}

func (f *F) report() {
	if *isFuzzWorker || f.parent == nil {
		return
	}
	dstr := fmtDuration(f.duration)
	format := "--- %s: %s (%s)\n"
	if f.Failed() {
		f.flushToParent(f.name, format, "FAIL", f.name, dstr)
	} else if f.chatty != nil {
		if f.Skipped() {
			f.flushToParent(f.name, format, "SKIP", f.name, dstr)
		} else {
			f.flushToParent(f.name, format, "PASS", f.name, dstr)
		}
	}
}

// fuzzResult contains the results of a fuzz run.
type fuzzResult struct {
	N     int           // The number of iterations.
	T     time.Duration // The total time taken.
	Error error         // Error is the error from the failing input
}

func (r fuzzResult) String() string {
	if r.Error == nil {
		return ""
	}
	return r.Error.Error()
}

// fuzzCrashError is satisfied by a failing input detected while fuzzing.
// These errors are written to the seed corpus and can be re-run with 'go test'.
// Errors within the fuzzing framework (like I/O errors between coordinator
// and worker processes) don't satisfy this interface.
type fuzzCrashError interface {
	error
	Unwrap() error

	// CrashPath returns the path of the subtest that corresponds to the saved
	// crash input file in the seed corpus. The test can be re-run with go test
	// -run=$test/$name $test is the fuzz test name, and $name is the
	// filepath.Base of the string returned here.
	CrashPath() string
}

// fuzzState holds fields common to all fuzz tests.
type fuzzState struct {
	deps testDeps
	mode fuzzMode
}

type fuzzMode uint8

const (
	seedCorpusOnly fuzzMode = iota
	fuzzCoordinator
	fuzzWorker
)

// runFuzzTests runs the fuzz tests matching the pattern for -run. This will
// only run the (*F).Fuzz function for each seed corpus without using the
// fuzzing engine to generate or mutate inputs.
func runFuzzTests(deps testDeps, fuzzTests []InternalFuzzTarget, deadline time.Time) (ran, ok bool) {
	ok = true
	if len(fuzzTests) == 0 || *isFuzzWorker {
		return ran, ok
	}
	m := newMatcher(deps.MatchString, *match, "-test.run", *skip)
	var mFuzz *matcher
	if *matchFuzz != "" {
		mFuzz = newMatcher(deps.MatchString, *matchFuzz, "-test.fuzz", *skip)
	}

	for _, procs := range cpuList {
		runtime.GOMAXPROCS(procs)
		for i := uint(0); i < *count; i++ {
			if shouldFailFast() {
				break
			}

			tstate := newTestState(*parallel, m)
			tstate.deadline = deadline
			fstate := &fuzzState{deps: deps, mode: seedCorpusOnly}
			root := common{w: os.Stdout} // gather output in one place
			if Verbose() {
				root.chatty = newChattyPrinter(root.w)
			}
			for _, ft := range fuzzTests {
				if shouldFailFast() {
					break
				}
				testName, matched, _ := tstate.match.fullName(nil, ft.Name)
				if !matched {
					continue
				}
				if mFuzz != nil {
					if _, fuzzMatched, _ := mFuzz.fullName(nil, ft.Name); fuzzMatched {
						// If this will be fuzzed, then don't run the seed corpus
						// right now. That will happen later.
						continue
					}
				}
				ctx, cancelCtx := context.WithCancel(context.Background())
				f := &F{
					common: common{
						signal:    make(chan bool),
						barrier:   make(chan bool),
						name:      testName,
						parent:    &root,
						level:     root.level + 1,
						chatty:    root.chatty,
						ctx:       ctx,
						cancelCtx: cancelCtx,
					},
					tstate: tstate,
					fstate: fstate,
				}
				f.w = indenter{&f.common}
				if f.chatty != nil {
					f.chatty.Updatef(f.name, "=== RUN   %s\n", f.name)
				}
				go fRunner(f, ft.Fn)
				<-f.signal
				if f.chatty != nil && f.chatty.json {
					f.chatty.Updatef(f.parent.name, "=== NAME  %s\n", f.parent.name)
				}
				ok = ok && !f.Failed()
				ran = ran || f.ran
			}
			if !ran {
				// There were no tests to run on this iteration.
				// This won't change, so no reason to keep trying.
				break
			}
		}
	}

	return ran, ok
}

// runFuzzing runs the fuzz test matching the pattern for -fuzz. Only one such
// fuzz test must match. This will run the fuzzing engine to generate and
// mutate new inputs against the fuzz target.
//
// If fuzzing is disabled (-test.fuzz is not set), runFuzzing
// returns immediately.
func runFuzzing(deps testDeps, fuzzTests []InternalFuzzTarget) (ok bool) {
	if len(fuzzTests) == 0 || *matchFuzz == "" {
		return true
	}
	m := newMatcher(deps.MatchString, *matchFuzz, "-test.fuzz", *skip)
	tstate := newTestState(1, m)
	tstate.isFuzzing = true
	fstate := &fuzzState{
		deps: deps,
	}
	root := common{w: os.Stdout}
	if *isFuzzWorker {
		root.w = io.Discard
		fstate.mode = fuzzWorker
	} else {
		fstate.mode = fuzzCoordinator
	}
	if Verbose() && !*isFuzzWorker {
		root.chatty = newChattyPrinter(root.w)
	}
	var fuzzTest *InternalFuzzTarget
	var testName string
	var matched []string
	for i := range fuzzTests {
		name, ok, _ := tstate.match.fullName(nil, fuzzTests[i].Name)
		if !ok {
			continue
		}
		matched = append(matched, name)
		fuzzTest = &fuzzTests[i]
		testName = name
	}
	if len(matched) == 0 {
		fmt.Fprintln(os.Stderr, "testing: warning: no fuzz tests to fuzz")
		return true
	}
	if len(matched) > 1 {
		fmt.Fprintf(os.Stderr, "testing: will not fuzz, -fuzz matches more than one fuzz test: %v\n", matched)
		return false
	}

	ctx, cancelCtx := context.WithCancel(context.Background())
	f := &F{
		common: common{
			signal:    make(chan bool),
			barrier:   nil, // T.Parallel has no effect when fuzzing.
			name:      testName,
			parent:    &root,
			level:     root.level + 1,
			chatty:    root.chatty,
			ctx:       ctx,
			cancelCtx: cancelCtx,
		},
		fstate: fstate,
		tstate: tstate,
	}
	f.w = indenter{&f.common}
	if f.chatty != nil {
		f.chatty.Updatef(f.name, "=== RUN   %s\n", f.name)
	}
	go fRunner(f, fuzzTest.Fn)
	<-f.signal
	if f.chatty != nil {
		f.chatty.Updatef(f.parent.name, "=== NAME  %s\n", f.parent.name)
	}
	return !f.failed
}

// fRunner wraps a call to a fuzz test and ensures that cleanup functions are
// called and status flags are set. fRunner should be called in its own
// goroutine. To wait for its completion, receive from f.signal.
//
// fRunner is analogous to tRunner, which wraps subtests started with T.Run.
// Unit tests and fuzz tests work a little differently, so for now, these
// functions aren't consolidated. In particular, because there are no F.Run and
// F.Parallel methods, i.e., no fuzz sub-tests or parallel fuzz tests, a few
// simplifications are made. We also require that F.Fuzz, F.Skip, or F.Fail is
// called.
func fRunner(f *F, fn func(*F)) {
	// When this goroutine is done, either because runtime.Goexit was called, a
	// panic started, or fn returned normally, record the duration and send
	// t.signal, indicating the fuzz test is done.
	defer func() {
		// Detect whether the fuzz test panicked or called runtime.Goexit
		// without calling F.Fuzz, F.Fail, or F.Skip. If it did, panic (possibly
		// replacing a nil panic value). Nothing should recover after fRunner
		// unwinds, so this should crash the process and print stack.
		// Unfortunately, recovering here adds stack frames, but the location of
		// the original panic should still be
		// clear.
		f.checkRaces()
		if f.Failed() {
			numFailed.Add(1)
		}
		err := recover()
		if err == nil {
			f.mu.RLock()
			fuzzNotCalled := !f.fuzzCalled && !f.skipped && !f.failed
			if !f.finished && !f.skipped && !f.failed {
				err = errNilPanicOrGoexit
			}
			f.mu.RUnlock()
			if fuzzNotCalled && err == nil {
				f.Error("returned without calling F.Fuzz, F.Fail, or F.Skip")
			}
		}

		// Use a deferred call to ensure that we report that the test is
		// complete even if a cleanup function calls F.FailNow. See issue 41355.
		didPanic := false
		defer func() {
			if !didPanic {
				// Only report that the test is complete if it doesn't panic,
				// as otherwise the test binary can exit before the panic is
				// reported to the user. See issue 41479.
				f.signal <- true
			}
		}()

		// If we recovered a panic or inappropriate runtime.Goexit, fail the test,
		// flush the output log up to the root, then panic.
		doPanic := func(err any) {
			f.Fail()
			if r := f.runCleanup(recoverAndReturnPanic); r != nil {
				f.Logf("cleanup panicked with %v", r)
			}
			for root := &f.common; root.parent != nil; root = root.parent {
				root.mu.Lock()
				root.duration += highPrecisionTimeSince(root.start)
				d := root.duration
				root.mu.Unlock()
				root.flushToParent(root.name, "--- FAIL: %s (%s)\n", root.name, fmtDuration(d))
			}
			didPanic = true
			panic(err)
		}
		if err != nil {
			doPanic(err)
		}

		// No panic or inappropriate Goexit.
		f.duration += highPrecisionTimeSince(f.start)

		if len(f.sub) > 0 {
			// Unblock inputs that called T.Parallel while running the seed corpus.
			// This only affects fuzz tests run as normal tests.
			// While fuzzing, T.Parallel has no effect, so f.sub is empty, and this
			// branch is not taken. f.barrier is nil in that case.
			f.tstate.release()
			close(f.barrier)
			// Wait for the subtests to complete.
			for _, sub := range f.sub {
				<-sub.signal
			}
			cleanupStart := highPrecisionTimeNow()
			err := f.runCleanup(recoverAndReturnPanic)
			f.duration += highPrecisionTimeSince(cleanupStart)
			if err != nil {
				doPanic(err)
			}
		}

		// Report after all subtests have finished.
		f.report()
		f.doneRaceCanary = true
		f.done = true
		f.setRan()
	}()
	defer func() {
		if len(f.sub) == 0 {
			f.runCleanup(normalPanic)
		}
	}()

	f.start = highPrecisionTimeNow()
	f.resetRaces()
	fn(f)

	// Code beyond this point will not be executed when FailNow or SkipNow
	// is invoked.
	f.mu.Lock()
	f.finished = true
	f.mu.Unlock()
}
