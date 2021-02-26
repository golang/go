// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sync/atomic"
	"time"
)

func initFuzzFlags() {
	matchFuzz = flag.String("test.fuzz", "", "run the fuzz target matching `regexp`")
	fuzzDuration = flag.Duration("test.fuzztime", 0, "time to spend fuzzing; default (0) is to run indefinitely")
	fuzzCacheDir = flag.String("test.fuzzcachedir", "", "directory where interesting fuzzing inputs are stored")
	isFuzzWorker = flag.Bool("test.fuzzworker", false, "coordinate with the parent process to fuzz random values")
}

var (
	matchFuzz    *string
	fuzzDuration *time.Duration
	fuzzCacheDir *string
	isFuzzWorker *bool

	// corpusDir is the parent directory of the target's seed corpus within
	// the package.
	corpusDir = "testdata/corpus"
)

// InternalFuzzTarget is an internal type but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
type InternalFuzzTarget struct {
	Name string
	Fn   func(f *F)
}

// F is a type passed to fuzz targets for fuzz testing.
type F struct {
	common
	fuzzContext *fuzzContext
	testContext *testContext
	inFuzzFn    bool          // set to true when fuzz function is running
	corpus      []corpusEntry // corpus is the in-memory corpus
	result      FuzzResult    // result is the result of running the fuzz target
	fuzzCalled  bool
}

var _ TB = (*F)(nil)

// corpusEntry is an alias to the same type as internal/fuzz.CorpusEntry.
// We use a type alias because we don't want to export this type, and we can't
// importing internal/fuzz from testing.
type corpusEntry = struct {
	Name   string
	Data   []byte
	Values []interface{}
}

// Cleanup registers a function to be called when the test and all its
// subtests complete. Cleanup functions will be called in last added,
// first called order.
func (f *F) Cleanup(fn func()) {
	if f.inFuzzFn {
		panic("testing: f.Cleanup was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Cleanup(fn)
}

// Error is equivalent to Log followed by Fail.
func (f *F) Error(args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Error was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Error(args...)
}

// Errorf is equivalent to Logf followed by Fail.
func (f *F) Errorf(format string, args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Errorf was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Errorf(format, args...)
}

// Fail marks the function as having failed but continues execution.
func (f *F) Fail() {
	if f.inFuzzFn {
		panic("testing: f.Fail was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Fail()
}

// FailNow marks the function as having failed and stops its execution
// by calling runtime.Goexit (which then runs all deferred calls in the
// current goroutine).
// Execution will continue at the next test or benchmark.
// FailNow must be called from the goroutine running the
// test or benchmark function, not from other goroutines
// created during the test. Calling FailNow does not stop
// those other goroutines.
func (f *F) FailNow() {
	if f.inFuzzFn {
		panic("testing: f.FailNow was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.FailNow()
}

// Fatal is equivalent to Log followed by FailNow.
func (f *F) Fatal(args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Fatal was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Fatal(args...)
}

// Fatalf is equivalent to Logf followed by FailNow.
func (f *F) Fatalf(format string, args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Fatalf was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Fatalf(format, args...)
}

// Helper marks the calling function as a test helper function.
// When printing file and line information, that function will be skipped.
// Helper may be called simultaneously from multiple goroutines.
func (f *F) Helper() {
	if f.inFuzzFn {
		panic("testing: f.Helper was called inside the f.Fuzz function")
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

// Skip is equivalent to Log followed by SkipNow.
func (f *F) Skip(args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Skip was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Skip(args...)
}

// SkipNow marks the test as having been skipped and stops its execution
// by calling runtime.Goexit.
// If a test fails (see Error, Errorf, Fail) and is then skipped,
// it is still considered to have failed.
// Execution will continue at the next test or benchmark. See also FailNow.
// SkipNow must be called from the goroutine running the test, not from
// other goroutines created during the test. Calling SkipNow does not stop
// those other goroutines.
func (f *F) SkipNow() {
	if f.inFuzzFn {
		panic("testing: f.SkipNow was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.SkipNow()
}

// Skipf is equivalent to Logf followed by SkipNow.
func (f *F) Skipf(format string, args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Skipf was called inside the f.Fuzz function")
	}
	f.common.Helper()
	f.common.Skipf(format, args...)
}

// TempDir returns a temporary directory for the test to use.
// The directory is automatically removed by Cleanup when the test and
// all its subtests complete.
// Each subsequent call to t.TempDir returns a unique directory;
// if the directory creation fails, TempDir terminates the test by calling Fatal.
func (f *F) TempDir() string {
	if f.inFuzzFn {
		panic("testing: f.TempDir was called inside the f.Fuzz function")
	}
	f.common.Helper()
	return f.common.TempDir()
}

// Add will add the arguments to the seed corpus for the fuzz target. This will
// be a no-op if called after or within the Fuzz function. The args must match
// those in the Fuzz function.
func (f *F) Add(args ...interface{}) {
	var values []interface{}
	for i := range args {
		if t := reflect.TypeOf(args[i]); !supportedTypes[t] {
			panic(fmt.Sprintf("testing: unsupported type to Add %v", t))
		}
		values = append(values, args[i])
	}
	f.corpus = append(f.corpus, corpusEntry{Values: values})
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
// This is a terminal function which will terminate the currently running fuzz
// target by calling runtime.Goexit. To run any code after this function, use
// Cleanup.
func (f *F) Fuzz(ff interface{}) {
	if f.fuzzCalled {
		panic("testing: F.Fuzz called more than once")
	}
	f.fuzzCalled = true
	f.Helper()

	// ff should be in the form func(*testing.T, ...interface{})
	fn := reflect.ValueOf(ff)
	fnType := fn.Type()
	if fnType.Kind() != reflect.Func {
		panic("testing: F.Fuzz must receive a function")
	}
	if fnType.NumIn() < 2 || fnType.In(0) != reflect.TypeOf((*T)(nil)) {
		panic("testing: F.Fuzz function must receive at least two arguments, where the first argument is a *T")
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

	// Load seed corpus
	c, err := f.fuzzContext.readCorpus(filepath.Join(corpusDir, f.name), types)
	if err != nil {
		f.Fatal(err)
	}
	f.corpus = append(f.corpus, c...)

	// run calls fn on a given input, as a subtest with its own T.
	// run is analogous to T.Run. The test filtering and cleanup works similarly.
	// fn is called in its own goroutine.
	//
	// TODO(jayconrod,katiehockman): dedupe testdata corpus with entries from f.Add
	// TODO(jayconrod,katiehockman): handle T.Parallel calls within fuzz function.
	run := func(e corpusEntry) error {
		if e.Values == nil {
			// Every code path should have already unmarshaled Data into Values.
			// It's our fault if it didn't.
			panic(fmt.Sprintf("corpus file %q was not unmarshaled", e.Name))
		}
		testName, ok, _ := f.testContext.match.fullName(&f.common, e.Name)
		if !ok || shouldFailFast() {
			return nil
		}
		// Record the stack trace at the point of this call so that if the subtest
		// function - which runs in a separate stack - is marked as a helper, we can
		// continue walking the stack into the parent test.
		var pc [maxStackLen]uintptr
		n := runtime.Callers(2, pc[:])
		t := &T{
			common: common{
				barrier: make(chan bool),
				signal:  make(chan bool),
				name:    testName,
				parent:  &f.common,
				level:   f.level + 1,
				creator: pc[:n],
				chatty:  f.chatty,
			},
			context: f.testContext,
		}
		t.w = indenter{&t.common}
		if t.chatty != nil {
			t.chatty.Updatef(t.name, "=== RUN  %s\n", t.name)
		}
		f.inFuzzFn = true
		go tRunner(t, func(t *T) {
			args := []reflect.Value{reflect.ValueOf(t)}
			for _, v := range e.Values {
				args = append(args, reflect.ValueOf(v))
			}
			fn.Call(args)
		})
		<-t.signal
		f.inFuzzFn = false
		if t.Failed() {
			return errors.New(string(t.output))
		}
		return nil
	}

	switch {
	case f.fuzzContext.coordinateFuzzing != nil:
		// Fuzzing is enabled, and this is the test process started by 'go test'.
		// Act as the coordinator process, and coordinate workers to perform the
		// actual fuzzing.
		corpusTargetDir := filepath.Join(corpusDir, f.name)
		cacheTargetDir := filepath.Join(*fuzzCacheDir, f.name)
		err := f.fuzzContext.coordinateFuzzing(*fuzzDuration, *parallel, f.corpus, types, corpusTargetDir, cacheTargetDir)
		if err != nil {
			f.result = FuzzResult{Error: err}
			f.Error(err)
			if crashErr, ok := err.(fuzzCrashError); ok {
				crashName := crashErr.CrashName()
				f.Logf("Crash written to %s", filepath.Join("testdata/corpus", f.name, crashName))
				f.Logf("To re-run:\ngo test %s -run=%s/%s", f.fuzzContext.importPath(), f.name, crashName)
			}
		}
		// TODO(jayconrod,katiehockman): Aggregate statistics across workers
		// and add to FuzzResult (ie. time taken, num iterations)

	case f.fuzzContext.runFuzzWorker != nil:
		// Fuzzing is enabled, and this is a worker process. Follow instructions
		// from the coordinator.
		if err := f.fuzzContext.runFuzzWorker(run); err != nil {
			// TODO(jayconrod,katiehockman): how should we handle a failure to
			// communicate with the coordinator? Might be caused by the coordinator
			// terminating early.
			f.Errorf("communicating with fuzzing coordinator: %v", err)
		}

	default:
		// Fuzzing is not enabled. Only run the seed corpus.
		for _, e := range f.corpus {
			run(e)
		}
	}

	// Record that the fuzz function (or coordinateFuzzing or runFuzzWorker)
	// returned normally. This is used to distinguish runtime.Goexit below
	// from panic(nil).
	f.finished = true

	// Terminate the goroutine. F.Fuzz should not return.
	// We cannot call runtime.Goexit from a deferred function: if there is a
	// panic, that would replace the panic value with nil.
	runtime.Goexit()
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

// FuzzResult contains the results of a fuzz run.
type FuzzResult struct {
	N     int           // The number of iterations.
	T     time.Duration // The total time taken.
	Error error         // Error is the error from the crash
}

func (r FuzzResult) String() string {
	s := ""
	if r.Error == nil {
		return s
	}
	s = fmt.Sprintf("%s", r.Error.Error())
	return s
}

// fuzzCrashError is satisfied by a crash detected within the fuzz function.
// These errors are written to the seed corpus and can be re-run with 'go test'.
// Errors within the fuzzing framework (like I/O errors between coordinator
// and worker processes) don't satisfy this interface.
type fuzzCrashError interface {
	error
	Unwrap() error

	// CrashName returns the name of the subtest that corresponds to the saved
	// crash input file in the seed corpus. The test can be re-run with
	// go test $pkg -run=$target/$name where $pkg is the package's import path,
	// $target is the fuzz target name, and $name is the string returned here.
	CrashName() string
}

// fuzzContext holds all fields that are common to all fuzz targets.
type fuzzContext struct {
	importPath        func() string
	coordinateFuzzing func(time.Duration, int, []corpusEntry, []reflect.Type, string, string) error
	runFuzzWorker     func(func(corpusEntry) error) error
	readCorpus        func(string, []reflect.Type) ([]corpusEntry, error)
}

// runFuzzTargets runs the fuzz targets matching the pattern for -run. This will
// only run the f.Fuzz function for each seed corpus without using the fuzzing
// engine to generate or mutate inputs.
func runFuzzTargets(deps testDeps, fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	ok = true
	if len(fuzzTargets) == 0 || *isFuzzWorker {
		return ran, ok
	}
	m := newMatcher(deps.MatchString, *match, "-test.run")
	tctx := newTestContext(*parallel, m)
	fctx := &fuzzContext{
		importPath: deps.ImportPath,
		readCorpus: deps.ReadCorpus,
	}
	root := common{w: os.Stdout} // gather output in one place
	if Verbose() {
		root.chatty = newChattyPrinter(root.w)
	}
	for _, ft := range fuzzTargets {
		if shouldFailFast() {
			break
		}
		testName, matched, _ := tctx.match.fullName(nil, ft.Name)
		if !matched {
			continue
		}
		f := &F{
			common: common{
				signal: make(chan bool),
				name:   testName,
				parent: &root,
				level:  root.level + 1,
				chatty: root.chatty,
			},
			testContext: tctx,
			fuzzContext: fctx,
		}
		f.w = indenter{&f.common}
		if f.chatty != nil {
			f.chatty.Updatef(f.name, "=== RUN  %s\n", f.name)
		}

		go fRunner(f, ft.Fn)
		<-f.signal
	}
	return root.ran, !root.Failed()
}

// runFuzzing runs the fuzz target matching the pattern for -fuzz. Only one such
// fuzz target must match. This will run the fuzzing engine to generate and
// mutate new inputs against the f.Fuzz function.
//
// If fuzzing is disabled (-test.fuzz is not set), runFuzzing
// returns immediately.
func runFuzzing(deps testDeps, fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	if len(fuzzTargets) == 0 || *matchFuzz == "" {
		return false, true
	}
	m := newMatcher(deps.MatchString, *matchFuzz, "-test.fuzz")
	tctx := newTestContext(1, m)
	fctx := &fuzzContext{
		importPath: deps.ImportPath,
		readCorpus: deps.ReadCorpus,
	}
	if *isFuzzWorker {
		fctx.runFuzzWorker = deps.RunFuzzWorker
	} else {
		fctx.coordinateFuzzing = deps.CoordinateFuzzing
	}
	root := common{w: os.Stdout}
	if Verbose() && !*isFuzzWorker {
		root.chatty = newChattyPrinter(root.w)
	}
	var target *InternalFuzzTarget
	var f *F
	for i := range fuzzTargets {
		ft := &fuzzTargets[i]
		testName, matched, _ := tctx.match.fullName(nil, ft.Name)
		if !matched {
			continue
		}
		if target != nil {
			fmt.Fprintln(os.Stderr, "testing: warning: -fuzz matches more than one target, won't fuzz")
			return false, true
		}
		target = ft
		f = &F{
			common: common{
				signal: make(chan bool),
				name:   testName,
				parent: &root,
				level:  root.level + 1,
				chatty: root.chatty,
			},
			fuzzContext: fctx,
			testContext: tctx,
		}
		f.w = indenter{&f.common}
	}
	if target == nil {
		return false, true
	}
	if f.chatty != nil {
		f.chatty.Updatef(f.name, "=== FUZZ  %s\n", f.name)
	}
	go fRunner(f, target.Fn)
	<-f.signal
	return f.ran, !f.failed
}

// fRunner wraps a call to a fuzz target and ensures that cleanup functions are
// called and status flags are set. fRunner should be called in its own
// goroutine. To wait for its completion, receive f.signal.
//
// fRunner is analogous with tRunner, which wraps subtests started with T.Run.
// Tests and fuzz targets work a little differently, so for now, these functions
// aren't consoldiated.
func fRunner(f *F, fn func(*F)) {
	// When this goroutine is done, either because runtime.Goexit was called,
	// a panic started, or fn returned normally, record the duration and send
	// t.signal, indicating the fuzz target is done.
	defer func() {
		// Detect whether the fuzz target panicked or called runtime.Goexit without
		// calling F.Fuzz, F.Fail, or F.Skip. If it did, panic (possibly replacing
		// a nil panic value). Nothing should recover after fRunner unwinds,
		// so this should crash the process with a stack. Unfortunately, recovering
		// here adds stack frames, but the location of the original panic should
		// still be clear.
		if f.Failed() {
			atomic.AddUint32(&numFailed, 1)
		}
		err := recover()
		f.mu.RLock()
		ok := f.skipped || f.failed || (f.fuzzCalled && f.finished)
		f.mu.RUnlock()
		if err == nil && !ok {
			err = errNilPanicOrGoexit
		}

		// If we recovered a panic or inappropriate runtime.Goexit, fail the test,
		// flush the output log up to the root, then panic.
		if err != nil {
			f.Fail()
			for root := &f.common; root.parent != nil; root = root.parent {
				root.mu.Lock()
				root.duration += time.Since(root.start)
				d := root.duration
				root.mu.Unlock()
				root.flushToParent(root.name, "--- FAIL: %s (%s)\n", root.name, fmtDuration(d))
			}
			panic(err)
		}

		// No panic or inappropriate Goexit. Record duration and report the result.
		f.duration += time.Since(f.start)
		f.report()
		f.done = true
		f.setRan()

		// Only report that the test is complete if it doesn't panic,
		// as otherwise the test binary can exit before the panic is
		// reported to the user. See issue 41479.
		f.signal <- true
	}()
	defer func() {
		f.runCleanup(normalPanic)
	}()

	f.start = time.Now()
	fn(f)

	// Code beyond this point is only executed if fn returned normally.
	// That means fn did not call F.Fuzz or F.Skip. It should have called F.Fail.
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.failed {
		panic(f.name + " returned without calling F.Fuzz, F.Fail, or F.Skip")
	}
}
