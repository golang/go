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
	"runtime"
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
	context  *fuzzContext
	inFuzzFn bool          // set to true when fuzz function is running
	corpus   []corpusEntry // corpus is the in-memory corpus
	result   FuzzResult    // result is the result of running the fuzz target
	fuzzFunc func(f *F)    // fuzzFunc is the function which makes up the fuzz target
}

var _ TB = (*F)(nil)

// corpusEntry is an alias to the same type as internal/fuzz.CorpusEntry.
// We use a type alias because we don't want to export this type, and we can't
// importing internal/fuzz from testing.
type corpusEntry = struct {
	Name string
	Data []byte
}

// Cleanup registers a function to be called when the test and all its
// subtests complete. Cleanup functions will be called in last added,
// first called order.
func (f *F) Cleanup(fn func()) {
	if f.inFuzzFn {
		panic("testing: f.Cleanup was called inside the f.Fuzz function")
	}
	f.common.Cleanup(fn)
}

// Error is equivalent to Log followed by Fail.
func (f *F) Error(args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Error was called inside the f.Fuzz function")
	}
	f.common.Error(args...)
}

// Errorf is equivalent to Logf followed by Fail.
func (f *F) Errorf(format string, args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Errorf was called inside the f.Fuzz function")
	}
	f.common.Errorf(format, args...)
}

// Fail marks the function as having failed but continues execution.
func (f *F) Fail() {
	if f.inFuzzFn {
		panic("testing: f.Fail was called inside the f.Fuzz function")
	}
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
	f.common.FailNow()
}

// Fatal is equivalent to Log followed by FailNow.
func (f *F) Fatal(args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Fatal was called inside the f.Fuzz function")
	}
	f.common.Fatal(args...)
}

// Fatalf is equivalent to Logf followed by FailNow.
func (f *F) Fatalf(format string, args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Fatalf was called inside the f.Fuzz function")
	}
	f.common.Fatalf(format, args...)
}

// Helper marks the calling function as a test helper function.
// When printing file and line information, that function will be skipped.
// Helper may be called simultaneously from multiple goroutines.
func (f *F) Helper() {
	if f.inFuzzFn {
		panic("testing: f.Helper was called inside the f.Fuzz function")
	}
	f.common.Helper()
}

// Skip is equivalent to Log followed by SkipNow.
func (f *F) Skip(args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Skip was called inside the f.Fuzz function")
	}
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
	f.common.SkipNow()
}

// Skipf is equivalent to Logf followed by SkipNow.
func (f *F) Skipf(format string, args ...interface{}) {
	if f.inFuzzFn {
		panic("testing: f.Skipf was called inside the f.Fuzz function")
	}
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
	return f.common.TempDir()
}

// Add will add the arguments to the seed corpus for the fuzz target. This will
// be a no-op if called after or within the Fuzz function. The args must match
// those in the Fuzz function.
func (f *F) Add(args ...interface{}) {
	if len(args) == 0 {
		panic("testing: Add must have at least one argument")
	}
	if len(args) != 1 {
		// TODO: support more than one argument
		panic("testing: Add only supports one argument currently")
	}
	switch v := args[0].(type) {
	case []byte:
		f.corpus = append(f.corpus, corpusEntry{Data: v})
	// TODO: support other types
	default:
		panic("testing: Add only supports []byte currently")
	}
}

// Fuzz runs the fuzz function, ff, for fuzz testing. If ff fails for a set of
// arguments, those arguments will be added to the seed corpus.
//
// This is a terminal function which will terminate the currently running fuzz
// target by calling runtime.Goexit. To run any code after this function, use
// Cleanup.
func (f *F) Fuzz(ff interface{}) {
	defer runtime.Goexit() // exit after this function

	fn, ok := ff.(func(*T, []byte))
	if !ok {
		panic("testing: Fuzz function must have type func(*testing.T, []byte)")
	}

	// Load seed corpus
	c, err := f.context.readCorpus(filepath.Join(corpusDir, f.name))
	if err != nil {
		f.Fatal(err)
	}
	f.corpus = append(f.corpus, c...)
	// TODO(jayconrod,katiehockman): dedupe testdata corpus with entries from f.Add

	var errStr string
	run := func(t *T, b []byte) {
		defer func() {
			err := recover()
			// If the function has recovered but the test hasn't finished,
			// it is due to a nil panic or runtime.GoExit.
			if !t.finished && err == nil {
				err = errNilPanicOrGoexit
			}
			if err != nil {
				t.Fail()
				t.output = []byte(fmt.Sprintf("    %s", err))
			}
			f.inFuzzFn = false
			t.signal <- true // signal that the test has finished
		}()
		// TODO(katiehockman, jayconrod): consider replacing inFuzzFn with
		// general purpose flag that checks whether specific methods can be
		// called.
		f.inFuzzFn = true
		fn(t, b)
		t.finished = true
	}

	switch {
	case f.context.coordinateFuzzing != nil:
		// Fuzzing is enabled, and this is the test process started by 'go test'.
		// Act as the coordinator process, and coordinate workers to perform the
		// actual fuzzing.
		corpusTargetDir := filepath.Join(corpusDir, f.name)
		cacheTargetDir := filepath.Join(*fuzzCacheDir, f.name)
		err := f.context.coordinateFuzzing(*fuzzDuration, *parallel, f.corpus, corpusTargetDir, cacheTargetDir)
		if err != nil {
			f.Fail()
			f.result = FuzzResult{Error: err}
		}
		f.setRan()
		f.finished = true
		// TODO(jayconrod,katiehockman): Aggregate statistics across workers
		// and add to FuzzResult (ie. time taken, num iterations)

	case f.context.runFuzzWorker != nil:
		// Fuzzing is enabled, and this is a worker process. Follow instructions
		// from the coordinator.
		err := f.context.runFuzzWorker(func(e corpusEntry) error {
			t := &T{
				common: common{
					signal: make(chan bool),
					w:      f.w,
					chatty: f.chatty,
				},
				context: newTestContext(1, nil),
			}
			go run(t, e.Data)
			<-t.signal
			if t.Failed() {
				return errors.New(string(t.output))
			}
			return nil
		})
		if err != nil {
			// TODO(jayconrod,katiehockman): how should we handle a failure to
			// communicate with the coordinator? Might be caused by the coordinator
			// terminating early.
			fmt.Fprintf(os.Stderr, "testing: communicating with fuzz coordinator: %v\n", err)
			os.Exit(1)
		}
		f.setRan()
		f.finished = true

	default:
		// Fuzzing is not enabled. Only run the seed corpus.
		for _, c := range f.corpus {
			t := &T{
				common: common{
					signal: make(chan bool),
					w:      f.w,
					chatty: f.chatty,
				},
				context: newTestContext(1, nil),
			}
			go run(t, c.Data)
			<-t.signal
			if t.Failed() {
				f.Fail()
				errStr += string(t.output)
			}
			f.setRan()
		}
		f.finished = true
		if f.Failed() {
			f.result = FuzzResult{Error: errors.New(errStr)}
			return
		}
	}
}

func (f *F) report() {
	if *isFuzzWorker {
		return
	}
	if f.Failed() {
		fmt.Fprintf(f.w, "--- FAIL: %s\n%s\n", f.name, f.result.String())
	} else if f.chatty != nil {
		if f.Skipped() {
			f.chatty.Updatef(f.name, "SKIP\n")
		} else {
			f.chatty.Updatef(f.name, "PASS\n")
		}
	}
}

// run runs each fuzz target in its own goroutine with its own *F.
func (f *F) run(ft InternalFuzzTarget) (ran, ok bool) {
	f = &F{
		common: common{
			signal: make(chan bool),
			name:   ft.Name,
			chatty: f.chatty,
			w:      f.w,
		},
		context: f.context,
	}
	if f.chatty != nil {
		f.chatty.Updatef(ft.Name, "=== RUN   %s\n", ft.Name)
	}
	go f.runTarget(ft.Fn)
	<-f.signal
	return f.ran, !f.failed
}

// runTarget runs the given target, handling panics and exits
// within the test, and reporting errors.
func (f *F) runTarget(fn func(*F)) {
	defer func() {
		err := recover()
		// If the function has recovered but the test hasn't finished,
		// it is due to a nil panic or runtime.GoExit.
		if !f.finished && err == nil {
			err = errNilPanicOrGoexit
		}
		if err != nil {
			f.Fail()
			f.result = FuzzResult{Error: fmt.Errorf("    %s", err)}
		}
		f.report()
		f.setRan()
		f.signal <- true // signal that the test has finished
	}()
	defer f.runCleanup(normalPanic)
	fn(f)
	f.finished = true
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

// fuzzContext holds all fields that are common to all fuzz targets.
type fuzzContext struct {
	runMatch          *matcher
	fuzzMatch         *matcher
	coordinateFuzzing func(time.Duration, int, []corpusEntry, string, string) error
	runFuzzWorker     func(func(corpusEntry) error) error
	readCorpus        func(string) ([]corpusEntry, error)
}

// runFuzzTargets runs the fuzz targets matching the pattern for -run. This will
// only run the f.Fuzz function for each seed corpus without using the fuzzing
// engine to generate or mutate inputs.
func runFuzzTargets(deps testDeps, fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	ok = true
	if len(fuzzTargets) == 0 || *isFuzzWorker {
		return ran, ok
	}
	ctx := &fuzzContext{
		runMatch:   newMatcher(deps.MatchString, *match, "-test.run"),
		readCorpus: deps.ReadCorpus,
	}
	var fts []InternalFuzzTarget
	for _, ft := range fuzzTargets {
		if _, matched, _ := ctx.runMatch.fullName(nil, ft.Name); matched {
			fts = append(fts, ft)
		}
	}
	f := &F{
		common: common{
			w: os.Stdout,
		},
		fuzzFunc: func(f *F) {
			for _, ft := range fts {
				// Run each fuzz target in it's own goroutine.
				ftRan, ftOk := f.run(ft)
				ran = ran || ftRan
				ok = ok && ftOk
			}
		},
		context: ctx,
	}
	if Verbose() {
		f.chatty = newChattyPrinter(f.w)
	}
	f.fuzzFunc(f)
	return ran, ok
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
	ctx := &fuzzContext{
		fuzzMatch:  newMatcher(deps.MatchString, *matchFuzz, "-test.fuzz"),
		readCorpus: deps.ReadCorpus,
	}
	if *isFuzzWorker {
		ctx.runFuzzWorker = deps.RunFuzzWorker
	} else {
		ctx.coordinateFuzzing = deps.CoordinateFuzzing
	}
	f := &F{
		common: common{
			signal: make(chan bool),
			w:      os.Stdout,
		},
		context: ctx,
	}
	var target *InternalFuzzTarget
	for i := range fuzzTargets {
		ft := &fuzzTargets[i]
		testName, matched, _ := ctx.fuzzMatch.fullName(&f.common, ft.Name)
		if !matched {
			continue
		}
		if target != nil {
			fmt.Fprintln(os.Stderr, "testing: warning: -fuzz matches more than one target, won't fuzz")
			return false, true
		}
		target = ft
		f.name = testName
	}
	if target == nil {
		return false, true
	}
	if Verbose() {
		f.chatty = newChattyPrinter(f.w)
		if !*isFuzzWorker {
			f.chatty.Updatef(f.name, "--- FUZZ: %s\n", f.name)
		}
	}
	go f.runTarget(target.Fn)
	<-f.signal
	return f.ran, !f.failed
}
