// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"flag"
	"fmt"
	"os"
	"time"
)

func initFuzzFlags() {
	matchFuzz = flag.String("test.fuzz", "", "run the fuzz target matching `regexp`")
}

var matchFuzz *string

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
	corpus   []corpusEntry // corpus is the in-memory corpus
	result   FuzzResult    // result is the result of running the fuzz target
	fuzzFunc func(f *F)    // fuzzFunc is the function which makes up the fuzz target
	fuzz     bool          // fuzz indicates whether or not the fuzzing engine should run
}

// corpus corpusEntry
type corpusEntry struct {
	b []byte
}

// Add will add the arguments to the seed corpus for the fuzz target. This
// cannot be invoked after or within the Fuzz function. The args must match
// those in the Fuzz function.
func (f *F) Add(args ...interface{}) {
	return
}

// Fuzz runs the fuzz function, ff, for fuzz testing. It runs ff in a separate
// goroutine. Only one call to Fuzz is allowed per fuzz target, and any
// subsequent calls will panic. If ff fails for a set of arguments, those
// arguments will be added to the seed corpus.
func (f *F) Fuzz(ff interface{}) {
	return
}

func (f *F) report(name string) {
	if f.Failed() {
		fmt.Fprintf(f.w, "--- FAIL: %s\n%s\n", name, f.result.String())
	} else if f.chatty != nil {
		if f.Skipped() {
			f.chatty.Updatef(name, "SKIP\n")
		} else {
			f.chatty.Updatef(name, "PASS\n")
		}
	}
}

// run runs each fuzz target in its own goroutine with its own *F.
func (f *F) run(name string, fn func(f *F)) (ran, ok bool) {
	innerF := &F{
		common: common{
			signal: make(chan bool),
			name:   name,
			chatty: f.chatty,
			w:      f.w,
		},
		context: f.context,
	}
	if innerF.chatty != nil {
		innerF.chatty.Updatef(name, "=== RUN   %s\n", name)
	}
	go innerF.runTarget(fn)
	<-innerF.signal
	return innerF.ran, !innerF.failed
}

// runTarget runs the given target, handling panics and exits
// within the test, and reporting errors.
func (f *F) runTarget(fn func(f *F)) {
	defer func() {
		err := recover()
		// If the function has recovered but the test hasn't finished,
		// it is due to a nil panic or runtime.GoExit.
		if !f.finished && err == nil {
			err = errNilPanicOrGoexit
		}
		if err != nil {
			f.Fail()
			f.result = FuzzResult{Error: fmt.Errorf("%s", err)}
		}
		f.report(f.name)
		f.setRan()
		f.signal <- true // signal that the test has finished
	}()
	fn(f)
	f.finished = true
}

// FuzzResult contains the results of a fuzz run.
type FuzzResult struct {
	N       int           // The number of iterations.
	T       time.Duration // The total time taken.
	Crasher *corpusEntry  // Crasher is the corpus entry that caused the crash
	Error   error         // Error is the error from the crash
}

func (r FuzzResult) String() string {
	s := ""
	if r.Error == nil {
		return s
	}
	s = fmt.Sprintf("error: %s", r.Error.Error())
	if r.Crasher != nil {
		s += fmt.Sprintf("\ncrasher: %b", r.Crasher)
	}
	return s
}

// fuzzContext holds all fields that are common to all fuzz targets.
type fuzzContext struct {
	runMatch  *matcher
	fuzzMatch *matcher
}

// RunFuzzTargets is an internal function but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
func RunFuzzTargets(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ok bool) {
	_, ok = runFuzzTargets(matchString, fuzzTargets)
	return ok
}

// runFuzzTargets runs the fuzz targets matching the pattern for -run. This will
// only run the f.Fuzz function for each seed corpus without using the fuzzing
// engine to generate or mutate inputs.
func runFuzzTargets(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	ok = true
	if len(fuzzTargets) == 0 {
		return ran, ok
	}
	ctx := &fuzzContext{runMatch: newMatcher(matchString, *match, "-test.run")}
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
				ftRan, ftOk := f.run(ft.Name, ft.Fn)
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

// RunFuzzing is an internal function but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
func RunFuzzing(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ok bool) {
	_, ok = runFuzzing(matchString, fuzzTargets)
	return ok
}

// runFuzzing runs the fuzz target matching the pattern for -fuzz. Only one such
// fuzz target must match. This will run the fuzzing engine to generate and
// mutate new inputs against the f.Fuzz function.
func runFuzzing(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	if len(fuzzTargets) == 0 {
		return false, true
	}
	ctx := &fuzzContext{
		fuzzMatch: newMatcher(matchString, *matchFuzz, "-test.fuzz"),
	}
	if *matchFuzz == "" {
		return false, true
	}
	f := &F{
		common: common{
			signal: make(chan bool),
			w:      os.Stdout,
		},
		context: ctx,
		fuzz:    true,
	}
	var (
		ft    InternalFuzzTarget
		found int
	)
	for _, ft = range fuzzTargets {
		testName, matched, _ := ctx.fuzzMatch.fullName(&f.common, ft.Name)
		if matched {
			found++
			if found > 1 {
				fmt.Fprintln(os.Stderr, "testing: warning: -fuzz matches more than one target, won't fuzz")
				return false, true
			}
			f.name = testName
		}
	}
	if found == 0 {
		return false, true
	}
	if Verbose() {
		f.chatty = newChattyPrinter(f.w)
		f.chatty.Updatef(f.name, "--- FUZZ: %s\n", f.name)
	}
	go f.runTarget(ft.Fn)
	<-f.signal
	return f.ran, !f.failed
}

// Fuzz runs a single fuzz target. It is useful for creating
// custom fuzz targets that do not use the "go test" command.
//
// If fn depends on testing flags, then Init must be used to register
// those flags before calling Fuzz and before calling flag.Parse.
func Fuzz(fn func(f *F)) FuzzResult {
	f := &F{
		common: common{
			w: discard{},
		},
		fuzzFunc: fn,
	}
	// TODO(katiehockman): run the test
	return f.result
}
