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

// FuzzResult contains the results of a fuzz run.
type FuzzResult struct {
	N       int           // The number of iterations.
	T       time.Duration // The total time taken.
	Crasher corpusEntry   // Crasher is the corpus entry that caused the crash
	Error   error         // Error is the error from the crash
}

func (r FuzzResult) String() string {
	s := ""
	if len(r.Error.Error()) != 0 {
		s = fmt.Sprintf("error: %s\ncrasher: %b", r.Error.Error(), r.Crasher)
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
// engine to generate or mutate inputs. If -fuzz matches a given fuzz target,
// then such test will be skipped and run later during fuzzing.
func runFuzzTargets(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	ran, ok = true, true
	if len(fuzzTargets) == 0 {
		return false, ok
	}
	for _, ft := range fuzzTargets {
		ctx := &fuzzContext{runMatch: newMatcher(matchString, *match, "-test.run")}
		f := &F{
			common: common{
				signal:  make(chan bool),
				barrier: make(chan bool),
				w:       os.Stdout,
				name:    ft.Name,
			},
			context: ctx,
		}
		testName, matched, _ := ctx.runMatch.fullName(&f.common, f.name)
		if !matched {
			continue
		}
		if *matchFuzz != "" {
			ctx.fuzzMatch = newMatcher(matchString, *matchFuzz, "-test.fuzz")
			if _, doFuzz, partial := ctx.fuzzMatch.fullName(&f.common, f.name); doFuzz && !partial {
				continue // this will be run later when fuzzed
			}
		}
		if Verbose() {
			f.chatty = newChattyPrinter(f.w)
		}
		if f.chatty != nil {
			f.chatty.Updatef(f.name, "=== RUN  %s\n", testName)
		}
	}
	return ran, ok
}

// RunFuzzing is an internal function but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
func RunFuzzing(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ok bool) {
	_, ok = runFuzzing(matchString, fuzzTargets)
	return ok
}

// runFuzzing runs the fuzz target matching the pattern for -fuzz. Only one such
// fuzz target must match. This will run the f.Fuzz function for each seed
// corpus and will run the fuzzing engine to generate and mutate new inputs
// against f.Fuzz.
func runFuzzing(matchString func(pat, str string) (bool, error), fuzzTargets []InternalFuzzTarget) (ran, ok bool) {
	ran, ok = true, true
	if len(fuzzTargets) == 0 {
		return false, ok
	}
	ctx := &fuzzContext{
		fuzzMatch: newMatcher(matchString, *matchFuzz, "-test.fuzz"),
	}
	if *matchFuzz == "" {
		return false, true
	}
	f := &F{
		common: common{
			signal:  make(chan bool),
			barrier: make(chan bool),
			w:       os.Stdout,
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
				fmt.Fprintf(f.w, "testing: warning: -fuzz matched more than one target, won't run\n")
				return false, ok
			}
			f.name = testName
		}
	}
	if Verbose() {
		f.chatty = newChattyPrinter(f.w)
	}
	if f.chatty != nil {
		f.chatty.Updatef(f.name, "--- FUZZ  %s\n", f.name)
	}
	return ran, ok
}

// Fuzz runs a single fuzz target. It is useful for creating
// custom fuzz targets that do not use the "go test" command.
//
// If fn depends on testing flags, then Init must be used to register
// those flags before calling Fuzz and before calling flag.Parse.
func Fuzz(fn func(f *F)) FuzzResult {
	f := &F{
		common: common{
			signal: make(chan bool),
			w:      discard{},
		},
		fuzzFunc: fn,
	}
	// TODO(katiehockman): run the test
	return f.result
}
