// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt"
	"runtime"
	"slices"
	"strings"
	"time"
)

type InternalExample struct {
	Name      string
	F         func()
	Output    string
	Unordered bool
}

// RunExamples is an internal function but exported because it is cross-package;
// it is part of the implementation of the "go test" command.
func RunExamples(matchString func(pat, str string) (bool, error), examples []InternalExample) (ok bool) {
	_, ok = runExamples(matchString, examples)
	return ok
}

func runExamples(matchString func(pat, str string) (bool, error), examples []InternalExample) (ran, ok bool) {
	ok = true

	m := newMatcher(matchString, *match, "-test.run", *skip)

	var eg InternalExample
	for _, eg = range examples {
		_, matched, _ := m.fullName(nil, eg.Name)
		if !matched {
			continue
		}
		ran = true
		if !runExample(eg) {
			ok = false
		}
	}

	return ran, ok
}

// processRunResult computes a summary and status of the result of running an example test.
// stdout is the captured output from stdout of the test.
// recovered is the result of invoking recover after running the test, in case it panicked.
//
// If stdout doesn't match the expected output or if recovered is non-nil, it'll print the cause of failure to stdout.
// If the test is chatty/verbose, it'll print a success message to stdout.
// If recovered is non-nil, it'll panic with that value.
// If the test panicked with nil, or invoked runtime.Goexit, it'll be
// made to fail and panic with errNilPanicOrGoexit
func (eg *InternalExample) processRunResult(stdout string, timeSpent time.Duration, finished bool, recovered any) (passed bool) {
	passed = true
	dstr := fmtDuration(timeSpent)
	var fail string
	got := strings.TrimSpace(stdout)
	want := strings.TrimSpace(eg.Output)
	if runtime.GOOS == "windows" {
		got = strings.ReplaceAll(got, "\r\n", "\n")
		want = strings.ReplaceAll(want, "\r\n", "\n")
	}
	if eg.Unordered {
		gotLines := slices.Sorted(strings.SplitSeq(got, "\n"))
		wantLines := slices.Sorted(strings.SplitSeq(want, "\n"))
		if !slices.Equal(gotLines, wantLines) && recovered == nil {
			fail = fmt.Sprintf("got:\n%s\nwant (unordered):\n%s\n", stdout, eg.Output)
		}
	} else {
		if got != want && recovered == nil {
			fail = fmt.Sprintf("got:\n%s\nwant:\n%s\n", got, want)
		}
	}
	if fail != "" || !finished || recovered != nil {
		fmt.Printf("%s--- FAIL: %s (%s)\n%s", chatty.prefix(), eg.Name, dstr, fail)
		passed = false
	} else if chatty.on {
		fmt.Printf("%s--- PASS: %s (%s)\n", chatty.prefix(), eg.Name, dstr)
	}

	if chatty.on && chatty.json {
		fmt.Printf("%s=== NAME   %s\n", chatty.prefix(), "")
	}

	if recovered != nil {
		// Propagate the previously recovered result, by panicking.
		panic(recovered)
	} else if !finished {
		panic(errNilPanicOrGoexit)
	}

	return
}
