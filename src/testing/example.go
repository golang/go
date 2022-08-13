// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt"
	"os"
	"sort"
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

	var eg InternalExample

	for _, eg = range examples {
		matched, err := matchString(*match, eg.Name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for -test.run: %s\n", err)
			os.Exit(1)
		}
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

func sortLines(output string) string {
	lines := strings.Split(output, "\n")
	sort.Strings(lines)
	return strings.Join(lines, "\n")
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
	if eg.Unordered {
		if sortLines(got) != sortLines(want) && recovered == nil {
			fail = fmt.Sprintf("got:\n%s\nwant (unordered):\n%s\n", stdout, eg.Output)
		}
	} else {
		if got != want && recovered == nil {
			fail = fmt.Sprintf("got:\n%s\nwant:\n%s\n", got, want)
		}
	}
	if fail != "" || !finished || recovered != nil {
		fmt.Printf("--- FAIL: %s (%s)\n%s", eg.Name, dstr, fail)
		passed = false
	} else if *chatty {
		fmt.Printf("--- PASS: %s (%s)\n", eg.Name, dstr)
	}

	if recovered != nil {
		// Propagate the previously recovered result, by panicking.
		panic(recovered)
	}
	if !finished && recovered == nil {
		panic(errNilPanicOrGoexit)
	}

	return
}
