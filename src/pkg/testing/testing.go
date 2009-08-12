// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The testing package provides support for automated testing of Go packages.
// It is intended to be used in concert with the ``gotest'' utility, which automates
// execution of any function of the form
//     func TestXxx(*testing.T)
// where Xxx can by any alphanumeric string (but the first letter must not be in
// [a-z]) and serves to identify the test routine.
// These TestXxx routines should be declared within the package they are testing.
package testing

import (
	"flag";
	"fmt";
	"os";
	"runtime";
)

// Report as tests are run; default is silent for success.
var chatty = flag.Bool("v", false, "verbose: print additional output")
var match = flag.String("match", "", "regular expression to select tests to run")


// Insert final newline if needed and tabs after internal newlines.
func tabify(s string) string {
	n := len(s);
	if n > 0 && s[n-1] != '\n' {
		s += "\n";
		n++;
	}
	for i := 0; i < n - 1; i++ {	// -1 to avoid final newline
		if s[i] == '\n' {
			return s[0:i+1] + "\t" + tabify(s[i+1:n]);
		}
	}
	return s
}

// T is a type passed to Test functions to manage test state and support formatted test logs.
// Logs are accumulated during execution and dumped to standard error when done.
type T struct {
	errors	string;
	failed	bool;
	ch	chan *T;
}

// Fail marks the Test function as having failed but continues execution.
func (t *T) Fail() {
	t.failed = true
}

// Failed returns whether the Test function has failed.
func (t *T) Failed() bool {
	return t.failed
}

// FailNow marks the Test function as having failed and stops its execution.
// Execution will continue at the next Test.
func (t *T) FailNow() {
	t.Fail();
	t.ch <- t;
	runtime.Goexit();
}

// Log formats its arguments using default formatting, analogous to Print(),
// and records the text in the error log.
func (t *T) Log(args ...) {
	t.errors += "\t" + tabify(fmt.Sprintln(args));
}

// Log formats its arguments according to the format, analogous to Printf(),
// and records the text in the error log.
func (t *T) Logf(format string, args ...) {
	t.errors += "\t" + tabify(fmt.Sprintf(format, args));
}

// Error is equivalent to Log() followed by Fail().
func (t *T) Error(args ...) {
	t.Log(args);
	t.Fail();
}

// Errorf is equivalent to Logf() followed by Fail().
func (t *T) Errorf(format string, args ...) {
	t.Logf(format, args);
	t.Fail();
}

// Fatal is equivalent to Log() followed by FailNow().
func (t *T) Fatal(args ...) {
	t.Log(args);
	t.FailNow();
}

// Fatalf is equivalent to Logf() followed by FailNow().
func (t *T) Fatalf(format string, args ...) {
	t.Logf(format, args);
	t.FailNow();
}

// An internal type but exported because it is cross-package; part of the implementation
// of gotest.
type Test struct {
	Name string;
	F func(*T);
}

func tRunner(t *T, test *Test) {
	test.F(t);
	t.ch <- t;
}

// An internal function but exported because it is cross-package; part of the implementation
// of gotest.
func Main(tests []Test) {
	flag.Parse();
	args := flag.Args();
	ok := true;
	if len(tests) == 0 {
		println("testing: warning: no tests to run");
	}
	re, err := CompileRegexp(*match);
	if err != "" {
		println("invalid regexp for -match:", err);
		os.Exit(1);
	}
	for i := 0; i < len(tests); i++ {
		if !re.MatchString(tests[i].Name) {
			continue;
		}
		if *chatty {
			println("=== RUN ", tests[i].Name);
		}
		t := new(T);
		t.ch = make(chan *T);
		go tRunner(t, &tests[i]);
		<-t.ch;
		if t.failed {
			println("--- FAIL:", tests[i].Name);
			print(t.errors);
			ok = false;
		} else if *chatty {
			println("--- PASS:", tests[i].Name);
			print(t.errors);
		}
	}
	if !ok {
		println("FAIL");
		os.Exit(1);
	}
	println("PASS");
}
