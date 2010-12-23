// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The testing package provides support for automated testing of Go packages.
// It is intended to be used in concert with the ``gotest'' utility, which automates
// execution of any function of the form
//     func TestXxx(*testing.T)
// where Xxx can be any alphanumeric string (but the first letter must not be in
// [a-z]) and serves to identify the test routine.
// These TestXxx routines should be declared within the package they are testing.
//
// Functions of the form
//     func BenchmarkXxx(*testing.B)
// are considered benchmarks, and are executed by gotest when the -benchmarks
// flag is provided.
//
// A sample benchmark function looks like this:
//     func BenchmarkHello(b *testing.B) {
//         for i := 0; i < b.N; i++ {
//             fmt.Sprintf("hello")
//         }
//     }
// The benchmark package will vary b.N until the benchmark function lasts
// long enough to be timed reliably.  The output
//     testing.BenchmarkHello	500000	      4076 ns/op
// means that the loop ran 500000 times at a speed of 4076 ns per loop.
//
// If a benchmark needs some expensive setup before running, the timer
// may be stopped:
//     func BenchmarkBigLen(b *testing.B) {
//         b.StopTimer()
//         big := NewBig()
//         b.StartTimer()
//         for i := 0; i < b.N; i++ {
//             big.Len()
//         }
//     }
package testing

import (
	"flag"
	"fmt"
	"os"
	"runtime"
)

// Report as tests are run; default is silent for success.
var chatty = flag.Bool("v", false, "verbose: print additional output")
var match = flag.String("match", "", "regular expression to select tests to run")


// Insert final newline if needed and tabs after internal newlines.
func tabify(s string) string {
	n := len(s)
	if n > 0 && s[n-1] != '\n' {
		s += "\n"
		n++
	}
	for i := 0; i < n-1; i++ { // -1 to avoid final newline
		if s[i] == '\n' {
			return s[0:i+1] + "\t" + tabify(s[i+1:n])
		}
	}
	return s
}

// T is a type passed to Test functions to manage test state and support formatted test logs.
// Logs are accumulated during execution and dumped to standard error when done.
type T struct {
	errors string
	failed bool
	ch     chan *T
}

// Fail marks the Test function as having failed but continues execution.
func (t *T) Fail() { t.failed = true }

// Failed returns whether the Test function has failed.
func (t *T) Failed() bool { return t.failed }

// FailNow marks the Test function as having failed and stops its execution.
// Execution will continue at the next Test.
func (t *T) FailNow() {
	t.Fail()
	t.ch <- t
	runtime.Goexit()
}

// Log formats its arguments using default formatting, analogous to Print(),
// and records the text in the error log.
func (t *T) Log(args ...interface{}) { t.errors += "\t" + tabify(fmt.Sprintln(args...)) }

// Log formats its arguments according to the format, analogous to Printf(),
// and records the text in the error log.
func (t *T) Logf(format string, args ...interface{}) {
	t.errors += "\t" + tabify(fmt.Sprintf(format, args...))
}

// Error is equivalent to Log() followed by Fail().
func (t *T) Error(args ...interface{}) {
	t.Log(args...)
	t.Fail()
}

// Errorf is equivalent to Logf() followed by Fail().
func (t *T) Errorf(format string, args ...interface{}) {
	t.Logf(format, args...)
	t.Fail()
}

// Fatal is equivalent to Log() followed by FailNow().
func (t *T) Fatal(args ...interface{}) {
	t.Log(args...)
	t.FailNow()
}

// Fatalf is equivalent to Logf() followed by FailNow().
func (t *T) Fatalf(format string, args ...interface{}) {
	t.Logf(format, args...)
	t.FailNow()
}

// An internal type but exported because it is cross-package; part of the implementation
// of gotest.
type InternalTest struct {
	Name string
	F    func(*T)
}

func tRunner(t *T, test *InternalTest) {
	test.F(t)
	t.ch <- t
}

// An internal function but exported because it is cross-package; part of the implementation
// of gotest.
func Main(matchString func(pat, str string) (bool, os.Error), tests []InternalTest) {
	flag.Parse()
	ok := true
	if len(tests) == 0 {
		println("testing: warning: no tests to run")
	}
	for i := 0; i < len(tests); i++ {
		matched, err := matchString(*match, tests[i].Name)
		if err != nil {
			println("invalid regexp for -match:", err.String())
			os.Exit(1)
		}
		if !matched {
			continue
		}
		if *chatty {
			println("=== RUN ", tests[i].Name)
		}
		t := new(T)
		t.ch = make(chan *T)
		go tRunner(t, &tests[i])
		<-t.ch
		if t.failed {
			println("--- FAIL:", tests[i].Name)
			print(t.errors)
			ok = false
		} else if *chatty {
			println("--- PASS:", tests[i].Name)
			print(t.errors)
		}
	}
	if !ok {
		println("FAIL")
		os.Exit(1)
	}
	println("PASS")
}
