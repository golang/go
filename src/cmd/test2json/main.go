// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test2json converts go test output to a machine-readable JSON stream.
//
// Usage:
//
//	go test ... | go tool test2json [-p pkg] [-t]
//	./test.out 2>&1 | go tool test2json [-p pkg] [-t]
//
// Test2json expects to read go test output from standard input.
// It writes a corresponding stream of JSON events to standard output.
// There is no unnecessary input or output buffering, so that
// the JSON stream can be read for “live updates” of test status.
//
// The -p flag sets the package reported in each test event.
//
// The -t flag requests that time stamps be added to each test event.
//
// Output Format
//
// The JSON stream is a newline-separated sequence of TestEvent objects
// corresponding to the Go struct:
//
//	type TestEvent struct {
//		Time    time.Time // encodes as an RFC3339-format string
//		Event   string
//		Package string
//		Test    string
//		Elapsed float64 // seconds
//		Output  string
//	}
//
// The Time field holds the time the event happened.
// It is conventionally omitted for cached test results.
//
// The Event field is one of a fixed set of event descriptions:
//
//	run    - the test has started running
//	pause  - the test has been paused
//	cont   - the test has continued running
//	pass   - the test passed
//	fail   - the test failed
//	output - the test printed output
//
// The Package field, if present, specifies the package being tested.
// When the go command runs parallel tests in -json mode, events from
// different tests are interlaced; the Package field allows readers to
// separate them.
//
// The Test field, if present, specifies the test or example, or benchmark
// function that caused the event. Events for the overall package test
// do not set Test.
//
// The Elapsed field is set for "pass" and "fail" events. It gives the time
// elapsed for the specific test or the overall package test that passed or failed.
//
// The Output field is set for Event == "output" and is a portion of the test's output
// (standard output and standard error merged together). The output is
// unmodified except that invalid UTF-8 output from a test is coerced
// into valid UTF-8 by use of replacement characters. With that one exception,
// the concatenation of the Output fields of all output events is the exact
// output of the test execution.
//
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"cmd/internal/test2json"
)

var (
	flagP = flag.String("p", "", "report `pkg` as the package being tested in each event")
	flagT = flag.Bool("t", false, "include timestamps in events")
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go test ... | go tool test2json [-p pkg] [-t]\n")
	os.Exit(2)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() > 0 {
		usage()
	}

	var mode test2json.Mode
	if *flagT {
		mode |= test2json.Timestamp
	}
	c := test2json.NewConverter(os.Stdout, *flagP, mode)
	defer c.Close()
	io.Copy(c, os.Stdin)
}
