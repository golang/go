// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testdeps provides access to dependencies needed by test execution.
//
// This package is imported by the generated main package, which passes
// TestDeps into testing.Main. This allows tests to use packages at run time
// without making those packages direct dependencies of package testing.
// Direct dependencies of package testing are harder to write tests for.
package testdeps

import (
	"bufio"
	"internal/testlog"
	"io"
	"regexp"
	"runtime/pprof"
	"strings"
	"sync"
)

// TestDeps is an implementation of the testing.testDeps interface,
// suitable for passing to testing.MainStart.
type TestDeps struct{}

var matchPat string
var matchRe *regexp.Regexp

func (TestDeps) MatchString(pat, str string) (result bool, err error) {
	if matchRe == nil || matchPat != pat {
		matchPat = pat
		matchRe, err = regexp.Compile(matchPat)
		if err != nil {
			return
		}
	}
	return matchRe.MatchString(str), nil
}

func (TestDeps) StartCPUProfile(w io.Writer) error {
	return pprof.StartCPUProfile(w)
}

func (TestDeps) StopCPUProfile() {
	pprof.StopCPUProfile()
}

func (TestDeps) WriteProfileTo(name string, w io.Writer, debug int) error {
	return pprof.Lookup(name).WriteTo(w, debug)
}

// ImportPath is the import path of the testing binary, set by the generated main function.
var ImportPath string

func (TestDeps) ImportPath() string {
	return ImportPath
}

// testLog implements testlog.Interface, logging actions by package os.
type testLog struct {
	mu  sync.Mutex
	w   *bufio.Writer
	set bool
}

func (l *testLog) Getenv(key string) {
	l.add("getenv", key)
}

func (l *testLog) Open(name string) {
	l.add("open", name)
}

func (l *testLog) Stat(name string) {
	l.add("stat", name)
}

func (l *testLog) Chdir(name string) {
	l.add("chdir", name)
}

// add adds the (op, name) pair to the test log.
func (l *testLog) add(op, name string) {
	if strings.Contains(name, "\n") || name == "" {
		return
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.w == nil {
		return
	}
	l.w.WriteString(op)
	l.w.WriteByte(' ')
	l.w.WriteString(name)
	l.w.WriteByte('\n')
}

var log testLog

func (TestDeps) StartTestLog(w io.Writer) {
	log.mu.Lock()
	log.w = bufio.NewWriter(w)
	if !log.set {
		// Tests that define TestMain and then run m.Run multiple times
		// will call StartTestLog/StopTestLog multiple times.
		// Checking log.set avoids calling testlog.SetLogger multiple times
		// (which will panic) and also avoids writing the header multiple times.
		log.set = true
		testlog.SetLogger(&log)
		log.w.WriteString("# test log\n") // known to cmd/go/internal/test/test.go
	}
	log.mu.Unlock()
}

func (TestDeps) StopTestLog() error {
	log.mu.Lock()
	defer log.mu.Unlock()
	err := log.w.Flush()
	log.w = nil
	return err
}

// SetPanicOnExit0 tells the os package whether to panic on os.Exit(0).
func (TestDeps) SetPanicOnExit0(v bool) {
	testlog.SetPanicOnExit0(v)
}
