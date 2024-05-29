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
	"context"
	"internal/fuzz"
	"internal/testlog"
	"io"
	"os"
	"os/signal"
	"reflect"
	"regexp"
	"runtime/pprof"
	"strings"
	"sync"
	"time"
)

// Cover indicates whether coverage is enabled.
var Cover bool

// TestDeps is an implementation of the testing.testDeps interface,
// suitable for passing to [testing.MainStart].
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

// SetPanicOnExit tells the os package whether to panic on os.Exit.
func (TestDeps) SetPanicOnExit(v bool) {
	testlog.SetPanicOnExit(v)
}

func (TestDeps) CoordinateFuzzing(
	timeout time.Duration,
	limit int64,
	minimizeTimeout time.Duration,
	minimizeLimit int64,
	parallel int,
	seed []fuzz.CorpusEntry,
	types []reflect.Type,
	corpusDir,
	cacheDir string) (err error) {
	// Fuzzing may be interrupted with a timeout or if the user presses ^C.
	// In either case, we'll stop worker processes gracefully and save
	// crashers and interesting values.
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	err = fuzz.CoordinateFuzzing(ctx, fuzz.CoordinateFuzzingOpts{
		Log:             os.Stderr,
		Timeout:         timeout,
		Limit:           limit,
		MinimizeTimeout: minimizeTimeout,
		MinimizeLimit:   minimizeLimit,
		Parallel:        parallel,
		Seed:            seed,
		Types:           types,
		CorpusDir:       corpusDir,
		CacheDir:        cacheDir,
	})
	if err == ctx.Err() {
		return nil
	}
	return err
}

func (TestDeps) RunFuzzWorker(fn func(fuzz.CorpusEntry) error) error {
	// Worker processes may or may not receive a signal when the user presses ^C
	// On POSIX operating systems, a signal sent to a process group is delivered
	// to all processes in that group. This is not the case on Windows.
	// If the worker is interrupted, return quickly and without error.
	// If only the coordinator process is interrupted, it tells each worker
	// process to stop by closing its "fuzz_in" pipe.
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	err := fuzz.RunFuzzWorker(ctx, fn)
	if err == ctx.Err() {
		return nil
	}
	return err
}

func (TestDeps) ReadCorpus(dir string, types []reflect.Type) ([]fuzz.CorpusEntry, error) {
	return fuzz.ReadCorpus(dir, types)
}

func (TestDeps) CheckCorpus(vals []any, types []reflect.Type) error {
	return fuzz.CheckCorpus(vals, types)
}

func (TestDeps) ResetCoverage() {
	fuzz.ResetCoverage()
}

func (TestDeps) SnapshotCoverage() {
	fuzz.SnapshotCoverage()
}

var CoverMode string
var Covered string

// These variables below are set at runtime (via code in testmain) to point
// to the equivalent functions in package internal/coverage/cfile; doing
// things this way allows us to have tests import internal/coverage/cfile
// only when -cover is in effect (as opposed to importing for all tests).
var (
	CoverSnapshotFunc           func() float64
	CoverProcessTestDirFunc     func(dir string, cfile string, cm string, cpkg string, w io.Writer) error
	CoverMarkProfileEmittedFunc func(val bool)
)

func (TestDeps) InitRuntimeCoverage() (mode string, tearDown func(string, string) (string, error), snapcov func() float64) {
	if CoverMode == "" {
		return
	}
	return CoverMode, coverTearDown, CoverSnapshotFunc
}

func coverTearDown(coverprofile string, gocoverdir string) (string, error) {
	var err error
	if gocoverdir == "" {
		gocoverdir, err = os.MkdirTemp("", "gocoverdir")
		if err != nil {
			return "error setting GOCOVERDIR: bad os.MkdirTemp return", err
		}
		defer os.RemoveAll(gocoverdir)
	}
	CoverMarkProfileEmittedFunc(true)
	cmode := CoverMode
	if err := CoverProcessTestDirFunc(gocoverdir, coverprofile, cmode, Covered, os.Stdout); err != nil {
		return "error generating coverage report", err
	}
	return "", nil
}
