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
	"io"
	"regexp"
	"runtime/pprof"
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

func (TestDeps) WriteHeapProfile(w io.Writer) error {
	return pprof.WriteHeapProfile(w)
}

func (TestDeps) WriteProfileTo(name string, w io.Writer, debug int) error {
	return pprof.Lookup(name).WriteTo(w, debug)
}

// ImportPath is the import path of the testing binary, set by the generated main function.
var ImportPath string

func (TestDeps) ImportPath() string {
	return ImportPath
}
