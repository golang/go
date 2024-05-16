// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for test coverage with redesigned coverage implementation.

package testing

import (
	"fmt"
	"internal/goexperiment"
	"os"
	_ "unsafe" // for linkname
)

// cover2 variable stores the current coverage mode and a
// tear-down function to be called at the end of the testing run.
var cover2 struct {
	mode        string
	tearDown    func(coverprofile string, gocoverdir string) (string, error)
	snapshotcov func() float64
}

// registerCover2 is injected in testmain.
//go:linkname registerCover2

// registerCover2 is invoked during "go test -cover" runs by the test harness
// code in _testmain.go; it is used to record a 'tear down' function
// (to be called when the test is complete) and the coverage mode.
func registerCover2(mode string, tearDown func(coverprofile string, gocoverdir string) (string, error), snapcov func() float64) {
	cover2.mode = mode
	cover2.tearDown = tearDown
	cover2.snapshotcov = snapcov
}

// coverReport2 invokes a callback in _testmain.go that will
// emit coverage data at the point where test execution is complete,
// for "go test -cover" runs.
func coverReport2() {
	if !goexperiment.CoverageRedesign {
		panic("unexpected")
	}
	if errmsg, err := cover2.tearDown(*coverProfile, *gocoverdir); err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", errmsg, err)
		os.Exit(2)
	}
}

// testGoCoverDir is used in runtime/coverage tests.
//go:linkname testGoCoverDir

// testGoCoverDir returns the value passed to the -test.gocoverdir
// flag by the Go command, if goexperiment.CoverageRedesign is
// in effect.
func testGoCoverDir() string {
	return *gocoverdir
}

// coverage2 returns a rough "coverage percentage so far"
// number to support the testing.Coverage() function.
func coverage2() float64 {
	if cover2.mode == "" {
		return 0.0
	}
	return cover2.snapshotcov()
}
