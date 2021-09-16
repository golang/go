// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func mkGenState() *genstate {

	return &genstate{
		GenConfig: GenConfig{
			Tag:              "gen",
			OutDir:           "/tmp",
			NumTestPackages:  1,
			NumTestFunctions: 10,
		},
		ipref:       "foo/",
		derefFuncs:  make(map[string]string),
		assignFuncs: make(map[string]string),
		allocFuncs:  make(map[string]string),
		globVars:    make(map[string]string),
	}
}

func TestBasic(t *testing.T) {
	checkTunables(tunables)
	s := mkGenState()
	for i := 0; i < 1000; i++ {
		s.wr = NewWrapRand(int64(i), RandCtlChecks|RandCtlPanic)
		fp := s.GenFunc(i, i)
		var buf bytes.Buffer
		var b *bytes.Buffer = &buf
		wr := NewWrapRand(int64(i), RandCtlChecks|RandCtlPanic)
		s.wr = wr
		s.emitCaller(fp, b, i)
		s.wr = NewWrapRand(int64(i), RandCtlChecks|RandCtlPanic)
		s.emitChecker(fp, b, i, true)
		wr.Check(s.wr)
	}
	if s.errs != 0 {
		t.Errorf("%d errors during Generate", s.errs)
	}
}

func TestMoreComplicated(t *testing.T) {
	saveit := tunables
	defer func() { tunables = saveit }()

	checkTunables(tunables)
	s := mkGenState()
	for i := 0; i < 10000; i++ {
		s.wr = NewWrapRand(int64(i), RandCtlChecks|RandCtlPanic)
		fp := s.GenFunc(i, i)
		var buf bytes.Buffer
		var b *bytes.Buffer = &buf
		wr := NewWrapRand(int64(i), RandCtlChecks|RandCtlPanic)
		s.wr = wr
		s.emitCaller(fp, b, i)
		verb(1, "finished iter %d caller", i)
		s.wr = NewWrapRand(int64(i), RandCtlChecks|RandCtlPanic)
		s.emitChecker(fp, b, i, true)
		verb(1, "finished iter %d checker", i)
		wr.Check(s.wr)
		if s.errs != 0 {
			t.Errorf("%d errors during Generate iter %d", s.errs, i)
		}
	}
}

func TestIsBuildable(t *testing.T) {
	testenv.NeedsTool(t, "go")
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
	}

	td := t.TempDir()
	verb(1, "generating into temp dir %s", td)
	checkTunables(tunables)
	pack := filepath.Base(td)
	s := GenConfig{
		Tag:              "x",
		OutDir:           td,
		PkgPath:          pack,
		NumTestFunctions: 10,
		NumTestPackages:  10,
		MaxFail:          10,
		RandCtl:          RandCtlChecks | RandCtlPanic,
	}
	errs := Generate(s)
	if errs != 0 {
		t.Errorf("%d errors during Generate", errs)
	}

	verb(1, "building %s\n", td)

	cmd := exec.Command("go", "run", ".")
	cmd.Dir = td
	coutput, cerr := cmd.CombinedOutput()
	if cerr != nil {
		t.Errorf("go build command failed: %s\n", string(coutput))
	}
	verb(1, "output is: %s\n", string(coutput))
}

// TestExhaustive does a series of code genreation runs, starting with
// (relatively) simple code and then getting progressively more
// complex (more params, deeper structs, turning on additional
// features such as address-taken vars and reflect testing). The
// intent here is mainly to insure that the tester still works if you
// turn things on and off, e.g. that each feature is separately
// controllable and not linked to other things.
func TestExhaustive(t *testing.T) {
	testenv.NeedsTool(t, "go")
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
	}

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	td := t.TempDir()
	verb(1, "generating into temp dir %s", td)

	scenarios := []struct {
		name     string
		adjuster func()
	}{
		{
			"minimal",
			func() {
				tunables.nParmRange = 3
				tunables.nReturnRange = 3
				tunables.structDepth = 1
				tunables.recurPerc = 0
				tunables.methodPerc = 0
				tunables.doReflectCall = false
				tunables.doDefer = false
				tunables.takeAddress = false
				tunables.doFuncCallValues = false
				tunables.doSkipCompare = false
				checkTunables(tunables)
			},
		},
		{
			"moreparms",
			func() {
				tunables.nParmRange = 15
				tunables.nReturnRange = 7
				tunables.structDepth = 3
				checkTunables(tunables)
			},
		},
		{
			"addrecur",
			func() {
				tunables.recurPerc = 20
				checkTunables(tunables)
			},
		},
		{
			"addmethod",
			func() {
				tunables.methodPerc = 25
				tunables.pointerMethodCallPerc = 30
				checkTunables(tunables)
			},
		},
		{
			"addtakeaddr",
			func() {
				tunables.takeAddress = true
				tunables.takenFraction = 20
				checkTunables(tunables)
			},
		},
		{
			"addreflect",
			func() {
				tunables.doReflectCall = true
				checkTunables(tunables)
			},
		},
		{
			"adddefer",
			func() {
				tunables.doDefer = true
				checkTunables(tunables)
			},
		},
		{
			"addfuncval",
			func() {
				tunables.doFuncCallValues = true
				checkTunables(tunables)
			},
		},
		{
			"addfuncval",
			func() {
				tunables.doSkipCompare = true
				checkTunables(tunables)
			},
		},
	}

	// Loop over scenarios and make sure each one works properly.
	for i, s := range scenarios {
		t.Logf("running %s\n", s.name)
		s.adjuster()
		os.RemoveAll(td)
		pack := filepath.Base(td)
		c := GenConfig{
			Tag:              "x",
			OutDir:           td,
			PkgPath:          pack,
			NumTestFunctions: 10,
			NumTestPackages:  10,
			Seed:             int64(i + 9),
			MaxFail:          10,
			RandCtl:          RandCtlChecks | RandCtlPanic,
		}
		errs := Generate(c)
		if errs != 0 {
			t.Errorf("%d errors during scenarios %q Generate", errs, s.name)
		}
		cmd := exec.Command("go", "run", ".")
		cmd.Dir = td
		coutput, cerr := cmd.CombinedOutput()
		if cerr != nil {
			t.Fatalf("run failed for scenario %q:  %s\n", s.name, string(coutput))
		}
		verb(1, "output is: %s\n", string(coutput))
	}
}

func TestEmitBadBuildFailure(t *testing.T) {
	testenv.NeedsTool(t, "go")
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
	}

	td := t.TempDir()
	verb(1, "generating into temp dir %s", td)

	checkTunables(tunables)
	pack := filepath.Base(td)
	s := GenConfig{
		Tag:              "x",
		OutDir:           td,
		PkgPath:          pack,
		NumTestFunctions: 10,
		NumTestPackages:  10,
		MaxFail:          10,
		RandCtl:          RandCtlChecks | RandCtlPanic,
		EmitBad:          1,
	}
	errs := Generate(s)
	if errs != 0 {
		t.Errorf("%d errors during Generate", errs)
	}

	cmd := exec.Command("go", "build", ".")
	cmd.Dir = td
	coutput, cerr := cmd.CombinedOutput()
	if cerr == nil {
		t.Errorf("go build command passed, expected failure. output: %s\n", string(coutput))
	}
}

func TestEmitBadRunFailure(t *testing.T) {
	testenv.NeedsTool(t, "go")
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
	}

	td := t.TempDir()
	verb(1, "generating into temp dir %s", td)

	checkTunables(tunables)
	pack := filepath.Base(td)
	s := GenConfig{
		Tag:              "x",
		OutDir:           td,
		PkgPath:          pack,
		NumTestFunctions: 10,
		NumTestPackages:  10,
		MaxFail:          10,
		RandCtl:          RandCtlChecks | RandCtlPanic,
		EmitBad:          2,
	}
	errs := Generate(s)
	if errs != 0 {
		t.Errorf("%d errors during Generate", errs)
	}

	// build
	cmd := exec.Command("go", "build", ".")
	cmd.Dir = td
	coutput, cerr := cmd.CombinedOutput()
	if cerr != nil {
		t.Fatalf("build failed: %s\n", string(coutput))
	}

	// run
	cmd = exec.Command("./" + pack)
	cmd.Dir = td
	coutput, cerr = cmd.CombinedOutput()
	if cerr == nil {
		t.Fatalf("run passed, expected failure -- run output: %s", string(coutput))
	}
}
