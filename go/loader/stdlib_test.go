// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package loader_test

// This file enumerates all packages beneath $GOROOT, loads them, plus
// their external tests if any, runs the type checker on them, and
// prints some summary information.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"go/types"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
)

func TestStdlib(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skipf("incomplete std lib on %s", runtime.GOOS)
	}
	if testing.Short() {
		t.Skip("skipping in short mode; uses tons of memory (golang.org/issue/14113)")
	}

	runtime.GC()
	t0 := time.Now()
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	alloc := memstats.Alloc

	// Load, parse and type-check the program.
	ctxt := build.Default // copy
	ctxt.GOPATH = ""      // disable GOPATH
	conf := loader.Config{Build: &ctxt}
	for _, path := range buildutil.AllPackages(conf.Build) {
		conf.ImportWithTests(path)
	}

	prog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	t1 := time.Now()
	runtime.GC()
	runtime.ReadMemStats(&memstats)

	numPkgs := len(prog.AllPackages)
	if want := 205; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Dump package members.
	if false {
		for pkg := range prog.AllPackages {
			fmt.Printf("Package %s:\n", pkg.Path())
			scope := pkg.Scope()
			qualifier := types.RelativeTo(pkg)
			for _, name := range scope.Names() {
				if ast.IsExported(name) {
					fmt.Printf("\t%s\n", types.ObjectString(scope.Lookup(name), qualifier))
				}
			}
			fmt.Println()
		}
	}

	// Check that Test functions for io/ioutil, regexp and
	// compress/bzip2 are all simultaneously present.
	// (The apparent cycle formed when augmenting all three of
	// these packages by their tests was the original motivation
	// for reporting b/7114.)
	//
	// compress/bzip2.TestBitReader in bzip2_test.go    imports io/ioutil
	// io/ioutil.TestTempFile       in tempfile_test.go imports regexp
	// regexp.TestRE2Search         in exec_test.go     imports compress/bzip2
	for _, test := range []struct{ pkg, fn string }{
		{"io/ioutil", "TestTempFile"},
		{"regexp", "TestRE2Search"},
		{"compress/bzip2", "TestBitReader"},
	} {
		info := prog.Imported[test.pkg]
		if info == nil {
			t.Errorf("failed to load package %q", test.pkg)
			continue
		}
		obj, _ := info.Pkg.Scope().Lookup(test.fn).(*types.Func)
		if obj == nil {
			t.Errorf("package %q has no func %q", test.pkg, test.fn)
			continue
		}
	}

	// Dump some statistics.

	// determine line count
	var lineCount int
	prog.Fset.Iterate(func(f *token.File) bool {
		lineCount += f.LineCount()
		return true
	})

	t.Log("GOMAXPROCS:           ", runtime.GOMAXPROCS(0))
	t.Log("#Source lines:        ", lineCount)
	t.Log("Load/parse/typecheck: ", t1.Sub(t0))
	t.Log("#MB:                  ", int64(memstats.Alloc-alloc)/1000000)
}

func TestCgoOption(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode; uses tons of memory (golang.org/issue/14113)")
	}
	switch runtime.GOOS {
	// On these systems, the net and os/user packages don't use cgo
	// or the std library is incomplete (Android).
	case "android", "plan9", "solaris", "windows":
		t.Skipf("no cgo or incomplete std lib on %s", runtime.GOOS)
	}
	// In nocgo builds (e.g. linux-amd64-nocgo),
	// there is no "runtime/cgo" package,
	// so cgo-generated Go files will have a failing import.
	if !build.Default.CgoEnabled {
		return
	}
	// Test that we can load cgo-using packages with
	// CGO_ENABLED=[01], which causes go/build to select pure
	// Go/native implementations, respectively, based on build
	// tags.
	//
	// Each entry specifies a package-level object and the generic
	// file expected to define it when cgo is disabled.
	// When cgo is enabled, the exact file is not specified (since
	// it varies by platform), but must differ from the generic one.
	//
	// The test also loads the actual file to verify that the
	// object is indeed defined at that location.
	for _, test := range []struct {
		pkg, name, genericFile string
	}{
		{"net", "cgoLookupHost", "cgo_stub.go"},
		{"os/user", "lookupId", "lookup_stubs.go"},
	} {
		ctxt := build.Default
		for _, ctxt.CgoEnabled = range []bool{false, true} {
			conf := loader.Config{Build: &ctxt}
			conf.Import(test.pkg)
			prog, err := conf.Load()
			if err != nil {
				t.Errorf("Load failed: %v", err)
				continue
			}
			info := prog.Imported[test.pkg]
			if info == nil {
				t.Errorf("package %s not found", test.pkg)
				continue
			}
			obj := info.Pkg.Scope().Lookup(test.name)
			if obj == nil {
				t.Errorf("no object %s.%s", test.pkg, test.name)
				continue
			}
			posn := prog.Fset.Position(obj.Pos())
			t.Logf("%s: %s (CgoEnabled=%t)", posn, obj, ctxt.CgoEnabled)

			gotFile := filepath.Base(posn.Filename)
			filesMatch := gotFile == test.genericFile

			if ctxt.CgoEnabled && filesMatch {
				t.Errorf("CGO_ENABLED=1: %s found in %s, want native file",
					obj, gotFile)
			} else if !ctxt.CgoEnabled && !filesMatch {
				t.Errorf("CGO_ENABLED=0: %s found in %s, want %s",
					obj, gotFile, test.genericFile)
			}

			// Load the file and check the object is declared at the right place.
			b, err := ioutil.ReadFile(posn.Filename)
			if err != nil {
				t.Errorf("can't read %s: %s", posn.Filename, err)
				continue
			}
			line := string(bytes.Split(b, []byte("\n"))[posn.Line-1])
			ident := line[posn.Column-1:]
			if !strings.HasPrefix(ident, test.name) {
				t.Errorf("%s: %s not declared here (looking at %q)", posn, obj, ident)
			}
		}
	}
}
