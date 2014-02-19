// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader_test

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"sort"
	"strings"
	"testing"

	"code.google.com/p/go.tools/go/loader"
)

func loadFromArgs(args []string) (prog *loader.Program, rest []string, err error) {
	conf := &loader.Config{}
	rest, err = conf.FromArgs(args, true)
	if err == nil {
		prog, err = conf.Load()
	}
	return
}

func TestLoadFromArgs(t *testing.T) {
	// Failed load: bad first import path causes parsePackageFiles to fail.
	args := []string{"nosuchpkg", "errors"}
	if _, _, err := loadFromArgs(args); err == nil {
		t.Errorf("loadFromArgs(%q) succeeded, want failure", args)
	} else {
		// cannot find package: ok.
	}

	// Failed load: bad second import path proceeds to doImport0, which fails.
	args = []string{"errors", "nosuchpkg"}
	if _, _, err := loadFromArgs(args); err == nil {
		t.Errorf("loadFromArgs(%q) succeeded, want failure", args)
	} else {
		// cannot find package: ok
	}

	// Successful load.
	args = []string{"fmt", "errors", "--", "surplus"}
	prog, rest, err := loadFromArgs(args)
	if err != nil {
		t.Fatalf("loadFromArgs(%q) failed: %s", args, err)
	}
	if got, want := fmt.Sprint(rest), "[surplus]"; got != want {
		t.Errorf("loadFromArgs(%q) rest: got %s, want %s", args, got, want)
	}
	// Check list of Created packages.
	var pkgnames []string
	for _, info := range prog.Created {
		pkgnames = append(pkgnames, info.Pkg.Path())
	}
	// Only the first import path (currently) contributes tests.
	if got, want := fmt.Sprint(pkgnames), "[fmt_test]"; got != want {
		t.Errorf("Created: got %s, want %s", got, want)
	}

	// Check set of Imported packages.
	pkgnames = nil
	for path := range prog.Imported {
		pkgnames = append(pkgnames, path)
	}
	sort.Strings(pkgnames)
	// Only the first import path (currently) contributes tests.
	if got, want := fmt.Sprint(pkgnames), "[errors fmt]"; got != want {
		t.Errorf("Loaded: got %s, want %s", got, want)
	}

	// Check set of transitive packages.
	// There are >30 and the set may grow over time, so only check a few.
	all := map[string]struct{}{}
	for _, info := range prog.AllPackages {
		all[info.Pkg.Path()] = struct{}{}
	}
	want := []string{"strings", "time", "runtime", "testing", "unicode"}
	for _, w := range want {
		if _, ok := all[w]; !ok {
			t.Errorf("AllPackages: want element %s, got set %v", w, all)
		}
	}
}

func TestLoadFromArgsSource(t *testing.T) {
	// mixture of *.go/non-go.
	args := []string{"testdata/a.go", "fmt"}
	prog, _, err := loadFromArgs(args)
	if err == nil {
		t.Errorf("loadFromArgs(%q) succeeded, want failure", args)
	} else {
		// "named files must be .go files: fmt": ok
	}

	// successful load
	args = []string{"testdata/a.go", "testdata/b.go"}
	prog, _, err = loadFromArgs(args)
	if err != nil {
		t.Errorf("loadFromArgs(%q) failed: %s", args, err)
		return
	}
	if len(prog.Created) != 1 || prog.Created[0].Pkg.Path() != "P" {
		t.Errorf("loadFromArgs(%q): got %v, want [P]", prog.Created)
	}
}

func TestTransitivelyErrorFreeFlag(t *testing.T) {
	conf := loader.Config{
		AllowTypeErrors: true,
		SourceImports:   true,
	}
	conf.Import("a")

	// Fake the following packages:
	//
	// a --> b --> c!   c has a TypeError
	//   \              d and e are transitively error free.
	//    e --> d

	// Temporary hack until we expose a principled PackageLocator.
	pfn := loader.PackageLocatorFunc()
	saved := *pfn
	*pfn = func(_ *build.Context, fset *token.FileSet, path string, which string) (files []*ast.File, err error) {
		if !strings.Contains(which, "g") {
			return nil, nil // no test/xtest files
		}
		var contents string
		switch path {
		case "a":
			contents = `package a; import (_ "b"; _ "e")`
		case "b":
			contents = `package b; import _ "c"`
		case "c":
			contents = `package c; func f() { _ = int(false) }` // type error within function body
		case "d":
			contents = `package d;`
		case "e":
			contents = `package e; import _ "d"`
		default:
			return nil, fmt.Errorf("no such package %q", path)
		}
		f, err := conf.ParseFile(fmt.Sprintf("%s/x.go", path), contents, 0)
		return []*ast.File{f}, err
	}
	defer func() { *pfn = saved }()

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed: %s", err)
	}
	if prog == nil {
		t.Fatalf("Load returnd nil *Program")
	}

	for pkg, info := range prog.AllPackages {
		var wantErr, wantTEF bool
		switch pkg.Path() {
		case "a", "b":
		case "c":
			wantErr = true
		case "d", "e":
			wantTEF = true
		default:
			t.Errorf("unexpected package: %q", pkg.Path())
			continue
		}

		if (info.TypeError != nil) != wantErr {
			if wantErr {
				t.Errorf("Package %q.TypeError = nil, want error", pkg.Path())
			} else {
				t.Errorf("Package %q has unexpected TypeError: %s",
					pkg.Path(), info.TypeError)
			}
		}

		if info.TransitivelyErrorFree != wantTEF {
			t.Errorf("Package %q.TransitivelyErrorFree=%t, want %t",
				info.TransitivelyErrorFree, wantTEF)
		}
	}
}
