// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader_test

import (
	"bytes"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

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
	// All import paths may contribute tests.
	if got, want := fmt.Sprint(pkgnames), "[fmt_test errors_test]"; got != want {
		t.Errorf("Created: got %s, want %s", got, want)
	}

	// Check set of Imported packages.
	pkgnames = nil
	for path := range prog.Imported {
		pkgnames = append(pkgnames, path)
	}
	sort.Strings(pkgnames)
	// All import paths may contribute tests.
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
		t.Fatalf("loadFromArgs(%q) failed: %s", args, err)
	}
	if len(prog.Created) != 1 {
		t.Errorf("loadFromArgs(%q): got %d items, want 1", len(prog.Created))
	}
	if len(prog.Created) > 0 {
		path := prog.Created[0].Pkg.Path()
		if path != "P" {
			t.Errorf("loadFromArgs(%q): got %v, want [P]", prog.Created, path)
		}
	}
}

type fakeFileInfo struct{}

func (fakeFileInfo) Name() string       { return "x.go" }
func (fakeFileInfo) Sys() interface{}   { return nil }
func (fakeFileInfo) ModTime() time.Time { return time.Time{} }
func (fakeFileInfo) IsDir() bool        { return false }
func (fakeFileInfo) Size() int64        { return 0 }
func (fakeFileInfo) Mode() os.FileMode  { return 0644 }

var justXgo = [1]os.FileInfo{fakeFileInfo{}} // ["x.go"]

func fakeContext(pkgs map[string]string) *build.Context {
	ctxt := build.Default // copy
	ctxt.GOROOT = "/go"
	ctxt.GOPATH = ""
	ctxt.IsDir = func(path string) bool { return true }
	ctxt.ReadDir = func(dir string) ([]os.FileInfo, error) { return justXgo[:], nil }
	ctxt.OpenFile = func(path string) (io.ReadCloser, error) {
		path = path[len("/go/src/"):]
		return ioutil.NopCloser(bytes.NewBufferString(pkgs[path[0:1]])), nil
	}
	return &ctxt
}

func TestTransitivelyErrorFreeFlag(t *testing.T) {
	// Create an minimal custom build.Context
	// that fakes the following packages:
	//
	// a --> b --> c!   c has an error
	//   \              d and e are transitively error-free.
	//    e --> d
	//
	// Each package [a-e] consists of one file, x.go.
	pkgs := map[string]string{
		"a": `package a; import (_ "b"; _ "e")`,
		"b": `package b; import _ "c"`,
		"c": `package c; func f() { _ = int(false) }`, // type error within function body
		"d": `package d;`,
		"e": `package e; import _ "d"`,
	}
	conf := loader.Config{
		AllowErrors:   true,
		SourceImports: true,
		Build:         fakeContext(pkgs),
	}
	conf.Import("a")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed: %s", err)
	}
	if prog == nil {
		t.Fatalf("Load returned nil *Program")
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

		if (info.Errors != nil) != wantErr {
			if wantErr {
				t.Errorf("Package %q.Error = nil, want error", pkg.Path())
			} else {
				t.Errorf("Package %q has unexpected Errors: %v",
					pkg.Path(), info.Errors)
			}
		}

		if info.TransitivelyErrorFree != wantTEF {
			t.Errorf("Package %q.TransitivelyErrorFree=%t, want %t",
				pkg.Path(), info.TransitivelyErrorFree, wantTEF)
		}
	}
}

// Test that both syntax (scan/parse) and type errors are both recorded
// (in PackageInfo.Errors) and reported (via Config.TypeChecker.Error).
func TestErrorReporting(t *testing.T) {
	pkgs := map[string]string{
		"a": `package a; import _ "b"; var x int = false`,
		"b": `package b; 'syntax error!`,
	}
	conf := loader.Config{
		AllowErrors:   true,
		SourceImports: true,
		Build:         fakeContext(pkgs),
	}
	var allErrors []error
	conf.TypeChecker.Error = func(err error) {
		allErrors = append(allErrors, err)
	}
	conf.Import("a")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed: %s", err)
	}
	if prog == nil {
		t.Fatalf("Load returned nil *Program")
	}

	hasError := func(errors []error, substr string) bool {
		for _, err := range errors {
			if strings.Contains(err.Error(), substr) {
				return true
			}
		}
		return false
	}

	// TODO(adonovan): test keys of ImportMap.

	// Check errors recorded in each PackageInfo.
	for pkg, info := range prog.AllPackages {
		switch pkg.Path() {
		case "a":
			if !hasError(info.Errors, "cannot convert false") {
				t.Errorf("a.Errors = %v, want bool conversion (type) error", info.Errors)
			}
		case "b":
			if !hasError(info.Errors, "rune literal not terminated") {
				t.Errorf("b.Errors = %v, want unterminated literal (syntax) error", info.Errors)
			}
		}
	}

	// Check errors reported via error handler.
	if !hasError(allErrors, "cannot convert false") ||
		!hasError(allErrors, "rune literal not terminated") {
		t.Errorf("allErrors = %v, want both syntax and type errors", allErrors)
	}
}
