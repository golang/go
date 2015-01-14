// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader_test

import (
	"fmt"
	"go/build"
	"sort"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
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
		t.Errorf("loadFromArgs(%q): got %d items, want 1", args, len(prog.Created))
	}
	if len(prog.Created) > 0 {
		path := prog.Created[0].Pkg.Path()
		if path != "P" {
			t.Errorf("loadFromArgs(%q): got %v, want [P]", prog.Created, path)
		}
	}
}

// Simplifying wrapper around buildutil.FakeContext for single-file packages.
func fakeContext(pkgs map[string]string) *build.Context {
	pkgs2 := make(map[string]map[string]string)
	for path, content := range pkgs {
		pkgs2[path] = map[string]string{"x.go": content}
	}
	return buildutil.FakeContext(pkgs2)
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

func hasError(errors []error, substr string) bool {
	for _, err := range errors {
		if strings.Contains(err.Error(), substr) {
			return true
		}
	}
	return false
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
	var mu sync.Mutex
	var allErrors []error
	conf.TypeChecker.Error = func(err error) {
		mu.Lock()
		allErrors = append(allErrors, err)
		mu.Unlock()
	}
	conf.Import("a")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed: %s", err)
	}
	if prog == nil {
		t.Fatalf("Load returned nil *Program")
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

func TestCycles(t *testing.T) {
	for _, test := range []struct {
		descr   string
		ctxt    *build.Context
		wantErr string
	}{
		{
			"self-cycle",
			fakeContext(map[string]string{
				"main":      `package main; import _ "selfcycle"`,
				"selfcycle": `package selfcycle; import _ "selfcycle"`,
			}),
			`import cycle: selfcycle -> selfcycle`,
		},
		{
			"three-package cycle",
			fakeContext(map[string]string{
				"main": `package main; import _ "a"`,
				"a":    `package a; import _ "b"`,
				"b":    `package b; import _ "c"`,
				"c":    `package c; import _ "a"`,
			}),
			`import cycle: c -> a -> b -> c`,
		},
		{
			"self-cycle in dependency of test file",
			buildutil.FakeContext(map[string]map[string]string{
				"main": {
					"main.go":      `package main`,
					"main_test.go": `package main; import _ "a"`,
				},
				"a": {
					"a.go": `package a; import _ "a"`,
				},
			}),
			`import cycle: a -> a`,
		},
		// TODO(adonovan): fix: these fail
		// {
		// 	"two-package cycle in dependency of test file",
		// 	buildutil.FakeContext(map[string]map[string]string{
		// 		"main": {
		// 			"main.go":      `package main`,
		// 			"main_test.go": `package main; import _ "a"`,
		// 		},
		// 		"a": {
		// 			"a.go": `package a; import _ "main"`,
		// 		},
		// 	}),
		// 	`import cycle: main -> a -> main`,
		// },
		// {
		// 	"self-cycle in augmented package",
		// 	buildutil.FakeContext(map[string]map[string]string{
		// 		"main": {
		// 			"main.go":      `package main`,
		// 			"main_test.go": `package main; import _ "main"`,
		// 		},
		// 	}),
		// 	`import cycle: main -> main`,
		// },
	} {
		conf := loader.Config{
			AllowErrors:   true,
			SourceImports: true,
			Build:         test.ctxt,
		}
		var mu sync.Mutex
		var allErrors []error
		conf.TypeChecker.Error = func(err error) {
			mu.Lock()
			allErrors = append(allErrors, err)
			mu.Unlock()
		}
		conf.ImportWithTests("main")

		prog, err := conf.Load()
		if err != nil {
			t.Errorf("%s: Load failed: %s", test.descr, err)
		}
		if prog == nil {
			t.Fatalf("%s: Load returned nil *Program", test.descr)
		}

		if !hasError(allErrors, test.wantErr) {
			t.Errorf("%s: Load() errors = %q, want %q",
				test.descr, allErrors, test.wantErr)
		}
	}

	// TODO(adonovan):
	// - Test that in a legal test cycle, none of the symbols
	//   defined by augmentation are visible via import.
}
