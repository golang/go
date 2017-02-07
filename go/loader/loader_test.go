// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

// +build !android

package loader_test

import (
	"fmt"
	"go/build"
	"go/constant"
	"go/types"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
)

// TestFromArgs checks that conf.FromArgs populates conf correctly.
// It does no I/O.
func TestFromArgs(t *testing.T) {
	type result struct {
		Err        string
		Rest       []string
		ImportPkgs map[string]bool
		CreatePkgs []loader.PkgSpec
	}
	for _, test := range []struct {
		args  []string
		tests bool
		want  result
	}{
		// Mix of existing and non-existent packages.
		{
			args: []string{"nosuchpkg", "errors"},
			want: result{
				ImportPkgs: map[string]bool{"errors": false, "nosuchpkg": false},
			},
		},
		// Same, with -test flag.
		{
			args:  []string{"nosuchpkg", "errors"},
			tests: true,
			want: result{
				ImportPkgs: map[string]bool{"errors": true, "nosuchpkg": true},
			},
		},
		// Surplus arguments.
		{
			args: []string{"fmt", "errors", "--", "surplus"},
			want: result{
				Rest:       []string{"surplus"},
				ImportPkgs: map[string]bool{"errors": false, "fmt": false},
			},
		},
		// Ad hoc package specified as *.go files.
		{
			args: []string{"foo.go", "bar.go"},
			want: result{CreatePkgs: []loader.PkgSpec{{
				Filenames: []string{"foo.go", "bar.go"},
			}}},
		},
		// Mixture of *.go and import paths.
		{
			args: []string{"foo.go", "fmt"},
			want: result{
				Err: "named files must be .go files: fmt",
			},
		},
	} {
		var conf loader.Config
		rest, err := conf.FromArgs(test.args, test.tests)
		got := result{
			Rest:       rest,
			ImportPkgs: conf.ImportPkgs,
			CreatePkgs: conf.CreatePkgs,
		}
		if err != nil {
			got.Err = err.Error()
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("FromArgs(%q) = %+v, want %+v", test.args, got, test.want)
		}
	}
}

func TestLoad_NoInitialPackages(t *testing.T) {
	var conf loader.Config

	const wantErr = "no initial packages were loaded"

	prog, err := conf.Load()
	if err == nil {
		t.Errorf("Load succeeded unexpectedly, want %q", wantErr)
	} else if err.Error() != wantErr {
		t.Errorf("Load failed with wrong error %q, want %q", err, wantErr)
	}
	if prog != nil {
		t.Errorf("Load unexpectedly returned a Program")
	}
}

func TestLoad_MissingInitialPackage(t *testing.T) {
	var conf loader.Config
	conf.Import("nosuchpkg")
	conf.Import("errors")

	const wantErr = "couldn't load packages due to errors: nosuchpkg"

	prog, err := conf.Load()
	if err == nil {
		t.Errorf("Load succeeded unexpectedly, want %q", wantErr)
	} else if err.Error() != wantErr {
		t.Errorf("Load failed with wrong error %q, want %q", err, wantErr)
	}
	if prog != nil {
		t.Errorf("Load unexpectedly returned a Program")
	}
}

func TestLoad_MissingInitialPackage_AllowErrors(t *testing.T) {
	var conf loader.Config
	conf.AllowErrors = true
	conf.Import("nosuchpkg")
	conf.ImportWithTests("errors")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed unexpectedly: %v", err)
	}
	if prog == nil {
		t.Fatalf("Load returned a nil Program")
	}
	if got, want := created(prog), "errors_test"; got != want {
		t.Errorf("Created = %s, want %s", got, want)
	}
	if got, want := imported(prog), "errors"; got != want {
		t.Errorf("Imported = %s, want %s", got, want)
	}
}

func TestCreateUnnamedPackage(t *testing.T) {
	var conf loader.Config
	conf.CreateFromFilenames("")
	prog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if got, want := fmt.Sprint(prog.InitialPackages()), "[(unnamed)]"; got != want {
		t.Errorf("InitialPackages = %s, want %s", got, want)
	}
}

func TestLoad_MissingFileInCreatedPackage(t *testing.T) {
	var conf loader.Config
	conf.CreateFromFilenames("", "missing.go")

	const wantErr = "couldn't load packages due to errors: (unnamed)"

	prog, err := conf.Load()
	if prog != nil {
		t.Errorf("Load unexpectedly returned a Program")
	}
	if err == nil {
		t.Fatalf("Load succeeded unexpectedly, want %q", wantErr)
	}
	if err.Error() != wantErr {
		t.Fatalf("Load failed with wrong error %q, want %q", err, wantErr)
	}
}

func TestLoad_MissingFileInCreatedPackage_AllowErrors(t *testing.T) {
	conf := loader.Config{AllowErrors: true}
	conf.CreateFromFilenames("", "missing.go")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed: %v", err)
	}
	if got, want := fmt.Sprint(prog.InitialPackages()), "[(unnamed)]"; got != want {
		t.Fatalf("InitialPackages = %s, want %s", got, want)
	}
}

func TestLoad_ParseError(t *testing.T) {
	var conf loader.Config
	conf.CreateFromFilenames("badpkg", "testdata/badpkgdecl.go")

	const wantErr = "couldn't load packages due to errors: badpkg"

	prog, err := conf.Load()
	if prog != nil {
		t.Errorf("Load unexpectedly returned a Program")
	}
	if err == nil {
		t.Fatalf("Load succeeded unexpectedly, want %q", wantErr)
	}
	if err.Error() != wantErr {
		t.Fatalf("Load failed with wrong error %q, want %q", err, wantErr)
	}
}

func TestLoad_ParseError_AllowErrors(t *testing.T) {
	var conf loader.Config
	conf.AllowErrors = true
	conf.CreateFromFilenames("badpkg", "testdata/badpkgdecl.go")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed unexpectedly: %v", err)
	}
	if prog == nil {
		t.Fatalf("Load returned a nil Program")
	}
	if got, want := created(prog), "badpkg"; got != want {
		t.Errorf("Created = %s, want %s", got, want)
	}

	badpkg := prog.Created[0]
	if len(badpkg.Files) != 1 {
		t.Errorf("badpkg has %d files, want 1", len(badpkg.Files))
	}
	wantErr := filepath.Join("testdata", "badpkgdecl.go") + ":1:34: expected 'package', found 'EOF'"
	if !hasError(badpkg.Errors, wantErr) {
		t.Errorf("badpkg.Errors = %v, want %s", badpkg.Errors, wantErr)
	}
}

func TestLoad_FromSource_Success(t *testing.T) {
	var conf loader.Config
	conf.CreateFromFilenames("P", "testdata/a.go", "testdata/b.go")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed unexpectedly: %v", err)
	}
	if prog == nil {
		t.Fatalf("Load returned a nil Program")
	}
	if got, want := created(prog), "P"; got != want {
		t.Errorf("Created = %s, want %s", got, want)
	}
}

func TestLoad_FromImports_Success(t *testing.T) {
	var conf loader.Config
	conf.ImportWithTests("fmt")
	conf.ImportWithTests("errors")

	prog, err := conf.Load()
	if err != nil {
		t.Errorf("Load failed unexpectedly: %v", err)
	}
	if prog == nil {
		t.Fatalf("Load returned a nil Program")
	}
	if got, want := created(prog), "errors_test fmt_test"; got != want {
		t.Errorf("Created = %q, want %s", got, want)
	}
	if got, want := imported(prog), "errors fmt"; got != want {
		t.Errorf("Imported = %s, want %s", got, want)
	}
	// Check set of transitive packages.
	// There are >30 and the set may grow over time, so only check a few.
	want := map[string]bool{
		"strings": true,
		"time":    true,
		"runtime": true,
		"testing": true,
		"unicode": true,
	}
	for _, path := range all(prog) {
		delete(want, path)
	}
	if len(want) > 0 {
		t.Errorf("AllPackages is missing these keys: %q", keys(want))
	}
}

func TestLoad_MissingIndirectImport(t *testing.T) {
	pkgs := map[string]string{
		"a": `package a; import _ "b"`,
		"b": `package b; import _ "c"`,
	}
	conf := loader.Config{Build: fakeContext(pkgs)}
	conf.Import("a")

	const wantErr = "couldn't load packages due to errors: b"

	prog, err := conf.Load()
	if err == nil {
		t.Errorf("Load succeeded unexpectedly, want %q", wantErr)
	} else if err.Error() != wantErr {
		t.Errorf("Load failed with wrong error %q, want %q", err, wantErr)
	}
	if prog != nil {
		t.Errorf("Load unexpectedly returned a Program")
	}
}

func TestLoad_BadDependency_AllowErrors(t *testing.T) {
	for _, test := range []struct {
		descr    string
		pkgs     map[string]string
		wantPkgs string
	}{

		{
			descr: "missing dependency",
			pkgs: map[string]string{
				"a": `package a; import _ "b"`,
				"b": `package b; import _ "c"`,
			},
			wantPkgs: "a b",
		},
		{
			descr: "bad package decl in dependency",
			pkgs: map[string]string{
				"a": `package a; import _ "b"`,
				"b": `package b; import _ "c"`,
				"c": `package`,
			},
			wantPkgs: "a b",
		},
		{
			descr: "parse error in dependency",
			pkgs: map[string]string{
				"a": `package a; import _ "b"`,
				"b": `package b; import _ "c"`,
				"c": `package c; var x = `,
			},
			wantPkgs: "a b c",
		},
	} {
		conf := loader.Config{
			AllowErrors: true,
			Build:       fakeContext(test.pkgs),
		}
		conf.Import("a")

		prog, err := conf.Load()
		if err != nil {
			t.Errorf("%s: Load failed unexpectedly: %v", test.descr, err)
		}
		if prog == nil {
			t.Fatalf("%s: Load returned a nil Program", test.descr)
		}

		if got, want := imported(prog), "a"; got != want {
			t.Errorf("%s: Imported = %s, want %s", test.descr, got, want)
		}
		if got := all(prog); strings.Join(got, " ") != test.wantPkgs {
			t.Errorf("%s: AllPackages = %s, want %s", test.descr, got, test.wantPkgs)
		}
	}
}

func TestCwd(t *testing.T) {
	ctxt := fakeContext(map[string]string{"one/two/three": `package three`})
	for _, test := range []struct {
		cwd, arg, want string
	}{
		{cwd: "/go/src/one", arg: "./two/three", want: "one/two/three"},
		{cwd: "/go/src/one", arg: "../one/two/three", want: "one/two/three"},
		{cwd: "/go/src/one", arg: "one/two/three", want: "one/two/three"},
		{cwd: "/go/src/one/two/three", arg: ".", want: "one/two/three"},
		{cwd: "/go/src/one", arg: "two/three", want: ""},
	} {
		conf := loader.Config{
			Cwd:   test.cwd,
			Build: ctxt,
		}
		conf.Import(test.arg)

		var got string
		prog, err := conf.Load()
		if prog != nil {
			got = imported(prog)
		}
		if got != test.want {
			t.Errorf("Load(%s) from %s: Imported = %s, want %s",
				test.arg, test.cwd, got, test.want)
			if err != nil {
				t.Errorf("Load failed: %v", err)
			}
		}
	}
}

func TestLoad_vendor(t *testing.T) {
	pkgs := map[string]string{
		"a":          `package a; import _ "x"`,
		"a/vendor":   ``, // mkdir a/vendor
		"a/vendor/x": `package xa`,
		"b":          `package b; import _ "x"`,
		"b/vendor":   ``, // mkdir b/vendor
		"b/vendor/x": `package xb`,
		"c":          `package c; import _ "x"`,
		"x":          `package xc`,
	}
	conf := loader.Config{Build: fakeContext(pkgs)}
	conf.Import("a")
	conf.Import("b")
	conf.Import("c")

	prog, err := conf.Load()
	if err != nil {
		t.Fatal(err)
	}

	// Check that a, b, and c see different versions of x.
	for _, r := range "abc" {
		name := string(r)
		got := prog.Package(name).Pkg.Imports()[0]
		want := "x" + name
		if got.Name() != want {
			t.Errorf("package %s import %q = %s, want %s",
				name, "x", got.Name(), want)
		}
	}
}

func TestVendorCwd(t *testing.T) {
	// Test the interaction of cwd and vendor directories.
	ctxt := fakeContext(map[string]string{
		"net":          ``, // mkdir net
		"net/http":     `package http; import _ "hpack"`,
		"vendor":       ``, // mkdir vendor
		"vendor/hpack": `package vendorhpack`,
		"hpack":        `package hpack`,
	})
	for i, test := range []struct {
		cwd, arg, want string
	}{
		{cwd: "/go/src/net", arg: "http"}, // not found
		{cwd: "/go/src/net", arg: "./http", want: "net/http vendor/hpack"},
		{cwd: "/go/src/net", arg: "hpack", want: "vendor/hpack"},
		{cwd: "/go/src/vendor", arg: "hpack", want: "vendor/hpack"},
		{cwd: "/go/src/vendor", arg: "./hpack", want: "vendor/hpack"},
	} {
		conf := loader.Config{
			Cwd:   test.cwd,
			Build: ctxt,
		}
		conf.Import(test.arg)

		var got string
		prog, err := conf.Load()
		if prog != nil {
			got = strings.Join(all(prog), " ")
		}
		if got != test.want {
			t.Errorf("#%d: Load(%s) from %s: got %s, want %s",
				i, test.arg, test.cwd, got, test.want)
			if err != nil {
				t.Errorf("Load failed: %v", err)
			}
		}
	}
}

func TestVendorCwdIssue16580(t *testing.T) {
	// Regression test for Go issue 16580.
	// Import decls in "created" packages were vendor-resolved
	// w.r.t. cwd, not the parent directory of the package's files.
	ctxt := fakeContext(map[string]string{
		"a":          ``, // mkdir a
		"a/vendor":   ``, // mkdir a/vendor
		"a/vendor/b": `package b; const X = true`,
		"b":          `package b; const X = false`,
	})
	for _, test := range []struct {
		filename, cwd string
		want          bool // expected value of b.X; depends on filename, not on cwd
	}{
		{filename: "c.go", cwd: "/go/src", want: false},
		{filename: "c.go", cwd: "/go/src/a", want: false},
		{filename: "c.go", cwd: "/go/src/a/b", want: false},
		{filename: "c.go", cwd: "/go/src/a/vendor/b", want: false},

		{filename: "/go/src/a/c.go", cwd: "/go/src", want: true},
		{filename: "/go/src/a/c.go", cwd: "/go/src/a", want: true},
		{filename: "/go/src/a/c.go", cwd: "/go/src/a/b", want: true},
		{filename: "/go/src/a/c.go", cwd: "/go/src/a/vendor/b", want: true},

		{filename: "/go/src/c/c.go", cwd: "/go/src", want: false},
		{filename: "/go/src/c/c.go", cwd: "/go/src/a", want: false},
		{filename: "/go/src/c/c.go", cwd: "/go/src/a/b", want: false},
		{filename: "/go/src/c/c.go", cwd: "/go/src/a/vendor/b", want: false},
	} {
		conf := loader.Config{
			Cwd:   test.cwd,
			Build: ctxt,
		}
		f, err := conf.ParseFile(test.filename, `package dummy; import "b"; const X = b.X`)
		if err != nil {
			t.Fatal(f)
		}
		conf.CreateFromFiles("dummy", f)

		prog, err := conf.Load()
		if err != nil {
			t.Errorf("%+v: Load failed: %v", test, err)
			continue
		}

		x := constant.BoolVal(prog.Created[0].Pkg.Scope().Lookup("X").(*types.Const).Val())
		if x != test.want {
			t.Errorf("%+v: b.X = %t", test, x)
		}
	}

	// TODO(adonovan): also test imports within XTestGoFiles.
}

// TODO(adonovan): more Load tests:
//
// failures:
// - to parse package decl of *_test.go files
// - to parse package decl of external *_test.go files
// - to parse whole of *_test.go files
// - to parse whole of external *_test.go files
// - to open a *.go file during import scanning
// - to import from binary

// features:
// - InitialPackages
// - PackageCreated hook
// - TypeCheckFuncBodies hook

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
		AllowErrors: true,
		Build:       fakeContext(pkgs),
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

// Test that syntax (scan/parse), type, and loader errors are recorded
// (in PackageInfo.Errors) and reported (via Config.TypeChecker.Error).
func TestErrorReporting(t *testing.T) {
	pkgs := map[string]string{
		"a": `package a; import (_ "b"; _ "c"); var x int = false`,
		"b": `package b; 'syntax error!`,
	}
	conf := loader.Config{
		AllowErrors: true,
		Build:       fakeContext(pkgs),
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
			if !hasError(info.Errors, "could not import c") {
				t.Errorf("a.Errors = %v, want import (loader) error", info.Errors)
			}
		case "b":
			if !hasError(info.Errors, "rune literal not terminated") {
				t.Errorf("b.Errors = %v, want unterminated literal (syntax) error", info.Errors)
			}
		}
	}

	// Check errors reported via error handler.
	if !hasError(allErrors, "cannot convert false") ||
		!hasError(allErrors, "rune literal not terminated") ||
		!hasError(allErrors, "could not import c") {
		t.Errorf("allErrors = %v, want syntax, type and loader errors", allErrors)
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
			AllowErrors: true,
			Build:       test.ctxt,
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

// ---- utilities ----

// Simplifying wrapper around buildutil.FakeContext for single-file packages.
func fakeContext(pkgs map[string]string) *build.Context {
	pkgs2 := make(map[string]map[string]string)
	for path, content := range pkgs {
		pkgs2[path] = map[string]string{"x.go": content}
	}
	return buildutil.FakeContext(pkgs2)
}

func hasError(errors []error, substr string) bool {
	for _, err := range errors {
		if strings.Contains(err.Error(), substr) {
			return true
		}
	}
	return false
}

func keys(m map[string]bool) (keys []string) {
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return
}

// Returns all loaded packages.
func all(prog *loader.Program) []string {
	var pkgs []string
	for _, info := range prog.AllPackages {
		pkgs = append(pkgs, info.Pkg.Path())
	}
	sort.Strings(pkgs)
	return pkgs
}

// Returns initially imported packages, as a string.
func imported(prog *loader.Program) string {
	var pkgs []string
	for _, info := range prog.Imported {
		pkgs = append(pkgs, info.Pkg.Path())
	}
	sort.Strings(pkgs)
	return strings.Join(pkgs, " ")
}

// Returns initially created packages, as a string.
func created(prog *loader.Program) string {
	var pkgs []string
	for _, info := range prog.Created {
		pkgs = append(pkgs, info.Pkg.Path())
	}
	return strings.Join(pkgs, " ")
}
