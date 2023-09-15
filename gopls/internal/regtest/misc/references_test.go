// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/regtest"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

func TestStdlibReferences(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Print()
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		loc := env.GoToDefinition(env.RegexpSearch("main.go", `fmt.(Print)`))
		refs, err := env.Editor.References(env.Ctx, loc)
		if err != nil {
			t.Fatal(err)
		}
		if len(refs) != 2 {
			// TODO(adonovan): make this assertion less maintainer-hostile.
			t.Fatalf("got %v reference(s), want 2", len(refs))
		}
		// The first reference is guaranteed to be the definition.
		if got, want := refs[1].URI, env.Sandbox.Workdir.URI("main.go"); got != want {
			t.Errorf("found reference in %v, wanted %v", got, want)
		}
	})
}

// This is a regression test for golang/go#48400 (a panic).
func TestReferencesOnErrorMethod(t *testing.T) {
	// Ideally this would actually return the correct answer,
	// instead of merely failing gracefully.
	const files = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

type t interface {
	error
}

type s struct{}

func (*s) Error() string {
	return ""
}

func _() {
	var s s
	_ = s.Error()
}
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		loc := env.GoToDefinition(env.RegexpSearch("main.go", `Error`))
		refs, err := env.Editor.References(env.Ctx, loc)
		if err != nil {
			t.Fatalf("references on (*s).Error failed: %v", err)
		}
		// TODO(adonovan): this test is crying out for marker support in regtests.
		var buf strings.Builder
		for _, ref := range refs {
			fmt.Fprintf(&buf, "%s %s\n", env.Sandbox.Workdir.URIToPath(ref.URI), ref.Range)
		}
		got := buf.String()
		want := "main.go 8:10-8:15\n" + // (*s).Error decl
			"main.go 14:7-14:12\n" // s.Error() call
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("unexpected references on (*s).Error (-want +got):\n%s", diff)
		}
	})
}

func TestDefsRefsBuiltins(t *testing.T) {
	testenv.NeedsGo1Point(t, 17) // for unsafe.{Add,Slice}
	// TODO(adonovan): add unsafe.{SliceData,String,StringData} in later go versions.
	const files = `
-- go.mod --
module example.com
go 1.16

-- a.go --
package a

import "unsafe"

const _ = iota
var _ error
var _ int
var _ = append()
var _ = unsafe.Pointer(nil)
var _ = unsafe.Add(nil, nil)
var _ = unsafe.Sizeof(0)
var _ = unsafe.Alignof(0)
var _ = unsafe.Slice(nil, 0)
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		for _, name := range strings.Fields(
			"iota error int nil append iota Pointer Sizeof Alignof Add Slice") {
			loc := env.RegexpSearch("a.go", `\b`+name+`\b`)

			// definition -> {builtin,unsafe}.go
			def := env.GoToDefinition(loc)
			if (!strings.HasSuffix(string(def.URI), "builtin.go") &&
				!strings.HasSuffix(string(def.URI), "unsafe.go")) ||
				def.Range.Start.Line == 0 {
				t.Errorf("definition(%q) = %v, want {builtin,unsafe}.go",
					name, def)
			}

			// "references to (builtin "Foo"|unsafe.Foo) are not supported"
			_, err := env.Editor.References(env.Ctx, loc)
			gotErr := fmt.Sprint(err)
			if !strings.Contains(gotErr, "references to") ||
				!strings.Contains(gotErr, "not supported") ||
				!strings.Contains(gotErr, name) {
				t.Errorf("references(%q) error: got %q, want %q",
					name, gotErr, "references to ... are not supported")
			}
		}
	})
}

func TestPackageReferences(t *testing.T) {
	tests := []struct {
		packageName  string
		wantRefCount int
		wantFiles    []string
	}{
		{
			"lib1",
			3,
			[]string{
				"main.go",
				"lib1/a.go",
				"lib1/b.go",
			},
		},
		{
			"lib2",
			2,
			[]string{
				"main.go",
				"lib2/a.go",
			},
		},
	}

	const files = `
-- go.mod --
module mod.com

go 1.18
-- lib1/a.go --
package lib1

const A = 1

-- lib1/b.go --
package lib1

const B = 1

-- lib2/a.go --
package lib2

const C = 1

-- main.go --
package main

import (
	"mod.com/lib1"
	"mod.com/lib2"
)

func main() {
	println("Hello")
}
`
	Run(t, files, func(t *testing.T, env *Env) {
		for _, test := range tests {
			file := fmt.Sprintf("%s/a.go", test.packageName)
			env.OpenFile(file)
			loc := env.RegexpSearch(file, test.packageName)
			refs := env.References(loc)
			if len(refs) != test.wantRefCount {
				// TODO(adonovan): make this assertion less maintainer-hostile.
				t.Fatalf("got %v reference(s), want %d", len(refs), test.wantRefCount)
			}
			var refURIs []string
			for _, ref := range refs {
				refURIs = append(refURIs, string(ref.URI))
			}
			for _, base := range test.wantFiles {
				hasBase := false
				for _, ref := range refURIs {
					if strings.HasSuffix(ref, base) {
						hasBase = true
						break
					}
				}
				if !hasBase {
					t.Fatalf("got [%v], want reference ends with \"%v\"", strings.Join(refURIs, ","), base)
				}
			}
		}
	})
}

// Test for golang/go#43144.
//
// Verify that we search for references and implementations in intermediate
// test variants.
func TestReferencesInTestVariants(t *testing.T) {
	const files = `
-- go.mod --
module foo.mod

go 1.12
-- foo/foo.go --
package foo

import "foo.mod/bar"

const Foo = 42

type T int
type InterfaceM interface{ M() }
type InterfaceF interface{ F() }

func _() {
	_ = bar.Blah
}

-- foo/foo_test.go --
package foo

type Fer struct{}
func (Fer) F() {}

-- bar/bar.go --
package bar

var Blah = 123

-- bar/bar_test.go --
package bar

type Mer struct{}
func (Mer) M() {}

func TestBar() {
	_ = Blah
}
-- bar/bar_x_test.go --
package bar_test

import (
	"foo.mod/bar"
	"foo.mod/foo"
)

type Mer struct{}
func (Mer) M() {}

func _() {
	_ = bar.Blah
	_ = foo.Foo
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("foo/foo.go")

		refTests := []struct {
			re       string
			wantRefs []string
		}{
			// Blah is referenced:
			// - inside the foo.mod/bar (ordinary) package
			// - inside the foo.mod/bar [foo.mod/bar.test] test variant package
			// - from the foo.mod/bar_test [foo.mod/bar.test] x_test package
			// - from the foo.mod/foo package
			{"Blah", []string{"bar/bar.go:3", "bar/bar_test.go:7", "bar/bar_x_test.go:12", "foo/foo.go:12"}},

			// Foo is referenced in bar_x_test.go via the intermediate test variant
			// foo.mod/foo [foo.mod/bar.test].
			{"Foo", []string{"bar/bar_x_test.go:13", "foo/foo.go:5"}},
		}

		for _, test := range refTests {
			loc := env.RegexpSearch("foo/foo.go", test.re)
			refs := env.References(loc)

			got := fileLocations(env, refs)
			if diff := cmp.Diff(test.wantRefs, got); diff != "" {
				t.Errorf("References(%q) returned unexpected diff (-want +got):\n%s", test.re, diff)
			}
		}

		implTests := []struct {
			re        string
			wantImpls []string
		}{
			// InterfaceM is implemented both in foo.mod/bar [foo.mod/bar.test] (which
			// doesn't import foo), and in foo.mod/bar_test [foo.mod/bar.test], which
			// imports the test variant of foo.
			{"InterfaceM", []string{"bar/bar_test.go:3", "bar/bar_x_test.go:8"}},

			// A search within the ordinary package to should find implementations
			// (Fer) within the augmented test package.
			{"InterfaceF", []string{"foo/foo_test.go:3"}},
		}

		for _, test := range implTests {
			loc := env.RegexpSearch("foo/foo.go", test.re)
			impls := env.Implementations(loc)

			got := fileLocations(env, impls)
			if diff := cmp.Diff(test.wantImpls, got); diff != "" {
				t.Errorf("Implementations(%q) returned unexpected diff (-want +got):\n%s", test.re, diff)
			}
		}
	})
}

// This is a regression test for Issue #56169, in which interface
// implementations in vendored modules were not found. The actual fix
// was the same as for #55995; see TestVendoringInvalidatesMetadata.
func TestImplementationsInVendor(t *testing.T) {
	t.Skip("golang/go#56169: file watching does not capture vendor dirs")

	const proxy = `
-- other.com/b@v1.0.0/go.mod --
module other.com/b
go 1.14

-- other.com/b@v1.0.0/b.go --
package b
type B int
func (B) F() {}
`
	const src = `
-- go.mod --
module example.com/a
go 1.14
require other.com/b v1.0.0

-- go.sum --
other.com/b v1.0.0 h1:9WyCKS+BLAMRQM0CegP6zqP2beP+ShTbPaARpNY31II=
other.com/b v1.0.0/go.mod h1:TgHQFucl04oGT+vrUm/liAzukYHNxCwKNkQZEyn3m9g=

-- a.go --
package a
import "other.com/b"
type I interface { F() }
var _ b.B

`
	WithOptions(
		ProxyFiles(proxy),
		Modes(Default), // fails in 'experimental' mode
	).Run(t, src, func(t *testing.T, env *Env) {
		// Enable to debug go.sum mismatch, which may appear as
		// "module lookup disabled by GOPROXY=off", confusingly.
		if false {
			env.DumpGoSum(".")
		}

		checkVendor := func(locs []protocol.Location, wantVendor bool) {
			if len(locs) != 1 {
				t.Errorf("got %d locations, want 1", len(locs))
			} else if strings.Contains(string(locs[0].URI), "/vendor/") != wantVendor {
				t.Errorf("got location %s, wantVendor=%t", locs[0], wantVendor)
			}
		}

		env.OpenFile("a.go")
		refLoc := env.RegexpSearch("a.go", "I") // find "I" reference

		// Initially, a.I has one implementation b.B in
		// the module cache, not the vendor tree.
		checkVendor(env.Implementations(refLoc), false)

		// Run 'go mod vendor' outside the editor.
		if err := env.Sandbox.RunGoCommand(env.Ctx, ".", "mod", []string{"vendor"}, nil, true); err != nil {
			t.Fatalf("go mod vendor: %v", err)
		}

		// Synchronize changes to watched files.
		env.Await(env.DoneWithChangeWatchedFiles())

		// Now, b.B is found in the vendor tree.
		checkVendor(env.Implementations(refLoc), true)

		// Delete the vendor tree.
		if err := os.RemoveAll(env.Sandbox.Workdir.AbsPath("vendor")); err != nil {
			t.Fatal(err)
		}
		// Notify the server of the deletion.
		if err := env.Sandbox.Workdir.CheckForFileChanges(env.Ctx); err != nil {
			t.Fatal(err)
		}

		// Synchronize again.
		env.Await(env.DoneWithChangeWatchedFiles())

		// b.B is once again defined in the module cache.
		checkVendor(env.Implementations(refLoc), false)
	})
}

// This test can't be expressed as a marker test because the marker
// test framework opens all files (which is a bit of a hack), creating
// a <command-line-arguments> package for packages that otherwise
// wouldn't be found from the go.work file.
func TestReferencesFromWorkspacePackages59674(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // for go.work support
	const src = `
-- a/go.mod --
module example.com/a
go 1.12

-- b/go.mod --
module example.com/b
go 1.12

-- c/go.mod --
module example.com/c
go 1.12

-- lib/go.mod --
module example.com/lib
go 1.12

-- go.work --
use ./a
use ./b
// don't use ./c
use ./lib

-- a/a.go --
package a

import "example.com/lib"

var _ = lib.F // query here

-- b/b.go --
package b

import "example.com/lib"

var _ = lib.F // also found by references

-- c/c.go --
package c

import "example.com/lib"

var _ = lib.F // this reference should not be reported

-- lib/lib.go --
package lib

func F() {} // declaration
`
	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		refLoc := env.RegexpSearch("a/a.go", "F")
		got := fileLocations(env, env.References(refLoc))
		want := []string{"a/a.go:5", "b/b.go:5", "lib/lib.go:3"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("incorrect References (-want +got):\n%s", diff)
		}
	})
}

// Test an 'implementation' query on a type that implements 'error'.
// (Unfortunately builtin locations cannot be expressed using @loc
// in the marker test framework.)
func TestImplementationsOfError(t *testing.T) {
	const src = `
-- go.mod --
module example.com
go 1.12

-- a.go --
package a

type Error2 interface {
	Error() string
}

type MyError int
func (MyError) Error() string { return "" }

type MyErrorPtr int
func (*MyErrorPtr) Error() string { return "" }
`
	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")

		for _, test := range []struct {
			re   string
			want []string
		}{
			// error type
			{"Error2", []string{"a.go:10", "a.go:7", "std:builtin/builtin.go"}},
			{"MyError", []string{"a.go:3", "std:builtin/builtin.go"}},
			{"MyErrorPtr", []string{"a.go:3", "std:builtin/builtin.go"}},
			// error.Error method
			{"(Error).. string", []string{"a.go:11", "a.go:8", "std:builtin/builtin.go"}},
			{"MyError. (Error)", []string{"a.go:4", "std:builtin/builtin.go"}},
			{"MyErrorPtr. (Error)", []string{"a.go:4", "std:builtin/builtin.go"}},
		} {
			matchLoc := env.RegexpSearch("a.go", test.re)
			impls := env.Implementations(matchLoc)
			got := fileLocations(env, impls)
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("Implementations(%q) = %q, want %q",
					test.re, got, test.want)
			}
		}
	})
}

// fileLocations returns a new sorted array of the
// relative file name and line number of each location.
// Duplicates are not removed.
// Standard library filenames are abstracted for robustness.
func fileLocations(env *regtest.Env, locs []protocol.Location) []string {
	got := make([]string, 0, len(locs))
	for _, loc := range locs {
		path := env.Sandbox.Workdir.URIToPath(loc.URI) // (slashified)
		if i := strings.LastIndex(path, "/src/"); i >= 0 && filepath.IsAbs(path) {
			// Absolute path with "src" segment: assume it's in GOROOT.
			// Strip directory and don't add line/column since they are fragile.
			path = "std:" + path[i+len("/src/"):]
		} else {
			path = fmt.Sprintf("%s:%d", path, loc.Range.Start.Line+1)
		}
		got = append(got, path)
	}
	sort.Strings(got)
	return got
}
