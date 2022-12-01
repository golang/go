// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
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
		file, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `fmt.(Print)`))
		refs, err := env.Editor.References(env.Ctx, file, pos)
		if err != nil {
			t.Fatal(err)
		}
		if len(refs) != 2 {
			t.Fatalf("got %v reference(s), want 2", len(refs))
		}
		// The first reference is guaranteed to be the definition.
		if got, want := refs[1].URI, env.Sandbox.Workdir.URI("main.go"); got != want {
			t.Errorf("found reference in %v, wanted %v", got, want)
		}
	})
}

// This reproduces and tests golang/go#48400.
func TestReferencesPanicOnError(t *testing.T) {
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
		file, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `Error`))
		refs, err := env.Editor.References(env.Ctx, file, pos)
		if err == nil {
			t.Fatalf("expected error for references, instead got %v", refs)
		}
		wantErr := "no position for func (error).Error() string"
		if err.Error() != wantErr {
			t.Fatalf("expected error with message %s, instead got %s", wantErr, err.Error())
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
			f := fmt.Sprintf("%s/a.go", test.packageName)
			env.OpenFile(f)
			pos := env.RegexpSearch(f, test.packageName)
			refs := env.References(fmt.Sprintf("%s/a.go", test.packageName), pos)
			if len(refs) != test.wantRefCount {
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
type Interface interface{ M() }

func _() {
	_ = bar.Blah
}

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

		// Helper to map locations relative file paths.
		fileLocations := func(locs []protocol.Location) []string {
			var got []string
			for _, loc := range locs {
				got = append(got, env.Sandbox.Workdir.URIToPath(loc.URI))
			}
			sort.Strings(got)
			return got
		}

		refTests := []struct {
			re       string
			wantRefs []string
		}{
			// Blah is referenced:
			// - inside the foo.mod/bar (ordinary) package
			// - inside the foo.mod/bar [foo.mod/bar.test] test variant package
			// - from the foo.mod/bar_test [foo.mod/bar.test] x_test package
			// - from the foo.mod/foo package
			{"Blah", []string{"bar/bar.go", "bar/bar_test.go", "bar/bar_x_test.go", "foo/foo.go"}},

			// Foo is referenced in bar_x_test.go via the intermediate test variant
			// foo.mod/foo [foo.mod/bar.test].
			{"Foo", []string{"bar/bar_x_test.go", "foo/foo.go"}},
		}

		for _, test := range refTests {
			pos := env.RegexpSearch("foo/foo.go", test.re)
			refs := env.References("foo/foo.go", pos)

			got := fileLocations(refs)
			if diff := cmp.Diff(test.wantRefs, got); diff != "" {
				t.Errorf("References(%q) returned unexpected diff (-want +got):\n%s", test.re, diff)
			}
		}

		implTests := []struct {
			re        string
			wantImpls []string
		}{
			// Interface is implemented both in foo.mod/bar [foo.mod/bar.test] (which
			// doesn't import foo), and in foo.mod/bar_test [foo.mod/bar.test], which
			// imports the test variant of foo.
			{"Interface", []string{"bar/bar_test.go", "bar/bar_x_test.go"}},
		}

		for _, test := range implTests {
			pos := env.RegexpSearch("foo/foo.go", test.re)
			refs := env.Implementations("foo/foo.go", pos)

			got := fileLocations(refs)
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
	testenv.NeedsGo1Point(t, 14)
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
		refPos := env.RegexpSearch("a.go", "I") // find "I" reference

		// Initially, a.I has one implementation b.B in
		// the module cache, not the vendor tree.
		checkVendor(env.Implementations("a.go", refPos), false)

		// Run 'go mod vendor' outside the editor.
		if err := env.Sandbox.RunGoCommand(env.Ctx, ".", "mod", []string{"vendor"}, true); err != nil {
			t.Fatalf("go mod vendor: %v", err)
		}

		// Synchronize changes to watched files.
		env.Await(env.DoneWithChangeWatchedFiles())

		// Now, b.B is found in the vendor tree.
		checkVendor(env.Implementations("a.go", refPos), true)

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
		checkVendor(env.Implementations("a.go", refPos), false)
	})
}
