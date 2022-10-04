// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
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
