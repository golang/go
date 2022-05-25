// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"fmt"
	"strings"
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
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
