// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

const internalDefinition = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println(message)
}
-- const.go --
package main

const message = "Hello World."
`

func TestGoToInternalDefinition(t *testing.T) {
	Run(t, internalDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		loc := env.GoToDefinition(env.RegexpSearch("main.go", "message"))
		name := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "const.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := env.RegexpSearch("const.go", "message"); loc != want {
			t.Errorf("GoToDefinition: got location %v, want %v", loc, want)
		}
	})
}

const linknameDefinition = `
-- go.mod --
module mod.com

-- upper/upper.go --
package upper

import (
	_ "unsafe"

	_ "mod.com/middle"
)

//go:linkname foo mod.com/lower.bar
func foo() string

-- middle/middle.go --
package middle

import (
	_ "mod.com/lower"
)

-- lower/lower.s --

-- lower/lower.go --
package lower

func bar() string {
	return "bar as foo"
}`

func TestGoToLinknameDefinition(t *testing.T) {
	Run(t, linknameDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("upper/upper.go")

		// Jump from directives 2nd arg.
		start := env.RegexpSearch("upper/upper.go", `lower.bar`)
		loc := env.GoToDefinition(start)
		name := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "lower/lower.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := env.RegexpSearch("lower/lower.go", `bar`); loc != want {
			t.Errorf("GoToDefinition: got position %v, want %v", loc, want)
		}
	})
}

const linknameDefinitionReverse = `
-- go.mod --
module mod.com

-- upper/upper.s --

-- upper/upper.go --
package upper

import (
	_ "mod.com/middle"
)

func foo() string

-- middle/middle.go --
package middle

import (
	_ "mod.com/lower"
)

-- lower/lower.go --
package lower

import _ "unsafe"

//go:linkname bar mod.com/upper.foo
func bar() string {
	return "bar as foo"
}`

func TestGoToLinknameDefinitionInReverseDep(t *testing.T) {
	Run(t, linknameDefinitionReverse, func(t *testing.T, env *Env) {
		env.OpenFile("lower/lower.go")

		// Jump from directives 2nd arg.
		start := env.RegexpSearch("lower/lower.go", `upper.foo`)
		loc := env.GoToDefinition(start)
		name := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "upper/upper.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := env.RegexpSearch("upper/upper.go", `foo`); loc != want {
			t.Errorf("GoToDefinition: got position %v, want %v", loc, want)
		}
	})
}

// The linkname directive connects two packages not related in the import graph.
const linknameDefinitionDisconnected = `
-- go.mod --
module mod.com

-- a/a.go --
package a

import (
	_ "unsafe"
)

//go:linkname foo mod.com/b.bar
func foo() string

-- b/b.go --
package b

func bar() string {
	return "bar as foo"
}`

func TestGoToLinknameDefinitionDisconnected(t *testing.T) {
	Run(t, linknameDefinitionDisconnected, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")

		// Jump from directives 2nd arg.
		start := env.RegexpSearch("a/a.go", `b.bar`)
		loc := env.GoToDefinition(start)
		name := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "b/b.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := env.RegexpSearch("b/b.go", `bar`); loc != want {
			t.Errorf("GoToDefinition: got position %v, want %v", loc, want)
		}
	})
}

const stdlibDefinition = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Printf()
}`

func TestGoToStdlibDefinition_Issue37045(t *testing.T) {
	Run(t, stdlibDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		loc := env.GoToDefinition(env.RegexpSearch("main.go", `fmt.(Printf)`))
		name := env.Sandbox.Workdir.URIToPath(loc.URI)
		if got, want := path.Base(name), "print.go"; got != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}

		// Test that we can jump to definition from outside our workspace.
		// See golang.org/issues/37045.
		newLoc := env.GoToDefinition(loc)
		newName := env.Sandbox.Workdir.URIToPath(newLoc.URI)
		if newName != name {
			t.Errorf("GoToDefinition is not idempotent: got %q, want %q", newName, name)
		}
		if newLoc != loc {
			t.Errorf("GoToDefinition is not idempotent: got %v, want %v", newLoc, loc)
		}
	})
}

func TestUnexportedStdlib_Issue40809(t *testing.T) {
	Run(t, stdlibDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		loc := env.GoToDefinition(env.RegexpSearch("main.go", `fmt.(Printf)`))
		name := env.Sandbox.Workdir.URIToPath(loc.URI)

		loc = env.RegexpSearch(name, `:=\s*(newPrinter)\(\)`)

		// Check that we can find references on a reference
		refs := env.References(loc)
		if len(refs) < 5 {
			t.Errorf("expected 5+ references to newPrinter, found: %#v", refs)
		}

		loc = env.GoToDefinition(loc)
		content, _ := env.Hover(loc)
		if !strings.Contains(content.Value, "newPrinter") {
			t.Fatal("definition of newPrinter went to the incorrect place")
		}
		// And on the definition too.
		refs = env.References(loc)
		if len(refs) < 5 {
			t.Errorf("expected 5+ references to newPrinter, found: %#v", refs)
		}
	})
}

// Test the hover on an error's Error function.
// This can't be done via the marker tests because Error is a builtin.
func TestHoverOnError(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
	var err error
	err.Error()
}`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		content, _ := env.Hover(env.RegexpSearch("main.go", "Error"))
		if content == nil {
			t.Fatalf("nil hover content for Error")
		}
		want := "```go\nfunc (error).Error() string\n```"
		if content.Value != want {
			t.Fatalf("hover failed:\n%s", compare.Text(want, content.Value))
		}
	})
}

func TestImportShortcut(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {}
`
	for _, tt := range []struct {
		wantLinks      int
		importShortcut string
	}{
		{1, "Link"},
		{0, "Definition"},
		{1, "Both"},
	} {
		t.Run(tt.importShortcut, func(t *testing.T) {
			WithOptions(
				Settings{"importShortcut": tt.importShortcut},
			).Run(t, mod, func(t *testing.T, env *Env) {
				env.OpenFile("main.go")
				loc := env.GoToDefinition(env.RegexpSearch("main.go", `"fmt"`))
				if loc == (protocol.Location{}) {
					t.Fatalf("expected definition, got none")
				}
				links := env.DocumentLink("main.go")
				if len(links) != tt.wantLinks {
					t.Fatalf("expected %v links, got %v", tt.wantLinks, len(links))
				}
			})
		})
	}
}

func TestGoToTypeDefinition_Issue38589(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

type Int int

type Struct struct{}

func F1() {}
func F2() (int, error) { return 0, nil }
func F3() (**Struct, bool, *Int, error) { return nil, false, nil, nil }
func F4() (**Struct, bool, *float64, error) { return nil, false, nil, nil }

func main() {}
`

	for _, tt := range []struct {
		re         string
		wantError  bool
		wantTypeRe string
	}{
		{re: `F1`, wantError: true},
		{re: `F2`, wantError: true},
		{re: `F3`, wantError: true},
		{re: `F4`, wantError: false, wantTypeRe: `type (Struct)`},
	} {
		t.Run(tt.re, func(t *testing.T) {
			Run(t, mod, func(t *testing.T, env *Env) {
				env.OpenFile("main.go")

				loc, err := env.Editor.TypeDefinition(env.Ctx, env.RegexpSearch("main.go", tt.re))
				if tt.wantError {
					if err == nil {
						t.Fatal("expected error, got nil")
					}
					return
				}
				if err != nil {
					t.Fatalf("expected nil error, got %s", err)
				}

				typeLoc := env.RegexpSearch("main.go", tt.wantTypeRe)
				if loc != typeLoc {
					t.Errorf("invalid pos: want %+v, got %+v", typeLoc, loc)
				}
			})
		})
	}
}

func TestGoToTypeDefinition_Issue60544(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.19
-- main.go --
package main

func F[T comparable]() {}
`

	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")

		_ = env.TypeDefinition(env.RegexpSearch("main.go", "comparable")) // must not panic
	})
}

// Test for golang/go#47825.
func TestImportTestVariant(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- client/test/role.go --
package test

import _ "mod.com/client"

type RoleSetup struct{}
-- client/client_role_test.go --
package client_test

import (
	"testing"
	_ "mod.com/client"
	ctest "mod.com/client/test"
)

func TestClient(t *testing.T) {
	_ = ctest.RoleSetup{}
}
-- client/client_test.go --
package client

import "testing"

func TestClient(t *testing.T) {}
-- client.go --
package client
`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("client/client_role_test.go")
		env.GoToDefinition(env.RegexpSearch("client/client_role_test.go", "RoleSetup"))
	})
}

// This test exercises a crashing pattern from golang/go#49223.
func TestGoToCrashingDefinition_Issue49223(t *testing.T) {
	Run(t, "", func(t *testing.T, env *Env) {
		params := &protocol.DefinitionParams{}
		params.TextDocument.URI = protocol.DocumentURI("fugitive%3A///Users/user/src/mm/ems/.git//0/pkg/domain/treasury/provider.go")
		params.Position.Character = 18
		params.Position.Line = 0
		env.Editor.Server.Definition(env.Ctx, params)
	})
}

// TestVendoringInvalidatesMetadata ensures that gopls uses the
// correct metadata even after an external 'go mod vendor' command
// causes packages to move; see issue #55995.
// See also TestImplementationsInVendor, which tests the same fix.
func TestVendoringInvalidatesMetadata(t *testing.T) {
	t.Skip("golang/go#56169: file watching does not capture vendor dirs")

	const proxy = `
-- other.com/b@v1.0.0/go.mod --
module other.com/b
go 1.14

-- other.com/b@v1.0.0/b.go --
package b
const K = 0
`
	const src = `
-- go.mod --
module example.com/a
go 1.14
require other.com/b v1.0.0

-- go.sum --
other.com/b v1.0.0 h1:1wb3PMGdet5ojzrKl+0iNksRLnOM9Jw+7amBNqmYwqk=
other.com/b v1.0.0/go.mod h1:TgHQFucl04oGT+vrUm/liAzukYHNxCwKNkQZEyn3m9g=

-- a.go --
package a
import "other.com/b"
const _ = b.K

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

		env.OpenFile("a.go")
		refLoc := env.RegexpSearch("a.go", "K") // find "b.K" reference

		// Initially, b.K is defined in the module cache.
		gotLoc := env.GoToDefinition(refLoc)
		gotFile := env.Sandbox.Workdir.URIToPath(gotLoc.URI)
		wantCache := filepath.ToSlash(env.Sandbox.GOPATH()) + "/pkg/mod/other.com/b@v1.0.0/b.go"
		if gotFile != wantCache {
			t.Errorf("GoToDefinition, before: got file %q, want %q", gotFile, wantCache)
		}

		// Run 'go mod vendor' outside the editor.
		if err := env.Sandbox.RunGoCommand(env.Ctx, ".", "mod", []string{"vendor"}, nil, true); err != nil {
			t.Fatalf("go mod vendor: %v", err)
		}

		// Synchronize changes to watched files.
		env.Await(env.DoneWithChangeWatchedFiles())

		// Now, b.K is defined in the vendor tree.
		gotLoc = env.GoToDefinition(refLoc)
		wantVendor := "vendor/other.com/b/b.go"
		if gotFile != wantVendor {
			t.Errorf("GoToDefinition, after go mod vendor: got file %q, want %q", gotFile, wantVendor)
		}

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

		// b.K is once again defined in the module cache.
		gotLoc = env.GoToDefinition(gotLoc)
		gotFile = env.Sandbox.Workdir.URIToPath(gotLoc.URI)
		if gotFile != wantCache {
			t.Errorf("GoToDefinition, after rm -rf vendor: got file %q, want %q", gotFile, wantCache)
		}
	})
}

const embedDefinition = `
-- go.mod --
module mod.com

-- main.go --
package main

import (
	"embed"
)

//go:embed *.txt
var foo embed.FS

func main() {}

-- skip.sql --
SKIP

-- foo.txt --
FOO

-- skip.bat --
SKIP
`

func TestGoToEmbedDefinition(t *testing.T) {
	Run(t, embedDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")

		start := env.RegexpSearch("main.go", `\*.txt`)
		loc := env.GoToDefinition(start)

		name := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "foo.txt"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
	})
}
