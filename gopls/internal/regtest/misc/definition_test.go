// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"path"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/tests"
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
		name, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", "message"))
		if want := "const.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := env.RegexpSearch("const.go", "message"); pos != want {
			t.Errorf("GoToDefinition: got position %v, want %v", pos, want)
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
		name, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `fmt.(Printf)`))
		if got, want := path.Base(name), "print.go"; got != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}

		// Test that we can jump to definition from outside our workspace.
		// See golang.org/issues/37045.
		newName, newPos := env.GoToDefinition(name, pos)
		if newName != name {
			t.Errorf("GoToDefinition is not idempotent: got %q, want %q", newName, name)
		}
		if newPos != pos {
			t.Errorf("GoToDefinition is not idempotent: got %v, want %v", newPos, pos)
		}
	})
}

func TestUnexportedStdlib_Issue40809(t *testing.T) {
	Run(t, stdlibDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		name, _ := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `fmt.(Printf)`))
		env.OpenFile(name)

		pos := env.RegexpSearch(name, `:=\s*(newPrinter)\(\)`)

		// Check that we can find references on a reference
		refs := env.References(name, pos)
		if len(refs) < 5 {
			t.Errorf("expected 5+ references to newPrinter, found: %#v", refs)
		}

		name, pos = env.GoToDefinition(name, pos)
		content, _ := env.Hover(name, pos)
		if !strings.Contains(content.Value, "newPrinter") {
			t.Fatal("definition of newPrinter went to the incorrect place")
		}
		// And on the definition too.
		refs = env.References(name, pos)
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
		content, _ := env.Hover("main.go", env.RegexpSearch("main.go", "Error"))
		if content == nil {
			t.Fatalf("nil hover content for Error")
		}
		want := "```go\nfunc (error).Error() string\n```"
		if content.Value != want {
			t.Fatalf("hover failed:\n%s", tests.Diff(t, want, content.Value))
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
		wantDef        bool
		importShortcut string
	}{
		{1, false, "Link"},
		{0, true, "Definition"},
		{1, true, "Both"},
	} {
		t.Run(tt.importShortcut, func(t *testing.T) {
			WithOptions(
				EditorConfig{
					ImportShortcut: tt.importShortcut,
				},
			).Run(t, mod, func(t *testing.T, env *Env) {
				env.OpenFile("main.go")
				file, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `"fmt"`))
				if !tt.wantDef && (file != "" || pos != (fake.Pos{})) {
					t.Fatalf("expected no definition, got one: %s:%v", file, pos)
				} else if tt.wantDef && file == "" && pos == (fake.Pos{}) {
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
