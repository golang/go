// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/internal/testenv"
)

const unformattedProgram = `
-- main.go --
package main
import "fmt"
func main(  ) {
	fmt.Println("Hello World.")
}
-- main.go.golden --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}
`

func TestFormatting(t *testing.T) {
	Run(t, unformattedProgram, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.FormatBuffer("main.go")
		got := env.Editor.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.golden")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}

// Tests golang/go#36824.
func TestFormattingOneLine36824(t *testing.T) {
	const onelineProgram = `
-- a.go --
package main; func f() {}

-- a.go.formatted --
package main

func f() {}
`
	Run(t, onelineProgram, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.FormatBuffer("a.go")
		got := env.Editor.BufferText("a.go")
		want := env.ReadWorkspaceFile("a.go.formatted")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}

// Tests golang/go#36824.
func TestFormattingOneLineImports36824(t *testing.T) {
	const onelineProgramA = `
-- a.go --
package x; func f() {fmt.Println()}

-- a.go.imported --
package x

import "fmt"

func f() { fmt.Println() }
`
	Run(t, onelineProgramA, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.OrganizeImports("a.go")
		got := env.Editor.BufferText("a.go")
		want := env.ReadWorkspaceFile("a.go.imported")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}

func TestFormattingOneLineRmImports36824(t *testing.T) {
	const onelineProgramB = `
-- a.go --
package x; import "os"; func f() {}

-- a.go.imported --
package x

func f() {}
`
	Run(t, onelineProgramB, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.OrganizeImports("a.go")
		got := env.Editor.BufferText("a.go")
		want := env.ReadWorkspaceFile("a.go.imported")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}

const disorganizedProgram = `
-- main.go --
package main

import (
	"fmt"
	"errors"
)
func main(  ) {
	fmt.Println(errors.New("bad"))
}
-- main.go.organized --
package main

import (
	"errors"
	"fmt"
)
func main(  ) {
	fmt.Println(errors.New("bad"))
}
-- main.go.formatted --
package main

import (
	"errors"
	"fmt"
)

func main() {
	fmt.Println(errors.New("bad"))
}
`

func TestOrganizeImports(t *testing.T) {
	Run(t, disorganizedProgram, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OrganizeImports("main.go")
		got := env.Editor.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.organized")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}

func TestFormattingOnSave(t *testing.T) {
	Run(t, disorganizedProgram, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.SaveBuffer("main.go")
		got := env.Editor.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.formatted")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}

// Tests various possibilities for comments in files with CRLF line endings.
// Import organization in these files has historically been a source of bugs.
func TestCRLFLineEndings(t *testing.T) {
	for _, tt := range []struct {
		issue, input, want string
	}{
		{
			issue: "41057",
			want: `package main

/*
Hi description
*/
func Hi() {
}
`,
		},
		{
			issue: "42646",
			want: `package main

import (
	"fmt"
)

/*
func upload(c echo.Context) error {
	if err := r.ParseForm(); err != nil {
		fmt.Fprintf(w, "ParseForm() err: %v", err)
		return
	}
	fmt.Fprintf(w, "POST request successful")
	path_ver := r.FormValue("path_ver")
	ukclin_ver := r.FormValue("ukclin_ver")

	fmt.Fprintf(w, "Name = %s\n", path_ver)
	fmt.Fprintf(w, "Address = %s\n", ukclin_ver)
}
*/

func main() {
	const server_port = 8080
	fmt.Printf("port: %d\n", server_port)
}
`,
		},
		{
			issue: "42923",
			want: `package main

// Line 1.
// aa
type Tree struct {
	arr []string
}
`,
		},
		{
			issue: "47200",
			input: `package main

import "fmt"

func main() {
	math.Sqrt(9)
	fmt.Println("hello")
}
`,
			want: `package main

import (
	"fmt"
	"math"
)

func main() {
	math.Sqrt(9)
	fmt.Println("hello")
}
`,
		},
	} {
		t.Run(tt.issue, func(t *testing.T) {
			Run(t, "-- main.go --", func(t *testing.T, env *Env) {
				input := tt.input
				if input == "" {
					input = tt.want
				}
				crlf := strings.ReplaceAll(input, "\n", "\r\n")
				env.CreateBuffer("main.go", crlf)
				env.Await(env.DoneWithOpen())
				env.OrganizeImports("main.go")
				got := env.Editor.BufferText("main.go")
				got = strings.ReplaceAll(got, "\r\n", "\n") // convert everything to LF for simplicity
				if tt.want != got {
					t.Errorf("unexpected content after save:\n%s", compare.Text(tt.want, got))
				}
			})
		})
	}
}

func TestFormattingOfGeneratedFile_Issue49555(t *testing.T) {
	const input = `
-- main.go --
// Code generated by generator.go. DO NOT EDIT.

package main

import "fmt"

func main() {




	fmt.Print("hello")
}
`

	Run(t, input, func(t *testing.T, env *Env) {
		wantErrSuffix := "file is generated"

		env.OpenFile("main.go")
		err := env.Editor.FormatBuffer(env.Ctx, "main.go")
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		// Check only the suffix because an error contains a dynamic path to main.go
		if !strings.HasSuffix(err.Error(), wantErrSuffix) {
			t.Fatalf("unexpected error %q, want suffix %q", err.Error(), wantErrSuffix)
		}
	})
}

func TestGofumptFormatting(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)

	// Exercise some gofumpt formatting rules:
	//  - No empty lines following an assignment operator
	//  - Octal integer literals should use the 0o prefix on modules using Go
	//    1.13 and later. Requires LangVersion to be correctly resolved.
	//  - std imports must be in a separate group at the top. Requires ModulePath
	//    to be correctly resolved.
	const input = `
-- go.mod --
module foo

go 1.17
-- foo.go --
package foo

import (
	"foo/bar"
	"fmt"
)

const perm = 0755

func foo() {
	foo :=
		"bar"
	fmt.Println(foo, bar.Bar)
}
-- foo.go.formatted --
package foo

import (
	"fmt"

	"foo/bar"
)

const perm = 0o755

func foo() {
	foo := "bar"
	fmt.Println(foo, bar.Bar)
}
-- bar/bar.go --
package bar

const Bar = 42
`

	WithOptions(
		Settings{
			"gofumpt": true,
		},
	).Run(t, input, func(t *testing.T, env *Env) {
		env.OpenFile("foo.go")
		env.FormatBuffer("foo.go")
		got := env.Editor.BufferText("foo.go")
		want := env.ReadWorkspaceFile("foo.go.formatted")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", compare.Text(want, got))
		}
	})
}
