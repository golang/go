package regtest

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/tests"
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
	runner.Run(t, unformattedProgram, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.FormatBuffer("main.go")
		got := env.Editor.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.golden")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", tests.Diff(want, got))
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
	runner.Run(t, onelineProgram, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.FormatBuffer("a.go")
		got := env.Editor.BufferText("a.go")
		want := env.ReadWorkspaceFile("a.go.formatted")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", tests.Diff(want, got))
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
	runner.Run(t, onelineProgramA, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.OrganizeImports("a.go")
		got := env.Editor.BufferText("a.go")
		want := env.ReadWorkspaceFile("a.go.imported")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", tests.Diff(want, got))
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
	runner.Run(t, onelineProgramB, func(t *testing.T, env *Env) {
		env.OpenFile("a.go")
		env.OrganizeImports("a.go")
		got := env.Editor.BufferText("a.go")
		want := env.ReadWorkspaceFile("a.go.imported")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", tests.Diff(want, got))
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
	runner.Run(t, disorganizedProgram, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OrganizeImports("main.go")
		got := env.Editor.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.organized")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", tests.Diff(want, got))
		}
	})
}

func TestFormattingOnSave(t *testing.T) {
	runner.Run(t, disorganizedProgram, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.SaveBuffer("main.go")
		got := env.Editor.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.formatted")
		if got != want {
			t.Errorf("unexpected formatting result:\n%s", tests.Diff(want, got))
		}
	})
}

// Reproduce golang/go#41057.
func TestCRLF(t *testing.T) {
	runner.Run(t, "-- main.go --", func(t *testing.T, env *Env) {
		want := `package main

/*
Hi description
*/
func Hi() {
}
`
		crlf := strings.ReplaceAll(want, "\n", "\r\n")
		env.CreateBuffer("main.go", crlf)
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1))
		env.SaveBuffer("main.go")
		got := env.Editor.BufferText("main.go")
		if want != got {
			t.Errorf("unexpected content after save:\n%s", tests.Diff(want, got))
		}
	})
}

func TestCRLF_42646(t *testing.T) {
	runner.Run(t, "-- main.go --", func(t *testing.T, env *Env) {
		want := `package main

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
`
		crlf := strings.ReplaceAll(want, "\n", "\r\n")
		env.CreateBuffer("main.go", crlf)
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1))
		env.OrganizeImports("main.go")
		got := env.Editor.BufferText("main.go")
		got = strings.ReplaceAll(got, "\r\n", "\n") // convert everything to LF for simplicity
		if want != got {
			t.Errorf("unexpected content after save:\n%s", tests.Diff(want, got))
		}
	})
}
