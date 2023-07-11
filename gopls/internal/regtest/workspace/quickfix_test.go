// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/internal/testenv"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestQuickFix_UseModule(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // needs go.work

	const files = `
-- go.work --
go 1.20

use (
	./a
)
-- a/go.mod --
module mod.com/a

go 1.18

-- a/main.go --
package main

import "mod.com/a/lib"

func main() {
	_ = lib.C
}

-- a/lib/lib.go --
package lib

const C = "b"
-- b/go.mod --
module mod.com/b

go 1.18

-- b/main.go --
package main

import "mod.com/b/lib"

func main() {
	_ = lib.C
}

-- b/lib/lib.go --
package lib

const C = "b"
`

	for _, title := range []string{
		"Use this module",
		"Use all modules",
	} {
		t.Run(title, func(t *testing.T) {
			Run(t, files, func(t *testing.T, env *Env) {
				env.OpenFile("b/main.go")
				var d protocol.PublishDiagnosticsParams
				env.AfterChange(ReadDiagnostics("b/main.go", &d))
				fixes := env.GetQuickFixes("b/main.go", d.Diagnostics)
				var toApply []protocol.CodeAction
				for _, fix := range fixes {
					if strings.Contains(fix.Title, title) {
						toApply = append(toApply, fix)
					}
				}
				if len(toApply) != 1 {
					t.Fatalf("codeAction: got %d quick fixes matching %q, want 1; got: %v", len(toApply), title, toApply)
				}
				env.ApplyCodeAction(toApply[0])
				env.AfterChange(NoDiagnostics())
				want := `go 1.20

use (
	./a
	./b
)
`
				got := env.ReadWorkspaceFile("go.work")
				if diff := compare.Text(want, got); diff != "" {
					t.Errorf("unexpeced go.work content:\n%s", diff)
				}
			})
		})
	}
}

func TestQuickFix_AddGoWork(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // needs go.work

	const files = `
-- a/go.mod --
module mod.com/a

go 1.18

-- a/main.go --
package main

import "mod.com/a/lib"

func main() {
	_ = lib.C
}

-- a/lib/lib.go --
package lib

const C = "b"
-- b/go.mod --
module mod.com/b

go 1.18

-- b/main.go --
package main

import "mod.com/b/lib"

func main() {
	_ = lib.C
}

-- b/lib/lib.go --
package lib

const C = "b"
`

	tests := []struct {
		name  string
		file  string
		title string
		want  string // expected go.work content, excluding go directive line
	}{
		{
			"use b",
			"b/main.go",
			"Add a go.work file using this module",
			`
use ./b
`,
		},
		{
			"use a",
			"a/main.go",
			"Add a go.work file using this module",
			`
use ./a
`,
		},
		{
			"use all",
			"a/main.go",
			"Add a go.work file using all modules",
			`
use (
	./a
	./b
)
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			Run(t, files, func(t *testing.T, env *Env) {
				env.OpenFile(test.file)
				var d protocol.PublishDiagnosticsParams
				env.AfterChange(ReadDiagnostics(test.file, &d))
				fixes := env.GetQuickFixes(test.file, d.Diagnostics)
				var toApply []protocol.CodeAction
				for _, fix := range fixes {
					if strings.Contains(fix.Title, test.title) {
						toApply = append(toApply, fix)
					}
				}
				if len(toApply) != 1 {
					t.Fatalf("codeAction: got %d quick fixes matching %q, want 1; got: %v", len(toApply), test.title, toApply)
				}
				env.ApplyCodeAction(toApply[0])
				env.AfterChange(
					NoDiagnostics(ForFile(test.file)),
				)

				got := env.ReadWorkspaceFile("go.work")
				// Ignore the `go` directive, which we assume is on the first line of
				// the go.work file. This allows the test to be independent of go version.
				got = strings.Join(strings.Split(got, "\n")[1:], "\n")
				if diff := compare.Text(test.want, got); diff != "" {
					t.Errorf("unexpected go.work content:\n%s", diff)
				}
			})
		})
	}
}

func TestQuickFix_UnsavedGoWork(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // needs go.work

	const files = `
-- go.work --
go 1.21

use (
	./a
)
-- a/go.mod --
module mod.com/a

go 1.18

-- a/main.go --
package main

func main() {}
-- b/go.mod --
module mod.com/b

go 1.18

-- b/main.go --
package main

func main() {}
`

	for _, title := range []string{
		"Use this module",
		"Use all modules",
	} {
		t.Run(title, func(t *testing.T) {
			Run(t, files, func(t *testing.T, env *Env) {
				env.OpenFile("go.work")
				env.OpenFile("b/main.go")
				env.RegexpReplace("go.work", "go 1.21", "go 1.21 // arbitrary comment")
				var d protocol.PublishDiagnosticsParams
				env.AfterChange(ReadDiagnostics("b/main.go", &d))
				fixes := env.GetQuickFixes("b/main.go", d.Diagnostics)
				var toApply []protocol.CodeAction
				for _, fix := range fixes {
					if strings.Contains(fix.Title, title) {
						toApply = append(toApply, fix)
					}
				}
				if len(toApply) != 1 {
					t.Fatalf("codeAction: got %d quick fixes matching %q, want 1; got: %v", len(toApply), title, toApply)
				}
				fix := toApply[0]
				err := env.Editor.ApplyCodeAction(env.Ctx, fix)
				if err == nil {
					t.Fatalf("codeAction(%q) succeeded unexpectedly", fix.Title)
				}

				if got := err.Error(); !strings.Contains(got, "must save") {
					t.Errorf("codeAction(%q) returned error %q, want containing \"must save\"", fix.Title, err)
				}
			})
		})
	}
}

func TestQuickFix_GOWORKOff(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // needs go.work

	const files = `
-- go.work --
go 1.21

use (
	./a
)
-- a/go.mod --
module mod.com/a

go 1.18

-- a/main.go --
package main

func main() {}
-- b/go.mod --
module mod.com/b

go 1.18

-- b/main.go --
package main

func main() {}
`

	for _, title := range []string{
		"Use this module",
		"Use all modules",
	} {
		t.Run(title, func(t *testing.T) {
			WithOptions(
				EnvVars{"GOWORK": "off"},
			).Run(t, files, func(t *testing.T, env *Env) {
				env.OpenFile("go.work")
				env.OpenFile("b/main.go")
				var d protocol.PublishDiagnosticsParams
				env.AfterChange(ReadDiagnostics("b/main.go", &d))
				fixes := env.GetQuickFixes("b/main.go", d.Diagnostics)
				var toApply []protocol.CodeAction
				for _, fix := range fixes {
					if strings.Contains(fix.Title, title) {
						toApply = append(toApply, fix)
					}
				}
				if len(toApply) != 1 {
					t.Fatalf("codeAction: got %d quick fixes matching %q, want 1; got: %v", len(toApply), title, toApply)
				}
				fix := toApply[0]
				err := env.Editor.ApplyCodeAction(env.Ctx, fix)
				if err == nil {
					t.Fatalf("codeAction(%q) succeeded unexpectedly", fix.Title)
				}

				if got := err.Error(); !strings.Contains(got, "GOWORK=off") {
					t.Errorf("codeAction(%q) returned error %q, want containing \"GOWORK=off\"", fix.Title, err)
				}
			})
		})
	}
}
