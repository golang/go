// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

const basicProxy = `
-- golang.org/x/hello@v1.2.3/go.mod --
module golang.org/x/hello

go 1.14
-- golang.org/x/hello@v1.2.3/hi/hi.go --
package hi

var Goodbye error
`

func TestInconsistentVendoring(t *testing.T) {
	const pkgThatUsesVendoring = `
-- go.mod --
module mod.com

go 1.14

require golang.org/x/hello v1.2.3
-- go.sum --
golang.org/x/hello v1.2.3 h1:EcMp5gSkIhaTkPXp8/3+VH+IFqTpk3ZbpOhqk0Ncmho=
golang.org/x/hello v1.2.3/go.mod h1:WW7ER2MRNXWA6c8/4bDIek4Hc/+DofTrMaQQitGXcco=
-- vendor/modules.txt --
-- a/a1.go --
package a

import "golang.org/x/hello/hi"

func _() {
	_ = hi.Goodbye
	var q int // hardcode a diagnostic
}
`
	WithOptions(
		Modes(Default),
		ProxyFiles(basicProxy),
	).Run(t, pkgThatUsesVendoring, func(t *testing.T, env *Env) {
		env.OpenFile("a/a1.go")
		d := &protocol.PublishDiagnosticsParams{}
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("go.mod", "module mod.com"), WithMessage("Inconsistent vendoring")),
			ReadDiagnostics("go.mod", d),
		)
		env.ApplyQuickFixes("go.mod", d.Diagnostics)

		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a1.go", `q int`), WithMessage("not used")),
		)
	})
}

func TestWindowsVendoring_Issue56291(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.14

require golang.org/x/hello v1.2.3
-- go.sum --
golang.org/x/hello v1.2.3 h1:EcMp5gSkIhaTkPXp8/3+VH+IFqTpk3ZbpOhqk0Ncmho=
golang.org/x/hello v1.2.3/go.mod h1:WW7ER2MRNXWA6c8/4bDIek4Hc/+DofTrMaQQitGXcco=
-- main.go --
package main

import "golang.org/x/hello/hi"

func main() {
	_ = hi.Goodbye
}
`
	WithOptions(
		Modes(Default),
		ProxyFiles(basicProxy),
	).Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.AfterChange(NoDiagnostics())
		env.RunGoCommand("mod", "tidy")
		env.RunGoCommand("mod", "vendor")
		env.AfterChange(NoDiagnostics())
		env.RegexpReplace("main.go", `import "golang.org/x/hello/hi"`, "")
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "hi.Goodbye")),
		)
		env.SaveBuffer("main.go")
		env.AfterChange(NoDiagnostics())
	})
}
