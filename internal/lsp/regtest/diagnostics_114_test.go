// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 1.14

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

const ardanLabsProxy = `
-- github.com/ardanlabs/conf@v1.2.3/go.mod --
module github.com/ardanlabs/conf

go 1.12
-- github.com/ardanlabs/conf@v1.2.3/conf.go --
package conf

var ErrHelpWanted error
`

// -modfile flag that is used to provide modfile diagnostics is only available
// with 1.14.
func Test_Issue38211(t *testing.T) {
	const ardanLabs = `
-- go.mod --
module mod.com

go 1.14
-- main.go --
package main

import "github.com/ardanlabs/conf"

func main() {
	_ = conf.ErrHelpWanted
}
`
	runner.Run(t, ardanLabs, func(t *testing.T, env *Env) {
		// Expect a diagnostic with a suggested fix to add
		// "github.com/ardanlabs/conf" to the go.mod file.
		env.OpenFile("go.mod")
		env.OpenFile("main.go")
		metBy := env.Await(
			env.DiagnosticAtRegexp("main.go", `"github.com/ardanlabs/conf"`),
		)
		d, ok := metBy[0].(*protocol.PublishDiagnosticsParams)
		if !ok {
			t.Fatalf("unexpected type for metBy (%T)", metBy)
		}
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.SaveBuffer("go.mod")
		env.Await(
			EmptyDiagnostics("main.go"),
		)
		// Comment out the line that depends on conf and expect a
		// diagnostic and a fix to remove the import.
		env.RegexpReplace("main.go", "_ = conf.ErrHelpWanted", "//_ = conf.ErrHelpWanted")
		env.Await(
			env.DiagnosticAtRegexp("main.go", `"github.com/ardanlabs/conf"`),
		)
		env.SaveBuffer("main.go")
		// Expect a diagnostic and fix to remove the dependency in the go.mod.
		metBy = env.Await(
			EmptyDiagnostics("main.go"),
			env.DiagnosticAtRegexp("go.mod", "require github.com/ardanlabs/conf"),
		)
		d, ok = metBy[1].(*protocol.PublishDiagnosticsParams)
		if !ok {
			t.Fatalf("unexpected type for metBy (%T)", metBy)
		}
		env.ApplyQuickFixes("go.mod", d.Diagnostics)
		env.SaveBuffer("go.mod")
		env.Await(
			EmptyDiagnostics("go.mod"),
		)
		// Uncomment the lines and expect a new diagnostic for the import.
		env.RegexpReplace("main.go", "//_ = conf.ErrHelpWanted", "_ = conf.ErrHelpWanted")
		env.SaveBuffer("main.go")
		env.Await(
			env.DiagnosticAtRegexp("main.go", `"github.com/ardanlabs/conf"`),
		)
	}, WithProxy(ardanLabsProxy))
}

// Test for golang/go#38207.
func TestNewModule_Issue38207(t *testing.T) {
	const emptyFile = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
`
	runner.Run(t, emptyFile, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OpenFile("go.mod")
		env.EditBuffer("main.go", fake.NewEdit(0, 0, 0, 0, `package main

import "github.com/ardanlabs/conf"

func main() {
	_ = conf.ErrHelpWanted
}
`))
		env.SaveBuffer("main.go")
		metBy := env.Await(
			env.DiagnosticAtRegexp("main.go", `"github.com/ardanlabs/conf"`),
		)
		d, ok := metBy[0].(*protocol.PublishDiagnosticsParams)
		if !ok {
			t.Fatalf("unexpected type for diagnostics (%T)", d)
		}
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.Await(
			EmptyDiagnostics("main.go"),
		)
	}, WithProxy(ardanLabsProxy))
}
