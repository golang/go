// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"
)

// Tests golang/go#38815.
func TestIssue38815(t *testing.T) {
	const needs = `
-- go.mod --
module foo

go 1.12
-- a.go --
package main
func f() {}
`
	const ntest = `package main
func TestZ(t *testing.T) {
	f()
}
`
	const want = `package main

import "testing"

func TestZ(t *testing.T) {
	f()
}
`

	// it was returning
	// "package main\nimport \"testing\"\npackage main..."
	Run(t, needs, func(t *testing.T, env *Env) {
		env.CreateBuffer("a_test.go", ntest)
		env.SaveBuffer("a_test.go")
		got := env.Editor.BufferText("a_test.go")
		if want != got {
			t.Errorf("got\n%q, wanted\n%q", got, want)
		}
	})
}

func TestVim1(t *testing.T) {
	const vim1 = `package main

import "fmt"

var foo = 1
var bar = 2

func main() {
	fmt.Printf("This is a test %v\n", foo)
	fmt.Printf("This is another test %v\n", foo)
	fmt.Printf("This is also a test %v\n", foo)
}
`

	// The file remains unchanged, but if there are any CodeActions returned, they confuse vim.
	// Therefore check for no CodeActions
	Run(t, "", func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", vim1)
		env.OrganizeImports("main.go")
		actions := env.CodeAction("main.go", nil)
		if len(actions) > 0 {
			got := env.Editor.BufferText("main.go")
			t.Errorf("unexpected actions %#v", actions)
			if got == vim1 {
				t.Errorf("no changes")
			} else {
				t.Errorf("got\n%q", got)
				t.Errorf("was\n%q", vim1)
			}
		}
	})
}

func TestVim2(t *testing.T) {
	const vim2 = `package main

import (
	"fmt"

	"example.com/blah"

	"rubbish.com/useless"
)

func main() {
	fmt.Println(blah.Name, useless.Name)
}
`

	Run(t, "", func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", vim2)
		env.OrganizeImports("main.go")
		actions := env.CodeAction("main.go", nil)
		if len(actions) > 0 {
			t.Errorf("unexpected actions %#v", actions)
		}
	})
}

func TestGOMODCACHE(t *testing.T) {
	const proxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/x/x.go --
package x

const X = 1
-- example.com@v1.2.3/y/y.go --
package y

const Y = 2
`
	const files = `
-- go.mod --
module mod.com

go 1.12

require example.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:6vTQqzX+pnwngZF1+5gcO3ZEWmix1jJ/h+pWS8wUxK0=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- main.go --
package main

import "example.com/x"

var _, _ = x.X, y.Y
`
	testenv.NeedsGo1Point(t, 15)

	modcache, err := ioutil.TempDir("", "TestGOMODCACHE-modcache")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(modcache)
	WithOptions(
		EnvVars{"GOMODCACHE": modcache},
		ProxyFiles(proxy),
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Await(env.DiagnosticAtRegexp("main.go", `y.Y`))
		env.SaveBuffer("main.go")
		env.Await(EmptyDiagnostics("main.go"))
		path, _ := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `y.(Y)`))
		if !strings.HasPrefix(path, filepath.ToSlash(modcache)) {
			t.Errorf("found module dependency outside of GOMODCACHE: got %v, wanted subdir of %v", path, filepath.ToSlash(modcache))
		}
	})
}

// Tests golang/go#40685.
func TestAcceptImportsQuickFixTestVariant(t *testing.T) {
	const pkg = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

import (
	"fmt"
)

func _() {
	fmt.Println("")
	os.Stat("")
}
-- a/a_test.go --
package a

import (
	"os"
	"testing"
)

func TestA(t *testing.T) {
	os.Stat("")
}
`
	Run(t, pkg, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("a/a.go", "os.Stat"),
				ReadDiagnostics("a/a.go", &d),
			),
		)
		env.ApplyQuickFixes("a/a.go", d.Diagnostics)
		env.Await(
			EmptyDiagnostics("a/a.go"),
		)
	})
}

// Test for golang/go#52784
func TestGoWorkImports(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const pkg = `
-- go.work --
go 1.19

use (
        ./caller
        ./mod
)
-- caller/go.mod --
module caller.com

go 1.18

require mod.com v0.0.0

replace mod.com => ../mod
-- caller/caller.go --
package main

func main() {
        a.Test()
}
-- mod/go.mod --
module mod.com

go 1.18
-- mod/a/a.go --
package a

func Test() {
}
`
	Run(t, pkg, func(t *testing.T, env *Env) {
		env.OpenFile("caller/caller.go")
		env.Await(env.DiagnosticAtRegexp("caller/caller.go", "a.Test"))

		// Saving caller.go should trigger goimports, which should find a.Test in
		// the mod.com module, thanks to the go.work file.
		env.SaveBuffer("caller/caller.go")
		env.Await(EmptyDiagnostics("caller/caller.go"))
	})
}
