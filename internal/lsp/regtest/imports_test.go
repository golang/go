package regtest

import (
	"testing"
)

const needs = `
-- go.mod --
module foo

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

func TestIssue38815(t *testing.T) {
	// it was returning
	// "package main\nimport \"testing\"\npackage main..."
	runner.Run(t, needs, func(t *testing.T, env *Env) {
		env.CreateBuffer("a_test.go", ntest)
		env.SaveBuffer("a_test.go")
		got := env.Editor.BufferText("a_test.go")
		if want != got {
			t.Errorf("got\n%q, wanted\n%q", got, want)
		}
	})
}

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

func TestVim1(t *testing.T) {
	// The file remains unchanged, but if there are any CodeActions returned, they confuse vim.
	// Therefore check for no CodeActions
	runner.Run(t, vim1, func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", vim1)
		env.OrganizeImports("main.go")
		actions := env.CodeAction("main.go")
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

func TestVim2(t *testing.T) {
	runner.Run(t, vim1, func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", vim2)
		env.OrganizeImports("main.go")
		actions := env.CodeAction("main.go")
		if len(actions) > 0 {
			t.Errorf("unexpected actions %#v", actions)
		}
	})
}
