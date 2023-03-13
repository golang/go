package testy

import (
	"testing"

	sig "golang.org/lsptests/signature"
	"golang.org/lsptests/snippets"
)

func TestSomething(t *testing.T) { //@item(TestSomething, "TestSomething(t *testing.T)", "", "func")
	var x int //@mark(testyX, "x"),diag("x", "compiler", "x declared (and|but) not used", "error")
	a()       //@mark(testyA, "a")
}

func _() {
	_ = snippets.X(nil) //@signature("nil", "X(_ map[sig.Alias]types.CoolAlias) map[sig.Alias]types.CoolAlias", 0)
	var _ sig.Alias
}
