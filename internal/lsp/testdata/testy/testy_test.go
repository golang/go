package testy

import "testing"

func TestSomething(t *testing.T) { //@item(TestSomething, "TestSomething(t *testing.T)", "", "func")
	var x int //@mark(testyX, "x"),diag("x", "LSP", "x declared but not used"),refs("x", testyX)
	a()       //@mark(testyA, "a")
}
