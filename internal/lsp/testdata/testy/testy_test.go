package testy

import "testing"

func TestSomething(t *testing.T) { //@item(TestSomething, "TestSomething(t *testing.T)", "", "func")
	var x int //@diag("x", "LSP", "x declared but not used")
	a()
}
