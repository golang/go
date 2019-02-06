package testy

import "testing"

func TestSomething(t *testing.T) { //@item(TestSomething, "TestSomething(t *testing.T)", "", "func")
	var x int //@diag("x", "x declared but not used")
	a()
}
