package testy

import "testing"

func TestSomething(t *testing.T) {
	var x int //@rename("x", "testyX")
	a()       //@rename("a", "b")
}
