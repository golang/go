package main

import "testing"

func TestSomething(t *testing.T) {
	go infloop()
	println(blah(1) + blah(0))
}
