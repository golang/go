package main

import "a"

var Glob int

func main() {
	a.Another()
	Glob += a.ConstIf() + a.CallConstIf()
}
