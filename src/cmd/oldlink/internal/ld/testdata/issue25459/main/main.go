package main

import "cmd/oldlink/internal/ld/testdata/issue25459/a"

var Glob int

func main() {
	a.Another()
	Glob += a.ConstIf() + a.CallConstIf()
}
