package main // @referrers package-decl "main"

// Tests of 'referrers' query.
// See go.tools/oracle/oracle_test.go for explanation.
// See referrers.golden for expected query results.

import "lib"

type s struct { // @referrers type " s "
	f int
}

type T int

func main() {
	var v lib.Type = lib.Const // @referrers ref-package "lib"
	_ = v.Method               // @referrers ref-method "Method"
	_ = v.Method
	v++ //@referrers ref-local "v"
	v++

	_ = s{}.f // @referrers ref-field "f"

	var s2 s
	s2.f = 1
}
