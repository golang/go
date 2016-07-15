package main

import "lib"

// Tests of 'what' queries, -format=json.
// See go.tools/guru/guru_test.go for explanation.
// See what-json.golden for expected query results.

func main() {
	f() // @what call "f"
}

var _ lib.Var // @what pkg "lib"
type _ lib.T
