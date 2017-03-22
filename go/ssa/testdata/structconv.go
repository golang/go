//+build ignore

// This file is the input to TestValueForExprStructConv in identical_test.go,
// which uses the same framework as TestValueForExpr does in source_test.go.
//
// In Go 1.8, struct conversions are permitted even when the struct types have
// different tags. This wasn't permitted in earlier versions of Go, so this file
// exists separately from valueforexpr.go to just test this behavior in Go 1.8
// and later.

package main

type t1 struct {
	x int
}
type t2 struct {
	x int `tag`
}

func main() {
	var tv1 t1
	var tv2 t2 = /*@ChangeType*/ (t2(tv1))
	_ = tv2
}
