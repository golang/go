package main

import "fmt"  // fmt

const c0 = 0;  // zero
const (
	c1 = iota;  // c1
	c2;  // c2
)


// The T type.
type T struct {
	a, b, c int  // 3 fields
}


var x int;  // x
var ()


func f0() {
	const pi = 3.14;  // pi
	var s1 struct {}
	var s2 struct {} = struct {}{};
	x := pi;
}
