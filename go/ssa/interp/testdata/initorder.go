package main

import "fmt"

// Test of initialization order of package-level vars.

var counter int

func next() int {
	c := counter
	counter++
	return c
}

func next2() (x int, y int) {
	x = next()
	y = next()
	return
}

func makeOrder() int {
	_, _, _, _ = f, b, d, e
	return 0
}

func main() {
	// Initialization constraints:
	// - {f,b,c/d,e} < order  (ref graph traversal)
	// - order < {a}          (lexical order)
	// - b < c/d < e < f      (lexical order)
	// Solution: a b c/d e f
	abcdef := [6]int{a, b, c, d, e, f}
	if abcdef != [6]int{0, 1, 2, 3, 4, 5} {
		panic(abcdef)
	}
}

var order = makeOrder()

var a, b = next(), next()
var c, d = next2()
var e, f = next(), next()

// ------------------------------------------------------------------------

var order2 []string

func create(x int, name string) int {
	order2 = append(order2, name)
	return x
}

var C = create(B+1, "C")
var A, B = create(1, "A"), create(2, "B")

// Initialization order of package-level value specs.
func init() {
	x := fmt.Sprint(order2)
	// Result varies by toolchain.  This is a spec bug.
	if x != "[B C A]" && // gc
		x != "[A B C]" { // go/types
		panic(x)
	}
	if C != 3 {
		panic(c)
	}
}
