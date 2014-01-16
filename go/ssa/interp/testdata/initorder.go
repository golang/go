package main

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
	// Solution: b c/d e f a
	abcdef := [6]int{a, b, c, d, e, f}
	if abcdef != [6]int{5, 0, 1, 2, 3, 4} {
		panic(abcdef)
	}
}

var order = makeOrder()

var a, b = next(), next()
var c, d = next2()
var e, f = next(), next()
