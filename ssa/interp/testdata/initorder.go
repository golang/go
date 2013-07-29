package main

// Test of initialization order of package-level vars.

type T int

var counter int

func next() int {
	c := counter
	counter++
	return c
}

func (T) next() int {
	return next()
}

var t T

func makeOrder1() [6]int {
	return [6]int{f1, b1, d1, e1, c1, a1}
}

func makeOrder2() [6]int {
	return [6]int{f2, b2, d2, e2, c2, a2}
}

var order1 = makeOrder1()

func main() {
	// order1 is a package-level variable:
	// [a-f]1 are initialized is reference order.
	if order1 != [6]int{0, 1, 2, 3, 4, 5} {
		panic(order1)
	}

	// order2 is a local variable:
	// [a-f]2 are initialized in lexical order.
	var order2 = makeOrder2()
	if order2 != [6]int{11, 7, 9, 10, 8, 6} {
		panic(order2)
	}
}

// The references traversal visits through calls to package-level
// functions (next), method expressions (T.next) and methods (t.next).

var a1, b1 = next(), next()
var c1, d1 = T.next(0), T.next(0)
var e1, f1 = t.next(), t.next()

var a2, b2 = next(), next()
var c2, d2 = T.next(0), T.next(0)
var e2, f2 = t.next(), t.next()
