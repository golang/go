package main

// Test of initialization order of package-level vars.

var counter int

func next() int {
	c := counter
	counter++
	return c
}

func makeOrder1() [6]int {
	// The values of these vars are determined by the (arbitrary)
	// order in which we refer to them here. f=0, b=1, d=2, etc.
	return [6]int{f1, b1, d1, e1, c1, a1}
}

func makeOrder2() [6]int {
	// The values of these vars are independent of the order in
	// which we refer to them here.  a=6, b=7, c=8, etc.
	return [6]int{f2, b2, d2, e2, c2, a2}
}

var order1 = makeOrder1()

func main() {
	// order1 is a package-level variable:
	// [a-f]1 are initialized in reference order.
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

var a1, b1 = next(), next()
var c1, d1 = next(), next()
var e1, f1 = next(), next()

var a2, b2 = next(), next()
var c2, d2 = next(), next()
var e2, f2 = next(), next()
