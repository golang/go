package main

// Test of initialization order of package-level vars.

var counter int

func next() int {
	c := counter
	counter++
	return c
}

func makeOrder1() [6]int {
	return [6]int{f1, b1, d1, e1, c1, a1}
}

func makeOrder2() [6]int {
	return [6]int{f2, b2, d2, e2, c2, a2}
}

var order1 = makeOrder1()

func main() {
	// order1 is a package-level variable
	if order1 != [6]int{5, 1, 3, 4, 2, 0} {
		panic(order1)
	}

	// order2 is a local variable
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
