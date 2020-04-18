// errorcheck

// Verify that the Go compiler will not
// die after running into an undefined
// type in the argument list for a
// function.
// Does not compile.

package main

func mine1(int b) int { // ERROR "undefined.*b"
	return b + 2 // ERROR "undefined.*b"
}

func mine2(b int) int {
	return b
}

func main() {
	mine2()     // GCCGO_ERROR "not enough arguments in call to mine2"
	с = mine1() // ERROR "undefined.*c"
}
