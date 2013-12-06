package main

// Tests of panic/recover.

import "fmt"

func fortyTwo() (r int) {
	r = 42
	// The next two statements simulate a 'return' statement.
	defer func() { recover() }()
	panic(nil)
}

func zero() int {
	defer func() { recover() }()
	panic(1)
}

func zeroEmpty() (int, string) {
	defer func() { recover() }()
	panic(1)
}

func main() {
	if r := fortyTwo(); r != 42 {
		panic(r)
	}
	if r := zero(); r != 0 {
		panic(r)
	}
	if r, s := zeroEmpty(); r != 0 || s != "" {
		panic(fmt.Sprint(r, s))
	}
}
