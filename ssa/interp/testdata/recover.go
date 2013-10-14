package main

// Tests of panic/recover.

func fortyTwo() (r int) {
	r = 42
	// The next two statements simulate a 'return' statement.
	defer func() { recover() }()
	panic(nil)
}

func main() {
	if r := fortyTwo(); r != 42 {
		panic(r)
	}
}
