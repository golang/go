package main

// Don't import "testing", it adds a lot of callgraph edges.

func Example() {
	C(0).f()
}
