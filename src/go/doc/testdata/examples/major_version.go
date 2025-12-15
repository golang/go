package foo_test

import (
	"example.com/foo/v3"
	"example.com/go-bar"
)

func Example() {
	foo.Print("hello")
	bar.Print("world")
	// Output:
	// hello
	// world
}
