package main

var sink int

//go:noinline
func test() {
	// This is for #30167, incorrect line numbers in an infinite loop
	go func() {}()

	for {
	}
}

func main() {
	test()
}
