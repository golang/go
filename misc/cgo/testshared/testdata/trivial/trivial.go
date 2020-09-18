package main

func main() {
	// This is enough to make sure that the executable references
	// a type descriptor, which was the cause of
	// https://golang.org/issue/25970.
	c := make(chan int)
	_ = c
}
