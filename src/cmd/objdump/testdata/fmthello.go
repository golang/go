package main

import "fmt"

func main() {
	Println("hello, world")
}

//go:noinline
func Println(s string) {
	fmt.Println(s)
}
