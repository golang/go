package main

import "fmt"

func F[T any](n T) {
	fmt.Printf("called\n")
}

func G[T any](n T) {
	F(n)
	fmt.Printf("after\n")
}

func main() {
	G(3)
}
