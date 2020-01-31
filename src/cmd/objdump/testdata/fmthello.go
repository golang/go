package main

import "fmt"

func main() {
	Println("hello, world")
	if flag {
		for {
		}
	}
}

//go:noinline
func Println(s string) {
	fmt.Println(s)
}

var flag bool
