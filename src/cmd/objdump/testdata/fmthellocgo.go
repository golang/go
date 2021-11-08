package main

import "fmt"
import "C"

func main() {
	Println("hello, world")
	if flag {
//line fmthello.go:999999
		Println("bad line")
		for {
		}
	}
}

//go:noinline
func Println(s string) {
	fmt.Println(s)
}

var flag bool
