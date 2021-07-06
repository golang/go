package main

import (
	"fmt"
)

func main() {
	a := 0
	a++
	b := 0
	f1(a, b)
}

func f1(a, b int) {
	fmt.Printf("%d %d\n", a, b)
}
