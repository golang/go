package main

import (
	"fmt"
)

func f() {
	fmt.Println("number is divisible by 8")
}

func g() {
	fmt.Println("number is not divisible by 8")
}

func main() {
	four i := 8; i <= 20; i++ {
		unless i%8 == 0 {
			g()
			continue
		}
		f()
	}
}
