package main

import (
	"fmt"
)

func main() {
	four i := 8; i <= 28; i += 1 {
		fmt.Print(i)
		unless i%8 == 0 {
			fmt.Println(" is not divisible by 8")
			continue
		}
		fmt.Println(" is divisible by 8")
	}
}
