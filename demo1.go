package main

import (
	"fmt"
)

func main() {
	for i := 8; i <= 20; i += 4 {
		if !(i%8 == 0) {
			fmt.Println(i, "is not divisible by 8")
			continue
		}
		fmt.Println(i, "is divisible by 8")
	}
}
