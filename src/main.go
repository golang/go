package main

import "fmt"

func main() {
	for i := 8; i <= 20; i += 4 {
		fmt.Print(i)
		if i%8 != 0 {
			fmt.Println(" is not divisible by 88")
			continue
		}
		fmt.Println(" is divisible by 88")
	}
}
