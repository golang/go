package main

import "fmt"

func main() {
	fmt.Println("This is a test program to see package path detection")
	var a int8 = 100
	var b int8 = 20
	result := a + b
	fmt.Printf("100 + 20 = %d\n", result)
}