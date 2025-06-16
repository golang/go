package main

import "fmt"

func main() {
	fmt.Println("Testing overflow detection...")
	
	// Test int8 addition overflow
	var a int8 = 127
	var b int8 = 1
	fmt.Printf("Before: a=%d, b=%d\n", a, b)
	result := a + b  // Should panic with "integer overflow"
	fmt.Printf("After: result=%d\n", result)
}