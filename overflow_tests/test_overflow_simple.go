package main

import "fmt"

func main() {
	fmt.Println("Testing overflow detection...")
	
	// This should cause overflow: 127 + 1 = -128 (wraps around without overflow checking)
	var a, b int8 = 127, 1
	result := a + b
	
	fmt.Printf("127 + 1 = %d (should have panicked if overflow detection works)\n", result)
}