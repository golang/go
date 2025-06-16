package main

import (
	"fmt"
	"example.com/overflow-test/overflow"
)

func main() {
	fmt.Println("Testing overflow detection in module...")
	
	// This should cause overflow in a subpackage
	result := overflow.TestOverflow()
	
	fmt.Printf("127 + 1 = %d (should have panicked if overflow detection works)\n", result)
}