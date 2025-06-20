package main

import "fmt"

func main() {
	fmt.Println("Testing type truncation detection...")

	// Test uint16 to uint8 truncation (should panic)
	var u16 uint16 = 256
	fmt.Printf("Before: u16=%d\n", u16)
	
	// This should panic with "integer truncation"
	u8 := uint8(u16)
	
	fmt.Printf("After: u8=%d\n", u8) // Should not reach here
}