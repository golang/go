package main

import "fmt"

func main() {
	fmt.Println("Testing type truncation in module context...")
	
	// Test uint16 to uint8 truncation (should panic)
	var u16 uint16 = 256
	fmt.Printf("Before: u16=%d (0x%X)\n", u16, u16)
	
	// This should panic with "integer truncation"
	u8 := uint8(u16)
	
	fmt.Printf("After: u8=%d (0x%X)\n", u8, u8) // Should not reach here
}