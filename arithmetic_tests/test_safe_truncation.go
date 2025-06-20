package main

import "fmt"

func main() {
	fmt.Println("Testing safe type conversion (no truncation)...")

	// Test uint16 to uint8 conversion that should NOT panic (value fits in target type)
	var u16 uint16 = 255
	fmt.Printf("Before: u16=%d\n", u16)
	
	// This should NOT panic because 255 fits in uint8
	u8 := uint8(u16)
	
	fmt.Printf("After: u8=%d\n", u8) // Should reach here successfully
	fmt.Println("Safe conversion completed successfully!")
}