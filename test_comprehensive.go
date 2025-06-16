package main

import "fmt"

func testSafeInt8() {
	fmt.Println("Testing safe int8 operations...")
	var a int8 = 100
	var b int8 = 20
	result := a + b
	fmt.Printf("Safe: 100 + 20 = %d\n", result)
}

func testOverflowInt8() {
	fmt.Println("Testing int8 overflow...")
	var a int8 = 127
	var b int8 = 1
	result := a + b  // Should panic
	fmt.Printf("This should not print: %d\n", result)
}

func testSubtraction() {
	fmt.Println("Testing int8 subtraction underflow...")
	var a int8 = -128
	var b int8 = 1
	result := a - b  // Should panic
	fmt.Printf("This should not print: %d\n", result)
}

func testInt16() {
	fmt.Println("Testing int16 overflow...")
	var a int16 = 32767
	var b int16 = 1
	result := a + b  // Should panic
	fmt.Printf("This should not print: %d\n", result)
}

func main() {
	// Test safe operation first
	testSafeInt8()
	
	// Test overflow cases (each should panic)
	fmt.Println("\n--- Testing overflow cases (should panic) ---")
	testOverflowInt8()
}