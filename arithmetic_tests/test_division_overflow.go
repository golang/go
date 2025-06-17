package main

import "fmt"

func testInt8Division() {
	var a int8 = -128 // MIN_INT8
	var b int8 = -1
	fmt.Printf("Testing int8: %d / %d\n", a, b)
	result := a / b // Should panic: -128 / -1 = 128, but max int8 is 127
	fmt.Printf("Result: %d\n", result)
}

func testInt16Division() {
	var a int16 = -32768 // MIN_INT16
	var b int16 = -1
	fmt.Printf("Testing int16: %d / %d\n", a, b)
	result := a / b // Should panic: -32768 / -1 = 32768, but max int16 is 32767
	fmt.Printf("Result: %d\n", result)
}

func testInt32Division() {
	var a int32 = -2147483648 // MIN_INT32
	var b int32 = -1
	fmt.Printf("Testing int32: %d / %d\n", a, b)
	result := a / b // Should panic: -2147483648 / -1 = 2147483648, but max int32 is 2147483647
	fmt.Printf("Result: %d\n", result)
}

func main() {
	fmt.Println("Testing division overflow detection...")
	
	// Test int8 division overflow
	testInt8Division()
	
	// Test int16 division overflow  
	testInt16Division()
	
	// Test int32 division overflow
	testInt32Division()
	
	fmt.Println("All tests completed without panic - this means division overflow detection is not implemented yet!")
}