package main

import "fmt"

func testAddition() {
	fmt.Println("Testing int8 addition overflow...")
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

func testMultiplication() {
	fmt.Println("Testing int8 multiplication overflow...")
	var a int8 = 127
	var b int8 = 2
	result := a * b  // Should panic  
	fmt.Printf("This should not print: %d\n", result)
}

func testInt16() {
	fmt.Println("Testing int16 overflow...")
	var a int16 = 32767
	var b int16 = 1
	result := a + b  // Should panic
	fmt.Printf("This should not print: %d\n", result)
}

func testInt32() {
	fmt.Println("Testing int32 overflow...")
	var a int32 = 2147483647
	var b int32 = 1
	result := a + b  // Should panic
	fmt.Printf("This should not print: %d\n", result)
}

func main() {
	fmt.Println("Testing overflow detection for all operations...")
	testInt32()
}