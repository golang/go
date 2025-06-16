package main

import "fmt"

func testAdd() {
	fmt.Println("Testing addition overflow...")
	var a, b int8 = 127, 1
	result := a + b // Should overflow
	fmt.Printf("127 + 1 = %d\n", result)
}

func testSub() {
	fmt.Println("Testing subtraction overflow...")
	var a, b int8 = -128, 1
	result := a - b // Should overflow
	fmt.Printf("-128 - 1 = %d\n", result)
}

func testMul() {
	fmt.Println("Testing multiplication overflow...")
	var a, b int8 = 64, 2
	result := a * b // Should overflow
	fmt.Printf("64 * 2 = %d\n", result)
}

func main() {
	testAdd()
	testSub() 
	testMul()
}