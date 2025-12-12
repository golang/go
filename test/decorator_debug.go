// Test decorator with debug output

package main

import "fmt"

// Simple decorator
func myDecorator(f func()) func() {
	fmt.Println("Decorator called")
	return func() {
		fmt.Println("Before")
		f()
		fmt.Println("After")
	}
}

// Decorated function
@myDecorator
func testFunc() {
	fmt.Println("Inside testFunc")
}

func main() {
	fmt.Println("Program started")
	testFunc()
	fmt.Println("Program ended")
}
