package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("Testing runtime overflow detection...")
	
	// Get values from command line args to prevent constant folding
	var a int8 = 127
	if len(os.Args) > 1 {
		a = 126 // still high enough to overflow
	}
	var b int8 = 2
	
	fmt.Printf("Before: a=%d, b=%d\n", a, b)
	result := a + b  // Should panic with "integer overflow"
	fmt.Printf("After: result=%d\n", result)
}