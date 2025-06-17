package main

import (
	"fmt"
	"github.com/thirdparty/lib"
)

func main() {
	fmt.Println("Testing vendor package (should NOT trigger overflow)...")
	
	// This should not panic because vendor packages are excluded
	result := lib.UnsafeOperation()
	fmt.Printf("Vendor package result: %d\n", result)
	
	fmt.Println("Vendor test completed successfully!")
}