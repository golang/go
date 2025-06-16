package main

import (
	"fmt"
	"github.com/example/realworld-app/pkg/math"
	"github.com/example/realworld-app/internal/utils"
)

func main() {
	fmt.Println("Testing real-world package overflow detection...")
	
	// This should trigger overflow detection
	result1 := math.AddInt8(127, 1)
	fmt.Printf("Math package result: %d\n", result1)
	
	// This should also trigger overflow detection  
	result2 := utils.ProcessData(-128, 1)
	fmt.Printf("Utils package result: %d\n", result2)
}