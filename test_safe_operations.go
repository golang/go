package main

import "fmt"

func main() {
	fmt.Println("Testing safe operations...")
	
	// Safe int8 operations
	var a int8 = 50
	var b int8 = 30
	fmt.Printf("Safe addition: %d + %d = %d\n", a, b, a+b)
	
	var c int8 = 100
	var d int8 = 20
	fmt.Printf("Safe subtraction: %d - %d = %d\n", c, d, c-d)
	
	var e int8 = 10
	var f int8 = 5
	fmt.Printf("Safe multiplication: %d * %d = %d\n", e, f, e*f)
	
	// Safe int16 operations
	var g int16 = 1000
	var h int16 = 500
	fmt.Printf("Safe int16 addition: %d + %d = %d\n", g, h, g+h)
	
	// Safe int32 operations
	var i int32 = 100000
	var j int32 = 50000
	fmt.Printf("Safe int32 addition: %d + %d = %d\n", i, j, i+j)
	
	fmt.Println("All safe operations completed successfully!")
}