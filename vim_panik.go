package main

import (
    "fmt"
)

func main() {
    fmt.Println("=== Integer Overflow Examples in Go ===\n")

    // Example 1: Signed int8 overflow
    fmt.Println("1. Signed int8 overflow:")
    var a int8 = 127  // Maximum value for int8
    fmt.Printf("   Max int8: %d\n", a)
    a = a + 1
    fmt.Printf("   After adding 1: %d (wrapped to minimum)\n\n", a)

}
