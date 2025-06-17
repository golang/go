package main

import (
	"fmt"
	"testing"
)

func FuzzIntegerOverflow(f *testing.F) {
	f.Fuzz(func(t *testing.T, a, b int8) {
		result := checkedAdd8(a, b)
		fmt.Println("Testing:", a, "+", b, "=", result)
	})
}

func checkedAdd8(a, b int8) int8 {
	return a + b
}
