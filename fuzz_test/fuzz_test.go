package fuzztest

import (
	"testing"
)

func FuzzIntegerOverflow(f *testing.F) {
	// Add seed values that will cause overflow
	f.Add(int8(127), int8(1))    // Max + 1 = overflow
	f.Add(int8(-128), int8(-1))  // Min - 1 = underflow
	f.Add(int8(100), int8(50))   // Large values
	
	f.Fuzz(func(t *testing.T, a, b int8) {
		result := a + b
		t.Logf("Testing: %d + %d = %d", a, b, result)
	})
}
