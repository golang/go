package fuzztest
import "testing"

func FuzzAdd(f *testing.F) {
	f.Fuzz(func(t *testing.T, a, b int8) {
		result := a + b
		t.Logf("%d + %d = %d", a, b, result)
	})
}