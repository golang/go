package main

import (
	"slices"
	"testing"
)

func Reverse(s string) string {
	runes := []rune(s)
	slices.Reverse(runes)
	return string(runes)
}

// This fuzz test should quickly fail, because Reverse doesn't
// work for strings that are not valid UTF-8.
// What we are testing for is whether we see a failure from ASAN;
// we should see a fuzzing failure, not an ASAN failure.

func FuzzReverse(f *testing.F) {
	f.Add("Go")
	f.Add("Gopher")
	f.Add("Hello, 世界")
	f.Fuzz(func(t *testing.T, s string) {
		r1 := Reverse(s)
		r2 := Reverse(r1)
		if s != r2 {
			t.Errorf("FUZZ FAILED: got %q want %q", r2, s)
		}
	})
}
