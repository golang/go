package cmp_test

import (
	"cmp"
	"slices"
	"strconv"
	"testing"
)

func BenchmarkCompare_int(b *testing.B) {
	var lst [1_000_000]int
	for i := range lst {
		lst[i] = i
	}
	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		slices.SortFunc(lst[:], cmp.Compare)
	}
}

func BenchmarkCompare_float64(b *testing.B) {
	var lst [1_000_000]float64
	for i := range lst {
		lst[i] = float64(i)
	}
	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		slices.SortFunc(lst[:], cmp.Compare)
	}
}

func BenchmarkCompare_string(b *testing.B) {
	var lst [1_000_000]string
	for i := range lst {
		lst[i] = strconv.Itoa(i)
	}
	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		slices.SortFunc(lst[:], cmp.Compare)
	}
}
