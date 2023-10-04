package cmp_test

import (
	"cmp"
	"math"
	"testing"
)

var sum int

func BenchmarkCompare_int(b *testing.B) {
	lst := [...]int{
		0xfa3, 0x7fe, 0x03c, 0xcb9,
		0x4ce, 0x4fb, 0x7d5, 0x38f,
		0x73b, 0x322, 0x85c, 0xf4d,
		0xbbc, 0x032, 0x059, 0xb93,
	}
	for n := 0; n < b.N; n++ {
		sum += cmp.Compare(lst[n%len(lst)], lst[(2*n)%len(lst)])
	}
}

func BenchmarkCompare_float64(b *testing.B) {
	lst := [...]float64{
		0.35573281, 0.77552566, 0.19006500, 0.66436280,
		0.02769279, 0.97572397, 0.40945068, 0.26422857,
		0.10985792, 0.35659522, 0.82752613, 0.18875522,
		0.16410543, 0.03578153, 0.51636871, math.NaN(),
	}
	for n := 0; n < b.N; n++ {
		sum += cmp.Compare(lst[n%len(lst)], lst[(2*n)%len(lst)])
	}
}

func BenchmarkCompare_strings(b *testing.B) {
	lst := [...]string{
		"time",
		"person",
		"year",
		"way",
		"day",
		"thing",
		"man",
		"world",
		"life",
		"hand",
		"part",
		"child",
		"eye",
		"woman",
		"place",
		"work",
	}
	for n := 0; n < b.N; n++ {
		sum += cmp.Compare(lst[n%len(lst)], lst[(2*n)%len(lst)])
	}
}
