package slices_test

import (
	"slices"
	"testing"
)

type cloneTest struct {
	input []byte
	desc  string
}

var cloneTests = []cloneTest{
	{
		input: []byte{'a', 'b', 'c', 'd', 'e'},
		desc:  "non-empty slice",
	},
	{
		input: []byte(nil),
		desc:  "empty slice",
	},
}

func BenchmarkClone(b *testing.B) {
	for _, tt := range cloneTests {
		b.Run(tt.desc, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StartTimer()
				_ = slices.Clone(tt.input)
				b.StopTimer()
			}
		})
	}
}
