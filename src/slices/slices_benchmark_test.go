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

var bsink []byte
var x = []byte{'a', 'b', 'c', 'd', 'e'}

func BenchmarkClone(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bsink = slices.Clone(x)
		bsink = slices.Clone([]byte(nil))
		bsink = slices.Clone[[]byte](nil)
	}
}

func BenchmarkCloneTable1(b *testing.B) {
	for _, tt := range cloneTests {
		b.Run(tt.desc, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = slices.Clone(tt.input)
			}
		})
	}
}

func BenchmarkCloneTable(b *testing.B) {
	for _, tt := range cloneTests {
		b.Run(tt.desc, func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_ = slices.Clone(tt.input)
				}
			})
			//for i := 0; i < b.N; i++ {
			//	b.StartTimer()
			//	_ = slices.Clone(tt.input)
			//	b.StopTimer()
			//}
		})
	}
}
