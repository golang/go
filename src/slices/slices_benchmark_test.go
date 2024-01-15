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
		desc:  "abcde",
	},
	{
		input: []byte("abcdefghijklmnopqrstuvwxyz"),
		desc:  "abcdefghijklmnopqrstuvwxyz",
	},
	{
		input: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 0},
		desc:  "1234567890",
	},
	{
		[]byte("hello world"),
		"hello world",
	},
	{
		input: nil,
		desc:  "empty slice",
	},
	{
		input: []byte(" "),
		desc:  "empty slice1",
	},
	{
		input: []byte(""),
		desc:  "empty slice2",
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

func BenchmarkCloneRealLife(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a := Clone0(x)
		_ = append(a, 'h')
	}
}

func BenchmarkClone0RealLife(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a := Clone0(x)
		_ = append(a, 'h')
	}
}

func BenchmarkClone1RealLife(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a := Clone1(x)
		_ = append(a, 'h')
	}
}
