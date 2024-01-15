package slices_test

import (
	"testing"
)

type CloneFuncType[S ~[]E, E any] func(s S) S

func Clone0[S ~[]E, E any](s S) S {
	return append(s[:0:0], s...)
}

func Clone1[S ~[]E, E any](s S) S {
	if s == nil {
		return nil
	}
	clone := make(S, 0, len(s))
	copy(clone, s)
	return clone
}

type CloneFuncTable[S ~[]E, E any] struct {
	fn   CloneFuncType[S, E]
	name string
}

func BenchmarkCloneX(b *testing.B) {
	//b.ReportAllocs()
	var cloneFuncs = []CloneFuncTable[[]byte, byte]{
		{Clone0[[]byte, byte], "Clone0"},
		{Clone1[[]byte, byte], "Clone1"},
	}
	for _, CloneX := range cloneFuncs {
		b.Run(CloneX.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = CloneX.fn(x)
				_ = CloneX.fn(nil)
				_ = CloneX.fn(nil)
				_ = CloneX.fn(nil)
				_ = CloneX.fn([]byte(nil))
				for _, tt := range cloneTests {
					b.Run(tt.desc, func(b *testing.B) {
						b.ReportAllocs()

						for i := 0; i < b.N; i++ {
							_ = CloneX.fn(tt.input)
						}
					})
				}
			}
		})
	}
}
