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
	clone := make(S, len(s))
	copy(clone, s)
	return clone
}

type CloneFuncTypeS[S ~[]E, E any] struct {
	fn   CloneFuncType[S, E]
	name string
}

var cloneFuncs = []CloneFuncTypeS[[]byte, byte]{
	{Clone0[[]byte, byte], "Clone0"},
	{Clone1[[]byte, byte], "Clone1"},
}

func BenchmarkClones(b *testing.B) {
	b.ReportAllocs()
	x := []byte("some bytes")
	for _, CloneX := range cloneFuncs {
		b.Run(CloneX.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = CloneX.fn(x)
				_ = CloneX.fn([]byte(nil))
			}
		})
	}
}
