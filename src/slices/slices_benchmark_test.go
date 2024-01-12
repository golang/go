package slices_test

import (
	"slices"
	"testing"
)

var bsink []byte

func BenchmarkClone(b *testing.B) {
	b.ReportAllocs()
	x := []byte{'a', 'b', 'c', 'd', 'e'}
	for i := 0; i < b.N; i++ {
		bsink = slices.Clone(x)
		bsink = slices.Clone([]byte(nil))
	}
}
