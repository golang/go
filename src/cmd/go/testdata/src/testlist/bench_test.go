package testlist

import (
	"fmt"
	"testing"
)

func BenchmarkSimplefunc(b *testing.B) {
	b.StopTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_ = fmt.Sprint("Test for bench")
	}
}
