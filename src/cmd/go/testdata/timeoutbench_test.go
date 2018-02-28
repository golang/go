package timeoutbench_test

import (
	"testing"
	"time"
)

func BenchmarkSleep1s(b *testing.B) {
	time.Sleep(1 * time.Second)
}
