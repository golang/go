package b

import "testing"

func NotATest(t *testing.T) {
	t.Error("foo")
}

func NotABenchmark(b *testing.B) {
	b.Error("wiz")
}
