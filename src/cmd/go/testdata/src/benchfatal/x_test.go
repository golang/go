package benchfatal

import "testing"

func BenchmarkThatCallsFatal(b *testing.B) {
	b.Fatal("called by benchmark")
}
