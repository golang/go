package x

import "testing"

func TestX(t *testing.T) {
	t.Logf("LOG: X running")
	t.Run("Y", func(t *testing.T) {
		t.Logf("LOG: Y running")
	})
}

func BenchmarkX(b *testing.B) {
	b.Logf("LOG: X running N=%d", b.N)
	b.Run("Y", func(b *testing.B) {
		b.Logf("LOG: Y running N=%d", b.N)
	})
}
