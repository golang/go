package x

import "testing"

func TestZ(t *testing.T) {
	t.Logf("LOG: Z running")
}

func TestXX(t *testing.T) {
	t.Logf("LOG: XX running")
}

func BenchmarkZ(b *testing.B) {
	b.Logf("LOG: Z running N=%d", b.N)
}

func BenchmarkXX(b *testing.B) {
	b.Logf("LOG: XX running N=%d", b.N)
}
