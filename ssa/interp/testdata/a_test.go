package a

import "testing"

func TestFoo(t *testing.T) {
	t.Error("foo")
}

func TestBar(t *testing.T) {
	t.Error("bar")
}

func BenchmarkWiz(b *testing.B) {
	b.Error("wiz")
}

// Don't test Examples since that testing package needs pipe(2) for that.
