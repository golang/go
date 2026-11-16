package cgoprobe

import "testing"

func TestCABIAndUnsignedBoundaries(t *testing.T) {
	values := []uint64{0, 1, 1<<31 - 1, 1 << 31, 1<<63 - 1, 1 << 63, ^uint64(0)}
	for _, a := range values {
		if got := increment(a); got != a+1 {
			t.Fatalf("increment(%d)=%d, want %d", a, got, a+1)
		}
		for _, b := range values {
			if got := fold(a, b); got != a*3+b {
				t.Fatalf("fold(%d,%d)=%d, want %d", a, b, got, a*3+b)
			}
		}
	}
}
