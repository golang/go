package bytes_test

import (
	. "bytes"
	"testing"
)

var compareTests = []struct {
	a, b []byte
	i    int
}{
	{[]byte(""), []byte(""), 0},
	{[]byte("a"), []byte(""), 1},
	{[]byte(""), []byte("a"), -1},
	{[]byte("abc"), []byte("abc"), 0},
	{[]byte("ab"), []byte("abc"), -1},
	{[]byte("abc"), []byte("ab"), 1},
	{[]byte("x"), []byte("ab"), 1},
	{[]byte("ab"), []byte("x"), -1},
	{[]byte("x"), []byte("a"), 1},
	{[]byte("b"), []byte("x"), -1},
	// test runtime·memeq's chunked implementation
	{[]byte("abcdefgh"), []byte("abcdefgh"), 0},
	{[]byte("abcdefghi"), []byte("abcdefghi"), 0},
	{[]byte("abcdefghi"), []byte("abcdefghj"), -1},
	// nil tests
	{nil, nil, 0},
	{[]byte(""), nil, 0},
	{nil, []byte(""), 0},
	{[]byte("a"), nil, 1},
	{nil, []byte("a"), -1},
}

func TestCompare(t *testing.T) {
	for _, tt := range compareTests {
		cmp := Compare(tt.a, tt.b)
		if cmp != tt.i {
			t.Errorf(`Compare(%q, %q) = %v`, tt.a, tt.b, cmp)
		}
	}
}

func TestCompareIdenticalSlice(t *testing.T) {
	var b = []byte("Hello Gophers!")
	if Compare(b, b) != 0 {
		t.Error("b != b")
	}
	if Compare(b, b[:1]) != 1 {
		t.Error("b > b[:1] failed")
	}
}

func TestCompareBytes(t *testing.T) {
	n := 128
	a := make([]byte, n+1)
	b := make([]byte, n+1)
	for len := 0; len < 128; len++ {
		// randomish but deterministic data.  No 0 or 255.
		for i := 0; i < len; i++ {
			a[i] = byte(1 + 31*i%254)
			b[i] = byte(1 + 31*i%254)
		}
		// data past the end is different
		for i := len; i <= n; i++ {
			a[i] = 8
			b[i] = 9
		}
		cmp := Compare(a[:len], b[:len])
		if cmp != 0 {
			t.Errorf(`CompareIdentical(%d) = %d`, len, cmp)
		}
		if len > 0 {
			cmp = Compare(a[:len-1], b[:len])
			if cmp != -1 {
				t.Errorf(`CompareAshorter(%d) = %d`, len, cmp)
			}
			cmp = Compare(a[:len], b[:len-1])
			if cmp != 1 {
				t.Errorf(`CompareBshorter(%d) = %d`, len, cmp)
			}
		}
		for k := 0; k < len; k++ {
			b[k] = a[k] - 1
			cmp = Compare(a[:len], b[:len])
			if cmp != 1 {
				t.Errorf(`CompareAbigger(%d,%d) = %d`, len, k, cmp)
			}
			b[k] = a[k] + 1
			cmp = Compare(a[:len], b[:len])
			if cmp != -1 {
				t.Errorf(`CompareBbigger(%d,%d) = %d`, len, k, cmp)
			}
			b[k] = a[k]
		}
	}
}

func BenchmarkCompareBytesEqual(b *testing.B) {
	b1 := []byte("Hello Gophers!")
	b2 := []byte("Hello Gophers!")
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != 0 {
			b.Fatal("b1 != b2")
		}
	}
}

func BenchmarkCompareBytesToNil(b *testing.B) {
	b1 := []byte("Hello Gophers!")
	var b2 []byte
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != 1 {
			b.Fatal("b1 > b2 failed")
		}
	}
}

func BenchmarkCompareBytesEmpty(b *testing.B) {
	b1 := []byte("")
	b2 := b1
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != 0 {
			b.Fatal("b1 != b2")
		}
	}
}

func BenchmarkCompareBytesIdentical(b *testing.B) {
	b1 := []byte("Hello Gophers!")
	b2 := b1
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != 0 {
			b.Fatal("b1 != b2")
		}
	}
}

func BenchmarkCompareBytesSameLength(b *testing.B) {
	b1 := []byte("Hello Gophers!")
	b2 := []byte("Hello, Gophers")
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != -1 {
			b.Fatal("b1 < b2 failed")
		}
	}
}

func BenchmarkCompareBytesDifferentLength(b *testing.B) {
	b1 := []byte("Hello Gophers!")
	b2 := []byte("Hello, Gophers!")
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != -1 {
			b.Fatal("b1 < b2 failed")
		}
	}
}

func BenchmarkCompareBytesBigUnaligned(b *testing.B) {
	b.StopTimer()
	b1 := make([]byte, 0, 1<<20)
	for len(b1) < 1<<20 {
		b1 = append(b1, "Hello Gophers!"...)
	}
	b2 := append([]byte("hello"), b1...)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2[len("hello"):]) != 0 {
			b.Fatal("b1 != b2")
		}
	}
	b.SetBytes(int64(len(b1)))
}

func BenchmarkCompareBytesBig(b *testing.B) {
	b.StopTimer()
	b1 := make([]byte, 0, 1<<20)
	for len(b1) < 1<<20 {
		b1 = append(b1, "Hello Gophers!"...)
	}
	b2 := append([]byte{}, b1...)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != 0 {
			b.Fatal("b1 != b2")
		}
	}
	b.SetBytes(int64(len(b1)))
}

func BenchmarkCompareBytesBigIdentical(b *testing.B) {
	b.StopTimer()
	b1 := make([]byte, 0, 1<<20)
	for len(b1) < 1<<20 {
		b1 = append(b1, "Hello Gophers!"...)
	}
	b2 := b1
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if Compare(b1, b2) != 0 {
			b.Fatal("b1 != b2")
		}
	}
	b.SetBytes(int64(len(b1)))
}
