package reverse

import (
	"testing"
	"unicode/utf8"
)

func FuzzReverse(f *testing.F) {
	f.Add([]byte("Hello, world"))
	f.Add([]byte(" "))
	f.Add([]byte("!12345"))
	f.Add([]byte("AAAAAAAA"))

	f.Fuzz(func(t *testing.T, data []byte) {
		s := string(data)
		if !utf8.ValidString(s) {
			return
		}

		rev := Reverse(s)
		if s != Reverse(rev) {
			t.Fatalf("double reverse mismatch: %q -> %q", s, rev)
		}

		if s == "FUZZING!" {
			t.Fatalf("found magic input")
		}
	})
}
