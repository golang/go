package harness

import (
	"testing"
)

func FuzzMe(f *testing.F) {
	f.Add([]byte("1234"))
	f.Fuzz(func(t *testing.T, input []byte) {
		harness(input)
	})
}
