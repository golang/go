//go:build gofuzz
package harness

import (
	"github.com/AdamKorcz/go-118-fuzz-build/testing"
)

func FuzzMe(f *testing.F) {
	f.Add([]byte("hello"))
	f.Fuzz(func(t *testing.T, input []byte) {
		harness(input)
	})
}
