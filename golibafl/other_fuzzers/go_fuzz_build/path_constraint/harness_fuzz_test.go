//go:build gofuzz

package harness

import (
	"github.com/AdamKorcz/go-118-fuzz-build/testing"
)

func FuzzMe(f *testing.F) {
	f.Fuzz(func(t *testing.T, input []byte) {
		if harness(input) {
			panic("crash")
			//t.Fatalf("Found input: %s", input)
		}
	})
}
