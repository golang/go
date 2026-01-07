package harness

import (
	"testing"
)

func FuzzMe(f *testing.F) {
	f.Add([]byte("1234"))
	f.Fuzz(func(t *testing.T, input []byte) {
		if harness(input) {
			t.Errorf("Found input: %s", input)
		}
	})
}
