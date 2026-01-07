package multiparams

import (
	"bytes"
	"testing"
)

func FuzzMultiParams(f *testing.F) {
	// A near-miss seed so the fuzzer quickly exercises the full comparison chain
	// (bool + uint16 + string + bytes) instead of starting from entirely random bytes.
	f.Add([]byte("LIBAFL"), "cybergo", uint16(31337), true)
	f.Fuzz(func(t *testing.T, data []byte, s string, n uint16, ok bool) {
		if ok && n == 31337 && s == "cybergo" && bytes.Equal(data, []byte("libafl")) {
			t.Fatalf("boom")
		}
	})
}
