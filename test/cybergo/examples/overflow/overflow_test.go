package overflow

import "testing"

func FuzzUint8Overflow(f *testing.F) {
	// Trigger cybergo's integer overflow instrumentation.
	f.Fuzz(func(t *testing.T, a, b uint8) {
		_ = a + b
	})
}
