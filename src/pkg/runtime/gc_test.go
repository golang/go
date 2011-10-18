package runtime_test

import (
	"runtime"
	"testing"
)

func TestGcSys(t *testing.T) {
	for i := 0; i < 1000000; i++ {
		workthegc()
	}

	// Should only be using a few MB.
	runtime.UpdateMemStats()
	sys := runtime.MemStats.Sys
	t.Logf("using %d MB", sys>>20)
	if sys > 10e6 {
		t.Fatalf("using too much memory: %d MB", sys>>20)
	}
}

func workthegc() []byte {
	return make([]byte, 1029)
}
