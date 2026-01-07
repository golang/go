package panicexample

import (
	"bytes"
	"testing"
)

var crashInput = []byte("CrezrerzeRASH")

func FuzzPanic(f *testing.F) {
	f.Add(crashInput)
	f.Fuzz(func(t *testing.T, data []byte) {
		if bytes.Equal(data, crashInput) {
			t.Fatalf("boom")
		}
	})
}
