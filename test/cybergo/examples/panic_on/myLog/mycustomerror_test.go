package myLog

import "testing"

func FuzzMyCustomError(f *testing.F) {
	f.Add("seed")
	f.Fuzz(func(t *testing.T, s string) {
		_, _ = t, s
		_ = myCustomError()
	})
}
