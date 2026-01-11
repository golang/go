package main

import "testing"

func FuzzLoggerError(f *testing.F) {
	f.Add("seed")
	f.Fuzz(func(t *testing.T, s string) {
		_, _ = t, s
		(&Logger{}).Error()
	})
}

