package testing_test

import (
	"testing"
)

func TestFuzzAdd(t *testing.T) {
	matchFunc := func(a, b string) (bool, error) { return true, nil }
	tests := []struct {
		name string
		fn   func(f *testing.F)
		ok   bool
	}{
		{
			"empty",
			func(f *testing.F) { f.Add() },
			false,
		},
		{
			"multiple arguments",
			func(f *testing.F) { f.Add([]byte("hello"), []byte("bye")) },
			false,
		},
		{
			"string",
			func(f *testing.F) { f.Add("hello") },
			false,
		},
		{
			"bytes",
			func(f *testing.F) { f.Add([]byte("hello")) },
			true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, want := testing.RunFuzzTargets(matchFunc, []testing.InternalFuzzTarget{{Fn: tc.fn}}), tc.ok; got != want {
				t.Errorf("testing.Add: ok %t, want %t", got, want)
			}
		})
	}
}
