package main

import (
	"reflect"
	"testing"
)

func TestForwardArgs(t *testing.T) {
	for _, c := range []struct {
		command string
		in, out []string
	}{
		{"run", []string{".", "a.gox", "--"}, []string{".", "a.gox", "--"}},
		{"run", []string{"-tags", "demo", "a.gox", "b.go", "--", "x.gox", "--"}, []string{"-tags", "demo", "a.go", "b.go", "x.gox", "--"}},
		{"run", []string{"a.gox", "hello", "x.gox"}, []string{"a.go", "hello", "x.gox"}},
		{"build", []string{"-o", "output.gox", "a.gox"}, []string{"-o", "output.gox", "a.go"}},
	} {
		if got := forwardArgs(c.command, c.in); !reflect.DeepEqual(got, c.out) {
			t.Errorf("%v: got %v want %v", c.in, got, c.out)
		}
	}
}
