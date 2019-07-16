package xtestonly_test

import (
	"testing"
	"xtestonly"
)

func TestF(t *testing.T) {
	if x := xtestonly.F(); x != 42 {
		t.Errorf("f.F() = %d, want 42", x)
	}
}
