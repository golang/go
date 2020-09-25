package tls_test

import (
	"testing"
)

func TestCloneNilConfig(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatal("Clone a nil Config should not produce panic")
		}
	}()

	if cc := c.Clone(); cc != nil {
		t.Fatalf("Clone with nil should return nil, got: %+v", cc)
	}
}
