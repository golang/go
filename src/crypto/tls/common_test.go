package tls_test

import (
	"crypto/tls"
	"testing"
)

func TestCloneNilConfig(t *testing.T) {
	defer func() {
		r := recover()
		if r != nil {
			t.Fatal("Clone a nil Config should not produce panic")
		}
	}()

	var c *tls.Config
	if c.Clone() != nil {
		t.Fatal("Clone a nil Config should output nil")
	}
}
