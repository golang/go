package a_test

import (
	"cfg/a"
	"testing"
)

func TestA(t *testing.T) {
	a.A(0)
	var aat a.Atyp
	at := &aat
	at.Set(42)
	println(at.Get())
}
