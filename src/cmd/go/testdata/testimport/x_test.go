package p_test

import (
	. "../testimport"

	"./p2"

	"testing"
)

func TestF1(t *testing.T) {
	if F() != p2.F() {
		t.Fatal(F())
	}
}
