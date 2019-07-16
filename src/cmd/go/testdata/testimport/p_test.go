package p

import (
	"./p1"

	"testing"
)

func TestF(t *testing.T) {
	if F() != p1.F() {
		t.Fatal(F())
	}
}
