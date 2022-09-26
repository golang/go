package a_test

import (
	"testing"
)

func TestA2(t *testing.T) { //@TestA2,godef(TestA2, TestA2)
	Nonexistant() //@diag("Nonexistant", "compiler", "(undeclared name|undefined): Nonexistant", "error")
}
