package unimported_test

import (
	"testing"
)

func TestSomething(t *testing.T) {
	_ = unimported.TestExport //@unimported("TestExport", testexport)
}
