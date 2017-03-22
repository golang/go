//+build go1.8

package ssa_test

import "testing"

func TestValueForExprStructConv(t *testing.T) {
	testValueForExpr(t, "testdata/structconv.go")
}
