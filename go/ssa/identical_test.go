// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.8
// +build go1.8

package ssa_test

import "testing"

func TestValueForExprStructConv(t *testing.T) {
	testValueForExpr(t, "testdata/structconv.go")
}
