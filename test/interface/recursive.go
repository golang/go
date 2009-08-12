// $G $D/$F.go || echo BUG: should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check mutually recursive interfaces

package main

type I1 interface {
	foo() I2
}

type I2 interface {
	bar() I1
}

type T int
func (t T) foo() I2 { return t }
func (t T) bar() I1 { return t }
