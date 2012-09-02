// compile

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gobs1

type T struct{ X, Y, Z int } // Only exported fields are encoded and decoded.
var t = T{X: 7, Y: 0, Z: 8}

// STOP OMIT

type U struct{ X, Y *int8 } // Note: pointers to int8s
var u U

// STOP OMIT

type Node struct {
	Value       int
	Left, Right *Node
}

// STOP OMIT
