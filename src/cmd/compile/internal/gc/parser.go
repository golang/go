// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// The recursive-descent parser is built around a slighty modified grammar
// of Go to accommodate for the constraints imposed by strict one token look-
// ahead, and for better error handling. Subsequent checks of the constructed
// syntax tree restrict the language accepted by the compiler to proper Go.
//
// Semicolons are inserted by the lexer. The parser uses one-token look-ahead
// to handle optional commas and semicolons before a closing ) or } .

const trace = false // if set, parse tracing can be enabled with -x

func mkname(sym *Sym) *Node {
	n := oldname(sym)
	if n.Name != nil && n.Name.Pack != nil {
		n.Name.Pack.Used = true
	}
	return n
}

func unparen(x *Node) *Node {
	for x.Op == OPAREN {
		x = x.Left
	}
	return x
}
