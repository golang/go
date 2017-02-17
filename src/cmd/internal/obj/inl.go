// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import "cmd/internal/src"

// InlTree s a collection of inlined calls. The Parent field of an
// InlinedCall is the index of another InlinedCall in InlTree.
//
// The compiler maintains a global inlining tree and adds a node to it
// every time a function is inlined. For example, suppose f() calls g()
// and g has two calls to h(), and that f, g, and h are inlineable:
//
//  1 func main() {
//  2     f()
//  3 }
//  4 func f() {
//  5     g()
//  6 }
//  7 func g() {
//  8     h()
//  9     h()
// 10 }
//
// Assuming the global tree starts empty, inlining will produce the
// following tree:
//
//   []InlinedCall{
//     {Parent: -1, Func: "f", Pos: <line 2>},
//     {Parent:  0, Func: "g", Pos: <line 5>},
//     {Parent:  1, Func: "h", Pos: <line 8>},
//     {Parent:  1, Func: "h", Pos: <line 9>},
//   }
//
// The nodes of h inlined into main will have inlining indexes 2 and 3.
//
// Eventually, the compiler extracts a per-function inlining tree from
// the global inlining tree (see pcln.go).
type InlTree struct {
	nodes []InlinedCall
}

// InlinedCall is a node in an InlTree.
type InlinedCall struct {
	Parent int      // index of the parent in the InlTree or < 0 if outermost call
	Pos    src.XPos // position of the inlined call
	Func   *LSym    // function that was inlined
}

// Add adds a new call to the tree, returning its index.
func (tree *InlTree) Add(parent int, pos src.XPos, func_ *LSym) int {
	r := len(tree.nodes)
	call := InlinedCall{
		Parent: parent,
		Pos:    pos,
		Func:   func_,
	}
	tree.nodes = append(tree.nodes, call)
	return r
}

// OutermostPos returns the outermost position corresponding to xpos,
// which is where xpos was ultimately inlined to. In the example for
// InlTree, main() contains inlined AST nodes from h(), but the
// outermost position for those nodes is line 2.
func (ctxt *Link) OutermostPos(xpos src.XPos) src.Pos {
	pos := ctxt.PosTable.Pos(xpos)

	outerxpos := xpos
	for ix := pos.Base().InliningIndex(); ix >= 0; {
		call := ctxt.InlTree.nodes[ix]
		ix = call.Parent
		outerxpos = call.Pos
	}
	return ctxt.PosTable.Pos(outerxpos)
}

func dumpInlTree(ctxt *Link, tree InlTree) {
	for i, call := range tree.nodes {
		pos := ctxt.PosTable.Pos(call.Pos)
		ctxt.Logf("%0d | %0d | %s (%s)\n", i, call.Parent, call.Func, pos)
	}
}
