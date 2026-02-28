// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import "cmd/internal/src"

// InlTree is a collection of inlined calls. The Parent field of an
// InlinedCall is the index of another InlinedCall in InlTree.
//
// The compiler maintains a global inlining tree and adds a node to it
// every time a function is inlined. For example, suppose f() calls g()
// and g has two calls to h(), and that f, g, and h are inlineable:
//
//	 1 func main() {
//	 2     f()
//	 3 }
//	 4 func f() {
//	 5     g()
//	 6 }
//	 7 func g() {
//	 8     h()
//	 9     h()
//	10 }
//	11 func h() {
//	12     println("H")
//	13 }
//
// Assuming the global tree starts empty, inlining will produce the
// following tree:
//
//	[]InlinedCall{
//	  {Parent: -1, Func: "f", Pos: <line 2>},
//	  {Parent:  0, Func: "g", Pos: <line 5>},
//	  {Parent:  1, Func: "h", Pos: <line 8>},
//	  {Parent:  1, Func: "h", Pos: <line 9>},
//	}
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
	Parent   int      // index of the parent in the InlTree or < 0 if outermost call
	Pos      src.XPos // position of the inlined call
	Func     *LSym    // function that was inlined
	ParentPC int32    // PC of instruction just before inlined body. Only valid in local trees.
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

func (tree *InlTree) Parent(inlIndex int) int {
	return tree.nodes[inlIndex].Parent
}

func (tree *InlTree) InlinedFunction(inlIndex int) *LSym {
	return tree.nodes[inlIndex].Func
}

func (tree *InlTree) CallPos(inlIndex int) src.XPos {
	return tree.nodes[inlIndex].Pos
}

func (tree *InlTree) setParentPC(inlIndex int, pc int32) {
	tree.nodes[inlIndex].ParentPC = pc
}

// OutermostPos returns the outermost position corresponding to xpos,
// which is where xpos was ultimately inlined to. In the example for
// InlTree, main() contains inlined AST nodes from h(), but the
// outermost position for those nodes is line 2.
func (ctxt *Link) OutermostPos(xpos src.XPos) src.Pos {
	pos := ctxt.InnermostPos(xpos)

	outerxpos := xpos
	for ix := pos.Base().InliningIndex(); ix >= 0; {
		call := ctxt.InlTree.nodes[ix]
		ix = call.Parent
		outerxpos = call.Pos
	}
	return ctxt.PosTable.Pos(outerxpos)
}

// InnermostPos returns the innermost position corresponding to xpos,
// that is, the code that is inlined and that inlines nothing else.
// In the example for InlTree above, the code for println within h
// would have an innermost position with line number 12, whether
// h was not inlined, inlined into g, g-then-f, or g-then-f-then-main.
// This corresponds to what someone debugging main, f, g, or h might
// expect to see while single-stepping.
func (ctxt *Link) InnermostPos(xpos src.XPos) src.Pos {
	return ctxt.PosTable.Pos(xpos)
}

// AllPos returns a slice of the positions inlined at xpos, from
// innermost (index zero) to outermost.  To avoid gratuitous allocation
// the result is passed in and extended if necessary.
func (ctxt *Link) AllPos(xpos src.XPos, result []src.Pos) []src.Pos {
	pos := ctxt.InnermostPos(xpos)
	result = result[:0]
	result = append(result, ctxt.PosTable.Pos(xpos))
	for ix := pos.Base().InliningIndex(); ix >= 0; {
		call := ctxt.InlTree.nodes[ix]
		ix = call.Parent
		result = append(result, ctxt.PosTable.Pos(call.Pos))
	}
	return result
}

func dumpInlTree(ctxt *Link, tree InlTree) {
	for i, call := range tree.nodes {
		pos := ctxt.PosTable.Pos(call.Pos)
		ctxt.Logf("%0d | %0d | %s (%s) pc=%d\n", i, call.Parent, call.Func, pos, call.ParentPC)
	}
}
