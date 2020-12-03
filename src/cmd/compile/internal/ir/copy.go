// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/internal/src"
)

// A Node may implement the Orig and SetOrig method to
// maintain a pointer to the "unrewritten" form of a Node.
// If a Node does not implement OrigNode, it is its own Orig.
//
// Note that both SepCopy and Copy have definitions compatible
// with a Node that does not implement OrigNode: such a Node
// is its own Orig, and in that case, that's what both want to return
// anyway (SepCopy unconditionally, and Copy only when the input
// is its own Orig as well, but if the output does not implement
// OrigNode, then neither does the input, making the condition true).
type OrigNode interface {
	Node
	Orig() Node
	SetOrig(Node)
}

// Orig returns the “original” node for n.
// If n implements OrigNode, Orig returns n.Orig().
// Otherwise Orig returns n itself.
func Orig(n Node) Node {
	if n, ok := n.(OrigNode); ok {
		o := n.Orig()
		if o == nil {
			Dump("Orig nil", n)
			base.Fatalf("Orig returned nil")
		}
		return o
	}
	return n
}

// SepCopy returns a separate shallow copy of n,
// breaking any Orig link to any other nodes.
func SepCopy(n Node) Node {
	n = n.rawCopy()
	if n, ok := n.(OrigNode); ok {
		n.SetOrig(n)
	}
	return n
}

// Copy returns a shallow copy of n.
// If Orig(n) == n, then Orig(Copy(n)) == the copy.
// Otherwise the Orig link is preserved as well.
//
// The specific semantics surrounding Orig are subtle but right for most uses.
// See issues #26855 and #27765 for pitfalls.
func Copy(n Node) Node {
	copy := n.rawCopy()
	if n, ok := n.(OrigNode); ok && n.Orig() == n {
		copy.(OrigNode).SetOrig(copy)
	}

	// Copy lists so that updates to n.List[0]
	// don't affect copy.List[0] and vice versa,
	// same as updates to Left and Right.
	// TODO(rsc): Eventually the Node implementations will need to do this.
	if l := copy.List(); l.Len() > 0 {
		copy.SetList(copyList(l))
	}
	if l := copy.Rlist(); l.Len() > 0 {
		copy.SetRlist(copyList(l))
	}
	if l := copy.Init(); l.Len() > 0 {
		copy.SetInit(copyList(l))
	}
	if l := copy.Body(); l.Len() > 0 {
		copy.SetBody(copyList(l))
	}

	return copy
}

func copyList(x Nodes) Nodes {
	out := make([]Node, x.Len())
	copy(out, x.Slice())
	return AsNodes(out)
}

// A Node can implement DeepCopyNode to provide a custom implementation
// of DeepCopy. If the compiler only needs access to a Node's structure during
// DeepCopy, then a Node can implement DeepCopyNode instead of providing
// fine-grained mutable access with Left, SetLeft, Right, SetRight, and so on.
type DeepCopyNode interface {
	Node
	DeepCopy(pos src.XPos) Node
}

// DeepCopy returns a “deep” copy of n, with its entire structure copied
// (except for shared nodes like ONAME, ONONAME, OLITERAL, and OTYPE).
// If pos.IsKnown(), it sets the source position of newly allocated Nodes to pos.
//
// The default implementation is to traverse the Node graph, making
// a shallow copy of each node and then updating each field to point
// at shallow copies of children, recursively, using Left, SetLeft, and so on.
//
// If a Node wishes to provide an alternate implementation, it can
// implement a DeepCopy method: see the DeepCopyNode interface.
//
// TODO(rsc): Once Nodes implement EditChildren, remove the DeepCopyNode interface.
func DeepCopy(pos src.XPos, n Node) Node {
	var edit func(Node) Node
	edit = func(x Node) Node {
		if x, ok := x.(DeepCopyNode); ok {
			return x.DeepCopy(pos)
		}
		switch x.Op() {
		case OPACK, ONAME, ONONAME, OLITERAL, ONIL, OTYPE:
			return x
		}
		x = Copy(x)
		if pos.IsKnown() {
			x.SetPos(pos)
		}
		EditChildren(x, edit)
		return x
	}
	return edit(n)
}

// DeepCopyList returns a list of deep copies (using DeepCopy) of the nodes in list.
func DeepCopyList(pos src.XPos, list []Node) []Node {
	var out []Node
	for _, n := range list {
		out = append(out, DeepCopy(pos, n))
	}
	return out
}
