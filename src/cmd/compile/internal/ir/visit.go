// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IR visitors for walking the IR tree.
//
// The lowest level helpers are DoChildren and EditChildren, which
// nodes help implement and provide control over whether and when
// recursion happens during the walk of the IR.
//
// Although these are both useful directly, two simpler patterns
// are fairly common and also provided: Visit and Any.

package ir

// DoChildren calls do(x) on each of n's non-nil child nodes x.
// If any call returns true, DoChildren stops and returns true.
// Otherwise, DoChildren returns false.
//
// Note that DoChildren(n, do) only calls do(x) for n's immediate children.
// If x's children should be processed, then do(x) must call DoChildren(x, do).
//
// DoChildren allows constructing general traversals of the IR graph
// that can stop early if needed. The most general usage is:
//
//	var do func(ir.Node) bool
//	do = func(x ir.Node) bool {
//		... processing BEFORE visiting children ...
//		if ... should visit children ... {
//			ir.DoChildren(x, do)
//			... processing AFTER visiting children ...
//		}
//		if ... should stop parent DoChildren call from visiting siblings ... {
//			return true
//		}
//		return false
//	}
//	do(root)
//
// Since DoChildren does not return true itself, if the do function
// never wants to stop the traversal, it can assume that DoChildren
// itself will always return false, simplifying to:
//
//	var do func(ir.Node) bool
//	do = func(x ir.Node) bool {
//		... processing BEFORE visiting children ...
//		if ... should visit children ... {
//			ir.DoChildren(x, do)
//		}
//		... processing AFTER visiting children ...
//		return false
//	}
//	do(root)
//
// The Visit function illustrates a further simplification of the pattern,
// only processing before visiting children and never stopping:
//
//	func Visit(n ir.Node, visit func(ir.Node)) {
//		if n == nil {
//			return
//		}
//		var do func(ir.Node) bool
//		do = func(x ir.Node) bool {
//			visit(x)
//			return ir.DoChildren(x, do)
//		}
//		do(n)
//	}
//
// The Any function illustrates a different simplification of the pattern,
// visiting each node and then its children, recursively, until finding
// a node x for which cond(x) returns true, at which point the entire
// traversal stops and returns true.
//
//	func Any(n ir.Node, cond(ir.Node) bool) bool {
//		if n == nil {
//			return false
//		}
//		var do func(ir.Node) bool
//		do = func(x ir.Node) bool {
//			return cond(x) || ir.DoChildren(x, do)
//		}
//		return do(n)
//	}
//
// Visit and Any are presented above as examples of how to use
// DoChildren effectively, but of course, usage that fits within the
// simplifications captured by Visit or Any will be best served
// by directly calling the ones provided by this package.
func DoChildren(n Node, do func(Node) bool) bool {
	if n == nil {
		return false
	}
	return n.doChildren(do)
}

// Visit visits each non-nil node x in the IR tree rooted at n
// in a depth-first preorder traversal, calling visit on each node visited.
func Visit(n Node, visit func(Node)) {
	if n == nil {
		return
	}
	var do func(Node) bool
	do = func(x Node) bool {
		visit(x)
		return DoChildren(x, do)
	}
	do(n)
}

// VisitList calls Visit(x, visit) for each node x in the list.
func VisitList(list Nodes, visit func(Node)) {
	for _, x := range list {
		Visit(x, visit)
	}
}

// Any looks for a non-nil node x in the IR tree rooted at n
// for which cond(x) returns true.
// Any considers nodes in a depth-first, preorder traversal.
// When Any finds a node x such that cond(x) is true,
// Any ends the traversal and returns true immediately.
// Otherwise Any returns false after completing the entire traversal.
func Any(n Node, cond func(Node) bool) bool {
	if n == nil {
		return false
	}
	var do func(Node) bool
	do = func(x Node) bool {
		return cond(x) || DoChildren(x, do)
	}
	return do(n)
}

// AnyList calls Any(x, cond) for each node x in the list, in order.
// If any call returns true, AnyList stops and returns true.
// Otherwise, AnyList returns false after calling Any(x, cond)
// for every x in the list.
func AnyList(list Nodes, cond func(Node) bool) bool {
	for _, x := range list {
		if Any(x, cond) {
			return true
		}
	}
	return false
}

// EditChildren edits the child nodes of n, replacing each child x with edit(x).
//
// Note that EditChildren(n, edit) only calls edit(x) for n's immediate children.
// If x's children should be processed, then edit(x) must call EditChildren(x, edit).
//
// EditChildren allows constructing general editing passes of the IR graph.
// The most general usage is:
//
//	var edit func(ir.Node) ir.Node
//	edit = func(x ir.Node) ir.Node {
//		... processing BEFORE editing children ...
//		if ... should edit children ... {
//			EditChildren(x, edit)
//			... processing AFTER editing children ...
//		}
//		... return x ...
//	}
//	n = edit(n)
//
// EditChildren edits the node in place. To edit a copy, call Copy first.
// As an example, a simple deep copy implementation would be:
//
//	func deepCopy(n ir.Node) ir.Node {
//		var edit func(ir.Node) ir.Node
//		edit = func(x ir.Node) ir.Node {
//			x = ir.Copy(x)
//			ir.EditChildren(x, edit)
//			return x
//		}
//		return edit(n)
//	}
//
// Of course, in this case it is better to call ir.DeepCopy than to build one anew.
func EditChildren(n Node, edit func(Node) Node) {
	if n == nil {
		return
	}
	n.editChildren(edit)
}
