// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IR visitors for walking the IR tree.
//
// The lowest level helpers are DoChildren and EditChildren,
// which nodes help implement (TODO(rsc): eventually) and
// provide control over whether and when recursion happens
// during the walk of the IR.
//
// Although these are both useful directly, two simpler patterns
// are fairly common and also provided: Inspect and Scan.

package ir

import (
	"errors"
)

// DoChildren calls do(x) on each of n's non-nil child nodes x.
// If any call returns a non-nil error, DoChildren stops and returns that error.
// Otherwise, DoChildren returns nil.
//
// Note that DoChildren(n, do) only calls do(x) for n's immediate children.
// If x's children should be processed, then do(x) must call DoChildren(x, do).
//
// DoChildren allows constructing general traversals of the IR graph
// that can stop early if needed. The most general usage is:
//
//	var do func(ir.Node) error
//	do = func(x ir.Node) error {
//		... processing BEFORE visting children ...
//		if ... should visit children ... {
//			ir.DoChildren(x, do)
//			... processing AFTER visting children ...
//		}
//		if ... should stop parent DoChildren call from visiting siblings ... {
//			return non-nil error
//		}
//		return nil
//	}
//	do(root)
//
// Since DoChildren does not generate any errors itself, if the do function
// never wants to stop the traversal, it can assume that DoChildren itself
// will always return nil, simplifying to:
//
//	var do func(ir.Node) error
//	do = func(x ir.Node) error {
//		... processing BEFORE visting children ...
//		if ... should visit children ... {
//			ir.DoChildren(x, do)
//		}
//		... processing AFTER visting children ...
//		return nil
//	}
//	do(root)
//
// The Visit function illustrates a further simplification of the pattern,
// only processing before visiting children and never stopping:
//
//	func Visit(n ir.Node, visit func(ir.Node)) {
//		var do func(ir.Node) error
//		do = func(x ir.Node) error {
//			visit(x)
//			return ir.DoChildren(x, do)
//		}
//		if n != nil {
//			visit(n)
//		}
//	}
//
// The Find function illustrates a different simplification of the pattern,
// visiting each node and then its children, recursively, until finding
// a node x for which find(x) returns true, at which point the entire
// traversal stops and returns true.
//
//	func Find(n ir.Node, find func(ir.Node)) bool {
//		stop := errors.New("stop")
//		var do func(ir.Node) error
//		do = func(x ir.Node) error {
//			if find(x) {
//				return stop
//			}
//			return ir.DoChildren(x, do)
//		}
//		return do(n) == stop
//	}
//
// Visit and Find are presented above as examples of how to use
// DoChildren effectively, but of course, usage that fits within the
// simplifications captured by Visit or Find will be best served
// by directly calling the ones provided by this package.
func DoChildren(n Node, do func(Node) error) error {
	if n == nil {
		return nil
	}
	return n.doChildren(do)
}

// DoList calls f on each non-nil node x in the list, in list order.
// If any call returns a non-nil error, DoList stops and returns that error.
// Otherwise DoList returns nil.
//
// Note that DoList only calls do on the nodes in the list, not their children.
// If x's children should be processed, do(x) must call DoChildren(x, do) itself.
func DoList(list Nodes, do func(Node) error) error {
	for _, x := range list.Slice() {
		if x != nil {
			if err := do(x); err != nil {
				return err
			}
		}
	}
	return nil
}

// Visit visits each non-nil node x in the IR tree rooted at n
// in a depth-first preorder traversal, calling visit on each node visited.
func Visit(n Node, visit func(Node)) {
	var do func(Node) error
	do = func(x Node) error {
		visit(x)
		return DoChildren(x, do)
	}
	if n != nil {
		do(n)
	}
}

// VisitList calls Visit(x, visit) for each node x in the list.
func VisitList(list Nodes, visit func(Node)) {
	for _, x := range list.Slice() {
		Visit(x, visit)
	}
}

var stop = errors.New("stop")

// Find looks for a non-nil node x in the IR tree rooted at n
// for which find(x) returns true.
// Find considers nodes in a depth-first, preorder traversal.
// When Find finds a node x such that find(x) is true,
// Find ends the traversal and returns true immediately.
// Otherwise Find returns false after completing the entire traversal.
func Find(n Node, find func(Node) bool) bool {
	if n == nil {
		return false
	}
	var do func(Node) error
	do = func(x Node) error {
		if find(x) {
			return stop
		}
		return DoChildren(x, do)
	}
	return do(n) == stop
}

// FindList calls Find(x, find) for each node x in the list, in order.
// If any call Find(x, find) returns true, FindList stops and
// returns that result, skipping the remainder of the list.
// Otherwise FindList returns false.
func FindList(list Nodes, find func(Node) bool) bool {
	for _, x := range list.Slice() {
		if Find(x, find) {
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

// editList calls edit on each non-nil node x in the list,
// saving the result of edit back into the list.
//
// Note that editList only calls edit on the nodes in the list, not their children.
// If x's children should be processed, edit(x) must call EditChildren(x, edit) itself.
func editList(list Nodes, edit func(Node) Node) {
	s := list.Slice()
	for i, x := range list.Slice() {
		if x != nil {
			s[i] = edit(x)
		}
	}
}
