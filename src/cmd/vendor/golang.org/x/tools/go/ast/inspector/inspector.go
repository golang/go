// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package inspector provides helper functions for traversal over the
// syntax trees of a package, including node filtering by type, and
// materialization of the traversal stack.
//
// During construction, the inspector does a complete traversal and
// builds a list of push/pop events and their node type. Subsequent
// method calls that request a traversal scan this list, rather than walk
// the AST, and perform type filtering using efficient bit sets.
//
// Experiments suggest the inspector's traversals are about 2.5x faster
// than ast.Inspect, but it may take around 5 traversals for this
// benefit to amortize the inspector's construction cost.
// If efficiency is the primary concern, do not use use Inspector for
// one-off traversals.
package inspector

// There are four orthogonal features in a traversal:
//  1 type filtering
//  2 pruning
//  3 postorder calls to f
//  4 stack
// Rather than offer all of them in the API,
// only a few combinations are exposed:
// - Preorder is the fastest and has fewest features,
//   but is the most commonly needed traversal.
// - Nodes and WithStack both provide pruning and postorder calls,
//   even though few clients need it, because supporting two versions
//   is not justified.
// More combinations could be supported by expressing them as
// wrappers around a more generic traversal, but this was measured
// and found to degrade performance significantly (30%).

import (
	"go/ast"
)

// An Inspector provides methods for inspecting
// (traversing) the syntax trees of a package.
type Inspector struct {
	events []event
}

// New returns an Inspector for the specified syntax trees.
func New(files []*ast.File) *Inspector {
	return &Inspector{traverse(files)}
}

// An event represents a push or a pop
// of an ast.Node during a traversal.
type event struct {
	node  ast.Node
	typ   uint64 // typeOf(node)
	index int    // 1 + index of corresponding pop event, or 0 if this is a pop
}

// Preorder visits all the nodes of the files supplied to New in
// depth-first order. It calls f(n) for each node n before it visits
// n's children.
//
// The types argument, if non-empty, enables type-based filtering of
// events. The function f if is called only for nodes whose type
// matches an element of the types slice.
func (in *Inspector) Preorder(types []ast.Node, f func(ast.Node)) {
	// Because it avoids postorder calls to f, and the pruning
	// check, Preorder is almost twice as fast as Nodes. The two
	// features seem to contribute similar slowdowns (~1.4x each).

	mask := maskOf(types)
	for i := 0; i < len(in.events); {
		ev := in.events[i]
		if ev.typ&mask != 0 {
			if ev.index > 0 {
				f(ev.node)
			}
		}
		i++
	}
}

// Nodes visits the nodes of the files supplied to New in depth-first
// order. It calls f(n, true) for each node n before it visits n's
// children. If f returns true, Nodes invokes f recursively for each
// of the non-nil children of the node, followed by a call of
// f(n, false).
//
// The types argument, if non-empty, enables type-based filtering of
// events. The function f if is called only for nodes whose type
// matches an element of the types slice.
func (in *Inspector) Nodes(types []ast.Node, f func(n ast.Node, push bool) (prune bool)) {
	mask := maskOf(types)
	for i := 0; i < len(in.events); {
		ev := in.events[i]
		if ev.typ&mask != 0 {
			if ev.index > 0 {
				// push
				if !f(ev.node, true) {
					i = ev.index // jump to corresponding pop + 1
					continue
				}
			} else {
				// pop
				f(ev.node, false)
			}
		}
		i++
	}
}

// WithStack visits nodes in a similar manner to Nodes, but it
// supplies each call to f an additional argument, the current
// traversal stack. The stack's first element is the outermost node,
// an *ast.File; its last is the innermost, n.
func (in *Inspector) WithStack(types []ast.Node, f func(n ast.Node, push bool, stack []ast.Node) (prune bool)) {
	mask := maskOf(types)
	var stack []ast.Node
	for i := 0; i < len(in.events); {
		ev := in.events[i]
		if ev.index > 0 {
			// push
			stack = append(stack, ev.node)
			if ev.typ&mask != 0 {
				if !f(ev.node, true, stack) {
					i = ev.index
					stack = stack[:len(stack)-1]
					continue
				}
			}
		} else {
			// pop
			if ev.typ&mask != 0 {
				f(ev.node, false, stack)
			}
			stack = stack[:len(stack)-1]
		}
		i++
	}
}

// traverse builds the table of events representing a traversal.
func traverse(files []*ast.File) []event {
	// Preallocate approximate number of events
	// based on source file extent.
	// This makes traverse faster by 4x (!).
	var extent int
	for _, f := range files {
		extent += int(f.End() - f.Pos())
	}
	// This estimate is based on the net/http package.
	events := make([]event, 0, extent*33/100)

	var stack []event
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if n != nil {
				// push
				ev := event{
					node:  n,
					typ:   typeOf(n),
					index: len(events), // push event temporarily holds own index
				}
				stack = append(stack, ev)
				events = append(events, ev)
			} else {
				// pop
				ev := stack[len(stack)-1]
				stack = stack[:len(stack)-1]

				events[ev.index].index = len(events) + 1 // make push refer to pop

				ev.index = 0 // turn ev into a pop event
				events = append(events, ev)
			}
			return true
		})
	}

	return events
}
