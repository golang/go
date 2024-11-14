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
// If efficiency is the primary concern, do not use Inspector for
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
	typ   uint64 // typeOf(node) on push event, or union of typ strictly between push and pop events on pop events
	index int    // index of corresponding push or pop event
}

// TODO: Experiment with storing only the second word of event.node (unsafe.Pointer).
// Type can be recovered from the sole bit in typ.

// Preorder visits all the nodes of the files supplied to New in
// depth-first order. It calls f(n) for each node n before it visits
// n's children.
//
// The complete traversal sequence is determined by ast.Inspect.
// The types argument, if non-empty, enables type-based filtering of
// events. The function f is called only for nodes whose type
// matches an element of the types slice.
func (in *Inspector) Preorder(types []ast.Node, f func(ast.Node)) {
	// Because it avoids postorder calls to f, and the pruning
	// check, Preorder is almost twice as fast as Nodes. The two
	// features seem to contribute similar slowdowns (~1.4x each).

	mask := maskOf(types)
	for i := 0; i < len(in.events); {
		ev := in.events[i]
		if ev.index > i {
			// push
			if ev.typ&mask != 0 {
				f(ev.node)
			}
			pop := ev.index
			if in.events[pop].typ&mask == 0 {
				// Subtrees do not contain types: skip them and pop.
				i = pop + 1
				continue
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
// The complete traversal sequence is determined by ast.Inspect.
// The types argument, if non-empty, enables type-based filtering of
// events. The function f if is called only for nodes whose type
// matches an element of the types slice.
func (in *Inspector) Nodes(types []ast.Node, f func(n ast.Node, push bool) (proceed bool)) {
	mask := maskOf(types)
	for i := 0; i < len(in.events); {
		ev := in.events[i]
		if ev.index > i {
			// push
			pop := ev.index
			if ev.typ&mask != 0 {
				if !f(ev.node, true) {
					i = pop + 1 // jump to corresponding pop + 1
					continue
				}
			}
			if in.events[pop].typ&mask == 0 {
				// Subtrees do not contain types: skip them.
				i = pop
				continue
			}
		} else {
			// pop
			push := ev.index
			if in.events[push].typ&mask != 0 {
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
func (in *Inspector) WithStack(types []ast.Node, f func(n ast.Node, push bool, stack []ast.Node) (proceed bool)) {
	mask := maskOf(types)
	var stack []ast.Node
	for i := 0; i < len(in.events); {
		ev := in.events[i]
		if ev.index > i {
			// push
			pop := ev.index
			stack = append(stack, ev.node)
			if ev.typ&mask != 0 {
				if !f(ev.node, true, stack) {
					i = pop + 1
					stack = stack[:len(stack)-1]
					continue
				}
			}
			if in.events[pop].typ&mask == 0 {
				// Subtrees does not contain types: skip them.
				i = pop
				continue
			}
		} else {
			// pop
			push := ev.index
			if in.events[push].typ&mask != 0 {
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
	capacity := extent * 33 / 100
	if capacity > 1e6 {
		capacity = 1e6 // impose some reasonable maximum
	}
	events := make([]event, 0, capacity)

	var stack []event
	stack = append(stack, event{}) // include an extra event so file nodes have a parent
	for _, f := range files {
		ast.Inspect(f, func { n ->
			if n != nil {
				// push
				ev := event{
					node:  n,
					typ:   0,           // temporarily used to accumulate type bits of subtree
					index: len(events), // push event temporarily holds own index
				}
				stack = append(stack, ev)
				events = append(events, ev)
			} else {
				// pop
				top := len(stack) - 1
				ev := stack[top]
				typ := typeOf(ev.node)
				push := ev.index
				parent := top - 1

				events[push].typ = typ            // set type of push
				stack[parent].typ |= typ | ev.typ // parent's typ contains push and pop's typs.
				events[push].index = len(events)  // make push refer to pop

				stack = stack[:top]
				events = append(events, ev)
			}
			return true
		})
	}

	return events
}
