// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inspector

import (
	"fmt"
	"go/ast"
	"go/token"
	"iter"
	"reflect"

	"golang.org/x/tools/go/ast/edge"
)

// A Cursor represents an [ast.Node]. It is immutable.
//
// Two Cursors compare equal if they represent the same node.
//
// Call [Inspector.Root] to obtain a valid cursor for the virtual root
// node of the traversal.
//
// Use the following methods to navigate efficiently around the tree:
//   - for ancestors, use [Cursor.Parent] and [Cursor.Enclosing];
//   - for children, use [Cursor.Child], [Cursor.Children],
//     [Cursor.FirstChild], and [Cursor.LastChild];
//   - for siblings, use [Cursor.PrevSibling] and [Cursor.NextSibling];
//   - for descendants, use [Cursor.FindByPos], [Cursor.FindNode],
//     [Cursor.Inspect], and [Cursor.Preorder].
//
// Use the [Cursor.ChildAt] and [Cursor.ParentEdge] methods for
// information about the edges in a tree: which field (and slice
// element) of the parent node holds the child.
type Cursor struct {
	in    *Inspector
	index int32 // index of push node; -1 for virtual root node
}

// Root returns a cursor for the virtual root node,
// whose children are the files provided to [New].
//
// Its [Cursor.Node] and [Cursor.Stack] methods return nil.
func (in *Inspector) Root() Cursor {
	return Cursor{in, -1}
}

// At returns the cursor at the specified index in the traversal,
// which must have been obtained from [Cursor.Index] on a Cursor
// belonging to the same Inspector (see [Cursor.Inspector]).
func (in *Inspector) At(index int32) Cursor {
	if index < 0 {
		panic("negative index")
	}
	if int(index) >= len(in.events) {
		panic("index out of range for this inspector")
	}
	if in.events[index].index < index {
		panic("invalid index") // (a push, not a pop)
	}
	return Cursor{in, index}
}

// Inspector returns the cursor's Inspector.
func (c Cursor) Inspector() *Inspector { return c.in }

// Index returns the index of this cursor position within the package.
//
// Clients should not assume anything about the numeric Index value
// except that it increases monotonically throughout the traversal.
// It is provided for use with [At].
//
// Index must not be called on the Root node.
func (c Cursor) Index() int32 {
	if c.index < 0 {
		panic("Index called on Root node")
	}
	return c.index
}

// Node returns the node at the current cursor position,
// or nil for the cursor returned by [Inspector.Root].
func (c Cursor) Node() ast.Node {
	if c.index < 0 {
		return nil
	}
	return c.in.events[c.index].node
}

// String returns information about the cursor's node, if any.
func (c Cursor) String() string {
	if c.in == nil {
		return "(invalid)"
	}
	if c.index < 0 {
		return "(root)"
	}
	return reflect.TypeOf(c.Node()).String()
}

// indices return the [start, end) half-open interval of event indices.
func (c Cursor) indices() (int32, int32) {
	if c.index < 0 {
		return 0, int32(len(c.in.events)) // root: all events
	} else {
		return c.index, c.in.events[c.index].index + 1 // just one subtree
	}
}

// Preorder returns an iterator over the nodes of the subtree
// represented by c in depth-first order. Each node in the sequence is
// represented by a Cursor that allows access to the Node, but may
// also be used to start a new traversal, or to obtain the stack of
// nodes enclosing the cursor.
//
// The traversal sequence is determined by [ast.Inspect]. The types
// argument, if non-empty, enables type-based filtering of events. The
// function f if is called only for nodes whose type matches an
// element of the types slice.
//
// If you need control over descent into subtrees,
// or need both pre- and post-order notifications, use [Cursor.Inspect]
func (c Cursor) Preorder(types ...ast.Node) iter.Seq[Cursor] {
	mask := maskOf(types)

	return func(yield func(Cursor) bool) {
		events := c.in.events

		for i, limit := c.indices(); i < limit; {
			ev := events[i]
			if ev.index > i { // push?
				if ev.typ&mask != 0 && !yield(Cursor{c.in, i}) {
					break
				}
				pop := ev.index
				if events[pop].typ&mask == 0 {
					// Subtree does not contain types: skip.
					i = pop + 1
					continue
				}
			}
			i++
		}
	}
}

// Inspect visits the nodes of the subtree represented by c in
// depth-first order. It calls f(n) for each node n before it
// visits n's children. If f returns true, Inspect invokes f
// recursively for each of the non-nil children of the node.
//
// Each node is represented by a Cursor that allows access to the
// Node, but may also be used to start a new traversal, or to obtain
// the stack of nodes enclosing the cursor.
//
// The complete traversal sequence is determined by [ast.Inspect].
// The types argument, if non-empty, enables type-based filtering of
// events. The function f if is called only for nodes whose type
// matches an element of the types slice.
func (c Cursor) Inspect(types []ast.Node, f func(c Cursor) (descend bool)) {
	mask := maskOf(types)
	events := c.in.events
	for i, limit := c.indices(); i < limit; {
		ev := events[i]
		if ev.index > i {
			// push
			pop := ev.index
			if ev.typ&mask != 0 && !f(Cursor{c.in, i}) ||
				events[pop].typ&mask == 0 {
				// The user opted not to descend, or the
				// subtree does not contain types:
				// skip past the pop.
				i = pop + 1
				continue
			}
		}
		i++
	}
}

// Enclosing returns an iterator over the nodes enclosing the current
// current node, starting with the Cursor itself.
//
// Enclosing must not be called on the Root node (whose [Cursor.Node] returns nil).
//
// The types argument, if non-empty, enables type-based filtering of
// events: the sequence includes only enclosing nodes whose type
// matches an element of the types slice.
func (c Cursor) Enclosing(types ...ast.Node) iter.Seq[Cursor] {
	if c.index < 0 {
		panic("Cursor.Enclosing called on Root node")
	}

	mask := maskOf(types)

	return func(yield func(Cursor) bool) {
		events := c.in.events
		for i := c.index; i >= 0; i = events[i].parent {
			if events[i].typ&mask != 0 && !yield(Cursor{c.in, i}) {
				break
			}
		}
	}
}

// Parent returns the parent of the current node.
//
// Parent must not be called on the Root node (whose [Cursor.Node] returns nil).
func (c Cursor) Parent() Cursor {
	if c.index < 0 {
		panic("Cursor.Parent called on Root node")
	}

	return Cursor{c.in, c.in.events[c.index].parent}
}

// ParentEdge returns the identity of the field in the parent node
// that holds this cursor's node, and if it is a list, the index within it.
//
// For example, f(x, y) is a CallExpr whose three children are Idents.
// f has edge kind [edge.CallExpr_Fun] and index -1.
// x and y have kind [edge.CallExpr_Args] and indices 0 and 1, respectively.
//
// If called on a child of the Root node, it returns ([edge.Invalid], -1).
//
// ParentEdge must not be called on the Root node (whose [Cursor.Node] returns nil).
func (c Cursor) ParentEdge() (edge.Kind, int) {
	if c.index < 0 {
		panic("Cursor.ParentEdge called on Root node")
	}
	events := c.in.events
	pop := events[c.index].index
	return unpackEdgeKindAndIndex(events[pop].parent)
}

// ChildAt returns the cursor for the child of the
// current node identified by its edge and index.
// The index must be -1 if the edge.Kind is not a slice.
// The indicated child node must exist.
//
// ChildAt must not be called on the Root node (whose [Cursor.Node] returns nil).
//
// Invariant: c.Parent().ChildAt(c.ParentEdge()) == c.
func (c Cursor) ChildAt(k edge.Kind, idx int) Cursor {
	target := packEdgeKindAndIndex(k, idx)

	// Unfortunately there's no shortcut to looping.
	events := c.in.events
	i := c.index + 1
	for {
		pop := events[i].index
		if pop < i {
			break
		}
		if events[pop].parent == target {
			return Cursor{c.in, i}
		}
		i = pop + 1
	}
	panic(fmt.Sprintf("ChildAt(%v, %d): no such child of %v", k, idx, c))
}

// Child returns the cursor for n, which must be a direct child of c's Node.
//
// Child must not be called on the Root node (whose [Cursor.Node] returns nil).
func (c Cursor) Child(n ast.Node) Cursor {
	if c.index < 0 {
		panic("Cursor.Child called on Root node")
	}

	if false {
		// reference implementation
		for child := range c.Children() {
			if child.Node() == n {
				return child
			}
		}

	} else {
		// optimized implementation
		events := c.in.events
		for i := c.index + 1; events[i].index > i; i = events[i].index + 1 {
			if events[i].node == n {
				return Cursor{c.in, i}
			}
		}
	}
	panic(fmt.Sprintf("Child(%T): not a child of %v", n, c))
}

// NextSibling returns the cursor for the next sibling node in the same list
// (for example, of files, decls, specs, statements, fields, or expressions) as
// the current node. It returns (zero, false) if the node is the last node in
// the list, or is not part of a list.
//
// NextSibling must not be called on the Root node.
//
// See note at [Cursor.Children].
func (c Cursor) NextSibling() (Cursor, bool) {
	if c.index < 0 {
		panic("Cursor.NextSibling called on Root node")
	}

	events := c.in.events
	i := events[c.index].index + 1 // after corresponding pop
	if i < int32(len(events)) {
		if events[i].index > i { // push?
			return Cursor{c.in, i}, true
		}
	}
	return Cursor{}, false
}

// PrevSibling returns the cursor for the previous sibling node in the
// same list (for example, of files, decls, specs, statements, fields,
// or expressions) as the current node. It returns zero if the node is
// the first node in the list, or is not part of a list.
//
// It must not be called on the Root node.
//
// See note at [Cursor.Children].
func (c Cursor) PrevSibling() (Cursor, bool) {
	if c.index < 0 {
		panic("Cursor.PrevSibling called on Root node")
	}

	events := c.in.events
	i := c.index - 1
	if i >= 0 {
		if j := events[i].index; j < i { // pop?
			return Cursor{c.in, j}, true
		}
	}
	return Cursor{}, false
}

// FirstChild returns the first direct child of the current node,
// or zero if it has no children.
func (c Cursor) FirstChild() (Cursor, bool) {
	events := c.in.events
	i := c.index + 1                                   // i=0 if c is root
	if i < int32(len(events)) && events[i].index > i { // push?
		return Cursor{c.in, i}, true
	}
	return Cursor{}, false
}

// LastChild returns the last direct child of the current node,
// or zero if it has no children.
func (c Cursor) LastChild() (Cursor, bool) {
	events := c.in.events
	if c.index < 0 { // root?
		if len(events) > 0 {
			// return push of final event (a pop)
			return Cursor{c.in, events[len(events)-1].index}, true
		}
	} else {
		j := events[c.index].index - 1 // before corresponding pop
		// Inv: j == c.index if c has no children
		//  or  j is last child's pop.
		if j > c.index { // c has children
			return Cursor{c.in, events[j].index}, true
		}
	}
	return Cursor{}, false
}

// Children returns an iterator over the direct children of the
// current node, if any.
//
// When using Children, NextChild, and PrevChild, bear in mind that a
// Node's children may come from different fields, some of which may
// be lists of nodes without a distinguished intervening container
// such as [ast.BlockStmt].
//
// For example, [ast.CaseClause] has a field List of expressions and a
// field Body of statements, so the children of a CaseClause are a mix
// of expressions and statements. Other nodes that have "uncontained"
// list fields include:
//
//   - [ast.ValueSpec] (Names, Values)
//   - [ast.CompositeLit] (Type, Elts)
//   - [ast.IndexListExpr] (X, Indices)
//   - [ast.CallExpr] (Fun, Args)
//   - [ast.AssignStmt] (Lhs, Rhs)
//
// So, do not assume that the previous sibling of an ast.Stmt is also
// an ast.Stmt, or if it is, that they are executed sequentially,
// unless you have established that, say, its parent is a BlockStmt
// or its [Cursor.ParentEdge] is [edge.BlockStmt_List].
// For example, given "for S1; ; S2 {}", the predecessor of S2 is S1,
// even though they are not executed in sequence.
func (c Cursor) Children() iter.Seq[Cursor] {
	return func(yield func(Cursor) bool) {
		c, ok := c.FirstChild()
		for ok && yield(c) {
			c, ok = c.NextSibling()
		}
	}
}

// Contains reports whether c contains or is equal to c2.
//
// Both Cursors must belong to the same [Inspector];
// neither may be its Root node.
func (c Cursor) Contains(c2 Cursor) bool {
	if c.in != c2.in {
		panic("different inspectors")
	}
	events := c.in.events
	return c.index <= c2.index && events[c2.index].index <= events[c.index].index
}

// FindNode returns the cursor for node n if it belongs to the subtree
// rooted at c. It returns zero if n is not found.
func (c Cursor) FindNode(n ast.Node) (Cursor, bool) {

	// FindNode is equivalent to this code,
	// but more convenient and 15-20% faster:
	if false {
		for candidate := range c.Preorder(n) {
			if candidate.Node() == n {
				return candidate, true
			}
		}
		return Cursor{}, false
	}

	// TODO(adonovan): opt: should we assume Node.Pos is accurate
	// and combine type-based filtering with position filtering
	// like FindByPos?

	mask := maskOf([]ast.Node{n})
	events := c.in.events

	for i, limit := c.indices(); i < limit; i++ {
		ev := events[i]
		if ev.index > i { // push?
			if ev.typ&mask != 0 && ev.node == n {
				return Cursor{c.in, i}, true
			}
			pop := ev.index
			if events[pop].typ&mask == 0 {
				// Subtree does not contain type of n: skip.
				i = pop
			}
		}
	}
	return Cursor{}, false
}

// FindByPos returns the cursor for the innermost node n in the tree
// rooted at c such that n.Pos() <= start && end <= n.End().
// (For an *ast.File, it uses the bounds n.FileStart-n.FileEnd.)
//
// It returns zero if none is found.
// Precondition: start <= end.
//
// See also [astutil.PathEnclosingInterval], which
// tolerates adjoining whitespace.
func (c Cursor) FindByPos(start, end token.Pos) (Cursor, bool) {
	if end < start {
		panic("end < start")
	}
	events := c.in.events

	// This algorithm could be implemented using c.Inspect,
	// but it is about 2.5x slower.

	best := int32(-1) // push index of latest (=innermost) node containing range
	for i, limit := c.indices(); i < limit; i++ {
		ev := events[i]
		if ev.index > i { // push?
			n := ev.node
			var nodeEnd token.Pos
			if file, ok := n.(*ast.File); ok {
				nodeEnd = file.FileEnd
				// Note: files may be out of Pos order.
				if file.FileStart > start {
					i = ev.index // disjoint, after; skip to next file
					continue
				}
			} else {
				nodeEnd = n.End()
				if n.Pos() > start {
					break // disjoint, after; stop
				}
			}
			// Inv: node.{Pos,FileStart} <= start
			if end <= nodeEnd {
				// node fully contains target range
				best = i
			} else if nodeEnd < start {
				i = ev.index // disjoint, before; skip forward
			}
		}
	}
	if best >= 0 {
		return Cursor{c.in, best}, true
	}
	return Cursor{}, false
}
