// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

// A NodeType is the type of a Node.
type NodeType int

const (
	ErrorNode NodeType = iota
	TextNode
	DocumentNode
	ElementNode
	CommentNode
	DoctypeNode
	scopeMarkerNode
)

// Section 12.2.3.3 says "scope markers are inserted when entering applet
// elements, buttons, object elements, marquees, table cells, and table
// captions, and are used to prevent formatting from 'leaking'".
var scopeMarker = Node{Type: scopeMarkerNode}

// A Node consists of a NodeType and some Data (tag name for element nodes,
// content for text) and are part of a tree of Nodes. Element nodes may also
// have a Namespace and contain a slice of Attributes. Data is unescaped, so
// that it looks like "a<b" rather than "a&lt;b".
//
// An empty Namespace implies a "http://www.w3.org/1999/xhtml" namespace.
// Similarly, "math" is short for "http://www.w3.org/1998/Math/MathML", and
// "svg" is short for "http://www.w3.org/2000/svg".
type Node struct {
	Parent    *Node
	Child     []*Node
	Type      NodeType
	Data      string
	Namespace string
	Attr      []Attribute
}

// Add adds a node as a child of n.
// It will panic if the child's parent is not nil.
func (n *Node) Add(child *Node) {
	if child.Parent != nil {
		panic("html: Node.Add called for a child Node that already has a parent")
	}
	child.Parent = n
	n.Child = append(n.Child, child)
}

// Remove removes a node as a child of n.
// It will panic if the child's parent is not n.
func (n *Node) Remove(child *Node) {
	if child.Parent == n {
		child.Parent = nil
		for i, m := range n.Child {
			if m == child {
				copy(n.Child[i:], n.Child[i+1:])
				j := len(n.Child) - 1
				n.Child[j] = nil
				n.Child = n.Child[:j]
				return
			}
		}
	}
	panic("html: Node.Remove called for a non-child Node")
}

// reparentChildren reparents all of src's child nodes to dst.
func reparentChildren(dst, src *Node) {
	for _, n := range src.Child {
		if n.Parent != src {
			panic("html: nodes have an inconsistent parent/child relationship")
		}
		n.Parent = dst
	}
	dst.Child = append(dst.Child, src.Child...)
	src.Child = nil
}

// clone returns a new node with the same type, data and attributes.
// The clone has no parent and no children.
func (n *Node) clone() *Node {
	m := &Node{
		Type: n.Type,
		Data: n.Data,
		Attr: make([]Attribute, len(n.Attr)),
	}
	copy(m.Attr, n.Attr)
	return m
}

// nodeStack is a stack of nodes.
type nodeStack []*Node

// pop pops the stack. It will panic if s is empty.
func (s *nodeStack) pop() *Node {
	i := len(*s)
	n := (*s)[i-1]
	*s = (*s)[:i-1]
	return n
}

// top returns the most recently pushed node, or nil if s is empty.
func (s *nodeStack) top() *Node {
	if i := len(*s); i > 0 {
		return (*s)[i-1]
	}
	return nil
}

// index returns the index of the top-most occurence of n in the stack, or -1
// if n is not present.
func (s *nodeStack) index(n *Node) int {
	for i := len(*s) - 1; i >= 0; i-- {
		if (*s)[i] == n {
			return i
		}
	}
	return -1
}

// insert inserts a node at the given index.
func (s *nodeStack) insert(i int, n *Node) {
	(*s) = append(*s, nil)
	copy((*s)[i+1:], (*s)[i:])
	(*s)[i] = n
}

// remove removes a node from the stack. It is a no-op if n is not present.
func (s *nodeStack) remove(n *Node) {
	i := s.index(n)
	if i == -1 {
		return
	}
	copy((*s)[i:], (*s)[i+1:])
	j := len(*s) - 1
	(*s)[j] = nil
	*s = (*s)[:j]
}

// TODO(nigeltao): forTag no longer used. Should it be deleted?

// forTag returns the top-most element node with the given tag.
func (s *nodeStack) forTag(tag string) *Node {
	for i := len(*s) - 1; i >= 0; i-- {
		n := (*s)[i]
		if n.Type == ElementNode && n.Data == tag {
			return n
		}
	}
	return nil
}
