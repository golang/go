// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"exp/html/atom"
)

// A NodeType is the type of a Node.
type NodeType uint32

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
// that it looks like "a<b" rather than "a&lt;b". For element nodes, DataAtom
// is the atom for Data, or zero if Data is not a known tag name.
//
// An empty Namespace implies a "http://www.w3.org/1999/xhtml" namespace.
// Similarly, "math" is short for "http://www.w3.org/1998/Math/MathML", and
// "svg" is short for "http://www.w3.org/2000/svg".
type Node struct {
	Parent, FirstChild, LastChild, PrevSibling, NextSibling *Node

	Type      NodeType
	DataAtom  atom.Atom
	Data      string
	Namespace string
	Attr      []Attribute
}

// InsertBefore inserts newChild as a child of n, immediately before oldChild
// in the sequence of n's children. oldChild may be nil, in which case newChild
// is appended to the end of n's children.
//
// It will panic if newChild already has a parent or siblings.
func (n *Node) InsertBefore(newChild, oldChild *Node) {
	if newChild.Parent != nil || newChild.PrevSibling != nil || newChild.NextSibling != nil {
		panic("html: InsertBefore called for an attached child Node")
	}
	var prev, next *Node
	if oldChild != nil {
		prev, next = oldChild.PrevSibling, oldChild
	} else {
		prev = n.LastChild
	}
	if prev != nil {
		prev.NextSibling = newChild
	} else {
		n.FirstChild = newChild
	}
	if next != nil {
		next.PrevSibling = newChild
	} else {
		n.LastChild = newChild
	}
	newChild.Parent = n
	newChild.PrevSibling = prev
	newChild.NextSibling = next
}

// AppendChild adds a node c as a child of n.
//
// It will panic if c already has a parent or siblings.
func (n *Node) AppendChild(c *Node) {
	if c.Parent != nil || c.PrevSibling != nil || c.NextSibling != nil {
		panic("html: AppendChild called for an attached child Node")
	}
	last := n.LastChild
	if last != nil {
		last.NextSibling = c
	} else {
		n.FirstChild = c
	}
	n.LastChild = c
	c.Parent = n
	c.PrevSibling = last
}

// RemoveChild removes a node c that is a child of n. Afterwards, c will have
// no parent and no siblings.
//
// It will panic if c's parent is not n.
func (n *Node) RemoveChild(c *Node) {
	if c.Parent != n {
		panic("html: RemoveChild called for a non-child Node")
	}
	if n.FirstChild == c {
		n.FirstChild = c.NextSibling
	}
	if c.NextSibling != nil {
		c.NextSibling.PrevSibling = c.PrevSibling
	}
	if n.LastChild == c {
		n.LastChild = c.PrevSibling
	}
	if c.PrevSibling != nil {
		c.PrevSibling.NextSibling = c.NextSibling
	}
	c.Parent = nil
	c.PrevSibling = nil
	c.NextSibling = nil
}

// reparentChildren reparents all of src's child nodes to dst.
func reparentChildren(dst, src *Node) {
	for {
		child := src.FirstChild
		if child == nil {
			break
		}
		src.RemoveChild(child)
		dst.AppendChild(child)
	}
}

// clone returns a new node with the same type, data and attributes.
// The clone has no parent, no siblings and no children.
func (n *Node) clone() *Node {
	m := &Node{
		Type:     n.Type,
		DataAtom: n.DataAtom,
		Data:     n.Data,
		Attr:     make([]Attribute, len(n.Attr)),
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

// index returns the index of the top-most occurrence of n in the stack, or -1
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
