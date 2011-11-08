// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"text/template/parse"
)

// clone clones a template Node.
func clone(n parse.Node) parse.Node {
	switch t := n.(type) {
	case *parse.ActionNode:
		return cloneAction(t)
	case *parse.IfNode:
		b := new(parse.IfNode)
		copyBranch(&b.BranchNode, &t.BranchNode)
		return b
	case *parse.ListNode:
		return cloneList(t)
	case *parse.RangeNode:
		b := new(parse.RangeNode)
		copyBranch(&b.BranchNode, &t.BranchNode)
		return b
	case *parse.TemplateNode:
		return cloneTemplate(t)
	case *parse.TextNode:
		return cloneText(t)
	case *parse.WithNode:
		b := new(parse.WithNode)
		copyBranch(&b.BranchNode, &t.BranchNode)
		return b
	}
	panic("cloning " + n.String() + " is unimplemented")
}

// cloneAction returns a deep clone of n.
func cloneAction(n *parse.ActionNode) *parse.ActionNode {
	// We use keyless fields because they won't compile if a field is added.
	return &parse.ActionNode{n.NodeType, n.Line, clonePipe(n.Pipe)}
}

// cloneList returns a deep clone of n.
func cloneList(n *parse.ListNode) *parse.ListNode {
	if n == nil {
		return nil
	}
	// We use keyless fields because they won't compile if a field is added.
	c := parse.ListNode{n.NodeType, make([]parse.Node, len(n.Nodes))}
	for i, child := range n.Nodes {
		c.Nodes[i] = clone(child)
	}
	return &c
}

// clonePipe returns a shallow clone of n.
// The escaper does not modify pipe descendants in place so there's no need to
// clone deeply.
func clonePipe(n *parse.PipeNode) *parse.PipeNode {
	if n == nil {
		return nil
	}
	// We use keyless fields because they won't compile if a field is added.
	return &parse.PipeNode{n.NodeType, n.Line, n.Decl, n.Cmds}
}

// cloneTemplate returns a deep clone of n.
func cloneTemplate(n *parse.TemplateNode) *parse.TemplateNode {
	// We use keyless fields because they won't compile if a field is added.
	return &parse.TemplateNode{n.NodeType, n.Line, n.Name, clonePipe(n.Pipe)}
}

// cloneText clones the given node sharing its []byte.
func cloneText(n *parse.TextNode) *parse.TextNode {
	// We use keyless fields because they won't compile if a field is added.
	return &parse.TextNode{n.NodeType, n.Text}
}

// copyBranch clones src into dst.
func copyBranch(dst, src *parse.BranchNode) {
	// We use keyless fields because they won't compile if a field is added.
	*dst = parse.BranchNode{
		src.NodeType,
		src.Line,
		clonePipe(src.Pipe),
		cloneList(src.List),
		cloneList(src.ElseList),
	}
}
