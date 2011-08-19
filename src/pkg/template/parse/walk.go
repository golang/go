// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import "fmt"

// Walk walks the parse tree, calling before for each node, then
// recurring for any non-nil child nodes that node may have, and
// then calling after.  The before and after functions can be nil.
func (t *Tree) Walk(before, after func(n Node)) {
	walk(t.Root, before, after)
}

func walk(n Node, before, after func(n Node)) {
	if before != nil {
		before(n)
	}
	switch n := n.(type) {
	case nil:
	case *ActionNode:
		if n.Pipe != nil {
			walk(n.Pipe, before, after)
		}
	case *BoolNode:
	case *CommandNode:
		for _, arg := range n.Args {
			walk(arg, before, after)
		}
	case *DotNode:
	case *FieldNode:
	case *IdentifierNode:
	case *IfNode:
		if n.Pipe != nil {
			walk(n.Pipe, before, after)
		}
		if n.List != nil {
			walk(n.List, before, after)
		}
		if n.ElseList != nil {
			walk(n.ElseList, before, after)
		}
	case *ListNode:
		for _, node := range n.Nodes {
			walk(node, before, after)
		}
	case *NumberNode:
	case *PipeNode:
		for _, decl := range n.Decl {
			walk(decl, before, after)
		}
		for _, cmd := range n.Cmds {
			walk(cmd, before, after)
		}
	case *RangeNode:
		if n.Pipe != nil {
			walk(n.Pipe, before, after)
		}
		if n.List != nil {
			walk(n.List, before, after)
		}
		if n.ElseList != nil {
			walk(n.ElseList, before, after)
		}
	case *StringNode:
	case *TemplateNode:
		if n.Pipe != nil {
			walk(n.Pipe, before, after)
		}
	case *TextNode:
	case *VariableNode:
	case *WithNode:
		if n.Pipe != nil {
			walk(n.Pipe, before, after)
		}
		if n.List != nil {
			walk(n.List, before, after)
		}
		if n.ElseList != nil {
			walk(n.ElseList, before, after)
		}
	default:
		panic("unknown node of type " + fmt.Sprintf("%T", n))
	}
	if after != nil {
		after(n)
	}
}
