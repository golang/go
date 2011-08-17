// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package html is a specialization of exp/template that automates the
// construction of safe HTML output.
// At the moment, the escaping is naive.  All dynamic content is assumed to be
// plain text interpolated in an HTML PCDATA context.
package html

import (
	"template"
	"template/parse"
)

// Escape rewrites each action in the template to guarantee the output is
// HTML-escaped.
func Escape(t *template.Template) {
	// If the parser shares trees based on common-subexpression
	// joining then we will need to avoid multiply escaping the same action.
	escapeListNode(t.Tree.Root)
}

// escapeNode dispatches to escape<NodeType> helpers by type.
func escapeNode(node parse.Node) {
	switch n := node.(type) {
	case *parse.ListNode:
		escapeListNode(n)
	case *parse.TextNode:
		// Nothing to do.
	case *parse.ActionNode:
		escapeActionNode(n)
	case *parse.IfNode:
		escapeIfNode(n)
	case *parse.RangeNode:
		escapeRangeNode(n)
	case *parse.TemplateNode:
		// Nothing to do.
	case *parse.WithNode:
		escapeWithNode(n)
	default:
		panic("handling for " + node.String() + " not implemented")
		// TODO: Handle other inner node types.
	}
}

// escapeListNode recursively escapes its input's children.
func escapeListNode(node *parse.ListNode) {
	if node == nil {
		return
	}
	children := node.Nodes
	for _, child := range children {
		escapeNode(child)
	}
}

// escapeActionNode adds a pipeline call to the end that escapes the result
// of the expression before it is interpolated into the template output.
func escapeActionNode(node *parse.ActionNode) {
	pipe := node.Pipe

	cmds := pipe.Cmds
	nCmds := len(cmds)

	// If it already has an escaping command, do not interfere.
	if nCmds != 0 {
		if lastCmd := cmds[nCmds-1]; len(lastCmd.Args) != 0 {
			// TODO: Recognize url and js as escaping functions once
			// we have enough context to know whether additional
			// escaping is necessary.
			if arg, ok := lastCmd.Args[0].(*parse.IdentifierNode); ok && arg.Ident == "html" {
				return
			}
		}
	}

	htmlEscapeCommand := parse.CommandNode{
		NodeType: parse.NodeCommand,
		Args:     []parse.Node{parse.NewIdentifier("html")},
	}

	node.Pipe.Cmds = append(node.Pipe.Cmds, &htmlEscapeCommand)
}

// escapeIfNode recursively escapes the if and then clauses but leaves the
// condition unchanged.
func escapeIfNode(node *parse.IfNode) {
	escapeListNode(node.List)
	escapeListNode(node.ElseList)
}

// escapeRangeNode recursively escapes the loop body and else clause but
// leaves the series unchanged.
func escapeRangeNode(node *parse.RangeNode) {
	escapeListNode(node.List)
	escapeListNode(node.ElseList)
}

// escapeWithNode recursively escapes the scope body and else clause but
// leaves the pipeline unchanged.
func escapeWithNode(node *parse.WithNode) {
	escapeListNode(node.List)
	escapeListNode(node.ElseList)
}
