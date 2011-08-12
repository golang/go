// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package html is a specialization of exp/template that automates the
// construction of safe HTML output.
// At the moment it is just skeleton code that demonstrates how to derive
// templates via AST -> AST transformations.
package html

import (
	"exp/template"
	"exp/template/parse"
	"fmt"
)

// Reverse reverses a template.
// After Reverse(t), t.Execute(wr, data) writes to wr the byte-wise reverse of
// what would have been written otherwise.
//
// E.g.
// Reverse(template.Parse("{{if .Coming}}Hello{{else}}Bye{{end}}, {{.World}}")
// behaves like
// template.Parse("{{.World | reverse}} ,{{if .Coming}}olleH{{else}}eyB{{end}}")
func Reverse(t *template.Template) {
	t.Funcs(supportFuncs)

	// If the parser shares trees based on common-subexpression
	// joining then we will need to avoid multiply reversing the same tree.
	reverseListNode(t.Tree.Root)
}

// reverseNode dispatches to reverse<NodeType> helpers by type.
func reverseNode(node parse.Node) {
	switch n := node.(type) {
	case *parse.ListNode:
		reverseListNode(n)
	case *parse.TextNode:
		reverseTextNode(n)
	case *parse.ActionNode:
		reverseActionNode(n)
	case *parse.IfNode:
		reverseIfNode(n)
	default:
		panic("handling for " + node.String() + " not implemented")
		// TODO: Handle other inner node types.
	}
}

// reverseListNode recursively reverses its input's children and reverses their
// order.
func reverseListNode(node *parse.ListNode) {
	if node == nil {
		return
	}
	children := node.Nodes
	for _, child := range children {
		reverseNode(child)
	}
	for i, j := 0, len(children)-1; i < j; i, j = i+1, j-1 {
		children[i], children[j] = children[j], children[i]
	}
}

// reverseTextNode reverses the text UTF-8 sequence by UTF-8 sequence.
func reverseTextNode(node *parse.TextNode) {
	runes := []int(string(node.Text))
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	node.Text = []byte(string(runes))
}

// reverseActionNode adds a pipeline call to the end that reverses the result
// of the expression before it is interpolated into the template output.
func reverseActionNode(node *parse.ActionNode) {
	pipe := node.Pipe

	cmds := pipe.Cmds
	nCmds := len(cmds)

	// If it's already been reversed, just slice out the reverse command.
	// This makes (Reverse o Reverse) almost the identity function
	// modulo changes to the templates FuncMap.
	if nCmds != 0 {
		if lastCmd := cmds[nCmds-1]; len(lastCmd.Args) != 0 {
			if arg, ok := lastCmd.Args[0].(*parse.IdentifierNode); ok && arg.Ident == "reverse" {
				pipe.Cmds = pipe.Cmds[:nCmds-1]
				return
			}
		}
	}

	reverseCommand := parse.CommandNode{
		NodeType: parse.NodeCommand,
		Args:     []parse.Node{parse.NewIdentifier("reverse")},
	}

	node.Pipe.Cmds = append(node.Pipe.Cmds, &reverseCommand)
}

// reverseIfNode recursively reverses the if and then clauses but leaves the
// condition unchanged.
func reverseIfNode(node *parse.IfNode) {
	reverseListNode(node.List)
	reverseListNode(node.ElseList)
}

// reverse writes the reverse of the given byte buffer to the given Writer.
func reverse(x interface{}) string {
	var s string
	switch y := x.(type) {
	case nil:
		s = "<nil>"
	case []byte:
		// TODO: unnecessary buffer copy.
		s = string(y)
	case string:
		s = y
	case fmt.Stringer:
		s = y.String()
	default:
		s = fmt.Sprintf("<inconvertible of type %T>", x)
	}
	n := len(s)
	bytes := make([]byte, n)
	for i := 0; i < n; i++ {
		bytes[n-i-1] = s[i]
	}
	return string(bytes)
}

// supportFuncs contains functions required by reversed template nodes.
var supportFuncs = template.FuncMap{"reverse": reverse}
