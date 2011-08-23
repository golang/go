// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package html is a specialization of template that automates the
// construction of safe HTML output.
// INCOMPLETE.
package html

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"template"
	"template/parse"
)

// Escape rewrites each action in the template to guarantee that the output is
// HTML-escaped.
func Escape(t *template.Template) (*template.Template, os.Error) {
	c := escapeList(context{}, t.Tree.Root)
	if c.errStr != "" {
		return nil, fmt.Errorf("%s:%d: %s", t.Name(), c.errLine, c.errStr)
	}
	return t, nil
}

// escape escapes a template node.
func escape(c context, n parse.Node) context {
	switch n := n.(type) {
	case *parse.ActionNode:
		return escapeAction(c, n)
	case *parse.IfNode:
		return escapeBranch(c, &n.BranchNode, "if")
	case *parse.ListNode:
		return escapeList(c, n)
	case *parse.RangeNode:
		return escapeBranch(c, &n.BranchNode, "range")
	case *parse.TextNode:
		return escapeText(c, n)
	case *parse.WithNode:
		return escapeBranch(c, &n.BranchNode, "with")
	}
	// TODO: handle a *parse.TemplateNode. Should Escape take a *template.Set?
	panic("escaping " + n.String() + " is unimplemented")
}

// escapeAction escapes an action template node.
func escapeAction(c context, n *parse.ActionNode) context {
	sanitizer := "html"
	if c.state == stateURL {
		sanitizer = "urlquery"
	}
	// If the pipe already ends with the sanitizer, do not interfere.
	if m := len(n.Pipe.Cmds); m != 0 {
		if last := n.Pipe.Cmds[m-1]; len(last.Args) != 0 {
			if i, ok := last.Args[0].(*parse.IdentifierNode); ok && i.Ident == sanitizer {
				return c
			}
		}
	}
	// Otherwise, append the sanitizer.
	n.Pipe.Cmds = append(n.Pipe.Cmds, &parse.CommandNode{
		NodeType: parse.NodeCommand,
		Args:     []parse.Node{parse.NewIdentifier(sanitizer)},
	})
	return c
}

// join joins the two contexts of a branch template node. The result is an
// error context if either of the input contexts are error contexts, or if the
// the input contexts differ.
func join(a, b context, line int, nodeName string) context {
	if a.state == stateError {
		return a
	}
	if b.state == stateError {
		return b
	}
	if a.eq(b) {
		return a
	}
	return context{
		state:   stateError,
		errLine: line,
		errStr:  fmt.Sprintf("{{%s}} branches end in different contexts: %v, %v", nodeName, a, b),
	}
}

// escapeBranch escapes a branch template node: "if", "range" and "with".
func escapeBranch(c context, n *parse.BranchNode, nodeName string) context {
	c0 := escapeList(c, n.List)
	if nodeName == "range" {
		// The "true" branch of a "range" node can execute multiple times.
		// We check that executing n.List once results in the same context
		// as executing n.List twice.
		c0 = join(c0, escapeList(c0, n.List), n.Line, nodeName)
	}
	c1 := escapeList(c, n.ElseList)
	return join(c0, c1, n.Line, nodeName)
}

// escapeList escapes a list template node.
func escapeList(c context, n *parse.ListNode) context {
	if n == nil {
		return c
	}
	for _, m := range n.Nodes {
		c = escape(c, m)
	}
	return c
}

// escapeText escapes a text template node.
func escapeText(c context, n *parse.TextNode) context {
	for s := n.Text; len(s) > 0; {
		c, s = transitionFunc[c.state](c, s)
	}
	return c
}

// transitionFunc is the array of context transition functions for text nodes.
// A transition function takes a context and template text input, and returns
// the updated context and any unconsumed text.
var transitionFunc = [...]func(context, []byte) (context, []byte){
	stateText:  tText,
	stateTag:   tTag,
	stateURL:   tURL,
	stateAttr:  tAttr,
	stateError: tError,
}

// tText is the context transition function for the text state.
func tText(c context, s []byte) (context, []byte) {
	for {
		i := bytes.IndexByte(s, '<')
		if i == -1 || i+1 == len(s) {
			return c, nil
		}
		i++
		if s[i] == '/' {
			if i+1 == len(s) {
				return c, nil
			}
			i++
		}
		j := eatTagName(s, i)
		if j != i {
			// We've found an HTML tag.
			return context{state: stateTag}, s[j:]
		}
		s = s[j:]
	}
	panic("unreachable")
}

// tTag is the context transition function for the tag state.
func tTag(c context, s []byte) (context, []byte) {
	// Skip to the end tag, if there is one.
	i := bytes.IndexByte(s, '>')
	if i != -1 {
		return context{state: stateText}, s[i+1:]
	}

	// Otherwise, find the attribute name.
	i = eatWhiteSpace(s, 0)
	attrStart, i := i, eatAttrName(s, i)
	if i == len(s) {
		return context{state: stateTag}, nil
	}
	state := stateAttr
	if urlAttr[strings.ToLower(string(s[attrStart:i]))] {
		state = stateURL
	}

	// Consume the "=".
	i = eatWhiteSpace(s, i)
	if i == len(s) || s[i] != '=' {
		return context{state: stateTag}, s[i:]
	}
	i = eatWhiteSpace(s, i+1)

	// Find the delimiter.
	if i == len(s) {
		return context{state: state, delim: delimSpaceOrTagEnd}, nil
	}
	switch s[i] {
	case '\'':
		return context{state: state, delim: delimSingleQuote}, s[i+1:]
	case '"':
		return context{state: state, delim: delimDoubleQuote}, s[i+1:]
	}

	// TODO: This shouldn't be an error: `<a b=1 c={{.X}}` should be valid.
	return context{state: stateError}, nil
}

// tAttr is the context transition function for the attribute state.
func tAttr(c context, s []byte) (context, []byte) {
	// TODO: look for the delimiter.
	return c, nil
}

// tURL is the context transition function for the URL state.
func tURL(c context, s []byte) (context, []byte) {
	// TODO: look for the delimiter.
	return c, nil
}

// tError is the context transition function for the error state.
func tError(c context, s []byte) (context, []byte) {
	return c, nil
}

// eatAttrName returns the largest j such that s[i:j] is an attribute name.
func eatAttrName(s []byte, i int) int {
	for j := i; j < len(s); j++ {
		switch s[j] {
		case ' ', '\n', '\r', '\t', '=':
			return j
		default:
			// No-op.
		}
	}
	return len(s)
}

// eatTagName returns the largest j such that s[i:j] is a tag name.
func eatTagName(s []byte, i int) int {
	for j := i; j < len(s); j++ {
		x := s[j]
		switch {
		case 'a' <= x && x <= 'z':
			// No-op.
		case 'A' <= x && x <= 'Z':
			// No-op.
		case '0' <= x && x <= '9' && i != j:
			// No-op.
		default:
			return j
		}
	}
	return len(s)
}

// eatWhiteSpace returns the largest j such that s[i:j] is white space.
func eatWhiteSpace(s []byte, i int) int {
	for j := i; j < len(s); j++ {
		switch s[j] {
		case ' ', '\n', '\r', '\t':
			// No-op.
		default:
			return j
		}
	}
	return len(s)
}

// urlAttr is the set of attribute names whose values are URLs.
// It consists of all "%URI"-typed attributes from
// http://www.w3.org/TR/html4/index/attributes.html
// as well as those attributes defined at
// http://dev.w3.org/html5/spec/index.html#attributes-1
// whose Value column in that table matches
// "Valid [non-empty] URL potentially surrounded by spaces".
var urlAttr = map[string]bool{
	"action":     true,
	"archive":    true,
	"background": true,
	"cite":       true,
	"classid":    true,
	"codebase":   true,
	"data":       true,
	"formaction": true,
	"href":       true,
	"icon":       true,
	"longdesc":   true,
	"manifest":   true,
	"poster":     true,
	"profile":    true,
	"src":        true,
	"usemap":     true,
}
