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
	"html"
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
	if c.state != stateText {
		return nil, fmt.Errorf("%s ends in a non-text context: %v", t.Name(), c)
	}
	t.Funcs(funcMap)
	return t, nil
}

// funcMap maps command names to functions that render their inputs safe.
var funcMap = template.FuncMap{
	"exp_template_html_urlfilter":       urlFilter,
	"exp_template_html_jsvalescaper":    jsValEscaper,
	"exp_template_html_jsstrescaper":    jsStrEscaper,
	"exp_template_html_jsregexpescaper": jsRegexpEscaper,
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
		return escapeText(c, n.Text)
	case *parse.WithNode:
		return escapeBranch(c, &n.BranchNode, "with")
	}
	// TODO: handle a *parse.TemplateNode. Should Escape take a *template.Set?
	panic("escaping " + n.String() + " is unimplemented")
}

// escapeAction escapes an action template node.
func escapeAction(c context, n *parse.ActionNode) context {
	s := make([]string, 0, 2)
	switch c.state {
	case stateURL:
		switch c.urlPart {
		case urlPartNone:
			s = append(s, "exp_template_html_urlfilter")
		case urlPartQueryOrFrag:
			s = append(s, "urlquery")
		case urlPartPreQuery:
			s = append(s, "html")
		case urlPartUnknown:
			return context{
				state:   stateError,
				errLine: n.Line,
				errStr:  fmt.Sprintf("%s appears in an ambiguous URL context", n),
			}
		default:
			panic(c.urlPart.String())
		}
	case stateJS:
		s = append(s, "exp_template_html_jsvalescaper")
		if c.delim != delimNone {
			s = append(s, "html")
		}
	case stateJSDqStr, stateJSSqStr:
		s = append(s, "exp_template_html_jsstrescaper")
	case stateJSRegexp:
		s = append(s, "exp_template_html_jsregexpescaper")
	case stateJSBlockCmt, stateJSLineCmt:
		return context{
			state:   stateError,
			errLine: n.Line,
			errStr:  fmt.Sprintf("%s appears inside a comment", n),
		}
	default:
		s = append(s, "html")
	}
	ensurePipelineContains(n.Pipe, s)
	return c
}

// ensurePipelineContains ensures that the pipeline has commands with
// the identifiers in s in order.
// If the pipeline already has some of the sanitizers, do not interfere.
// For example, if p is (.X | html) and s is ["escapeJSVal", "html"] then it
// has one matching, "html", and one to insert, "escapeJSVal", to produce
// (.X | escapeJSVal | html).
func ensurePipelineContains(p *parse.PipeNode, s []string) {
	if len(s) == 0 {
		return
	}
	n := len(p.Cmds)
	// Find the identifiers at the end of the command chain.
	idents := p.Cmds
	for i := n - 1; i >= 0; i-- {
		if cmd := p.Cmds[i]; len(cmd.Args) != 0 {
			if _, ok := cmd.Args[0].(*parse.IdentifierNode); ok {
				continue
			}
		}
		idents = p.Cmds[i+1:]
	}
	dups := 0
	for _, id := range idents {
		if s[dups] == (id.Args[0].(*parse.IdentifierNode)).Ident {
			dups++
			if dups == len(s) {
				return
			}
		}
	}
	newCmds := make([]*parse.CommandNode, n-len(idents), n+len(s)-dups)
	copy(newCmds, p.Cmds)
	// Merge existing identifier commands with the sanitizers needed.
	for _, id := range idents {
		i := indexOfStr((id.Args[0].(*parse.IdentifierNode)).Ident, s)
		if i != -1 {
			for _, name := range s[:i] {
				newCmds = append(newCmds, newIdentCmd(name))
			}
			s = s[i+1:]
		}
		newCmds = append(newCmds, id)
	}
	// Create any remaining sanitizers.
	for _, name := range s {
		newCmds = append(newCmds, newIdentCmd(name))
	}
	p.Cmds = newCmds
}

// indexOfStr is the least i such that strs[i] == s or -1 if s is not in strs.
func indexOfStr(s string, strs []string) int {
	for i, t := range strs {
		if s == t {
			return i
		}
	}
	return -1
}

// newIdentCmd produces a command containing a single identifier node.
func newIdentCmd(identifier string) *parse.CommandNode {
	return &parse.CommandNode{
		NodeType: parse.NodeCommand,
		Args:     []parse.Node{parse.NewIdentifier(identifier)},
	}
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

	c := a
	c.urlPart = b.urlPart
	if c.eq(b) {
		// The contexts differ only by urlPart.
		c.urlPart = urlPartUnknown
		return c
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
	if nodeName == "range" && c0.state != stateError {
		// The "true" branch of a "range" node can execute multiple times.
		// We check that executing n.List once results in the same context
		// as executing n.List twice.
		c0 = join(c0, escapeList(c0, n.List), n.Line, nodeName)
		if c0.state == stateError {
			// Make clear that this is a problem on loop re-entry
			// since developers tend to overlook that branch when
			// debugging templates.
			c0.errLine = n.Line
			c0.errStr = "on range loop re-entry: " + c0.errStr
			return c0
		}
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

// delimEnds maps each delim to a string of characters that terminate it.
var delimEnds = [...]string{
	delimDoubleQuote: `"`,
	delimSingleQuote: "'",
	// Determined empirically by running the below in various browsers.
	// var div = document.createElement("DIV");
	// for (var i = 0; i < 0x10000; ++i) {
	//   div.innerHTML = "<span title=x" + String.fromCharCode(i) + "-bar>";
	//   if (div.getElementsByTagName("SPAN")[0].title.indexOf("bar") < 0)
	//     document.write("<p>U+" + i.toString(16));
	// }
	delimSpaceOrTagEnd: " \t\n\f\r>",
}

// escapeText escapes a text template node.
func escapeText(c context, s []byte) context {
	for len(s) > 0 {
		if c.delim == delimNone {
			c, s = transitionFunc[c.state](c, s)
			continue
		}

		i := bytes.IndexAny(s, delimEnds[c.delim])
		if i == -1 {
			// Remain inside the attribute.
			// Decode the value so non-HTML rules can easily handle
			//     <button onclick="alert(&quot;Hi!&quot;)">
			// without having to entity decode token boundaries.
			d := c.delim
			c.delim = delimNone
			c = escapeText(c, []byte(html.UnescapeString(string(s))))
			if c.state != stateError {
				c.delim = d
			}
			return c
		}
		if c.delim != delimSpaceOrTagEnd {
			// Consume any quote.
			i++
		}
		c, s = context{state: stateTag}, s[i:]
	}
	return c
}

// transitionFunc is the array of context transition functions for text nodes.
// A transition function takes a context and template text input, and returns
// the updated context and any unconsumed text.
var transitionFunc = [...]func(context, []byte) (context, []byte){
	stateText:       tText,
	stateTag:        tTag,
	stateURL:        tURL,
	stateJS:         tJS,
	stateJSDqStr:    tJSStr,
	stateJSSqStr:    tJSStr,
	stateJSRegexp:   tJSRegexp,
	stateJSBlockCmt: tJSBlockCmt,
	stateJSLineCmt:  tJSLineCmt,
	stateAttr:       tAttr,
	stateError:      tError,
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
	// Find the attribute name.
	attrStart := eatWhiteSpace(s, 0)
	i, err := eatAttrName(s, attrStart)
	if err != nil {
		return context{
			state:  stateError,
			errStr: err.String(),
		}, nil
	}
	if i == len(s) {
		return context{state: stateTag}, nil
	}
	state := stateAttr
	canonAttrName := strings.ToLower(string(s[attrStart:i]))
	if urlAttr[canonAttrName] {
		state = stateURL
	} else if strings.HasPrefix(canonAttrName, "on") {
		state = stateJS
	}

	// Look for the start of the value.
	i = eatWhiteSpace(s, i)
	if i == len(s) {
		return context{state: stateTag}, s[i:]
	}
	if s[i] == '>' {
		return context{state: stateText}, s[i+1:]
	} else if s[i] != '=' {
		// Possible due to a valueless attribute or '/' in "<input />".
		return context{state: stateTag}, s[i:]
	}
	// Consume the "=".
	i = eatWhiteSpace(s, i+1)

	// Find the attribute delimiter.
	delim := delimSpaceOrTagEnd
	if i < len(s) {
		switch s[i] {
		case '\'':
			delim, i = delimSingleQuote, i+1
		case '"':
			delim, i = delimDoubleQuote, i+1
		}
	}

	return context{state: state, delim: delim}, s[i:]
}

// tAttr is the context transition function for the attribute state.
func tAttr(c context, s []byte) (context, []byte) {
	return c, nil
}

// tURL is the context transition function for the URL state.
func tURL(c context, s []byte) (context, []byte) {
	if bytes.IndexAny(s, "#?") >= 0 {
		c.urlPart = urlPartQueryOrFrag
	} else if c.urlPart == urlPartNone {
		c.urlPart = urlPartPreQuery
	}
	return c, nil
}

// tJS is the context transition function for the JS state.
func tJS(c context, s []byte) (context, []byte) {
	// TODO: delegate to tSpecialTagEnd to find any </script> once that CL
	// has been merged.

	i := bytes.IndexAny(s, `"'/`)
	if i == -1 {
		// Entire input is non string, comment, regexp tokens.
		c.jsCtx = nextJSCtx(s, c.jsCtx)
		return c, nil
	}
	c.jsCtx = nextJSCtx(s[:i], c.jsCtx)
	switch s[i] {
	case '"':
		c.state, c.jsCtx = stateJSDqStr, jsCtxRegexp
	case '\'':
		c.state, c.jsCtx = stateJSSqStr, jsCtxRegexp
	case '/':
		switch {
		case i+1 < len(s) && s[i+1] == '/':
			c.state = stateJSLineCmt
		case i+1 < len(s) && s[i+1] == '*':
			c.state = stateJSBlockCmt
		case c.jsCtx == jsCtxRegexp:
			c.state = stateJSRegexp
		default:
			c.jsCtx = jsCtxRegexp
		}
	default:
		panic("unreachable")
	}
	return c, s[i+1:]
}

// tJSStr is the context transition function for the JS string states.
func tJSStr(c context, s []byte) (context, []byte) {
	// TODO: delegate to tSpecialTagEnd to find any </script> once that CL
	// has been merged.

	quoteAndEsc := `\"`
	if c.state == stateJSSqStr {
		quoteAndEsc = `\'`
	}

	b := s
	for {
		i := bytes.IndexAny(b, quoteAndEsc)
		if i == -1 {
			return c, nil
		}
		if b[i] == '\\' {
			i++
			if i == len(b) {
				return context{
					state:  stateError,
					errStr: fmt.Sprintf("unfinished escape sequence in JS string: %q", s),
				}, nil
			}
		} else {
			c.state, c.jsCtx = stateJS, jsCtxDivOp
			return c, b[i+1:]
		}
		b = b[i+1:]
	}
	panic("unreachable")
}

// tJSRegexp is the context transition function for the /RegExp/ literal state.
func tJSRegexp(c context, s []byte) (context, []byte) {
	// TODO: delegate to tSpecialTagEnd to find any </script> once that CL
	// has been merged.

	b := s
	inCharset := false
	for {
		i := bytes.IndexAny(b, `/[\]`)
		if i == -1 {
			break
		}
		switch b[i] {
		case '/':
			if !inCharset {
				c.state, c.jsCtx = stateJS, jsCtxDivOp
				return c, b[i+1:]
			}
		case '\\':
			i++
			if i == len(b) {
				return context{
					state:  stateError,
					errStr: fmt.Sprintf("unfinished escape sequence in JS regexp: %q", s),
				}, nil
			}
		case '[':
			inCharset = true
		case ']':
			inCharset = false
		default:
			panic("unreachable")
		}
		b = b[i+1:]
	}

	if inCharset {
		// This can be fixed by making context richer if interpolation
		// into charsets is desired.
		return context{
			state:  stateError,
			errStr: fmt.Sprintf("unfinished JS regexp charset: %q", s),
		}, nil
	}

	return c, nil
}

var blockCommentEnd = []byte("*/")

// tJSBlockCmt is the context transition function for the JS /*comment*/ state.
func tJSBlockCmt(c context, s []byte) (context, []byte) {
	// TODO: delegate to tSpecialTagEnd to find any </script> once that CL
	// has been merged.

	i := bytes.Index(s, blockCommentEnd)
	if i == -1 {
		return c, nil
	}
	c.state = stateJS
	return c, s[i+2:]
}

// tJSLineCmt is the context transition function for the JS //comment state.
func tJSLineCmt(c context, s []byte) (context, []byte) {
	// TODO: delegate to tSpecialTagEnd to find any </script> once that CL
	// has been merged.

	i := bytes.IndexAny(s, "\r\n\u2028\u2029")
	if i == -1 {
		return c, nil
	}
	c.state = stateJS
	// Per section 7.4 of EcmaScript 5 : http://es5.github.com/#x7.4
	// "However, the LineTerminator at the end of the line is not
	// considered to be part of the single-line comment; it is recognised
	// separately by the lexical grammar and becomes part of the stream of
	// input elements for the syntactic grammar."
	return c, s[i:]
}

// tError is the context transition function for the error state.
func tError(c context, s []byte) (context, []byte) {
	return c, nil
}

// eatAttrName returns the largest j such that s[i:j] is an attribute name.
// It returns an error if s[i:] does not look like it begins with an
// attribute name, such as encountering a quote mark without a preceding
// equals sign.
func eatAttrName(s []byte, i int) (int, os.Error) {
	for j := i; j < len(s); j++ {
		switch s[j] {
		case ' ', '\t', '\n', '\f', '\r', '=', '>':
			return j, nil
		case '\'', '"', '<':
			// These result in a parse warning in HTML5 and are
			// indicative of serious problems if seen in an attr
			// name in a template.
			return 0, fmt.Errorf("%q in attribute name: %.32q", s[j:j+1], s)
		default:
			// No-op.
		}
	}
	return len(s), nil
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
		case ' ', '\t', '\n', '\f', '\r':
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

// urlFilter returns the HTML equivalent of its input unless it contains an
// unsafe protocol in which case it defangs the entire URL.
func urlFilter(args ...interface{}) string {
	ok := false
	var s string
	if len(args) == 1 {
		s, ok = args[0].(string)
	}
	if !ok {
		s = fmt.Sprint(args...)
	}
	i := strings.IndexRune(s, ':')
	if i >= 0 && strings.IndexRune(s[:i], '/') < 0 {
		protocol := strings.ToLower(s[:i])
		if protocol != "http" && protocol != "https" && protocol != "mailto" {
			// Return a value that someone investigating a bug
			// report can put into a search engine.
			return "#ZgotmplZ"
		}
	}
	// TODO: Once we handle <style>#id { background: url({{.Img}}) }</style>
	// we will need to stop this from HTML escaping and pipeline sanitizers.
	return template.HTMLEscapeString(s)
}
