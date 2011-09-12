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
	"exp_template_html_cssescaper":      cssEscaper,
	"exp_template_html_cssvaluefilter":  cssValueFilter,
	"exp_template_html_jsregexpescaper": jsRegexpEscaper,
	"exp_template_html_jsstrescaper":    jsStrEscaper,
	"exp_template_html_jsvalescaper":    jsValEscaper,
	"exp_template_html_nospaceescaper":  htmlNospaceEscaper,
	"exp_template_html_urlescaper":      urlEscaper,
	"exp_template_html_urlfilter":       urlFilter,
	"exp_template_html_urlnormalizer":   urlNormalizer,
}

// filterFailsafe is an innocuous word that is emitted in place of unsafe values
// by sanitizer functions.  It is not a keyword in any programming language,
// contains no special characters, is not empty, and when it appears in output
// it is distinct enough that a developer can find the source of the problem
// via a search engine.
const filterFailsafe = "ZgotmplZ"

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
	s := make([]string, 0, 3)
	switch c.state {
	case stateURL, stateCSSDqStr, stateCSSSqStr, stateCSSDqURL, stateCSSSqURL, stateCSSURL:
		switch c.urlPart {
		case urlPartNone:
			s = append(s, "exp_template_html_urlfilter")
			fallthrough
		case urlPartPreQuery:
			switch c.state {
			case stateCSSDqStr, stateCSSSqStr:
				s = append(s, "exp_template_html_cssescaper")
			case stateCSSDqURL, stateCSSSqURL, stateCSSURL:
				s = append(s, "exp_template_html_urlnormalizer")
			}
		case urlPartQueryOrFrag:
			s = append(s, "exp_template_html_urlescaper")
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
	case stateJSDqStr, stateJSSqStr:
		s = append(s, "exp_template_html_jsstrescaper")
	case stateJSRegexp:
		s = append(s, "exp_template_html_jsregexpescaper")
	case stateComment, stateJSBlockCmt, stateJSLineCmt, stateCSSBlockCmt, stateCSSLineCmt:
		return context{
			state:   stateError,
			errLine: n.Line,
			errStr:  fmt.Sprintf("%s appears inside a comment", n),
		}
	case stateCSS:
		s = append(s, "exp_template_html_cssvaluefilter")
	case stateText:
		s = append(s, "html")
	}
	switch c.delim {
	case delimNone:
		// No extra-escaping needed for raw text content.
	case delimSpaceOrTagEnd:
		s = append(s, "exp_template_html_nospaceescaper")
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
		// On exiting an attribute, we discard all state information
		// except the state and element.
		c, s = context{state: stateTag, element: c.element}, s[i:]
	}
	return c
}

// transitionFunc is the array of context transition functions for text nodes.
// A transition function takes a context and template text input, and returns
// the updated context and any unconsumed text.
var transitionFunc = [...]func(context, []byte) (context, []byte){
	stateText:        tText,
	stateTag:         tTag,
	stateComment:     tComment,
	stateRCDATA:      tSpecialTagEnd,
	stateAttr:        tAttr,
	stateURL:         tURL,
	stateJS:          tJS,
	stateJSDqStr:     tJSStr,
	stateJSSqStr:     tJSStr,
	stateJSRegexp:    tJSRegexp,
	stateJSBlockCmt:  tBlockCmt,
	stateJSLineCmt:   tLineCmt,
	stateCSS:         tCSS,
	stateCSSDqStr:    tCSSStr,
	stateCSSSqStr:    tCSSStr,
	stateCSSDqURL:    tCSSStr,
	stateCSSSqURL:    tCSSStr,
	stateCSSURL:      tCSSStr,
	stateCSSBlockCmt: tBlockCmt,
	stateCSSLineCmt:  tLineCmt,
	stateError:       tError,
}

var commentStart = []byte("<!--")
var commentEnd = []byte("-->")

// tText is the context transition function for the text state.
func tText(c context, s []byte) (context, []byte) {
	for {
		i := bytes.IndexByte(s, '<')
		if i == -1 || i+1 == len(s) {
			return c, nil
		} else if i+4 <= len(s) && bytes.Equal(commentStart, s[i:i+4]) {
			return context{state: stateComment}, s[i+4:]
		}
		i++
		if s[i] == '/' {
			if i+1 == len(s) {
				return c, nil
			}
			i++
		}
		j, e := eatTagName(s, i)
		if j != i {
			// We've found an HTML tag.
			return context{state: stateTag, element: e}, s[j:]
		}
		s = s[j:]
	}
	panic("unreachable")
}

var elementContentType = [...]state{
	elementNone:     stateText,
	elementScript:   stateJS,
	elementStyle:    stateCSS,
	elementTextarea: stateRCDATA,
	elementTitle:    stateRCDATA,
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
		return c, nil
	}
	state := stateAttr
	canonAttrName := strings.ToLower(string(s[attrStart:i]))
	if urlAttr[canonAttrName] {
		state = stateURL
	} else if strings.HasPrefix(canonAttrName, "on") {
		state = stateJS
	} else if canonAttrName == "style" {
		state = stateCSS
	}

	// Look for the start of the value.
	i = eatWhiteSpace(s, i)
	if i == len(s) {
		return c, s[i:]
	}
	if s[i] == '>' {
		state = elementContentType[c.element]
		return context{state: state, element: c.element}, s[i+1:]
	} else if s[i] != '=' {
		// Possible due to a valueless attribute or '/' in "<input />".
		return c, s[i:]
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

	return context{state: state, delim: delim, element: c.element}, s[i:]
}

// tComment is the context transition function for stateComment.
func tComment(c context, s []byte) (context, []byte) {
	i := bytes.Index(s, commentEnd)
	if i != -1 {
		return context{}, s[i+3:]
	}
	return c, nil
}

// specialTagEndMarkers maps element types to the character sequence that
// case-insensitively signals the end of the special tag body.
var specialTagEndMarkers = [...]string{
	elementScript:   "</script",
	elementStyle:    "</style",
	elementTextarea: "</textarea",
	elementTitle:    "</title",
}

// tSpecialTagEnd is the context transition function for raw text and RCDATA
// element states.
func tSpecialTagEnd(c context, s []byte) (context, []byte) {
	if c.element != elementNone {
		end := specialTagEndMarkers[c.element]
		i := strings.Index(strings.ToLower(string(s)), end)
		if i != -1 {
			return context{state: stateTag}, s[i+len(end):]
		}
	}
	return c, nil
}

// tAttr is the context transition function for the attribute state.
func tAttr(c context, s []byte) (context, []byte) {
	return c, nil
}

// tURL is the context transition function for the URL state.
func tURL(c context, s []byte) (context, []byte) {
	if bytes.IndexAny(s, "#?") >= 0 {
		c.urlPart = urlPartQueryOrFrag
	} else if len(s) != 0 && c.urlPart == urlPartNone {
		c.urlPart = urlPartPreQuery
	}
	return c, nil
}

// tJS is the context transition function for the JS state.
func tJS(c context, s []byte) (context, []byte) {
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}

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
			c.state, i = stateJSLineCmt, i+1
		case i+1 < len(s) && s[i+1] == '*':
			c.state, i = stateJSBlockCmt, i+1
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
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}

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
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}

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

// tBlockCmt is the context transition function for /*comment*/ states.
func tBlockCmt(c context, s []byte) (context, []byte) {
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}
	i := bytes.Index(s, blockCommentEnd)
	if i == -1 {
		return c, nil
	}
	switch c.state {
	case stateJSBlockCmt:
		c.state = stateJS
	case stateCSSBlockCmt:
		c.state = stateCSS
	default:
		panic(c.state.String())
	}
	return c, s[i+2:]
}

// tLineCmt is the context transition function for //comment states.
func tLineCmt(c context, s []byte) (context, []byte) {
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}
	var lineTerminators string
	var endState state
	switch c.state {
	case stateJSLineCmt:
		lineTerminators, endState = "\n\r\u2028\u2029", stateJS
	case stateCSSLineCmt:
		lineTerminators, endState = "\n\f\r", stateCSS
		// Line comments are not part of any published CSS standard but
		// are supported by the 4 major browsers.
		// This defines line comments as
		//     LINECOMMENT ::= "//" [^\n\f\d]*
		// since http://www.w3.org/TR/css3-syntax/#SUBTOK-nl defines
		// newlines:
		//     nl ::= #xA | #xD #xA | #xD | #xC
	default:
		panic(c.state.String())
	}

	i := bytes.IndexAny(s, lineTerminators)
	if i == -1 {
		return c, nil
	}
	c.state = endState
	// Per section 7.4 of EcmaScript 5 : http://es5.github.com/#x7.4
	// "However, the LineTerminator at the end of the line is not
	// considered to be part of the single-line comment; it is recognised
	// separately by the lexical grammar and becomes part of the stream of
	// input elements for the syntactic grammar."
	return c, s[i:]
}

// tCSS is the context transition function for the CSS state.
func tCSS(c context, s []byte) (context, []byte) {
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}

	// CSS quoted strings are almost never used except for:
	// (1) URLs as in background: "/foo.png"
	// (2) Multiword font-names as in font-family: "Times New Roman"
	// (3) List separators in content values as in inline-lists:
	//    <style>
	//    ul.inlineList { list-style: none; padding:0 }
	//    ul.inlineList > li { display: inline }
	//    ul.inlineList > li:before { content: ", " }
	//    ul.inlineList > li:first-child:before { content: "" }
	//    </style>
	//    <ul class=inlineList><li>One<li>Two<li>Three</ul>
	// (4) Attribute value selectors as in a[href="http://example.com/"]
	//
	// We conservatively treat all strings as URLs, but make some
	// allowances to avoid confusion.
	//
	// In (1), our conservative assumption is justified.
	// In (2), valid font names do not contain ':', '?', or '#', so our
	// conservative assumption is fine since we will never transition past
	// urlPartPreQuery.
	// In (3), our protocol heuristic should not be tripped, and there
	// should not be non-space content after a '?' or '#', so as long as
	// we only %-encode RFC 3986 reserved characters we are ok.
	// In (4), we should URL escape for URL attributes, and for others we
	// have the attribute name available if our conservative assumption
	// proves problematic for real code.

	for {
		i := bytes.IndexAny(s, `("'/`)
		if i == -1 {
			return c, nil
		}
		switch s[i] {
		case '(':
			// Look for url to the left.
			p := bytes.TrimRight(s[:i], "\t\n\f\r ")
			if endsWithCSSKeyword(p, "url") {
				q := bytes.TrimLeft(s[i+1:], "\t\n\f\r ")
				switch {
				case len(q) != 0 && q[0] == '"':
					c.state, s = stateCSSDqURL, q[1:]
				case len(q) != 0 && q[0] == '\'':
					c.state, s = stateCSSSqURL, q[1:]

				default:
					c.state, s = stateCSSURL, q
				}
				return c, s
			}
		case '/':
			if i+1 < len(s) {
				switch s[i+1] {
				case '/':
					c.state = stateCSSLineCmt
					return c, s[i+2:]
				case '*':
					c.state = stateCSSBlockCmt
					return c, s[i+2:]
				}
			}
		case '"':
			c.state = stateCSSDqStr
			return c, s[i+1:]
		case '\'':
			c.state = stateCSSSqStr
			return c, s[i+1:]
		}
		s = s[i+1:]
	}
	panic("unreachable")
}

// tCSSStr is the context transition function for the CSS string and URL states.
func tCSSStr(c context, s []byte) (context, []byte) {
	if d, t := tSpecialTagEnd(c, s); t != nil {
		return d, t
	}

	var endAndEsc string
	switch c.state {
	case stateCSSDqStr, stateCSSDqURL:
		endAndEsc = `\"`
	case stateCSSSqStr, stateCSSSqURL:
		endAndEsc = `\'`
	case stateCSSURL:
		// Unquoted URLs end with a newline or close parenthesis.
		// The below includes the wc (whitespace character) and nl.
		endAndEsc = "\\\t\n\f\r )"
	default:
		panic(c.state.String())
	}

	b := s
	for {
		i := bytes.IndexAny(b, endAndEsc)
		if i == -1 {
			return tURL(c, decodeCSS(b))
		}
		if b[i] == '\\' {
			i++
			if i == len(b) {
				return context{
					state:  stateError,
					errStr: fmt.Sprintf("unfinished escape sequence in CSS string: %q", s),
				}, nil
			}
		} else {
			c.state = stateCSS
			return c, b[i+1:]
		}
		c, _ = tURL(c, decodeCSS(b[:i+1]))
		b = b[i+1:]
	}
	panic("unreachable")
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

var elementNameMap = map[string]element{
	"script":   elementScript,
	"style":    elementStyle,
	"textarea": elementTextarea,
	"title":    elementTitle,
}

// eatTagName returns the largest j such that s[i:j] is a tag name and the tag type.
func eatTagName(s []byte, i int) (int, element) {
	j := i
	for ; j < len(s); j++ {
		x := s[j]
		if !(('a' <= x && x <= 'z') ||
			('A' <= x && x <= 'Z') ||
			('0' <= x && x <= '9' && i != j)) {
			break
		}
	}
	return j, elementNameMap[strings.ToLower(string(s[i:j]))]
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
