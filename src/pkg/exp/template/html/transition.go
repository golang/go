// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"strings"
)

// TODO: ensure transition error messages contain template name and ideally
// line info.

// transitionFunc is the array of context transition functions for text nodes.
// A transition function takes a context and template text input, and returns
// the updated context and any unconsumed text.
var transitionFunc = [...]func(context, []byte) (context, []byte){
	stateText:        tText,
	stateTag:         tTag,
	stateAttrName:    tAttrName,
	stateAfterName:   tAfterName,
	stateBeforeValue: tBeforeValue,
	stateHTMLCmt:     tHTMLCmt,
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
			return context{state: stateHTMLCmt}, s[i+4:]
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
	i := eatWhiteSpace(s, 0)
	if i == len(s) {
		return c, nil
	}
	if s[i] == '>' {
		return context{
			state:   elementContentType[c.element],
			element: c.element,
		}, s[i+1:]
	}
	j, err := eatAttrName(s, i)
	if err != nil {
		return context{state: stateError, err: err}, nil
	}
	state, attr := stateTag, attrNone
	if i != j {
		canonAttrName := strings.ToLower(string(s[i:j]))
		if urlAttr[canonAttrName] {
			attr = attrURL
		} else if strings.HasPrefix(canonAttrName, "on") {
			attr = attrScript
		} else if canonAttrName == "style" {
			attr = attrStyle
		}
		if j == len(s) {
			state = stateAttrName
		} else {
			state = stateAfterName
		}
	}
	return context{state: state, element: c.element, attr: attr}, s[j:]
}

// tAttrName is the context transition function for stateAttrName.
func tAttrName(c context, s []byte) (context, []byte) {
	i, err := eatAttrName(s, 0)
	if err != nil {
		return context{state: stateError, err: err}, nil
	} else if i == len(s) {
		return c, nil
	}
	c.state = stateAfterName
	return c, s[i:]
}

// tAfterName is the context transition function for stateAfterName.
func tAfterName(c context, s []byte) (context, []byte) {
	// Look for the start of the value.
	i := eatWhiteSpace(s, 0)
	if i == len(s) {
		return c, nil
	} else if s[i] != '=' {
		// Occurs due to tag ending '>', and valueless attribute.
		c.state = stateTag
		return c, s[i:]
	}
	c.state = stateBeforeValue
	// Consume the "=".
	return c, s[i+1:]
}

var attrStartStates = [...]state{
	attrNone:   stateAttr,
	attrScript: stateJS,
	attrStyle:  stateCSS,
	attrURL:    stateURL,
}

// tBeforeValue is the context transition function for stateBeforeValue.
func tBeforeValue(c context, s []byte) (context, []byte) {
	i := eatWhiteSpace(s, 0)
	if i == len(s) {
		return c, nil
	}
	// Find the attribute delimiter.
	delim := delimSpaceOrTagEnd
	switch s[i] {
	case '\'':
		delim, i = delimSingleQuote, i+1
	case '"':
		delim, i = delimDoubleQuote, i+1
	}
	c.state, c.delim, c.attr = attrStartStates[c.attr], delim, attrNone
	return c, s[i:]
}

// tHTMLCmt is the context transition function for stateHTMLCmt.
func tHTMLCmt(c context, s []byte) (context, []byte) {
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
	} else if len(s) != eatWhiteSpace(s, 0) && c.urlPart == urlPartNone {
		// HTML5 uses "Valid URL potentially surrounded by spaces" for
		// attrs: http://www.w3.org/TR/html5/index.html#attributes-1
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
		case c.jsCtx == jsCtxDivOp:
			c.jsCtx = jsCtxRegexp
		default:
			return context{
				state: stateError,
				err:   errorf(ErrSlashAmbig, 0, "'/' could start div or regexp: %.32q", s[i:]),
			}, nil
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
					state: stateError,
					err:   errorf(ErrPartialEscape, 0, "unfinished escape sequence in JS string: %q", s),
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
					state: stateError,
					err:   errorf(ErrPartialEscape, 0, "unfinished escape sequence in JS regexp: %q", s),
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
			state: stateError,
			err:   errorf(ErrPartialCharset, 0, "unfinished JS regexp charset: %q", s),
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
					state: stateError,
					err:   errorf(ErrPartialEscape, 0, "unfinished escape sequence in CSS string: %q", s),
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
func eatAttrName(s []byte, i int) (int, *Error) {
	for j := i; j < len(s); j++ {
		switch s[j] {
		case ' ', '\t', '\n', '\f', '\r', '=', '>':
			return j, nil
		case '\'', '"', '<':
			// These result in a parse warning in HTML5 and are
			// indicative of serious problems if seen in an attr
			// name in a template.
			return -1, errorf(ErrBadHTML, 0, "%q in attribute name: %.32q", s[j:j+1], s)
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
