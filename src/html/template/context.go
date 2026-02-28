// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"text/template/parse"
)

// context describes the state an HTML parser must be in when it reaches the
// portion of HTML produced by evaluating a particular template node.
//
// The zero value of type context is the start context for a template that
// produces an HTML fragment as defined at
// https://www.w3.org/TR/html5/syntax.html#the-end
// where the context element is null.
type context struct {
	state   state
	delim   delim
	urlPart urlPart
	jsCtx   jsCtx
	// jsBraceDepth contains the current depth, for each JS template literal
	// string interpolation expression, of braces we've seen. This is used to
	// determine if the next } will close a JS template literal string
	// interpolation expression or not.
	jsBraceDepth []int
	attr         attr
	element      element
	n            parse.Node // for range break/continue
	err          *Error
}

func (c context) String() string {
	var err error
	if c.err != nil {
		err = c.err
	}
	return fmt.Sprintf("{%v %v %v %v %v %v %v}", c.state, c.delim, c.urlPart, c.jsCtx, c.attr, c.element, err)
}

// eq reports whether two contexts are equal.
func (c context) eq(d context) bool {
	return c.state == d.state &&
		c.delim == d.delim &&
		c.urlPart == d.urlPart &&
		c.jsCtx == d.jsCtx &&
		c.attr == d.attr &&
		c.element == d.element &&
		c.err == d.err
}

// mangle produces an identifier that includes a suffix that distinguishes it
// from template names mangled with different contexts.
func (c context) mangle(templateName string) string {
	// The mangled name for the default context is the input templateName.
	if c.state == stateText {
		return templateName
	}
	s := templateName + "$htmltemplate_" + c.state.String()
	if c.delim != delimNone {
		s += "_" + c.delim.String()
	}
	if c.urlPart != urlPartNone {
		s += "_" + c.urlPart.String()
	}
	if c.jsCtx != jsCtxRegexp {
		s += "_" + c.jsCtx.String()
	}
	if c.attr != attrNone {
		s += "_" + c.attr.String()
	}
	if c.element != elementNone {
		s += "_" + c.element.String()
	}
	return s
}

// state describes a high-level HTML parser state.
//
// It bounds the top of the element stack, and by extension the HTML insertion
// mode, but also contains state that does not correspond to anything in the
// HTML5 parsing algorithm because a single token production in the HTML
// grammar may contain embedded actions in a template. For instance, the quoted
// HTML attribute produced by
//
//	<div title="Hello {{.World}}">
//
// is a single token in HTML's grammar but in a template spans several nodes.
type state uint8

//go:generate stringer -type state

const (
	// stateText is parsed character data. An HTML parser is in
	// this state when its parse position is outside an HTML tag,
	// directive, comment, and special element body.
	stateText state = iota
	// stateTag occurs before an HTML attribute or the end of a tag.
	stateTag
	// stateAttrName occurs inside an attribute name.
	// It occurs between the ^'s in ` ^name^ = value`.
	stateAttrName
	// stateAfterName occurs after an attr name has ended but before any
	// equals sign. It occurs between the ^'s in ` name^ ^= value`.
	stateAfterName
	// stateBeforeValue occurs after the equals sign but before the value.
	// It occurs between the ^'s in ` name =^ ^value`.
	stateBeforeValue
	// stateHTMLCmt occurs inside an <!-- HTML comment -->.
	stateHTMLCmt
	// stateRCDATA occurs inside an RCDATA element (<textarea> or <title>)
	// as described at https://www.w3.org/TR/html5/syntax.html#elements-0
	stateRCDATA
	// stateAttr occurs inside an HTML attribute whose content is text.
	stateAttr
	// stateURL occurs inside an HTML attribute whose content is a URL.
	stateURL
	// stateSrcset occurs inside an HTML srcset attribute.
	stateSrcset
	// stateJS occurs inside an event handler or script element.
	stateJS
	// stateJSDqStr occurs inside a JavaScript double quoted string.
	stateJSDqStr
	// stateJSSqStr occurs inside a JavaScript single quoted string.
	stateJSSqStr
	// stateJSTmplLit occurs inside a JavaScript back quoted string.
	stateJSTmplLit
	// stateJSRegexp occurs inside a JavaScript regexp literal.
	stateJSRegexp
	// stateJSBlockCmt occurs inside a JavaScript /* block comment */.
	stateJSBlockCmt
	// stateJSLineCmt occurs inside a JavaScript // line comment.
	stateJSLineCmt
	// stateJSHTMLOpenCmt occurs inside a JavaScript <!-- HTML-like comment.
	stateJSHTMLOpenCmt
	// stateJSHTMLCloseCmt occurs inside a JavaScript --> HTML-like comment.
	stateJSHTMLCloseCmt
	// stateCSS occurs inside a <style> element or style attribute.
	stateCSS
	// stateCSSDqStr occurs inside a CSS double quoted string.
	stateCSSDqStr
	// stateCSSSqStr occurs inside a CSS single quoted string.
	stateCSSSqStr
	// stateCSSDqURL occurs inside a CSS double quoted url("...").
	stateCSSDqURL
	// stateCSSSqURL occurs inside a CSS single quoted url('...').
	stateCSSSqURL
	// stateCSSURL occurs inside a CSS unquoted url(...).
	stateCSSURL
	// stateCSSBlockCmt occurs inside a CSS /* block comment */.
	stateCSSBlockCmt
	// stateCSSLineCmt occurs inside a CSS // line comment.
	stateCSSLineCmt
	// stateError is an infectious error state outside any valid
	// HTML/CSS/JS construct.
	stateError
	// stateDead marks unreachable code after a {{break}} or {{continue}}.
	stateDead
)

// isComment is true for any state that contains content meant for template
// authors & maintainers, not for end-users or machines.
func isComment(s state) bool {
	switch s {
	case stateHTMLCmt, stateJSBlockCmt, stateJSLineCmt, stateJSHTMLOpenCmt, stateJSHTMLCloseCmt, stateCSSBlockCmt, stateCSSLineCmt:
		return true
	}
	return false
}

// isInTag return whether s occurs solely inside an HTML tag.
func isInTag(s state) bool {
	switch s {
	case stateTag, stateAttrName, stateAfterName, stateBeforeValue, stateAttr:
		return true
	}
	return false
}

// isInScriptLiteral returns true if s is one of the literal states within a
// <script> tag, and as such occurrences of "<!--", "<script", and "</script"
// need to be treated specially.
func isInScriptLiteral(s state) bool {
	// Ignore the comment states (stateJSBlockCmt, stateJSLineCmt,
	// stateJSHTMLOpenCmt, stateJSHTMLCloseCmt) because their content is already
	// omitted from the output.
	switch s {
	case stateJSDqStr, stateJSSqStr, stateJSTmplLit, stateJSRegexp:
		return true
	}
	return false
}

// delim is the delimiter that will end the current HTML attribute.
type delim uint8

//go:generate stringer -type delim

const (
	// delimNone occurs outside any attribute.
	delimNone delim = iota
	// delimDoubleQuote occurs when a double quote (") closes the attribute.
	delimDoubleQuote
	// delimSingleQuote occurs when a single quote (') closes the attribute.
	delimSingleQuote
	// delimSpaceOrTagEnd occurs when a space or right angle bracket (>)
	// closes the attribute.
	delimSpaceOrTagEnd
)

// urlPart identifies a part in an RFC 3986 hierarchical URL to allow different
// encoding strategies.
type urlPart uint8

//go:generate stringer -type urlPart

const (
	// urlPartNone occurs when not in a URL, or possibly at the start:
	// ^ in "^http://auth/path?k=v#frag".
	urlPartNone urlPart = iota
	// urlPartPreQuery occurs in the scheme, authority, or path; between the
	// ^s in "h^ttp://auth/path^?k=v#frag".
	urlPartPreQuery
	// urlPartQueryOrFrag occurs in the query portion between the ^s in
	// "http://auth/path?^k=v#frag^".
	urlPartQueryOrFrag
	// urlPartUnknown occurs due to joining of contexts both before and
	// after the query separator.
	urlPartUnknown
)

// jsCtx determines whether a '/' starts a regular expression literal or a
// division operator.
type jsCtx uint8

//go:generate stringer -type jsCtx

const (
	// jsCtxRegexp occurs where a '/' would start a regexp literal.
	jsCtxRegexp jsCtx = iota
	// jsCtxDivOp occurs where a '/' would start a division operator.
	jsCtxDivOp
	// jsCtxUnknown occurs where a '/' is ambiguous due to context joining.
	jsCtxUnknown
)

// element identifies the HTML element when inside a start tag or special body.
// Certain HTML element (for example <script> and <style>) have bodies that are
// treated differently from stateText so the element type is necessary to
// transition into the correct context at the end of a tag and to identify the
// end delimiter for the body.
type element uint8

//go:generate stringer -type element

const (
	// elementNone occurs outside a special tag or special element body.
	elementNone element = iota
	// elementScript corresponds to the raw text <script> element
	// with JS MIME type or no type attribute.
	elementScript
	// elementStyle corresponds to the raw text <style> element.
	elementStyle
	// elementTextarea corresponds to the RCDATA <textarea> element.
	elementTextarea
	// elementTitle corresponds to the RCDATA <title> element.
	elementTitle
)

//go:generate stringer -type attr

// attr identifies the current HTML attribute when inside the attribute,
// that is, starting from stateAttrName until stateTag/stateText (exclusive).
type attr uint8

const (
	// attrNone corresponds to a normal attribute or no attribute.
	attrNone attr = iota
	// attrScript corresponds to an event handler attribute.
	attrScript
	// attrScriptType corresponds to the type attribute in script HTML element
	attrScriptType
	// attrStyle corresponds to the style attribute whose value is CSS.
	attrStyle
	// attrURL corresponds to an attribute whose value is a URL.
	attrURL
	// attrSrcset corresponds to a srcset attribute.
	attrSrcset
)
