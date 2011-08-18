// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"fmt"
)

// context describes the state an HTML parser must be in when it reaches the
// portion of HTML produced by evaluating a particular template node.
//
// The zero value of type context is the start context for a template that
// produces an HTML fragment as defined at
// http://www.w3.org/TR/html5/the-end.html#parsing-html-fragments
// where the context element is null.
type context struct {
	state state
	delim delim
}

func (c context) String() string {
	return fmt.Sprintf("context{state: %s, delim: %s", c.state, c.delim)
}

// eq is true if the two contexts are identical field-wise.
func (c context) eq(d context) bool {
	return c.state == d.state && c.delim == d.delim
}

// state describes a high-level HTML parser state.
//
// It bounds the top of the element stack, and by extension the HTML
// insertion mode, but also contains state that does not correspond to
// anything in the HTML5 parsing algorithm because a single token 
// production in the HTML grammar may contain embedded actions in a template.
// For instance, the quoted HTML attribute produced by
//     <div title="Hello {{.World}}">
// is a single token in HTML's grammar but in a template spans several nodes.
type state uint8

const (
	// statePCDATA is parsed character data.  An HTML parser is in
	// this state when its parse position is outside an HTML tag,
	// directive, comment, and special element body.
	statePCDATA state = iota
	// stateTag occurs before an HTML attribute or the end of a tag.
	stateTag
	// stateURI occurs inside an HTML attribute whose content is a URI.
	stateURI
	// stateError is an infectious error state outside any valid
	// HTML/CSS/JS construct.
	stateError
)

var stateNames = [...]string{
	statePCDATA: "statePCDATA",
	stateTag:    "stateTag",
	stateURI:    "stateURI",
	stateError:  "stateError",
}

func (s state) String() string {
	if uint(s) < uint(len(stateNames)) {
		return stateNames[s]
	}
	return fmt.Sprintf("illegal state %d", uint(s))
}

// delim is the delimiter that will end the current HTML attribute.
type delim uint8

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

var delimNames = [...]string{
	delimNone:          "delimNone",
	delimDoubleQuote:   "delimDoubleQuote",
	delimSingleQuote:   "delimSingleQuote",
	delimSpaceOrTagEnd: "delimSpaceOrTagEnd",
}

func (d delim) String() string {
	if uint(d) < uint(len(delimNames)) {
		return delimNames[d]
	}
	return fmt.Sprintf("illegal delim %d", uint(d))
}
