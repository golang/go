// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"io"
	"os"
)

// A NodeType is the type of a Node.
type NodeType int

const (
	ErrorNode NodeType = iota
	TextNode
	DocumentNode
	ElementNode
	CommentNode
)

// A Node consists of a NodeType and some Data (tag name for element nodes,
// content for text) and are part of a tree of Nodes. Element nodes may also
// contain a slice of Attributes. Data is unescaped, so that it looks like
// "a<b" rather than "a&lt;b".
type Node struct {
	Parent *Node
	Child  []*Node
	Type   NodeType
	Data   string
	Attr   []Attribute
}

// A parser implements the HTML5 parsing algorithm:
// http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html#tree-construction
type parser struct {
	// tokenizer provides the tokens for the parser.
	tokenizer *Tokenizer
	// tok is the most recently read token.
	tok Token
	// Self-closing tags like <hr/> are re-interpreted as a two-token sequence:
	// <hr> followed by </hr>. hasSelfClosingToken is true if we have just read
	// the synthetic start tag and the next one due is the matching end tag.
	hasSelfClosingToken bool
	// doc is the document root element.
	doc *Node
	// The stack of open elements (section 10.2.3.2).
	stack []*Node
	// Element pointers (section 10.2.3.4).
	head, form *Node
	// Other parsing state flags (section 10.2.3.5).
	scripting, framesetOK bool
}

// push pushes onto the stack of open elements.
func (p *parser) push(n *Node) {
	p.stack = append(p.stack, n)
}

// top returns the top of the stack of open elements.
// This is also known as the current node.
func (p *parser) top() *Node {
	if n := len(p.stack); n > 0 {
		return p.stack[n-1]
	}
	return p.doc
}

// pop pops the top of the stack of open elements.
// It will panic if the stack is empty.
func (p *parser) pop() *Node {
	n := len(p.stack)
	ret := p.stack[n-1]
	p.stack = p.stack[:n-1]
	return ret
}

// stopTags for use in popUntil. These come from section 10.2.3.2.
var (
	defaultScopeStopTags  = []string{"applet", "caption", "html", "table", "td", "th", "marquee", "object"}
	listItemScopeStopTags = []string{"applet", "caption", "html", "table", "td", "th", "marquee", "object", "ol", "ul"}
	buttonScopeStopTags   = []string{"applet", "caption", "html", "table", "td", "th", "marquee", "object", "button"}
	tableScopeStopTags    = []string{"html", "table"}
)

// popUntil pops the stack of open elements at the highest element whose tag
// is in matchTags, provided there is no higher element in stopTags. It returns
// whether or not there was such an element. If there was not, popUntil leaves
// the stack unchanged.
//
// For example, if the stack was:
// ["html", "body", "font", "table", "b", "i", "u"]
// then popUntil([]string{"html, "table"}, "font") would return false, but
// popUntil([]string{"html, "table"}, "i") would return true and the resultant
// stack would be:
// ["html", "body", "font", "table", "b"]
//
// If an element's tag is in both stopTags and matchTags, then the stack will
// be popped and the function returns true (provided, of course, there was no
// higher element in the stack that was also in stopTags). For example,
// popUntil([]string{"html, "table"}, "table") would return true and leave:
// ["html", "body", "font"]
func (p *parser) popUntil(stopTags []string, matchTags ...string) bool {
	for i := len(p.stack) - 1; i >= 0; i-- {
		tag := p.stack[i].Data
		for _, t := range matchTags {
			if t == tag {
				p.stack = p.stack[:i]
				return true
			}
		}
		for _, t := range stopTags {
			if t == tag {
				return false
			}
		}
	}
	return false
}

// addChild adds a child node n to the top element, and pushes n if it is an
// element node (text nodes are not part of the stack of open elements).
func (p *parser) addChild(n *Node) {
	m := p.top()
	m.Child = append(m.Child, n)
	if n.Type == ElementNode {
		p.push(n)
	}
}

// addText calls addChild with a text node.
func (p *parser) addText(text string) {
	// TODO: merge s with previous text, if the preceding node is a text node.
	// TODO: distinguish whitespace text from others.
	p.addChild(&Node{
		Type: TextNode,
		Data: text,
	})
}

// addElement calls addChild with an element node.
func (p *parser) addElement(tag string, attr []Attribute) {
	p.addChild(&Node{
		Type: ElementNode,
		Data: tag,
		Attr: attr,
	})
}

// Section 10.2.3.3.
func (p *parser) addFormattingElement(tag string, attr []Attribute) {
	p.addElement(tag, attr)
	// TODO.
}

// Section 10.2.3.3.
func (p *parser) reconstructActiveFormattingElements() {
	// TODO.
}

// read reads the next token. This is usually from the tokenizer, but it may
// be the synthesized end tag implied by a self-closing tag.
func (p *parser) read() os.Error {
	if p.hasSelfClosingToken {
		p.hasSelfClosingToken = false
		p.tok.Type = EndTagToken
		p.tok.Attr = nil
		return nil
	}
	p.tokenizer.Next()
	p.tok = p.tokenizer.Token()
	switch p.tok.Type {
	case ErrorToken:
		return p.tokenizer.Error()
	case SelfClosingTagToken:
		p.hasSelfClosingToken = true
		p.tok.Type = StartTagToken
	}
	return nil
}

// Section 10.2.4.
func (p *parser) acknowledgeSelfClosingTag() {
	p.hasSelfClosingToken = false
}

// An insertion mode (section 10.2.3.1) is the state transition function from
// a particular state in the HTML5 parser's state machine. It updates the
// parser's fields depending on parser.token (where ErrorToken means EOF). In
// addition to returning the next insertionMode state, it also returns whether
// the token was consumed.
type insertionMode func(*parser) (insertionMode, bool)

// useTheRulesFor runs the delegate insertionMode over p, returning the actual
// insertionMode unless the delegate caused a state transition.
// Section 10.2.3.1, "using the rules for".
func useTheRulesFor(p *parser, actual, delegate insertionMode) (insertionMode, bool) {
	im, consumed := delegate(p)
	if im != delegate {
		return im, consumed
	}
	return actual, consumed
}

// Section 10.2.5.4.
func initialIM(p *parser) (insertionMode, bool) {
	// TODO: check p.tok for DOCTYPE.
	return beforeHTMLIM, false
}

// Section 10.2.5.5.
func beforeHTMLIM(p *parser) (insertionMode, bool) {
	var (
		add     bool
		attr    []Attribute
		implied bool
	)
	switch p.tok.Type {
	case ErrorToken:
		implied = true
	case TextToken:
		// TODO: distinguish whitespace text from others.
		implied = true
	case StartTagToken:
		if p.tok.Data == "html" {
			add = true
			attr = p.tok.Attr
		} else {
			implied = true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "head", "body", "html", "br":
			implied = true
		default:
			// Ignore the token.
		}
	}
	if add || implied {
		p.addElement("html", attr)
	}
	return beforeHeadIM, !implied
}

// Section 10.2.5.6.
func beforeHeadIM(p *parser) (insertionMode, bool) {
	var (
		add     bool
		attr    []Attribute
		implied bool
	)
	switch p.tok.Type {
	case ErrorToken:
		implied = true
	case TextToken:
		// TODO: distinguish whitespace text from others.
		implied = true
	case StartTagToken:
		switch p.tok.Data {
		case "head":
			add = true
			attr = p.tok.Attr
		case "html":
			return useTheRulesFor(p, beforeHeadIM, inBodyIM)
		default:
			implied = true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "head", "body", "html", "br":
			implied = true
		default:
			// Ignore the token.
		}
	}
	if add || implied {
		p.addElement("head", attr)
	}
	return inHeadIM, !implied
}

// Section 10.2.5.7.
func inHeadIM(p *parser) (insertionMode, bool) {
	var (
		pop     bool
		implied bool
	)
	switch p.tok.Type {
	case ErrorToken, TextToken:
		implied = true
	case StartTagToken:
		switch p.tok.Data {
		case "meta":
			// TODO.
		case "script":
			// TODO.
		default:
			implied = true
		}
	case EndTagToken:
		if p.tok.Data == "head" {
			pop = true
		}
		// TODO.
	}
	if pop || implied {
		n := p.pop()
		if n.Data != "head" {
			panic("html: bad parser state")
		}
		return afterHeadIM, !implied
	}
	return inHeadIM, !implied
}

// Section 10.2.5.9.
func afterHeadIM(p *parser) (insertionMode, bool) {
	var (
		add        bool
		attr       []Attribute
		framesetOK bool
		implied    bool
	)
	switch p.tok.Type {
	case ErrorToken, TextToken:
		implied = true
		framesetOK = true
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			// TODO.
		case "body":
			add = true
			attr = p.tok.Attr
			framesetOK = false
		case "frameset":
			// TODO.
		case "base", "basefont", "bgsound", "link", "meta", "noframes", "script", "style", "title":
			// TODO.
		case "head":
			// TODO.
		default:
			implied = true
			framesetOK = true
		}
	case EndTagToken:
		// TODO.
	}
	if add || implied {
		p.addElement("body", attr)
		p.framesetOK = framesetOK
	}
	return inBodyIM, !implied
}

// Section 10.2.5.10.
func inBodyIM(p *parser) (insertionMode, bool) {
	var endP bool
	switch p.tok.Type {
	case TextToken:
		p.addText(p.tok.Data)
		p.framesetOK = false
	case StartTagToken:
		switch p.tok.Data {
		case "address", "article", "aside", "blockquote", "center", "details", "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "header", "hgroup", "menu", "nav", "ol", "p", "section", "summary", "ul":
			// TODO: Do the proper "does the stack of open elements has a p element in button scope" algorithm in section 10.2.3.2.
			n := p.top()
			if n.Type == ElementNode && n.Data == "p" {
				endP = true
			} else {
				p.addElement(p.tok.Data, p.tok.Attr)
			}
		case "h1", "h2", "h3", "h4", "h5", "h6":
			// TODO: auto-insert </p> if necessary.
			switch n := p.top(); n.Data {
			case "h1", "h2", "h3", "h4", "h5", "h6":
				p.pop()
			}
			p.addElement(p.tok.Data, p.tok.Attr)
		case "b", "big", "code", "em", "font", "i", "s", "small", "strike", "strong", "tt", "u":
			p.reconstructActiveFormattingElements()
			p.addFormattingElement(p.tok.Data, p.tok.Attr)
		case "area", "br", "embed", "img", "input", "keygen", "wbr":
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.pop()
			p.acknowledgeSelfClosingTag()
			p.framesetOK = false
		case "table":
			// TODO: auto-insert </p> if necessary, depending on quirks mode.
			p.addElement(p.tok.Data, p.tok.Attr)
			p.framesetOK = false
			return inTableIM, true
		case "hr":
			// TODO: auto-insert </p> if necessary.
			p.addElement(p.tok.Data, p.tok.Attr)
			p.pop()
			p.acknowledgeSelfClosingTag()
			p.framesetOK = false
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "body":
			// TODO: autoclose the stack of open elements.
			return afterBodyIM, true
		case "a", "b", "big", "code", "em", "font", "i", "nobr", "s", "small", "strike", "strong", "tt", "u":
			// TODO: implement the "adoption agency" algorithm:
			// http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html#adoptionAgency
			if p.tok.Data == p.top().Data {
				p.pop()
			}
		default:
			// TODO.
		}
	}
	if endP {
		// TODO: do the proper algorithm.
		n := p.pop()
		if n.Type != ElementNode || n.Data != "p" {
			panic("unreachable")
		}
	}
	return inBodyIM, !endP
}

// Section 10.2.5.12.
func inTableIM(p *parser) (insertionMode, bool) {
	var (
		add      bool
		data     string
		attr     []Attribute
		consumed bool
	)
	switch p.tok.Type {
	case ErrorToken:
		// Stop parsing.
		return nil, true
	case TextToken:
		// TODO.
	case StartTagToken:
		switch p.tok.Data {
		case "tbody", "tfoot", "thead":
			add = true
			data = p.tok.Data
			attr = p.tok.Attr
			consumed = true
		case "td", "th", "tr":
			add = true
			data = "tbody"
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "table":
			if p.popUntil(tableScopeStopTags, "table") {
				// TODO: "reset the insertion mode appropriately" as per 10.2.3.1.
				return inBodyIM, false
			}
			// Ignore the token.
			return inTableIM, true
		case "body", "caption", "col", "colgroup", "html", "tbody", "td", "tfoot", "th", "thead", "tr":
			// Ignore the token.
			return inTableIM, true
		}
	}
	if add {
		// TODO: clear the stack back to a table context.
		p.addElement(data, attr)
		return inTableBodyIM, consumed
	}
	// TODO: return useTheRulesFor(inTableIM, inBodyIM, p) unless etc. etc. foster parenting.
	return inTableIM, true
}

// Section 10.2.5.16.
func inTableBodyIM(p *parser) (insertionMode, bool) {
	var (
		add      bool
		data     string
		attr     []Attribute
		consumed bool
	)
	switch p.tok.Type {
	case ErrorToken:
		// TODO.
	case TextToken:
		// TODO.
	case StartTagToken:
		switch p.tok.Data {
		case "tr":
			add = true
			data = p.tok.Data
			attr = p.tok.Attr
			consumed = true
		case "td", "th":
			add = true
			data = "tr"
			consumed = false
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "table":
			if p.popUntil(tableScopeStopTags, "tbody", "thead", "tfoot") {
				return inTableIM, false
			}
			// Ignore the token.
			return inTableBodyIM, true
		case "body", "caption", "col", "colgroup", "html", "td", "th", "tr":
			// Ignore the token.
			return inTableBodyIM, true
		}
	}
	if add {
		// TODO: clear the stack back to a table body context.
		p.addElement(data, attr)
		return inRowIM, consumed
	}
	return useTheRulesFor(p, inTableBodyIM, inTableIM)
}

// Section 10.2.5.17.
func inRowIM(p *parser) (insertionMode, bool) {
	switch p.tok.Type {
	case ErrorToken:
		// TODO.
	case TextToken:
		// TODO.
	case StartTagToken:
		switch p.tok.Data {
		case "td", "th":
			// TODO: clear the stack back to a table row context.
			p.addElement(p.tok.Data, p.tok.Attr)
			// TODO: insert a marker at the end of the list of active formatting elements.
			return inCellIM, true
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "tr":
			// TODO.
		case "table":
			if p.popUntil(tableScopeStopTags, "tr") {
				return inTableBodyIM, false
			}
			// Ignore the token.
			return inRowIM, true
		case "tbody", "tfoot", "thead":
			// TODO.
		case "body", "caption", "col", "colgroup", "html", "td", "th":
			// Ignore the token.
			return inRowIM, true
		default:
			// TODO.
		}
	}
	return useTheRulesFor(p, inRowIM, inTableIM)
}

// Section 10.2.5.18.
func inCellIM(p *parser) (insertionMode, bool) {
	var (
		closeTheCellAndReprocess bool
	)
	switch p.tok.Type {
	case StartTagToken:
		switch p.tok.Data {
		case "caption", "col", "colgroup", "tbody", "td", "tfoot", "th", "thead", "tr":
			// TODO: check for "td" or "th" in table scope.
			closeTheCellAndReprocess = true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "td", "th":
			// TODO.
		case "body", "caption", "col", "colgroup", "html":
			// TODO.
		case "table", "tbody", "tfoot", "thead", "tr":
			// TODO: check for matching element in table scope.
			closeTheCellAndReprocess = true
		}
	}
	if closeTheCellAndReprocess {
		if p.popUntil(tableScopeStopTags, "td") || p.popUntil(tableScopeStopTags, "th") {
			// TODO: clear the list of active formatting elements up to the last marker.
			return inRowIM, false
		}
	}
	return useTheRulesFor(p, inCellIM, inBodyIM)
}

// Section 10.2.5.22.
func afterBodyIM(p *parser) (insertionMode, bool) {
	switch p.tok.Type {
	case ErrorToken:
		// TODO.
	case TextToken:
		// TODO.
	case StartTagToken:
		// TODO.
	case EndTagToken:
		switch p.tok.Data {
		case "html":
			// TODO: autoclose the stack of open elements.
			return afterAfterBodyIM, true
		default:
			// TODO.
		}
	}
	return afterBodyIM, true
}

// Section 10.2.5.25.
func afterAfterBodyIM(p *parser) (insertionMode, bool) {
	switch p.tok.Type {
	case ErrorToken:
		// Stop parsing.
		return nil, true
	case TextToken:
		// TODO.
	case StartTagToken:
		if p.tok.Data == "html" {
			return useTheRulesFor(p, afterAfterBodyIM, inBodyIM)
		}
	}
	return inBodyIM, false
}

// Parse returns the parse tree for the HTML from the given Reader.
// The input is assumed to be UTF-8 encoded.
func Parse(r io.Reader) (*Node, os.Error) {
	p := &parser{
		tokenizer: NewTokenizer(r),
		doc: &Node{
			Type: DocumentNode,
		},
		scripting:  true,
		framesetOK: true,
	}
	// Iterate until EOF. Any other error will cause an early return.
	im, consumed := initialIM, true
	for {
		if consumed {
			if err := p.read(); err != nil {
				if err == os.EOF {
					break
				}
				return nil, err
			}
		}
		im, consumed = im(p)
	}
	// Loop until the final token (the ErrorToken signifying EOF) is consumed.
	for {
		if im, consumed = im(p); consumed {
			break
		}
	}
	return p.doc, nil
}
