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

// An insertion mode (section 10.2.3.1) is the state transition function from
// a particular state in the HTML5 parser's state machine. In addition to
// returning the next state, it also returns whether the token was consumed.
type insertionMode func(*parser) (insertionMode, bool)

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

// pop pops the top of the stack of open elements.
// It will panic if the stack is empty.
func (p *parser) pop() *Node {
	n := len(p.stack)
	ret := p.stack[n-1]
	p.stack = p.stack[:n-1]
	return ret
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

// addChild adds a child node n to the top element, and pushes n
// if it is an element node (text nodes do not have children).
func (p *parser) addChild(n *Node) {
	m := p.top()
	m.Child = append(m.Child, n)
	if n.Type == ElementNode {
		p.push(n)
	}
}

// addText adds text to the current node.
func (p *parser) addText(s string) {
	// TODO(nigeltao): merge s with previous text, if the preceding node is a text node.
	// TODO(nigeltao): distinguish whitespace text from others.
	p.addChild(&Node{
		Type: TextNode,
		Data: s,
	})
}

// Section 10.2.3.3.
func (p *parser) addFormattingElement(n *Node) {
	p.addChild(n)
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
	if tokenType := p.tokenizer.Next(); tokenType == ErrorToken {
		return p.tokenizer.Error()
	}
	p.tok = p.tokenizer.Token()
	if p.tok.Type == SelfClosingTagToken {
		p.hasSelfClosingToken = true
		p.tok.Type = StartTagToken
	}
	return nil
}

// Section 10.2.4.
func (p *parser) acknowledgeSelfClosingTag() {
	p.hasSelfClosingToken = false
}

// Section 10.2.5.4.
func initialInsertionMode(p *parser) (insertionMode, bool) {
	// TODO(nigeltao): check p.tok for DOCTYPE.
	return beforeHTMLInsertionMode, false
}

// Section 10.2.5.5.
func beforeHTMLInsertionMode(p *parser) (insertionMode, bool) {
	var (
		add     bool
		attr    []Attribute
		implied bool
	)
	switch p.tok.Type {
	case TextToken:
		// TODO(nigeltao): distinguish whitespace text from others.
		implied = true
	case StartTagToken:
		if p.tok.Data == "html" {
			add = true
			attr = p.tok.Attr
		} else {
			implied = true
		}
	case EndTagToken:
		// TODO.
	}
	if add || implied {
		p.addChild(&Node{
			Type: ElementNode,
			Data: "html",
			Attr: attr,
		})
	}
	return beforeHeadInsertionMode, !implied
}

// Section 10.2.5.6.
func beforeHeadInsertionMode(p *parser) (insertionMode, bool) {
	var (
		add     bool
		attr    []Attribute
		implied bool
	)
	switch p.tok.Type {
	case TextToken:
		// TODO(nigeltao): distinguish whitespace text from others.
		implied = true
	case StartTagToken:
		switch p.tok.Data {
		case "head":
			add = true
			attr = p.tok.Attr
		case "html":
			// TODO.
		default:
			implied = true
		}
	case EndTagToken:
		// TODO.
	}
	if add || implied {
		p.addChild(&Node{
			Type: ElementNode,
			Data: "head",
			Attr: attr,
		})
	}
	return inHeadInsertionMode, !implied
}

// Section 10.2.5.7.
func inHeadInsertionMode(p *parser) (insertionMode, bool) {
	var (
		pop     bool
		implied bool
	)
	switch p.tok.Type {
	case TextToken:
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
		return afterHeadInsertionMode, !implied
	}
	return inHeadInsertionMode, !implied
}

// Section 10.2.5.9.
func afterHeadInsertionMode(p *parser) (insertionMode, bool) {
	var (
		add        bool
		attr       []Attribute
		framesetOK bool
		implied    bool
	)
	switch p.tok.Type {
	case TextToken:
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
		p.addChild(&Node{
			Type: ElementNode,
			Data: "body",
			Attr: attr,
		})
		p.framesetOK = framesetOK
	}
	return inBodyInsertionMode, !implied
}

// Section 10.2.5.10.
func inBodyInsertionMode(p *parser) (insertionMode, bool) {
	var endP bool
	switch p.tok.Type {
	case TextToken:
		p.addText(p.tok.Data)
		p.framesetOK = false
	case StartTagToken:
		switch p.tok.Data {
		case "address", "article", "aside", "blockquote", "center", "details", "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "header", "hgroup", "menu", "nav", "ol", "p", "section", "summary", "ul":
			// TODO(nigeltao): Do the proper "does the stack of open elements has a p element in button scope" algorithm in section 10.2.3.2.
			n := p.top()
			if n.Type == ElementNode && n.Data == "p" {
				endP = true
			} else {
				p.addChild(&Node{
					Type: ElementNode,
					Data: p.tok.Data,
					Attr: p.tok.Attr,
				})
			}
		case "b", "big", "code", "em", "font", "i", "s", "small", "strike", "strong", "tt", "u":
			p.reconstructActiveFormattingElements()
			p.addFormattingElement(&Node{
				Type: ElementNode,
				Data: p.tok.Data,
				Attr: p.tok.Attr,
			})
		case "area", "br", "embed", "img", "input", "keygen", "wbr":
			p.reconstructActiveFormattingElements()
			p.addChild(&Node{
				Type: ElementNode,
				Data: p.tok.Data,
				Attr: p.tok.Attr,
			})
			p.pop()
			p.acknowledgeSelfClosingTag()
			p.framesetOK = false
		case "hr":
			// TODO(nigeltao): auto-insert </p> if necessary.
			p.addChild(&Node{
				Type: ElementNode,
				Data: p.tok.Data,
				Attr: p.tok.Attr,
			})
			p.pop()
			p.acknowledgeSelfClosingTag()
			p.framesetOK = false
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "body":
			// TODO(nigeltao): autoclose the stack of open elements.
			return afterBodyInsertionMode, true
		case "a", "b", "big", "code", "em", "font", "i", "nobr", "s", "small", "strike", "strong", "tt", "u":
			// TODO(nigeltao): implement the "adoption agency" algorithm:
			// http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html#adoptionAgency
			p.pop()
		default:
			// TODO.
		}
	}
	if endP {
		// TODO(nigeltao): do the proper algorithm.
		n := p.pop()
		if n.Type != ElementNode || n.Data != "p" {
			panic("unreachable")
		}
	}
	return inBodyInsertionMode, !endP
}

// Section 10.2.5.22.
func afterBodyInsertionMode(p *parser) (insertionMode, bool) {
	switch p.tok.Type {
	case TextToken:
		// TODO.
	case StartTagToken:
		// TODO.
	case EndTagToken:
		switch p.tok.Data {
		case "html":
			// TODO(nigeltao): autoclose the stack of open elements.
			return afterAfterBodyInsertionMode, true
		default:
			// TODO.
		}
	}
	return afterBodyInsertionMode, true
}

// Section 10.2.5.25.
func afterAfterBodyInsertionMode(p *parser) (insertionMode, bool) {
	return inBodyInsertionMode, false
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
	im, consumed := initialInsertionMode, true
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
	// TODO(nigeltao): clean up, depending on the value of im.
	// The specification's algorithm does clean up on reading an EOF 'token',
	// but in go we represent EOF by an os.Error instead.
	return p.doc, nil
}
