// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"io"
	"strings"
)

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
	// The stack of open elements (section 12.2.3.2) and active formatting
	// elements (section 12.2.3.3).
	oe, afe nodeStack
	// Element pointers (section 12.2.3.4).
	head, form *Node
	// Other parsing state flags (section 12.2.3.5).
	scripting, framesetOK bool
	// im is the current insertion mode.
	im insertionMode
	// originalIM is the insertion mode to go back to after completing a text
	// or inTableText insertion mode.
	originalIM insertionMode
	// fosterParenting is whether new elements should be inserted according to
	// the foster parenting rules (section 12.2.5.3).
	fosterParenting bool
	// quirks is whether the parser is operating in "quirks mode."
	quirks bool
	// context is the context element when parsing an HTML fragment
	// (section 12.4).
	context *Node
}

func (p *parser) top() *Node {
	if n := p.oe.top(); n != nil {
		return n
	}
	return p.doc
}

// Stop tags for use in popUntil. These come from section 12.2.3.2.
var (
	defaultScopeStopTags = map[string][]string{
		"":     {"applet", "caption", "html", "table", "td", "th", "marquee", "object"},
		"math": {"annotation-xml", "mi", "mn", "mo", "ms", "mtext"},
		"svg":  {"desc", "foreignObject", "title"},
	}
)

type scope int

const (
	defaultScope scope = iota
	listItemScope
	buttonScope
	tableScope
	tableRowScope
)

// popUntil pops the stack of open elements at the highest element whose tag
// is in matchTags, provided there is no higher element in the scope's stop
// tags (as defined in section 12.2.3.2). It returns whether or not there was
// such an element. If there was not, popUntil leaves the stack unchanged.
//
// For example, the set of stop tags for table scope is: "html", "table". If
// the stack was:
// ["html", "body", "font", "table", "b", "i", "u"]
// then popUntil(tableScope, "font") would return false, but
// popUntil(tableScope, "i") would return true and the stack would become:
// ["html", "body", "font", "table", "b"]
//
// If an element's tag is in both the stop tags and matchTags, then the stack
// will be popped and the function returns true (provided, of course, there was
// no higher element in the stack that was also in the stop tags). For example,
// popUntil(tableScope, "table") returns true and leaves:
// ["html", "body", "font"]
func (p *parser) popUntil(s scope, matchTags ...string) bool {
	if i := p.indexOfElementInScope(s, matchTags...); i != -1 {
		p.oe = p.oe[:i]
		return true
	}
	return false
}

// indexOfElementInScope returns the index in p.oe of the highest element whose
// tag is in matchTags that is in scope. If no matching element is in scope, it
// returns -1.
func (p *parser) indexOfElementInScope(s scope, matchTags ...string) int {
	for i := len(p.oe) - 1; i >= 0; i-- {
		tag := p.oe[i].Data
		if p.oe[i].Namespace == "" {
			for _, t := range matchTags {
				if t == tag {
					return i
				}
			}
			switch s {
			case defaultScope:
				// No-op.
			case listItemScope:
				if tag == "ol" || tag == "ul" {
					return -1
				}
			case buttonScope:
				if tag == "button" {
					return -1
				}
			case tableScope:
				if tag == "html" || tag == "table" {
					return -1
				}
			default:
				panic("unreachable")
			}
		}
		switch s {
		case defaultScope, listItemScope, buttonScope:
			for _, t := range defaultScopeStopTags[p.oe[i].Namespace] {
				if t == tag {
					return -1
				}
			}
		}
	}
	return -1
}

// elementInScope is like popUntil, except that it doesn't modify the stack of
// open elements.
func (p *parser) elementInScope(s scope, matchTags ...string) bool {
	return p.indexOfElementInScope(s, matchTags...) != -1
}

// clearStackToContext pops elements off the stack of open elements until a
// scope-defined element is found.
func (p *parser) clearStackToContext(s scope) {
	for i := len(p.oe) - 1; i >= 0; i-- {
		tag := p.oe[i].Data
		switch s {
		case tableScope:
			if tag == "html" || tag == "table" {
				p.oe = p.oe[:i+1]
				return
			}
		case tableRowScope:
			if tag == "html" || tag == "tr" {
				p.oe = p.oe[:i+1]
				return
			}
		default:
			panic("unreachable")
		}
	}
}

// addChild adds a child node n to the top element, and pushes n onto the stack
// of open elements if it is an element node.
func (p *parser) addChild(n *Node) {
	if p.fosterParenting {
		p.fosterParent(n)
	} else {
		p.top().Add(n)
	}

	if n.Type == ElementNode {
		p.oe = append(p.oe, n)
	}
}

// fosterParent adds a child node according to the foster parenting rules.
// Section 12.2.5.3, "foster parenting".
func (p *parser) fosterParent(n *Node) {
	p.fosterParenting = false
	var table, parent *Node
	var i int
	for i = len(p.oe) - 1; i >= 0; i-- {
		if p.oe[i].Data == "table" {
			table = p.oe[i]
			break
		}
	}

	if table == nil {
		// The foster parent is the html element.
		parent = p.oe[0]
	} else {
		parent = table.Parent
	}
	if parent == nil {
		parent = p.oe[i-1]
	}

	var child *Node
	for i, child = range parent.Child {
		if child == table {
			break
		}
	}

	if i > 0 && parent.Child[i-1].Type == TextNode && n.Type == TextNode {
		parent.Child[i-1].Data += n.Data
		return
	}

	if i == len(parent.Child) {
		parent.Add(n)
	} else {
		// Insert n into parent.Child at index i.
		parent.Child = append(parent.Child[:i+1], parent.Child[i:]...)
		parent.Child[i] = n
		n.Parent = parent
	}
}

// addText adds text to the preceding node if it is a text node, or else it
// calls addChild with a new text node.
func (p *parser) addText(text string) {
	// TODO: distinguish whitespace text from others.
	t := p.top()
	if i := len(t.Child); i > 0 && t.Child[i-1].Type == TextNode {
		t.Child[i-1].Data += text
		return
	}
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

// Section 12.2.3.3.
func (p *parser) addFormattingElement(tag string, attr []Attribute) {
	p.addElement(tag, attr)
	p.afe = append(p.afe, p.top())
	// TODO.
}

// Section 12.2.3.3.
func (p *parser) clearActiveFormattingElements() {
	for {
		n := p.afe.pop()
		if len(p.afe) == 0 || n.Type == scopeMarkerNode {
			return
		}
	}
}

// Section 12.2.3.3.
func (p *parser) reconstructActiveFormattingElements() {
	n := p.afe.top()
	if n == nil {
		return
	}
	if n.Type == scopeMarkerNode || p.oe.index(n) != -1 {
		return
	}
	i := len(p.afe) - 1
	for n.Type != scopeMarkerNode && p.oe.index(n) == -1 {
		if i == 0 {
			i = -1
			break
		}
		i--
		n = p.afe[i]
	}
	for {
		i++
		clone := p.afe[i].clone()
		p.addChild(clone)
		p.afe[i] = clone
		if i == len(p.afe)-1 {
			break
		}
	}
}

// read reads the next token. This is usually from the tokenizer, but it may
// be the synthesized end tag implied by a self-closing tag.
func (p *parser) read() error {
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
		return p.tokenizer.Err()
	case SelfClosingTagToken:
		p.hasSelfClosingToken = true
		p.tok.Type = StartTagToken
	}
	return nil
}

// Section 12.2.4.
func (p *parser) acknowledgeSelfClosingTag() {
	p.hasSelfClosingToken = false
}

// An insertion mode (section 12.2.3.1) is the state transition function from
// a particular state in the HTML5 parser's state machine. It updates the
// parser's fields depending on parser.tok (where ErrorToken means EOF).
// It returns whether the token was consumed.
type insertionMode func(*parser) bool

// setOriginalIM sets the insertion mode to return to after completing a text or
// inTableText insertion mode.
// Section 12.2.3.1, "using the rules for".
func (p *parser) setOriginalIM() {
	if p.originalIM != nil {
		panic("html: bad parser state: originalIM was set twice")
	}
	p.originalIM = p.im
}

// Section 12.2.3.1, "reset the insertion mode".
func (p *parser) resetInsertionMode() {
	for i := len(p.oe) - 1; i >= 0; i-- {
		n := p.oe[i]
		if i == 0 && p.context != nil {
			n = p.context
		}

		switch n.Data {
		case "select":
			p.im = inSelectIM
		case "td", "th":
			p.im = inCellIM
		case "tr":
			p.im = inRowIM
		case "tbody", "thead", "tfoot":
			p.im = inTableBodyIM
		case "caption":
			p.im = inCaptionIM
		case "colgroup":
			p.im = inColumnGroupIM
		case "table":
			p.im = inTableIM
		case "head":
			p.im = inBodyIM
		case "body":
			p.im = inBodyIM
		case "frameset":
			p.im = inFramesetIM
		case "html":
			p.im = beforeHeadIM
		default:
			continue
		}
		return
	}
	p.im = inBodyIM
}

const whitespace = " \t\r\n\f"

// Section 12.2.5.4.1.
func initialIM(p *parser) bool {
	switch p.tok.Type {
	case TextToken:
		p.tok.Data = strings.TrimLeft(p.tok.Data, whitespace)
		if len(p.tok.Data) == 0 {
			// It was all whitespace, so ignore it.
			return true
		}
	case CommentToken:
		p.doc.Add(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	case DoctypeToken:
		n, quirks := parseDoctype(p.tok.Data)
		p.doc.Add(n)
		p.quirks = quirks
		p.im = beforeHTMLIM
		return true
	}
	p.quirks = true
	p.im = beforeHTMLIM
	return false
}

// Section 12.2.5.4.2.
func beforeHTMLIM(p *parser) bool {
	switch p.tok.Type {
	case TextToken:
		p.tok.Data = strings.TrimLeft(p.tok.Data, whitespace)
		if len(p.tok.Data) == 0 {
			// It was all whitespace, so ignore it.
			return true
		}
	case StartTagToken:
		if p.tok.Data == "html" {
			p.addElement(p.tok.Data, p.tok.Attr)
			p.im = beforeHeadIM
			return true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "head", "body", "html", "br":
			// Drop down to creating an implied <html> tag.
		default:
			// Ignore the token.
			return true
		}
	case CommentToken:
		p.doc.Add(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	// Create an implied <html> tag.
	p.addElement("html", nil)
	p.im = beforeHeadIM
	return false
}

// Section 12.2.5.4.3.
func beforeHeadIM(p *parser) bool {
	var (
		add     bool
		attr    []Attribute
		implied bool
	)
	switch p.tok.Type {
	case ErrorToken:
		implied = true
	case TextToken:
		p.tok.Data = strings.TrimLeft(p.tok.Data, whitespace)
		if len(p.tok.Data) == 0 {
			// It was all whitespace, so ignore it.
			return true
		}
		implied = true
	case StartTagToken:
		switch p.tok.Data {
		case "head":
			add = true
			attr = p.tok.Attr
		case "html":
			return inBodyIM(p)
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
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	if add || implied {
		p.addElement("head", attr)
		p.head = p.top()
	}
	p.im = inHeadIM
	return !implied
}

// Section 12.2.5.4.4.
func inHeadIM(p *parser) bool {
	var (
		pop     bool
		implied bool
	)
	switch p.tok.Type {
	case ErrorToken:
		implied = true
	case TextToken:
		s := strings.TrimLeft(p.tok.Data, whitespace)
		if len(s) < len(p.tok.Data) {
			// Add the initial whitespace to the current node.
			p.addText(p.tok.Data[:len(p.tok.Data)-len(s)])
			if s == "" {
				return true
			}
			p.tok.Data = s
		}
		implied = true
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			return inBodyIM(p)
		case "base", "basefont", "bgsound", "command", "link", "meta":
			p.addElement(p.tok.Data, p.tok.Attr)
			p.oe.pop()
			p.acknowledgeSelfClosingTag()
		case "script", "title", "noscript", "noframes", "style":
			p.addElement(p.tok.Data, p.tok.Attr)
			p.setOriginalIM()
			p.im = textIM
			return true
		case "head":
			// Ignore the token.
			return true
		default:
			implied = true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "head":
			pop = true
		case "body", "html", "br":
			implied = true
		default:
			// Ignore the token.
			return true
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	if pop || implied {
		n := p.oe.pop()
		if n.Data != "head" {
			panic("html: bad parser state: <head> element not found, in the in-head insertion mode")
		}
		p.im = afterHeadIM
		return !implied
	}
	return true
}

// Section 12.2.5.4.6.
func afterHeadIM(p *parser) bool {
	var (
		add        bool
		attr       []Attribute
		framesetOK bool
		implied    bool
	)
	switch p.tok.Type {
	case ErrorToken:
		implied = true
		framesetOK = true
	case TextToken:
		s := strings.TrimLeft(p.tok.Data, whitespace)
		if len(s) < len(p.tok.Data) {
			// Add the initial whitespace to the current node.
			p.addText(p.tok.Data[:len(p.tok.Data)-len(s)])
			if s == "" {
				return true
			}
			p.tok.Data = s
		}
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
			p.addElement(p.tok.Data, p.tok.Attr)
			p.im = inFramesetIM
			return true
		case "base", "basefont", "bgsound", "link", "meta", "noframes", "script", "style", "title":
			p.oe = append(p.oe, p.head)
			defer p.oe.pop()
			return inHeadIM(p)
		case "head":
			// Ignore the token.
			return true
		default:
			implied = true
			framesetOK = true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "body", "html", "br":
			implied = true
			framesetOK = true
		default:
			// Ignore the token.
			return true
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	if add || implied {
		p.addElement("body", attr)
		p.framesetOK = framesetOK
	}
	p.im = inBodyIM
	return !implied
}

// copyAttributes copies attributes of src not found on dst to dst.
func copyAttributes(dst *Node, src Token) {
	if len(src.Attr) == 0 {
		return
	}
	attr := map[string]string{}
	for _, a := range dst.Attr {
		attr[a.Key] = a.Val
	}
	for _, a := range src.Attr {
		if _, ok := attr[a.Key]; !ok {
			dst.Attr = append(dst.Attr, a)
			attr[a.Key] = a.Val
		}
	}
}

// Section 12.2.5.4.7.
func inBodyIM(p *parser) bool {
	switch p.tok.Type {
	case TextToken:
		switch n := p.oe.top(); n.Data {
		case "pre", "listing", "textarea":
			if len(n.Child) == 0 {
				// Ignore a newline at the start of a <pre> block.
				d := p.tok.Data
				if d != "" && d[0] == '\r' {
					d = d[1:]
				}
				if d != "" && d[0] == '\n' {
					d = d[1:]
				}
				if d == "" {
					return true
				}
				p.tok.Data = d
			}
		}
		p.reconstructActiveFormattingElements()
		p.addText(p.tok.Data)
		p.framesetOK = false
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			copyAttributes(p.oe[0], p.tok)
		case "address", "article", "aside", "blockquote", "center", "details", "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "header", "hgroup", "menu", "nav", "ol", "p", "section", "summary", "ul":
			p.popUntil(buttonScope, "p")
			p.addElement(p.tok.Data, p.tok.Attr)
		case "h1", "h2", "h3", "h4", "h5", "h6":
			p.popUntil(buttonScope, "p")
			switch n := p.top(); n.Data {
			case "h1", "h2", "h3", "h4", "h5", "h6":
				p.oe.pop()
			}
			p.addElement(p.tok.Data, p.tok.Attr)
		case "a":
			for i := len(p.afe) - 1; i >= 0 && p.afe[i].Type != scopeMarkerNode; i-- {
				if n := p.afe[i]; n.Type == ElementNode && n.Data == "a" {
					p.inBodyEndTagFormatting("a")
					p.oe.remove(n)
					p.afe.remove(n)
					break
				}
			}
			p.reconstructActiveFormattingElements()
			p.addFormattingElement(p.tok.Data, p.tok.Attr)
		case "b", "big", "code", "em", "font", "i", "s", "small", "strike", "strong", "tt", "u":
			p.reconstructActiveFormattingElements()
			p.addFormattingElement(p.tok.Data, p.tok.Attr)
		case "nobr":
			p.reconstructActiveFormattingElements()
			if p.elementInScope(defaultScope, "nobr") {
				p.inBodyEndTagFormatting("nobr")
				p.reconstructActiveFormattingElements()
			}
			p.addFormattingElement(p.tok.Data, p.tok.Attr)
		case "applet", "marquee", "object":
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.afe = append(p.afe, &scopeMarker)
			p.framesetOK = false
		case "area", "br", "embed", "img", "input", "keygen", "wbr":
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.oe.pop()
			p.acknowledgeSelfClosingTag()
			p.framesetOK = false
		case "table":
			if !p.quirks {
				p.popUntil(buttonScope, "p")
			}
			p.addElement(p.tok.Data, p.tok.Attr)
			p.framesetOK = false
			p.im = inTableIM
			return true
		case "hr":
			p.popUntil(buttonScope, "p")
			p.addElement(p.tok.Data, p.tok.Attr)
			p.oe.pop()
			p.acknowledgeSelfClosingTag()
			p.framesetOK = false
		case "select":
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.framesetOK = false
			p.im = inSelectIM
			return true
		case "form":
			if p.form == nil {
				p.popUntil(buttonScope, "p")
				p.addElement(p.tok.Data, p.tok.Attr)
				p.form = p.top()
			}
		case "li":
			p.framesetOK = false
			for i := len(p.oe) - 1; i >= 0; i-- {
				node := p.oe[i]
				switch node.Data {
				case "li":
					p.popUntil(listItemScope, "li")
				case "address", "div", "p":
					continue
				default:
					if !isSpecialElement(node) {
						continue
					}
				}
				break
			}
			p.popUntil(buttonScope, "p")
			p.addElement(p.tok.Data, p.tok.Attr)
		case "dd", "dt":
			p.framesetOK = false
			for i := len(p.oe) - 1; i >= 0; i-- {
				node := p.oe[i]
				switch node.Data {
				case "dd", "dt":
					p.oe = p.oe[:i]
				case "address", "div", "p":
					continue
				default:
					if !isSpecialElement(node) {
						continue
					}
				}
				break
			}
			p.popUntil(buttonScope, "p")
			p.addElement(p.tok.Data, p.tok.Attr)
		case "plaintext":
			p.popUntil(buttonScope, "p")
			p.addElement(p.tok.Data, p.tok.Attr)
		case "button":
			p.popUntil(defaultScope, "button")
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.framesetOK = false
		case "optgroup", "option":
			if p.top().Data == "option" {
				p.oe.pop()
			}
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
		case "body":
			if len(p.oe) >= 2 {
				body := p.oe[1]
				if body.Type == ElementNode && body.Data == "body" {
					p.framesetOK = false
					copyAttributes(body, p.tok)
				}
			}
		case "frameset":
			if !p.framesetOK || len(p.oe) < 2 || p.oe[1].Data != "body" {
				// Ignore the token.
				return true
			}
			body := p.oe[1]
			if body.Parent != nil {
				body.Parent.Remove(body)
			}
			p.oe = p.oe[:1]
			p.addElement(p.tok.Data, p.tok.Attr)
			p.im = inFramesetIM
			return true
		case "base", "basefont", "bgsound", "command", "link", "meta", "noframes", "script", "style", "title":
			return inHeadIM(p)
		case "image":
			p.tok.Data = "img"
			return false
		case "isindex":
			if p.form != nil {
				// Ignore the token.
				return true
			}
			action := ""
			prompt := "This is a searchable index. Enter search keywords: "
			attr := []Attribute{{Key: "name", Val: "isindex"}}
			for _, a := range p.tok.Attr {
				switch a.Key {
				case "action":
					action = a.Val
				case "name":
					// Ignore the attribute.
				case "prompt":
					prompt = a.Val
				default:
					attr = append(attr, a)
				}
			}
			p.acknowledgeSelfClosingTag()
			p.popUntil(buttonScope, "p")
			p.addElement("form", nil)
			p.form = p.top()
			if action != "" {
				p.form.Attr = []Attribute{{Key: "action", Val: action}}
			}
			p.addElement("hr", nil)
			p.oe.pop()
			p.addElement("label", nil)
			p.addText(prompt)
			p.addElement("input", attr)
			p.oe.pop()
			p.oe.pop()
			p.addElement("hr", nil)
			p.oe.pop()
			p.oe.pop()
			p.form = nil
		case "xmp":
			p.popUntil(buttonScope, "p")
			p.reconstructActiveFormattingElements()
			p.framesetOK = false
			p.addElement(p.tok.Data, p.tok.Attr)
		case "math", "svg":
			p.reconstructActiveFormattingElements()
			if p.tok.Data == "math" {
				// TODO: adjust MathML attributes.
			} else {
				// TODO: adjust SVG attributes.
			}
			adjustForeignAttributes(p.tok.Attr)
			p.addElement(p.tok.Data, p.tok.Attr)
			p.top().Namespace = p.tok.Data
			return true
		case "caption", "col", "colgroup", "frame", "head", "tbody", "td", "tfoot", "th", "thead", "tr":
			// Ignore the token.
		default:
			// TODO.
			p.addElement(p.tok.Data, p.tok.Attr)
		}
	case EndTagToken:
		switch p.tok.Data {
		case "body":
			// TODO: autoclose the stack of open elements.
			p.im = afterBodyIM
			return true
		case "p":
			if !p.elementInScope(buttonScope, "p") {
				p.addElement("p", nil)
			}
			p.popUntil(buttonScope, "p")
		case "a", "b", "big", "code", "em", "font", "i", "nobr", "s", "small", "strike", "strong", "tt", "u":
			p.inBodyEndTagFormatting(p.tok.Data)
		case "address", "article", "aside", "blockquote", "button", "center", "details", "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "header", "hgroup", "listing", "menu", "nav", "ol", "pre", "section", "summary", "ul":
			p.popUntil(defaultScope, p.tok.Data)
		case "applet", "marquee", "object":
			if p.popUntil(defaultScope, p.tok.Data) {
				p.clearActiveFormattingElements()
			}
		case "br":
			p.tok.Type = StartTagToken
			return false
		default:
			p.inBodyEndTagOther(p.tok.Data)
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
	}

	return true
}

func (p *parser) inBodyEndTagFormatting(tag string) {
	// This is the "adoption agency" algorithm, described at
	// http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html#adoptionAgency

	// TODO: this is a fairly literal line-by-line translation of that algorithm.
	// Once the code successfully parses the comprehensive test suite, we should
	// refactor this code to be more idiomatic.

	// Steps 1-3. The outer loop.
	for i := 0; i < 8; i++ {
		// Step 4. Find the formatting element.
		var formattingElement *Node
		for j := len(p.afe) - 1; j >= 0; j-- {
			if p.afe[j].Type == scopeMarkerNode {
				break
			}
			if p.afe[j].Data == tag {
				formattingElement = p.afe[j]
				break
			}
		}
		if formattingElement == nil {
			p.inBodyEndTagOther(tag)
			return
		}
		feIndex := p.oe.index(formattingElement)
		if feIndex == -1 {
			p.afe.remove(formattingElement)
			return
		}
		if !p.elementInScope(defaultScope, tag) {
			// Ignore the tag.
			return
		}

		// Steps 5-6. Find the furthest block.
		var furthestBlock *Node
		for _, e := range p.oe[feIndex:] {
			if isSpecialElement(e) {
				furthestBlock = e
				break
			}
		}
		if furthestBlock == nil {
			e := p.oe.pop()
			for e != formattingElement {
				e = p.oe.pop()
			}
			p.afe.remove(e)
			return
		}

		// Steps 7-8. Find the common ancestor and bookmark node.
		commonAncestor := p.oe[feIndex-1]
		bookmark := p.afe.index(formattingElement)

		// Step 9. The inner loop. Find the lastNode to reparent.
		lastNode := furthestBlock
		node := furthestBlock
		x := p.oe.index(node)
		// Steps 9.1-9.3.
		for j := 0; j < 3; j++ {
			// Step 9.4.
			x--
			node = p.oe[x]
			// Step 9.5.
			if p.afe.index(node) == -1 {
				p.oe.remove(node)
				continue
			}
			// Step 9.6.
			if node == formattingElement {
				break
			}
			// Step 9.7.
			clone := node.clone()
			p.afe[p.afe.index(node)] = clone
			p.oe[p.oe.index(node)] = clone
			node = clone
			// Step 9.8.
			if lastNode == furthestBlock {
				bookmark = p.afe.index(node) + 1
			}
			// Step 9.9.
			if lastNode.Parent != nil {
				lastNode.Parent.Remove(lastNode)
			}
			node.Add(lastNode)
			// Step 9.10.
			lastNode = node
		}

		// Step 10. Reparent lastNode to the common ancestor,
		// or for misnested table nodes, to the foster parent.
		if lastNode.Parent != nil {
			lastNode.Parent.Remove(lastNode)
		}
		switch commonAncestor.Data {
		case "table", "tbody", "tfoot", "thead", "tr":
			p.fosterParent(lastNode)
		default:
			commonAncestor.Add(lastNode)
		}

		// Steps 11-13. Reparent nodes from the furthest block's children
		// to a clone of the formatting element.
		clone := formattingElement.clone()
		reparentChildren(clone, furthestBlock)
		furthestBlock.Add(clone)

		// Step 14. Fix up the list of active formatting elements.
		if oldLoc := p.afe.index(formattingElement); oldLoc != -1 && oldLoc < bookmark {
			// Move the bookmark with the rest of the list.
			bookmark--
		}
		p.afe.remove(formattingElement)
		p.afe.insert(bookmark, clone)

		// Step 15. Fix up the stack of open elements.
		p.oe.remove(formattingElement)
		p.oe.insert(p.oe.index(furthestBlock)+1, clone)
	}
}

// inBodyEndTagOther performs the "any other end tag" algorithm for inBodyIM.
func (p *parser) inBodyEndTagOther(tag string) {
	for i := len(p.oe) - 1; i >= 0; i-- {
		if p.oe[i].Data == tag {
			p.oe = p.oe[:i]
			break
		}
		if isSpecialElement(p.oe[i]) {
			break
		}
	}
}

// Section 12.2.5.4.8.
func textIM(p *parser) bool {
	switch p.tok.Type {
	case ErrorToken:
		p.oe.pop()
	case TextToken:
		p.addText(p.tok.Data)
		return true
	case EndTagToken:
		p.oe.pop()
	}
	p.im = p.originalIM
	p.originalIM = nil
	return p.tok.Type == EndTagToken
}

// Section 12.2.5.4.9.
func inTableIM(p *parser) bool {
	switch p.tok.Type {
	case ErrorToken:
		// Stop parsing.
		return true
	case TextToken:
		// TODO.
	case StartTagToken:
		switch p.tok.Data {
		case "caption":
			p.clearStackToContext(tableScope)
			p.afe = append(p.afe, &scopeMarker)
			p.addElement(p.tok.Data, p.tok.Attr)
			p.im = inCaptionIM
			return true
		case "tbody", "tfoot", "thead":
			p.clearStackToContext(tableScope)
			p.addElement(p.tok.Data, p.tok.Attr)
			p.im = inTableBodyIM
			return true
		case "td", "th", "tr":
			p.clearStackToContext(tableScope)
			p.addElement("tbody", nil)
			p.im = inTableBodyIM
			return false
		case "table":
			if p.popUntil(tableScope, "table") {
				p.resetInsertionMode()
				return false
			}
			// Ignore the token.
			return true
		case "colgroup":
			p.clearStackToContext(tableScope)
			p.addElement(p.tok.Data, p.tok.Attr)
			p.im = inColumnGroupIM
			return true
		case "col":
			p.clearStackToContext(tableScope)
			p.addElement("colgroup", p.tok.Attr)
			p.im = inColumnGroupIM
			return false
		case "select":
			p.reconstructActiveFormattingElements()
			switch p.top().Data {
			case "table", "tbody", "tfoot", "thead", "tr":
				p.fosterParenting = true
			}
			p.addElement(p.tok.Data, p.tok.Attr)
			p.fosterParenting = false
			p.framesetOK = false
			p.im = inSelectInTableIM
			return true
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "table":
			if p.popUntil(tableScope, "table") {
				p.resetInsertionMode()
				return true
			}
			// Ignore the token.
			return true
		case "body", "caption", "col", "colgroup", "html", "tbody", "td", "tfoot", "th", "thead", "tr":
			// Ignore the token.
			return true
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}

	switch p.top().Data {
	case "table", "tbody", "tfoot", "thead", "tr":
		p.fosterParenting = true
		defer func() { p.fosterParenting = false }()
	}

	return inBodyIM(p)
}

// Section 12.2.5.4.11.
func inCaptionIM(p *parser) bool {
	switch p.tok.Type {
	case StartTagToken:
		switch p.tok.Data {
		case "caption", "col", "colgroup", "tbody", "td", "tfoot", "thead", "tr":
			if p.popUntil(tableScope, "caption") {
				p.clearActiveFormattingElements()
				p.im = inTableIM
				return false
			} else {
				// Ignore the token.
				return true
			}
		case "select":
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.framesetOK = false
			p.im = inSelectInTableIM
			return true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "caption":
			if p.popUntil(tableScope, "caption") {
				p.clearActiveFormattingElements()
				p.im = inTableIM
			}
			return true
		case "table":
			if p.popUntil(tableScope, "caption") {
				p.clearActiveFormattingElements()
				p.im = inTableIM
				return false
			} else {
				// Ignore the token.
				return true
			}
		case "body", "col", "colgroup", "html", "tbody", "td", "tfoot", "th", "thead", "tr":
			// Ignore the token.
			return true
		}
	}
	return inBodyIM(p)
}

// Section 12.2.5.4.12.
func inColumnGroupIM(p *parser) bool {
	switch p.tok.Type {
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	case DoctypeToken:
		// Ignore the token.
		return true
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			return inBodyIM(p)
		case "col":
			p.addElement(p.tok.Data, p.tok.Attr)
			p.oe.pop()
			p.acknowledgeSelfClosingTag()
			return true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "colgroup":
			if p.oe.top().Data != "html" {
				p.oe.pop()
				p.im = inTableIM
			}
			return true
		case "col":
			// Ignore the token.
			return true
		}
	}
	if p.oe.top().Data != "html" {
		p.oe.pop()
		p.im = inTableIM
		return false
	}
	return true
}

// Section 12.2.5.4.13.
func inTableBodyIM(p *parser) bool {
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
		case "caption", "col", "colgroup", "tbody", "tfoot", "thead":
			if !p.popUntil(tableScope, "tbody", "thead", "tfoot") {
				// Ignore the token.
				return true
			}
			p.im = inTableIM
			return false
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "table":
			if p.popUntil(tableScope, "tbody", "thead", "tfoot") {
				p.im = inTableIM
				return false
			}
			// Ignore the token.
			return true
		case "body", "caption", "col", "colgroup", "html", "td", "th", "tr":
			// Ignore the token.
			return true
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	if add {
		// TODO: clear the stack back to a table body context.
		p.addElement(data, attr)
		p.im = inRowIM
		return consumed
	}
	return inTableIM(p)
}

// Section 12.2.5.4.14.
func inRowIM(p *parser) bool {
	switch p.tok.Type {
	case ErrorToken:
		// TODO.
	case TextToken:
		// TODO.
	case StartTagToken:
		switch p.tok.Data {
		case "td", "th":
			p.clearStackToContext(tableRowScope)
			p.addElement(p.tok.Data, p.tok.Attr)
			p.afe = append(p.afe, &scopeMarker)
			p.im = inCellIM
			return true
		case "caption", "col", "colgroup", "tbody", "tfoot", "thead", "tr":
			if p.popUntil(tableScope, "tr") {
				p.im = inTableBodyIM
				return false
			}
			// Ignore the token.
			return true
		default:
			// TODO.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "tr":
			if p.popUntil(tableScope, "tr") {
				p.im = inTableBodyIM
				return true
			}
			// Ignore the token.
			return true
		case "table":
			if p.popUntil(tableScope, "tr") {
				p.im = inTableBodyIM
				return false
			}
			// Ignore the token.
			return true
		case "tbody", "tfoot", "thead":
			// TODO.
		case "body", "caption", "col", "colgroup", "html", "td", "th":
			// Ignore the token.
			return true
		default:
			// TODO.
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	return inTableIM(p)
}

// Section 12.2.5.4.15.
func inCellIM(p *parser) bool {
	var (
		closeTheCellAndReprocess bool
	)
	switch p.tok.Type {
	case StartTagToken:
		switch p.tok.Data {
		case "caption", "col", "colgroup", "tbody", "td", "tfoot", "th", "thead", "tr":
			// TODO: check for "td" or "th" in table scope.
			closeTheCellAndReprocess = true
		case "select":
			p.reconstructActiveFormattingElements()
			p.addElement(p.tok.Data, p.tok.Attr)
			p.framesetOK = false
			p.im = inSelectInTableIM
			return true
		}
	case EndTagToken:
		switch p.tok.Data {
		case "td", "th":
			if !p.popUntil(tableScope, p.tok.Data) {
				// Ignore the token.
				return true
			}
			p.clearActiveFormattingElements()
			p.im = inRowIM
			return true
		case "body", "caption", "col", "colgroup", "html":
			// TODO.
		case "table", "tbody", "tfoot", "thead", "tr":
			// TODO: check for matching element in table scope.
			closeTheCellAndReprocess = true
		}
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	if closeTheCellAndReprocess {
		if p.popUntil(tableScope, "td") || p.popUntil(tableScope, "th") {
			p.clearActiveFormattingElements()
			p.im = inRowIM
			return false
		}
	}
	return inBodyIM(p)
}

// Section 12.2.5.4.16.
func inSelectIM(p *parser) bool {
	endSelect := false
	switch p.tok.Type {
	case ErrorToken:
		// TODO.
	case TextToken:
		p.addText(p.tok.Data)
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			// TODO.
		case "option":
			if p.top().Data == "option" {
				p.oe.pop()
			}
			p.addElement(p.tok.Data, p.tok.Attr)
		case "optgroup":
			if p.top().Data == "option" {
				p.oe.pop()
			}
			if p.top().Data == "optgroup" {
				p.oe.pop()
			}
			p.addElement(p.tok.Data, p.tok.Attr)
		case "select":
			endSelect = true
		case "input", "keygen", "textarea":
			// TODO.
		case "script":
			// TODO.
		default:
			// Ignore the token.
		}
	case EndTagToken:
		switch p.tok.Data {
		case "option":
			if p.top().Data == "option" {
				p.oe.pop()
			}
		case "optgroup":
			i := len(p.oe) - 1
			if p.oe[i].Data == "option" {
				i--
			}
			if p.oe[i].Data == "optgroup" {
				p.oe = p.oe[:i]
			}
		case "select":
			endSelect = true
		default:
			// Ignore the token.
		}
	case CommentToken:
		p.doc.Add(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
	}
	if endSelect {
		p.endSelect()
	}
	return true
}

// Section 12.2.5.4.17.
func inSelectInTableIM(p *parser) bool {
	switch p.tok.Type {
	case StartTagToken, EndTagToken:
		switch p.tok.Data {
		case "caption", "table", "tbody", "tfoot", "thead", "tr", "td", "th":
			if p.tok.Type == StartTagToken || p.elementInScope(tableScope, p.tok.Data) {
				p.endSelect()
				return false
			} else {
				// Ignore the token.
				return true
			}
		}
	}
	return inSelectIM(p)
}

func (p *parser) endSelect() {
	for i := len(p.oe) - 1; i >= 0; i-- {
		switch p.oe[i].Data {
		case "option", "optgroup":
			continue
		case "select":
			p.oe = p.oe[:i]
			p.resetInsertionMode()
		}
		return
	}
}

// Section 12.2.5.4.18.
func afterBodyIM(p *parser) bool {
	switch p.tok.Type {
	case ErrorToken:
		// Stop parsing.
		return true
	case StartTagToken:
		if p.tok.Data == "html" {
			return inBodyIM(p)
		}
	case EndTagToken:
		if p.tok.Data == "html" {
			p.im = afterAfterBodyIM
			return true
		}
	case CommentToken:
		// The comment is attached to the <html> element.
		if len(p.oe) < 1 || p.oe[0].Data != "html" {
			panic("html: bad parser state: <html> element not found, in the after-body insertion mode")
		}
		p.oe[0].Add(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	p.im = inBodyIM
	return false
}

// Section 12.2.5.4.19.
func inFramesetIM(p *parser) bool {
	switch p.tok.Type {
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
	case TextToken:
		// Ignore all text but whitespace.
		s := strings.Map(func(c rune) rune {
			switch c {
			case ' ', '\t', '\n', '\f', '\r':
				return c
			}
			return -1
		}, p.tok.Data)
		if s != "" {
			p.addText(s)
		}
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			return inBodyIM(p)
		case "frameset":
			p.addElement(p.tok.Data, p.tok.Attr)
		case "frame":
			p.addElement(p.tok.Data, p.tok.Attr)
			p.oe.pop()
			p.acknowledgeSelfClosingTag()
		case "noframes":
			return inHeadIM(p)
		}
	case EndTagToken:
		switch p.tok.Data {
		case "frameset":
			if p.oe.top().Data != "html" {
				p.oe.pop()
				if p.oe.top().Data != "frameset" {
					p.im = afterFramesetIM
					return true
				}
			}
		}
	default:
		// Ignore the token.
	}
	return true
}

// Section 12.2.5.4.20.
func afterFramesetIM(p *parser) bool {
	switch p.tok.Type {
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
	case TextToken:
		// Ignore all text but whitespace.
		s := strings.Map(func(c rune) rune {
			switch c {
			case ' ', '\t', '\n', '\f', '\r':
				return c
			}
			return -1
		}, p.tok.Data)
		if s != "" {
			p.addText(s)
		}
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			return inBodyIM(p)
		case "noframes":
			return inHeadIM(p)
		}
	case EndTagToken:
		switch p.tok.Data {
		case "html":
			p.im = afterAfterFramesetIM
			return true
		}
	default:
		// Ignore the token.
	}
	return true
}

// Section 12.2.5.4.21.
func afterAfterBodyIM(p *parser) bool {
	switch p.tok.Type {
	case ErrorToken:
		// Stop parsing.
		return true
	case TextToken:
		// TODO.
	case StartTagToken:
		if p.tok.Data == "html" {
			return inBodyIM(p)
		}
	case CommentToken:
		p.doc.Add(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
		return true
	}
	p.im = inBodyIM
	return false
}

// Section 12.2.5.4.22.
func afterAfterFramesetIM(p *parser) bool {
	switch p.tok.Type {
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
	case TextToken:
		// Ignore all text but whitespace.
		s := strings.Map(func(c rune) rune {
			switch c {
			case ' ', '\t', '\n', '\f', '\r':
				return c
			}
			return -1
		}, p.tok.Data)
		if s != "" {
			p.reconstructActiveFormattingElements()
			p.addText(s)
		}
	case StartTagToken:
		switch p.tok.Data {
		case "html":
			return inBodyIM(p)
		case "noframes":
			return inHeadIM(p)
		}
	default:
		// Ignore the token.
	}
	return true
}

// Section 12.2.5.5.
func parseForeignContent(p *parser) bool {
	switch p.tok.Type {
	case TextToken:
		// TODO: HTML integration points.
		if p.top().Namespace == "" {
			inBodyIM(p)
			p.resetInsertionMode()
			return true
		}
		if p.framesetOK {
			p.framesetOK = strings.TrimLeft(p.tok.Data, whitespace) == ""
		}
		p.addText(p.tok.Data)
	case CommentToken:
		p.addChild(&Node{
			Type: CommentNode,
			Data: p.tok.Data,
		})
	case StartTagToken:
		if htmlIntegrationPoint(p.top()) {
			inBodyIM(p)
			p.resetInsertionMode()
			return true
		}
		if breakout[p.tok.Data] {
			for i := len(p.oe) - 1; i >= 0; i-- {
				// TODO: HTML, MathML integration points.
				if p.oe[i].Namespace == "" {
					p.oe = p.oe[:i+1]
					break
				}
			}
			return false
		}
		switch p.top().Namespace {
		case "math":
			// TODO: adjust MathML attributes.
		case "svg":
			// Adjust SVG tag names. The tokenizer lower-cases tag names, but
			// SVG wants e.g. "foreignObject" with a capital second "O".
			if x := svgTagNameAdjustments[p.tok.Data]; x != "" {
				p.tok.Data = x
			}
			// TODO: adjust SVG attributes.
		default:
			panic("html: bad parser state: unexpected namespace")
		}
		adjustForeignAttributes(p.tok.Attr)
		namespace := p.top().Namespace
		p.addElement(p.tok.Data, p.tok.Attr)
		p.top().Namespace = namespace
	case EndTagToken:
		for i := len(p.oe) - 1; i >= 0; i-- {
			if p.oe[i].Namespace == "" {
				return p.im(p)
			}
			if strings.EqualFold(p.oe[i].Data, p.tok.Data) {
				p.oe = p.oe[:i]
				break
			}
		}
		return true
	default:
		// Ignore the token.
	}
	return true
}

// Section 12.2.5.
func (p *parser) inForeignContent() bool {
	if len(p.oe) == 0 {
		return false
	}
	n := p.oe[len(p.oe)-1]
	if n.Namespace == "" {
		return false
	}
	// TODO: MathML, HTML integration points.
	// TODO: MathML's annotation-xml combining with SVG's svg.
	return true
}

func (p *parser) parse() error {
	// Iterate until EOF. Any other error will cause an early return.
	consumed := true
	for {
		if consumed {
			if err := p.read(); err != nil {
				if err == io.EOF {
					break
				}
				return err
			}
		}
		if p.inForeignContent() {
			consumed = parseForeignContent(p)
		} else {
			consumed = p.im(p)
		}
	}
	// Loop until the final token (the ErrorToken signifying EOF) is consumed.
	for {
		if consumed = p.im(p); consumed {
			break
		}
	}
	return nil
}

// Parse returns the parse tree for the HTML from the given Reader.
// The input is assumed to be UTF-8 encoded.
func Parse(r io.Reader) (*Node, error) {
	p := &parser{
		tokenizer: NewTokenizer(r),
		doc: &Node{
			Type: DocumentNode,
		},
		scripting:  true,
		framesetOK: true,
		im:         initialIM,
	}
	err := p.parse()
	if err != nil {
		return nil, err
	}
	return p.doc, nil
}

// ParseFragment parses a fragment of HTML and returns the nodes that were 
// found. If the fragment is the InnerHTML for an existing element, pass that
// element in context.
func ParseFragment(r io.Reader, context *Node) ([]*Node, error) {
	p := &parser{
		tokenizer: NewTokenizer(r),
		doc: &Node{
			Type: DocumentNode,
		},
		scripting: true,
		context:   context,
	}

	if context != nil {
		switch context.Data {
		case "iframe", "noembed", "noframes", "noscript", "plaintext", "script", "style", "title", "textarea", "xmp":
			p.tokenizer.rawTag = context.Data
		}
	}

	root := &Node{
		Type: ElementNode,
		Data: "html",
	}
	p.doc.Add(root)
	p.oe = nodeStack{root}
	p.resetInsertionMode()

	for n := context; n != nil; n = n.Parent {
		if n.Type == ElementNode && n.Data == "form" {
			p.form = n
			break
		}
	}

	err := p.parse()
	if err != nil {
		return nil, err
	}

	parent := p.doc
	if context != nil {
		parent = root
	}

	result := parent.Child
	parent.Child = nil
	for _, n := range result {
		n.Parent = nil
	}
	return result, nil
}
