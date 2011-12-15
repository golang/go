// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

// Section 12.2.3.2 of the HTML5 specification says "The following elements
// have varying levels of special parsing rules".
// http://www.whatwg.org/specs/web-apps/current-work/multipage/parsing.html#the-stack-of-open-elements
var isSpecialElementMap = map[string]bool{
	"address":    true,
	"applet":     true,
	"area":       true,
	"article":    true,
	"aside":      true,
	"base":       true,
	"basefont":   true,
	"bgsound":    true,
	"blockquote": true,
	"body":       true,
	"br":         true,
	"button":     true,
	"caption":    true,
	"center":     true,
	"col":        true,
	"colgroup":   true,
	"command":    true,
	"dd":         true,
	"details":    true,
	"dir":        true,
	"div":        true,
	"dl":         true,
	"dt":         true,
	"embed":      true,
	"fieldset":   true,
	"figcaption": true,
	"figure":     true,
	"footer":     true,
	"form":       true,
	"frame":      true,
	"frameset":   true,
	"h1":         true,
	"h2":         true,
	"h3":         true,
	"h4":         true,
	"h5":         true,
	"h6":         true,
	"head":       true,
	"header":     true,
	"hgroup":     true,
	"hr":         true,
	"html":       true,
	"iframe":     true,
	"img":        true,
	"input":      true,
	"isindex":    true,
	"li":         true,
	"link":       true,
	"listing":    true,
	"marquee":    true,
	"menu":       true,
	"meta":       true,
	"nav":        true,
	"noembed":    true,
	"noframes":   true,
	"noscript":   true,
	"object":     true,
	"ol":         true,
	"p":          true,
	"param":      true,
	"plaintext":  true,
	"pre":        true,
	"script":     true,
	"section":    true,
	"select":     true,
	"style":      true,
	"summary":    true,
	"table":      true,
	"tbody":      true,
	"td":         true,
	"textarea":   true,
	"tfoot":      true,
	"th":         true,
	"thead":      true,
	"title":      true,
	"tr":         true,
	"ul":         true,
	"wbr":        true,
	"xmp":        true,
}

func isSpecialElement(element *Node) bool {
	switch element.Namespace {
	case "", "html":
		return isSpecialElementMap[element.Data]
	case "svg":
		return element.Data == "foreignObject"
	}
	return false
}
