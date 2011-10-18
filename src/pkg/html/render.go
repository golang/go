// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

type writer interface {
	io.Writer
	WriteByte(byte) os.Error
	WriteString(string) (int, os.Error)
}

// Render renders the parse tree n to the given writer.
//
// For 'well-formed' parse trees, calling Parse on the output of Render will
// result in a clone of the original tree.
//
// 'Well-formed' is not formally specified, but calling Parse on arbitrary
// input results in a 'well-formed' parse tree if Parse does not return an
// error. Programmatically constructed trees are typically also 'well-formed',
// but it is possible to construct a tree that, when rendered and re-parsed,
// results in a different tree. A simple example is that a solitary text node
// would become a tree containing <html>, <head> and <body> elements. Another
// example is that the programmatic equivalent of "a<head>b</head>c" becomes
// "<html><head><head/><body>abc</body></html>".
//
// Comment nodes are elided from the output, analogous to Parse skipping over
// any <!--comment--> input.
func Render(w io.Writer, n *Node) os.Error {
	if x, ok := w.(writer); ok {
		return render(x, n)
	}
	buf := bufio.NewWriter(w)
	if err := render(buf, n); err != nil {
		return err
	}
	return buf.Flush()
}

func render(w writer, n *Node) os.Error {
	// Render non-element nodes; these are the easy cases.
	switch n.Type {
	case ErrorNode:
		return os.NewError("html: cannot render an ErrorNode node")
	case TextNode:
		return escape(w, n.Data)
	case DocumentNode:
		for _, c := range n.Child {
			if err := render(w, c); err != nil {
				return err
			}
		}
		return nil
	case ElementNode:
		// No-op.
	case CommentNode:
		return nil
	case DoctypeNode:
		if _, err := w.WriteString("<!DOCTYPE "); err != nil {
			return err
		}
		if _, err := w.WriteString(n.Data); err != nil {
			return err
		}
		return w.WriteByte('>')
	default:
		return os.NewError("html: unknown node type")
	}

	// Render the <xxx> opening tag.
	if err := w.WriteByte('<'); err != nil {
		return err
	}
	if _, err := w.WriteString(n.Data); err != nil {
		return err
	}
	for _, a := range n.Attr {
		if err := w.WriteByte(' '); err != nil {
			return err
		}
		if _, err := w.WriteString(a.Key); err != nil {
			return err
		}
		if _, err := w.WriteString(`="`); err != nil {
			return err
		}
		if err := escape(w, a.Val); err != nil {
			return err
		}
		if err := w.WriteByte('"'); err != nil {
			return err
		}
	}
	if voidElements[n.Data] {
		if len(n.Child) != 0 {
			return fmt.Errorf("html: void element <%s> has child nodes", n.Data)
		}
		_, err := w.WriteString("/>")
		return err
	}
	if err := w.WriteByte('>'); err != nil {
		return err
	}

	// Render any child nodes.
	switch n.Data {
	case "noembed", "noframes", "noscript", "script", "style":
		for _, c := range n.Child {
			if c.Type != TextNode {
				return fmt.Errorf("html: raw text element <%s> has non-text child node", n.Data)
			}
			if _, err := w.WriteString(c.Data); err != nil {
				return err
			}
		}
	case "textarea", "title":
		for _, c := range n.Child {
			if c.Type != TextNode {
				return fmt.Errorf("html: RCDATA element <%s> has non-text child node", n.Data)
			}
			if err := render(w, c); err != nil {
				return err
			}
		}
	default:
		for _, c := range n.Child {
			if err := render(w, c); err != nil {
				return err
			}
		}
	}

	// Render the </xxx> closing tag.
	if _, err := w.WriteString("</"); err != nil {
		return err
	}
	if _, err := w.WriteString(n.Data); err != nil {
		return err
	}
	return w.WriteByte('>')
}

// Section 13.1.2, "Elements", gives this list of void elements. Void elements
// are those that can't have any contents.
var voidElements = map[string]bool{
	"area":    true,
	"base":    true,
	"br":      true,
	"col":     true,
	"command": true,
	"embed":   true,
	"hr":      true,
	"img":     true,
	"input":   true,
	"keygen":  true,
	"link":    true,
	"meta":    true,
	"param":   true,
	"source":  true,
	"track":   true,
	"wbr":     true,
}
