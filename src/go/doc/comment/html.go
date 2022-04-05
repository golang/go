// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"bytes"
	"fmt"
)

// An htmlPrinter holds the state needed for printing a Doc as HTML.
type htmlPrinter struct {
	*Printer
}

// HTML returns an HTML formatting of the Doc.
// See the [Printer] documentation for ways to customize the HTML output.
func (p *Printer) HTML(d *Doc) []byte {
	hp := &htmlPrinter{Printer: p}
	var out bytes.Buffer
	for _, x := range d.Content {
		hp.block(&out, x)
	}
	return out.Bytes()
}

// block prints the block x to out.
func (p *htmlPrinter) block(out *bytes.Buffer, x Block) {
	switch x := x.(type) {
	default:
		fmt.Fprintf(out, "?%T", x)

	case *Paragraph:
		out.WriteString("<p>")
		p.text(out, x.Text)
		out.WriteString("\n")
	}
}

// text prints the text sequence x to out.
func (p *htmlPrinter) text(out *bytes.Buffer, x []Text) {
	for _, t := range x {
		switch t := t.(type) {
		case Plain:
			p.escape(out, string(t))
		case Italic:
			out.WriteString("<i>")
			p.escape(out, string(t))
			out.WriteString("</i>")
		case *Link:
			out.WriteString(`<a href="`)
			p.escape(out, t.URL)
			out.WriteString(`">`)
			p.text(out, t.Text)
			out.WriteString("</a>")
		case *DocLink:
			url := p.docLinkURL(t)
			if url != "" {
				out.WriteString(`<a href="`)
				p.escape(out, url)
				out.WriteString(`">`)
			}
			p.text(out, t.Text)
			if url != "" {
				out.WriteString("</a>")
			}
		}
	}
}

// escape prints s to out as plain text,
// escaping < & " ' and > to avoid being misinterpreted
// in larger HTML constructs.
func (p *htmlPrinter) escape(out *bytes.Buffer, s string) {
	start := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '<':
			out.WriteString(s[start:i])
			out.WriteString("&lt;")
			start = i + 1
		case '&':
			out.WriteString(s[start:i])
			out.WriteString("&amp;")
			start = i + 1
		case '"':
			out.WriteString(s[start:i])
			out.WriteString("&quot;")
			start = i + 1
		case '\'':
			out.WriteString(s[start:i])
			out.WriteString("&apos;")
			start = i + 1
		case '>':
			out.WriteString(s[start:i])
			out.WriteString("&gt;")
			start = i + 1
		}
	}
	out.WriteString(s[start:])
}
