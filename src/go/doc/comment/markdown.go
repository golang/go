// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"bytes"
	"fmt"
	"strings"
)

// An mdPrinter holds the state needed for printing a Doc as Markdown.
type mdPrinter struct {
	*Printer
	headingPrefix string
	raw           bytes.Buffer
}

// Markdown returns a Markdown formatting of the Doc.
// See the [Printer] documentation for ways to customize the Markdown output.
func (p *Printer) Markdown(d *Doc) []byte {
	mp := &mdPrinter{
		Printer:       p,
		headingPrefix: strings.Repeat("#", p.headingLevel()) + " ",
	}

	var out bytes.Buffer
	for i, x := range d.Content {
		if i > 0 {
			out.WriteByte('\n')
		}
		mp.block(&out, x)
	}
	return out.Bytes()
}

// block prints the block x to out.
func (p *mdPrinter) block(out *bytes.Buffer, x Block) {
	switch x := x.(type) {
	default:
		fmt.Fprintf(out, "?%T", x)

	case *Paragraph:
		p.text(out, x.Text)
		out.WriteString("\n")

	case *Heading:
		out.WriteString(p.headingPrefix)
		p.text(out, x.Text)
		if id := p.headingID(x); id != "" {
			out.WriteString(" {#")
			out.WriteString(id)
			out.WriteString("}")
		}
		out.WriteString("\n")

	case *Code:
		md := x.Text
		for md != "" {
			var line string
			line, md, _ = strings.Cut(md, "\n")
			if line != "" {
				out.WriteString("\t")
				out.WriteString(line)
			}
			out.WriteString("\n")
		}

	case *List:
		loose := x.BlankBetween()
		for i, item := range x.Items {
			if i > 0 && loose {
				out.WriteString("\n")
			}
			if n := item.Number; n != "" {
				out.WriteString(" ")
				out.WriteString(n)
				out.WriteString(". ")
			} else {
				out.WriteString("  - ") // SP SP - SP
			}
			for i, blk := range item.Content {
				const fourSpace = "    "
				if i > 0 {
					out.WriteString("\n" + fourSpace)
				}
				p.text(out, blk.(*Paragraph).Text)
				out.WriteString("\n")
			}
		}
	}
}

// text prints the text sequence x to out.
func (p *mdPrinter) text(out *bytes.Buffer, x []Text) {
	p.raw.Reset()
	p.rawText(&p.raw, x)
	line := bytes.TrimSpace(p.raw.Bytes())
	if len(line) == 0 {
		return
	}
	switch line[0] {
	case '+', '-', '*', '#':
		// Escape what would be the start of an unordered list or heading.
		out.WriteByte('\\')
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		i := 1
		for i < len(line) && '0' <= line[i] && line[i] <= '9' {
			i++
		}
		if i < len(line) && (line[i] == '.' || line[i] == ')') {
			// Escape what would be the start of an ordered list.
			out.Write(line[:i])
			out.WriteByte('\\')
			line = line[i:]
		}
	}
	out.Write(line)
}

// rawText prints the text sequence x to out,
// without worrying about escaping characters
// that have special meaning at the start of a Markdown line.
func (p *mdPrinter) rawText(out *bytes.Buffer, x []Text) {
	for _, t := range x {
		switch t := t.(type) {
		case Plain:
			p.escape(out, string(t))
		case Italic:
			out.WriteString("*")
			p.escape(out, string(t))
			out.WriteString("*")
		case *Link:
			out.WriteString("[")
			p.rawText(out, t.Text)
			out.WriteString("](")
			out.WriteString(t.URL)
			out.WriteString(")")
		case *DocLink:
			url := p.docLinkURL(t)
			if url != "" {
				out.WriteString("[")
			}
			p.rawText(out, t.Text)
			if url != "" {
				out.WriteString("](")
				url = strings.ReplaceAll(url, "(", "%28")
				url = strings.ReplaceAll(url, ")", "%29")
				out.WriteString(url)
				out.WriteString(")")
			}
		}
	}
}

// escape prints s to out as plain text,
// escaping special characters to avoid being misinterpreted
// as Markdown markup sequences.
func (p *mdPrinter) escape(out *bytes.Buffer, s string) {
	start := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\n':
			// Turn all \n into spaces, for a few reasons:
			//   - Avoid introducing paragraph breaks accidentally.
			//   - Avoid the need to reindent after the newline.
			//   - Avoid problems with Markdown renderers treating
			//     every mid-paragraph newline as a <br>.
			out.WriteString(s[start:i])
			out.WriteByte(' ')
			start = i + 1
			continue
		case '`', '_', '*', '[', '<', '\\':
			// Not all of these need to be escaped all the time,
			// but is valid and easy to do so.
			// We assume the Markdown is being passed to a
			// Markdown renderer, not edited by a person,
			// so it's fine to have escapes that are not strictly
			// necessary in some cases.
			out.WriteString(s[start:i])
			out.WriteByte('\\')
			out.WriteByte(s[i])
			start = i + 1
		}
	}
	out.WriteString(s[start:])
}
