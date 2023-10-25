// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"bytes"
	"fmt"
	"strings"
)

// A Printer is a doc comment printer.
// The fields in the struct can be filled in before calling
// any of the printing methods
// in order to customize the details of the printing process.
type Printer struct {
	// HeadingLevel is the nesting level used for
	// HTML and Markdown headings.
	// If HeadingLevel is zero, it defaults to level 3,
	// meaning to use <h3> and ###.
	HeadingLevel int

	// HeadingID is a function that computes the heading ID
	// (anchor tag) to use for the heading h when generating
	// HTML and Markdown. If HeadingID returns an empty string,
	// then the heading ID is omitted.
	// If HeadingID is nil, h.DefaultID is used.
	HeadingID func(h *Heading) string

	// DocLinkURL is a function that computes the URL for the given DocLink.
	// If DocLinkURL is nil, then link.DefaultURL(p.DocLinkBaseURL) is used.
	DocLinkURL func(link *DocLink) string

	// DocLinkBaseURL is used when DocLinkURL is nil,
	// passed to [DocLink.DefaultURL] to construct a DocLink's URL.
	// See that method's documentation for details.
	DocLinkBaseURL string

	// TextPrefix is a prefix to print at the start of every line
	// when generating text output using the Text method.
	TextPrefix string

	// TextCodePrefix is the prefix to print at the start of each
	// preformatted (code block) line when generating text output,
	// instead of (not in addition to) TextPrefix.
	// If TextCodePrefix is the empty string, it defaults to TextPrefix+"\t".
	TextCodePrefix string

	// TextWidth is the maximum width text line to generate,
	// measured in Unicode code points,
	// excluding TextPrefix and the newline character.
	// If TextWidth is zero, it defaults to 80 minus the number of code points in TextPrefix.
	// If TextWidth is negative, there is no limit.
	TextWidth int
}

func (p *Printer) headingLevel() int {
	if p.HeadingLevel <= 0 {
		return 3
	}
	return p.HeadingLevel
}

func (p *Printer) headingID(h *Heading) string {
	if p.HeadingID == nil {
		return h.DefaultID()
	}
	return p.HeadingID(h)
}

func (p *Printer) docLinkURL(link *DocLink) string {
	if p.DocLinkURL != nil {
		return p.DocLinkURL(link)
	}
	return link.DefaultURL(p.DocLinkBaseURL)
}

// DefaultURL constructs and returns the documentation URL for l,
// using baseURL as a prefix for links to other packages.
//
// The possible forms returned by DefaultURL are:
//   - baseURL/ImportPath, for a link to another package
//   - baseURL/ImportPath#Name, for a link to a const, func, type, or var in another package
//   - baseURL/ImportPath#Recv.Name, for a link to a method in another package
//   - #Name, for a link to a const, func, type, or var in this package
//   - #Recv.Name, for a link to a method in this package
//
// If baseURL ends in a trailing slash, then DefaultURL inserts
// a slash between ImportPath and # in the anchored forms.
// For example, here are some baseURL values and URLs they can generate:
//
//	"/pkg/" → "/pkg/math/#Sqrt"
//	"/pkg"  → "/pkg/math#Sqrt"
//	"/"     → "/math/#Sqrt"
//	""      → "/math#Sqrt"
func (l *DocLink) DefaultURL(baseURL string) string {
	if l.ImportPath != "" {
		slash := ""
		if strings.HasSuffix(baseURL, "/") {
			slash = "/"
		} else {
			baseURL += "/"
		}
		switch {
		case l.Name == "":
			return baseURL + l.ImportPath + slash
		case l.Recv != "":
			return baseURL + l.ImportPath + slash + "#" + l.Recv + "." + l.Name
		default:
			return baseURL + l.ImportPath + slash + "#" + l.Name
		}
	}
	if l.Recv != "" {
		return "#" + l.Recv + "." + l.Name
	}
	return "#" + l.Name
}

// DefaultID returns the default anchor ID for the heading h.
//
// The default anchor ID is constructed by converting every
// rune that is not alphanumeric ASCII to an underscore
// and then adding the prefix “hdr-”.
// For example, if the heading text is “Go Doc Comments”,
// the default ID is “hdr-Go_Doc_Comments”.
func (h *Heading) DefaultID() string {
	// Note: The “hdr-” prefix is important to avoid DOM clobbering attacks.
	// See https://pkg.go.dev/github.com/google/safehtml#Identifier.
	var out strings.Builder
	var p textPrinter
	p.oneLongLine(&out, h.Text)
	s := strings.TrimSpace(out.String())
	if s == "" {
		return ""
	}
	out.Reset()
	out.WriteString("hdr-")
	for _, r := range s {
		if r < 0x80 && isIdentASCII(byte(r)) {
			out.WriteByte(byte(r))
		} else {
			out.WriteByte('_')
		}
	}
	return out.String()
}

type commentPrinter struct {
	*Printer
}

// Comment returns the standard Go formatting of the [Doc],
// without any comment markers.
func (p *Printer) Comment(d *Doc) []byte {
	cp := &commentPrinter{Printer: p}
	var out bytes.Buffer
	for i, x := range d.Content {
		if i > 0 && blankBefore(x) {
			out.WriteString("\n")
		}
		cp.block(&out, x)
	}

	// Print one block containing all the link definitions that were used,
	// and then a second block containing all the unused ones.
	// This makes it easy to clean up the unused ones: gofmt and
	// delete the final block. And it's a nice visual signal without
	// affecting the way the comment formats for users.
	for i := 0; i < 2; i++ {
		used := i == 0
		first := true
		for _, def := range d.Links {
			if def.Used == used {
				if first {
					out.WriteString("\n")
					first = false
				}
				out.WriteString("[")
				out.WriteString(def.Text)
				out.WriteString("]: ")
				out.WriteString(def.URL)
				out.WriteString("\n")
			}
		}
	}

	return out.Bytes()
}

// blankBefore reports whether the block x requires a blank line before it.
// All blocks do, except for Lists that return false from x.BlankBefore().
func blankBefore(x Block) bool {
	if x, ok := x.(*List); ok {
		return x.BlankBefore()
	}
	return true
}

// block prints the block x to out.
func (p *commentPrinter) block(out *bytes.Buffer, x Block) {
	switch x := x.(type) {
	default:
		fmt.Fprintf(out, "?%T", x)

	case *Paragraph:
		p.text(out, "", x.Text)
		out.WriteString("\n")

	case *Heading:
		out.WriteString("# ")
		p.text(out, "", x.Text)
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
			out.WriteString(" ")
			if item.Number == "" {
				out.WriteString(" - ")
			} else {
				out.WriteString(item.Number)
				out.WriteString(". ")
			}
			for i, blk := range item.Content {
				const fourSpace = "    "
				if i > 0 {
					out.WriteString("\n" + fourSpace)
				}
				p.text(out, fourSpace, blk.(*Paragraph).Text)
				out.WriteString("\n")
			}
		}
	}
}

// text prints the text sequence x to out.
func (p *commentPrinter) text(out *bytes.Buffer, indent string, x []Text) {
	for _, t := range x {
		switch t := t.(type) {
		case Plain:
			p.indent(out, indent, string(t))
		case Italic:
			p.indent(out, indent, string(t))
		case *Link:
			if t.Auto {
				p.text(out, indent, t.Text)
			} else {
				out.WriteString("[")
				p.text(out, indent, t.Text)
				out.WriteString("]")
			}
		case *DocLink:
			out.WriteString("[")
			p.text(out, indent, t.Text)
			out.WriteString("]")
		}
	}
}

// indent prints s to out, indenting with the indent string
// after each newline in s.
func (p *commentPrinter) indent(out *bytes.Buffer, indent, s string) {
	for s != "" {
		line, rest, ok := strings.Cut(s, "\n")
		out.WriteString(line)
		if ok {
			out.WriteString("\n")
			out.WriteString(indent)
		}
		s = rest
	}
}
