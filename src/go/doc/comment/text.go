// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"bytes"
	"fmt"
	"strings"
)

// A textPrinter holds the state needed for printing a Doc as plain text.
type textPrinter struct {
	*Printer
	long bytes.Buffer
}

// Text returns a textual formatting of the Doc.
// See the [Printer] documentation for ways to customize the text output.
func (p *Printer) Text(d *Doc) []byte {
	tp := &textPrinter{
		Printer: p,
	}
	var out bytes.Buffer
	for i, x := range d.Content {
		if i > 0 && blankBefore(x) {
			writeNL(&out)
		}
		tp.block(&out, x)
	}
	return out.Bytes()
}

// writeNL calls out.WriteByte('\n')
// but first trims trailing spaces on the previous line.
func writeNL(out *bytes.Buffer) {
	// Trim trailing spaces.
	data := out.Bytes()
	n := 0
	for n < len(data) && (data[len(data)-n-1] == ' ' || data[len(data)-n-1] == '\t') {
		n++
	}
	if n > 0 {
		out.Truncate(len(data) - n)
	}
	out.WriteByte('\n')
}

// block prints the block x to out.
func (p *textPrinter) block(out *bytes.Buffer, x Block) {
	switch x := x.(type) {
	default:
		fmt.Fprintf(out, "?%T\n", x)

	case *Paragraph:
		p.text(out, x.Text)
	}
}

// text prints the text sequence x to out.
// TODO: Wrap lines.
func (p *textPrinter) text(out *bytes.Buffer, x []Text) {
	p.oneLongLine(&p.long, x)
	out.WriteString(strings.ReplaceAll(p.long.String(), "\n", " "))
	p.long.Reset()
	writeNL(out)
}

// oneLongLine prints the text sequence x to out as one long line,
// without worrying about line wrapping.
// Explicit links have the [ ] dropped to improve readability.
func (p *textPrinter) oneLongLine(out *bytes.Buffer, x []Text) {
	for _, t := range x {
		switch t := t.(type) {
		case Plain:
			out.WriteString(string(t))
		case Italic:
			out.WriteString(string(t))
		case *Link:
			p.oneLongLine(out, t.Text)
		case *DocLink:
			p.oneLongLine(out, t.Text)
		}
	}
}
