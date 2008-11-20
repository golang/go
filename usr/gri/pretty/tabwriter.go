// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tabwriter

import (
	"os";
	"io";
	"array";
)


// ----------------------------------------------------------------------------
// ByteArray
// TODO move this into std lib eventually

type ByteArray struct {
	a *[]byte;
}


func (b *ByteArray) Init(initial_size int) {
	b.a = new([]byte, initial_size)[0 : 0];
}


func (b *ByteArray) Clear() {
	b.a = b.a[0 : 0];
}


func (b *ByteArray) Len() int {
	return len(b.a);
}


func (b *ByteArray) At(i int) byte {
	return b.a[i];
}


func (b *ByteArray) Set(i int, x byte) {
	b.a[i] = x;
}


func (b *ByteArray) Slice(i, j int) *[]byte {
	return b.a[i : j];  // BUG should really be &b.a[i : j]
}


func (b *ByteArray) Append(s *[]byte) {
	a := b.a;
	n := len(a);
	m := n + len(s);

	if m > cap(a) {
		n2 := 2*n;
		if m > n2 {
			n2 = m;
		}
		b := new([]byte, n2);
		for i := 0; i < n; i++ {
			b[i] = a[i];
		}
		a = b;
	}

	a = a[0 : m];
	for i := len(s) - 1; i >= 0; i-- {
		a[n + i] = s[i];
	}
	b.a = a;
}


// ----------------------------------------------------------------------------
// Implementation of flexible tab stops.

// TabWriter is a representation for a list of lines consisting of
// cells. A new cell is added for each Tab() call, and a new line
// is added for each Newline() call.
//
// The lines are formatted and printed such that all cells in a column
// of adjacent cells have the same width (by adding padding). For more
// details see: http://nickgravgaard.com/elastictabstops/index.html .

export type TabWriter struct {
	// configuration
	writer io.Write;
	usetabs bool;
	tabwidth int;

	// current state
	buf ByteArray;  // the collected text w/o tabs and newlines
	width int;  // width of last incomplete cell
	lines array.Array;  // list of lines; each line is a list of cell widths
	widths array.IntArray;  // list of column widths - (re-)used during formatting
}


func (b *TabWriter) AddLine() {
	b.lines.Push(array.NewIntArray(0));
}


func (b *TabWriter) Init(writer io.Write, usetabs bool, tabwidth int) {
	b.writer = writer;
	b.usetabs = usetabs;
	b.tabwidth = tabwidth;
	
	b.buf.Init(1024);
	b.lines.Init(0);
	b.widths.Init(0);
	b.AddLine();  // the very first line
}


func (b *TabWriter) Line(i int) *array.IntArray {
	return b.lines.At(i).(*array.IntArray);
}


func (b *TabWriter) LastLine() *array.IntArray {
	return b.lines.At(b.lines.Len() - 1).(*array.IntArray);
}


// debugging support
func (b *TabWriter) Dump() {
	pos := 0;
	for i := 0; i < b.lines.Len(); i++ {
		line := b.Line(i);
		print("(", i, ") ");
		for j := 0; j < line.Len(); j++ {
			w := line.At(j);
			print("[", string(b.buf.a[pos : pos + w]), "]");
			pos += w;
		}
		print("\n");
	}
	print("\n");
}


var Tabs = &[]byte{'\t', '\t', '\t', '\t', '\t', '\t', '\t', '\t'}
var Blanks = &[]byte{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '}
var Newline = &[]byte{'\n'}


func (b *TabWriter) Padding(textwidth, cellwidth int) {
	n := cellwidth - textwidth;
	if n < 0 {
		panic("internal error");
	}
	if b.usetabs {
		if cellwidth % b.tabwidth != 0 {
			panic("internal error");  // cellwidth should be a multiple of tabwidth
		}
		n = (n + b.tabwidth - 1) / b.tabwidth;
		for n > len(Tabs) {
			m, err := b.writer.Write(Tabs);
			n -= len(Tabs);
		}
		m, err := b.writer.Write(Tabs[0 : n]);
	} else {
		for n > len(Blanks) {
			m, err := b.writer.Write(Blanks);
			n -= len(Blanks);
		}
		m, err := b.writer.Write(Blanks[0 : n]);
	}
}


func (b *TabWriter) PrintLines(pos int, line0, line1 int) int {
	for i := line0; i < line1; i++ {
		line := b.Line(i);
		for j := 0; j < line.Len(); j++ {
			w := line.At(j);
			m, err := b.writer.Write(b.buf.a[pos : pos + w]);
			if m != w {
				panic();
			}
			pos += w;
			if j < b.widths.Len() {
				b.Padding(w, b.widths.At(j));
			}
		}
		m, err := b.writer.Write(Newline);
	}
	return pos;
}


func (b *TabWriter) Format(pos int, line0, line1 int) int {
	column := b.widths.Len();
	
	last := line0;
	for this := line0; this < line1; this++ {
		line := b.Line(this);
		
		if column < line.Len() - 1 {
			// cell exists in this column
			// (note that the last cell per line is ignored)
			
			// print unprinted lines until beginning of block
			pos = b.PrintLines(pos, last, this);
			last = this;
			
			// column block begin
			width := b.tabwidth;  // minimal width
			for ; this < line1; this++ {
				line = b.Line(this);
				if column < line.Len() - 1 {
					// cell exists in this column
					// update width
					w := line.At(column) + 1; // 1 = minimum space between cells
					if w > width {
						width = w;
					}
				} else {
					break
				}
			}
			// column block end

			if b.usetabs {
				// make width a multiple of the tab width
				width = ((width + b.tabwidth - 1) / b.tabwidth) * b.tabwidth;
			}
			
			// format and print all columns to the right of this column
			// (we know the widths of this column and all columns to the left)
			b.widths.Push(width);
			pos = b.Format(pos, last, this);
			b.widths.Pop();
			last = this;
		}
	}

	// print unprinted lines until end
	return b.PrintLines(pos, last, line1);
}


func (b *TabWriter) EmptyLine() bool {
	return b.LastLine().Len() == 0 && b.width == 0;
}


func (b *TabWriter) Tab() {
	b.LastLine().Push(b.width);
	b.width = 0;
}


func (b *TabWriter) Newline() {
	b.Tab();  // add last cell to current line

	if b.LastLine().Len() == 1 {
		// The current line has only one cell which does not have an impact
		// on the formatting of the following lines (the last cell per line
		// is ignored by Format), thus we can print the TabWriter contents.
		if b.widths.Len() != 0 {
			panic("internal error");
		}
		b.Format(0, 0, b.lines.Len());
		if b.widths.Len() != 0 {
			panic("internal error");
		}

		// reset TabWriter
		b.width = 0;
		b.buf.Clear();
		b.lines.Init(0);
	}

	b.AddLine();
}


func (b *TabWriter) Write(buf *[]byte) (i int, err *os.Error) {
	i0, n := 0, len(buf);
	for i = 0; i < n; i++ {
		switch buf[i] {
		case '\t':
			b.width += i - i0;
			b.buf.Append(buf[i0 : i]);
			i0 = i + 1;  // don't append '\t'
			b.Tab();
		case '\n':
			b.width += i - i0;
			b.buf.Append(buf[i0 : i]);
			i0 = i + 1;  // don't append '\n'
			b.Newline();
		}
	}
	b.width += n - i0;
	b.buf.Append(buf[i0 : n]);
	return i, nil;
}


export func MakeTabWriter(writer io.Write, usetabs bool, tabwidth int) *TabWriter {
	b := new(TabWriter);
	b.Init(writer, usetabs, tabwidth);
	return b;
}
