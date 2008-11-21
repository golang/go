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
// TODO should use a ByteArray library eventually

type ByteArray struct {
	a *[]byte;
}


func (b *ByteArray) Init(initial_size int) {
	b.a = new([]byte, initial_size)[0 : 0];
}


func (b *ByteArray) Clear() {
	b.a = b.a[0 : 0];
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
// Writer is a filter implementing the io.Write interface. It assumes
// that the incoming bytes represent ASCII encoded text consisting of
// lines of tab-separated "cells". Cells in adjacent lines constitute
// a column. Writer rewrites the incoming text such that all cells in
// a column have the same width; thus it effectively aligns cells. It
// does this by adding padding where necessary.
//
// Formatting can be controlled via parameters:
//
// tabwidth  the minimal with of a cell
// padding   additional padding
// usetabs   use tabs instead of blanks for padding
//           (for correct-looking results, tabwidth must correspond
//           to the tabwidth in the editor used to look at the result)
//
// (See alse http://nickgravgaard.com/elastictabstops/index.html)

// TODO Should support UTF-8
// TODO Should probably implement a couple of trivial customization options
//      such as arbitrary padding character, left/right alignment, and inde-
//      pendant cell and tab width.


export type Writer struct {
	// TODO should not export any of the fields
	// configuration
	writer io.Write;
	tabwidth int;
	padding int;
	usetabs bool;

	// current state
	buf ByteArray;  // the collected text w/o tabs and newlines
	width int;  // width of last incomplete cell
	lines array.Array;  // list of lines; each line is a list of cell widths
	widths array.IntArray;  // list of column widths - re-used during formatting
}


func (b *Writer) AddLine() {
	b.lines.Push(array.NewIntArray(0));
}


func (b *Writer) Init(writer io.Write, tabwidth, padding int, usetabs bool) *Writer {
	b.writer = writer;
	b.tabwidth = tabwidth;
	b.padding = padding;
	b.usetabs = usetabs;
	
	b.buf.Init(1024);
	b.lines.Init(0);
	b.widths.Init(0);
	b.AddLine();  // the very first line
	
	return b;
}


func (b *Writer) Line(i int) *array.IntArray {
	return b.lines.At(i).(*array.IntArray);
}


func (b *Writer) LastLine() *array.IntArray {
	return b.lines.At(b.lines.Len() - 1).(*array.IntArray);
}


// debugging support
func (b *Writer) Dump() {
	pos := 0;
	for i := 0; i < b.lines.Len(); i++ {
		line := b.Line(i);
		print("(", i, ") ");
		for j := 0; j < line.Len(); j++ {
			w := line.At(j);
			print("[", string(b.buf.Slice(pos, pos + w)), "]");
			pos += w;
		}
		print("\n");
	}
	print("\n");
}


func (b *Writer) Write0(buf *[]byte) *os.Error {
	n, err := b.writer.Write(buf);
	if n != len(buf) && err == nil {
		err = os.EIO;
	}
	return err;
}


var Tabs = &[]byte{'\t', '\t', '\t', '\t', '\t', '\t', '\t', '\t'}
var Blanks = &[]byte{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '}
var Newline = &[]byte{'\n'}


func (b *Writer) WritePadding(textw, cellw int) (err *os.Error) {
	if b.usetabs {
		// make cell width a multiple of tabwidth
		cellw = ((cellw + b.tabwidth - 1) / b.tabwidth) * b.tabwidth;
	}

	n := cellw - textw;
	if n < 0 {
		panic("internal error");
	}

	padding := Blanks;
	if b.usetabs {
		n = (n + b.tabwidth - 1) / b.tabwidth;
		padding = Tabs;
	}
	
	for n > len(padding) {
		err = b.Write0(padding);
		if err != nil {
			goto exit;
		}
		n -= len(padding);
	}
	err = b.Write0(padding[0 : n]);

exit:
	return err;
}


func (b *Writer) WriteLines(pos0 int, line0, line1 int) (pos int, err *os.Error) {
	pos = pos0;
	for i := line0; i < line1; i++ {
		line := b.Line(i);
		for j := 0; j < line.Len(); j++ {
			w := line.At(j);
			err = b.Write0(b.buf.a[pos : pos + w]);
			if err != nil {
				goto exit;
			}
			pos += w;
			if j < b.widths.Len() {
				err = b.WritePadding(w, b.widths.At(j));
				if err != nil {
					goto exit;
				}
			}
		}
		err = b.Write0(Newline);
		if err != nil {
			goto exit;
		}
	}

exit:
	return pos, err;
}


// TODO use utflen for correct formatting
func utflen(buf *[]byte) int {
	n := 0;
	for i := 0; i < len(buf); i++ {
		if buf[i]&0xC0 != 0x80 {
			n++
		}
	}
	return n
}


func (b *Writer) Format(pos0 int, line0, line1 int) (pos int, err *os.Error) {
	pos = pos0;
	column := b.widths.Len();	
	last := line0;
	for this := line0; this < line1; this++ {
		line := b.Line(this);
		
		if column < line.Len() - 1 {
			// cell exists in this column
			// (note that the last cell per line is ignored)
			
			// print unprinted lines until beginning of block
			pos, err = b.WriteLines(pos, last, this);
			if err != nil {
				goto exit;
			}
			last = this;
			
			// column block begin
			width := b.tabwidth;  // minimal width
			for ; this < line1; this++ {
				line = b.Line(this);
				if column < line.Len() - 1 {
					// cell exists in this column => update width
					w := line.At(column) + b.padding;
					if w > width {
						width = w;
					}
				} else {
					break
				}
			}
			// column block end

			// format and print all columns to the right of this column
			// (we know the widths of this column and all columns to the left)
			b.widths.Push(width);
			pos, err = b.Format(pos, last, this);
			b.widths.Pop();
			last = this;
		}
	}

	// print unprinted lines until end
	pos, err = b.WriteLines(pos, last, line1);
	
exit:
	return pos, err;
}


func (b *Writer) Append(buf *[]byte) {
	b.buf.Append(buf);
	b.width += len(buf);
}


/* export */ func (b *Writer) Flush() *os.Error {
	dummy, err := b.Format(0, 0, b.lines.Len());
	// reset (even in the presence of errors)
	b.buf.Clear();
	b.width = 0;
	b.lines.Init(0);
	b.AddLine();
	return err;
}


/* export */ func (b *Writer) Write(buf *[]byte) (written int, err *os.Error) {
	i0, n := 0, len(buf);
	
	// split text into cells
	for i := 0; i < n; i++ {
		if ch := buf[i]; ch == '\t' || ch == '\n' {
			b.Append(buf[i0 : i]);
			i0 = i + 1;  // exclude ch from (next) cell

			// terminate cell
			b.LastLine().Push(b.width);
			b.width = 0;

			if ch == '\n' {
				if b.LastLine().Len() == 1 {
					// The last line has only one cell which does not have an
					// impact on the formatting of the following lines (the
					// last cell per line is ignored by Format), thus we can
					// flush the Writer contents.
					err = b.Flush();
					if err != nil {
						return i0, err;
					}
				} else {
					// We can't flush yet - just add a new line.
					b.AddLine();
				}
			}
		}
	}
	
	// append leftover text
	b.Append(buf[i0 : n]);
	return n, nil;
}


export func New(writer io.Write, tabwidth, padding int, usetabs bool) *Writer {
	return new(Writer).Init(writer, tabwidth, padding, usetabs)
}
