// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tabwriter

import (
	"os";
	"io";
	"vector";
	"utf8";
)


// ----------------------------------------------------------------------------
// Basic byteArray support

type byteArray struct {
	a []byte;
}


func (b *byteArray) Init(initial_size int) {
	b.a = make([]byte, initial_size)[0 : 0];
}


func (b *byteArray) Len() int {
	return len(b.a);
}


func (b *byteArray) clear() {
	b.a = b.a[0 : 0];
}


func (b *byteArray) slice(i, j int) []byte {
	return b.a[i : j];  // BUG should really be &b.a[i : j]
}


func (b *byteArray) append(s []byte) {
	a := b.a;
	n := len(a);
	m := n + len(s);

	if m > cap(a) {
		n2 := 2*n;
		if m > n2 {
			n2 = m;
		}
		b := make([]byte, n2);
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
// that the incoming bytes represent UTF-8 encoded text consisting of
// lines of tab-terminated "cells". Cells in adjacent lines constitute
// a column. Writer rewrites the incoming text such that all cells in
// a column have the same width; thus it effectively aligns cells. It
// does this by adding padding where necessary. All characters (ASCII
// or not) are assumed to be of the same width - this may not be true
// for arbitrary UTF-8 characters visualized on the screen.
//
// Note that any text at the end of a line that is not tab-terminated
// is not a cell and does not enforce alignment of cells in adjacent
// rows. To make it a cell it needs to be tab-terminated. (For more
// information see http://nickgravgaard.com/elastictabstops/index.html)
//
// Formatting can be controlled via parameters:
//
// cellwidth	minimal cell width
// padding      additional cell padding
// padchar      ASCII char used for padding
//              if padchar == '\t', the Writer will assume that the
//              width of a '\t' in the formatted output is cellwidth,
//              and cells are left-aligned independent of align_left
//              (for correct-looking results, cellwidth must correspond
//              to the tabwidth in the viewer displaying the result)
// filter_html  ignores html tags and handles entities (starting with '&'
//              and ending in ';') as single characters (width = 1)

type Writer struct {
	// configuration
	writer io.Write;
	cellwidth int;
	padding int;
	padbytes [8]byte;
	align_left bool;
	filter_html bool;

	// current state
	html_char byte;  // terminating char of html tag/entity, or 0 ('>', ';', or 0)
	buf byteArray;  // collected text w/o tabs and newlines
	size int;  // size of incomplete cell in bytes
	width int;  // width of incomplete cell in runes up to buf[pos] w/o ignored sections
	pos int;  // buffer position up to which width of incomplete cell has been computed
	lines_size vector.Vector;  // list of lines; each line is a list of cell sizes in bytes
	lines_width vector.Vector;  // list of lines; each line is a list of cell widths in runes
	widths vector.IntVector;  // list of column widths in runes - re-used during formatting
}

// Internal representation (current state):
//
// - all text written is appended to buf; tabs and newlines are stripped away
// - at any given time there is a (possibly empty) incomplete cell at the end
//   (the cell starts after a tab or newline)
// - size is the number of bytes belonging to the cell so far
// - width is text width in runes of that cell from the start of the cell to
//   position pos; html tags and entities are excluded from this width if html
//   filtering is enabled
// - the sizes and widths of processed text are kept in the lines_size and
//   lines_width arrays, which contain an array of sizes or widths for each line
// - the widths array is a temporary array with current widths used during
//   formatting; it is kept in Writer because it's re-used
//
//                    |<---------- size ---------->|
//                    |                            |
//                    |<- width ->|<- ignored ->|  |
//                    |           |             |  |
// [---processed---tab------------<tag>...</tag>...]
// ^                  ^                         ^
// |                  |                         |
// buf                start of incomplete cell  pos


func (b *Writer) addLine() {
	b.lines_size.Push(vector.NewIntVector(0));
	b.lines_width.Push(vector.NewIntVector(0));
}


func (b *Writer) Init(writer io.Write, cellwidth, padding int, padchar byte, align_left, filter_html bool) *Writer {
	if cellwidth < 0 {
		panic("negative cellwidth");
	}
	if padding < 0 {
		panic("negative padding");
	}
	b.writer = writer;
	b.cellwidth = cellwidth;
	b.padding = padding;
	for i := len(b.padbytes) - 1; i >= 0; i-- {
		b.padbytes[i] = padchar;
	}
	b.align_left = align_left || padchar == '\t';  // tab enforces left-alignment
	b.filter_html = filter_html;

	b.buf.Init(1024);
	b.lines_size.Init(0);
	b.lines_width.Init(0);
	b.widths.Init(0);
	b.addLine();  // the very first line

	return b;
}


func (b *Writer) line(i int) (*vector.IntVector, *vector.IntVector) {
	return
		b.lines_size.At(i).(*vector.IntVector),
		b.lines_width.At(i).(*vector.IntVector);
}


// debugging support
func (b *Writer) dump() {
	pos := 0;
	for i := 0; i < b.lines_size.Len(); i++ {
		line_size, line_width := b.line(i);
		print("(", i, ") ");
		for j := 0; j < line_size.Len(); j++ {
			s := line_size.At(j);
			print("[", string(b.buf.slice(pos, pos + s)), "]");
			pos += s;
		}
		print("\n");
	}
	print("\n");
}


func (b *Writer) write0(buf []byte) *os.Error {
	n, err := b.writer.Write(buf);
	if n != len(buf) && err == nil {
		err = os.EIO;
	}
	return err;
}


var newline = []byte{'\n'}

func (b *Writer) writePadding(textw, cellw int) (err *os.Error) {
	if b.padbytes[0] == '\t' {
		// make cell width a multiple of cellwidth
		cellw = ((cellw + b.cellwidth - 1) / b.cellwidth) * b.cellwidth;
	}

	n := cellw - textw;
	if n < 0 {
		panic("internal error");
	}

	if b.padbytes[0] == '\t' {
		n = (n + b.cellwidth - 1) / b.cellwidth;
	}

	for n > len(b.padbytes) {
		err = b.write0(b.padbytes);
		if err != nil {
			goto exit;
		}
		n -= len(b.padbytes);
	}
	err = b.write0(b.padbytes[0 : n]);

exit:
	return err;
}


func (b *Writer) writeLines(pos0 int, line0, line1 int) (pos int, err *os.Error) {
	pos = pos0;
	for i := line0; i < line1; i++ {
		line_size, line_width := b.line(i);
		for j := 0; j < line_size.Len(); j++ {
			s, w := line_size.At(j), line_width.At(j);

			if b.align_left {
				err = b.write0(b.buf.slice(pos, pos + s));
				if err != nil {
					goto exit;
				}
				pos += s;
				if j < b.widths.Len() {
					err = b.writePadding(w, b.widths.At(j));
					if err != nil {
						goto exit;
					}
				}

			} else {  // align right

				if j < b.widths.Len() {
					err = b.writePadding(w, b.widths.At(j));
					if err != nil {
						goto exit;
					}
				}
				err = b.write0(b.buf.slice(pos, pos + s));
				if err != nil {
					goto exit;
				}
				pos += s;
			}
		}

		if i+1 == b.lines_size.Len() {
			// last buffered line - we don't have a newline, so just write
			// any outstanding buffered data
			err = b.write0(b.buf.slice(pos, pos + b.size));
			pos += b.size;
		} else {
			// not the last line - write newline
			err = b.write0(newline);
		}
		if err != nil {
			goto exit;
		}
	}

exit:
	return pos, err;
}


func (b *Writer) format(pos0 int, line0, line1 int) (pos int, err *os.Error) {
	pos = pos0;
	column := b.widths.Len();
	last := line0;
	for this := line0; this < line1; this++ {
		line_size, line_width := b.line(this);

		if column < line_size.Len() - 1 {
			// cell exists in this column
			// (note that the last cell per line is ignored)

			// print unprinted lines until beginning of block
			pos, err = b.writeLines(pos, last, this);
			if err != nil {
				goto exit;
			}
			last = this;

			// column block begin
			width := b.cellwidth;  // minimal width
			for ; this < line1; this++ {
				line_size, line_width = b.line(this);
				if column < line_size.Len() - 1 {
					// cell exists in this column => update width
					w := line_width.At(column) + b.padding;
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
			pos, err = b.format(pos, last, this);
			b.widths.Pop();
			last = this;
		}
	}

	// print unprinted lines until end
	pos, err = b.writeLines(pos, last, line1);

exit:
	return pos, err;
}


func (b *Writer) Flush() *os.Error {
	dummy, err := b.format(0, 0, b.lines_size.Len());
	// reset (even in the presence of errors)
	b.buf.clear();
	b.size, b.width = 0, 0;
	b.pos = 0;
	b.lines_size.Init(0);
	b.lines_width.Init(0);
	b.addLine();
	return err;
}


func unicodeLen(buf []byte) int {
	l := 0;
	for i := 0; i < len(buf); {
		if buf[i] < utf8.RuneSelf {
			i++;
		} else {
			rune, size := utf8.DecodeRune(buf[i : len(buf)]);
			i += size;
		}
		l++;
	}
	return l;
}


func (b *Writer) append(buf []byte) {
	b.buf.append(buf);
	b.size += len(buf);
}


func (b *Writer) Write(buf []byte) (written int, err *os.Error) {
	i0, n := 0, len(buf);

	// split text into cells
	for i := 0; i < n; i++ {
		ch := buf[i];

		if b.html_char == 0 {
			// outside html tag/entity
			switch ch {
			case '\t', '\n':
				b.append(buf[i0 : i]);
				i0 = i + 1;  // exclude ch from (next) cell
				b.width += unicodeLen(b.buf.slice(b.pos, b.buf.Len()));
				b.pos = b.buf.Len();

				// terminate cell
				last_size, last_width := b.line(b.lines_size.Len() - 1);
				last_size.Push(b.size);
				last_width.Push(b.width);
				b.size, b.width = 0, 0;

				if ch == '\n' {
					b.addLine();
					if last_size.Len() == 1 {
						// The previous line has only one cell which does not have
						// an impact on the formatting of the following lines (the
						// last cell per line is ignored by format()), thus we can
						// flush the Writer contents.
						err = b.Flush();
						if err != nil {
							return i0, err;
						}
					}
				}

			case '<', '&':
				if b.filter_html {
					b.append(buf[i0 : i]);
					i0 = i;
					b.width += unicodeLen(b.buf.slice(b.pos, b.buf.Len()));
					b.pos = -1;  // preventative - should not be used (will cause index out of bounds)
					if ch == '<' {
						b.html_char = '>';
					} else {
						b.html_char = ';';
					}
				}
			}

		} else {
			// inside html tag/entity
			if ch == b.html_char {
				// reached the end of tag/entity
				b.append(buf[i0 : i + 1]);
				i0 = i + 1;
				if b.html_char == ';' {
					b.width++;  // count as one char
				}
				b.pos = b.buf.Len();
				b.html_char = 0;
			}
		}
	}

	// append leftover text
	b.append(buf[i0 : n]);
	return n, nil;
}


func New(writer io.Write, cellwidth, padding int, padchar byte, align_left, filter_html bool) *Writer {
	return new(Writer).Init(writer, cellwidth, padding, padchar, align_left, filter_html)
}
