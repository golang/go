// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The tabwriter package implements a write filter (tabwriter.Writer)
// that translates tabbed columns in input into properly aligned text,
// using the Elastic Tabstops algorithm described at
// http://nickgravgaard.com/elastictabstops/index.html.
//
package tabwriter

import (
	"bytes";
	"container/vector";
	"io";
	"os";
	"utf8";
)


// ----------------------------------------------------------------------------
// Filter implementation

// A cell represents a segment of text delineated by tabs, form-feed,
// or newline chars. The text itself is stored in a separate buffer;
// cell only describes the segment's size in bytes and width in runes.
//
type cell struct {
	size int;  // cell size in bytes
	width int;  // cell width in runes
}


// A Writer is a filter that inserts padding around
// tab-delimited columns in its input to align them
// in the output.
//
// The Writer treats incoming bytes as UTF-8 encoded text
// consisting of tab-terminated cells. Cells in adjacent lines
// constitute a column. The Writer inserts padding as needed
// to make all cells in a column have the same width, effectively
// aligning the columns. Note that cells are tab-terminated,
// not tab-separated: trailing non-tab text at the end of a line
// is not part of any cell.
//
// The Writer assumes that all characters have the same width;
// this may not be true in some fonts, especially with certain
// UTF-8 characters.
//
// If a Writer is configured to filter HTML, HTML tags and entities
// are simply passed through. The widths of tags and entities are
// assumed to be zero (tags) and one (entities) for formatting purposes.
//
// The form feed character ('\f') acts like a newline but it also
// terminates all columns in the current line (effectively calling
// Flush). Cells in the next line start new columns. Unless found
// inside an HTML tag, form feed characters appear as newlines in
// the output.
//
// The Writer must buffer input internally, because proper spacing
// of one line may depend on the cells in future lines. Clients must
// call Flush when done calling Write.
//
type Writer struct {
	// configuration
	output io.Writer;
	cellwidth int;
	padding int;
	padbytes [8]byte;
	flags uint;

	// current state
	buf bytes.Buffer;  // collected text w/o tabs, newlines, or form feed chars
	pos int;  // buffer position up to which width of incomplete cell has been computed
	cell cell;  // current incomplete cell; cell.width is up to buf[pos] w/o ignored sections
	html_char byte;  // terminating char of html tag/entity, or 0 ('>', ';', or 0)
	lines vector.Vector;  // list if lines; each line is a list of cells
	widths vector.IntVector;  // list of column widths in runes - re-used during formatting
}


func (b *Writer) addLine() {
	b.lines.Push(vector.New(0));
}


func (b *Writer) line(i int) *vector.Vector {
	return b.lines.At(i).(*vector.Vector);
}


// Reset the current state.
func (b *Writer) reset() {
	b.buf.Reset();
	b.pos = 0;
	b.cell = cell{};
	b.html_char = 0;
	b.lines.Init(0);
	b.widths.Init(0);
	b.addLine();
}


// Internal representation (current state):
//
// - all text written is appended to buf; form feed chars, tabs and newlines are stripped away
// - at any given time there is a (possibly empty) incomplete cell at the end
//   (the cell starts after a tab, form feed, or newline)
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


// Formatting can be controlled with these flags.
const (
	// Ignore html tags and treat entities (starting with '&'
	// and ending in ';') as single characters (width = 1).
	FilterHTML = 1 << iota;

	// Force right-alignment of cell content.
	// Default is left-alignment.
	AlignRight;
)


// A Writer must be initialized with a call to Init. The first parameter (output)
// specifies the filter output. The remaining parameters control the formatting:
//
//	cellwidth	minimal cell width
//	padding		additional cell padding
//	padchar		ASCII char used for padding
//				if padchar == '\t', the Writer will assume that the
//				width of a '\t' in the formatted output is cellwidth,
//				and cells are left-aligned independent of align_left
//				(for correct-looking results, cellwidth must correspond
//				to the tab width in the viewer displaying the result)
//	flags		formatting control
//
func (b *Writer) Init(output io.Writer, cellwidth, padding int, padchar byte, flags uint) *Writer {
	if cellwidth < 0 {
		panic("negative cellwidth");
	}
	if padding < 0 {
		panic("negative padding");
	}
	b.output = output;
	b.cellwidth = cellwidth;
	b.padding = padding;
	for i := len(b.padbytes) - 1; i >= 0; i-- {
		b.padbytes[i] = padchar;
	}
	if padchar == '\t' {
		// tab enforces left-alignment
		flags &^= AlignRight;
	}
	b.flags = flags;

	b.reset();

	return b;
}


// debugging support (keep code around)
/*
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
*/


func (b *Writer) write0(buf []byte) os.Error {
	n, err := b.output.Write(buf);
	if n != len(buf) && err == nil {
		err = os.EIO;
	}
	return err;
}


var newline = []byte{'\n'}

func (b *Writer) writePadding(textw, cellw int) os.Error {
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
		if err := b.write0(&b.padbytes); err != nil {
			return err;
		}
		n -= len(b.padbytes);
	}

	return b.write0(b.padbytes[0 : n]);
}


func (b *Writer) writeLines(pos0 int, line0, line1 int) (int, os.Error) {
	pos := pos0;
	for i := line0; i < line1; i++ {
		line := b.line(i);
		for j := 0; j < line.Len(); j++ {
			c := line.At(j).(cell);

			switch {
			default: // align left

				if err := b.write0(b.buf.Data()[pos : pos + c.size]); err != nil {
					return pos, err;
				}
				pos += c.size;
				if j < b.widths.Len() {
					if err := b.writePadding(c.width, b.widths.At(j)); err != nil {
						return pos, err;
					}
				}

			case b.flags & AlignRight != 0:  // align right

				if j < b.widths.Len() {
					if err := b.writePadding(c.width, b.widths.At(j)); err != nil {
						return pos, err;
					}
				}
				if err := b.write0(b.buf.Data()[pos : pos + c.size]); err != nil {
					return pos, err;
				}
				pos += c.size;
			}
		}

		if i+1 == b.lines.Len() {
			// last buffered line - we don't have a newline, so just write
			// any outstanding buffered data
			if err := b.write0(b.buf.Data()[pos : pos + b.cell.size]); err != nil {
				return pos, err;
			}
			pos += b.cell.size;
		} else {
			// not the last line - write newline
			if err := b.write0(newline); err != nil {
				return pos, err;
			}
		}
	}
	return pos, nil;
}


func (b *Writer) format(pos0 int, line0, line1 int) (pos int, err os.Error) {
	pos = pos0;
	column := b.widths.Len();
	last := line0;
	for this := line0; this < line1; this++ {
		line := b.line(this);

		if column < line.Len() - 1 {
			// cell exists in this column
			// (note that the last cell per line is ignored)

			// print unprinted lines until beginning of block
			pos, err = b.writeLines(pos, last, this);
			if err != nil {
				return pos, err;
			}
			last = this;

			// column block begin
			width := b.cellwidth;  // minimal width
			for ; this < line1; this++ {
				line = b.line(this);
				if column < line.Len() - 1 {
					// cell exists in this column => update width
					w := line.At(column).(cell).width + b.padding;
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
	return b.writeLines(pos, last, line1);
}


// Append text to current cell. Only update the cell width if updateWidth
// is set (the cell width can only be updated if we know that we cannot be
// in the middle of a UTF-8 encoded Unicode character).
//
func (b *Writer) append(text []byte, updateWidth bool) {
	b.buf.Write(text);
	b.cell.size += len(text);
	if updateWidth {
		b.cell.width += utf8.RuneCount(b.buf.Data()[b.pos : b.buf.Len()]);
		b.pos = b.buf.Len();
	}
}


// Start HTML-escape mode.
func (b *Writer) startHTML(ch byte) {
	if ch == '<' {
		b.html_char = '>';
	} else {
		b.html_char = ';';
	}
}


// Terminate HTML-escape mode. If the HTML text was an entity, its width
// is assumed to be one for formatting purposes; otherwise it assumed to
// be zero.
//
func (b *Writer) terminateHTML() {
	if b.html_char == ';' {
		// was entity, count as one rune
		b.cell.width++;
	}
	b.pos = b.buf.Len();
	b.html_char = 0;
}


// Terminate the current cell by adding it to the list of cells of the
// current line. Returns the number of cells in that line.
//
func (b *Writer) terminateCell() int {
	line := b.line(b.lines.Len() - 1);
	line.Push(b.cell);
	b.cell = cell{};
	return line.Len();
}


// Flush should be called after the last call to Write to ensure
// that any data buffered in the Writer is written to output. Any
// incomplete HTML tag or entity at the end is simply considered
// complete for formatting purposes.
//
func (b *Writer) Flush() os.Error {
	// add current cell if not empty
	if b.cell.size > 0 {
		if b.html_char != 0 {
			// inside html tag/entity - terminate it even if incomplete
			b.terminateHTML();
		}
		b.terminateCell();
	}

	// format contents of buffer
	_, err := b.format(0, 0, b.lines.Len());

	// reset, even in the presence of errors
	b.reset();

	return err;
}


// Write writes buf to the writer b.
// The only errors returned are ones encountered
// while writing to the underlying output stream.
//
func (b *Writer) Write(buf []byte) (written int, err os.Error) {
	// split text into cells
	i0 := 0;
	for i, ch := range buf {
		if b.html_char == 0 {
			// outside html tag/entity
			switch ch {
			case '\t', '\n', '\f':
				// end of cell
				b.append(buf[i0 : i], true);
				i0 = i+1;  // exclude ch from (next) cell
				ncells := b.terminateCell();
				if ch != '\t' {
					// terminate line
					b.addLine();
					if ch == '\f' || ncells == 1 {
						// A '\f' always forces a flush. Otherwise, if the previous
						// line has only one cell which does not have an impact on
						// the formatting of the following lines (the last cell per
						// line is ignored by format()), thus we can flush the
						// Writer contents.
						if err = b.Flush(); err != nil {
							return i0, err;
						}
					}
				}

			case '<', '&':
				// possibly an html tag/entity
				if b.flags & FilterHTML != 0 {
					// begin of tag/entity
					b.append(buf[i0 : i], true);
					i0 = i;
					b.startHTML(ch);
				}
			}

		} else {
			// inside html tag/entity
			if ch == b.html_char {
				// end of tag/entity
				b.append(buf[i0 : i+1], false);
				i0 = i+1;  // exclude ch from (next) cell
				b.terminateHTML();
			}
		}
	}

	// append leftover text
	b.append(buf[i0 : len(buf)], false);
	return len(buf), nil;
}


// NewWriter allocates and initializes a new tabwriter.Writer.
// The parameters are the same as for the the Init function.
//
func NewWriter(output io.Writer, cellwidth, padding int, padchar byte, flags uint) *Writer {
	return new(Writer).Init(output, cellwidth, padding, padchar, flags)
}
