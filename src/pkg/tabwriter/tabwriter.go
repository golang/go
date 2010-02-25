// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The tabwriter package implements a write filter (tabwriter.Writer)
// that translates tabbed columns in input into properly aligned text.
//
// The package is using the Elastic Tabstops algorithm described at
// http://nickgravgaard.com/elastictabstops/index.html.
//
package tabwriter

import (
	"bytes"
	"container/vector"
	"io"
	"os"
	"utf8"
)


// ----------------------------------------------------------------------------
// Filter implementation

// A cell represents a segment of text terminated by tabs or line breaks.
// The text itself is stored in a separate buffer; cell only describes the
// segment's size in bytes, its width in runes, and whether it's an htab
// ('\t') terminated cell.
//
type cell struct {
	size  int  // cell size in bytes
	width int  // cell width in runes
	htab  bool // true if the cell is terminated by an htab ('\t')
}


// A Writer is a filter that inserts padding around
// tab-delimited columns in its input to align them
// in the output.
//
// The Writer treats incoming bytes as UTF-8 encoded text consisting
// of cells terminated by (horizontal or vertical) tabs or line
// breaks (newline or formfeed characters). Cells in adjacent lines
// constitute a column. The Writer inserts padding as needed to
// make all cells in a column have the same width, effectively
// aligning the columns. It assumes that all characters have the
// same width except for tabs for which a tabwidth must be specified.
// Note that cells are tab-terminated, not tab-separated: trailing
// non-tab text at the end of a line does not form a column cell.
//
// If DiscardEmptyColumns is set, empty columns that are terminated
// entirely by vertical (or "soft") tabs are discarded. Columns
// terminated by horizontal (or "hard") tabs are not affected by
// this flag.
//
// A segment of text may be escaped by bracketing it with Escape
// characters. The tabwriter strips the Escape characters but otherwise
// passes escaped text segments through unchanged. In particular, it
// does not interpret any tabs or line breaks within the segment.
//
// The Writer assumes that all characters have the same width;
// this may not be true in some fonts, especially with certain
// UTF-8 characters.
//
// If a Writer is configured to filter HTML, HTML tags and entities
// are simply passed through. The widths of tags and entities are
// assumed to be zero (tags) and one (entities) for formatting purposes.
//
// The formfeed character ('\f') acts like a newline but it also
// terminates all columns in the current line (effectively calling
// Flush). Cells in the next line start new columns. Unless found
// inside an HTML tag or inside an escaped text segment, formfeed
// characters appear as newlines in the output.
//
// The Writer must buffer input internally, because proper spacing
// of one line may depend on the cells in future lines. Clients must
// call Flush when done calling Write.
//
type Writer struct {
	// configuration
	output   io.Writer
	minwidth int
	tabwidth int
	padding  int
	padbytes [8]byte
	flags    uint

	// current state
	buf     bytes.Buffer     // collected text excluding tabs or line breaks
	pos     int              // buffer position up to which cell.width of incomplete cell has been computed
	cell    cell             // current incomplete cell; cell.width is up to buf[pos] excluding ignored sections
	endChar byte             // terminating char of escaped sequence (Escape for escapes, '>', ';' for HTML tags/entities, or 0)
	lines   vector.Vector    // list of lines; each line is a list of cells
	widths  vector.IntVector // list of column widths in runes - re-used during formatting
}


func (b *Writer) addLine() { b.lines.Push(new(vector.Vector)) }


func (b *Writer) line(i int) *vector.Vector { return b.lines.At(i).(*vector.Vector) }


// Reset the current state.
func (b *Writer) reset() {
	b.buf.Reset()
	b.pos = 0
	b.cell = cell{}
	b.endChar = 0
	b.lines.Resize(0, 0)
	b.widths.Resize(0, 0)
	b.addLine()
}


// Internal representation (current state):
//
// - all text written is appended to buf; tabs and line breaks are stripped away
// - at any given time there is a (possibly empty) incomplete cell at the end
//   (the cell starts after a tab or line break)
// - cell.size is the number of bytes belonging to the cell so far
// - cell.width is text width in runes of that cell from the start of the cell to
//   position pos; html tags and entities are excluded from this width if html
//   filtering is enabled
// - the sizes and widths of processed text are kept in the lines vector
//   which contains a vector of cells for each line
// - the widths vector is a temporary vector with current widths used during
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
	FilterHTML uint = 1 << iota

	// Force right-alignment of cell content.
	// Default is left-alignment.
	AlignRight

	// Handle empty columns as if they were not present in
	// the input in the first place.
	DiscardEmptyColumns

	// Always use tabs for indentation columns (i.e., padding of
	// leading empty cells on the left) independent of padchar.
	TabIndent

	// Print a vertical bar ('|') between columns (after formatting).
	// Discarded colums appear as zero-width columns ("||").
	Debug
)


// A Writer must be initialized with a call to Init. The first parameter (output)
// specifies the filter output. The remaining parameters control the formatting:
//
//	minwidth	minimal cell width including any padding
//      tabwidth	width of tab characters (equivalent number of spaces)
//	padding		padding added to a cell before computing its width
//	padchar		ASCII char used for padding
//			if padchar == '\t', the Writer will assume that the
//			width of a '\t' in the formatted output is tabwidth,
//			and cells are left-aligned independent of align_left
//			(for correct-looking results, tabwidth must correspond
//			to the tab width in the viewer displaying the result)
//	flags		formatting control
//
// To format in tab-separated columns with a tab stop of 8:
//	b.Init(w, 8, 1, 8, '\t', 0);
//
// To format in space-separated columns with at least 4 spaces between columns:
//	b.Init(w, 0, 4, 8, ' ', 0);
//
func (b *Writer) Init(output io.Writer, minwidth, tabwidth, padding int, padchar byte, flags uint) *Writer {
	if minwidth < 0 || tabwidth < 0 || padding < 0 {
		panic("negative minwidth, tabwidth, or padding")
	}
	b.output = output
	b.minwidth = minwidth
	b.tabwidth = tabwidth
	b.padding = padding
	for i := range b.padbytes {
		b.padbytes[i] = padchar
	}
	if padchar == '\t' {
		// tab padding enforces left-alignment
		flags &^= AlignRight
	}
	b.flags = flags

	b.reset()

	return b
}


// debugging support (keep code around)
func (b *Writer) dump() {
	pos := 0
	for i := 0; i < b.lines.Len(); i++ {
		line := b.line(i)
		print("(", i, ") ")
		for j := 0; j < line.Len(); j++ {
			c := line.At(j).(cell)
			print("[", string(b.buf.Bytes()[pos:pos+c.size]), "]")
			pos += c.size
		}
		print("\n")
	}
	print("\n")
}


func (b *Writer) write0(buf []byte) os.Error {
	n, err := b.output.Write(buf)
	if n != len(buf) && err == nil {
		err = os.EIO
	}
	return err
}


func (b *Writer) writeN(src []byte, n int) os.Error {
	for n > len(src) {
		if err := b.write0(src); err != nil {
			return err
		}
		n -= len(src)
	}
	return b.write0(src[0:n])
}


var (
	newline = []byte{'\n'}
	tabs    = []byte{'\t', '\t', '\t', '\t', '\t', '\t', '\t', '\t'}
)


func (b *Writer) writePadding(textw, cellw int, useTabs bool) os.Error {
	if b.padbytes[0] == '\t' || useTabs {
		// padding is done with tabs
		if b.tabwidth == 0 {
			return nil // tabs have no width - can't do any padding
		}
		// make cellw the smallest multiple of b.tabwidth
		cellw = (cellw + b.tabwidth - 1) / b.tabwidth * b.tabwidth
		n := cellw - textw // amount of padding
		if n < 0 {
			panic("internal error")
		}
		return b.writeN(tabs, (n+b.tabwidth-1)/b.tabwidth)
	}

	// padding is done with non-tab characters
	return b.writeN(&b.padbytes, cellw-textw)
}


var vbar = []byte{'|'}

func (b *Writer) writeLines(pos0 int, line0, line1 int) (pos int, err os.Error) {
	pos = pos0
	for i := line0; i < line1; i++ {
		line := b.line(i)

		// if TabIndent is set, use tabs to pad leading empty cells
		useTabs := b.flags&TabIndent != 0

		for j := 0; j < line.Len(); j++ {
			c := line.At(j).(cell)

			if j > 0 && b.flags&Debug != 0 {
				// indicate column break
				if err = b.write0(vbar); err != nil {
					return
				}
			}

			if c.size == 0 {
				// empty cell
				if j < b.widths.Len() {
					if err = b.writePadding(c.width, b.widths.At(j), useTabs); err != nil {
						return
					}
				}
			} else {
				// non-empty cell
				useTabs = false
				if b.flags&AlignRight == 0 { // align left
					if err = b.write0(b.buf.Bytes()[pos : pos+c.size]); err != nil {
						return
					}
					pos += c.size
					if j < b.widths.Len() {
						if err = b.writePadding(c.width, b.widths.At(j), false); err != nil {
							return
						}
					}
				} else { // align right
					if j < b.widths.Len() {
						if err = b.writePadding(c.width, b.widths.At(j), false); err != nil {
							return
						}
					}
					if err = b.write0(b.buf.Bytes()[pos : pos+c.size]); err != nil {
						return
					}
					pos += c.size
				}
			}
		}

		if i+1 == b.lines.Len() {
			// last buffered line - we don't have a newline, so just write
			// any outstanding buffered data
			if err = b.write0(b.buf.Bytes()[pos : pos+b.cell.size]); err != nil {
				return
			}
			pos += b.cell.size
		} else {
			// not the last line - write newline
			if err = b.write0(newline); err != nil {
				return
			}
		}
	}
	return
}


// Format the text between line0 and line1 (excluding line1); pos
// is the buffer position corresponding to the beginning of line0.
// Returns the buffer position corresponding to the beginning of
// line1 and an error, if any.
//
func (b *Writer) format(pos0 int, line0, line1 int) (pos int, err os.Error) {
	pos = pos0
	column := b.widths.Len()
	for this := line0; this < line1; this++ {
		line := b.line(this)

		if column < line.Len()-1 {
			// cell exists in this column => this line
			// has more cells than the previous line
			// (the last cell per line is ignored because cells are
			// tab-terminated; the last cell per line describes the
			// text before the newline/formfeed and does not belong
			// to a column)

			// print unprinted lines until beginning of block
			if pos, err = b.writeLines(pos, line0, this); err != nil {
				return
			}
			line0 = this

			// column block begin
			width := b.minwidth // minimal column width
			discardable := true // true if all cells in this column are empty and "soft"
			for ; this < line1; this++ {
				line = b.line(this)
				if column < line.Len()-1 {
					// cell exists in this column
					c := line.At(column).(cell)
					// update width
					if w := c.width + b.padding; w > width {
						width = w
					}
					// update discardable
					if c.width > 0 || c.htab {
						discardable = false
					}
				} else {
					break
				}
			}
			// column block end

			// discard empty columns if necessary
			if discardable && b.flags&DiscardEmptyColumns != 0 {
				width = 0
			}

			// format and print all columns to the right of this column
			// (we know the widths of this column and all columns to the left)
			b.widths.Push(width)
			pos, err = b.format(pos, line0, this)
			b.widths.Pop()
			line0 = this
		}
	}

	// print unprinted lines until end
	return b.writeLines(pos, line0, line1)
}


// Append text to current cell.
func (b *Writer) append(text []byte) {
	b.buf.Write(text)
	b.cell.size += len(text)
}


// Update the cell width.
func (b *Writer) updateWidth() {
	b.cell.width += utf8.RuneCount(b.buf.Bytes()[b.pos:b.buf.Len()])
	b.pos = b.buf.Len()
}


// To escape a text segment, bracket it with Escape characters.
// For instance, the tab in this string "Ignore this tab: \xff\t\xff"
// does not terminate a cell and constitutes a single character of
// width one for formatting purposes.
//
// The value 0xff was chosen because it cannot appear in a valid UTF-8 sequence.
//
const Escape = '\xff'


// Start escaped mode.
func (b *Writer) startEscape(ch byte) {
	switch ch {
	case Escape:
		b.endChar = Escape
	case '<':
		b.endChar = '>'
	case '&':
		b.endChar = ';'
	}
}


// Terminate escaped mode. If the escaped text was an HTML tag, its width
// is assumed to be zero for formatting purposes; if it was an HTML entity,
// its width is assumed to be one. In all other cases, the width is the
// unicode width of the text.
//
func (b *Writer) endEscape() {
	switch b.endChar {
	case Escape:
		b.updateWidth()
	case '>': // tag of zero width
	case ';':
		b.cell.width++ // entity, count as one rune
	}
	b.pos = b.buf.Len()
	b.endChar = 0
}


// Terminate the current cell by adding it to the list of cells of the
// current line. Returns the number of cells in that line.
//
func (b *Writer) terminateCell(htab bool) int {
	b.cell.htab = htab
	line := b.line(b.lines.Len() - 1)
	line.Push(b.cell)
	b.cell = cell{}
	return line.Len()
}


// Flush should be called after the last call to Write to ensure
// that any data buffered in the Writer is written to output. Any
// incomplete escape sequence at the end is simply considered
// complete for formatting purposes.
//
func (b *Writer) Flush() os.Error {
	// add current cell if not empty
	if b.cell.size > 0 {
		if b.endChar != 0 {
			// inside escape - terminate it even if incomplete
			b.endEscape()
		}
		b.terminateCell(false)
	}

	// format contents of buffer
	_, err := b.format(0, 0, b.lines.Len())

	// reset, even in the presence of errors
	b.reset()

	return err
}


var hbar = []byte{'-', '-', '-', '\n'}

// Write writes buf to the writer b.
// The only errors returned are ones encountered
// while writing to the underlying output stream.
//
func (b *Writer) Write(buf []byte) (n int, err os.Error) {
	// split text into cells
	n = 0
	for i, ch := range buf {
		if b.endChar == 0 {
			// outside escape
			switch ch {
			case '\t', '\v', '\n', '\f':
				// end of cell
				b.append(buf[n:i])
				b.updateWidth()
				n = i + 1 // ch consumed
				ncells := b.terminateCell(ch == '\t')
				if ch == '\n' || ch == '\f' {
					// terminate line
					b.addLine()
					if ch == '\f' || ncells == 1 {
						// A '\f' always forces a flush. Otherwise, if the previous
						// line has only one cell which does not have an impact on
						// the formatting of the following lines (the last cell per
						// line is ignored by format()), thus we can flush the
						// Writer contents.
						if err = b.Flush(); err != nil {
							return
						}
						if ch == '\f' && b.flags&Debug != 0 {
							// indicate section break
							if err = b.write0(hbar); err != nil {
								return
							}
						}
					}
				}

			case Escape:
				// start of escaped sequence
				b.append(buf[n:i])
				b.updateWidth()
				n = i + 1 // exclude Escape
				b.startEscape(Escape)

			case '<', '&':
				// possibly an html tag/entity
				if b.flags&FilterHTML != 0 {
					// begin of tag/entity
					b.append(buf[n:i])
					b.updateWidth()
					n = i
					b.startEscape(ch)
				}
			}

		} else {
			// inside escape
			if ch == b.endChar {
				// end of tag/entity
				j := i + 1
				if ch == Escape {
					j = i // exclude Escape
				}
				b.append(buf[n:j])
				n = i + 1 // ch consumed
				b.endEscape()
			}
		}
	}

	// append leftover text
	b.append(buf[n:])
	n = len(buf)
	return
}


// NewWriter allocates and initializes a new tabwriter.Writer.
// The parameters are the same as for the the Init function.
//
func NewWriter(output io.Writer, minwidth, tabwidth, padding int, padchar byte, flags uint) *Writer {
	return new(Writer).Init(output, minwidth, tabwidth, padding, padchar, flags)
}
