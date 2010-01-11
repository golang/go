// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The printer package implements printing of AST nodes.
package printer

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"os"
	"reflect"
	"runtime"
	"strings"
	"tabwriter"
)


const (
	debug       = false // enable for debugging
	maxNewlines = 3     // maximum vertical white space
)


type whiteSpace int

const (
	ignore   = whiteSpace(0)
	blank    = whiteSpace(' ')
	vtab     = whiteSpace('\v')
	newline  = whiteSpace('\n')
	formfeed = whiteSpace('\f')
	indent   = whiteSpace('>')
	unindent = whiteSpace('<')
)


var (
	esc       = []byte{tabwriter.Escape}
	htab      = []byte{'\t'}
	htabs     = [...]byte{'\t', '\t', '\t', '\t', '\t', '\t', '\t', '\t'}
	newlines  = [...]byte{'\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n'} // more than maxNewlines
	formfeeds = [...]byte{'\f', '\f', '\f', '\f', '\f', '\f', '\f', '\f'} // more than maxNewlines

	esc_quot = strings.Bytes("&#34;") // shorter than "&quot;"
	esc_apos = strings.Bytes("&#39;") // shorter than "&apos;"
	esc_amp  = strings.Bytes("&amp;")
	esc_lt   = strings.Bytes("&lt;")
	esc_gt   = strings.Bytes("&gt;")
)


// Use noPos when a position is needed but not known.
var noPos token.Position


// Use ignoreMultiLine if the multiLine information is not important.
var ignoreMultiLine = new(bool)


type printer struct {
	// Configuration (does not change after initialization)
	output io.Writer
	Config
	errors chan os.Error

	// Current state
	written int  // number of bytes written
	indent  int  // current indentation
	escape  bool // true if in escape sequence

	// Buffered whitespace
	buffer []whiteSpace

	// The (possibly estimated) position in the generated output;
	// in AST space (i.e., pos is set whenever a token position is
	// known accurately, and updated dependending on what has been
	// written)
	pos token.Position

	// The value of pos immediately after the last item has been
	// written using writeItem.
	last token.Position

	// HTML support
	lastTaggedLine int // last line for which a line tag was written

	// The list of comments; or nil.
	comment *ast.CommentGroup
}


func (p *printer) init(output io.Writer, cfg *Config) {
	p.output = output
	p.Config = *cfg
	p.errors = make(chan os.Error)
	p.buffer = make([]whiteSpace, 0, 16) // whitespace sequences are short
}


func (p *printer) internalError(msg ...) {
	if debug {
		fmt.Print(p.pos.String() + ": ")
		fmt.Println(msg)
		panic()
	}
}


// write0 writes raw (uninterpreted) data to p.output and handles errors.
// write0 does not indent after newlines, and does not HTML-escape or update p.pos.
//
func (p *printer) write0(data []byte) {
	n, err := p.output.Write(data)
	p.written += n
	if err != nil {
		p.errors <- err
		runtime.Goexit()
	}
}


// write interprets data and writes it to p.output. It inserts indentation
// after a line break unless in a tabwriter escape sequence, and it HTML-
// escapes characters if GenHTML is set. It updates p.pos as a side-effect.
//
func (p *printer) write(data []byte) {
	i0 := 0
	for i, b := range data {
		switch b {
		case '\n', '\f':
			// write segment ending in b
			p.write0(data[i0 : i+1])

			// update p.pos
			p.pos.Offset += i + 1 - i0
			p.pos.Line++
			p.pos.Column = 1

			if !p.escape {
				// write indentation
				// use "hard" htabs - indentation columns
				// must not be discarded by the tabwriter
				j := p.indent
				for ; j > len(htabs); j -= len(htabs) {
					p.write0(&htabs)
				}
				p.write0(htabs[0:j])

				// update p.pos
				p.pos.Offset += p.indent
				p.pos.Column += p.indent
			}

			// next segment start
			i0 = i + 1

		case '"', '\'', '&', '<', '>':
			if p.Mode&GenHTML != 0 {
				// write segment ending in b
				p.write0(data[i0:i])

				// write HTML-escaped b
				var esc []byte
				switch b {
				case '"':
					esc = esc_quot
				case '\'':
					esc = esc_apos
				case '&':
					esc = esc_amp
				case '<':
					esc = esc_lt
				case '>':
					esc = esc_gt
				}
				p.write0(esc)

				// update p.pos
				d := i + 1 - i0
				p.pos.Offset += d
				p.pos.Column += d

				// next segment start
				i0 = i + 1
			}

		case tabwriter.Escape:
			p.escape = !p.escape
		}
	}

	// write remaining segment
	p.write0(data[i0:])

	// update p.pos
	d := len(data) - i0
	p.pos.Offset += d
	p.pos.Column += d
}


func (p *printer) writeNewlines(n int) {
	if n > 0 {
		if n > maxNewlines {
			n = maxNewlines
		}
		p.write(newlines[0:n])
	}
}


func (p *printer) writeFormfeeds(n int) {
	if n > 0 {
		if n > maxNewlines {
			n = maxNewlines
		}
		p.write(formfeeds[0:n])
	}
}


func (p *printer) writeTaggedItem(data []byte, tag HTMLTag) {
	// write start tag, if any
	// (no html-escaping and no p.pos update for tags - use write0)
	if tag.Start != "" {
		p.write0(strings.Bytes(tag.Start))
	}
	p.write(data)
	// write end tag, if any
	if tag.End != "" {
		p.write0(strings.Bytes(tag.End))
	}
}


// writeItem writes data at position pos. data is the text corresponding to
// a single lexical token, but may also be comment text. pos is the actual
// (or at least very accurately estimated) position of the data in the original
// source text. If tags are present and GenHTML is set, the tags are written
// before and after the data. writeItem updates p.last to the position
// immediately following the data.
//
func (p *printer) writeItem(pos token.Position, data []byte, tag HTMLTag) {
	p.pos = pos
	if debug {
		// do not update p.pos - use write0
		p.write0(strings.Bytes(fmt.Sprintf("[%d:%d]", pos.Line, pos.Column)))
	}
	if p.Mode&GenHTML != 0 {
		// write line tag if on a new line
		// TODO(gri): should write line tags on each line at the start
		//            will be more useful (e.g. to show line numbers)
		if p.Styler != nil && pos.Line > p.lastTaggedLine {
			p.writeTaggedItem(p.Styler.LineTag(pos.Line))
			p.lastTaggedLine = pos.Line
		}
		p.writeTaggedItem(data, tag)
	} else {
		p.write(data)
	}
	p.last = p.pos
}


// writeCommentPrefix writes the whitespace before a comment.
// If there is any pending whitespace, it consumes as much of
// it as is likely to help the comment position properly.
// pos is the comment position, next the position of the item
// after all pending comments, isFirst indicates if this is the
// first comment in a group of comments, and isKeyword indicates
// if the next item is a keyword.
//
func (p *printer) writeCommentPrefix(pos, next token.Position, isFirst, isKeyword bool) {
	if !p.last.IsValid() {
		// there was no preceeding item and the comment is the
		// first item to be printed - don't write any whitespace
		return
	}

	if pos.Line == p.last.Line {
		// comment on the same line as last item:
		// separate with at least one separator
		hasSep := false
		if isFirst {
			j := 0
			for i, ch := range p.buffer {
				switch ch {
				case blank:
					// ignore any blanks before a comment
					p.buffer[i] = ignore
					continue
				case vtab:
					// respect existing tabs - important
					// for proper formatting of commented structs
					hasSep = true
					continue
				case indent:
					// apply pending indentation
					continue
				}
				j = i
				break
			}
			p.writeWhitespace(j)
		}
		// make sure there is at least one separator
		if !hasSep {
			if pos.Line == next.Line {
				// next item is on the same line as the comment
				// (which must be a /*-style comment): separate
				// with a blank instead of a tab
				p.write([]byte{' '})
			} else {
				p.write(htab)
			}
		}

	} else {
		// comment on a different line:
		// separate with at least one line break
		if isFirst {
			j := 0
			for i, ch := range p.buffer {
				switch ch {
				case blank, vtab:
					// ignore any horizontal whitespace before line breaks
					p.buffer[i] = ignore
					continue
				case indent:
					// apply pending indentation
					continue
				case unindent:
					// if the next token is a keyword, apply the outdent
					// if it appears that the comment is aligned with the
					// keyword; otherwise assume the outdent is part of a
					// closing block and stop (this scenario appears with
					// comments before a case label where the comments
					// apply to the next case instead of the current one)
					if isKeyword && pos.Column == next.Column {
						continue
					}
				case newline, formfeed:
					// TODO(gri): may want to keep formfeed info in some cases
					p.buffer[i] = ignore
				}
				j = i
				break
			}
			p.writeWhitespace(j)
		}
		// use formfeeds to break columns before a comment;
		// this is analogous to using formfeeds to separate
		// individual lines of /*-style comments
		p.writeFormfeeds(pos.Line - p.last.Line)
	}
}


func (p *printer) writeCommentLine(comment *ast.Comment, pos token.Position, line []byte) {
	// line must pass through unchanged, bracket it with tabwriter.Escape
	esc := []byte{tabwriter.Escape}
	line = bytes.Join([][]byte{esc, line, esc}, nil)

	// apply styler, if any
	var tag HTMLTag
	if p.Styler != nil {
		line, tag = p.Styler.Comment(comment, line)
	}

	p.writeItem(pos, line, tag)
}


// TODO(gri): Similar (but not quite identical) functionality for
//            comment processing can be found in go/doc/comment.go.
//            Perhaps this can be factored eventually.

// Split comment text into lines
func split(text []byte) [][]byte {
	// count lines (comment text never ends in a newline)
	n := 1
	for _, c := range text {
		if c == '\n' {
			n++
		}
	}

	// split
	lines := make([][]byte, n)
	n = 0
	i := 0
	for j, c := range text {
		if c == '\n' {
			lines[n] = text[i:j] // exclude newline
			i = j + 1            // discard newline
			n++
		}
	}
	lines[n] = text[i:]

	return lines
}


func isBlank(s []byte) bool {
	for _, b := range s {
		if b > ' ' {
			return false
		}
	}
	return true
}


func commonPrefix(a, b []byte) []byte {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] && (a[i] <= ' ' || a[i] == '*') {
		i++
	}
	return a[0:i]
}


func stripCommonPrefix(lines [][]byte) {
	if len(lines) < 2 {
		return // at most one line - nothing to do
	}

	// The heuristic in this function tries to handle a few
	// common patterns of /*-style comments: Comments where
	// the opening /* and closing */ are aligned and the
	// rest of the comment text is aligned and indented with
	// blanks or tabs, cases with a vertical "line of stars"
	// on the left, and cases where the closing */ is on the
	// same line as the last comment text.

	// Compute maximum common white prefix of all but the first,
	// last, and blank lines, and replace blank lines with empty
	// lines (the first line starts with /* and has no prefix).
	var prefix []byte
	for i, line := range lines {
		switch {
		case i == 0 || i == len(lines)-1:
			// ignore
		case isBlank(line):
			lines[i] = nil
		case prefix == nil:
			prefix = commonPrefix(line, line)
		default:
			prefix = commonPrefix(prefix, line)
		}
	}

	/*
	 * Check for vertical "line of stars" and correct prefix accordingly.
	 */
	lineOfStars := false
	if i := bytes.Index(prefix, []byte{'*'}); i >= 0 {
		// Line of stars present.
		if i > 0 && prefix[i-1] == ' ' {
			i-- // remove trailing blank from prefix so stars remain aligned
		}
		prefix = prefix[0:i]
		lineOfStars = true
	} else {
		// No line of stars present.
		// Determine the white space on the first line after the /*
		// and before the beginning of the comment text, assume two
		// blanks instead of the /* unless the first character after
		// the /* is a tab. If the first comment line is empty but
		// for the opening /*, assume up to 3 blanks or a tab. This
		// whitespace may be found as suffix in the common prefix.
		first := lines[0]
		if isBlank(first[2:]) {
			// no comment text on the first line:
			// reduce prefix by up to 3 blanks or a tab
			// if present - this keeps comment text indented
			// relative to the /* and */'s if it was indented
			// in the first place
			i := len(prefix)
			for n := 0; n < 3 && i > 0 && prefix[i-1] == ' '; n++ {
				i--
			}
			if i == len(prefix) && i > 0 && prefix[i-1] == '\t' {
				i--
			}
			prefix = prefix[0:i]
		} else {
			// comment text on the first line
			suffix := make([]byte, len(first))
			n := 2
			for n < len(first) && first[n] <= ' ' {
				suffix[n] = first[n]
				n++
			}
			if n > 2 && suffix[2] == '\t' {
				// assume the '\t' compensates for the /*
				suffix = suffix[2:n]
			} else {
				// otherwise assume two blanks
				suffix[0], suffix[1] = ' ', ' '
				suffix = suffix[0:n]
			}
			// Shorten the computed common prefix by the length of
			// suffix, if it is found as suffix of the prefix.
			if bytes.HasSuffix(prefix, suffix) {
				prefix = prefix[0 : len(prefix)-len(suffix)]
			}
		}
	}

	// Handle last line: If it only contains a closing */, align it
	// with the opening /*, otherwise align the text with the other
	// lines.
	last := lines[len(lines)-1]
	closing := []byte{'*', '/'}
	i := bytes.Index(last, closing)
	if isBlank(last[0:i]) {
		// last line only contains closing */
		var sep []byte
		if lineOfStars {
			// insert an aligning blank
			sep = []byte{' '}
		}
		lines[len(lines)-1] = bytes.Join([][]byte{prefix, closing}, sep)
	} else {
		// last line contains more comment text - assume
		// it is aligned like the other lines
		prefix = commonPrefix(prefix, last)
	}

	// Remove the common prefix from all but the first and empty lines.
	for i, line := range lines {
		if i > 0 && len(line) != 0 {
			lines[i] = line[len(prefix):]
		}
	}
}


func (p *printer) writeComment(comment *ast.Comment) {
	text := comment.Text

	// shortcut common case of //-style comments
	if text[1] == '/' {
		p.writeCommentLine(comment, comment.Pos(), text)
		return
	}

	// for /*-style comments, print line by line and let the
	// write function take care of the proper indentation
	lines := split(text)
	stripCommonPrefix(lines)

	// write comment lines, separated by formfeed,
	// without a line break after the last line
	linebreak := formfeeds[0:1]
	pos := comment.Pos()
	for i, line := range lines {
		if i > 0 {
			p.write(linebreak)
			pos = p.pos
		}
		if len(line) > 0 {
			p.writeCommentLine(comment, pos, line)
		}
	}
}


// writeCommentSuffix writes a line break after a comment if indicated
// and processes any leftover indentation information. If a line break
// is needed, the kind of break (newline vs formfeed) depends on the
// pending whitespace.
//
func (p *printer) writeCommentSuffix(needsLinebreak bool) {
	for i, ch := range p.buffer {
		switch ch {
		case blank, vtab:
			// ignore trailing whitespace
			p.buffer[i] = ignore
		case indent, unindent:
			// don't loose indentation information
		case newline, formfeed:
			// if we need a line break, keep exactly one
			if needsLinebreak {
				needsLinebreak = false
			} else {
				p.buffer[i] = ignore
			}
		}
	}
	p.writeWhitespace(len(p.buffer))

	// make sure we have a line break
	if needsLinebreak {
		p.write([]byte{'\n'})
	}
}


// intersperseComments consumes all comments that appear before the next token
// and prints it together with the buffered whitespace (i.e., the whitespace
// that needs to be written before the next token). A heuristic is used to mix
// the comments and whitespace. The isKeyword parameter indicates if the next
// token is a keyword or not.
//
func (p *printer) intersperseComments(next token.Position, isKeyword bool) {
	isFirst := true
	needsLinebreak := false
	var last *ast.Comment
	for ; p.commentBefore(next); p.comment = p.comment.Next {
		for _, c := range p.comment.List {
			p.writeCommentPrefix(c.Pos(), next, isFirst, isKeyword)
			isFirst = false
			p.writeComment(c)
			needsLinebreak = c.Text[1] == '/'
			last = c
		}
	}
	if last != nil && !needsLinebreak && last.Pos().Line == next.Line {
		// the last comment is a /*-style comment and the next item
		// follows on the same line: separate with an extra blank
		p.write([]byte{' '})
	}
	p.writeCommentSuffix(needsLinebreak)
}


// whiteWhitespace writes the first n whitespace entries.
func (p *printer) writeWhitespace(n int) {
	// write entries
	var data [1]byte
	for i := 0; i < n; i++ {
		switch ch := p.buffer[i]; ch {
		case ignore:
			// ignore!
		case indent:
			p.indent++
		case unindent:
			p.indent--
			if p.indent < 0 {
				p.internalError("negative indentation:", p.indent)
				p.indent = 0
			}
		case newline, formfeed:
			// A line break immediately followed by a "correcting"
			// unindent is swapped with the unindent - this permits
			// proper label positioning. If a comment is between
			// the line break and the label, the unindent is not
			// part of the comment whitespace prefix and the comment
			// will be positioned correctly indented.
			if i+1 < n && p.buffer[i+1] == unindent {
				// Use a formfeed to terminate the current section.
				// Otherwise, a long label name on the next line leading
				// to a wide column may increase the indentation column
				// of lines before the label; effectively leading to wrong
				// indentation.
				p.buffer[i], p.buffer[i+1] = unindent, formfeed
				i-- // do it again
				continue
			}
			fallthrough
		default:
			data[0] = byte(ch)
			p.write(&data)
		}
	}

	// shift remaining entries down
	i := 0
	for ; n < len(p.buffer); n++ {
		p.buffer[i] = p.buffer[n]
		i++
	}
	p.buffer = p.buffer[0:i]
}


// ----------------------------------------------------------------------------
// Printing interface

// print prints a list of "items" (roughly corresponding to syntactic
// tokens, but also including whitespace and formatting information).
// It is the only print function that should be called directly from
// any of the AST printing functions in nodes.go.
//
// Whitespace is accumulated until a non-whitespace token appears. Any
// comments that need to appear before that token are printed first,
// taking into account the amount and structure of any pending white-
// space for best comment placement. Then, any leftover whitespace is
// printed, followed by the actual token.
//
func (p *printer) print(args ...) {
	v := reflect.NewValue(args).(*reflect.StructValue)
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)

		next := p.pos // estimated position of next item
		var data []byte
		var tag HTMLTag
		isKeyword := false
		switch x := f.Interface().(type) {
		case whiteSpace:
			if x == ignore {
				// don't add ignore's to the buffer; they
				// may screw up "correcting" unindents (see
				// LabeledStmt)
				break
			}
			i := len(p.buffer)
			if i == cap(p.buffer) {
				// Whitespace sequences are very short so this should
				// never happen. Handle gracefully (but possibly with
				// bad comment placement) if it does happen.
				p.writeWhitespace(i)
				i = 0
			}
			p.buffer = p.buffer[0 : i+1]
			p.buffer[i] = x
		case []byte:
			// TODO(gri): remove this case once commentList
			//            handles comments correctly
			data = x
		case string:
			// TODO(gri): remove this case once fieldList
			//            handles comments correctly
			data = strings.Bytes(x)
		case *ast.Ident:
			if p.Styler != nil {
				data, tag = p.Styler.Ident(x)
			} else {
				data = strings.Bytes(x.Value)
			}
		case *ast.BasicLit:
			if p.Styler != nil {
				data, tag = p.Styler.BasicLit(x)
			} else {
				data = x.Value
			}
			// escape all literals so they pass through unchanged
			// (note that valid Go programs cannot contain esc ('\xff')
			// bytes since they do not appear in legal UTF-8 sequences)
			// TODO(gri): do this more efficiently.
			data = strings.Bytes("\xff" + string(data) + "\xff")
		case token.Token:
			if p.Styler != nil {
				data, tag = p.Styler.Token(x)
			} else {
				data = strings.Bytes(x.String())
			}
			isKeyword = x.IsKeyword()
		case token.Position:
			if x.IsValid() {
				next = x // accurate position of next item
			}
		default:
			panicln("print: unsupported argument type", f.Type().String())
		}
		p.pos = next

		if data != nil {
			p.flush(next, isKeyword)

			// intersperse extra newlines if present in the source
			// (don't do this in flush as it will cause extra newlines
			// at the end of a file)
			p.writeNewlines(next.Line - p.pos.Line)

			p.writeItem(next, data, tag)
		}
	}
}


// commentBefore returns true iff the current comment occurs
// before the next position in the source code.
//
func (p *printer) commentBefore(next token.Position) bool {
	return p.comment != nil && p.comment.List[0].Pos().Offset < next.Offset
}


// Flush prints any pending comments and whitespace occuring
// textually before the position of the next item.
//
func (p *printer) flush(next token.Position, isKeyword bool) {
	// if there are comments before the next item, intersperse them
	if p.commentBefore(next) {
		p.intersperseComments(next, isKeyword)
	}
	// write any leftover whitespace
	p.writeWhitespace(len(p.buffer))
}


// ----------------------------------------------------------------------------
// Trimmer

// A trimmer is an io.Writer filter for stripping tabwriter.Escape
// characters, trailing blanks and tabs, and for converting formfeed
// and vtab characters into newlines and htabs (in case no tabwriter
// is used).
//
type trimmer struct {
	output io.Writer
	buf    bytes.Buffer
}


// Design note: It is tempting to eliminate extra blanks occuring in
//              whitespace in this function as it could simplify some
//              of the blanks logic in the node printing functions.
//              However, this would mess up any formatting done by
//              the tabwriter.

func (p *trimmer) Write(data []byte) (n int, err os.Error) {
	// m < 0: no unwritten data except for whitespace
	// m >= 0: data[m:n] unwritten and no whitespace
	m := 0
	if p.buf.Len() > 0 {
		m = -1
	}

	var b byte
	for n, b = range data {
		switch b {
		default:
			// write any pending whitespace
			if m < 0 {
				if _, err = p.output.Write(p.buf.Bytes()); err != nil {
					return
				}
				p.buf.Reset()
				m = n
			}

		case '\v':
			b = '\t' // convert to htab
			fallthrough

		case '\t', ' ', tabwriter.Escape:
			// write any pending (non-whitespace) data
			if m >= 0 {
				if _, err = p.output.Write(data[m:n]); err != nil {
					return
				}
				m = -1
			}
			// collect whitespace but discard tabrwiter.Escapes.
			if b != tabwriter.Escape {
				p.buf.WriteByte(b) // WriteByte returns no errors
			}

		case '\f', '\n':
			// discard whitespace
			p.buf.Reset()
			// write any pending (non-whitespace) data
			if m >= 0 {
				if _, err = p.output.Write(data[m:n]); err != nil {
					return
				}
				m = -1
			}
			// convert formfeed into newline
			if _, err = p.output.Write(newlines[0:1]); err != nil {
				return
			}
		}
	}
	n = len(data)

	// write any pending non-whitespace
	if m >= 0 {
		if _, err = p.output.Write(data[m:n]); err != nil {
			return
		}
	}

	return
}


// ----------------------------------------------------------------------------
// Public interface

// General printing is controlled with these Config.Mode flags.
const (
	GenHTML   uint = 1 << iota // generate HTML
	RawFormat      // do not use a tabwriter; if set, UseSpaces is ignored
	TabIndent      // use tabs for indentation independent of UseSpaces
	UseSpaces      // use spaces instead of tabs for alignment
	NoSemis        // don't print semicolons at the end of a line
)


// An HTMLTag specifies a start and end tag.
type HTMLTag struct {
	Start, End string // empty if tags are absent
}


// A Styler specifies formatting of line tags and elementary Go words.
// A format consists of text and a (possibly empty) surrounding HTML tag.
//
type Styler interface {
	LineTag(line int) ([]byte, HTMLTag)
	Comment(c *ast.Comment, line []byte) ([]byte, HTMLTag)
	BasicLit(x *ast.BasicLit) ([]byte, HTMLTag)
	Ident(id *ast.Ident) ([]byte, HTMLTag)
	Token(tok token.Token) ([]byte, HTMLTag)
}


// A Config node controls the output of Fprint.
type Config struct {
	Mode     uint   // default: 0
	Tabwidth int    // default: 8
	Styler   Styler // default: nil
}


// Fprint "pretty-prints" an AST node to output and returns the number
// of bytes written and an error (if any) for a given configuration cfg.
// The node type must be *ast.File, or assignment-compatible to ast.Expr,
// ast.Decl, or ast.Stmt.
//
func (cfg *Config) Fprint(output io.Writer, node interface{}) (int, os.Error) {
	// redirect output through a trimmer to eliminate trailing whitespace
	// (Input to a tabwriter must be untrimmed since trailing tabs provide
	// formatting information. The tabwriter could provide trimming
	// functionality but no tabwriter is used when RawFormat is set.)
	output = &trimmer{output: output}

	// setup tabwriter if needed and redirect output
	var tw *tabwriter.Writer
	if cfg.Mode&RawFormat == 0 {
		minwidth := cfg.Tabwidth

		padchar := byte('\t')
		if cfg.Mode&UseSpaces != 0 {
			padchar = ' '
		}

		twmode := tabwriter.DiscardEmptyColumns
		if cfg.Mode&GenHTML != 0 {
			twmode |= tabwriter.FilterHTML
		}
		if cfg.Mode&TabIndent != 0 {
			minwidth = 0
			twmode |= tabwriter.TabIndent
		}

		tw = tabwriter.NewWriter(output, minwidth, cfg.Tabwidth, 1, padchar, twmode)
		output = tw
	}

	// setup printer and print node
	var p printer
	p.init(output, cfg)
	go func() {
		switch n := node.(type) {
		case ast.Expr:
			p.expr(n, ignoreMultiLine)
		case ast.Stmt:
			p.stmt(n, ignoreMultiLine)
		case ast.Decl:
			p.decl(n, atTop, ignoreMultiLine)
		case *ast.File:
			p.comment = n.Comments
			p.file(n)
		default:
			p.errors <- os.NewError(fmt.Sprintf("printer.Fprint: unsupported node type %T", n))
			runtime.Goexit()
		}
		p.flush(token.Position{Offset: 1 << 30, Line: 1 << 30}, false) // flush to "infinity"
		p.errors <- nil                                                // no errors
	}()
	err := <-p.errors // wait for completion of goroutine

	// flush tabwriter, if any
	if tw != nil {
		tw.Flush() // ignore errors
	}

	return p.written, err
}


// Fprint "pretty-prints" an AST node to output.
// It calls Config.Fprint with default settings.
//
func Fprint(output io.Writer, node interface{}) os.Error {
	_, err := (&Config{Tabwidth: 8}).Fprint(output, node) // don't care about number of bytes written
	return err
}
