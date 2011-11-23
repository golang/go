// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package printer implements printing of AST nodes.
package printer

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
)

const debug = false // enable for debugging
const infinity = 1 << 30

type whiteSpace byte

const (
	ignore   = whiteSpace(0)
	blank    = whiteSpace(' ')
	vtab     = whiteSpace('\v')
	newline  = whiteSpace('\n')
	formfeed = whiteSpace('\f')
	indent   = whiteSpace('>')
	unindent = whiteSpace('<')
)

// Use ignoreMultiLine if the multiLine information is not important.
var ignoreMultiLine = new(bool)

// A pmode value represents the current printer mode.
type pmode int

const (
	noExtraLinebreak pmode = 1 << iota
)

type printer struct {
	// Configuration (does not change after initialization)
	Config
	fset *token.FileSet

	// Current state
	output  bytes.Buffer // raw printer result
	indent  int          // current indentation
	mode    pmode        // current printer mode
	lastTok token.Token  // the last token printed (token.ILLEGAL if it's whitespace)
	wsbuf   []whiteSpace // delayed white space

	// The (possibly estimated) position in the generated output;
	// in AST space (i.e., pos is set whenever a token position is
	// known accurately, and updated dependending on what has been
	// written).
	pos token.Position

	// The value of pos immediately after the last item has been
	// written using writeItem.
	last token.Position

	// The list of all source comments, in order of appearance.
	comments        []*ast.CommentGroup // may be nil
	cindex          int                 // current comment index
	useNodeComments bool                // if not set, ignore lead and line comments of nodes

	// Cache of already computed node sizes.
	nodeSizes map[ast.Node]int
}

func (p *printer) init(cfg *Config, fset *token.FileSet, nodeSizes map[ast.Node]int) {
	p.Config = *cfg
	p.fset = fset
	p.wsbuf = make([]whiteSpace, 0, 16) // whitespace sequences are short
	p.nodeSizes = nodeSizes
}

func (p *printer) internalError(msg ...interface{}) {
	if debug {
		fmt.Print(p.pos.String() + ": ")
		fmt.Println(msg...)
		panic("go/printer")
	}
}

// nlines returns the adjusted number of linebreaks given the desired number
// of breaks n such that min <= result <= max.
//
func (p *printer) nlines(n, min int) int {
	const max = 2 // max. number of newlines
	switch {
	case n < min:
		return min
	case n > max:
		return max
	}
	return n
}

// writeByte writes a single byte to p.output and updates p.pos.
func (p *printer) writeByte(ch byte) {
	p.output.WriteByte(ch)
	p.pos.Offset++
	p.pos.Column++

	if ch == '\n' || ch == '\f' {
		// write indentation
		// use "hard" htabs - indentation columns
		// must not be discarded by the tabwriter
		const htabs = "\t\t\t\t\t\t\t\t"
		j := p.indent
		for j > len(htabs) {
			p.output.WriteString(htabs)
			j -= len(htabs)
		}
		p.output.WriteString(htabs[0:j])

		// update p.pos
		p.pos.Line++
		p.pos.Offset += p.indent
		p.pos.Column = 1 + p.indent
	}
}

// writeNewlines writes up to n newlines to p.output and updates p.pos.
// The actual number of newlines written is limited by nlines.
// nl must be one of '\n' or '\f'.
//
func (p *printer) writeNewlines(n int, nl byte) {
	for n = p.nlines(n, 0); n > 0; n-- {
		p.writeByte(nl)
	}
}

// writeString writes the string s to p.output and updates p.pos.
// If isLit is set, s is escaped w/ tabwriter.Escape characters
// to protect s from being interpreted by the tabwriter.
//
// Note: writeString is only used to write Go tokens, literals, and
// comments, all of which must be written literally. Thus, it is correct
// to always set isLit = true. However, setting it explicitly only when
// needed (i.e., when we don't know that s contains no tabs or line breaks)
// avoids processing extra escape characters and reduces run time of the
// printer benchmark by up to 10%.
//
func (p *printer) writeString(s string, isLit bool) {
	if isLit {
		// Protect s such that is passes through the tabwriter
		// unchanged. Note that valid Go programs cannot contain
		// tabwriter.Escape bytes since they do not appear in legal
		// UTF-8 sequences.
		p.output.WriteByte(tabwriter.Escape)
	}

	p.output.WriteString(s)

	// update p.pos
	nlines := 0
	column := p.pos.Column + len(s)
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			nlines++
			column = len(s) - i
		}
	}
	p.pos.Offset += len(s)
	p.pos.Line += nlines
	p.pos.Column = column

	if isLit {
		p.output.WriteByte(tabwriter.Escape)
	}
}

// writeItem writes data at position pos. data is the text corresponding to
// a single lexical token, but may also be comment text. pos is the actual
// (or at least very accurately estimated) position of the data in the original
// source text. writeItem updates p.last to the position immediately following
// the data.
//
func (p *printer) writeItem(pos token.Position, data string, isLit bool) {
	if pos.IsValid() {
		// continue with previous position if we don't have a valid pos
		if p.last.IsValid() && p.last.Filename != pos.Filename {
			// the file has changed - reset state
			// (used when printing merged ASTs of different files
			// e.g., the result of ast.MergePackageFiles)
			p.indent = 0
			p.mode = 0
			p.wsbuf = p.wsbuf[0:0]
		}
		p.pos = pos
	}
	if debug {
		// do not update p.pos - use write0
		_, filename := filepath.Split(pos.Filename)
		fmt.Fprintf(&p.output, "[%s:%d:%d]", filename, pos.Line, pos.Column)
	}
	p.writeString(data, isLit)
	p.last = p.pos
}

const linePrefix = "//line "

// writeCommentPrefix writes the whitespace before a comment.
// If there is any pending whitespace, it consumes as much of
// it as is likely to help position the comment nicely.
// pos is the comment position, next the position of the item
// after all pending comments, prev is the previous comment in
// a group of comments (or nil), and isKeyword indicates if the
// next item is a keyword.
//
func (p *printer) writeCommentPrefix(pos, next token.Position, prev, comment *ast.Comment, isKeyword bool) {
	if p.output.Len() == 0 {
		// the comment is the first item to be printed - don't write any whitespace
		return
	}

	if pos.IsValid() && pos.Filename != p.last.Filename {
		// comment in a different file - separate with newlines (writeNewlines will limit the number)
		p.writeNewlines(10, '\f')
		return
	}

	if pos.Line == p.last.Line && (prev == nil || prev.Text[1] != '/') {
		// comment on the same line as last item:
		// separate with at least one separator
		hasSep := false
		if prev == nil {
			// first comment of a comment group
			j := 0
			for i, ch := range p.wsbuf {
				switch ch {
				case blank:
					// ignore any blanks before a comment
					p.wsbuf[i] = ignore
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
			sep := byte('\t')
			if pos.Line == next.Line {
				// next item is on the same line as the comment
				// (which must be a /*-style comment): separate
				// with a blank instead of a tab
				sep = ' '
			}
			p.writeByte(sep)
		}

	} else {
		// comment on a different line:
		// separate with at least one line break
		if prev == nil {
			// first comment of a comment group
			j := 0
			for i, ch := range p.wsbuf {
				switch ch {
				case blank, vtab:
					// ignore any horizontal whitespace before line breaks
					p.wsbuf[i] = ignore
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
					p.wsbuf[i] = ignore
				}
				j = i
				break
			}
			p.writeWhitespace(j)
		}

		// turn off indent if we're about to print a line directive.
		indent := p.indent
		if strings.HasPrefix(comment.Text, linePrefix) {
			p.indent = 0
		}

		// use formfeeds to break columns before a comment;
		// this is analogous to using formfeeds to separate
		// individual lines of /*-style comments - but make
		// sure there is at least one line break if the previous
		// comment was a line comment
		n := pos.Line - p.last.Line // if !pos.IsValid(), pos.Line == 0, and n will be 0
		if n <= 0 && prev != nil && prev.Text[1] == '/' {
			n = 1
		}
		if n > 0 {
			p.writeNewlines(n, '\f')
		}
		p.indent = indent
	}
}

// Split comment text into lines
// (using strings.Split(text, "\n") is significantly slower for
// this specific purpose, as measured with: gotest -bench=Print)
func split(text string) []string {
	// count lines (comment text never ends in a newline)
	n := 1
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			n++
		}
	}

	// split
	lines := make([]string, n)
	n = 0
	i := 0
	for j := 0; j < len(text); j++ {
		if text[j] == '\n' {
			lines[n] = text[i:j] // exclude newline
			i = j + 1            // discard newline
			n++
		}
	}
	lines[n] = text[i:]

	return lines
}

// Returns true if s contains only white space
// (only tabs and blanks can appear in the printer's context).
func isBlank(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] > ' ' {
			return false
		}
	}
	return true
}

func commonPrefix(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] && (a[i] <= ' ' || a[i] == '*') {
		i++
	}
	return a[0:i]
}

func stripCommonPrefix(lines []string) {
	if len(lines) < 2 {
		return // at most one line - nothing to do
	}
	// len(lines) >= 2

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
	// In case of two-line comments, consider the last line for
	// the prefix computation since otherwise the prefix would
	// be empty.
	//
	// Note that the first and last line are never empty (they
	// contain the opening /* and closing */ respectively) and
	// thus they can be ignored by the blank line check.
	var prefix string
	if len(lines) > 2 {
		first := true
		for i, line := range lines[1 : len(lines)-1] {
			switch {
			case isBlank(line):
				lines[1+i] = "" // range starts at line 1
			case first:
				prefix = commonPrefix(line, line)
				first = false
			default:
				prefix = commonPrefix(prefix, line)
			}
		}
	} else { // len(lines) == 2, lines cannot be blank (contain /* and */)
		line := lines[1]
		prefix = commonPrefix(line, line)
	}

	/*
	 * Check for vertical "line of stars" and correct prefix accordingly.
	 */
	lineOfStars := false
	if i := strings.Index(prefix, "*"); i >= 0 {
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
			n := 2 // start after opening /*
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
			if strings.HasSuffix(prefix, string(suffix)) {
				prefix = prefix[0 : len(prefix)-len(suffix)]
			}
		}
	}

	// Handle last line: If it only contains a closing */, align it
	// with the opening /*, otherwise align the text with the other
	// lines.
	last := lines[len(lines)-1]
	closing := "*/"
	i := strings.Index(last, closing) // i >= 0 (closing is always present)
	if isBlank(last[0:i]) {
		// last line only contains closing */
		if lineOfStars {
			closing = " */" // add blank to align final star
		}
		lines[len(lines)-1] = prefix + closing
	} else {
		// last line contains more comment text - assume
		// it is aligned like the other lines and include
		// in prefix computation
		prefix = commonPrefix(prefix, last)
	}

	// Remove the common prefix from all but the first and empty lines.
	for i, line := range lines[1:] {
		if len(line) != 0 {
			lines[1+i] = line[len(prefix):] // range starts at line 1
		}
	}
}

func (p *printer) writeComment(comment *ast.Comment) {
	text := comment.Text

	if strings.HasPrefix(text, linePrefix) {
		pos := strings.TrimSpace(text[len(linePrefix):])
		i := strings.LastIndex(pos, ":")
		if i >= 0 {
			// The line directive we are about to print changed
			// the Filename and Line number used by go/token
			// as it was reading the input originally.
			// In order to match the original input, we have to
			// update our own idea of the file and line number
			// accordingly, after printing the directive.
			file := pos[:i]
			line, _ := strconv.Atoi(pos[i+1:])
			defer func() {
				p.pos.Filename = file
				p.pos.Line = line
				p.pos.Column = 1
			}()
		}
	}

	// shortcut common case of //-style comments
	if text[1] == '/' {
		p.writeItem(p.fset.Position(comment.Pos()), text, true)
		return
	}

	// for /*-style comments, print line by line and let the
	// write function take care of the proper indentation
	lines := split(text)
	stripCommonPrefix(lines)

	// write comment lines, separated by formfeed,
	// without a line break after the last line
	pos := p.fset.Position(comment.Pos())
	for i, line := range lines {
		if i > 0 {
			p.writeByte('\f')
			pos = p.pos
		}
		if len(line) > 0 {
			p.writeItem(pos, line, true)
		}
	}
}

// writeCommentSuffix writes a line break after a comment if indicated
// and processes any leftover indentation information. If a line break
// is needed, the kind of break (newline vs formfeed) depends on the
// pending whitespace. writeCommentSuffix returns true if a pending
// formfeed was dropped from the whitespace buffer.
//
func (p *printer) writeCommentSuffix(needsLinebreak bool) (droppedFF bool) {
	for i, ch := range p.wsbuf {
		switch ch {
		case blank, vtab:
			// ignore trailing whitespace
			p.wsbuf[i] = ignore
		case indent, unindent:
			// don't lose indentation information
		case newline, formfeed:
			// if we need a line break, keep exactly one
			// but remember if we dropped any formfeeds
			if needsLinebreak {
				needsLinebreak = false
			} else {
				if ch == formfeed {
					droppedFF = true
				}
				p.wsbuf[i] = ignore
			}
		}
	}
	p.writeWhitespace(len(p.wsbuf))

	// make sure we have a line break
	if needsLinebreak {
		p.writeByte('\n')
	}

	return
}

// intersperseComments consumes all comments that appear before the next token
// tok and prints it together with the buffered whitespace (i.e., the whitespace
// that needs to be written before the next token). A heuristic is used to mix
// the comments and whitespace. intersperseComments returns true if a pending
// formfeed was dropped from the whitespace buffer.
//
func (p *printer) intersperseComments(next token.Position, tok token.Token) (droppedFF bool) {
	var last *ast.Comment
	for ; p.commentBefore(next); p.cindex++ {
		for _, c := range p.comments[p.cindex].List {
			p.writeCommentPrefix(p.fset.Position(c.Pos()), next, last, c, tok.IsKeyword())
			p.writeComment(c)
			last = c
		}
	}

	if last != nil {
		if last.Text[1] == '*' && p.fset.Position(last.Pos()).Line == next.Line {
			// the last comment is a /*-style comment and the next item
			// follows on the same line: separate with an extra blank
			p.writeByte(' ')
		}
		// ensure that there is a line break after a //-style comment,
		// before a closing '}' unless explicitly disabled, or at eof
		needsLinebreak :=
			last.Text[1] == '/' ||
				tok == token.RBRACE && p.mode&noExtraLinebreak == 0 ||
				tok == token.EOF
		return p.writeCommentSuffix(needsLinebreak)
	}

	// no comment was written - we should never reach here since
	// intersperseComments should not be called in that case
	p.internalError("intersperseComments called without pending comments")
	return false
}

// whiteWhitespace writes the first n whitespace entries.
func (p *printer) writeWhitespace(n int) {
	// write entries
	for i := 0; i < n; i++ {
		switch ch := p.wsbuf[i]; ch {
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
			if i+1 < n && p.wsbuf[i+1] == unindent {
				// Use a formfeed to terminate the current section.
				// Otherwise, a long label name on the next line leading
				// to a wide column may increase the indentation column
				// of lines before the label; effectively leading to wrong
				// indentation.
				p.wsbuf[i], p.wsbuf[i+1] = unindent, formfeed
				i-- // do it again
				continue
			}
			fallthrough
		default:
			p.writeByte(byte(ch))
		}
	}

	// shift remaining entries down
	i := 0
	for ; n < len(p.wsbuf); n++ {
		p.wsbuf[i] = p.wsbuf[n]
		i++
	}
	p.wsbuf = p.wsbuf[0:i]
}

// ----------------------------------------------------------------------------
// Printing interface

func mayCombine(prev token.Token, next byte) (b bool) {
	switch prev {
	case token.INT:
		b = next == '.' // 1.
	case token.ADD:
		b = next == '+' // ++
	case token.SUB:
		b = next == '-' // --
	case token.QUO:
		b = next == '*' // /*
	case token.LSS:
		b = next == '-' || next == '<' // <- or <<
	case token.AND:
		b = next == '&' || next == '^' // && or &^
	}
	return
}

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
func (p *printer) print(args ...interface{}) {
	for _, f := range args {
		next := p.pos // estimated position of next item
		data := ""
		isLit := false
		var tok token.Token

		switch x := f.(type) {
		case pmode:
			// toggle printer mode
			p.mode ^= x
		case whiteSpace:
			if x == ignore {
				// don't add ignore's to the buffer; they
				// may screw up "correcting" unindents (see
				// LabeledStmt)
				break
			}
			i := len(p.wsbuf)
			if i == cap(p.wsbuf) {
				// Whitespace sequences are very short so this should
				// never happen. Handle gracefully (but possibly with
				// bad comment placement) if it does happen.
				p.writeWhitespace(i)
				i = 0
			}
			p.wsbuf = p.wsbuf[0 : i+1]
			p.wsbuf[i] = x
		case *ast.Ident:
			data = x.Name
			tok = token.IDENT
		case *ast.BasicLit:
			data = x.Value
			isLit = true
			tok = x.Kind
		case token.Token:
			s := x.String()
			if mayCombine(p.lastTok, s[0]) {
				// the previous and the current token must be
				// separated by a blank otherwise they combine
				// into a different incorrect token sequence
				// (except for token.INT followed by a '.' this
				// should never happen because it is taken care
				// of via binary expression formatting)
				if len(p.wsbuf) != 0 {
					p.internalError("whitespace buffer not empty")
				}
				p.wsbuf = p.wsbuf[0:1]
				p.wsbuf[0] = ' '
			}
			data = s
			tok = x
		case token.Pos:
			if x.IsValid() {
				next = p.fset.Position(x) // accurate position of next item
			}
			tok = p.lastTok
		default:
			fmt.Fprintf(os.Stderr, "print: unsupported argument type %T\n", f)
			panic("go/printer type")
		}
		p.lastTok = tok
		p.pos = next

		if data != "" {
			nl := byte('\n')
			if p.flush(next, tok) {
				nl = '\f' // dropped formfeed before
			}

			// intersperse extra newlines if present in the source
			// (don't do this in flush as it will cause extra newlines
			// at the end of a file) - use formfeeds if we dropped one
			// before
			if n := next.Line - p.pos.Line; n > 0 {
				p.writeNewlines(n, nl)
			}

			p.writeItem(next, data, isLit)
		}
	}
}

// commentBefore returns true iff the current comment occurs
// before the next position in the source code.
//
func (p *printer) commentBefore(next token.Position) bool {
	return p.cindex < len(p.comments) && p.fset.Position(p.comments[p.cindex].List[0].Pos()).Offset < next.Offset
}

// Flush prints any pending comments and whitespace occurring
// textually before the position of the next token tok. Flush
// returns true if a pending formfeed character was dropped
// from the whitespace buffer as a result of interspersing
// comments.
//
func (p *printer) flush(next token.Position, tok token.Token) (droppedFF bool) {
	if p.commentBefore(next) {
		// if there are comments before the next item, intersperse them
		droppedFF = p.intersperseComments(next, tok)
	} else {
		// otherwise, write any leftover whitespace
		p.writeWhitespace(len(p.wsbuf))
	}
	return
}

func (p *printer) printNode(node interface{}) error {
	switch n := node.(type) {
	case ast.Expr:
		p.useNodeComments = true
		p.expr(n, ignoreMultiLine)
	case ast.Stmt:
		p.useNodeComments = true
		// A labeled statement will un-indent to position the
		// label. Set indent to 1 so we don't get indent "underflow".
		if _, labeledStmt := n.(*ast.LabeledStmt); labeledStmt {
			p.indent = 1
		}
		p.stmt(n, false, ignoreMultiLine)
	case ast.Decl:
		p.useNodeComments = true
		p.decl(n, ignoreMultiLine)
	case ast.Spec:
		p.useNodeComments = true
		p.spec(n, 1, false, ignoreMultiLine)
	case *ast.File:
		p.comments = n.Comments
		p.useNodeComments = n.Comments == nil
		p.file(n)
	default:
		return fmt.Errorf("go/printer: unsupported node type %T", n)
	}
	return nil
}

// ----------------------------------------------------------------------------
// Trimmer

// A trimmer is an io.Writer filter for stripping tabwriter.Escape
// characters, trailing blanks and tabs, and for converting formfeed
// and vtab characters into newlines and htabs (in case no tabwriter
// is used). Text bracketed by tabwriter.Escape characters is passed
// through unchanged.
//
type trimmer struct {
	output io.Writer
	state  int
	space  bytes.Buffer
}

// trimmer is implemented as a state machine.
// It can be in one of the following states:
const (
	inSpace  = iota // inside space
	inEscape        // inside text bracketed by tabwriter.Escapes
	inText          // inside text
)

// Design note: It is tempting to eliminate extra blanks occurring in
//              whitespace in this function as it could simplify some
//              of the blanks logic in the node printing functions.
//              However, this would mess up any formatting done by
//              the tabwriter.

var aNewline = []byte("\n")

func (p *trimmer) Write(data []byte) (n int, err error) {
	// invariants:
	// p.state == inSpace:
	//	p.space is unwritten
	// p.state == inEscape, inText:
	//	data[m:n] is unwritten
	m := 0
	var b byte
	for n, b = range data {
		if b == '\v' {
			b = '\t' // convert to htab
		}
		switch p.state {
		case inSpace:
			switch b {
			case '\t', ' ':
				p.space.WriteByte(b) // WriteByte returns no errors
			case '\n', '\f':
				p.space.Reset() // discard trailing space
				_, err = p.output.Write(aNewline)
			case tabwriter.Escape:
				_, err = p.output.Write(p.space.Bytes())
				p.state = inEscape
				m = n + 1 // +1: skip tabwriter.Escape
			default:
				_, err = p.output.Write(p.space.Bytes())
				p.state = inText
				m = n
			}
		case inEscape:
			if b == tabwriter.Escape {
				_, err = p.output.Write(data[m:n])
				p.state = inSpace
				p.space.Reset()
			}
		case inText:
			switch b {
			case '\t', ' ':
				_, err = p.output.Write(data[m:n])
				p.state = inSpace
				p.space.Reset()
				p.space.WriteByte(b) // WriteByte returns no errors
			case '\n', '\f':
				_, err = p.output.Write(data[m:n])
				p.state = inSpace
				p.space.Reset()
				_, err = p.output.Write(aNewline)
			case tabwriter.Escape:
				_, err = p.output.Write(data[m:n])
				p.state = inEscape
				m = n + 1 // +1: skip tabwriter.Escape
			}
		default:
			panic("unreachable")
		}
		if err != nil {
			return
		}
	}
	n = len(data)

	switch p.state {
	case inEscape, inText:
		_, err = p.output.Write(data[m:n])
		p.state = inSpace
		p.space.Reset()
	}

	return
}

// ----------------------------------------------------------------------------
// Public interface

// General printing is controlled with these Config.Mode flags.
const (
	RawFormat uint = 1 << iota // do not use a tabwriter; if set, UseSpaces is ignored
	TabIndent                  // use tabs for indentation independent of UseSpaces
	UseSpaces                  // use spaces instead of tabs for alignment
)

// A Config node controls the output of Fprint.
type Config struct {
	Mode     uint // default: 0
	Tabwidth int  // default: 8
}

// fprint implements Fprint and takes a nodesSizes map for setting up the printer state.
func (cfg *Config) fprint(output io.Writer, fset *token.FileSet, node interface{}, nodeSizes map[ast.Node]int) (err error) {
	// print node
	var p printer
	p.init(cfg, fset, nodeSizes)
	if err = p.printNode(node); err != nil {
		return
	}
	p.flush(token.Position{Offset: infinity, Line: infinity}, token.EOF)

	// redirect output through a trimmer to eliminate trailing whitespace
	// (Input to a tabwriter must be untrimmed since trailing tabs provide
	// formatting information. The tabwriter could provide trimming
	// functionality but no tabwriter is used when RawFormat is set.)
	output = &trimmer{output: output}

	// redirect output through a tabwriter if necessary
	if cfg.Mode&RawFormat == 0 {
		minwidth := cfg.Tabwidth

		padchar := byte('\t')
		if cfg.Mode&UseSpaces != 0 {
			padchar = ' '
		}

		twmode := tabwriter.DiscardEmptyColumns
		if cfg.Mode&TabIndent != 0 {
			minwidth = 0
			twmode |= tabwriter.TabIndent
		}

		output = tabwriter.NewWriter(output, minwidth, cfg.Tabwidth, 1, padchar, twmode)
	}

	// write printer result via tabwriter/trimmer to output
	if _, err = output.Write(p.output.Bytes()); err != nil {
		return
	}

	// flush tabwriter, if any
	if tw, _ := (output).(*tabwriter.Writer); tw != nil {
		err = tw.Flush()
	}

	return
}

// Fprint "pretty-prints" an AST node to output for a given configuration cfg.
// Position information is interpreted relative to the file set fset.
// The node type must be *ast.File, or assignment-compatible to ast.Expr,
// ast.Decl, ast.Spec, or ast.Stmt.
//
func (cfg *Config) Fprint(output io.Writer, fset *token.FileSet, node interface{}) error {
	return cfg.fprint(output, fset, node, make(map[ast.Node]int))
}

// Fprint "pretty-prints" an AST node to output.
// It calls Config.Fprint with default settings.
//
func Fprint(output io.Writer, fset *token.FileSet, node interface{}) error {
	return (&Config{Tabwidth: 8}).Fprint(output, fset, node)
}
