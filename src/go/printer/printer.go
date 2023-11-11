// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package printer implements printing of AST nodes.
package printer

import (
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/token"
	"io"
	"os"
	"strings"
	"sync"
	"text/tabwriter"
	"unicode"
)

const (
	maxNewlines = 2     // max. number of newlines between source text
	debug       = false // enable for debugging
	infinity    = 1 << 30
)

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

// A pmode value represents the current printer mode.
type pmode int

const (
	noExtraBlank     pmode = 1 << iota // disables extra blank after /*-style comment
	noExtraLinebreak                   // disables extra line break after /*-style comment
)

type commentInfo struct {
	cindex         int               // current comment index
	comment        *ast.CommentGroup // = printer.comments[cindex]; or nil
	commentOffset  int               // = printer.posFor(printer.comments[cindex].List[0].Pos()).Offset; or infinity
	commentNewline bool              // true if the comment group contains newlines
}

type printer struct {
	// Configuration (does not change after initialization)
	Config
	fset *token.FileSet

	// Current state
	output       []byte       // raw printer result
	indent       int          // current indentation
	level        int          // level == 0: outside composite literal; level > 0: inside composite literal
	mode         pmode        // current printer mode
	endAlignment bool         // if set, terminate alignment immediately
	impliedSemi  bool         // if set, a linebreak implies a semicolon
	lastTok      token.Token  // last token printed (token.ILLEGAL if it's whitespace)
	prevOpen     token.Token  // previous non-brace "open" token (, [, or token.ILLEGAL
	wsbuf        []whiteSpace // delayed white space
	goBuild      []int        // start index of all //go:build comments in output
	plusBuild    []int        // start index of all // +build comments in output

	// Positions
	// The out position differs from the pos position when the result
	// formatting differs from the source formatting (in the amount of
	// white space). If there's a difference and SourcePos is set in
	// ConfigMode, //line directives are used in the output to restore
	// original source positions for a reader.
	pos          token.Position // current position in AST (source) space
	out          token.Position // current position in output space
	last         token.Position // value of pos after calling writeString
	linePtr      *int           // if set, record out.Line for the next token in *linePtr
	sourcePosErr error          // if non-nil, the first error emitting a //line directive

	// The list of all source comments, in order of appearance.
	comments        []*ast.CommentGroup // may be nil
	useNodeComments bool                // if not set, ignore lead and line comments of nodes

	// Information about p.comments[p.cindex]; set up by nextComment.
	commentInfo

	// Cache of already computed node sizes.
	nodeSizes map[ast.Node]int

	// Cache of most recently computed line position.
	cachedPos  token.Pos
	cachedLine int // line corresponding to cachedPos
}

func (p *printer) internalError(msg ...any) {
	if debug {
		fmt.Print(p.pos.String() + ": ")
		fmt.Println(msg...)
		panic("go/printer")
	}
}

// commentsHaveNewline reports whether a list of comments belonging to
// an *ast.CommentGroup contains newlines. Because the position information
// may only be partially correct, we also have to read the comment text.
func (p *printer) commentsHaveNewline(list []*ast.Comment) bool {
	// len(list) > 0
	line := p.lineFor(list[0].Pos())
	for i, c := range list {
		if i > 0 && p.lineFor(list[i].Pos()) != line {
			// not all comments on the same line
			return true
		}
		if t := c.Text; len(t) >= 2 && (t[1] == '/' || strings.Contains(t, "\n")) {
			return true
		}
	}
	_ = line
	return false
}

func (p *printer) nextComment() {
	for p.cindex < len(p.comments) {
		c := p.comments[p.cindex]
		p.cindex++
		if list := c.List; len(list) > 0 {
			p.comment = c
			p.commentOffset = p.posFor(list[0].Pos()).Offset
			p.commentNewline = p.commentsHaveNewline(list)
			return
		}
		// we should not reach here (correct ASTs don't have empty
		// ast.CommentGroup nodes), but be conservative and try again
	}
	// no more comments
	p.commentOffset = infinity
}

// commentBefore reports whether the current comment group occurs
// before the next position in the source code and printing it does
// not introduce implicit semicolons.
func (p *printer) commentBefore(next token.Position) bool {
	return p.commentOffset < next.Offset && (!p.impliedSemi || !p.commentNewline)
}

// commentSizeBefore returns the estimated size of the
// comments on the same line before the next position.
func (p *printer) commentSizeBefore(next token.Position) int {
	// save/restore current p.commentInfo (p.nextComment() modifies it)
	defer func(info commentInfo) {
		p.commentInfo = info
	}(p.commentInfo)

	size := 0
	for p.commentBefore(next) {
		for _, c := range p.comment.List {
			size += len(c.Text)
		}
		p.nextComment()
	}
	return size
}

// recordLine records the output line number for the next non-whitespace
// token in *linePtr. It is used to compute an accurate line number for a
// formatted construct, independent of pending (not yet emitted) whitespace
// or comments.
func (p *printer) recordLine(linePtr *int) {
	p.linePtr = linePtr
}

// linesFrom returns the number of output lines between the current
// output line and the line argument, ignoring any pending (not yet
// emitted) whitespace or comments. It is used to compute an accurate
// size (in number of lines) for a formatted construct.
func (p *printer) linesFrom(line int) int {
	return p.out.Line - line
}

func (p *printer) posFor(pos token.Pos) token.Position {
	// not used frequently enough to cache entire token.Position
	return p.fset.PositionFor(pos, false /* absolute position */)
}

func (p *printer) lineFor(pos token.Pos) int {
	if pos != p.cachedPos {
		p.cachedPos = pos
		p.cachedLine = p.fset.PositionFor(pos, false /* absolute position */).Line
	}
	return p.cachedLine
}

// writeLineDirective writes a //line directive if necessary.
func (p *printer) writeLineDirective(pos token.Position) {
	if pos.IsValid() && (p.out.Line != pos.Line || p.out.Filename != pos.Filename) {
		if strings.ContainsAny(pos.Filename, "\r\n") {
			if p.sourcePosErr == nil {
				p.sourcePosErr = fmt.Errorf("go/printer: source filename contains unexpected newline character: %q", pos.Filename)
			}
			return
		}

		p.output = append(p.output, tabwriter.Escape) // protect '\n' in //line from tabwriter interpretation
		p.output = append(p.output, fmt.Sprintf("//line %s:%d\n", pos.Filename, pos.Line)...)
		p.output = append(p.output, tabwriter.Escape)
		// p.out must match the //line directive
		p.out.Filename = pos.Filename
		p.out.Line = pos.Line
	}
}

// writeIndent writes indentation.
func (p *printer) writeIndent() {
	// use "hard" htabs - indentation columns
	// must not be discarded by the tabwriter
	n := p.Config.Indent + p.indent // include base indentation
	for i := 0; i < n; i++ {
		p.output = append(p.output, '\t')
	}

	// update positions
	p.pos.Offset += n
	p.pos.Column += n
	p.out.Column += n
}

// writeByte writes ch n times to p.output and updates p.pos.
// Only used to write formatting (white space) characters.
func (p *printer) writeByte(ch byte, n int) {
	if p.endAlignment {
		// Ignore any alignment control character;
		// and at the end of the line, break with
		// a formfeed to indicate termination of
		// existing columns.
		switch ch {
		case '\t', '\v':
			ch = ' '
		case '\n', '\f':
			ch = '\f'
			p.endAlignment = false
		}
	}

	if p.out.Column == 1 {
		// no need to write line directives before white space
		p.writeIndent()
	}

	for i := 0; i < n; i++ {
		p.output = append(p.output, ch)
	}

	// update positions
	p.pos.Offset += n
	if ch == '\n' || ch == '\f' {
		p.pos.Line += n
		p.out.Line += n
		p.pos.Column = 1
		p.out.Column = 1
		return
	}
	p.pos.Column += n
	p.out.Column += n
}

// writeString writes the string s to p.output and updates p.pos, p.out,
// and p.last. If isLit is set, s is escaped w/ tabwriter.Escape characters
// to protect s from being interpreted by the tabwriter.
//
// Note: writeString is only used to write Go tokens, literals, and
// comments, all of which must be written literally. Thus, it is correct
// to always set isLit = true. However, setting it explicitly only when
// needed (i.e., when we don't know that s contains no tabs or line breaks)
// avoids processing extra escape characters and reduces run time of the
// printer benchmark by up to 10%.
func (p *printer) writeString(pos token.Position, s string, isLit bool) {
	if p.out.Column == 1 {
		if p.Config.Mode&SourcePos != 0 {
			p.writeLineDirective(pos)
		}
		p.writeIndent()
	}

	if pos.IsValid() {
		// update p.pos (if pos is invalid, continue with existing p.pos)
		// Note: Must do this after handling line beginnings because
		// writeIndent updates p.pos if there's indentation, but p.pos
		// is the position of s.
		p.pos = pos
	}

	if isLit {
		// Protect s such that is passes through the tabwriter
		// unchanged. Note that valid Go programs cannot contain
		// tabwriter.Escape bytes since they do not appear in legal
		// UTF-8 sequences.
		p.output = append(p.output, tabwriter.Escape)
	}

	if debug {
		p.output = append(p.output, fmt.Sprintf("/*%s*/", pos)...) // do not update p.pos!
	}
	p.output = append(p.output, s...)

	// update positions
	nlines := 0
	var li int // index of last newline; valid if nlines > 0
	for i := 0; i < len(s); i++ {
		// Raw string literals may contain any character except back quote (`).
		if ch := s[i]; ch == '\n' || ch == '\f' {
			// account for line break
			nlines++
			li = i
			// A line break inside a literal will break whatever column
			// formatting is in place; ignore any further alignment through
			// the end of the line.
			p.endAlignment = true
		}
	}
	p.pos.Offset += len(s)
	if nlines > 0 {
		p.pos.Line += nlines
		p.out.Line += nlines
		c := len(s) - li
		p.pos.Column = c
		p.out.Column = c
	} else {
		p.pos.Column += len(s)
		p.out.Column += len(s)
	}

	if isLit {
		p.output = append(p.output, tabwriter.Escape)
	}

	p.last = p.pos
}

// writeCommentPrefix writes the whitespace before a comment.
// If there is any pending whitespace, it consumes as much of
// it as is likely to help position the comment nicely.
// pos is the comment position, next the position of the item
// after all pending comments, prev is the previous comment in
// a group of comments (or nil), and tok is the next token.
func (p *printer) writeCommentPrefix(pos, next token.Position, prev *ast.Comment, tok token.Token) {
	if len(p.output) == 0 {
		// the comment is the first item to be printed - don't write any whitespace
		return
	}

	if pos.IsValid() && pos.Filename != p.last.Filename {
		// comment in a different file - separate with newlines
		p.writeByte('\f', maxNewlines)
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
			p.writeByte(sep, 1)
		}

	} else {
		// comment on a different line:
		// separate with at least one line break
		droppedLinebreak := false
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
				// if this is not the last unindent, apply it
				// as it is (likely) belonging to the last
				// construct (e.g., a multi-line expression list)
				// and is not part of closing a block
				if i+1 < len(p.wsbuf) && p.wsbuf[i+1] == unindent {
					continue
				}
				// if the next token is not a closing }, apply the unindent
				// if it appears that the comment is aligned with the
				// token; otherwise assume the unindent is part of a
				// closing block and stop (this scenario appears with
				// comments before a case label where the comments
				// apply to the next case instead of the current one)
				if tok != token.RBRACE && pos.Column == next.Column {
					continue
				}
			case newline, formfeed:
				p.wsbuf[i] = ignore
				droppedLinebreak = prev == nil // record only if first comment of a group
			}
			j = i
			break
		}
		p.writeWhitespace(j)

		// determine number of linebreaks before the comment
		n := 0
		if pos.IsValid() && p.last.IsValid() {
			n = pos.Line - p.last.Line
			if n < 0 { // should never happen
				n = 0
			}
		}

		// at the package scope level only (p.indent == 0),
		// add an extra newline if we dropped one before:
		// this preserves a blank line before documentation
		// comments at the package scope level (issue 2570)
		if p.indent == 0 && droppedLinebreak {
			n++
		}

		// make sure there is at least one line break
		// if the previous comment was a line comment
		if n == 0 && prev != nil && prev.Text[1] == '/' {
			n = 1
		}

		if n > 0 {
			// use formfeeds to break columns before a comment;
			// this is analogous to using formfeeds to separate
			// individual lines of /*-style comments
			p.writeByte('\f', nlimit(n))
		}
	}
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

// commonPrefix returns the common prefix of a and b.
func commonPrefix(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] && (a[i] <= ' ' || a[i] == '*') {
		i++
	}
	return a[0:i]
}

// trimRight returns s with trailing whitespace removed.
func trimRight(s string) string {
	return strings.TrimRightFunc(s, unicode.IsSpace)
}

// stripCommonPrefix removes a common prefix from /*-style comment lines (unless no
// comment line is indented, all but the first line have some form of space prefix).
// The prefix is computed using heuristics such that is likely that the comment
// contents are nicely laid out after re-printing each line using the printer's
// current indentation.
func stripCommonPrefix(lines []string) {
	if len(lines) <= 1 {
		return // at most one line - nothing to do
	}
	// len(lines) > 1

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
	// In cases where only the first and last lines are not blank,
	// such as two-line comments, or comments where all inner lines
	// are blank, consider the last line for the prefix computation
	// since otherwise the prefix would be empty.
	//
	// Note that the first and last line are never empty (they
	// contain the opening /* and closing */ respectively) and
	// thus they can be ignored by the blank line check.
	prefix := ""
	prefixSet := false
	if len(lines) > 2 {
		for i, line := range lines[1 : len(lines)-1] {
			if isBlank(line) {
				lines[1+i] = "" // range starts with lines[1]
			} else {
				if !prefixSet {
					prefix = line
					prefixSet = true
				}
				prefix = commonPrefix(prefix, line)
			}

		}
	}
	// If we don't have a prefix yet, consider the last line.
	if !prefixSet {
		line := lines[len(lines)-1]
		prefix = commonPrefix(line, line)
	}

	/*
	 * Check for vertical "line of stars" and correct prefix accordingly.
	 */
	lineOfStars := false
	if p, _, ok := strings.Cut(prefix, "*"); ok {
		// remove trailing blank from prefix so stars remain aligned
		prefix = strings.TrimSuffix(p, " ")
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
			prefix = strings.TrimSuffix(prefix, string(suffix))
		}
	}

	// Handle last line: If it only contains a closing */, align it
	// with the opening /*, otherwise align the text with the other
	// lines.
	last := lines[len(lines)-1]
	closing := "*/"
	before, _, _ := strings.Cut(last, closing) // closing always present
	if isBlank(before) {
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
	for i, line := range lines {
		if i > 0 && line != "" {
			lines[i] = line[len(prefix):]
		}
	}
}

func (p *printer) writeComment(comment *ast.Comment) {
	text := comment.Text
	pos := p.posFor(comment.Pos())

	const linePrefix = "//line "
	if strings.HasPrefix(text, linePrefix) && (!pos.IsValid() || pos.Column == 1) {
		// Possibly a //-style line directive.
		// Suspend indentation temporarily to keep line directive valid.
		defer func(indent int) { p.indent = indent }(p.indent)
		p.indent = 0
	}

	// shortcut common case of //-style comments
	if text[1] == '/' {
		if constraint.IsGoBuild(text) {
			p.goBuild = append(p.goBuild, len(p.output))
		} else if constraint.IsPlusBuild(text) {
			p.plusBuild = append(p.plusBuild, len(p.output))
		}
		p.writeString(pos, trimRight(text), true)
		return
	}

	// for /*-style comments, print line by line and let the
	// write function take care of the proper indentation
	lines := strings.Split(text, "\n")

	// The comment started in the first column but is going
	// to be indented. For an idempotent result, add indentation
	// to all lines such that they look like they were indented
	// before - this will make sure the common prefix computation
	// is the same independent of how many times formatting is
	// applied (was issue 1835).
	if pos.IsValid() && pos.Column == 1 && p.indent > 0 {
		for i, line := range lines[1:] {
			lines[1+i] = "   " + line
		}
	}

	stripCommonPrefix(lines)

	// write comment lines, separated by formfeed,
	// without a line break after the last line
	for i, line := range lines {
		if i > 0 {
			p.writeByte('\f', 1)
			pos = p.pos
		}
		if len(line) > 0 {
			p.writeString(pos, trimRight(line), true)
		}
	}
}

// writeCommentSuffix writes a line break after a comment if indicated
// and processes any leftover indentation information. If a line break
// is needed, the kind of break (newline vs formfeed) depends on the
// pending whitespace. The writeCommentSuffix result indicates if a
// newline was written or if a formfeed was dropped from the whitespace
// buffer.
func (p *printer) writeCommentSuffix(needsLinebreak bool) (wroteNewline, droppedFF bool) {
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
				wroteNewline = true
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
		p.writeByte('\n', 1)
		wroteNewline = true
	}

	return
}

// containsLinebreak reports whether the whitespace buffer contains any line breaks.
func (p *printer) containsLinebreak() bool {
	for _, ch := range p.wsbuf {
		if ch == newline || ch == formfeed {
			return true
		}
	}
	return false
}

// intersperseComments consumes all comments that appear before the next token
// tok and prints it together with the buffered whitespace (i.e., the whitespace
// that needs to be written before the next token). A heuristic is used to mix
// the comments and whitespace. The intersperseComments result indicates if a
// newline was written or if a formfeed was dropped from the whitespace buffer.
func (p *printer) intersperseComments(next token.Position, tok token.Token) (wroteNewline, droppedFF bool) {
	var last *ast.Comment
	for p.commentBefore(next) {
		list := p.comment.List
		changed := false
		if p.lastTok != token.IMPORT && // do not rewrite cgo's import "C" comments
			p.posFor(p.comment.Pos()).Column == 1 &&
			p.posFor(p.comment.End()+1) == next {
			// Unindented comment abutting next token position:
			// a top-level doc comment.
			list = formatDocComment(list)
			changed = true

			if len(p.comment.List) > 0 && len(list) == 0 {
				// The doc comment was removed entirely.
				// Keep preceding whitespace.
				p.writeCommentPrefix(p.posFor(p.comment.Pos()), next, last, tok)
				// Change print state to continue at next.
				p.pos = next
				p.last = next
				// There can't be any more comments.
				p.nextComment()
				return p.writeCommentSuffix(false)
			}
		}
		for _, c := range list {
			p.writeCommentPrefix(p.posFor(c.Pos()), next, last, tok)
			p.writeComment(c)
			last = c
		}
		// In case list was rewritten, change print state to where
		// the original list would have ended.
		if len(p.comment.List) > 0 && changed {
			last = p.comment.List[len(p.comment.List)-1]
			p.pos = p.posFor(last.End())
			p.last = p.pos
		}
		p.nextComment()
	}

	if last != nil {
		// If the last comment is a /*-style comment and the next item
		// follows on the same line but is not a comma, and not a "closing"
		// token immediately following its corresponding "opening" token,
		// add an extra separator unless explicitly disabled. Use a blank
		// as separator unless we have pending linebreaks, they are not
		// disabled, and we are outside a composite literal, in which case
		// we want a linebreak (issue 15137).
		// TODO(gri) This has become overly complicated. We should be able
		// to track whether we're inside an expression or statement and
		// use that information to decide more directly.
		needsLinebreak := false
		if p.mode&noExtraBlank == 0 &&
			last.Text[1] == '*' && p.lineFor(last.Pos()) == next.Line &&
			tok != token.COMMA &&
			(tok != token.RPAREN || p.prevOpen == token.LPAREN) &&
			(tok != token.RBRACK || p.prevOpen == token.LBRACK) {
			if p.containsLinebreak() && p.mode&noExtraLinebreak == 0 && p.level == 0 {
				needsLinebreak = true
			} else {
				p.writeByte(' ', 1)
			}
		}
		// Ensure that there is a line break after a //-style comment,
		// before EOF, and before a closing '}' unless explicitly disabled.
		if last.Text[1] == '/' ||
			tok == token.EOF ||
			tok == token.RBRACE && p.mode&noExtraLinebreak == 0 {
			needsLinebreak = true
		}
		return p.writeCommentSuffix(needsLinebreak)
	}

	// no comment was written - we should never reach here since
	// intersperseComments should not be called in that case
	p.internalError("intersperseComments called without pending comments")
	return
}

// writeWhitespace writes the first n whitespace entries.
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
			p.writeByte(byte(ch), 1)
		}
	}

	// shift remaining entries down
	l := copy(p.wsbuf, p.wsbuf[n:])
	p.wsbuf = p.wsbuf[:l]
}

// ----------------------------------------------------------------------------
// Printing interface

// nlimit limits n to maxNewlines.
func nlimit(n int) int {
	return min(n, maxNewlines)
}

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

func (p *printer) setPos(pos token.Pos) {
	if pos.IsValid() {
		p.pos = p.posFor(pos) // accurate position of next item
	}
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
func (p *printer) print(args ...any) {
	for _, arg := range args {
		// information about the current arg
		var data string
		var isLit bool
		var impliedSemi bool // value for p.impliedSemi after this arg

		// record previous opening token, if any
		switch p.lastTok {
		case token.ILLEGAL:
			// ignore (white space)
		case token.LPAREN, token.LBRACK:
			p.prevOpen = p.lastTok
		default:
			// other tokens followed any opening token
			p.prevOpen = token.ILLEGAL
		}

		switch x := arg.(type) {
		case pmode:
			// toggle printer mode
			p.mode ^= x
			continue

		case whiteSpace:
			if x == ignore {
				// don't add ignore's to the buffer; they
				// may screw up "correcting" unindents (see
				// LabeledStmt)
				continue
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
			if x == newline || x == formfeed {
				// newlines affect the current state (p.impliedSemi)
				// and not the state after printing arg (impliedSemi)
				// because comments can be interspersed before the arg
				// in this case
				p.impliedSemi = false
			}
			p.lastTok = token.ILLEGAL
			continue

		case *ast.Ident:
			data = x.Name
			impliedSemi = true
			p.lastTok = token.IDENT

		case *ast.BasicLit:
			data = x.Value
			isLit = true
			impliedSemi = true
			p.lastTok = x.Kind

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
			// some keywords followed by a newline imply a semicolon
			switch x {
			case token.BREAK, token.CONTINUE, token.FALLTHROUGH, token.RETURN,
				token.INC, token.DEC, token.RPAREN, token.RBRACK, token.RBRACE:
				impliedSemi = true
			}
			p.lastTok = x

		case string:
			// incorrect AST - print error message
			data = x
			isLit = true
			impliedSemi = true
			p.lastTok = token.STRING

		default:
			fmt.Fprintf(os.Stderr, "print: unsupported argument %v (%T)\n", arg, arg)
			panic("go/printer type")
		}
		// data != ""

		next := p.pos // estimated/accurate position of next item
		wroteNewline, droppedFF := p.flush(next, p.lastTok)

		// intersperse extra newlines if present in the source and
		// if they don't cause extra semicolons (don't do this in
		// flush as it will cause extra newlines at the end of a file)
		if !p.impliedSemi {
			n := nlimit(next.Line - p.pos.Line)
			// don't exceed maxNewlines if we already wrote one
			if wroteNewline && n == maxNewlines {
				n = maxNewlines - 1
			}
			if n > 0 {
				ch := byte('\n')
				if droppedFF {
					ch = '\f' // use formfeed since we dropped one before
				}
				p.writeByte(ch, n)
				impliedSemi = false
			}
		}

		// the next token starts now - record its line number if requested
		if p.linePtr != nil {
			*p.linePtr = p.out.Line
			p.linePtr = nil
		}

		p.writeString(next, data, isLit)
		p.impliedSemi = impliedSemi
	}
}

// flush prints any pending comments and whitespace occurring textually
// before the position of the next token tok. The flush result indicates
// if a newline was written or if a formfeed was dropped from the whitespace
// buffer.
func (p *printer) flush(next token.Position, tok token.Token) (wroteNewline, droppedFF bool) {
	if p.commentBefore(next) {
		// if there are comments before the next item, intersperse them
		wroteNewline, droppedFF = p.intersperseComments(next, tok)
	} else {
		// otherwise, write any leftover whitespace
		p.writeWhitespace(len(p.wsbuf))
	}
	return
}

// getDoc returns the ast.CommentGroup associated with n, if any.
func getDoc(n ast.Node) *ast.CommentGroup {
	switch n := n.(type) {
	case *ast.Field:
		return n.Doc
	case *ast.ImportSpec:
		return n.Doc
	case *ast.ValueSpec:
		return n.Doc
	case *ast.TypeSpec:
		return n.Doc
	case *ast.GenDecl:
		return n.Doc
	case *ast.FuncDecl:
		return n.Doc
	case *ast.File:
		return n.Doc
	}
	return nil
}

func getLastComment(n ast.Node) *ast.CommentGroup {
	switch n := n.(type) {
	case *ast.Field:
		return n.Comment
	case *ast.ImportSpec:
		return n.Comment
	case *ast.ValueSpec:
		return n.Comment
	case *ast.TypeSpec:
		return n.Comment
	case *ast.GenDecl:
		if len(n.Specs) > 0 {
			return getLastComment(n.Specs[len(n.Specs)-1])
		}
	case *ast.File:
		if len(n.Comments) > 0 {
			return n.Comments[len(n.Comments)-1]
		}
	}
	return nil
}

func (p *printer) printNode(node any) error {
	// unpack *CommentedNode, if any
	var comments []*ast.CommentGroup
	if cnode, ok := node.(*CommentedNode); ok {
		node = cnode.Node
		comments = cnode.Comments
	}

	if comments != nil {
		// commented node - restrict comment list to relevant range
		n, ok := node.(ast.Node)
		if !ok {
			goto unsupported
		}
		beg := n.Pos()
		end := n.End()
		// if the node has associated documentation,
		// include that commentgroup in the range
		// (the comment list is sorted in the order
		// of the comment appearance in the source code)
		if doc := getDoc(n); doc != nil {
			beg = doc.Pos()
		}
		if com := getLastComment(n); com != nil {
			if e := com.End(); e > end {
				end = e
			}
		}
		// token.Pos values are global offsets, we can
		// compare them directly
		i := 0
		for i < len(comments) && comments[i].End() < beg {
			i++
		}
		j := i
		for j < len(comments) && comments[j].Pos() < end {
			j++
		}
		if i < j {
			p.comments = comments[i:j]
		}
	} else if n, ok := node.(*ast.File); ok {
		// use ast.File comments, if any
		p.comments = n.Comments
	}

	// if there are no comments, use node comments
	p.useNodeComments = p.comments == nil

	// get comments ready for use
	p.nextComment()

	p.print(pmode(0))

	// format node
	switch n := node.(type) {
	case ast.Expr:
		p.expr(n)
	case ast.Stmt:
		// A labeled statement will un-indent to position the label.
		// Set p.indent to 1 so we don't get indent "underflow".
		if _, ok := n.(*ast.LabeledStmt); ok {
			p.indent = 1
		}
		p.stmt(n, false)
	case ast.Decl:
		p.decl(n)
	case ast.Spec:
		p.spec(n, 1, false)
	case []ast.Stmt:
		// A labeled statement will un-indent to position the label.
		// Set p.indent to 1 so we don't get indent "underflow".
		for _, s := range n {
			if _, ok := s.(*ast.LabeledStmt); ok {
				p.indent = 1
			}
		}
		p.stmtList(n, 0, false)
	case []ast.Decl:
		p.declList(n)
	case *ast.File:
		p.file(n)
	default:
		goto unsupported
	}

	return p.sourcePosErr

unsupported:
	return fmt.Errorf("go/printer: unsupported node type %T", node)
}

// ----------------------------------------------------------------------------
// Trimmer

// A trimmer is an io.Writer filter for stripping tabwriter.Escape
// characters, trailing blanks and tabs, and for converting formfeed
// and vtab characters into newlines and htabs (in case no tabwriter
// is used). Text bracketed by tabwriter.Escape characters is passed
// through unchanged.
type trimmer struct {
	output io.Writer
	state  int
	space  []byte
}

// trimmer is implemented as a state machine.
// It can be in one of the following states:
const (
	inSpace  = iota // inside space
	inEscape        // inside text bracketed by tabwriter.Escapes
	inText          // inside text
)

func (p *trimmer) resetSpace() {
	p.state = inSpace
	p.space = p.space[0:0]
}

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
				p.space = append(p.space, b)
			case '\n', '\f':
				p.resetSpace() // discard trailing space
				_, err = p.output.Write(aNewline)
			case tabwriter.Escape:
				_, err = p.output.Write(p.space)
				p.state = inEscape
				m = n + 1 // +1: skip tabwriter.Escape
			default:
				_, err = p.output.Write(p.space)
				p.state = inText
				m = n
			}
		case inEscape:
			if b == tabwriter.Escape {
				_, err = p.output.Write(data[m:n])
				p.resetSpace()
			}
		case inText:
			switch b {
			case '\t', ' ':
				_, err = p.output.Write(data[m:n])
				p.resetSpace()
				p.space = append(p.space, b)
			case '\n', '\f':
				_, err = p.output.Write(data[m:n])
				p.resetSpace()
				if err == nil {
					_, err = p.output.Write(aNewline)
				}
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
		p.resetSpace()
	}

	return
}

// ----------------------------------------------------------------------------
// Public interface

// A Mode value is a set of flags (or 0). They control printing.
type Mode uint

const (
	RawFormat Mode = 1 << iota // do not use a tabwriter; if set, UseSpaces is ignored
	TabIndent                  // use tabs for indentation independent of UseSpaces
	UseSpaces                  // use spaces instead of tabs for alignment
	SourcePos                  // emit //line directives to preserve original source positions
)

// The mode below is not included in printer's public API because
// editing code text is deemed out of scope. Because this mode is
// unexported, it's also possible to modify or remove it based on
// the evolving needs of go/format and cmd/gofmt without breaking
// users. See discussion in CL 240683.
const (
	// normalizeNumbers means to canonicalize number
	// literal prefixes and exponents while printing.
	//
	// This value is known in and used by go/format and cmd/gofmt.
	// It is currently more convenient and performant for those
	// packages to apply number normalization during printing,
	// rather than by modifying the AST in advance.
	normalizeNumbers Mode = 1 << 30
)

// A Config node controls the output of Fprint.
type Config struct {
	Mode     Mode // default: 0
	Tabwidth int  // default: 8
	Indent   int  // default: 0 (all code is indented at least by this much)
}

var printerPool = sync.Pool{
	New: func() any {
		return &printer{
			// Whitespace sequences are short.
			wsbuf: make([]whiteSpace, 0, 16),
			// We start the printer with a 16K output buffer, which is currently
			// larger than about 80% of Go files in the standard library.
			output: make([]byte, 0, 16<<10),
		}
	},
}

func newPrinter(cfg *Config, fset *token.FileSet, nodeSizes map[ast.Node]int) *printer {
	p := printerPool.Get().(*printer)
	*p = printer{
		Config:    *cfg,
		fset:      fset,
		pos:       token.Position{Line: 1, Column: 1},
		out:       token.Position{Line: 1, Column: 1},
		wsbuf:     p.wsbuf[:0],
		nodeSizes: nodeSizes,
		cachedPos: -1,
		output:    p.output[:0],
	}
	return p
}

func (p *printer) free() {
	// Hard limit on buffer size; see https://golang.org/issue/23199.
	if cap(p.output) > 64<<10 {
		return
	}

	printerPool.Put(p)
}

// fprint implements Fprint and takes a nodesSizes map for setting up the printer state.
func (cfg *Config) fprint(output io.Writer, fset *token.FileSet, node any, nodeSizes map[ast.Node]int) (err error) {
	// print node
	p := newPrinter(cfg, fset, nodeSizes)
	defer p.free()
	if err = p.printNode(node); err != nil {
		return
	}
	// print outstanding comments
	p.impliedSemi = false // EOF acts like a newline
	p.flush(token.Position{Offset: infinity, Line: infinity}, token.EOF)

	// output is buffered in p.output now.
	// fix //go:build and // +build comments if needed.
	p.fixGoBuildLines()

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
	if _, err = output.Write(p.output); err != nil {
		return
	}

	// flush tabwriter, if any
	if tw, _ := output.(*tabwriter.Writer); tw != nil {
		err = tw.Flush()
	}

	return
}

// A CommentedNode bundles an AST node and corresponding comments.
// It may be provided as argument to any of the [Fprint] functions.
type CommentedNode struct {
	Node     any // *ast.File, or ast.Expr, ast.Decl, ast.Spec, or ast.Stmt
	Comments []*ast.CommentGroup
}

// Fprint "pretty-prints" an AST node to output for a given configuration cfg.
// Position information is interpreted relative to the file set fset.
// The node type must be *[ast.File], *[CommentedNode], [][ast.Decl], [][ast.Stmt],
// or assignment-compatible to [ast.Expr], [ast.Decl], [ast.Spec], or [ast.Stmt].
func (cfg *Config) Fprint(output io.Writer, fset *token.FileSet, node any) error {
	return cfg.fprint(output, fset, node, make(map[ast.Node]int))
}

// Fprint "pretty-prints" an AST node to output.
// It calls [Config.Fprint] with default settings.
// Note that gofmt uses tabs for indentation but spaces for alignment;
// use format.Node (package go/format) for output that matches gofmt.
func Fprint(output io.Writer, fset *token.FileSet, node any) error {
	return (&Config{Tabwidth: 8}).Fprint(output, fset, node)
}
