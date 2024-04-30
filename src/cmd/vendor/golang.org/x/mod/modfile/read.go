// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// A Position describes an arbitrary source position in a file, including the
// file, line, column, and byte offset.
type Position struct {
	Line     int // line in input (starting at 1)
	LineRune int // rune in line (starting at 1)
	Byte     int // byte in input (starting at 0)
}

// add returns the position at the end of s, assuming it starts at p.
func (p Position) add(s string) Position {
	p.Byte += len(s)
	if n := strings.Count(s, "\n"); n > 0 {
		p.Line += n
		s = s[strings.LastIndex(s, "\n")+1:]
		p.LineRune = 1
	}
	p.LineRune += utf8.RuneCountInString(s)
	return p
}

// An Expr represents an input element.
type Expr interface {
	// Span returns the start and end position of the expression,
	// excluding leading or trailing comments.
	Span() (start, end Position)

	// Comment returns the comments attached to the expression.
	// This method would normally be named 'Comments' but that
	// would interfere with embedding a type of the same name.
	Comment() *Comments
}

// A Comment represents a single // comment.
type Comment struct {
	Start  Position
	Token  string // without trailing newline
	Suffix bool   // an end of line (not whole line) comment
}

// Comments collects the comments associated with an expression.
type Comments struct {
	Before []Comment // whole-line comments before this expression
	Suffix []Comment // end-of-line comments after this expression

	// For top-level expressions only, After lists whole-line
	// comments following the expression.
	After []Comment
}

// Comment returns the receiver. This isn't useful by itself, but
// a [Comments] struct is embedded into all the expression
// implementation types, and this gives each of those a Comment
// method to satisfy the Expr interface.
func (c *Comments) Comment() *Comments {
	return c
}

// A FileSyntax represents an entire go.mod file.
type FileSyntax struct {
	Name string // file path
	Comments
	Stmt []Expr
}

func (x *FileSyntax) Span() (start, end Position) {
	if len(x.Stmt) == 0 {
		return
	}
	start, _ = x.Stmt[0].Span()
	_, end = x.Stmt[len(x.Stmt)-1].Span()
	return start, end
}

// addLine adds a line containing the given tokens to the file.
//
// If the first token of the hint matches the first token of the
// line, the new line is added at the end of the block containing hint,
// extracting hint into a new block if it is not yet in one.
//
// If the hint is non-nil buts its first token does not match,
// the new line is added after the block containing hint
// (or hint itself, if not in a block).
//
// If no hint is provided, addLine appends the line to the end of
// the last block with a matching first token,
// or to the end of the file if no such block exists.
func (x *FileSyntax) addLine(hint Expr, tokens ...string) *Line {
	if hint == nil {
		// If no hint given, add to the last statement of the given type.
	Loop:
		for i := len(x.Stmt) - 1; i >= 0; i-- {
			stmt := x.Stmt[i]
			switch stmt := stmt.(type) {
			case *Line:
				if stmt.Token != nil && stmt.Token[0] == tokens[0] {
					hint = stmt
					break Loop
				}
			case *LineBlock:
				if stmt.Token[0] == tokens[0] {
					hint = stmt
					break Loop
				}
			}
		}
	}

	newLineAfter := func(i int) *Line {
		new := &Line{Token: tokens}
		if i == len(x.Stmt) {
			x.Stmt = append(x.Stmt, new)
		} else {
			x.Stmt = append(x.Stmt, nil)
			copy(x.Stmt[i+2:], x.Stmt[i+1:])
			x.Stmt[i+1] = new
		}
		return new
	}

	if hint != nil {
		for i, stmt := range x.Stmt {
			switch stmt := stmt.(type) {
			case *Line:
				if stmt == hint {
					if stmt.Token == nil || stmt.Token[0] != tokens[0] {
						return newLineAfter(i)
					}

					// Convert line to line block.
					stmt.InBlock = true
					block := &LineBlock{Token: stmt.Token[:1], Line: []*Line{stmt}}
					stmt.Token = stmt.Token[1:]
					x.Stmt[i] = block
					new := &Line{Token: tokens[1:], InBlock: true}
					block.Line = append(block.Line, new)
					return new
				}

			case *LineBlock:
				if stmt == hint {
					if stmt.Token[0] != tokens[0] {
						return newLineAfter(i)
					}

					new := &Line{Token: tokens[1:], InBlock: true}
					stmt.Line = append(stmt.Line, new)
					return new
				}

				for j, line := range stmt.Line {
					if line == hint {
						if stmt.Token[0] != tokens[0] {
							return newLineAfter(i)
						}

						// Add new line after hint within the block.
						stmt.Line = append(stmt.Line, nil)
						copy(stmt.Line[j+2:], stmt.Line[j+1:])
						new := &Line{Token: tokens[1:], InBlock: true}
						stmt.Line[j+1] = new
						return new
					}
				}
			}
		}
	}

	new := &Line{Token: tokens}
	x.Stmt = append(x.Stmt, new)
	return new
}

func (x *FileSyntax) updateLine(line *Line, tokens ...string) {
	if line.InBlock {
		tokens = tokens[1:]
	}
	line.Token = tokens
}

// markRemoved modifies line so that it (and its end-of-line comment, if any)
// will be dropped by (*FileSyntax).Cleanup.
func (line *Line) markRemoved() {
	line.Token = nil
	line.Comments.Suffix = nil
}

// Cleanup cleans up the file syntax x after any edit operations.
// To avoid quadratic behavior, (*Line).markRemoved marks the line as dead
// by setting line.Token = nil but does not remove it from the slice
// in which it appears. After edits have all been indicated,
// calling Cleanup cleans out the dead lines.
func (x *FileSyntax) Cleanup() {
	w := 0
	for _, stmt := range x.Stmt {
		switch stmt := stmt.(type) {
		case *Line:
			if stmt.Token == nil {
				continue
			}
		case *LineBlock:
			ww := 0
			for _, line := range stmt.Line {
				if line.Token != nil {
					stmt.Line[ww] = line
					ww++
				}
			}
			if ww == 0 {
				continue
			}
			if ww == 1 && len(stmt.RParen.Comments.Before) == 0 {
				// Collapse block into single line.
				line := &Line{
					Comments: Comments{
						Before: commentsAdd(stmt.Before, stmt.Line[0].Before),
						Suffix: commentsAdd(stmt.Line[0].Suffix, stmt.Suffix),
						After:  commentsAdd(stmt.Line[0].After, stmt.After),
					},
					Token: stringsAdd(stmt.Token, stmt.Line[0].Token),
				}
				x.Stmt[w] = line
				w++
				continue
			}
			stmt.Line = stmt.Line[:ww]
		}
		x.Stmt[w] = stmt
		w++
	}
	x.Stmt = x.Stmt[:w]
}

func commentsAdd(x, y []Comment) []Comment {
	return append(x[:len(x):len(x)], y...)
}

func stringsAdd(x, y []string) []string {
	return append(x[:len(x):len(x)], y...)
}

// A CommentBlock represents a top-level block of comments separate
// from any rule.
type CommentBlock struct {
	Comments
	Start Position
}

func (x *CommentBlock) Span() (start, end Position) {
	return x.Start, x.Start
}

// A Line is a single line of tokens.
type Line struct {
	Comments
	Start   Position
	Token   []string
	InBlock bool
	End     Position
}

func (x *Line) Span() (start, end Position) {
	return x.Start, x.End
}

// A LineBlock is a factored block of lines, like
//
//	require (
//		"x"
//		"y"
//	)
type LineBlock struct {
	Comments
	Start  Position
	LParen LParen
	Token  []string
	Line   []*Line
	RParen RParen
}

func (x *LineBlock) Span() (start, end Position) {
	return x.Start, x.RParen.Pos.add(")")
}

// An LParen represents the beginning of a parenthesized line block.
// It is a place to store suffix comments.
type LParen struct {
	Comments
	Pos Position
}

func (x *LParen) Span() (start, end Position) {
	return x.Pos, x.Pos.add(")")
}

// An RParen represents the end of a parenthesized line block.
// It is a place to store whole-line (before) comments.
type RParen struct {
	Comments
	Pos Position
}

func (x *RParen) Span() (start, end Position) {
	return x.Pos, x.Pos.add(")")
}

// An input represents a single input file being parsed.
type input struct {
	// Lexing state.
	filename   string    // name of input file, for errors
	complete   []byte    // entire input
	remaining  []byte    // remaining input
	tokenStart []byte    // token being scanned to end of input
	token      token     // next token to be returned by lex, peek
	pos        Position  // current input position
	comments   []Comment // accumulated comments

	// Parser state.
	file        *FileSyntax // returned top-level syntax tree
	parseErrors ErrorList   // errors encountered during parsing

	// Comment assignment state.
	pre  []Expr // all expressions, in preorder traversal
	post []Expr // all expressions, in postorder traversal
}

func newInput(filename string, data []byte) *input {
	return &input{
		filename:  filename,
		complete:  data,
		remaining: data,
		pos:       Position{Line: 1, LineRune: 1, Byte: 0},
	}
}

// parse parses the input file.
func parse(file string, data []byte) (f *FileSyntax, err error) {
	// The parser panics for both routine errors like syntax errors
	// and for programmer bugs like array index errors.
	// Turn both into error returns. Catching bug panics is
	// especially important when processing many files.
	in := newInput(file, data)
	defer func() {
		if e := recover(); e != nil && e != &in.parseErrors {
			in.parseErrors = append(in.parseErrors, Error{
				Filename: in.filename,
				Pos:      in.pos,
				Err:      fmt.Errorf("internal error: %v", e),
			})
		}
		if err == nil && len(in.parseErrors) > 0 {
			err = in.parseErrors
		}
	}()

	// Prime the lexer by reading in the first token. It will be available
	// in the next peek() or lex() call.
	in.readToken()

	// Invoke the parser.
	in.parseFile()
	if len(in.parseErrors) > 0 {
		return nil, in.parseErrors
	}
	in.file.Name = in.filename

	// Assign comments to nearby syntax.
	in.assignComments()

	return in.file, nil
}

// Error is called to report an error.
// Error does not return: it panics.
func (in *input) Error(s string) {
	in.parseErrors = append(in.parseErrors, Error{
		Filename: in.filename,
		Pos:      in.pos,
		Err:      errors.New(s),
	})
	panic(&in.parseErrors)
}

// eof reports whether the input has reached end of file.
func (in *input) eof() bool {
	return len(in.remaining) == 0
}

// peekRune returns the next rune in the input without consuming it.
func (in *input) peekRune() int {
	if len(in.remaining) == 0 {
		return 0
	}
	r, _ := utf8.DecodeRune(in.remaining)
	return int(r)
}

// peekPrefix reports whether the remaining input begins with the given prefix.
func (in *input) peekPrefix(prefix string) bool {
	// This is like bytes.HasPrefix(in.remaining, []byte(prefix))
	// but without the allocation of the []byte copy of prefix.
	for i := 0; i < len(prefix); i++ {
		if i >= len(in.remaining) || in.remaining[i] != prefix[i] {
			return false
		}
	}
	return true
}

// readRune consumes and returns the next rune in the input.
func (in *input) readRune() int {
	if len(in.remaining) == 0 {
		in.Error("internal lexer error: readRune at EOF")
	}
	r, size := utf8.DecodeRune(in.remaining)
	in.remaining = in.remaining[size:]
	if r == '\n' {
		in.pos.Line++
		in.pos.LineRune = 1
	} else {
		in.pos.LineRune++
	}
	in.pos.Byte += size
	return int(r)
}

type token struct {
	kind   tokenKind
	pos    Position
	endPos Position
	text   string
}

type tokenKind int

const (
	_EOF tokenKind = -(iota + 1)
	_EOLCOMMENT
	_IDENT
	_STRING
	_COMMENT

	// newlines and punctuation tokens are allowed as ASCII codes.
)

func (k tokenKind) isComment() bool {
	return k == _COMMENT || k == _EOLCOMMENT
}

// isEOL returns whether a token terminates a line.
func (k tokenKind) isEOL() bool {
	return k == _EOF || k == _EOLCOMMENT || k == '\n'
}

// startToken marks the beginning of the next input token.
// It must be followed by a call to endToken, once the token's text has
// been consumed using readRune.
func (in *input) startToken() {
	in.tokenStart = in.remaining
	in.token.text = ""
	in.token.pos = in.pos
}

// endToken marks the end of an input token.
// It records the actual token string in tok.text.
// A single trailing newline (LF or CRLF) will be removed from comment tokens.
func (in *input) endToken(kind tokenKind) {
	in.token.kind = kind
	text := string(in.tokenStart[:len(in.tokenStart)-len(in.remaining)])
	if kind.isComment() {
		if strings.HasSuffix(text, "\r\n") {
			text = text[:len(text)-2]
		} else {
			text = strings.TrimSuffix(text, "\n")
		}
	}
	in.token.text = text
	in.token.endPos = in.pos
}

// peek returns the kind of the next token returned by lex.
func (in *input) peek() tokenKind {
	return in.token.kind
}

// lex is called from the parser to obtain the next input token.
func (in *input) lex() token {
	tok := in.token
	in.readToken()
	return tok
}

// readToken lexes the next token from the text and stores it in in.token.
func (in *input) readToken() {
	// Skip past spaces, stopping at non-space or EOF.
	for !in.eof() {
		c := in.peekRune()
		if c == ' ' || c == '\t' || c == '\r' {
			in.readRune()
			continue
		}

		// Comment runs to end of line.
		if in.peekPrefix("//") {
			in.startToken()

			// Is this comment the only thing on its line?
			// Find the last \n before this // and see if it's all
			// spaces from there to here.
			i := bytes.LastIndex(in.complete[:in.pos.Byte], []byte("\n"))
			suffix := len(bytes.TrimSpace(in.complete[i+1:in.pos.Byte])) > 0
			in.readRune()
			in.readRune()

			// Consume comment.
			for len(in.remaining) > 0 && in.readRune() != '\n' {
			}

			// If we are at top level (not in a statement), hand the comment to
			// the parser as a _COMMENT token. The grammar is written
			// to handle top-level comments itself.
			if !suffix {
				in.endToken(_COMMENT)
				return
			}

			// Otherwise, save comment for later attachment to syntax tree.
			in.endToken(_EOLCOMMENT)
			in.comments = append(in.comments, Comment{in.token.pos, in.token.text, suffix})
			return
		}

		if in.peekPrefix("/*") {
			in.Error("mod files must use // comments (not /* */ comments)")
		}

		// Found non-space non-comment.
		break
	}

	// Found the beginning of the next token.
	in.startToken()

	// End of file.
	if in.eof() {
		in.endToken(_EOF)
		return
	}

	// Punctuation tokens.
	switch c := in.peekRune(); c {
	case '\n', '(', ')', '[', ']', '{', '}', ',':
		in.readRune()
		in.endToken(tokenKind(c))
		return

	case '"', '`': // quoted string
		quote := c
		in.readRune()
		for {
			if in.eof() {
				in.pos = in.token.pos
				in.Error("unexpected EOF in string")
			}
			if in.peekRune() == '\n' {
				in.Error("unexpected newline in string")
			}
			c := in.readRune()
			if c == quote {
				break
			}
			if c == '\\' && quote != '`' {
				if in.eof() {
					in.pos = in.token.pos
					in.Error("unexpected EOF in string")
				}
				in.readRune()
			}
		}
		in.endToken(_STRING)
		return
	}

	// Checked all punctuation. Must be identifier token.
	if c := in.peekRune(); !isIdent(c) {
		in.Error(fmt.Sprintf("unexpected input character %#q", c))
	}

	// Scan over identifier.
	for isIdent(in.peekRune()) {
		if in.peekPrefix("//") {
			break
		}
		if in.peekPrefix("/*") {
			in.Error("mod files must use // comments (not /* */ comments)")
		}
		in.readRune()
	}
	in.endToken(_IDENT)
}

// isIdent reports whether c is an identifier rune.
// We treat most printable runes as identifier runes, except for a handful of
// ASCII punctuation characters.
func isIdent(c int) bool {
	switch r := rune(c); r {
	case ' ', '(', ')', '[', ']', '{', '}', ',':
		return false
	default:
		return !unicode.IsSpace(r) && unicode.IsPrint(r)
	}
}

// Comment assignment.
// We build two lists of all subexpressions, preorder and postorder.
// The preorder list is ordered by start location, with outer expressions first.
// The postorder list is ordered by end location, with outer expressions last.
// We use the preorder list to assign each whole-line comment to the syntax
// immediately following it, and we use the postorder list to assign each
// end-of-line comment to the syntax immediately preceding it.

// order walks the expression adding it and its subexpressions to the
// preorder and postorder lists.
func (in *input) order(x Expr) {
	if x != nil {
		in.pre = append(in.pre, x)
	}
	switch x := x.(type) {
	default:
		panic(fmt.Errorf("order: unexpected type %T", x))
	case nil:
		// nothing
	case *LParen, *RParen:
		// nothing
	case *CommentBlock:
		// nothing
	case *Line:
		// nothing
	case *FileSyntax:
		for _, stmt := range x.Stmt {
			in.order(stmt)
		}
	case *LineBlock:
		in.order(&x.LParen)
		for _, l := range x.Line {
			in.order(l)
		}
		in.order(&x.RParen)
	}
	if x != nil {
		in.post = append(in.post, x)
	}
}

// assignComments attaches comments to nearby syntax.
func (in *input) assignComments() {
	const debug = false

	// Generate preorder and postorder lists.
	in.order(in.file)

	// Split into whole-line comments and suffix comments.
	var line, suffix []Comment
	for _, com := range in.comments {
		if com.Suffix {
			suffix = append(suffix, com)
		} else {
			line = append(line, com)
		}
	}

	if debug {
		for _, c := range line {
			fmt.Fprintf(os.Stderr, "LINE %q :%d:%d #%d\n", c.Token, c.Start.Line, c.Start.LineRune, c.Start.Byte)
		}
	}

	// Assign line comments to syntax immediately following.
	for _, x := range in.pre {
		start, _ := x.Span()
		if debug {
			fmt.Fprintf(os.Stderr, "pre %T :%d:%d #%d\n", x, start.Line, start.LineRune, start.Byte)
		}
		xcom := x.Comment()
		for len(line) > 0 && start.Byte >= line[0].Start.Byte {
			if debug {
				fmt.Fprintf(os.Stderr, "ASSIGN LINE %q #%d\n", line[0].Token, line[0].Start.Byte)
			}
			xcom.Before = append(xcom.Before, line[0])
			line = line[1:]
		}
	}

	// Remaining line comments go at end of file.
	in.file.After = append(in.file.After, line...)

	if debug {
		for _, c := range suffix {
			fmt.Fprintf(os.Stderr, "SUFFIX %q :%d:%d #%d\n", c.Token, c.Start.Line, c.Start.LineRune, c.Start.Byte)
		}
	}

	// Assign suffix comments to syntax immediately before.
	for i := len(in.post) - 1; i >= 0; i-- {
		x := in.post[i]

		start, end := x.Span()
		if debug {
			fmt.Fprintf(os.Stderr, "post %T :%d:%d #%d :%d:%d #%d\n", x, start.Line, start.LineRune, start.Byte, end.Line, end.LineRune, end.Byte)
		}

		// Do not assign suffix comments to end of line block or whole file.
		// Instead assign them to the last element inside.
		switch x.(type) {
		case *FileSyntax:
			continue
		}

		// Do not assign suffix comments to something that starts
		// on an earlier line, so that in
		//
		//	x ( y
		//		z ) // comment
		//
		// we assign the comment to z and not to x ( ... ).
		if start.Line != end.Line {
			continue
		}
		xcom := x.Comment()
		for len(suffix) > 0 && end.Byte <= suffix[len(suffix)-1].Start.Byte {
			if debug {
				fmt.Fprintf(os.Stderr, "ASSIGN SUFFIX %q #%d\n", suffix[len(suffix)-1].Token, suffix[len(suffix)-1].Start.Byte)
			}
			xcom.Suffix = append(xcom.Suffix, suffix[len(suffix)-1])
			suffix = suffix[:len(suffix)-1]
		}
	}

	// We assigned suffix comments in reverse.
	// If multiple suffix comments were appended to the same
	// expression node, they are now in reverse. Fix that.
	for _, x := range in.post {
		reverseComments(x.Comment().Suffix)
	}

	// Remaining suffix comments go at beginning of file.
	in.file.Before = append(in.file.Before, suffix...)
}

// reverseComments reverses the []Comment list.
func reverseComments(list []Comment) {
	for i, j := 0, len(list)-1; i < j; i, j = i+1, j-1 {
		list[i], list[j] = list[j], list[i]
	}
}

func (in *input) parseFile() {
	in.file = new(FileSyntax)
	var cb *CommentBlock
	for {
		switch in.peek() {
		case '\n':
			in.lex()
			if cb != nil {
				in.file.Stmt = append(in.file.Stmt, cb)
				cb = nil
			}
		case _COMMENT:
			tok := in.lex()
			if cb == nil {
				cb = &CommentBlock{Start: tok.pos}
			}
			com := cb.Comment()
			com.Before = append(com.Before, Comment{Start: tok.pos, Token: tok.text})
		case _EOF:
			if cb != nil {
				in.file.Stmt = append(in.file.Stmt, cb)
			}
			return
		default:
			in.parseStmt()
			if cb != nil {
				in.file.Stmt[len(in.file.Stmt)-1].Comment().Before = cb.Before
				cb = nil
			}
		}
	}
}

func (in *input) parseStmt() {
	tok := in.lex()
	start := tok.pos
	end := tok.endPos
	tokens := []string{tok.text}
	for {
		tok := in.lex()
		switch {
		case tok.kind.isEOL():
			in.file.Stmt = append(in.file.Stmt, &Line{
				Start: start,
				Token: tokens,
				End:   end,
			})
			return

		case tok.kind == '(':
			if next := in.peek(); next.isEOL() {
				// Start of block: no more tokens on this line.
				in.file.Stmt = append(in.file.Stmt, in.parseLineBlock(start, tokens, tok))
				return
			} else if next == ')' {
				rparen := in.lex()
				if in.peek().isEOL() {
					// Empty block.
					in.lex()
					in.file.Stmt = append(in.file.Stmt, &LineBlock{
						Start:  start,
						Token:  tokens,
						LParen: LParen{Pos: tok.pos},
						RParen: RParen{Pos: rparen.pos},
					})
					return
				}
				// '( )' in the middle of the line, not a block.
				tokens = append(tokens, tok.text, rparen.text)
			} else {
				// '(' in the middle of the line, not a block.
				tokens = append(tokens, tok.text)
			}

		default:
			tokens = append(tokens, tok.text)
			end = tok.endPos
		}
	}
}

func (in *input) parseLineBlock(start Position, token []string, lparen token) *LineBlock {
	x := &LineBlock{
		Start:  start,
		Token:  token,
		LParen: LParen{Pos: lparen.pos},
	}
	var comments []Comment
	for {
		switch in.peek() {
		case _EOLCOMMENT:
			// Suffix comment, will be attached later by assignComments.
			in.lex()
		case '\n':
			// Blank line. Add an empty comment to preserve it.
			in.lex()
			if len(comments) == 0 && len(x.Line) > 0 || len(comments) > 0 && comments[len(comments)-1].Token != "" {
				comments = append(comments, Comment{})
			}
		case _COMMENT:
			tok := in.lex()
			comments = append(comments, Comment{Start: tok.pos, Token: tok.text})
		case _EOF:
			in.Error(fmt.Sprintf("syntax error (unterminated block started at %s:%d:%d)", in.filename, x.Start.Line, x.Start.LineRune))
		case ')':
			rparen := in.lex()
			x.RParen.Before = comments
			x.RParen.Pos = rparen.pos
			if !in.peek().isEOL() {
				in.Error("syntax error (expected newline after closing paren)")
			}
			in.lex()
			return x
		default:
			l := in.parseLine()
			x.Line = append(x.Line, l)
			l.Comment().Before = comments
			comments = nil
		}
	}
}

func (in *input) parseLine() *Line {
	tok := in.lex()
	if tok.kind.isEOL() {
		in.Error("internal parse error: parseLine at end of line")
	}
	start := tok.pos
	end := tok.endPos
	tokens := []string{tok.text}
	for {
		tok := in.lex()
		if tok.kind.isEOL() {
			return &Line{
				Start:   start,
				Token:   tokens,
				End:     end,
				InBlock: true,
			}
		}
		tokens = append(tokens, tok.text)
		end = tok.endPos
	}
}

var (
	slashSlash = []byte("//")
	moduleStr  = []byte("module")
)

// ModulePath returns the module path from the gomod file text.
// If it cannot find a module path, it returns an empty string.
// It is tolerant of unrelated problems in the go.mod file.
func ModulePath(mod []byte) string {
	for len(mod) > 0 {
		line := mod
		mod = nil
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, mod = line[:i], line[i+1:]
		}
		if i := bytes.Index(line, slashSlash); i >= 0 {
			line = line[:i]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, moduleStr) {
			continue
		}
		line = line[len(moduleStr):]
		n := len(line)
		line = bytes.TrimSpace(line)
		if len(line) == n || len(line) == 0 {
			continue
		}

		if line[0] == '"' || line[0] == '`' {
			p, err := strconv.Unquote(string(line))
			if err != nil {
				return "" // malformed quoted string or multiline module path
			}
			return p
		}

		return string(line)
	}
	return "" // missing module path
}
