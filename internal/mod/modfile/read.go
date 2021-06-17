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

	"golang.org/x/mod/modfile"
)

// An input represents a single input file being parsed.
type input struct {
	// Lexing state.
	filename   string            // name of input file, for errors
	complete   []byte            // entire input
	remaining  []byte            // remaining input
	tokenStart []byte            // token being scanned to end of input
	token      token             // next token to be returned by lex, peek
	pos        modfile.Position  // current input position
	comments   []modfile.Comment // accumulated comments

	// Parser state.
	file        *modfile.FileSyntax // returned top-level syntax tree
	parseErrors modfile.ErrorList   // errors encountered during parsing

	// Comment assignment state.
	pre  []modfile.Expr // all expressions, in preorder traversal
	post []modfile.Expr // all expressions, in postorder traversal
}

func newInput(filename string, data []byte) *input {
	return &input{
		filename:  filename,
		complete:  data,
		remaining: data,
		pos:       modfile.Position{Line: 1, LineRune: 1, Byte: 0},
	}
}

// parse parses the input file.
func parse(file string, data []byte) (f *modfile.FileSyntax, err error) {
	// The parser panics for both routine errors like syntax errors
	// and for programmer bugs like array index errors.
	// Turn both into error returns. Catching bug panics is
	// especially important when processing many files.
	in := newInput(file, data)
	defer func() {
		if e := recover(); e != nil && e != &in.parseErrors {
			in.parseErrors = append(in.parseErrors, modfile.Error{
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
	in.parseErrors = append(in.parseErrors, modfile.Error{
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
	pos    modfile.Position
	endPos modfile.Position
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

// peek returns the kind of the the next token returned by lex.
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
			in.comments = append(in.comments, modfile.Comment{in.token.pos, in.token.text, suffix})
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
func (in *input) order(x modfile.Expr) {
	if x != nil {
		in.pre = append(in.pre, x)
	}
	switch x := x.(type) {
	default:
		panic(fmt.Errorf("order: unexpected type %T", x))
	case nil:
		// nothing
	case *modfile.LParen, *modfile.RParen:
		// nothing
	case *modfile.CommentBlock:
		// nothing
	case *modfile.Line:
		// nothing
	case *modfile.FileSyntax:
		for _, stmt := range x.Stmt {
			in.order(stmt)
		}
	case *modfile.LineBlock:
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
	var line, suffix []modfile.Comment
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
		case *modfile.FileSyntax:
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
func reverseComments(list []modfile.Comment) {
	for i, j := 0, len(list)-1; i < j; i, j = i+1, j-1 {
		list[i], list[j] = list[j], list[i]
	}
}

func (in *input) parseFile() {
	in.file = new(modfile.FileSyntax)
	var cb *modfile.CommentBlock
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
				cb = &modfile.CommentBlock{Start: tok.pos}
			}
			com := cb.Comment()
			com.Before = append(com.Before, modfile.Comment{Start: tok.pos, Token: tok.text})
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
			in.file.Stmt = append(in.file.Stmt, &modfile.Line{
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
					in.file.Stmt = append(in.file.Stmt, &modfile.LineBlock{
						Start:  start,
						Token:  tokens,
						LParen: modfile.LParen{Pos: tok.pos},
						RParen: modfile.RParen{Pos: rparen.pos},
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

func (in *input) parseLineBlock(start modfile.Position, token []string, lparen token) *modfile.LineBlock {
	x := &modfile.LineBlock{
		Start:  start,
		Token:  token,
		LParen: modfile.LParen{Pos: lparen.pos},
	}
	var comments []modfile.Comment
	for {
		switch in.peek() {
		case _EOLCOMMENT:
			// Suffix comment, will be attached later by assignComments.
			in.lex()
		case '\n':
			// Blank line. Add an empty comment to preserve it.
			in.lex()
			if len(comments) == 0 && len(x.Line) > 0 || len(comments) > 0 && comments[len(comments)-1].Token != "" {
				comments = append(comments, modfile.Comment{})
			}
		case _COMMENT:
			tok := in.lex()
			comments = append(comments, modfile.Comment{Start: tok.pos, Token: tok.text})
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

func (in *input) parseLine() *modfile.Line {
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
			return &modfile.Line{
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
