// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type importReader struct {
	b    *bufio.Reader
	buf  []byte
	peek byte
	err  error
	eof  bool
	nerr int
	pos  token.Position
}

var bom = []byte{0xef, 0xbb, 0xbf}

func newImportReader(name string, r io.Reader) *importReader {
	b := bufio.NewReader(r)
	// Remove leading UTF-8 BOM.
	// Per https://golang.org/ref/spec#Source_code_representation:
	// a compiler may ignore a UTF-8-encoded byte order mark (U+FEFF)
	// if it is the first Unicode code point in the source text.
	if leadingBytes, err := b.Peek(3); err == nil && bytes.Equal(leadingBytes, bom) {
		b.Discard(3)
	}
	return &importReader{
		b: b,
		pos: token.Position{
			Filename: name,
			Line:     1,
			Column:   1,
		},
	}
}

func isIdent(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '_' || c >= utf8.RuneSelf
}

var (
	errSyntax = errors.New("syntax error")
	errNUL    = errors.New("unexpected NUL in input")
)

// syntaxError records a syntax error, but only if an I/O error has not already been recorded.
func (r *importReader) syntaxError() {
	if r.err == nil {
		r.err = errSyntax
	}
}

// readByte reads the next byte from the input, saves it in buf, and returns it.
// If an error occurs, readByte records the error in r.err and returns 0.
func (r *importReader) readByte() byte {
	c, err := r.b.ReadByte()
	if err == nil {
		r.buf = append(r.buf, c)
		if c == 0 {
			err = errNUL
		}
	}
	if err != nil {
		if err == io.EOF {
			r.eof = true
		} else if r.err == nil {
			r.err = err
		}
		c = 0
	}
	return c
}

// readByteNoBuf is like readByte but doesn't buffer the byte.
// It exhausts r.buf before reading from r.b.
func (r *importReader) readByteNoBuf() byte {
	var c byte
	var err error
	if len(r.buf) > 0 {
		c = r.buf[0]
		r.buf = r.buf[1:]
	} else {
		c, err = r.b.ReadByte()
		if err == nil && c == 0 {
			err = errNUL
		}
	}

	if err != nil {
		if err == io.EOF {
			r.eof = true
		} else if r.err == nil {
			r.err = err
		}
		return 0
	}
	r.pos.Offset++
	if c == '\n' {
		r.pos.Line++
		r.pos.Column = 1
	} else {
		r.pos.Column++
	}
	return c
}

// peekByte returns the next byte from the input reader but does not advance beyond it.
// If skipSpace is set, peekByte skips leading spaces and comments.
func (r *importReader) peekByte(skipSpace bool) byte {
	if r.err != nil {
		if r.nerr++; r.nerr > 10000 {
			panic("go/build: import reader looping")
		}
		return 0
	}

	// Use r.peek as first input byte.
	// Don't just return r.peek here: it might have been left by peekByte(false)
	// and this might be peekByte(true).
	c := r.peek
	if c == 0 {
		c = r.readByte()
	}
	for r.err == nil && !r.eof {
		if skipSpace {
			// For the purposes of this reader, semicolons are never necessary to
			// understand the input and are treated as spaces.
			switch c {
			case ' ', '\f', '\t', '\r', '\n', ';':
				c = r.readByte()
				continue

			case '/':
				c = r.readByte()
				if c == '/' {
					for c != '\n' && r.err == nil && !r.eof {
						c = r.readByte()
					}
				} else if c == '*' {
					var c1 byte
					for (c != '*' || c1 != '/') && r.err == nil {
						if r.eof {
							r.syntaxError()
						}
						c, c1 = c1, r.readByte()
					}
				} else {
					r.syntaxError()
				}
				c = r.readByte()
				continue
			}
		}
		break
	}
	r.peek = c
	return r.peek
}

// nextByte is like peekByte but advances beyond the returned byte.
func (r *importReader) nextByte(skipSpace bool) byte {
	c := r.peekByte(skipSpace)
	r.peek = 0
	return c
}

var goEmbed = []byte("go:embed")

// findEmbed advances the input reader to the next //go:embed comment.
// It reports whether it found a comment.
// (Otherwise it found an error or EOF.)
func (r *importReader) findEmbed(first bool) bool {
	// The import block scan stopped after a non-space character,
	// so the reader is not at the start of a line on the first call.
	// After that, each //go:embed extraction leaves the reader
	// at the end of a line.
	startLine := !first
	var c byte
	for r.err == nil && !r.eof {
		c = r.readByteNoBuf()
	Reswitch:
		switch c {
		default:
			startLine = false

		case '\n':
			startLine = true

		case ' ', '\t':
			// leave startLine alone

		case '"':
			startLine = false
			for r.err == nil {
				if r.eof {
					r.syntaxError()
				}
				c = r.readByteNoBuf()
				if c == '\\' {
					r.readByteNoBuf()
					if r.err != nil {
						r.syntaxError()
						return false
					}
					continue
				}
				if c == '"' {
					c = r.readByteNoBuf()
					goto Reswitch
				}
			}
			goto Reswitch

		case '`':
			startLine = false
			for r.err == nil {
				if r.eof {
					r.syntaxError()
				}
				c = r.readByteNoBuf()
				if c == '`' {
					c = r.readByteNoBuf()
					goto Reswitch
				}
			}

		case '/':
			c = r.readByteNoBuf()
			switch c {
			default:
				startLine = false
				goto Reswitch

			case '*':
				var c1 byte
				for (c != '*' || c1 != '/') && r.err == nil {
					if r.eof {
						r.syntaxError()
					}
					c, c1 = c1, r.readByteNoBuf()
				}
				startLine = false

			case '/':
				if startLine {
					// Try to read this as a //go:embed comment.
					for i := range goEmbed {
						c = r.readByteNoBuf()
						if c != goEmbed[i] {
							goto SkipSlashSlash
						}
					}
					c = r.readByteNoBuf()
					if c == ' ' || c == '\t' {
						// Found one!
						return true
					}
				}
			SkipSlashSlash:
				for c != '\n' && r.err == nil && !r.eof {
					c = r.readByteNoBuf()
				}
				startLine = true
			}
		}
	}
	return false
}

// readKeyword reads the given keyword from the input.
// If the keyword is not present, readKeyword records a syntax error.
func (r *importReader) readKeyword(kw string) {
	r.peekByte(true)
	for i := 0; i < len(kw); i++ {
		if r.nextByte(false) != kw[i] {
			r.syntaxError()
			return
		}
	}
	if isIdent(r.peekByte(false)) {
		r.syntaxError()
	}
}

// readIdent reads an identifier from the input.
// If an identifier is not present, readIdent records a syntax error.
func (r *importReader) readIdent() {
	c := r.peekByte(true)
	if !isIdent(c) {
		r.syntaxError()
		return
	}
	for isIdent(r.peekByte(false)) {
		r.peek = 0
	}
}

// readString reads a quoted string literal from the input.
// If an identifier is not present, readString records a syntax error.
func (r *importReader) readString() {
	switch r.nextByte(true) {
	case '`':
		for r.err == nil {
			if r.nextByte(false) == '`' {
				break
			}
			if r.eof {
				r.syntaxError()
			}
		}
	case '"':
		for r.err == nil {
			c := r.nextByte(false)
			if c == '"' {
				break
			}
			if r.eof || c == '\n' {
				r.syntaxError()
			}
			if c == '\\' {
				r.nextByte(false)
			}
		}
	default:
		r.syntaxError()
	}
}

// readImport reads an import clause - optional identifier followed by quoted string -
// from the input.
func (r *importReader) readImport() {
	c := r.peekByte(true)
	if c == '.' {
		r.peek = 0
	} else if isIdent(c) {
		r.readIdent()
	}
	r.readString()
}

// readComments is like io.ReadAll, except that it only reads the leading
// block of comments in the file.
func readComments(f io.Reader) ([]byte, error) {
	r := newImportReader("", f)
	r.peekByte(true)
	if r.err == nil && !r.eof {
		// Didn't reach EOF, so must have found a non-space byte. Remove it.
		r.buf = r.buf[:len(r.buf)-1]
	}
	return r.buf, r.err
}

// readGoInfo expects a Go file as input and reads the file up to and including the import section.
// It records what it learned in *info.
// If info.fset is non-nil, readGoInfo parses the file and sets info.parsed, info.parseErr,
// info.imports, info.embeds, and info.embedErr.
//
// It only returns an error if there are problems reading the file,
// not for syntax errors in the file itself.
func readGoInfo(f io.Reader, info *fileInfo) error {
	r := newImportReader(info.name, f)

	r.readKeyword("package")
	r.readIdent()
	for r.peekByte(true) == 'i' {
		r.readKeyword("import")
		if r.peekByte(true) == '(' {
			r.nextByte(false)
			for r.peekByte(true) != ')' && r.err == nil {
				r.readImport()
			}
			r.nextByte(false)
		} else {
			r.readImport()
		}
	}

	info.header = r.buf

	// If we stopped successfully before EOF, we read a byte that told us we were done.
	// Return all but that last byte, which would cause a syntax error if we let it through.
	if r.err == nil && !r.eof {
		info.header = r.buf[:len(r.buf)-1]
	}

	// If we stopped for a syntax error, consume the whole file so that
	// we are sure we don't change the errors that go/parser returns.
	if r.err == errSyntax {
		r.err = nil
		for r.err == nil && !r.eof {
			r.readByte()
		}
		info.header = r.buf
	}
	if r.err != nil {
		return r.err
	}

	if info.fset == nil {
		return nil
	}

	// Parse file header & record imports.
	info.parsed, info.parseErr = parser.ParseFile(info.fset, info.name, info.header, parser.ImportsOnly|parser.ParseComments)
	if info.parseErr != nil {
		return nil
	}

	hasEmbed := false
	for _, decl := range info.parsed.Decls {
		d, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, dspec := range d.Specs {
			spec, ok := dspec.(*ast.ImportSpec)
			if !ok {
				continue
			}
			quoted := spec.Path.Value
			path, err := strconv.Unquote(quoted)
			if err != nil {
				return fmt.Errorf("parser returned invalid quoted string: <%s>", quoted)
			}
			if path == "embed" {
				hasEmbed = true
			}

			doc := spec.Doc
			if doc == nil && len(d.Specs) == 1 {
				doc = d.Doc
			}
			info.imports = append(info.imports, fileImport{path, spec.Pos(), doc})
		}
	}

	// If the file imports "embed",
	// we have to look for //go:embed comments
	// in the remainder of the file.
	// The compiler will enforce the mapping of comments to
	// declared variables. We just need to know the patterns.
	// If there were //go:embed comments earlier in the file
	// (near the package statement or imports), the compiler
	// will reject them. They can be (and have already been) ignored.
	if hasEmbed {
		var line []byte
		for first := true; r.findEmbed(first); first = false {
			line = line[:0]
			pos := r.pos
			for {
				c := r.readByteNoBuf()
				if c == '\n' || r.err != nil || r.eof {
					break
				}
				line = append(line, c)
			}
			// Add args if line is well-formed.
			// Ignore badly-formed lines - the compiler will report them when it finds them,
			// and we can pretend they are not there to help go list succeed with what it knows.
			embs, err := parseGoEmbed(string(line), pos)
			if err == nil {
				info.embeds = append(info.embeds, embs...)
			}
		}
	}

	return nil
}

// parseGoEmbed parses the text following "//go:embed" to extract the glob patterns.
// It accepts unquoted space-separated patterns as well as double-quoted and back-quoted Go strings.
// This is based on a similar function in cmd/compile/internal/gc/noder.go;
// this version calculates position information as well.
func parseGoEmbed(args string, pos token.Position) ([]fileEmbed, error) {
	trimBytes := func(n int) {
		pos.Offset += n
		pos.Column += utf8.RuneCountInString(args[:n])
		args = args[n:]
	}
	trimSpace := func() {
		trim := strings.TrimLeftFunc(args, unicode.IsSpace)
		trimBytes(len(args) - len(trim))
	}

	var list []fileEmbed
	for trimSpace(); args != ""; trimSpace() {
		var path string
		pathPos := pos
	Switch:
		switch args[0] {
		default:
			i := len(args)
			for j, c := range args {
				if unicode.IsSpace(c) {
					i = j
					break
				}
			}
			path = args[:i]
			trimBytes(i)

		case '`':
			i := strings.Index(args[1:], "`")
			if i < 0 {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
			path = args[1 : 1+i]
			trimBytes(1 + i + 1)

		case '"':
			i := 1
			for ; i < len(args); i++ {
				if args[i] == '\\' {
					i++
					continue
				}
				if args[i] == '"' {
					q, err := strconv.Unquote(args[:i+1])
					if err != nil {
						return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args[:i+1])
					}
					path = q
					trimBytes(i + 1)
					break Switch
				}
			}
			if i >= len(args) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}

		if args != "" {
			r, _ := utf8.DecodeRuneInString(args)
			if !unicode.IsSpace(r) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}
		list = append(list, fileEmbed{path, pathPos})
	}
	return list, nil
}
