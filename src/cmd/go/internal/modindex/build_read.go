// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a lightly modified copy go/build/read.go with unused parts
// removed.

package modindex

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/scanner"
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

// readRest reads the entire rest of the file into r.buf.
func (r *importReader) readRest() {
	for {
		if len(r.buf) == cap(r.buf) {
			// Grow the buffer
			r.buf = append(r.buf, 0)[:len(r.buf)]
		}
		n, err := r.b.Read(r.buf[len(r.buf):cap(r.buf)])
		r.buf = r.buf[:len(r.buf)+n]
		if err != nil {
			if err == io.EOF {
				r.eof = true
			} else if r.err == nil {
				r.err = err
			}
			break
		}
	}
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
// info.imports and info.embeds.
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
		r.readRest()
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
			if !isValidImport(path) {
				// The parser used to return a parse error for invalid import paths, but
				// no longer does, so check for and create the error here instead.
				info.parseErr = scanner.Error{Pos: info.fset.Position(spec.Pos()), Msg: "invalid import path: " + path}
				info.imports = nil
				return nil
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

	// Extract directives.
	for _, group := range info.parsed.Comments {
		if group.Pos() >= info.parsed.Package {
			break
		}
		for _, c := range group.List {
			if strings.HasPrefix(c.Text, "//go:") {
				info.directives = append(info.directives, build.Directive{Text: c.Text, Pos: info.fset.Position(c.Slash)})
			}
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
		r.readRest()
		fset := token.NewFileSet()
		file := fset.AddFile(r.pos.Filename, -1, len(r.buf))
		var sc scanner.Scanner
		sc.Init(file, r.buf, nil, scanner.ScanComments)
		for {
			pos, tok, lit := sc.Scan()
			if tok == token.EOF {
				break
			}
			if tok == token.COMMENT && strings.HasPrefix(lit, "//go:embed") {
				// Ignore badly-formed lines - the compiler will report them when it finds them,
				// and we can pretend they are not there to help go list succeed with what it knows.
				embs, err := parseGoEmbed(fset, pos, lit)
				if err == nil {
					info.embeds = append(info.embeds, embs...)
				}
			}
		}
	}

	return nil
}

// isValidImport checks if the import is a valid import using the more strict
// checks allowed by the implementation restriction in https://go.dev/ref/spec#Import_declarations.
// It was ported from the function of the same name that was removed from the
// parser in CL 424855, when the parser stopped doing these checks.
func isValidImport(s string) bool {
	const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"
	for _, r := range s {
		if !unicode.IsGraphic(r) || unicode.IsSpace(r) || strings.ContainsRune(illegalChars, r) {
			return false
		}
	}
	return s != ""
}

// parseGoEmbed parses a "//go:embed" to extract the glob patterns.
// It accepts unquoted space-separated patterns as well as double-quoted and back-quoted Go strings.
// This must match the behavior of cmd/compile/internal/noder.go.
func parseGoEmbed(fset *token.FileSet, pos token.Pos, comment string) ([]fileEmbed, error) {
	dir, ok := ast.ParseDirective(pos, comment)
	if !ok || dir.Tool != "go" || dir.Name != "embed" {
		return nil, nil
	}
	args, err := dir.ParseArgs()
	if err != nil {
		return nil, err
	}
	var list []fileEmbed
	for _, arg := range args {
		list = append(list, fileEmbed{arg.Arg, fset.Position(arg.Pos)})
	}
	return list, nil
}
