// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expect

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strconv"
	"strings"
	"text/scanner"
)

const (
	commentStart = "@"
)

// Identifier is the type for an identifier in an Note argument list.
type Identifier string

// Parse collects all the notes present in a file.
// If content is nil, the filename specified is read and parsed, otherwise the
// content is used and the filename is used for positions and error messages.
// Each comment whose text starts with @ is parsed as a comma-separated
// sequence of notes.
// See the package documentation for details about the syntax of those
// notes.
func Parse(fset *token.FileSet, filename string, content []byte) ([]*Note, error) {
	var src interface{}
	if content != nil {
		src = content
	}
	// TODO: We should write this in terms of the scanner.
	// there are ways you can break the parser such that it will not add all the
	// comments to the ast, which may result in files where the tests are silently
	// not run.
	file, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if file == nil {
		return nil, err
	}
	return Extract(fset, file)
}

// Extract collects all the notes present in an AST.
// Each comment whose text starts with @ is parsed as a comma-separated
// sequence of notes.
// See the package documentation for details about the syntax of those
// notes.
func Extract(fset *token.FileSet, file *ast.File) ([]*Note, error) {
	var notes []*Note
	for _, g := range file.Comments {
		for _, c := range g.List {
			text := c.Text
			if strings.HasPrefix(text, "/*") {
				text = strings.TrimSuffix(text, "*/")
			}
			text = text[2:] // remove "//" or "/*" prefix

			// Allow notes to appear within comments.
			// For example:
			// "// //@mark()" is valid.
			// "// @mark()" is not valid.
			// "// /*@mark()*/" is not valid.
			var adjust int
			if i := strings.Index(text, commentStart); i > 2 {
				// Get the text before the commentStart.
				pre := text[i-2 : i]
				if pre != "//" {
					continue
				}
				text = text[i:]
				adjust = i
			}
			if !strings.HasPrefix(text, commentStart) {
				continue
			}
			text = text[len(commentStart):]
			parsed, err := parse(fset, token.Pos(int(c.Pos())+4+adjust), text)
			if err != nil {
				return nil, err
			}
			notes = append(notes, parsed...)
		}
	}
	return notes, nil
}

const invalidToken rune = 0

type tokens struct {
	scanner scanner.Scanner
	current rune
	err     error
	base    token.Pos
}

func (t *tokens) Init(base token.Pos, text string) *tokens {
	t.base = base
	t.scanner.Init(strings.NewReader(text))
	t.scanner.Mode = scanner.GoTokens
	t.scanner.Whitespace ^= 1 << '\n' // don't skip new lines
	t.scanner.Error = func(s *scanner.Scanner, msg string) {
		t.Errorf("%v", msg)
	}
	return t
}

func (t *tokens) Consume() string {
	t.current = invalidToken
	return t.scanner.TokenText()
}

func (t *tokens) Token() rune {
	if t.err != nil {
		return scanner.EOF
	}
	if t.current == invalidToken {
		t.current = t.scanner.Scan()
	}
	return t.current
}

func (t *tokens) Skip(r rune) int {
	i := 0
	for t.Token() == '\n' {
		t.Consume()
		i++
	}
	return i
}

func (t *tokens) TokenString() string {
	return scanner.TokenString(t.Token())
}

func (t *tokens) Pos() token.Pos {
	return t.base + token.Pos(t.scanner.Position.Offset)
}

func (t *tokens) Errorf(msg string, args ...interface{}) {
	if t.err != nil {
		return
	}
	t.err = fmt.Errorf(msg, args...)
}

func parse(fset *token.FileSet, base token.Pos, text string) ([]*Note, error) {
	t := new(tokens).Init(base, text)
	notes := parseComment(t)
	if t.err != nil {
		return nil, fmt.Errorf("%v:%s", fset.Position(t.Pos()), t.err)
	}
	return notes, nil
}

func parseComment(t *tokens) []*Note {
	var notes []*Note
	for {
		t.Skip('\n')
		switch t.Token() {
		case scanner.EOF:
			return notes
		case scanner.Ident:
			notes = append(notes, parseNote(t))
		default:
			t.Errorf("unexpected %s parsing comment, expect identifier", t.TokenString())
			return nil
		}
		switch t.Token() {
		case scanner.EOF:
			return notes
		case ',', '\n':
			t.Consume()
		default:
			t.Errorf("unexpected %s parsing comment, expect separator", t.TokenString())
			return nil
		}
	}
}

func parseNote(t *tokens) *Note {
	n := &Note{
		Pos:  t.Pos(),
		Name: t.Consume(),
	}

	switch t.Token() {
	case ',', '\n', scanner.EOF:
		// no argument list present
		return n
	case '(':
		n.Args = parseArgumentList(t)
		return n
	default:
		t.Errorf("unexpected %s parsing note", t.TokenString())
		return nil
	}
}

func parseArgumentList(t *tokens) []interface{} {
	args := []interface{}{} // @name() is represented by a non-nil empty slice.
	t.Consume()             // '('
	t.Skip('\n')
	for t.Token() != ')' {
		args = append(args, parseArgument(t))
		if t.Token() != ',' {
			break
		}
		t.Consume()
		t.Skip('\n')
	}
	if t.Token() != ')' {
		t.Errorf("unexpected %s parsing argument list", t.TokenString())
		return nil
	}
	t.Consume() // ')'
	return args
}

func parseArgument(t *tokens) interface{} {
	switch t.Token() {
	case scanner.Ident:
		v := t.Consume()
		switch v {
		case "true":
			return true
		case "false":
			return false
		case "nil":
			return nil
		case "re":
			if t.Token() != scanner.String && t.Token() != scanner.RawString {
				t.Errorf("re must be followed by string, got %s", t.TokenString())
				return nil
			}
			pattern, _ := strconv.Unquote(t.Consume()) // can't fail
			re, err := regexp.Compile(pattern)
			if err != nil {
				t.Errorf("invalid regular expression %s: %v", pattern, err)
				return nil
			}
			return re
		default:
			return Identifier(v)
		}

	case scanner.String, scanner.RawString:
		v, _ := strconv.Unquote(t.Consume()) // can't fail
		return v

	case scanner.Int:
		s := t.Consume()
		v, err := strconv.ParseInt(s, 0, 0)
		if err != nil {
			t.Errorf("cannot convert %v to int: %v", s, err)
		}
		return v

	case scanner.Float:
		s := t.Consume()
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			t.Errorf("cannot convert %v to float: %v", s, err)
		}
		return v

	case scanner.Char:
		t.Errorf("unexpected char literal %s", t.Consume())
		return nil

	default:
		t.Errorf("unexpected %s parsing argument", t.TokenString())
		return nil
	}
}
