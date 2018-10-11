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
			if !strings.HasPrefix(text, commentStart) {
				continue
			}
			text = text[len(commentStart):]
			parsed, err := parse(fset, c.Pos()+4, text)
			if err != nil {
				return nil, err
			}
			notes = append(notes, parsed...)
		}
	}
	return notes, nil
}

func parse(fset *token.FileSet, base token.Pos, text string) ([]*Note, error) {
	var scanErr error
	s := new(scanner.Scanner).Init(strings.NewReader(text))
	s.Mode = scanner.GoTokens
	s.Error = func(s *scanner.Scanner, msg string) {
		scanErr = fmt.Errorf("%v:%s", fset.Position(base+token.Pos(s.Position.Offset)), msg)
	}
	notes, err := parseComment(s)
	if err != nil {
		return nil, fmt.Errorf("%v:%s", fset.Position(base+token.Pos(s.Position.Offset)), err)
	}
	if scanErr != nil {
		return nil, scanErr
	}
	for _, n := range notes {
		n.Pos += base
	}
	return notes, nil
}

func parseComment(s *scanner.Scanner) ([]*Note, error) {
	var notes []*Note
	for {
		n, err := parseNote(s)
		if err != nil {
			return nil, err
		}
		notes = append(notes, n)
		tok := s.Scan()
		switch tok {
		case ',':
			// continue
		case scanner.EOF:
			return notes, nil
		default:
			return nil, fmt.Errorf("unexpected %s parsing comment", scanner.TokenString(tok))
		}
	}
}

func parseNote(s *scanner.Scanner) (*Note, error) {
	if tok := s.Scan(); tok != scanner.Ident {
		return nil, fmt.Errorf("expected identifier, got %s", scanner.TokenString(tok))
	}
	n := &Note{
		Pos:  token.Pos(s.Position.Offset),
		Name: s.TokenText(),
	}
	switch s.Peek() {
	case ',', scanner.EOF:
		// no argument list present
		return n, nil
	case '(':
		// parse the argument list
		if tok := s.Scan(); tok != '(' {
			return nil, fmt.Errorf("expected ( got %s", scanner.TokenString(tok))
		}
		// special case the empty argument list
		if s.Peek() == ')' {
			if tok := s.Scan(); tok != ')' {
				return nil, fmt.Errorf("expected ) got %s", scanner.TokenString(tok))
			}
			n.Args = []interface{}{} // @name() is represented by a non-nil empty slice.
			return n, nil
		}
		// handle a normal argument list
		for {
			arg, err := parseArgument(s)
			if err != nil {
				return nil, err
			}
			n.Args = append(n.Args, arg)
			switch s.Peek() {
			case ')':
				if tok := s.Scan(); tok != ')' {
					return nil, fmt.Errorf("expected ) got %s", scanner.TokenString(tok))
				}
				return n, nil
			case ',':
				if tok := s.Scan(); tok != ',' {
					return nil, fmt.Errorf("expected , got %s", scanner.TokenString(tok))
				}
				// continue
			default:
				return nil, fmt.Errorf("unexpected %s parsing argument list", scanner.TokenString(s.Scan()))
			}
		}
	default:
		return nil, fmt.Errorf("unexpected %s parsing note", scanner.TokenString(s.Scan()))
	}
}

func parseArgument(s *scanner.Scanner) (interface{}, error) {
	tok := s.Scan()
	switch tok {
	case scanner.Ident:
		v := s.TokenText()
		switch v {
		case "true":
			return true, nil
		case "false":
			return false, nil
		case "nil":
			return nil, nil
		case "re":
			tok := s.Scan()
			switch tok {
			case scanner.String, scanner.RawString:
				pattern, _ := strconv.Unquote(s.TokenText()) // can't fail
				re, err := regexp.Compile(pattern)
				if err != nil {
					return nil, fmt.Errorf("invalid regular expression %s: %v", pattern, err)
				}
				return re, nil
			default:
				return nil, fmt.Errorf("re must be followed by string, got %s", scanner.TokenString(tok))
			}
		default:
			return Identifier(v), nil
		}

	case scanner.String, scanner.RawString:
		v, _ := strconv.Unquote(s.TokenText()) // can't fail
		return v, nil

	case scanner.Int:
		v, err := strconv.ParseInt(s.TokenText(), 0, 0)
		if err != nil {
			return nil, fmt.Errorf("cannot convert %v to int: %v", s.TokenText(), err)
		}
		return v, nil

	case scanner.Float:
		v, err := strconv.ParseFloat(s.TokenText(), 64)
		if err != nil {
			return nil, fmt.Errorf("cannot convert %v to float: %v", s.TokenText(), err)
		}
		return v, nil

	case scanner.Char:
		return nil, fmt.Errorf("unexpected char literal %s", s.TokenText())

	default:
		return nil, fmt.Errorf("unexpected %s parsing argument", scanner.TokenString(tok))
	}
}
