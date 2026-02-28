// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// A Directive is a comment of this form:
//
//	//tool:name args
//
// For example, this directive:
//
//	//go:generate stringer -type Op -trimprefix Op
//
// would have Tool "go", Name "generate", and Args "stringer -type Op
// -trimprefix Op".
//
// While Args does not have a strict syntax, by convention it is a
// space-separated sequence of unquoted words, '"'-quoted Go strings, or
// '`'-quoted raw strings.
//
// See https://go.dev/doc/comment#directives for specification.
type Directive struct {
	Tool string
	Name string
	Args string // no leading or trailing whitespace

	// Slash is the position of the "//" at the beginning of the directive.
	Slash token.Pos

	// ArgsPos is the position where Args begins, based on the position passed
	// to ParseDirective.
	ArgsPos token.Pos
}

// ParseDirective parses a single comment line for a directive comment.
//
// If the line is not a directive comment, it returns false.
//
// The provided text must be a single line and should include the leading "//".
// If the text does not start with "//", it returns false.
//
// The caller may provide a file position of the start of c. This will be used
// to track the position of the arguments. This may be [Comment.Slash],
// synthesized by the caller, or simply 0. If the caller passes 0, then the
// positions are effectively byte offsets into the string c.
func ParseDirective(pos token.Pos, c string) (Directive, bool) {
	// Fast path to eliminate most non-directive comments. Must be a line
	// comment starting with [a-z0-9]
	if !(len(c) >= 3 && c[0] == '/' && c[1] == '/' && isalnum(c[2])) {
		return Directive{}, false
	}

	buf := directiveScanner{c, pos}
	buf.skip(len("//"))

	// Check for a valid directive and parse tool part.
	//
	// This logic matches isDirective. (We could combine them, but isDirective
	// itself is duplicated in several places.)
	colon := strings.Index(buf.str, ":")
	if colon <= 0 || colon+1 >= len(buf.str) {
		return Directive{}, false
	}
	for i := 0; i <= colon+1; i++ {
		if i == colon {
			continue
		}
		if !isalnum(buf.str[i]) {
			return Directive{}, false
		}
	}
	tool := buf.take(colon)
	buf.skip(len(":"))

	// Parse name and args.
	name := buf.takeNonSpace()
	buf.skipSpace()
	argsPos := buf.pos
	args := strings.TrimRightFunc(buf.str, unicode.IsSpace)

	return Directive{tool, name, args, pos, argsPos}, true
}

func isalnum(b byte) bool {
	return 'a' <= b && b <= 'z' || '0' <= b && b <= '9'
}

func (d *Directive) Pos() token.Pos { return d.Slash }
func (d *Directive) End() token.Pos { return token.Pos(int(d.ArgsPos) + len(d.Args)) }

// A DirectiveArg is an argument to a directive comment.
type DirectiveArg struct {
	// Arg is the parsed argument string. If the argument was a quoted string,
	// this is its unquoted form.
	Arg string
	// Pos is the position of the first character in this argument.
	Pos token.Pos
}

// ParseArgs parses a [Directive]'s arguments using the standard convention,
// which is a sequence of tokens, where each token may be a bare word, or a
// double quoted Go string, or a back quoted raw Go string. Each token must be
// separated by one or more Unicode spaces.
//
// If the arguments do not conform to this syntax, it returns an error.
func (d *Directive) ParseArgs() ([]DirectiveArg, error) {
	args := directiveScanner{d.Args, d.ArgsPos}

	list := []DirectiveArg{}
	for args.skipSpace(); args.str != ""; args.skipSpace() {
		var arg string
		argPos := args.pos

		switch args.str[0] {
		default:
			arg = args.takeNonSpace()

		case '`', '"':
			q, err := strconv.QuotedPrefix(args.str)
			if err != nil { // Always strconv.ErrSyntax
				return nil, fmt.Errorf("invalid quoted string in //%s:%s: %s", d.Tool, d.Name, args.str)
			}
			// Any errors will have been returned by QuotedPrefix
			arg, _ = strconv.Unquote(args.take(len(q)))

			// Check that the quoted string is followed by a space (or nothing)
			if args.str != "" {
				r, _ := utf8.DecodeRuneInString(args.str)
				if !unicode.IsSpace(r) {
					return nil, fmt.Errorf("invalid quoted string in //%s:%s: %s", d.Tool, d.Name, args.str)
				}
			}
		}

		list = append(list, DirectiveArg{arg, argPos})
	}
	return list, nil
}

// directiveScanner is a helper for parsing directive comments while maintaining
// position information.
type directiveScanner struct {
	str string
	pos token.Pos
}

func (s *directiveScanner) skip(n int) {
	s.pos += token.Pos(n)
	s.str = s.str[n:]
}

func (s *directiveScanner) take(n int) string {
	res := s.str[:n]
	s.skip(n)
	return res
}

func (s *directiveScanner) takeNonSpace() string {
	i := strings.IndexFunc(s.str, unicode.IsSpace)
	if i == -1 {
		i = len(s.str)
	}
	return s.take(i)
}

func (s *directiveScanner) skipSpace() {
	trim := strings.TrimLeftFunc(s.str, unicode.IsSpace)
	s.skip(len(s.str) - len(trim))
}
