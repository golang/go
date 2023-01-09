// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package glob implements an LSP-compliant glob pattern matcher for testing.
package glob

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// A Glob is an LSP-compliant glob pattern, as defined by the spec:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#documentFilter
//
// NOTE: this implementation is currently only intended for testing. In order
// to make it production ready, we'd need to:
//   - verify it against the VS Code implementation
//   - add more tests
//   - microbenchmark, likely avoiding the element interface
//   - resolve the question of what is meant by "character". If it's a UTF-16
//     code (as we suspect) it'll be a bit more work.
//
// Quoting from the spec:
// Glob patterns can have the following syntax:
//   - `*` to match one or more characters in a path segment
//   - `?` to match on one character in a path segment
//   - `**` to match any number of path segments, including none
//   - `{}` to group sub patterns into an OR expression. (e.g. `**/*.{ts,js}`
//     matches all TypeScript and JavaScript files)
//   - `[]` to declare a range of characters to match in a path segment
//     (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
//   - `[!...]` to negate a range of characters to match in a path segment
//     (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but
//     not `example.0`)
//
// Expanding on this:
//   - '/' matches one or more literal slashes.
//   - any other character matches itself literally.
type Glob struct {
	elems []element // pattern elements
}

// Parse builds a Glob for the given pattern, returning an error if the pattern
// is invalid.
func Parse(pattern string) (*Glob, error) {
	g, _, err := parse(pattern, false)
	return g, err
}

func parse(pattern string, nested bool) (*Glob, string, error) {
	g := new(Glob)
	for len(pattern) > 0 {
		switch pattern[0] {
		case '/':
			pattern = pattern[1:]
			g.elems = append(g.elems, slash{})

		case '*':
			if len(pattern) > 1 && pattern[1] == '*' {
				if (len(g.elems) > 0 && g.elems[len(g.elems)-1] != slash{}) || (len(pattern) > 2 && pattern[2] != '/') {
					return nil, "", errors.New("** may only be adjacent to '/'")
				}
				pattern = pattern[2:]
				g.elems = append(g.elems, starStar{})
				break
			}
			pattern = pattern[1:]
			g.elems = append(g.elems, star{})

		case '?':
			pattern = pattern[1:]
			g.elems = append(g.elems, anyChar{})

		case '{':
			var gs group
			for pattern[0] != '}' {
				pattern = pattern[1:]
				g, pat, err := parse(pattern, true)
				if err != nil {
					return nil, "", err
				}
				if len(pat) == 0 {
					return nil, "", errors.New("unmatched '{'")
				}
				pattern = pat
				gs = append(gs, g)
			}
			pattern = pattern[1:]
			g.elems = append(g.elems, gs)

		case '}', ',':
			if nested {
				return g, pattern, nil
			}
			pattern = g.parseLiteral(pattern, false)

		case '[':
			pattern = pattern[1:]
			if len(pattern) == 0 {
				return nil, "", errBadRange
			}
			negate := false
			if pattern[0] == '!' {
				pattern = pattern[1:]
				negate = true
			}
			low, sz, err := readRangeRune(pattern)
			if err != nil {
				return nil, "", err
			}
			pattern = pattern[sz:]
			if len(pattern) == 0 || pattern[0] != '-' {
				return nil, "", errBadRange
			}
			pattern = pattern[1:]
			high, sz, err := readRangeRune(pattern)
			if err != nil {
				return nil, "", err
			}
			pattern = pattern[sz:]
			if len(pattern) == 0 || pattern[0] != ']' {
				return nil, "", errBadRange
			}
			pattern = pattern[1:]
			g.elems = append(g.elems, charRange{negate, low, high})

		default:
			pattern = g.parseLiteral(pattern, nested)
		}
	}
	return g, "", nil
}

// helper for decoding a rune in range elements, e.g. [a-z]
func readRangeRune(input string) (rune, int, error) {
	r, sz := utf8.DecodeRuneInString(input)
	var err error
	if r == utf8.RuneError {
		// See the documentation for DecodeRuneInString.
		switch sz {
		case 0:
			err = errBadRange
		case 1:
			err = errInvalidUTF8
		}
	}
	return r, sz, err
}

var (
	errBadRange    = errors.New("'[' patterns must be of the form [x-y]")
	errInvalidUTF8 = errors.New("invalid UTF-8 encoding")
)

func (g *Glob) parseLiteral(pattern string, nested bool) string {
	var specialChars string
	if nested {
		specialChars = "*?{[/},"
	} else {
		specialChars = "*?{[/"
	}
	end := strings.IndexAny(pattern, specialChars)
	if end == -1 {
		end = len(pattern)
	}
	g.elems = append(g.elems, literal(pattern[:end]))
	return pattern[end:]
}

func (g *Glob) String() string {
	var b strings.Builder
	for _, e := range g.elems {
		fmt.Fprint(&b, e)
	}
	return b.String()
}

// element holds a glob pattern element, as defined below.
type element fmt.Stringer

// element types.
type (
	slash     struct{} // One or more '/' separators
	literal   string   // string literal, not containing /, *, ?, {}, or []
	star      struct{} // *
	anyChar   struct{} // ?
	starStar  struct{} // **
	group     []*Glob  // {foo, bar, ...} grouping
	charRange struct { // [a-z] character range
		negate    bool
		low, high rune
	}
)

func (s slash) String() string    { return "/" }
func (l literal) String() string  { return string(l) }
func (s star) String() string     { return "*" }
func (a anyChar) String() string  { return "?" }
func (s starStar) String() string { return "**" }
func (g group) String() string {
	var parts []string
	for _, g := range g {
		parts = append(parts, g.String())
	}
	return "{" + strings.Join(parts, ",") + "}"
}
func (r charRange) String() string {
	return "[" + string(r.low) + "-" + string(r.high) + "]"
}

// Match reports whether the input string matches the glob pattern.
func (g *Glob) Match(input string) bool {
	return match(g.elems, input)
}

func match(elems []element, input string) (ok bool) {
	var elem interface{}
	for len(elems) > 0 {
		elem, elems = elems[0], elems[1:]
		switch elem := elem.(type) {
		case slash:
			if len(input) == 0 || input[0] != '/' {
				return false
			}
			for input[0] == '/' {
				input = input[1:]
			}

		case starStar:
			// Special cases:
			//  - **/a matches "a"
			//  - **/ matches everything
			//
			// Note that if ** is followed by anything, it must be '/' (this is
			// enforced by Parse).
			if len(elems) > 0 {
				elems = elems[1:]
			}

			// A trailing ** matches anything.
			if len(elems) == 0 {
				return true
			}

			// Backtracking: advance pattern segments until the remaining pattern
			// elements match.
			for len(input) != 0 {
				if match(elems, input) {
					return true
				}
				_, input = split(input)
			}
			return false

		case literal:
			if !strings.HasPrefix(input, string(elem)) {
				return false
			}
			input = input[len(elem):]

		case star:
			var segInput string
			segInput, input = split(input)

			elemEnd := len(elems)
			for i, e := range elems {
				if e == (slash{}) {
					elemEnd = i
					break
				}
			}
			segElems := elems[:elemEnd]
			elems = elems[elemEnd:]

			// A trailing * matches the entire segment.
			if len(segElems) == 0 {
				break
			}

			// Backtracking: advance characters until remaining subpattern elements
			// match.
			matched := false
			for i := range segInput {
				if match(segElems, segInput[i:]) {
					matched = true
					break
				}
			}
			if !matched {
				return false
			}

		case anyChar:
			if len(input) == 0 || input[0] == '/' {
				return false
			}
			input = input[1:]

		case group:
			// Append remaining pattern elements to each group member looking for a
			// match.
			var branch []element
			for _, m := range elem {
				branch = branch[:0]
				branch = append(branch, m.elems...)
				branch = append(branch, elems...)
				if match(branch, input) {
					return true
				}
			}
			return false

		case charRange:
			if len(input) == 0 || input[0] == '/' {
				return false
			}
			c, sz := utf8.DecodeRuneInString(input)
			if c < elem.low || c > elem.high {
				return false
			}
			input = input[sz:]

		default:
			panic(fmt.Sprintf("segment type %T not implemented", elem))
		}
	}

	return len(input) == 0
}

// split returns the portion before and after the first slash
// (or sequence of consecutive slashes). If there is no slash
// it returns (input, nil).
func split(input string) (first, rest string) {
	i := strings.IndexByte(input, '/')
	if i < 0 {
		return input, ""
	}
	first = input[:i]
	for j := i; j < len(input); j++ {
		if input[j] != '/' {
			return first, input[j:]
		}
	}
	return first, ""
}
