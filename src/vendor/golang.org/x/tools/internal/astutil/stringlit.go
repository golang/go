// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"go/token"
	"strconv"
	"unicode/utf8"
)

// RangeInStringLiteral returns the range within the string literal
// that corresponds to the byte offsets [start, end) within the string
// value it denotes. See [PosInStringLiteral] for details.
func RangeInStringLiteral(lit *ast.BasicLit, start, end int) Range {
	return Range{PosInStringLiteral(lit, start), PosInStringLiteral(lit, end)}
}

// PosInStringLiteral returns the source position within the string
// literal lit that corresponds to the byte offset within the string
// value it denotes. If offset falls within the encoding of a single
// rune (an escape sequence or non-ASCII character), PosInStringLiteral
// returns the start position of that rune.
//
// Because the scanner removes carriage returns (\r) from raw string
// literals, the result is accurate only if, in the source of such a
// literal, either no line feed or every line feed is preceded by a
// carriage return.
//
// Precondition: 0 <= offset <= len(Unquote(lit.Value)).
// If lit is malformed (for example, due to a parse error), the
// precondition is relaxed and the result is a best-effort
// approximation; offsets beyond the end yield the end position.
//
// TODO(adonovan): propose as method [ast.BasicLit.StringPos].
func PosInStringLiteral(lit *ast.BasicLit, offset int) token.Pos {
	v := lit.Value
	if lit.Kind != token.STRING {
		panic("not a string literal")
	}
	if offset < 0 {
		panic("negative offset")
	}
	var (
		quote = v[0]                                              // '"' or '`'
		norm  = quote == '`' && int(lit.End()-lit.Pos()) > len(v) // carriage returns were removed
		s     = v[1:]                                             // remaining source text
		pos   = lit.Pos() + 1                                     // source position of s
		off   = 0                                                 // value offset of s
		ok    = true                                              // literal is well-formed (so far)
	)
	if s != "" && s[len(s)-1] == quote {
		s = s[:len(s)-1] // closing quote (or an escaped quote, detected below)
	} else {
		ok = false // unterminated
	}
	for s != "" {
		// Decode the next character, which occupies
		// n bytes of s, srcLen bytes of source, and valLen bytes of value.
		var n, srcLen, valLen int
		if quote == '`' {
			_, n = utf8.DecodeRuneInString(s)
			srcLen, valLen = n, n
			if norm && s[0] == '\n' {
				srcLen++ // assume a carriage return preceded each line feed
			}
		} else {
			r, multibyte, rest, err := strconv.UnquoteChar(s, '"')
			if err != nil {
				// Malformed: treat the next byte as a byte of value.
				ok = false
				n, srcLen, valLen = 1, 1, 1
			} else {
				n = len(s) - len(rest)
				srcLen = n
				// Within a string literal, hex and octal escapes denote single bytes.
				// (Invalid UTF-8 in the source denotes U+FFFD, like [strconv.Unquote].)
				valLen = 1
				if multibyte {
					valLen = utf8.RuneLen(r)
				}
			}
		}
		if off+valLen > offset {
			return pos // offset is at or within this character
		}
		s = s[n:]
		pos += token.Pos(srcLen)
		off += valLen
	}
	if off < offset && ok {
		panic("offset out of range")
	}
	return pos
}
