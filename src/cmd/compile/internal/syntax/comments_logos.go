// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"io"
)

// CommentsDoMode is CommentsDo with the scanner mode of the source: a
// loGo/s file is scanned with LogosMode so interpolated strings and the ?
// token cannot hide or invent a comment.
func CommentsDoMode(src io.Reader, mode Mode, handler func(line, col uint, text string)) {
	var s scanner
	s.init(src, handler, comments|uint(mode))
	for s.tok != _EOF {
		s.next()
	}
}

// ParenEnd returns the position just after the parenthesis that closes the
// one at open, scanning src, the whole text of the file open belongs to,
// with the tokens of the given mode. Strings, runes, and comments cannot
// hide or invent a parenthesis. It reports false when open is not a
// parenthesis of src or no closing parenthesis follows.
func ParenEnd(src []byte, open Pos, mode Mode) (Pos, bool) {
	return closeEnd(src, open, mode, _Lparen, _Rparen)
}

// BracketEnd is ParenEnd for brackets: it returns the position just after the
// bracket that closes the "[" at open, the opening bracket of an index or
// slice expression, under the same rules.
func BracketEnd(src []byte, open Pos, mode Mode) (Pos, bool) {
	return closeEnd(src, open, mode, _Lbrack, _Rbrack)
}

// closeEnd returns the position just after the token that closes the openTok
// at open, scanning src with the tokens of the given mode. Only openTok and
// closeTok are counted, so other kinds of bracket inside are skipped.
func closeEnd(src []byte, open Pos, mode Mode, openTok, closeTok token) (Pos, bool) {
	from, ok := lineOffset(src, open.Line())
	if !ok {
		return Pos{}, false
	}
	var s scanner
	s.init(bytes.NewReader(src[from:]), func(uint, uint, string) {}, uint(mode))
	depth := 0
	for s.next(); s.tok != _EOF; s.next() {
		if depth == 0 {
			if s.tok == openTok && s.line == linebase && s.col == open.Col() {
				depth = 1
			}
			continue
		}
		switch s.tok {
		case openTok:
			depth++
		case closeTok:
			depth--
		}
		if depth == 0 {
			return MakePos(open.Base(), open.Line()+s.line-linebase, s.col+1), true
		}
	}
	return Pos{}, false
}

// lineOffset returns the byte offset of the start of the 1-based line.
func lineOffset(src []byte, line uint) (int, bool) {
	off := 0
	for l := uint(linebase); l < line; l++ {
		i := bytes.IndexByte(src[off:], '\n')
		if i < 0 {
			return 0, false
		}
		off += i + 1
	}
	return off, true
}
