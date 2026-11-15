// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"io"
)

func CommentsDoMode(src io.Reader, mode Mode, handler func(line, col uint, text string)) {
	var s scanner
	s.init(src, handler, comments|uint(mode))
	for s.tok != _EOF {
		s.next()
	}
}

func ParenEnd(src []byte, open Pos, mode Mode) (Pos, bool) {
	return closeEnd(src, open, mode, _Lparen, _Rparen)
}

func BracketEnd(src []byte, open Pos, mode Mode) (Pos, bool) {
	return closeEnd(src, open, mode, _Lbrack, _Rbrack)
}

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
