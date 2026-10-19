// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"go/scanner"
	"go/token"
)

// This file finds exact token boundaries in original source text. The
// syntax package records start positions and the position of a call's
// opening parenthesis, but no end positions, and the eject text must splice
// user-written pieces (comments included) at exact offsets. Go's own
// scanner tokenizes the text; the loGo/s "?" scans as an illegal token,
// which is harmless here because only bracket depth and separators matter.

// span is a half-open byte range of the original source.
type span struct{ start, end int }

// tok is one token of the scanned source, in offsets of the whole text.
type tok struct {
	span
	kind    token.Token
	comment bool
	auto    bool
}

// tokenEnd returns the end offset of the token that starts at off.
func tokenEnd(seg []byte, off int, kind token.Token, lit string) int {
	switch {
	case kind == token.COMMENT && bytes.HasPrefix(seg[off:], []byte("//")):
		if i := bytes.IndexByte(seg[off:], '\n'); i >= 0 {
			return off + i
		}
		return len(seg)
	case kind == token.COMMENT:
		if i := bytes.Index(seg[off+2:], []byte("*/")); i >= 0 {
			return off + 2 + i + 2
		}
		return len(seg)
	case kind == token.STRING && lit[0] == '`':
		if i := bytes.IndexByte(seg[off+1:], '`'); i >= 0 {
			return off + 1 + i + 1
		}
		return len(seg)
	case lit != "":
		return off + len(lit)
	default:
		return off + len(kind.String())
	}
}

// scanSegment tokenizes src[from:to] and calls visit for each token,
// comments included, until visit returns false. Automatic semicolons are
// reported with auto set and an empty span at their offset.
func scanSegment(src []byte, from, to int, visit func(tok) bool) {
	seg := src[from:to]
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(seg))
	var s scanner.Scanner
	s.Init(file, seg, nil, scanner.ScanComments)
	for {
		pos, kind, lit := s.Scan()
		if kind == token.EOF {
			return
		}
		off := file.Offset(pos)
		t := tok{kind: kind, comment: kind == token.COMMENT}
		if kind == token.SEMICOLON && lit == "\n" {
			t.auto = true
			t.span = span{from + off, from + off}
		} else {
			t.span = span{from + off, from + tokenEnd(seg, off, kind, lit)}
		}
		if !visit(t) {
			return
		}
	}
}

// commentsIn returns the comment spans inside src[from:to].
func commentsIn(src []byte, from, to int) []span {
	var out []span
	scanSegment(src, from, to, func(t tok) bool {
		if t.comment {
			out = append(out, t.span)
		}
		return true
	})
	return out
}

// lastTokenEnd returns the end of the last token inside src[from:to] that
// is neither a comment nor an automatic semicolon, or from when there is
// none.
func lastTokenEnd(src []byte, from, to int) int {
	end := from
	scanSegment(src, from, to, func(t tok) bool {
		if !t.comment && !t.auto {
			end = t.end
		}
		return true
	})
	return end
}

// callSplit is the layout of one call's argument list.
type callSplit struct {
	args []span // one span per argument, comments excluded
	end  int    // offset just past the closing parenthesis
}

// splitCall splits the argument list whose opening parenthesis is at
// lparen. Commas and "..." at the call's own nesting depth separate
// arguments; brackets of any kind nest.
func splitCall(src []byte, lparen int) (callSplit, bool) {
	var out callSplit
	depth := 0
	cur := span{-1, -1}
	flush := func() {
		if cur.start >= 0 {
			out.args = append(out.args, cur)
		}
		cur = span{-1, -1}
	}
	done := false
	scanSegment(src, lparen, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
			if depth > 1 {
				cur = extend(cur, t.span)
			}
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
			if depth == 0 {
				flush()
				out.end = t.end
				done = true
				return false
			}
			cur = extend(cur, t.span)
		case token.COMMA, token.ELLIPSIS:
			if depth == 1 {
				flush()
			} else {
				cur = extend(cur, t.span)
			}
		default:
			cur = extend(cur, t.span)
		}
		return true
	})
	return out, done
}

// extend grows cur to cover t.
func extend(cur, t span) span {
	if cur.start < 0 {
		return t
	}
	return span{cur.start, t.end}
}

// stmtEnd returns the end of the statement that starts at from: the end of
// its last token before the terminating semicolon or closing brace.
func stmtEnd(src []byte, from int) int {
	end := from
	depth := 0
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK:
			depth--
		case token.RBRACE:
			if depth == 0 {
				return false
			}
			depth--
		case token.SEMICOLON:
			if depth == 0 {
				return false
			}
		}
		if !t.auto {
			end = t.end
		}
		return true
	})
	return end
}

// spanStartBack extends off backward past white space, comments, and open
// parentheses the parser dropped. A parenthesis before a statement start
// can only wrap that statement in checked code, so consuming it only ever
// removes a dropped wrapper. A line comment counts only when it opens its
// line, which keeps string contents out of reach.
func spanStartBack(src []byte, off int) int {
	start := off
	for {
		j := skipSpaceBack(src, start)
		if j > 0 && src[j-1] == '(' {
			start = j - 1
			continue
		}
		back, ok := commentStartBack(src, j)
		if ok {
			start = back
			continue
		}
		return start
	}
}

// skipSpaceBack skips ASCII white space before off.
func skipSpaceBack(src []byte, off int) int {
	for off > 0 && isSpace(src[off-1]) {
		off--
	}
	return off
}

// commentStartBack returns the start of the comment ending at j.
func commentStartBack(src []byte, j int) (int, bool) {
	if j >= 2 && src[j-2] == '*' && src[j-1] == '/' {
		if k := bytes.LastIndex(src[:j-2], []byte("/*")); k >= 0 {
			return k, true
		}
		return 0, false
	}
	line := src[:j]
	if k := bytes.LastIndexByte(src[:j], '\n'); k >= 0 {
		line = src[k+1 : j]
	}
	trimmed := line
	for len(trimmed) > 0 && (trimmed[0] == ' ' || trimmed[0] == '\t' || trimmed[0] == '\r') {
		trimmed = trimmed[1:]
	}
	if len(trimmed) >= 2 && trimmed[0] == '/' && trimmed[1] == '/' {
		return j - len(line) + (len(line) - len(trimmed)), true
	}
	return 0, false
}

// trimSpan shrinks s to exclude leading and trailing ASCII white space.
func trimSpan(src []byte, s span) span {
	for s.start < s.end && isSpace(src[s.start]) {
		s.start++
	}
	for s.end > s.start && isSpace(src[s.end-1]) {
		s.end--
	}
	return s
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
