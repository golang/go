// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"go/scanner"
	"go/token"
	"strings"
)

type span struct{ start, end int }

func insideAny(c span, pieces []span) bool {
	for _, p := range pieces {
		if c.start >= p.start && c.end <= p.end {
			return true
		}
	}
	return false
}

type tok struct {
	span
	kind    token.Token
	comment bool
	auto    bool
}

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

func scanSegment(src []byte, from, to int, visit func(tok) bool) {
	// The Go scanner knows nothing of $"..." literals: it pairs each
	// quote inside an interpolation hole as a string edge, and a rune,
	// comment, or raw string it opens there can run past the literal's
	// true end and swallow the real tokens that follow. Mask every
	// interp literal into a same-length raw string — the one Go token
	// that may span lines — so the scanner reads a single STRING token
	// for it and never sees the hole text. Masking preserves every
	// byte offset, newline, and the literal's semicolon insertion.
	seg := maskInterpLiterals(src[from:to])
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(seg))
	var s scanner.Scanner
	s.Init(file, seg, nil, scanner.ScanComments)
	for {
		p, kind, lit := s.Scan()
		if kind == token.EOF {
			return
		}
		off := file.Offset(p)
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

// maskInterpLiterals rewrites each $"..." or $`...` literal of seg as a
// raw string of the same length. Go literals and comments are skipped
// first, so a $ inside one is never mistaken for an interp opener.
func maskInterpLiterals(seg []byte) []byte {
	var out []byte
	for i := 0; i+1 < len(seg); i++ {
		if seg[i] != '$' || (seg[i+1] != '"' && seg[i+1] != '`') {
			i = skipGoLit(seg, i)
			continue
		}
		end := interpLitEnd(seg, i+1)
		if out == nil {
			out = append([]byte(nil), seg...)
		}
		out[i] = '`'
		for j := i + 1; j < end-1; j++ {
			if out[j] != '\n' {
				out[j] = ' '
			}
		}
		out[end-1] = '`'
		i = end - 1
	}
	if out == nil {
		return seg
	}
	return out
}

// skipGoLit returns the offset before the end of the Go literal or
// comment starting at seg[i], or i when seg[i] starts neither.
func skipGoLit(seg []byte, i int) int {
	switch seg[i] {
	case '"', '\'', '`':
		return goLitEnd(seg, i) - 1
	case '/':
		return commentEnd(seg, i) - 1
	}
	return i
}

// interpLitEnd returns the offset just past the $"..." or $`...`
// literal whose quote is seg[off]. {expr} holes nest and hold ordinary
// Go literals and comments of their own; {{ and }} escape literal
// braces outside holes; quoted escapes apply except in raw form.
func interpLitEnd(seg []byte, off int) int {
	q := seg[off]
	depth := 0
	for i := off + 1; i < len(seg); {
		switch c := seg[i]; {
		case depth == 0 && c == '\\' && q != '`':
			i += 2
		case depth == 0 && c == q:
			return i + 1
		case depth == 0 && c == '{' && i+1 < len(seg) && seg[i+1] == '{':
			i += 2
		case depth == 0 && c == '}':
			i++
			if i < len(seg) && seg[i] == '}' {
				i++
			}
		case c == '{':
			depth++
			i++
		case depth > 0 && c == '}':
			depth--
			i++
		case depth > 0 && (c == '"' || c == '\'' || c == '`'):
			i = goLitEnd(seg, i)
		case depth > 0 && c == '/':
			i = commentEnd(seg, i)
		default:
			i++
		}
	}
	return len(seg)
}

// commentEnd returns the offset just past the comment starting at
// seg[off], or off+1 when seg[off] does not start one.
func commentEnd(seg []byte, off int) int {
	if off+1 >= len(seg) {
		return off + 1
	}
	switch seg[off+1] {
	case '/':
		for i := off + 2; i < len(seg); i++ {
			if seg[i] == '\n' {
				return i
			}
		}
	case '*':
		for i := off + 2; i+1 < len(seg); i++ {
			if seg[i] == '*' && seg[i+1] == '/' {
				return i + 2
			}
		}
		return len(seg)
	default:
		return off + 1
	}
	return len(seg)
}

// goLitEnd returns the offset just past an ordinary Go string, rune, or
// raw literal starting at seg[off].
func goLitEnd(seg []byte, off int) int {
	q := seg[off]
	for i := off + 1; i < len(seg); {
		switch c := seg[i]; {
		case c == '\\' && q != '`':
			i += 2
		case c == q:
			return i + 1
		default:
			i++
		}
	}
	return len(seg)
}

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

func keepComments(src []byte, from, to int, text string) string {
	for _, c := range commentsIn(src, from, to) {
		body := string(src[c.start:c.end])
		if rest, ok := strings.CutPrefix(body, "//"); ok {
			body = "/*" + strings.ReplaceAll(rest, "*/", "* /") + "*/"
		}
		text += " " + strings.ReplaceAll(body, "\n", " ")
	}
	return text
}

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

func lineBreakBefore(src []byte, from, off int) string {
	last := lastTokenEnd(src, from, off)
	if last > from && bytes.IndexByte(src[last:off], '\n') < 0 {
		return "\n"
	}
	return ""
}

type callSplit struct {
	args []span
	end  int
}

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

func extend(cur, t span) span {
	if cur.start < 0 {
		return t
	}
	return span{cur.start, t.end}
}

func stmtEnd(src []byte, from int) int {
	end := from
	depth := 0
	header := -1
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment {
			return true
		}
		switch t.kind {
		case token.SWITCH:
			if header < 0 {
				header = depth
			}
		case token.LPAREN, token.LBRACK:
			depth++
		case token.LBRACE:
			if depth == header {
				header = -1
			}
			depth++
		case token.RPAREN, token.RBRACK:
			depth--
		case token.RBRACE:
			if depth == 0 {
				return false
			}
			depth--
		case token.SEMICOLON:
			if depth == 0 && header != 0 {
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

func skipSpaceBack(src []byte, off int) int {
	for off > 0 && isSpace(src[off-1]) {
		off--
	}
	return off
}

func skipBlankBack(src []byte, off int) int {
	for {
		off = skipSpaceBack(src, off)
		if off < 2 || src[off-2] != '*' || src[off-1] != '/' {
			return off
		}
		before, _, found := bytes.CutLast(src[:off-2], []byte("/*"))
		if !found {
			return off
		}
		off = len(before)
	}
}

func commentStartBack(src []byte, j int) (int, bool) {
	if j >= 2 && src[j-2] == '*' && src[j-1] == '/' {
		if k := bytes.LastIndex(src[:j-2], []byte("/*")); k >= 0 {
			indent := src[bytes.LastIndexByte(src[:k], '\n')+1 : k]
			return k, len(bytes.Trim(indent, " \t\r")) == 0
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
