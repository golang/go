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
