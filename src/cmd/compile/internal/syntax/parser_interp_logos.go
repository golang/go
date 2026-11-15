// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"strconv"
	"strings"
)

func (p *parser) logosInterpOperand() Expr {
	if p.tok != _Literal || p.kind != InterpLit {
		return nil
	}
	raw, pos := p.lit, p.pos()
	p.next()
	return p.parseInterp(raw, pos)
}

func (p *parser) parseInterp(raw string, pos Pos) Expr {
	t := new(InterpExpr)
	t.pos = pos
	t.Raw = raw
	inner := strings.TrimPrefix(raw, `$"`)
	inner = strings.TrimSuffix(inner, `"`)
	parts, exprs := p.splitInterp(inner, pos)
	t.Parts = parts
	for _, frag := range exprs {
		if frag.bad {
			t.Exprs = append(t.Exprs, p.badExpr())
			continue
		}
		t.Exprs = append(t.Exprs, p.parseInterpFrag(frag))
		t.Opens = append(t.Opens, MakePos(p.file, frag.openLine, frag.openCol))
		t.Closes = append(t.Closes, MakePos(p.file, frag.closeLine, frag.closeCol))
	}
	for len(t.Opens) < len(t.Exprs) {
		t.Opens = append(t.Opens, pos)
	}
	for len(t.Closes) < len(t.Exprs) {
		t.Closes = append(t.Closes, pos)
	}
	return t
}

type interpFrag struct {
	text      string
	line      uint
	col       uint
	openLine  uint
	openCol   uint
	closeLine uint
	closeCol  uint
	bracePos  Pos
	bad       bool
}

func (p *parser) splitInterp(inner string, pos Pos) ([]string, []interpFrag) {
	var parts []string
	var exprs []interpFrag
	var cur strings.Builder
	line, col := pos.Line(), pos.Col()+2
	flush := func() {
		parts = append(parts, logosInterpDecode(cur.String()))
		cur.Reset()
	}
	i := 0
	for i < len(inner) {
		c := inner[i]
		switch {
		case c == '{':
			if i+1 < len(inner) && inner[i+1] == '{' {
				cur.WriteByte('{')
				i += 2
				col += 2
				continue
			}
			frag, next, nextLine, nextCol, ok := p.scanInterpFrag(inner, i, line, col)
			flush()
			exprs = append(exprs, frag)
			if !ok {
				p.syntaxErrorAt(frag.bracePos, "unterminated interpolation")
				parts = append(parts, "")
				return parts, exprs
			}
			i, line, col = next, nextLine, nextCol
		case c == '}':
			if i+1 < len(inner) && inner[i+1] == '}' {
				cur.WriteByte('}')
				i += 2
				col += 2
				continue
			}
			cur.WriteByte(c)
			i++
			col++
		case c == '\n':
			cur.WriteByte(c)
			i++
			line++
			col = 1
		default:
			cur.WriteByte(c)
			i++
			col++
		}
	}
	flush()
	for len(exprs) < len(parts)-1 {
		exprs = append(exprs, interpFrag{bad: true})
	}
	return parts, exprs
}

func (p *parser) scanInterpFrag(inner string, open int, line, col uint) (interpFrag, int, uint, uint, bool) {
	bracePos := MakePos(p.file, line, col)
	openFrag := interpFrag{openLine: line, openCol: col, bracePos: bracePos}
	depth := 1
	i := open + 1
	fragLine, fragCol := line, col+1
	j := i
	for j < len(inner) {
		c := inner[j]
		switch {
		case c == '{':
			depth++
			j++
		case c == '}':
			depth--
			j++
			if depth == 0 {
				nl := strings.Count(inner[open:j], "\n")
				if nl == 0 {
					rcol := col + uint(j-open)
					openFrag.text = inner[i : j-1]
					openFrag.line, openFrag.col = fragLine, fragCol
					openFrag.closeLine, openFrag.closeCol = line, rcol-1
					return openFrag, j, line, rcol, true
				}
				last := strings.LastIndexByte(inner[open:j], '\n')
				rcol := uint(j-open-last-1) + 1
				openFrag.text = inner[i : j-1]
				openFrag.line, openFrag.col = fragLine, fragCol
				openFrag.closeLine, openFrag.closeCol = line+uint(nl), rcol-1
				return openFrag, j, line + uint(nl), rcol, true
			}
		case c == '"':
			j = skipInterpString(inner, j)
		case c == '\'':
			j = skipInterpRune(inner, j)
		case c == '`':
			j = skipInterpRaw(inner, j)
		case c == '/' && j+1 < len(inner) && (inner[j+1] == '/' || inner[j+1] == '*'):
			j = skipInterpComment(inner, j)
		case c == '\\':
			j += 2
			if j > len(inner) {
				j = len(inner)
			}
		case c == '\n':
			j++
		default:
			j++
		}
	}
	return interpFrag{bracePos: bracePos, bad: true}, len(inner), line, col, false
}

func (p *parser) parseInterpFrag(frag interpFrag) Expr {
	var sub parser
	sub.init(p.file, bytes.NewBufferString(frag.text), func(err error) {
		if e, ok := err.(Error); ok {
			if p.first == nil {
				p.first = e
			}
			p.errcnt++
		}
		if p.errh != nil {
			p.errh(err)
		}
	}, p.pragh, p.mode)
	sub.source.setStart(frag.line, frag.col)
	sub.fnest = p.fnest
	sub.next()
	x := sub.expr()
	if sub.tok != _EOF && sub.tok != _Semi {
		p.errorAt(sub.pos(), "syntax error: expected '}' to close interpolation")
		return p.badExpr()
	}
	if x == nil {
		return p.badExpr()
	}
	return x
}

func logosInterpDecode(part string) string {
	if !strings.Contains(part, "\\") {
		return part
	}
	if s, err := strconv.Unquote(`"` + part + `"`); err == nil {
		return s
	}
	return part
}

func skipInterpString(inner string, off int) int {
	j := off + 1
	for j < len(inner) {
		switch inner[j] {
		case '"':
			return j + 1
		case '\\':
			j += 2
			continue
		case '\n':
			return j
		}
		j++
	}
	return j
}

func skipInterpRune(inner string, off int) int {
	j := off + 1
	for j < len(inner) {
		switch inner[j] {
		case '\'':
			return j + 1
		case '\\':
			j += 2
			continue
		case '\n':
			return j
		}
		j++
	}
	return j
}

func skipInterpRaw(inner string, off int) int {
	if j := strings.IndexByte(inner[off+1:], '`'); j >= 0 {
		return off + 1 + j + 1
	}
	return len(inner)
}

func skipInterpComment(inner string, off int) int {
	if inner[off+1] == '/' {
		if j := strings.IndexByte(inner[off:], '\n'); j >= 0 {
			return off + j
		}
		return len(inner)
	}
	if j := strings.Index(inner[off+2:], "*/"); j >= 0 {
		return off + 2 + j + 2
	}
	return len(inner)
}
