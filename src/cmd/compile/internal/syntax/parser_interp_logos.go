// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"strconv"
	"strings"
)

// This file holds the loGo/s parser logic for M6-S1 interpolated strings
// (SPEC Sec 1.4, T-104.4). parser.go calls into it through one hook line
// only (G3); the rule runs solely in LogosMode, so plain Go parses
// exactly as upstream.
//
// The scanner delivers `$"..."` as one _Literal token with LitKind
// InterpLit and the full source spelling as its Value. The parser splits
// the inner text into literal parts and `{expr}` operands, parsing each
// operand with a sub-parser over the fragment bytes. Fragments run
// through the same expression grammar (including nested loGo/s forms) in
// the file's own position base with newline/space padding, so operand
// positions and errors are absolute file positions. A `{` without a
// matching `}` is `syntax error: unterminated interpolation` at the
// brace.

// logosInterpOperand parses a `$"..."` operand. It returns nil,
// consuming nothing, when the current token is not an interpolated
// string, so every other literal keeps the upstream path.
func (p *parser) logosInterpOperand() Expr {
	if p.tok != _Literal || p.kind != InterpLit {
		return nil
	}
	raw, pos, bad := p.lit, p.pos(), p.bad
	p.next()
	return p.parseInterp(raw, pos, bad)
}

// parseInterp splits raw (the full `$"..."` spelling at pos) into an
// InterpLit. bad reports a scanner-side error for this literal; with bad
// set the split stays lenient and emits no new error, so one cause keeps
// one error.
func (p *parser) parseInterp(raw string, pos Pos, bad bool) Expr {
	t := new(InterpExpr)
	t.pos = pos
	t.Raw = raw
	inner := strings.TrimPrefix(raw, `$"`)
	inner = strings.TrimSuffix(inner, `"`)
	parts, exprs := p.splitInterp(inner, pos, bad)
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

// interpFrag is one `{...}` operand: the raw fragment bytes, the
// absolute file position of its first byte, and the absolute positions
// of its opening `{` and closing `}`.
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

// splitInterp divides inner (the literal text between `$"` and the
// closing quote) into decoded parts and operand fragments. It mirrors
// the scanner's brace grammar; on a clean literal every `{` has its
// `}` and the only error is an unmatched `{`, reported at the brace.
func (p *parser) splitInterp(inner string, pos Pos, bad bool) ([]string, []interpFrag) {
	var parts []string
	var exprs []interpFrag
	var cur strings.Builder
	// line/col track the absolute file position of inner[i].
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
				// Unmatched `{`: the one strict-mode error.
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
			// A lone `}` reaches here only for a bad literal (the
			// scanner errored already); keep it literally.
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

// scanInterpFrag scans one `{...}` operand starting at inner[open]
// (the `{`, at absolute line/col). It returns the fragment (text without
// the braces), the resume offset and position past the matching `}`,
// and whether a match was found. Brace matching skips nested strings,
// runes, raw strings, and comments exactly like the scanner.
func (p *parser) scanInterpFrag(inner string, open int, line, col uint) (interpFrag, int, uint, uint, bool) {
	bracePos := MakePos(p.file, line, col)
	openFrag := interpFrag{openLine: line, openCol: col, bracePos: bracePos}
	depth := 1
	i := open + 1
	// fragLine/fragCol is the absolute position of inner[i]: one byte
	// past the `{` on the same line.
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
				// Resume past `}`. Newlines inside the fragment
				// advance the line; the column counts from the
				// last newline (or the `{` line). The closing
				// brace sits one byte before the resume point.
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

// parseInterpFrag parses one operand fragment with a sub-parser over the
// fragment bytes padded to its absolute file position, so the operand
// tree and any of its errors carry true file positions in the file base.
func (p *parser) parseInterpFrag(frag interpFrag) Expr {
	pad := strings.Repeat("\n", int(frag.line)-1) + strings.Repeat(" ", int(frag.col)-1)
	var sub parser
	sub.init(p.file, bytes.NewBufferString(pad+frag.text), func(err error) {
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

// logosInterpDecode decodes one literal part's escapes. The scanner
// validated them; on any failure the raw text stands (a bad literal
// already carries its error).
func logosInterpDecode(part string) string {
	if !strings.Contains(part, "\\") {
		return part
	}
	if s, err := strconv.Unquote(`"` + part + `"`); err == nil {
		return s
	}
	return part
}

// skipInterpString skips a nested interpreted string starting at the
// opening quote inner[off].
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

// skipInterpRune skips a nested rune literal starting at the opening
// quote inner[off].
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

// skipInterpRaw skips a nested raw string starting at the opening
// backquote inner[off].
func skipInterpRaw(inner string, off int) int {
	if j := strings.IndexByte(inner[off+1:], '`'); j >= 0 {
		return off + 1 + j + 1
	}
	return len(inner)
}

// skipInterpComment skips a `//` or `/* */` comment starting at inner[off].
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
