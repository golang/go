// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser and scanner logic for M3-S2 (option
// types `T?`, `??`, and `?.`; SPEC T-202, T-104.3). parser.go and
// scanner.go call into it through hook lines only (G3); every rule below
// runs solely in LogosMode, so plain Go scans and parses exactly as
// upstream. The scanner never produces _QuestionDot or _Coalesce outside
// LogosMode, so the parser cases for them are unreachable in plain mode.

// logosQuestionTail scans a `?.` or `??` tail. It is called in LogosMode
// right after the scanner consumed `?`, so s.ch is the following
// character; it mirrors logosFatArrow (scanner_logos.go). A `?` that
// joins a second `?` with no gap makes the pair one _Coalesce token, and
// a `?` that joins `.` with no gap makes the pair one _QuestionDot token
// unless a digit follows the dot (SPEC Sec 5.1). _Coalesce ends no
// operand (like `||`), _QuestionDot ends no operand (like `.`), and a
// lone `?` keeps the M1-S1 behavior. It reports false when the tail is
// neither, leaving the `.` or digit for the caller; plain Go never
// reaches it.
func (s *scanner) logosQuestionTail() bool {
	if s.mode&uint(LogosMode) == 0 {
		return false
	}
	switch s.ch {
	case '?':
		s.nextch()
		s.prec = precCoalesce
		s.tok = _Coalesce
		return true
	case '.':
		s.nextch()
		if !isDecimal(s.ch) {
			s.tok = _QuestionDot
			return true
		}
		// `?.5` is `?` plus a number, as before: rewind to `?` (the
		// active segment holds no newline) and consume it again.
		s.rewind()
		s.nextch()
	}
	return false
}

// logosOptionType wraps typ in an OptionType for each postfix `?` in
// LogosMode (SPEC T-202). The binding is outermost: `[]int?` is an
// option of a slice and `*U?` an option of a pointer, because element,
// key, value, and base positions parse through logosTypeElem (which never
// wraps) while every framing position parses through type_ (which wraps
// here). An element option needs parentheses: `[](int?)`. A nil typ
// passes through for the caller's missing-type path. In plain mode it
// returns typ unchanged.
func (p *parser) logosOptionType(typ Expr) Expr {
	if p.mode&LogosMode == 0 || typ == nil {
		return typ
	}
	for {
		switch p.tok {
		case _Question:
			t := new(OptionType)
			t.pos = p.pos()
			p.next()
			t.Elem = typ
			typ = t
		case _Coalesce:
			// The joined spelling `T??` is a syntax error (SPEC
			// T-202); `(T?)?` is the nested form. Report once,
			// consume, and keep a nested option so checking
			// continues without follow-on errors.
			p.syntaxError("T?? is not allowed; use (T?)?")
			pos := p.pos()
			p.next()
			for range 2 {
				t := new(OptionType)
				t.pos = pos
				t.Elem = typ
				typ = t
			}
		default:
			return typ
		}
	}
}

// logosOptionResult applies a postfix `?` that follows a parenthesized
// result list. In `func() (T?)?`, the parentheses group one unnamed type,
// so the `?` wraps that type and the result is the nested option (SPEC
// T-202). A `?` after a named or multi-value list is a syntax error,
// because a result list is not a type. With no `?`, it returns list
// unchanged. In plain mode it returns list unchanged.
func (p *parser) logosOptionResult(list []*Field) []*Field {
	if p.mode&LogosMode == 0 || p.tok != _Question && p.tok != _Coalesce {
		return list
	}
	if len(list) != 1 || list[0].Name != nil {
		p.syntaxError("? applies to one unnamed result type; write (T)? or (T?)?")
		p.next()
		return list
	}
	list[0].Type = p.logosOptionType(list[0].Type)
	return list
}

// logosTypeElem parses a component type: a slice, array, map, or channel
// element, a map key, or a pointer base. It is type_ without the
// postfix-`?` wrap, so in `[]int?` the `?` belongs to the slice, not the
// element (SPEC T-202, M3-S2). In plain mode it behaves exactly as
// type_, keeping the missing-type error path identical.
func (p *parser) logosTypeElem() Expr {
	if p.mode&LogosMode == 0 {
		return p.type_()
	}
	typ := p.typeOrNil()
	if typ == nil {
		typ = p.badExpr()
		p.syntaxError("expected type")
		p.advance(_Comma, _Colon, _Semi, _Rparen, _Rbrack, _Rbrace)
	}
	return typ
}

// logosCoalesce builds the `??` node for x at pos and parses its right
// side. `??` groups to the right, so the recursion runs one precedence
// level lower (SPEC T-104.3); `||` and tighter operators still bind first.
func (p *parser) logosCoalesce(x Expr, pos Pos) Expr {
	t := new(CoalesceExpr)
	t.pos = pos
	p.next()
	t.X = x
	t.Y = p.binaryExpr(nil, precCoalesce-1)
	return t
}

// logosOptSelect builds the `?.` node for x at pos. Only a name may
// follow `?.` (there is no `?.(` assertion form); anything else is a
// syntax error at the offending token with the M2 selector recovery, so
// an enclosing block still closes where the source closes it.
func (p *parser) logosOptSelect(x Expr, pos Pos) Expr {
	p.next()
	if p.tok != _Name {
		p.syntaxError("expected name after ?.")
		p.logosAdvanceSelector() // M2 hook: loGo/s recovery keeps `}`.
		return x
	}
	t := new(OptSelectExpr)
	t.pos = pos
	t.X = x
	t.Sel = p.name()
	return t
}
