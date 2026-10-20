// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"reflect"
	"slices"
	"strconv"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// lowerMakeLen rewrites every product inside a make length or capacity of
// one file (SPEC Sec 15.2, T-303; #169): `a*b` becomes
// _logos_rt.MulLen(a, b, "<product>"), so an overflowing product panics
// instead of allocating a wrapped buffer. The native tree calls MulLen and
// the eject text splices the same call over the product's source span. It
// runs before the rt import is spelled, because it may need that import.
func (fl *fileLower) lowerMakeLen() {
	prods := fl.makeLenProducts()
	if len(prods) == 0 {
		return
	}
	fl.needRT = true
	set := make(map[syntax.Expr]bool, len(prods))
	for _, op := range prods {
		set[op] = true
		fl.spliceMakeLen(op)
	}
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n != nil {
			replaceMulLenFields(n, set)
		}
		return true
	})
}

// isMakeCall reports whether c calls the predeclared make.
func (fl *fileLower) isMakeCall(c *syntax.CallExpr) bool {
	name, ok := c.Fun.(*syntax.Name)
	if !ok {
		return false
	}
	b, ok := fl.pkg.info.Uses[name].(*types2.Builtin)
	return ok && b.Name() == "make"
}

// makeLenProducts returns the multiplications inside the length and
// capacity arguments of every make call of the file. A constant product
// stays out: its type is fixed at compile time, and MulLen would infer
// float64 from untyped operands. A product comes after its ancestors in the
// walk, so the reversed list holds each product before the product that
// contains it, the order the splices need.
func (fl *fileLower) makeLenProducts() []*syntax.Operation {
	var out []*syntax.Operation
	seen := make(map[*syntax.Operation]bool)
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		c, ok := n.(*syntax.CallExpr)
		if !ok || !fl.isMakeCall(c) {
			return true
		}
		for _, arg := range c.ArgList[min(1, len(c.ArgList)):] {
			syntax.Inspect(arg, func(m syntax.Node) bool {
				if op, ok := lenProduct(m); ok && !seen[op] {
					seen[op] = true
					out = append(out, op)
				}
				return true
			})
		}
		return true
	})
	slices.Reverse(out)
	return out
}

// lenProduct returns n as a multiplication that types2 does not record as
// a constant and whose type has integer underlying type. The constant
// value sits in the node's type information, as everywhere else in this
// package. A product of any other type stays out: MulLen constrains its
// operands to integers, so wrapping a float product such as f*g under
// int(f*g) would fail the recheck.
func lenProduct(n syntax.Node) (*syntax.Operation, bool) {
	op, ok := n.(*syntax.Operation)
	if !ok || op.Op != syntax.Mul || op.Y == nil {
		return nil, false
	}
	tv := op.GetTypeInfo()
	if tv.Value != nil || !isIntegerType(tv.Type) {
		return nil, false
	}
	return op, true
}

// isIntegerType reports whether t has integer underlying type.
func isIntegerType(t types2.Type) bool {
	if t == nil {
		return false
	}
	b, ok := types2.Unalias(t).Underlying().(*types2.Basic)
	return ok && b.Info()&types2.IsInteger != 0
}

// spliceMakeLen records the eject edit that spells one product as a MulLen
// call. The left operand ends at the product's operator. The right operand
// starts just after the operator, so a comment in that gap stays, and ends
// where operandEnd finds it (EndPos would drop closing brackets). textOf
// absorbs the splices recorded inside each operand. The message names the
// product as the printer renders it, so "w*h" and "w * h" give the same
// panic text.
func (fl *fileLower) spliceMakeLen(op *syntax.Operation) {
	start := fl.off(syntax.StartPos(op))
	opOff := fl.off(op.Pos())
	yStart := fl.off(syntax.StartPos(op.Y))
	yEnd := fl.operandEnd(yStart)
	x := fl.textOf(span{start, opOff})
	y := fl.textOf(span{opOff + len("*"), yEnd})
	fl.add(start, yEnd, "_logos_rt.MulLen("+x+", "+y+", "+strconv.Quote(renderExpr(op))+")")
}

// operandEnd returns the offset just past the right operand of a product
// that starts at from. The operand ends at the first depth-zero token that
// cannot continue it: a binary operator, a separator, or a closing bracket.
// Prefix operators at its start belong to it.
func (fl *fileLower) operandEnd(from int) int {
	end := len(fl.src)
	scan := operandScan{prefix: true}
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment {
			return true
		}
		if scan.ends(t) {
			end = t.start
			return false
		}
		return true
	})
	return end
}

// operandScan is the state of one right-operand scan: the bracket depth
// and whether only prefix operators have been seen so far.
type operandScan struct {
	depth  int
	prefix bool
}

// ends reports whether t ends the operand: the end of the line or file, a
// closing bracket that matches no opener of the operand, or a depth-zero
// binary operator or separator. A prefix operator at the start continues it.
func (s *operandScan) ends(t tok) bool {
	if t.auto || t.kind == token.EOF {
		return true
	}
	if s.prefix && s.depth == 0 && prefixOp(t.kind) {
		return false
	}
	s.prefix = false
	return s.nest(t.kind)
}

// nest tracks the bracket depth for token k and reports whether k ends the
// operand at depth zero.
func (s *operandScan) nest(k token.Token) bool {
	switch k {
	case token.LPAREN, token.LBRACK, token.LBRACE:
		s.depth++
		return false
	case token.RPAREN, token.RBRACK, token.RBRACE:
		s.depth--
		return s.depth < 0
	}
	return s.depth == 0 && stopOp(k)
}

// prefixOp reports whether k can open a unary operand.
func prefixOp(k token.Token) bool {
	switch k {
	case token.ADD, token.SUB, token.MUL, token.AND, token.XOR, token.NOT, token.ARROW:
		return true
	}
	return false
}

// stopOp reports whether k ends a right operand at depth zero: a binary
// operator or a separator.
func stopOp(k token.Token) bool {
	switch k {
	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.AND, token.OR,
		token.XOR, token.SHL, token.SHR, token.AND_NOT, token.LAND, token.LOR,
		token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ,
		token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
		return true
	}
	return false
}

// replaceMulLenFields replaces each product held directly by n, as a field
// or a slice element, with its MulLen call. Inspect walks the new call's
// operands after the replacement, so nested products lower too.
func replaceMulLenFields(n syntax.Node, set map[syntax.Expr]bool) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	t := v.Type()
	for i := range v.NumField() {
		if t.Field(i).IsExported() {
			replaceMulLenField(v.Field(i), set)
		}
	}
}

// replaceMulLenField replaces a product held by the interface value f, or
// by an element of the slice or array f.
func replaceMulLenField(f reflect.Value, set map[syntax.Expr]bool) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if op, ok := f.Elem().Interface().(*syntax.Operation); ok && set[op] {
			f.Set(reflect.ValueOf(mulLenCall(op)))
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			replaceMulLenField(f.Index(i), set)
		}
	}
}

// mulLenCall builds `_logos_rt.MulLen(x, y, "<product>")` for one product
// of the native tree. The operands move under the call without copying.
func mulLenCall(op *syntax.Operation) syntax.Expr {
	pos := op.Pos()
	msg := litOf(pos, strconv.Quote(renderExpr(op)), syntax.StringLit)
	return callOf(pos, rtSel(pos, "MulLen"), []syntax.Expr{op.X, op.Y, msg})
}
