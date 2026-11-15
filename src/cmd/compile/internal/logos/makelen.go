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

func (fl *fileLower) isMakeCall(c *syntax.CallExpr) bool {
	name, ok := c.Fun.(*syntax.Name)
	if !ok {
		return false
	}
	b, ok := fl.pkg.info.Uses[name].(*types2.Builtin)
	return ok && b.Name() == "make"
}

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

func isIntegerType(t types2.Type) bool {
	if t == nil {
		return false
	}
	b, ok := types2.Unalias(t).Underlying().(*types2.Basic)
	return ok && b.Info()&types2.IsInteger != 0
}

func (fl *fileLower) spliceMakeLen(op *syntax.Operation) {
	start := fl.off(syntax.StartPos(op))
	opOff := fl.off(op.Pos())
	yStart := fl.off(syntax.StartPos(op.Y))
	yEnd := fl.operandEnd(yStart)
	x := fl.textOf(span{start, opOff})
	y := fl.textOf(span{opOff + len("*"), yEnd})
	fl.addT(start, yEnd, cat(
		gen("_logos_rt.MulLen("), x, gen(", "), y, gen(", "+strconv.Quote(renderExpr(op))+")")))
}

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

type operandScan struct {
	depth  int
	prefix bool
}

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

func prefixOp(k token.Token) bool {
	switch k {
	case token.ADD, token.SUB, token.MUL, token.AND, token.XOR, token.NOT, token.ARROW:
		return true
	}
	return false
}

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

func mulLenCall(op *syntax.Operation) syntax.Expr {
	pos := op.Pos()
	msg := litOf(pos, strconv.Quote(renderExpr(op)), syntax.StringLit)
	return callOf(pos, rtSel(pos, "MulLen"), []syntax.Expr{op.X, op.Y, msg})
}
