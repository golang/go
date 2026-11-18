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
			fl.replaceMulLenFields(n, set)
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

// spliceMakeLen wraps the product in _logos_rt.MulLen without covering the
// operands: three boundary edits leave [start, opOff) and [yStart, yEnd) as
// untouched source, so splices a later lowering records inside either operand
// — a lambda, a property, a frozen operand — are still applied. A single
// covering splice would drop them as orphans.
//
// The edits fuse with any wrap lowered around the operand: drainBoundaries
// pulls a pending wrap's owned boundary inserts — a `??` closure's head and
// `}()` tail — so `MulLen(` lands inside its return slot rather than after
// its closing text, and the narrowest move holding the operand gets the
// edits as inner splices for the same reason (an insert routed by its own
// offset stays outside a moved region's end boundary).
func (fl *fileLower) spliceMakeLen(op *syntax.Operation) {
	start := fl.off(syntax.StartPos(op))
	opOff := fl.off(op.Pos())
	yStart := fl.off(syntax.StartPos(op.Y))
	yEnd := fl.operandEnd(yStart)
	m := fl.narrowestMove(span{start: start, end: yEnd}, false)
	head, tail := fl.drainEnclosing(span{start: start, end: yEnd})
	put := func(start, end int, t otext) {
		fl.placeSplice(splice{start: start, end: end, text: t.s, chunks: t.chunks}, m)
	}
	put(start, start, cat(head, gen("_logos_rt.MulLen(")))
	put(opOff, opOff+len("*"), gen(","))
	// The trailer closes MulLen's argument list, so it lands after every
	// operand-internal suffix insert at the operand end — a fresh-address
	// rewrite's closing paren recorded by the later lowerAddrFresh pass, a
	// narrowed name's .Value — which placementAfterExpression sorts last at
	// one offset.
	trailer := cat(gen(", "+strconv.Quote(renderExpr(op))+")"), tail)
	fl.placeSplice(splice{start: yEnd, end: yEnd, text: trailer.s, chunks: trailer.chunks,
		placement: placementAfterExpression}, m)
}

func (fl *fileLower) operandEnd(from int) int {
	end := from
	scan := operandScan{prefix: true}
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment {
			return true
		}
		if scan.ends(t) {
			return false
		}
		end = t.end
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

func (fl *fileLower) replaceMulLenFields(n syntax.Node, set map[syntax.Expr]bool) {
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
			fl.replaceMulLenField(v.Field(i), set)
		}
	}
}

func (fl *fileLower) replaceMulLenField(f reflect.Value, set map[syntax.Expr]bool) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if op, ok := f.Elem().Interface().(*syntax.Operation); ok && set[op] {
			f.Set(reflect.ValueOf(fl.mulLenCall(op)))
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			fl.replaceMulLenField(f.Index(i), set)
		}
	}
}

func (fl *fileLower) mulLenCall(op *syntax.Operation) syntax.Expr {
	pos := op.Pos()
	msg := litOf(pos, strconv.Quote(renderExpr(op)), syntax.StringLit)
	call := callOf(pos, rtSel(pos, "MulLen"), []syntax.Expr{op.X, op.Y, msg})
	// The synthetic call carries no source parens: pin the product's
	// own span so source-extent walks — a group child's hoisted
	// operand, a chain boundary — see `n*m`, not a call at `*`.
	fl.wrapSpan[call] = span{fl.off(syntax.StartPos(op)), fl.off(syntax.EndPos(op))}
	return call
}
