// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"reflect"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
)

// M6-S2 lowering of `&expr` over non-addressable operands (SPEC
// T-104.5): each recorded operation becomes `new(expr)`, so every
// evaluation yields a fresh variable. The native tree moves the operand
// under a new call; the eject text replaces `&` with `new(` and closes
// past the operand end. Small splices keep nested fresh operations
// composable in input order.

// lowerAddrFresh rewrites every fresh `&expr` of one file. It runs after
// every pass that records splices inside operands, so operand text is
// final when the closing splice lands.
func (fl *fileLower) lowerAddrFresh() {
	fresh := fl.pkg.info.LogosAddrFresh
	if len(fresh) == 0 {
		return
	}
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		replaceAddrFresh(n, fresh, fl.lowerAddrOp)
		return true
	})
}

// replaceAddrFresh replaces each recorded fresh `&` operation held
// directly by n with repl of it. The caller walks n's children after
// the call, so nested operations lower inside out through input-ordered
// splices.
func replaceAddrFresh(n syntax.Node, fresh map[syntax.Expr]bool, repl func(*syntax.Operation) syntax.Expr) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	typ := v.Type()
	for i := range v.NumField() {
		if typ.Field(i).IsExported() {
			replaceAddrField(v.Field(i), fresh, repl)
		}
	}
}

// replaceAddrField replaces a recorded fresh `&` operation held by the
// interface value f, or by an element of the slice or array f.
func replaceAddrField(f reflect.Value, fresh map[syntax.Expr]bool, repl func(*syntax.Operation) syntax.Expr) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if op, ok := f.Elem().Interface().(*syntax.Operation); ok && op.Op == syntax.And && fresh[op] {
			f.Set(reflect.ValueOf(repl(op)))
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			replaceAddrField(f.Index(i), fresh, repl)
		}
	}
}

// lowerAddrOp lowers one fresh `&expr` to `new(expr)`.
func (fl *fileLower) lowerAddrOp(e *syntax.Operation) syntax.Expr {
	amp := fl.off(e.Pos())
	fl.add(amp, amp+1, "new(")
	fl.add(fl.unaryOperandEnd(amp), fl.unaryOperandEnd(amp), ")")
	pos := e.Pos()
	return at(&syntax.CallExpr{Fun: nameOf(pos, "new"), ArgList: []syntax.Expr{e.X}}, pos)
}

// unaryOperandEnd returns the offset just past the unary operand opening
// at from, where from holds the leading `&`. Depth-zero tokens that end
// an operand stop the scan: binary operators, commas, semicolons,
// colons, assignments, closing brackets, and block braces. Brackets of
// any kind nest; a `?` followed by another `?` ends (the `??` fallback
// binds looser than `&`), while any other `?` continues (postfix `?`
// and `?.` bind tighter).
func (fl *fileLower) unaryOperandEnd(from int) int {
	end := len(fl.src)
	const (
		needOperand  = false
		needOperator = true
	)
	operator := needOperand
	depth := 0
	question := -1
	done := false
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if done {
			return false
		}
		if t.comment {
			return true
		}
		if t.auto {
			end = t.start
			done = true
			return false
		}
		if question >= 0 && t.kind != token.ILLEGAL {
			question = -1
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
			operator = needOperand
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				end = t.start
				done = true
				return false
			}
			depth--
			operator = needOperator
		case token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
			if depth == 0 {
				end = t.start
				done = true
				return false
			}
			if t.kind == token.COMMA || t.kind == token.COLON {
				operator = needOperand
			}
		case token.ILLEGAL:
			if string(fl.src[t.start:t.end]) == "?" {
				if depth == 0 && operator {
					if question >= 0 {
						end = question
						done = true
						return false
					}
					question = t.start
				}
				return true
			}
			base.Fatalf("logos: unexpected token in `&` operand at byte offset %d", t.start)
		case token.PERIOD:
			operator = needOperand
		case token.ARROW:
			if depth == 0 && operator {
				end = t.start
				done = true
				return false
			}
		default:
			if isUnaryOperandEndOp(t.kind) {
				if depth == 0 && operator {
					end = t.start
					done = true
					return false
				}
				return true
			}
			operator = needOperator
		}
		return true
	})
	return end
}

// isUnaryOperandEndOp reports whether kind is a binary operator: one
// that ends a unary operand after a complete operand but continues it
// in operand position as a unary operator.
func isUnaryOperandEndOp(kind token.Token) bool {
	switch kind {
	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM,
		token.AND, token.OR, token.XOR, token.SHL, token.SHR, token.AND_NOT,
		token.LAND, token.LOR,
		token.EQL, token.NEQ, token.LSS, token.GTR, token.LEQ, token.GEQ:
		return true
	}
	return false
}
