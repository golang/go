// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
)

func (fl *fileLower) lowerAddrFresh() {
	fresh := fl.pkg.info.LogosAddrFresh
	if len(fresh) == 0 {
		return
	}
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		replaceFieldExprs(n, anyInterfaceFields, func(x syntax.Expr) syntax.Expr {
			if op, ok := x.(*syntax.Operation); ok && op.Op == syntax.And && fresh[op] {
				return fl.lowerAddrOp(op)
			}
			return x
		})
		return true
	})
}

func (fl *fileLower) lowerAddrOp(e *syntax.Operation) syntax.Expr {
	amp := fl.off(e.Pos())
	fl.add(amp, amp+1, "new(")
	fl.add(fl.unaryOperandEnd(amp), fl.unaryOperandEnd(amp), ")")
	pos := e.Pos()
	return at(&syntax.CallExpr{Fun: nameOf(pos, "new"), ArgList: []syntax.Expr{e.X}}, pos)
}

func (fl *fileLower) unaryOperandEnd(from int) int {
	last, lastBeforeQuestion := from, from
	end := -1
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
			end = last
			done = true
			return false
		}
		prev := last
		last = t.end
		if question >= 0 && t.kind != token.ILLEGAL {
			question = -1
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
			operator = needOperand
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				end = prev
				done = true
				return false
			}
			depth--
			operator = needOperator
		case token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
			if depth == 0 {
				end = prev
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
						end = lastBeforeQuestion
						done = true
						return false
					}
					question = t.start
					lastBeforeQuestion = prev
				}
				return true
			}
			base.Fatalf("logos: unexpected token in `&` operand at byte offset %d", t.start)
		case token.PERIOD:
			operator = needOperand
		case token.ARROW:
			if depth == 0 && operator {
				end = prev
				done = true
				return false
			}
		default:
			if isUnaryOperandEndOp(t.kind) {
				if depth == 0 && operator {
					end = prev
					done = true
					return false
				}
				return true
			}
			operator = needOperator
		}
		return true
	})
	if !done {
		end = last
	}
	return end
}

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
