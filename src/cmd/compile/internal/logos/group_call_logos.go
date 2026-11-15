// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type groupOperand struct {
	expr  syntax.Expr
	span  span
	names []string
	decl  types2.Type
}

func (fc *funcLower) groupCapture(pos syntax.Pos, call *syntax.CallExpr) ([]syntax.Stmt, string) {
	fl := fc.fl
	fun, args := fc.groupOperands(call)
	var native []syntax.Stmt
	var text strings.Builder
	var list []syntax.Expr
	for _, op := range append([]groupOperand{fun}, args...) {
		list = append(list, groupUses(op)...)
		if len(op.names) == 0 {
			continue
		}
		mark := fl.moveRegion(op.span, strings.Join(op.names, ", "))
		head, tail := fl.drainBoundaries(op.span)
		stmts, stmtText := fc.groupBind(pos, op, cat(head, gen(mark), tail).s)
		native = append(native, stmts...)
		text.WriteString(stmtText)
		scanForLiterals(fl, op.expr)
	}
	call.Fun, call.ArgList = list[0], list[1:]
	return native, text.String()
}

func groupUses(op groupOperand) []syntax.Expr {
	if len(op.names) == 0 {
		return syntax.UnpackListExpr(op.expr)
	}
	uses := make([]syntax.Expr, len(op.names))
	for i, name := range op.names {
		uses[i] = nameOf(syntax.StartPos(op.expr), name)
	}
	return uses
}

func (fc *funcLower) groupOperands(call *syntax.CallExpr) (groupOperand, []groupOperand) {
	fl := fc.fl
	src := fl.src
	lparen := fl.off(call.Pos())
	split, ok := splitCall(src, lparen)
	if !ok || len(split.args) != len(call.ArgList) {
		if ok && len(call.ArgList) == 0 {
			return groupOperand{expr: call.Fun}, nil
		}
		base.Fatalf("logos: cannot split the operands of the group call at %s", call.Pos().String())
	}
	start := fl.off(leftmostOf(call.Fun))
	fun := groupOperand{expr: call.Fun, span: fl.groupOperandSpan(call.Fun, span{start: start, end: lastTokenEnd(src, start, lparen)})}
	if !isBuiltinExpr(fc, call.Fun) && !fc.groupStaticFunc(call.Fun) {
		fun.names = []string{fc.nextFun()}
	}
	args := make([]groupOperand, len(call.ArgList))
	for i, raw := range call.ArgList {
		arg := groupOperand{expr: raw, span: fl.groupOperandSpan(raw, trimSpan(src, split.args[i]))}
		switch kind := argKind(fc, raw); {
		case kind == argInline || fc.groupStaticFunc(raw):
		case kind == argMulti:
			arg.names = fc.multiNames(raw)
		default:
			arg.names = []string{fc.nextArg()}
			arg.decl = fc.groupDeclaredType(raw)
		}
		args[i] = arg
	}
	return fun, args
}

func (fl *fileLower) groupOperandSpan(e syntax.Expr, source span) span {
	if ws, ok := fl.wrapSpan[syntax.Unparen(e)]; ok {
		return ws
	}
	return source
}

func (fc *funcLower) groupStaticFunc(e syntax.Expr) bool {
	info := fc.fl.pkg.info
	switch n := syntax.Unparen(e).(type) {
	case *syntax.Name:
		_, ok := info.ObjectOf(n).(*types2.Func)
		return ok
	case *syntax.SelectorExpr:
		if _, selected := info.Selections[n]; selected {
			return false
		}
		_, ok := info.ObjectOf(n.Sel).(*types2.Func)
		return ok
	case *syntax.IndexExpr:
		return fc.groupStaticFunc(n.X)
	}
	return false
}

func (fc *funcLower) groupDeclaredType(raw syntax.Expr) types2.Type {
	info := fc.fl.pkg.info
	typ := info.TypeOf(raw)
	if typ == nil {
		return nil
	}
	op, ok := syntax.Unparen(raw).(*syntax.Operation)
	if !ok {
		return nil
	}
	var natural types2.Type
	switch {
	case op.Y == nil && op.Op == syntax.Not:
		natural = types2.Typ[types2.UntypedBool]
	case op.Y != nil && (op.Op == syntax.Eql || op.Op == syntax.Neq || op.Op == syntax.Lss ||
		op.Op == syntax.Leq || op.Op == syntax.Gtr || op.Op == syntax.Geq ||
		op.Op == syntax.AndAnd || op.Op == syntax.OrOr):
		natural = types2.Typ[types2.UntypedBool]
	case op.Y != nil && (op.Op == syntax.Shl || op.Op == syntax.Shr):
		if _, constant := typeConst(info, op.X); !constant {
			return nil
		}
		natural = types2.Typ[types2.UntypedInt]
	default:
		return nil
	}
	if types2.Identical(types2.Default(typ), types2.Default(natural)) {
		return nil
	}
	return typ
}

func (fc *funcLower) groupBind(pos syntax.Pos, op groupOperand, mark string) ([]syntax.Stmt, string) {
	list := strings.Join(op.names, ", ")
	lhs := make([]syntax.Expr, len(op.names))
	for i, name := range op.names {
		lhs[i] = nameOf(pos, name)
	}
	if op.decl != nil {
		if typ, spelled, ok := fc.fl.spellTypeBoth(op.decl, pos); ok {
			native := []syntax.Stmt{
				typedVarDecl(pos, nameOf(pos, op.names[0]), typ),
				assignOf(pos, lhs, []syntax.Expr{op.expr}),
			}
			return native, "var " + list + " " + spelled + "; " + list + " = " + mark + "; "
		}
	}
	return []syntax.Stmt{defineOf(pos, lhs, []syntax.Expr{op.expr})}, list + " := " + mark + "; "
}
