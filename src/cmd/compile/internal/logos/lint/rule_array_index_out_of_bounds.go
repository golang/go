// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"go/constant"
	"go/token"
	"internal/buildcfg"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

var arrayIndexOutOfBounds = Rule{
	ID:      "correctness::array_index_out_of_bounds",
	Group:   Correctness,
	Default: Deny,
	Go:      true,
	Check:   checkArrayIndexOutOfBounds,
}

func checkArrayIndexOutOfBounds(pass *Pass) {
	sizes := types2.SizesFor("gc", buildcfg.GOARCH)
	seen := make(map[*syntax.IndexExpr]bool)
	for _, file := range pass.Files {
		for _, decl := range file.DeclList {
			if fn, ok := decl.(*syntax.FuncDecl); ok && fn.Body != nil {
				checkArrayIndices(pass, sizes, fn, seen)
			}
		}
	}
}

func checkArrayIndices(pass *Pass, sizes types2.Sizes, fn *syntax.FuncDecl, seen map[*syntax.IndexExpr]bool) {
	params := arrayIndexParameters(pass, fn)
	if len(params) == 0 {
		return
	}
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		if isFunctionBody(n) {
			return false
		}
		index, ok := n.(*syntax.IndexExpr)
		if !ok || seen[index] {
			return true
		}
		seen[index] = true
		param := arrayIndexVariable(pass, index.Index)
		array := arrayIndexArray(index.X)
		if !params[param] || array == nil {
			return true
		}
		for _, value := range arrayIndexWitnesses(sizes, param, array.Len()) {
			proof := arrayIndexProof{pass: pass, target: index, param: param, value: value}
			if proof.block(fn.Body.List) == arrayIndexReached {
				pass.Reportf(index.Index, "index %s can be %s at this access, outside the array's bounds [0, %d)",
					param.Name(), value.ExactString(), array.Len())
				break
			}
		}
		return true
	})
}

func arrayIndexParameters(pass *Pass, fn *syntax.FuncDecl) map[*types2.Var]bool {
	params := make(map[*types2.Var]bool)
	add := func(field *syntax.Field) {
		if field == nil || field.Name == nil {
			return
		}
		v, ok := pass.Info.Defs[field.Name].(*types2.Var)
		if ok {
			if basic, ok := v.Type().Underlying().(*types2.Basic); ok && basic.Info()&types2.IsInteger != 0 {
				params[v] = true
			}
		}
	}
	add(fn.Recv)
	for _, field := range fn.Type.ParamList {
		add(field)
	}
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr, *syntax.GroupStmt:
			syntax.Inspect(n, func(child syntax.Node) bool {
				if name, ok := child.(*syntax.Name); ok {
					if v, ok := pass.Info.Uses[name].(*types2.Var); ok {
						delete(params, v)
					}
				}
				return true
			})
			return false
		case *syntax.AssignStmt:
			for _, lhs := range exprList(n.Lhs) {
				delete(params, arrayIndexVariable(pass, lhs))
			}
		case *syntax.RangeClause:
			if n.Lhs != nil {
				for _, lhs := range exprList(n.Lhs) {
					delete(params, arrayIndexVariable(pass, lhs))
				}
			}
		case *syntax.Operation:
			if n.Op == syntax.And && n.Y == nil {
				delete(params, arrayIndexVariable(pass, n.X))
			}
		}
		return true
	})
	return params
}

func arrayIndexVariable(pass *Pass, expr syntax.Expr) *types2.Var {
	name, ok := syntax.Unparen(expr).(*syntax.Name)
	if !ok {
		return nil
	}
	v, _ := pass.Info.ObjectOf(name).(*types2.Var)
	return v
}

func arrayIndexArray(expr syntax.Expr) *types2.Array {
	if _, ok := syntax.Unparen(expr).(*syntax.Name); !ok {
		return nil
	}
	if typ := TypeOf(expr); typ != nil {
		array, _ := typ.Underlying().(*types2.Array)
		return array
	}
	return nil
}

func arrayIndexWitnesses(sizes types2.Sizes, param *types2.Var, length int64) []constant.Value {
	basic := param.Type().Underlying().(*types2.Basic)
	signed := basic.Info()&types2.IsUnsigned == 0
	bits := sizes.Sizeof(param.Type()) * 8
	if signed {
		bits--
	}
	var values []constant.Value
	upper := constant.MakeInt64(length)
	if int64(constant.BitLen(upper)) <= bits {
		values = append(values, upper)
	}
	if signed {
		values = append(values, constant.MakeInt64(-1))
	}
	return values
}

type arrayIndexState uint8

const (
	arrayIndexContinue arrayIndexState = iota
	arrayIndexReached
	arrayIndexStopped
)

// A proof reaches its target only through a fully understood prefix under
// one representable input. Stopped covers both termination and uncertainty.
type arrayIndexProof struct {
	pass   *Pass
	target *syntax.IndexExpr
	param  *types2.Var
	value  constant.Value
}

func (p *arrayIndexProof) block(stmts []syntax.Stmt) arrayIndexState {
	for _, stmt := range stmts {
		if state := p.stmt(stmt); state != arrayIndexContinue {
			return state
		}
	}
	return arrayIndexContinue
}

func (p *arrayIndexProof) stmt(stmt syntax.Stmt) arrayIndexState {
	switch stmt := stmt.(type) {
	case *syntax.EmptyStmt:
		return arrayIndexContinue
	case *syntax.BlockStmt:
		return p.block(stmt.List)
	case *syntax.DeclStmt:
		return p.declarations(stmt.DeclList)
	case *syntax.AssignStmt:
		return p.assignment(stmt)
	case *syntax.IfStmt:
		return p.ifStmt(stmt)
	case *syntax.ReturnStmt:
		if stmt.Results != nil && p.expression(stmt.Results) == arrayIndexReached {
			return arrayIndexReached
		}
	}
	return arrayIndexStopped
}

func (p *arrayIndexProof) ifStmt(stmt *syntax.IfStmt) arrayIndexState {
	if stmt.Init != nil {
		if state := p.stmt(stmt.Init); state != arrayIndexContinue {
			return state
		}
	}
	if state := p.expression(stmt.Cond); state != arrayIndexContinue {
		return state
	}
	taken, known := p.truth(stmt.Cond)
	if !known {
		return arrayIndexStopped
	}
	if taken {
		return p.block(stmt.Then.List)
	}
	if stmt.Else != nil {
		return p.stmt(stmt.Else)
	}
	return arrayIndexContinue
}

func (p *arrayIndexProof) declarations(decls []syntax.Decl) arrayIndexState {
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *syntax.ConstDecl, *syntax.TypeDecl:
			continue
		case *syntax.VarDecl:
			for _, name := range decl.NameList {
				if !arrayIndexPlainType(p.pass.Info.Defs[name].Type()) {
					return arrayIndexStopped
				}
			}
			if decl.Values != nil {
				if state := p.expression(decl.Values); state != arrayIndexContinue {
					return state
				}
			}
		default:
			return arrayIndexStopped
		}
	}
	return arrayIndexContinue
}

func (p *arrayIndexProof) assignment(stmt *syntax.AssignStmt) arrayIndexState {
	if stmt.Rhs == nil || (stmt.Op != 0 && stmt.Op != syntax.Def) {
		return arrayIndexStopped
	}
	for _, lhs := range exprList(stmt.Lhs) {
		if _, ok := syntax.Unparen(lhs).(*syntax.Name); !ok && !p.pure(lhs) {
			return arrayIndexStopped
		}
	}
	if !p.pure(stmt.Rhs) {
		return arrayIndexStopped
	}
	if state := p.access(stmt.Rhs); state != arrayIndexContinue {
		return state
	}
	return p.access(stmt.Lhs)
}

func arrayIndexScalar(typ types2.Type) bool {
	if typ == nil {
		return false
	}
	basic, ok := typ.Underlying().(*types2.Basic)
	return ok && basic.Info()&types2.IsConstType != 0
}

func arrayIndexPlainType(typ types2.Type) bool {
	if arrayIndexScalar(typ) {
		return true
	}
	array, ok := typ.Underlying().(*types2.Array)
	return ok && arrayIndexScalar(array.Elem())
}

// Preflight the entire expression before visiting the target: a later call
// or another potentially panicking operand may execute before the access.
func (p *arrayIndexProof) expression(expr syntax.Expr) arrayIndexState {
	if !p.pure(expr) {
		return arrayIndexStopped
	}
	return p.access(expr)
}

func (p *arrayIndexProof) pure(expr syntax.Expr) bool {
	if value := expr.GetTypeInfo().Value; value != nil && value.Kind() != constant.Unknown {
		return true
	}
	switch expr := syntax.Unparen(expr).(type) {
	case *syntax.Name:
		switch p.pass.Info.Uses[expr].(type) {
		case *types2.Var, *types2.Const, *types2.Nil, *types2.Func:
			return true
		}
	case *syntax.IndexExpr:
		return expr == p.target && p.pure(expr.X) && p.pure(expr.Index)
	case *syntax.ListExpr:
		for _, elem := range expr.ElemList {
			if !p.pure(elem) {
				return false
			}
		}
		return true
	case *syntax.CompositeLit:
		array, ok := TypeOf(expr).Underlying().(*types2.Array)
		if !ok || !arrayIndexScalar(array.Elem()) {
			return false
		}
		for _, elem := range expr.ElemList {
			if !p.pure(elem) {
				return false
			}
		}
		return true
	case *syntax.KeyValueExpr:
		return p.pure(expr.Key) && p.pure(expr.Value)
	case *syntax.Operation:
		if !arrayIndexScalar(TypeOf(expr.X)) || !p.pure(expr.X) {
			return false
		}
		if expr.Y == nil {
			return expr.Op == syntax.Not
		}
		allowed := arrayIndexComparison(expr.Op) != token.ILLEGAL || expr.Op == syntax.AndAnd || expr.Op == syntax.OrOr
		return allowed && arrayIndexScalar(TypeOf(expr.Y)) && p.pure(expr.Y)
	}
	return false
}

func (p *arrayIndexProof) access(expr syntax.Expr) arrayIndexState {
	if value := expr.GetTypeInfo().Value; value != nil && value.Kind() != constant.Unknown {
		return arrayIndexContinue
	}
	switch expr := syntax.Unparen(expr).(type) {
	case *syntax.IndexExpr:
		if expr == p.target {
			return arrayIndexReached
		}
	case *syntax.ListExpr:
		return p.accessList(expr.ElemList)
	case *syntax.CompositeLit:
		return p.accessList(expr.ElemList)
	case *syntax.KeyValueExpr:
		return p.accessList([]syntax.Expr{expr.Key, expr.Value})
	case *syntax.Operation:
		if state := p.access(expr.X); state != arrayIndexContinue {
			return state
		}
		if expr.Y == nil {
			return arrayIndexContinue
		}
		if expr.Op == syntax.AndAnd || expr.Op == syntax.OrOr {
			left, known := p.truth(expr.X)
			if !known {
				return arrayIndexStopped
			}
			if (expr.Op == syntax.AndAnd && !left) || (expr.Op == syntax.OrOr && left) {
				return arrayIndexContinue
			}
		}
		return p.access(expr.Y)
	}
	return arrayIndexContinue
}

func (p *arrayIndexProof) accessList(exprs []syntax.Expr) arrayIndexState {
	for _, expr := range exprs {
		if state := p.access(expr); state != arrayIndexContinue {
			return state
		}
	}
	return arrayIndexContinue
}

func (p *arrayIndexProof) truth(expr syntax.Expr) (bool, bool) {
	if value := p.operand(expr); value != nil && value.Kind() == constant.Bool {
		return constant.BoolVal(value), true
	}
	op, ok := syntax.Unparen(expr).(*syntax.Operation)
	if !ok {
		return false, false
	}
	if op.Op == syntax.Not && op.Y == nil {
		value, known := p.truth(op.X)
		return !value, known
	}
	if op.Op == syntax.AndAnd || op.Op == syntax.OrOr {
		left, known := p.truth(op.X)
		if !known || (op.Op == syntax.AndAnd && !left) || (op.Op == syntax.OrOr && left) {
			return left, known
		}
		return p.truth(op.Y)
	}
	cmp := arrayIndexComparison(op.Op)
	left, right := p.operand(op.X), p.operand(op.Y)
	if cmp == token.ILLEGAL || left == nil || right == nil {
		return false, false
	}
	return constant.Compare(left, cmp, right), true
}

func (p *arrayIndexProof) operand(expr syntax.Expr) constant.Value {
	if expr == nil {
		return nil
	}
	if value := expr.GetTypeInfo().Value; value != nil && value.Kind() != constant.Unknown {
		return value
	}
	if arrayIndexVariable(p.pass, expr) == p.param {
		return p.value
	}
	return nil
}

func arrayIndexComparison(op syntax.Operator) token.Token {
	switch op {
	case syntax.Eql:
		return token.EQL
	case syntax.Neq:
		return token.NEQ
	case syntax.Lss:
		return token.LSS
	case syntax.Leq:
		return token.LEQ
	case syntax.Gtr:
		return token.GTR
	case syntax.Geq:
		return token.GEQ
	}
	return token.ILLEGAL
}
