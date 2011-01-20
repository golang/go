// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"big"
	"fmt"
	"go/ast"
	"go/token"
	"log"
	"strconv"
	"strings"
	"os"
)

var (
	idealZero = big.NewInt(0)
	idealOne  = big.NewInt(1)
)

// An expr is the result of compiling an expression.  It stores the
// type of the expression and its evaluator function.
type expr struct {
	*exprInfo
	t Type

	// Evaluate this node as the given type.
	eval interface{}

	// Map index expressions permit special forms of assignment,
	// for which we need to know the Map and key.
	evalMapValue func(t *Thread) (Map, interface{})

	// Evaluate to the "address of" this value; that is, the
	// settable Value object.  nil for expressions whose address
	// cannot be taken.
	evalAddr func(t *Thread) Value

	// Execute this expression as a statement.  Only expressions
	// that are valid expression statements should set this.
	exec func(t *Thread)

	// If this expression is a type, this is its compiled type.
	// This is only permitted in the function position of a call
	// expression.  In this case, t should be nil.
	valType Type

	// A short string describing this expression for error
	// messages.
	desc string
}

// exprInfo stores information needed to compile any expression node.
// Each expr also stores its exprInfo so further expressions can be
// compiled from it.
type exprInfo struct {
	*compiler
	pos token.Pos
}

func (a *exprInfo) newExpr(t Type, desc string) *expr {
	return &expr{exprInfo: a, t: t, desc: desc}
}

func (a *exprInfo) diag(format string, args ...interface{}) {
	a.diagAt(a.pos, format, args...)
}

func (a *exprInfo) diagOpType(op token.Token, vt Type) {
	a.diag("illegal operand type for '%v' operator\n\t%v", op, vt)
}

func (a *exprInfo) diagOpTypes(op token.Token, lt Type, rt Type) {
	a.diag("illegal operand types for '%v' operator\n\t%v\n\t%v", op, lt, rt)
}

/*
 * Common expression manipulations
 */

// a.convertTo(t) converts the value of the analyzed expression a,
// which must be a constant, ideal number, to a new analyzed
// expression with a constant value of type t.
//
// TODO(austin) Rename to resolveIdeal or something?
func (a *expr) convertTo(t Type) *expr {
	if !a.t.isIdeal() {
		log.Panicf("attempted to convert from %v, expected ideal", a.t)
	}

	var rat *big.Rat

	// XXX(Spec)  The spec says "It is erroneous".
	//
	// It is an error to assign a value with a non-zero fractional
	// part to an integer, or if the assignment would overflow or
	// underflow, or in general if the value cannot be represented
	// by the type of the variable.
	switch a.t {
	case IdealFloatType:
		rat = a.asIdealFloat()()
		if t.isInteger() && !rat.IsInt() {
			a.diag("constant %v truncated to integer", rat.FloatString(6))
			return nil
		}
	case IdealIntType:
		i := a.asIdealInt()()
		rat = new(big.Rat).SetInt(i)
	default:
		log.Panicf("unexpected ideal type %v", a.t)
	}

	// Check bounds
	if t, ok := t.lit().(BoundedType); ok {
		if rat.Cmp(t.minVal()) < 0 {
			a.diag("constant %v underflows %v", rat.FloatString(6), t)
			return nil
		}
		if rat.Cmp(t.maxVal()) > 0 {
			a.diag("constant %v overflows %v", rat.FloatString(6), t)
			return nil
		}
	}

	// Convert rat to type t.
	res := a.newExpr(t, a.desc)
	switch t := t.lit().(type) {
	case *uintType:
		n, d := rat.Num(), rat.Denom()
		f := new(big.Int).Quo(n, d)
		f = f.Abs(f)
		v := uint64(f.Int64())
		res.eval = func(*Thread) uint64 { return v }
	case *intType:
		n, d := rat.Num(), rat.Denom()
		f := new(big.Int).Quo(n, d)
		v := f.Int64()
		res.eval = func(*Thread) int64 { return v }
	case *idealIntType:
		n, d := rat.Num(), rat.Denom()
		f := new(big.Int).Quo(n, d)
		res.eval = func() *big.Int { return f }
	case *floatType:
		n, d := rat.Num(), rat.Denom()
		v := float64(n.Int64()) / float64(d.Int64())
		res.eval = func(*Thread) float64 { return v }
	case *idealFloatType:
		res.eval = func() *big.Rat { return rat }
	default:
		log.Panicf("cannot convert to type %T", t)
	}

	return res
}

// convertToInt converts this expression to an integer, if possible,
// or produces an error if not.  This accepts ideal ints, uints, and
// ints.  If max is not -1, produces an error if possible if the value
// exceeds max.  If negErr is not "", produces an error if possible if
// the value is negative.
func (a *expr) convertToInt(max int64, negErr string, errOp string) *expr {
	switch a.t.lit().(type) {
	case *idealIntType:
		val := a.asIdealInt()()
		if negErr != "" && val.Sign() < 0 {
			a.diag("negative %s: %s", negErr, val)
			return nil
		}
		bound := max
		if negErr == "slice" {
			bound++
		}
		if max != -1 && val.Cmp(big.NewInt(bound)) >= 0 {
			a.diag("index %s exceeds length %d", val, max)
			return nil
		}
		return a.convertTo(IntType)

	case *uintType:
		// Convert to int
		na := a.newExpr(IntType, a.desc)
		af := a.asUint()
		na.eval = func(t *Thread) int64 { return int64(af(t)) }
		return na

	case *intType:
		// Good as is
		return a
	}

	a.diag("illegal operand type for %s\n\t%v", errOp, a.t)
	return nil
}

// derefArray returns an expression of array type if the given
// expression is a *array type.  Otherwise, returns the given
// expression.
func (a *expr) derefArray() *expr {
	if pt, ok := a.t.lit().(*PtrType); ok {
		if _, ok := pt.Elem.lit().(*ArrayType); ok {
			deref := a.compileStarExpr(a)
			if deref == nil {
				log.Panicf("failed to dereference *array")
			}
			return deref
		}
	}
	return a
}

/*
 * Assignments
 */

// An assignCompiler compiles assignment operations.  Anything other
// than short declarations should use the compileAssign wrapper.
//
// There are three valid types of assignment:
// 1) T = T
//    Assigning a single expression with single-valued type to a
//    single-valued type.
// 2) MT = T, T, ...
//    Assigning multiple expressions with single-valued types to a
//    multi-valued type.
// 3) MT = MT
//    Assigning a single expression with multi-valued type to a
//    multi-valued type.
type assignCompiler struct {
	*compiler
	pos token.Pos
	// The RHS expressions.  This may include nil's for
	// expressions that failed to compile.
	rs []*expr
	// The (possibly unary) MultiType of the RHS.
	rmt *MultiType
	// Whether this is an unpack assignment (case 3).
	isUnpack bool
	// Whether map special assignment forms are allowed.
	allowMap bool
	// Whether this is a "r, ok = a[x]" assignment.
	isMapUnpack bool
	// The operation name to use in error messages, such as
	// "assignment" or "function call".
	errOp string
	// The name to use for positions in error messages, such as
	// "argument".
	errPosName string
}

// Type check the RHS of an assignment, returning a new assignCompiler
// and indicating if the type check succeeded.  This always returns an
// assignCompiler with rmt set, but if type checking fails, slots in
// the MultiType may be nil.  If rs contains nil's, type checking will
// fail and these expressions given a nil type.
func (a *compiler) checkAssign(pos token.Pos, rs []*expr, errOp, errPosName string) (*assignCompiler, bool) {
	c := &assignCompiler{
		compiler:   a,
		pos:        pos,
		rs:         rs,
		errOp:      errOp,
		errPosName: errPosName,
	}

	// Is this an unpack?
	if len(rs) == 1 && rs[0] != nil {
		if rmt, isUnpack := rs[0].t.(*MultiType); isUnpack {
			c.rmt = rmt
			c.isUnpack = true
			return c, true
		}
	}

	// Create MultiType for RHS and check that all RHS expressions
	// are single-valued.
	rts := make([]Type, len(rs))
	ok := true
	for i, r := range rs {
		if r == nil {
			ok = false
			continue
		}

		if _, isMT := r.t.(*MultiType); isMT {
			r.diag("multi-valued expression not allowed in %s", errOp)
			ok = false
			continue
		}

		rts[i] = r.t
	}

	c.rmt = NewMultiType(rts)
	return c, ok
}

func (a *assignCompiler) allowMapForms(nls int) {
	a.allowMap = true

	// Update unpacking info if this is r, ok = a[x]
	if nls == 2 && len(a.rs) == 1 && a.rs[0] != nil && a.rs[0].evalMapValue != nil {
		a.isUnpack = true
		a.rmt = NewMultiType([]Type{a.rs[0].t, BoolType})
		a.isMapUnpack = true
	}
}

// compile type checks and compiles an assignment operation, returning
// a function that expects an l-value and the frame in which to
// evaluate the RHS expressions.  The l-value must have exactly the
// type given by lt.  Returns nil if type checking fails.
func (a *assignCompiler) compile(b *block, lt Type) func(Value, *Thread) {
	lmt, isMT := lt.(*MultiType)
	rmt, isUnpack := a.rmt, a.isUnpack

	// Create unary MultiType for single LHS
	if !isMT {
		lmt = NewMultiType([]Type{lt})
	}

	// Check that the assignment count matches
	lcount := len(lmt.Elems)
	rcount := len(rmt.Elems)
	if lcount != rcount {
		msg := "not enough"
		pos := a.pos
		if rcount > lcount {
			msg = "too many"
			if lcount > 0 {
				pos = a.rs[lcount-1].pos
			}
		}
		a.diagAt(pos, "%s %ss for %s\n\t%s\n\t%s", msg, a.errPosName, a.errOp, lt, rmt)
		return nil
	}

	bad := false

	// If this is an unpack, create a temporary to store the
	// multi-value and replace the RHS with expressions to pull
	// out values from the temporary.  Technically, this is only
	// necessary when we need to perform assignment conversions.
	var effect func(*Thread)
	if isUnpack {
		// This leaks a slot, but is definitely safe.
		temp := b.DefineTemp(a.rmt)
		tempIdx := temp.Index
		if tempIdx < 0 {
			panic(fmt.Sprintln("tempidx", tempIdx))
		}
		if a.isMapUnpack {
			rf := a.rs[0].evalMapValue
			vt := a.rmt.Elems[0]
			effect = func(t *Thread) {
				m, k := rf(t)
				v := m.Elem(t, k)
				found := boolV(true)
				if v == nil {
					found = boolV(false)
					v = vt.Zero()
				}
				t.f.Vars[tempIdx] = multiV([]Value{v, &found})
			}
		} else {
			rf := a.rs[0].asMulti()
			effect = func(t *Thread) { t.f.Vars[tempIdx] = multiV(rf(t)) }
		}
		orig := a.rs[0]
		a.rs = make([]*expr, len(a.rmt.Elems))
		for i, t := range a.rmt.Elems {
			if t.isIdeal() {
				log.Panicf("Right side of unpack contains ideal: %s", rmt)
			}
			a.rs[i] = orig.newExpr(t, orig.desc)
			index := i
			a.rs[i].genValue(func(t *Thread) Value { return t.f.Vars[tempIdx].(multiV)[index] })
		}
	}
	// Now len(a.rs) == len(a.rmt) and we've reduced any unpacking
	// to multi-assignment.

	// TODO(austin) Deal with assignment special cases.

	// Values of any type may always be assigned to variables of
	// compatible static type.
	for i, lt := range lmt.Elems {
		rt := rmt.Elems[i]

		// When [an ideal is] (used in an expression) assigned
		// to a variable or typed constant, the destination
		// must be able to represent the assigned value.
		if rt.isIdeal() {
			a.rs[i] = a.rs[i].convertTo(lmt.Elems[i])
			if a.rs[i] == nil {
				bad = true
				continue
			}
			rt = a.rs[i].t
		}

		// A pointer p to an array can be assigned to a slice
		// variable v with compatible element type if the type
		// of p or v is unnamed.
		if rpt, ok := rt.lit().(*PtrType); ok {
			if at, ok := rpt.Elem.lit().(*ArrayType); ok {
				if lst, ok := lt.lit().(*SliceType); ok {
					if lst.Elem.compat(at.Elem, false) && (rt.lit() == Type(rt) || lt.lit() == Type(lt)) {
						rf := a.rs[i].asPtr()
						a.rs[i] = a.rs[i].newExpr(lt, a.rs[i].desc)
						len := at.Len
						a.rs[i].eval = func(t *Thread) Slice { return Slice{rf(t).(ArrayValue), len, len} }
						rt = a.rs[i].t
					}
				}
			}
		}

		if !lt.compat(rt, false) {
			if len(a.rs) == 1 {
				a.rs[0].diag("illegal operand types for %s\n\t%v\n\t%v", a.errOp, lt, rt)
			} else {
				a.rs[i].diag("illegal operand types in %s %d of %s\n\t%v\n\t%v", a.errPosName, i+1, a.errOp, lt, rt)
			}
			bad = true
		}
	}
	if bad {
		return nil
	}

	// Compile
	if !isMT {
		// Case 1
		return genAssign(lt, a.rs[0])
	}
	// Case 2 or 3
	as := make([]func(lv Value, t *Thread), len(a.rs))
	for i, r := range a.rs {
		as[i] = genAssign(lmt.Elems[i], r)
	}
	return func(lv Value, t *Thread) {
		if effect != nil {
			effect(t)
		}
		lmv := lv.(multiV)
		for i, a := range as {
			a(lmv[i], t)
		}
	}
}

// compileAssign compiles an assignment operation without the full
// generality of an assignCompiler.  See assignCompiler for a
// description of the arguments.
func (a *compiler) compileAssign(pos token.Pos, b *block, lt Type, rs []*expr, errOp, errPosName string) func(Value, *Thread) {
	ac, ok := a.checkAssign(pos, rs, errOp, errPosName)
	if !ok {
		return nil
	}
	return ac.compile(b, lt)
}

/*
 * Expression compiler
 */

// An exprCompiler stores information used throughout the compilation
// of a single expression.  It does not embed funcCompiler because
// expressions can appear at top level.
type exprCompiler struct {
	*compiler
	// The block this expression is being compiled in.
	block *block
	// Whether this expression is used in a constant context.
	constant bool
}

// compile compiles an expression AST.  callCtx should be true if this
// AST is in the function position of a function call node; it allows
// the returned expression to be a type or a built-in function (which
// otherwise result in errors).
func (a *exprCompiler) compile(x ast.Expr, callCtx bool) *expr {
	ei := &exprInfo{a.compiler, x.Pos()}

	switch x := x.(type) {
	// Literals
	case *ast.BasicLit:
		switch x.Kind {
		case token.INT:
			return ei.compileIntLit(string(x.Value))
		case token.FLOAT:
			return ei.compileFloatLit(string(x.Value))
		case token.CHAR:
			return ei.compileCharLit(string(x.Value))
		case token.STRING:
			return ei.compileStringLit(string(x.Value))
		default:
			log.Panicf("unexpected basic literal type %v", x.Kind)
		}

	case *ast.CompositeLit:
		goto notimpl

	case *ast.FuncLit:
		decl := ei.compileFuncType(a.block, x.Type)
		if decl == nil {
			// TODO(austin) Try compiling the body,
			// perhaps with dummy argument definitions
			return nil
		}
		fn := ei.compileFunc(a.block, decl, x.Body)
		if fn == nil {
			return nil
		}
		if a.constant {
			a.diagAt(x.Pos(), "function literal used in constant expression")
			return nil
		}
		return ei.compileFuncLit(decl, fn)

	// Types
	case *ast.ArrayType:
		// TODO(austin) Use a multi-type case
		goto typeexpr

	case *ast.ChanType:
		goto typeexpr

	case *ast.Ellipsis:
		goto typeexpr

	case *ast.FuncType:
		goto typeexpr

	case *ast.InterfaceType:
		goto typeexpr

	case *ast.MapType:
		goto typeexpr

	// Remaining expressions
	case *ast.BadExpr:
		// Error already reported by parser
		a.silentErrors++
		return nil

	case *ast.BinaryExpr:
		l, r := a.compile(x.X, false), a.compile(x.Y, false)
		if l == nil || r == nil {
			return nil
		}
		return ei.compileBinaryExpr(x.Op, l, r)

	case *ast.CallExpr:
		l := a.compile(x.Fun, true)
		args := make([]*expr, len(x.Args))
		bad := false
		for i, arg := range x.Args {
			if i == 0 && l != nil && (l.t == Type(makeType) || l.t == Type(newType)) {
				argei := &exprInfo{a.compiler, arg.Pos()}
				args[i] = argei.exprFromType(a.compileType(a.block, arg))
			} else {
				args[i] = a.compile(arg, false)
			}
			if args[i] == nil {
				bad = true
			}
		}
		if bad || l == nil {
			return nil
		}
		if a.constant {
			a.diagAt(x.Pos(), "function call in constant context")
			return nil
		}

		if l.valType != nil {
			a.diagAt(x.Pos(), "type conversions not implemented")
			return nil
		} else if ft, ok := l.t.(*FuncType); ok && ft.builtin != "" {
			return ei.compileBuiltinCallExpr(a.block, ft, args)
		} else {
			return ei.compileCallExpr(a.block, l, args)
		}

	case *ast.Ident:
		return ei.compileIdent(a.block, a.constant, callCtx, x.Name)

	case *ast.IndexExpr:
		l, r := a.compile(x.X, false), a.compile(x.Index, false)
		if l == nil || r == nil {
			return nil
		}
		return ei.compileIndexExpr(l, r)

	case *ast.SliceExpr:
		var lo, hi *expr
		arr := a.compile(x.X, false)
		if x.Low == nil {
			// beginning was omitted, so we need to provide it
			ei := &exprInfo{a.compiler, x.Pos()}
			lo = ei.compileIntLit("0")
		} else {
			lo = a.compile(x.Low, false)
		}
		if x.High == nil {
			// End was omitted, so we need to compute len(x.X)
			ei := &exprInfo{a.compiler, x.Pos()}
			hi = ei.compileBuiltinCallExpr(a.block, lenType, []*expr{arr})
		} else {
			hi = a.compile(x.High, false)
		}
		if arr == nil || lo == nil || hi == nil {
			return nil
		}
		return ei.compileSliceExpr(arr, lo, hi)

	case *ast.KeyValueExpr:
		goto notimpl

	case *ast.ParenExpr:
		return a.compile(x.X, callCtx)

	case *ast.SelectorExpr:
		v := a.compile(x.X, false)
		if v == nil {
			return nil
		}
		return ei.compileSelectorExpr(v, x.Sel.Name)

	case *ast.StarExpr:
		// We pass down our call context because this could be
		// a pointer type (and thus a type conversion)
		v := a.compile(x.X, callCtx)
		if v == nil {
			return nil
		}
		if v.valType != nil {
			// Turns out this was a pointer type, not a dereference
			return ei.exprFromType(NewPtrType(v.valType))
		}
		return ei.compileStarExpr(v)

	case *ast.StructType:
		goto notimpl

	case *ast.TypeAssertExpr:
		goto notimpl

	case *ast.UnaryExpr:
		v := a.compile(x.X, false)
		if v == nil {
			return nil
		}
		return ei.compileUnaryExpr(x.Op, v)
	}
	log.Panicf("unexpected ast node type %T", x)
	panic("unreachable")

typeexpr:
	if !callCtx {
		a.diagAt(x.Pos(), "type used as expression")
		return nil
	}
	return ei.exprFromType(a.compileType(a.block, x))

notimpl:
	a.diagAt(x.Pos(), "%T expression node not implemented", x)
	return nil
}

func (a *exprInfo) exprFromType(t Type) *expr {
	if t == nil {
		return nil
	}
	expr := a.newExpr(nil, "type")
	expr.valType = t
	return expr
}

func (a *exprInfo) compileIdent(b *block, constant bool, callCtx bool, name string) *expr {
	bl, level, def := b.Lookup(name)
	if def == nil {
		a.diag("%s: undefined", name)
		return nil
	}
	switch def := def.(type) {
	case *Constant:
		expr := a.newExpr(def.Type, "constant")
		if ft, ok := def.Type.(*FuncType); ok && ft.builtin != "" {
			// XXX(Spec) I don't think anything says that
			// built-in functions can't be used as values.
			if !callCtx {
				a.diag("built-in function %s cannot be used as a value", ft.builtin)
				return nil
			}
			// Otherwise, we leave the evaluators empty
			// because this is handled specially
		} else {
			expr.genConstant(def.Value)
		}
		return expr
	case *Variable:
		if constant {
			a.diag("variable %s used in constant expression", name)
			return nil
		}
		if bl.global {
			return a.compileGlobalVariable(def)
		}
		return a.compileVariable(level, def)
	case Type:
		if callCtx {
			return a.exprFromType(def)
		}
		a.diag("type %v used as expression", name)
		return nil
	}
	log.Panicf("name %s has unknown type %T", name, def)
	panic("unreachable")
}

func (a *exprInfo) compileVariable(level int, v *Variable) *expr {
	if v.Type == nil {
		// Placeholder definition from an earlier error
		a.silentErrors++
		return nil
	}
	expr := a.newExpr(v.Type, "variable")
	expr.genIdentOp(level, v.Index)
	return expr
}

func (a *exprInfo) compileGlobalVariable(v *Variable) *expr {
	if v.Type == nil {
		// Placeholder definition from an earlier error
		a.silentErrors++
		return nil
	}
	if v.Init == nil {
		v.Init = v.Type.Zero()
	}
	expr := a.newExpr(v.Type, "variable")
	val := v.Init
	expr.genValue(func(t *Thread) Value { return val })
	return expr
}

func (a *exprInfo) compileIdealInt(i *big.Int, desc string) *expr {
	expr := a.newExpr(IdealIntType, desc)
	expr.eval = func() *big.Int { return i }
	return expr
}

func (a *exprInfo) compileIntLit(lit string) *expr {
	i, _ := new(big.Int).SetString(lit, 0)
	return a.compileIdealInt(i, "integer literal")
}

func (a *exprInfo) compileCharLit(lit string) *expr {
	if lit[0] != '\'' {
		// Caught by parser
		a.silentErrors++
		return nil
	}
	v, _, tail, err := strconv.UnquoteChar(lit[1:], '\'')
	if err != nil || tail != "'" {
		// Caught by parser
		a.silentErrors++
		return nil
	}
	return a.compileIdealInt(big.NewInt(int64(v)), "character literal")
}

func (a *exprInfo) compileFloatLit(lit string) *expr {
	f, ok := new(big.Rat).SetString(lit)
	if !ok {
		log.Panicf("malformed float literal %s at %v passed parser", lit, a.pos)
	}
	expr := a.newExpr(IdealFloatType, "float literal")
	expr.eval = func() *big.Rat { return f }
	return expr
}

func (a *exprInfo) compileString(s string) *expr {
	// Ideal strings don't have a named type but they are
	// compatible with type string.

	// TODO(austin) Use unnamed string type.
	expr := a.newExpr(StringType, "string literal")
	expr.eval = func(*Thread) string { return s }
	return expr
}

func (a *exprInfo) compileStringLit(lit string) *expr {
	s, err := strconv.Unquote(lit)
	if err != nil {
		a.diag("illegal string literal, %v", err)
		return nil
	}
	return a.compileString(s)
}

func (a *exprInfo) compileStringList(list []*expr) *expr {
	ss := make([]string, len(list))
	for i, s := range list {
		ss[i] = s.asString()(nil)
	}
	return a.compileString(strings.Join(ss, ""))
}

func (a *exprInfo) compileFuncLit(decl *FuncDecl, fn func(*Thread) Func) *expr {
	expr := a.newExpr(decl.Type, "function literal")
	expr.eval = fn
	return expr
}

func (a *exprInfo) compileSelectorExpr(v *expr, name string) *expr {
	// mark marks a field that matches the selector name.  It
	// tracks the best depth found so far and whether more than
	// one field has been found at that depth.
	bestDepth := -1
	ambig := false
	amberr := ""
	mark := func(depth int, pathName string) {
		switch {
		case bestDepth == -1 || depth < bestDepth:
			bestDepth = depth
			ambig = false
			amberr = ""

		case depth == bestDepth:
			ambig = true

		default:
			log.Panicf("Marked field at depth %d, but already found one at depth %d", depth, bestDepth)
		}
		amberr += "\n\t" + pathName[1:]
	}

	visited := make(map[Type]bool)

	// find recursively searches for the named field, starting at
	// type t.  If it finds the named field, it returns a function
	// which takes an expr that represents a value of type 't' and
	// returns an expr that retrieves the named field.  We delay
	// expr construction to avoid producing lots of useless expr's
	// as we search.
	//
	// TODO(austin) Now that the expression compiler works on
	// semantic values instead of AST's, there should be a much
	// better way of doing this.
	var find func(Type, int, string) func(*expr) *expr
	find = func(t Type, depth int, pathName string) func(*expr) *expr {
		// Don't bother looking if we've found something shallower
		if bestDepth != -1 && bestDepth < depth {
			return nil
		}

		// Don't check the same type twice and avoid loops
		if visited[t] {
			return nil
		}
		visited[t] = true

		// Implicit dereference
		deref := false
		if ti, ok := t.(*PtrType); ok {
			deref = true
			t = ti.Elem
		}

		// If it's a named type, look for methods
		if ti, ok := t.(*NamedType); ok {
			_, ok := ti.methods[name]
			if ok {
				mark(depth, pathName+"."+name)
				log.Panic("Methods not implemented")
			}
			t = ti.Def
		}

		// If it's a struct type, check fields and embedded types
		var builder func(*expr) *expr
		if t, ok := t.(*StructType); ok {
			for i, f := range t.Elems {
				var sub func(*expr) *expr
				switch {
				case f.Name == name:
					mark(depth, pathName+"."+name)
					sub = func(e *expr) *expr { return e }

				case f.Anonymous:
					sub = find(f.Type, depth+1, pathName+"."+f.Name)
					if sub == nil {
						continue
					}

				default:
					continue
				}

				// We found something.  Create a
				// builder for accessing this field.
				ft := f.Type
				index := i
				builder = func(parent *expr) *expr {
					if deref {
						parent = a.compileStarExpr(parent)
					}
					expr := a.newExpr(ft, "selector expression")
					pf := parent.asStruct()
					evalAddr := func(t *Thread) Value { return pf(t).Field(t, index) }
					expr.genValue(evalAddr)
					return sub(expr)
				}
			}
		}

		return builder
	}

	builder := find(v.t, 0, "")
	if builder == nil {
		a.diag("type %v has no field or method %s", v.t, name)
		return nil
	}
	if ambig {
		a.diag("field %s is ambiguous in type %v%s", name, v.t, amberr)
		return nil
	}

	return builder(v)
}

func (a *exprInfo) compileSliceExpr(arr, lo, hi *expr) *expr {
	// Type check object
	arr = arr.derefArray()

	var at Type
	var maxIndex int64 = -1

	switch lt := arr.t.lit().(type) {
	case *ArrayType:
		at = NewSliceType(lt.Elem)
		maxIndex = lt.Len

	case *SliceType:
		at = lt

	case *stringType:
		at = lt

	default:
		a.diag("cannot slice %v", arr.t)
		return nil
	}

	// Type check index and convert to int
	// XXX(Spec) It's unclear if ideal floats with no
	// fractional part are allowed here.  6g allows it.  I
	// believe that's wrong.
	lo = lo.convertToInt(maxIndex, "slice", "slice")
	hi = hi.convertToInt(maxIndex, "slice", "slice")
	if lo == nil || hi == nil {
		return nil
	}

	expr := a.newExpr(at, "slice expression")

	// Compile
	lof := lo.asInt()
	hif := hi.asInt()
	switch lt := arr.t.lit().(type) {
	case *ArrayType:
		arrf := arr.asArray()
		bound := lt.Len
		expr.eval = func(t *Thread) Slice {
			arr, lo, hi := arrf(t), lof(t), hif(t)
			if lo > hi || hi > bound || lo < 0 {
				t.Abort(SliceError{lo, hi, bound})
			}
			return Slice{arr.Sub(lo, bound-lo), hi - lo, bound - lo}
		}

	case *SliceType:
		arrf := arr.asSlice()
		expr.eval = func(t *Thread) Slice {
			arr, lo, hi := arrf(t), lof(t), hif(t)
			if lo > hi || hi > arr.Cap || lo < 0 {
				t.Abort(SliceError{lo, hi, arr.Cap})
			}
			return Slice{arr.Base.Sub(lo, arr.Cap-lo), hi - lo, arr.Cap - lo}
		}

	case *stringType:
		arrf := arr.asString()
		// TODO(austin) This pulls over the whole string in a
		// remote setting, instead of creating a substring backed
		// by remote memory.
		expr.eval = func(t *Thread) string {
			arr, lo, hi := arrf(t), lof(t), hif(t)
			if lo > hi || hi > int64(len(arr)) || lo < 0 {
				t.Abort(SliceError{lo, hi, int64(len(arr))})
			}
			return arr[lo:hi]
		}

	default:
		log.Panicf("unexpected left operand type %T", arr.t.lit())
	}

	return expr
}

func (a *exprInfo) compileIndexExpr(l, r *expr) *expr {
	// Type check object
	l = l.derefArray()

	var at Type
	intIndex := false
	var maxIndex int64 = -1

	switch lt := l.t.lit().(type) {
	case *ArrayType:
		at = lt.Elem
		intIndex = true
		maxIndex = lt.Len

	case *SliceType:
		at = lt.Elem
		intIndex = true

	case *stringType:
		at = Uint8Type
		intIndex = true

	case *MapType:
		at = lt.Elem
		if r.t.isIdeal() {
			r = r.convertTo(lt.Key)
			if r == nil {
				return nil
			}
		}
		if !lt.Key.compat(r.t, false) {
			a.diag("cannot use %s as index into %s", r.t, lt)
			return nil
		}

	default:
		a.diag("cannot index into %v", l.t)
		return nil
	}

	// Type check index and convert to int if necessary
	if intIndex {
		// XXX(Spec) It's unclear if ideal floats with no
		// fractional part are allowed here.  6g allows it.  I
		// believe that's wrong.
		r = r.convertToInt(maxIndex, "index", "index")
		if r == nil {
			return nil
		}
	}

	expr := a.newExpr(at, "index expression")

	// Compile
	switch lt := l.t.lit().(type) {
	case *ArrayType:
		lf := l.asArray()
		rf := r.asInt()
		bound := lt.Len
		expr.genValue(func(t *Thread) Value {
			l, r := lf(t), rf(t)
			if r < 0 || r >= bound {
				t.Abort(IndexError{r, bound})
			}
			return l.Elem(t, r)
		})

	case *SliceType:
		lf := l.asSlice()
		rf := r.asInt()
		expr.genValue(func(t *Thread) Value {
			l, r := lf(t), rf(t)
			if l.Base == nil {
				t.Abort(NilPointerError{})
			}
			if r < 0 || r >= l.Len {
				t.Abort(IndexError{r, l.Len})
			}
			return l.Base.Elem(t, r)
		})

	case *stringType:
		lf := l.asString()
		rf := r.asInt()
		// TODO(austin) This pulls over the whole string in a
		// remote setting, instead of just the one character.
		expr.eval = func(t *Thread) uint64 {
			l, r := lf(t), rf(t)
			if r < 0 || r >= int64(len(l)) {
				t.Abort(IndexError{r, int64(len(l))})
			}
			return uint64(l[r])
		}

	case *MapType:
		lf := l.asMap()
		rf := r.asInterface()
		expr.genValue(func(t *Thread) Value {
			m := lf(t)
			k := rf(t)
			if m == nil {
				t.Abort(NilPointerError{})
			}
			e := m.Elem(t, k)
			if e == nil {
				t.Abort(KeyError{k})
			}
			return e
		})
		// genValue makes things addressable, but map values
		// aren't addressable.
		expr.evalAddr = nil
		expr.evalMapValue = func(t *Thread) (Map, interface{}) {
			// TODO(austin) Key check?  nil check?
			return lf(t), rf(t)
		}

	default:
		log.Panicf("unexpected left operand type %T", l.t.lit())
	}

	return expr
}

func (a *exprInfo) compileCallExpr(b *block, l *expr, as []*expr) *expr {
	// TODO(austin) Variadic functions.

	// Type check

	// XXX(Spec) Calling a named function type is okay.  I really
	// think there needs to be a general discussion of named
	// types.  A named type creates a new, distinct type, but the
	// type of that type is still whatever it's defined to.  Thus,
	// in "type Foo int", Foo is still an integer type and in
	// "type Foo func()", Foo is a function type.
	lt, ok := l.t.lit().(*FuncType)
	if !ok {
		a.diag("cannot call non-function type %v", l.t)
		return nil
	}

	// The arguments must be single-valued expressions assignment
	// compatible with the parameters of F.
	//
	// XXX(Spec) The spec is wrong.  It can also be a single
	// multi-valued expression.
	nin := len(lt.In)
	assign := a.compileAssign(a.pos, b, NewMultiType(lt.In), as, "function call", "argument")
	if assign == nil {
		return nil
	}

	var t Type
	nout := len(lt.Out)
	switch nout {
	case 0:
		t = EmptyType
	case 1:
		t = lt.Out[0]
	default:
		t = NewMultiType(lt.Out)
	}
	expr := a.newExpr(t, "function call")

	// Gather argument and out types to initialize frame variables
	vts := make([]Type, nin+nout)
	copy(vts, lt.In)
	copy(vts[nin:], lt.Out)

	// Compile
	lf := l.asFunc()
	call := func(t *Thread) []Value {
		fun := lf(t)
		fr := fun.NewFrame()
		for i, t := range vts {
			fr.Vars[i] = t.Zero()
		}
		assign(multiV(fr.Vars[0:nin]), t)
		oldf := t.f
		t.f = fr
		fun.Call(t)
		t.f = oldf
		return fr.Vars[nin : nin+nout]
	}
	expr.genFuncCall(call)

	return expr
}

func (a *exprInfo) compileBuiltinCallExpr(b *block, ft *FuncType, as []*expr) *expr {
	checkCount := func(min, max int) bool {
		if len(as) < min {
			a.diag("not enough arguments to %s", ft.builtin)
			return false
		} else if len(as) > max {
			a.diag("too many arguments to %s", ft.builtin)
			return false
		}
		return true
	}

	switch ft {
	case capType:
		if !checkCount(1, 1) {
			return nil
		}
		arg := as[0].derefArray()
		expr := a.newExpr(IntType, "function call")
		switch t := arg.t.lit().(type) {
		case *ArrayType:
			// TODO(austin) It would be nice if this could
			// be a constant int.
			v := t.Len
			expr.eval = func(t *Thread) int64 { return v }

		case *SliceType:
			vf := arg.asSlice()
			expr.eval = func(t *Thread) int64 { return vf(t).Cap }

		//case *ChanType:

		default:
			a.diag("illegal argument type for cap function\n\t%v", arg.t)
			return nil
		}
		return expr

	case copyType:
		if !checkCount(2, 2) {
			return nil
		}
		src := as[1]
		dst := as[0]
		if src.t != dst.t {
			a.diag("arguments to built-in function 'copy' must have same type\nsrc: %s\ndst: %s\n", src.t, dst.t)
			return nil
		}
		if _, ok := src.t.lit().(*SliceType); !ok {
			a.diag("src argument to 'copy' must be a slice (got: %s)", src.t)
			return nil
		}
		if _, ok := dst.t.lit().(*SliceType); !ok {
			a.diag("dst argument to 'copy' must be a slice (got: %s)", dst.t)
			return nil
		}
		expr := a.newExpr(IntType, "function call")
		srcf := src.asSlice()
		dstf := dst.asSlice()
		expr.eval = func(t *Thread) int64 {
			src, dst := srcf(t), dstf(t)
			nelems := src.Len
			if nelems > dst.Len {
				nelems = dst.Len
			}
			dst.Base.Sub(0, nelems).Assign(t, src.Base.Sub(0, nelems))
			return nelems
		}
		return expr

	case lenType:
		if !checkCount(1, 1) {
			return nil
		}
		arg := as[0].derefArray()
		expr := a.newExpr(IntType, "function call")
		switch t := arg.t.lit().(type) {
		case *stringType:
			vf := arg.asString()
			expr.eval = func(t *Thread) int64 { return int64(len(vf(t))) }

		case *ArrayType:
			// TODO(austin) It would be nice if this could
			// be a constant int.
			v := t.Len
			expr.eval = func(t *Thread) int64 { return v }

		case *SliceType:
			vf := arg.asSlice()
			expr.eval = func(t *Thread) int64 { return vf(t).Len }

		case *MapType:
			vf := arg.asMap()
			expr.eval = func(t *Thread) int64 {
				// XXX(Spec) What's the len of an
				// uninitialized map?
				m := vf(t)
				if m == nil {
					return 0
				}
				return m.Len(t)
			}

		//case *ChanType:

		default:
			a.diag("illegal argument type for len function\n\t%v", arg.t)
			return nil
		}
		return expr

	case makeType:
		if !checkCount(1, 3) {
			return nil
		}
		// XXX(Spec) What are the types of the
		// arguments?  Do they have to be ints?  6g
		// accepts any integral type.
		var lenexpr, capexpr *expr
		var lenf, capf func(*Thread) int64
		if len(as) > 1 {
			lenexpr = as[1].convertToInt(-1, "length", "make function")
			if lenexpr == nil {
				return nil
			}
			lenf = lenexpr.asInt()
		}
		if len(as) > 2 {
			capexpr = as[2].convertToInt(-1, "capacity", "make function")
			if capexpr == nil {
				return nil
			}
			capf = capexpr.asInt()
		}

		switch t := as[0].valType.lit().(type) {
		case *SliceType:
			// A new, initialized slice value for a given
			// element type T is made using the built-in
			// function make, which takes a slice type and
			// parameters specifying the length and
			// optionally the capacity.
			if !checkCount(2, 3) {
				return nil
			}
			et := t.Elem
			expr := a.newExpr(t, "function call")
			expr.eval = func(t *Thread) Slice {
				l := lenf(t)
				// XXX(Spec) What if len or cap is
				// negative?  The runtime panics.
				if l < 0 {
					t.Abort(NegativeLengthError{l})
				}
				c := l
				if capf != nil {
					c = capf(t)
					if c < 0 {
						t.Abort(NegativeCapacityError{c})
					}
					// XXX(Spec) What happens if
					// len > cap?  The runtime
					// sets cap to len.
					if l > c {
						c = l
					}
				}
				base := arrayV(make([]Value, c))
				for i := int64(0); i < c; i++ {
					base[i] = et.Zero()
				}
				return Slice{&base, l, c}
			}
			return expr

		case *MapType:
			// A new, empty map value is made using the
			// built-in function make, which takes the map
			// type and an optional capacity hint as
			// arguments.
			if !checkCount(1, 2) {
				return nil
			}
			expr := a.newExpr(t, "function call")
			expr.eval = func(t *Thread) Map {
				if lenf == nil {
					return make(evalMap)
				}
				l := lenf(t)
				return make(evalMap, l)
			}
			return expr

		//case *ChanType:

		default:
			a.diag("illegal argument type for make function\n\t%v", as[0].valType)
			return nil
		}

	case closeType, closedType:
		a.diag("built-in function %s not implemented", ft.builtin)
		return nil

	case newType:
		if !checkCount(1, 1) {
			return nil
		}

		t := as[0].valType
		expr := a.newExpr(NewPtrType(t), "new")
		expr.eval = func(*Thread) Value { return t.Zero() }
		return expr

	case panicType, printType, printlnType:
		evals := make([]func(*Thread) interface{}, len(as))
		for i, x := range as {
			evals[i] = x.asInterface()
		}
		spaces := ft == printlnType
		newline := ft != printType
		printer := func(t *Thread) {
			for i, eval := range evals {
				if i > 0 && spaces {
					print(" ")
				}
				v := eval(t)
				type stringer interface {
					String() string
				}
				switch v1 := v.(type) {
				case bool:
					print(v1)
				case uint64:
					print(v1)
				case int64:
					print(v1)
				case float64:
					print(v1)
				case string:
					print(v1)
				case stringer:
					print(v1.String())
				default:
					print("???")
				}
			}
			if newline {
				print("\n")
			}
		}
		expr := a.newExpr(EmptyType, "print")
		expr.exec = printer
		if ft == panicType {
			expr.exec = func(t *Thread) {
				printer(t)
				t.Abort(os.NewError("panic"))
			}
		}
		return expr
	}

	log.Panicf("unexpected built-in function '%s'", ft.builtin)
	panic("unreachable")
}

func (a *exprInfo) compileStarExpr(v *expr) *expr {
	switch vt := v.t.lit().(type) {
	case *PtrType:
		expr := a.newExpr(vt.Elem, "indirect expression")
		vf := v.asPtr()
		expr.genValue(func(t *Thread) Value {
			v := vf(t)
			if v == nil {
				t.Abort(NilPointerError{})
			}
			return v
		})
		return expr
	}

	a.diagOpType(token.MUL, v.t)
	return nil
}

var unaryOpDescs = make(map[token.Token]string)

func (a *exprInfo) compileUnaryExpr(op token.Token, v *expr) *expr {
	// Type check
	var t Type
	switch op {
	case token.ADD, token.SUB:
		if !v.t.isInteger() && !v.t.isFloat() {
			a.diagOpType(op, v.t)
			return nil
		}
		t = v.t

	case token.NOT:
		if !v.t.isBoolean() {
			a.diagOpType(op, v.t)
			return nil
		}
		t = BoolType

	case token.XOR:
		if !v.t.isInteger() {
			a.diagOpType(op, v.t)
			return nil
		}
		t = v.t

	case token.AND:
		// The unary prefix address-of operator & generates
		// the address of its operand, which must be a
		// variable, pointer indirection, field selector, or
		// array or slice indexing operation.
		if v.evalAddr == nil {
			a.diag("cannot take the address of %s", v.desc)
			return nil
		}

		// TODO(austin) Implement "It is illegal to take the
		// address of a function result variable" once I have
		// function result variables.

		t = NewPtrType(v.t)

	case token.ARROW:
		log.Panicf("Unary op %v not implemented", op)

	default:
		log.Panicf("unknown unary operator %v", op)
	}

	desc, ok := unaryOpDescs[op]
	if !ok {
		desc = "unary " + op.String() + " expression"
		unaryOpDescs[op] = desc
	}

	// Compile
	expr := a.newExpr(t, desc)
	switch op {
	case token.ADD:
		// Just compile it out
		expr = v
		expr.desc = desc

	case token.SUB:
		expr.genUnaryOpNeg(v)

	case token.NOT:
		expr.genUnaryOpNot(v)

	case token.XOR:
		expr.genUnaryOpXor(v)

	case token.AND:
		vf := v.evalAddr
		expr.eval = func(t *Thread) Value { return vf(t) }

	default:
		log.Panicf("Compilation of unary op %v not implemented", op)
	}

	return expr
}

var binOpDescs = make(map[token.Token]string)

func (a *exprInfo) compileBinaryExpr(op token.Token, l, r *expr) *expr {
	// Save the original types of l.t and r.t for error messages.
	origlt := l.t
	origrt := r.t

	// XXX(Spec) What is the exact definition of a "named type"?

	// XXX(Spec) Arithmetic operators: "Integer types" apparently
	// means all types compatible with basic integer types, though
	// this is never explained.  Likewise for float types, etc.
	// This relates to the missing explanation of named types.

	// XXX(Spec) Operators: "If both operands are ideal numbers,
	// the conversion is to ideal floats if one of the operands is
	// an ideal float (relevant for / and %)."  How is that
	// relevant only for / and %?  If I add an ideal int and an
	// ideal float, I get an ideal float.

	if op != token.SHL && op != token.SHR {
		// Except in shift expressions, if one operand has
		// numeric type and the other operand is an ideal
		// number, the ideal number is converted to match the
		// type of the other operand.
		if (l.t.isInteger() || l.t.isFloat()) && !l.t.isIdeal() && r.t.isIdeal() {
			r = r.convertTo(l.t)
		} else if (r.t.isInteger() || r.t.isFloat()) && !r.t.isIdeal() && l.t.isIdeal() {
			l = l.convertTo(r.t)
		}
		if l == nil || r == nil {
			return nil
		}

		// Except in shift expressions, if both operands are
		// ideal numbers and one is an ideal float, the other
		// is converted to ideal float.
		if l.t.isIdeal() && r.t.isIdeal() {
			if l.t.isInteger() && r.t.isFloat() {
				l = l.convertTo(r.t)
			} else if l.t.isFloat() && r.t.isInteger() {
				r = r.convertTo(l.t)
			}
			if l == nil || r == nil {
				return nil
			}
		}
	}

	// Useful type predicates
	// TODO(austin) CL 33668 mandates identical types except for comparisons.
	compat := func() bool { return l.t.compat(r.t, false) }
	integers := func() bool { return l.t.isInteger() && r.t.isInteger() }
	floats := func() bool { return l.t.isFloat() && r.t.isFloat() }
	strings := func() bool {
		// TODO(austin) Deal with named types
		return l.t == StringType && r.t == StringType
	}
	booleans := func() bool { return l.t.isBoolean() && r.t.isBoolean() }

	// Type check
	var t Type
	switch op {
	case token.ADD:
		if !compat() || (!integers() && !floats() && !strings()) {
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}
		t = l.t

	case token.SUB, token.MUL, token.QUO:
		if !compat() || (!integers() && !floats()) {
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}
		t = l.t

	case token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		if !compat() || !integers() {
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}
		t = l.t

	case token.SHL, token.SHR:
		// XXX(Spec) Is it okay for the right operand to be an
		// ideal float with no fractional part?  "The right
		// operand in a shift operation must be always be of
		// unsigned integer type or an ideal number that can
		// be safely converted into an unsigned integer type
		// (§Arithmetic operators)" suggests so and 6g agrees.

		if !l.t.isInteger() || !(r.t.isInteger() || r.t.isIdeal()) {
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}

		// The right operand in a shift operation must be
		// always be of unsigned integer type or an ideal
		// number that can be safely converted into an
		// unsigned integer type.
		if r.t.isIdeal() {
			r2 := r.convertTo(UintType)
			if r2 == nil {
				return nil
			}

			// If the left operand is not ideal, convert
			// the right to not ideal.
			if !l.t.isIdeal() {
				r = r2
			}

			// If both are ideal, but the right side isn't
			// an ideal int, convert it to simplify things.
			if l.t.isIdeal() && !r.t.isInteger() {
				r = r.convertTo(IdealIntType)
				if r == nil {
					log.Panicf("conversion to uintType succeeded, but conversion to idealIntType failed")
				}
			}
		} else if _, ok := r.t.lit().(*uintType); !ok {
			a.diag("right operand of shift must be unsigned")
			return nil
		}

		if l.t.isIdeal() && !r.t.isIdeal() {
			// XXX(Spec) What is the meaning of "ideal >>
			// non-ideal"?  Russ says the ideal should be
			// converted to an int.  6g propagates the
			// type down from assignments as a hint.

			l = l.convertTo(IntType)
			if l == nil {
				return nil
			}
		}

		// At this point, we should have one of three cases:
		// 1) uint SHIFT uint
		// 2) int SHIFT uint
		// 3) ideal int SHIFT ideal int

		t = l.t

	case token.LOR, token.LAND:
		if !booleans() {
			return nil
		}
		// XXX(Spec) There's no mention of *which* boolean
		// type the logical operators return.  From poking at
		// 6g, it appears to be the named boolean type, NOT
		// the type of the left operand, and NOT an unnamed
		// boolean type.

		t = BoolType

	case token.ARROW:
		// The operands in channel sends differ in type: one
		// is always a channel and the other is a variable or
		// value of the channel's element type.
		log.Panic("Binary op <- not implemented")
		t = BoolType

	case token.LSS, token.GTR, token.LEQ, token.GEQ:
		// XXX(Spec) It's really unclear what types which
		// comparison operators apply to.  I feel like the
		// text is trying to paint a Venn diagram for me,
		// which it's really pretty simple: <, <=, >, >= apply
		// only to numeric types and strings.  == and != apply
		// to everything except arrays and structs, and there
		// are some restrictions on when it applies to slices.

		if !compat() || (!integers() && !floats() && !strings()) {
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}
		t = BoolType

	case token.EQL, token.NEQ:
		// XXX(Spec) The rules for type checking comparison
		// operators are spread across three places that all
		// partially overlap with each other: the Comparison
		// Compatibility section, the Operators section, and
		// the Comparison Operators section.  The Operators
		// section should just say that operators require
		// identical types (as it does currently) except that
		// there a few special cases for comparison, which are
		// described in section X.  Currently it includes just
		// one of the four special cases.  The Comparison
		// Compatibility section and the Comparison Operators
		// section should either be merged, or at least the
		// Comparison Compatibility section should be
		// exclusively about type checking and the Comparison
		// Operators section should be exclusively about
		// semantics.

		// XXX(Spec) Comparison operators: "All comparison
		// operators apply to basic types except bools."  This
		// is very difficult to parse.  It's explained much
		// better in the Comparison Compatibility section.

		// XXX(Spec) Comparison compatibility: "Function
		// values are equal if they refer to the same
		// function." is rather vague.  It should probably be
		// similar to the way the rule for map values is
		// written: Function values are equal if they were
		// created by the same execution of a function literal
		// or refer to the same function declaration.  This is
		// *almost* but not quite waht 6g implements.  If a
		// function literals does not capture any variables,
		// then multiple executions of it will result in the
		// same closure.  Russ says he'll change that.

		// TODO(austin) Deal with remaining special cases

		if !compat() {
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}
		// Arrays and structs may not be compared to anything.
		switch l.t.(type) {
		case *ArrayType, *StructType:
			a.diagOpTypes(op, origlt, origrt)
			return nil
		}
		t = BoolType

	default:
		log.Panicf("unknown binary operator %v", op)
	}

	desc, ok := binOpDescs[op]
	if !ok {
		desc = op.String() + " expression"
		binOpDescs[op] = desc
	}

	// Check for ideal divide by zero
	switch op {
	case token.QUO, token.REM:
		if r.t.isIdeal() {
			if (r.t.isInteger() && r.asIdealInt()().Sign() == 0) ||
				(r.t.isFloat() && r.asIdealFloat()().Sign() == 0) {
				a.diag("divide by zero")
				return nil
			}
		}
	}

	// Compile
	expr := a.newExpr(t, desc)
	switch op {
	case token.ADD:
		expr.genBinOpAdd(l, r)

	case token.SUB:
		expr.genBinOpSub(l, r)

	case token.MUL:
		expr.genBinOpMul(l, r)

	case token.QUO:
		expr.genBinOpQuo(l, r)

	case token.REM:
		expr.genBinOpRem(l, r)

	case token.AND:
		expr.genBinOpAnd(l, r)

	case token.OR:
		expr.genBinOpOr(l, r)

	case token.XOR:
		expr.genBinOpXor(l, r)

	case token.AND_NOT:
		expr.genBinOpAndNot(l, r)

	case token.SHL:
		if l.t.isIdeal() {
			lv := l.asIdealInt()()
			rv := r.asIdealInt()()
			const maxShift = 99999
			if rv.Cmp(big.NewInt(maxShift)) > 0 {
				a.diag("left shift by %v; exceeds implementation limit of %v", rv, maxShift)
				expr.t = nil
				return nil
			}
			val := new(big.Int).Lsh(lv, uint(rv.Int64()))
			expr.eval = func() *big.Int { return val }
		} else {
			expr.genBinOpShl(l, r)
		}

	case token.SHR:
		if l.t.isIdeal() {
			lv := l.asIdealInt()()
			rv := r.asIdealInt()()
			val := new(big.Int).Rsh(lv, uint(rv.Int64()))
			expr.eval = func() *big.Int { return val }
		} else {
			expr.genBinOpShr(l, r)
		}

	case token.LSS:
		expr.genBinOpLss(l, r)

	case token.GTR:
		expr.genBinOpGtr(l, r)

	case token.LEQ:
		expr.genBinOpLeq(l, r)

	case token.GEQ:
		expr.genBinOpGeq(l, r)

	case token.EQL:
		expr.genBinOpEql(l, r)

	case token.NEQ:
		expr.genBinOpNeq(l, r)

	case token.LAND:
		expr.genBinOpLogAnd(l, r)

	case token.LOR:
		expr.genBinOpLogOr(l, r)

	default:
		log.Panicf("Compilation of binary op %v not implemented", op)
	}

	return expr
}

// TODO(austin) This is a hack to eliminate a circular dependency
// between type.go and expr.go
func (a *compiler) compileArrayLen(b *block, expr ast.Expr) (int64, bool) {
	lenExpr := a.compileExpr(b, true, expr)
	if lenExpr == nil {
		return 0, false
	}

	// XXX(Spec) Are ideal floats with no fractional part okay?
	if lenExpr.t.isIdeal() {
		lenExpr = lenExpr.convertTo(IntType)
		if lenExpr == nil {
			return 0, false
		}
	}

	if !lenExpr.t.isInteger() {
		a.diagAt(expr.Pos(), "array size must be an integer")
		return 0, false
	}

	switch lenExpr.t.lit().(type) {
	case *intType:
		return lenExpr.asInt()(nil), true
	case *uintType:
		return int64(lenExpr.asUint()(nil)), true
	}
	log.Panicf("unexpected integer type %T", lenExpr.t)
	return 0, false
}

func (a *compiler) compileExpr(b *block, constant bool, expr ast.Expr) *expr {
	ec := &exprCompiler{a, b, constant}
	nerr := a.numError()
	e := ec.compile(expr, false)
	if e == nil && nerr == a.numError() {
		log.Panicf("expression compilation failed without reporting errors")
	}
	return e
}

// extractEffect separates out any effects that the expression may
// have, returning a function that will perform those effects and a
// new exprCompiler that is guaranteed to be side-effect free.  These
// are the moral equivalents of "temp := expr" and "temp" (or "temp :=
// &expr" and "*temp" for addressable exprs).  Because this creates a
// temporary variable, the caller should create a temporary block for
// the compilation of this expression and the evaluation of the
// results.
func (a *expr) extractEffect(b *block, errOp string) (func(*Thread), *expr) {
	// Create "&a" if a is addressable
	rhs := a
	if a.evalAddr != nil {
		rhs = a.compileUnaryExpr(token.AND, rhs)
	}

	// Create temp
	ac, ok := a.checkAssign(a.pos, []*expr{rhs}, errOp, "")
	if !ok {
		return nil, nil
	}
	if len(ac.rmt.Elems) != 1 {
		a.diag("multi-valued expression not allowed in %s", errOp)
		return nil, nil
	}
	tempType := ac.rmt.Elems[0]
	if tempType.isIdeal() {
		// It's too bad we have to duplicate this rule.
		switch {
		case tempType.isInteger():
			tempType = IntType
		case tempType.isFloat():
			tempType = Float64Type
		default:
			log.Panicf("unexpected ideal type %v", tempType)
		}
	}
	temp := b.DefineTemp(tempType)
	tempIdx := temp.Index

	// Create "temp := rhs"
	assign := ac.compile(b, tempType)
	if assign == nil {
		log.Panicf("compileAssign type check failed")
	}

	effect := func(t *Thread) {
		tempVal := tempType.Zero()
		t.f.Vars[tempIdx] = tempVal
		assign(tempVal, t)
	}

	// Generate "temp" or "*temp"
	getTemp := a.compileVariable(0, temp)
	if a.evalAddr == nil {
		return effect, getTemp
	}

	deref := a.compileStarExpr(getTemp)
	if deref == nil {
		return nil, nil
	}
	return effect, deref
}
