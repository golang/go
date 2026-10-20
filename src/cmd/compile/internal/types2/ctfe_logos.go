// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

// This file implements the M5-S2 types2 hooks for const composites and
// const func calls (SPEC Sec 2.1, T-201): unwrapping ConstFuncDecl into
// the ordinary function path, evaluating const initializers of
// composite type or const-func calls, and recording values for
// lowering. The Host adapter lives in ctfe_host_logos.go; the static
// purity check lives in ctfe_purity_logos.go.

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"go/constant"
	"go/token"
	. "internal/types/errors"
	"strconv"
)

// logosConstFuncBodies records plain package-level function bodies by
// name for transitive const evaluation, and logosConstFuncSyntax marks
// the inner FuncDecls declared through `const func`.
func (check *Checker) logosConstFuncBodies() (map[string]*syntax.FuncDecl, map[*syntax.FuncDecl]bool) {
	if check.logosBodies == nil {
		check.logosBodies = map[string]*syntax.FuncDecl{}
		check.logosConstSyntax = map[*syntax.FuncDecl]bool{}
	}
	return check.logosBodies, check.logosConstSyntax
}

// logosUnwrapConstFunc normalizes a file-scope declaration for the
// resolver loop: a ConstFuncDecl contributes its inner FuncDecl to the
// ordinary path while recording constness; plain FuncDecls record
// their bodies for transitive const evaluation. Every other
// declaration passes through untouched.
func (check *Checker) logosUnwrapConstFunc(decl syntax.Decl) syntax.Decl {
	if !check.conf.Logos {
		return decl
	}
	bodies, consts := check.logosConstFuncBodies()
	if cfd, ok := decl.(*syntax.ConstFuncDecl); ok {
		consts[cfd.Func] = true
		if cfd.Func.Recv == nil && cfd.Func.Name != nil {
			if _, dup := bodies[cfd.Func.Name.Value]; !dup {
				bodies[cfd.Func.Name.Value] = cfd.Func
			}
		}
		return cfd.Func
	}
	if fd, ok := decl.(*syntax.FuncDecl); ok {
		if fd.Recv == nil && fd.Name != nil {
			if _, dup := bodies[fd.Name.Value]; !dup {
				bodies[fd.Name.Value] = fd
			}
		}
	}
	return decl
}

// logosConstKind classifies a const initializer.
type logosConstKind int

const (
	logosConstOther logosConstKind = iota
	logosConstComposite
	logosConstCall
	logosConstAlias
	logosConstSelect
	logosConstIndex
	logosConstCoalesce
)

// logosClassifyInit reports how a const initializer evaluates.
func logosClassifyInit(init syntax.Expr) logosConstKind {
	switch syntax.Unparen(init).(type) {
	case *syntax.CompositeLit:
		return logosConstComposite
	case *syntax.CallExpr:
		return logosConstCall
	case *syntax.Name:
		return logosConstAlias
	case *syntax.SelectorExpr:
		return logosConstSelect
	case *syntax.IndexExpr:
		return logosConstIndex
	case *syntax.CoalesceExpr:
		return logosConstCoalesce
	}
	return logosConstOther
}

// logosConstEval evaluates a const initializer that upstream cannot:
// composite literals, calls to const funcs (and builtins over const
// composites), and aliases of const composites. Scalar results become
// real constants through the ordinary initConst path; composite results
// are recorded in Info.LogosCTFE for lowering. It reports whether it
// handled the declaration (including failures, which invalidate the
// object to avoid follow-on errors).
func (check *Checker) logosConstEval(obj *Const, init syntax.Expr, x *operand) bool {
	if !check.conf.Logos || init == nil || !x.isValid() {
		return false
	}
	if x.mode() == constant_ {
		return false
	}
	switch logosClassifyInit(init) {
	case logosConstAlias:
		return check.logosConstAlias(obj, init)
	case logosConstIndex:
		// Map indexes yield V?, whose M3-S6 lowering keys off the
		// read shape that literalization would erase; they stay
		// with the ordinary "not constant" error.
		if logosAsOption(x.typ()) != nil {
			return false
		}
		return check.logosConstCompute(obj, init, x)
	case logosConstComposite, logosConstCall, logosConstSelect, logosConstCoalesce:
		return check.logosConstCompute(obj, init, x)
	}
	return false
}

// logosConstUse reads a recorded const composite as a value operand,
// keeping unknown-valued composites out of the constant invariant
// (which only basic const types satisfy). Recorded scalars keep their
// constant behavior. It reports whether it handled the use.
func (check *Checker) logosConstUse(obj *Const, x *operand) bool {
	if !check.conf.Logos {
		return false
	}
	v, ok := check.logosRecorded(obj)
	if !ok {
		return false
	}
	return isCompositeValue(v)
}

// logosConstAlias copies a recorded const composite value.
func (check *Checker) logosConstAlias(obj *Const, init syntax.Expr) bool {
	name, ok := syntax.Unparen(init).(*syntax.Name)
	if !ok {
		return false
	}
	ref, ok := check.Info.Uses[name].(*Const)
	if !ok {
		return false
	}
	if ref.val != nil && ref.val.Kind() != constant.Unknown {
		// A scalar constant: the ordinary path handles it.
		return false
	}
	v, ok := check.logosRecorded(ref)
	if !ok {
		return false
	}
	check.logosRecord(obj, v)
	if obj.typ == nil {
		obj.typ = ref.typ
	}
	return true
}

// logosConstCompute runs the interpreter over a composite or call
// initializer.
func (check *Checker) logosConstCompute(obj *Const, init syntax.Expr, x *operand) bool {
	if v, ok := check.logosMemo(init); ok {
		return check.logosApplyMemo(obj, init, x, v)
	}
	host := &ctfeHost{check: check}
	cfg := ctfe.Config{MaxSteps: check.conf.LogosCTFEMaxSteps, WordSize: check.logosWordSize()}
	v, _, err := ctfe.Eval(host, cfg, init)
	if err != nil {
		return check.logosCTFEFailed(obj, init, err)
	}
	check.logosMemoize(init, v)
	// Every evaluated value records for lowering's literal splice
	// (scalars keep flowing through initConst for typing below).
	check.logosRecord(obj, v)
	if isCompositeValue(v) || !isConstType(x.typ()) {
		// Composite values and values of non-basic const type
		// (such as option results of map indexes) record for
		// lowering instead of flowing through initConst, whose
		// constant invariant covers basic types only.
		return check.logosRecordComposite(obj, init, x, v)
	}
	return check.logosApplyScalar(obj, init, x, v)
}

// logosApplyMemo replays a memoized initializer result.
func (check *Checker) logosApplyMemo(obj *Const, init syntax.Expr, x *operand, v ctfe.Value) bool {
	if !v.IsValid() {
		obj.typ = Typ[Invalid]
		return true
	}
	check.logosRecord(obj, v)
	if isCompositeValue(v) || !isConstType(x.typ()) {
		return check.logosRecordComposite(obj, init, x, v)
	}
	return check.logosApplyScalar(obj, init, x, v)
}

// logosApplyScalar converts an evaluated scalar into a constant
// operand for the ordinary initConst path.
func (check *Checker) logosApplyScalar(obj *Const, init syntax.Expr, x *operand, v ctfe.Value) bool {
	c, ok := ctfeScalarConstant(v)
	if !ok {
		check.errorf(obj, InvalidConstInit, "%s value of %s is not constant", logosConstEvalRule, x)
		obj.typ = Typ[Invalid]
		return true
	}
	x.val = c
	x.mode_ = constant_
	return false
}

// logosRecordComposite records a composite value and finishes the
// declaration without initConst (whose constant requirement cannot
// hold). A declared type must accept the value's Go type.
func (check *Checker) logosRecordComposite(obj *Const, init syntax.Expr, x *operand, v ctfe.Value) bool {
	if obj.typ != nil {
		if ok, _ := x.assignableTo(check, obj.typ, nil); !ok {
			check.errorf(obj, InvalidConstInit, "%s cannot use %s as %s value in constant declaration", logosConstEvalRule, x, obj.typ)
			obj.typ = Typ[Invalid]
			return true
		}
	} else {
		obj.typ = x.typ()
	}
	if mt, ok := obj.typ.Underlying().(*Map); ok && v.Type().Kind == ctfe.KMap {
		// A const map is a const view of its map type (M5-S4, SPEC T-Sec9).
		if !logosConstMapKeyOK(mt.Key()) {
			check.errorf(obj, InvalidConstInit, "[comptime::constmap_key_type] const map key type %s cannot be hashed: use a string, number, bool, or a struct or array of them", mt.Key())
			obj.typ = Typ[Invalid]
			return true
		}
		obj.typ = &ConstView{Elem: obj.typ, ConstMap: true}
	}
	check.logosRecord(obj, v)
	return true
}

// logosCTFEFailed reports an evaluation failure once per initializer;
// later names sharing the initializer invalidate silently.
func (check *Checker) logosCTFEFailed(obj *Const, init syntax.Expr, err error) bool {
	check.logosMemoizeFailure(init)
	if cerr, ok := err.(*ctfe.Error); ok {
		switch cerr.Code {
		case ctfe.CodeSilent:
			// The impurity sits in a const func body; the static
			// purity check reports it.
			obj.typ = Typ[Invalid]
			return true
		case ctfe.CodeStepLimit:
			check.errorf(obj, InvalidConstInit, "%s evaluation exceeds the step limit", logosStepLimitRule)
		case ctfe.CodeTooLarge:
			check.errorf(obj, InvalidConstInit, "%s %s", logosTooLargeRule, cerr.Detail)
		default:
			check.errorf(obj, InvalidConstInit, "%s %s", logosConstEvalRule, cerr.Detail)
		}
		obj.typ = Typ[Invalid]
		return true
	}
	check.errorf(obj, InvalidConstInit, "%s %s", logosConstEvalRule, err)
	obj.typ = Typ[Invalid]
	return true
}

// logosRecord stores an evaluated value for lowering.
func (check *Checker) logosRecord(obj *Const, v ctfe.Value) {
	if check.Info.LogosCTFE == nil {
		check.Info.LogosCTFE = map[*Const]ctfe.Value{}
	}
	check.Info.LogosCTFE[obj] = v
}

// logosRecorded returns a recorded value.
func (check *Checker) logosRecorded(obj *Const) (ctfe.Value, bool) {
	v, ok := check.Info.LogosCTFE[obj]
	return v, ok
}

// logosWordSize returns the target int width in bytes.
func (check *Checker) logosWordSize() int64 {
	if check.conf.Sizes != nil {
		if w := check.conf.Sizes.Sizeof(Typ[Int]); w == 4 || w == 8 {
			return w
		}
	}
	return 8
}

// logosMemo caches one evaluation per initializer expression.
func (check *Checker) logosMemo(init syntax.Expr) (ctfe.Value, bool) {
	if check.logosMemoized == nil {
		return ctfe.Value{}, false
	}
	m, ok := check.logosMemoized[init]
	if !ok {
		return ctfe.Value{}, false
	}
	return m.val, true
}

// logosMemoize stores a successful evaluation.
func (check *Checker) logosMemoize(init syntax.Expr, v ctfe.Value) {
	if check.logosMemoized == nil {
		check.logosMemoized = map[syntax.Expr]ctfeMemo{}
	}
	check.logosMemoized[init] = ctfeMemo{val: v}
}

// logosMemoizeFailure stores a failed evaluation.
func (check *Checker) logosMemoizeFailure(init syntax.Expr) {
	if check.logosMemoized == nil {
		check.logosMemoized = map[syntax.Expr]ctfeMemo{}
	}
	check.logosMemoized[init] = ctfeMemo{}
}

// ctfeMemo is one memoized initializer result: an invalid value marks
// a reported failure.
type ctfeMemo struct {
	val ctfe.Value
}

const (
	logosConstEvalRule = "[correctness::const_eval]"
	logosStepLimitRule = "[comptime::ctfe_step_limit]"
	logosTooLargeRule  = "[comptime::const_too_large]"
	logosImpureRule    = "[correctness::const_func_impure]"
)

// isCompositeValue reports whether v needs a composite literal in
// lowering (and a var declaration instead of a const).
func isCompositeValue(v ctfe.Value) bool {
	t := v.Type()
	if t == nil {
		return false
	}
	switch t.Kind {
	case ctfe.KSlice, ctfe.KArray, ctfe.KMap, ctfe.KStruct:
		return true
	}
	return false
}

// ctfeScalarConstant converts an evaluated scalar to a constant.Value
// for the ordinary initConst path.
func ctfeScalarConstant(v ctfe.Value) (constant.Value, bool) {
	t := v.Type()
	if t == nil {
		return nil, false
	}
	switch t.Kind {
	case ctfe.KBool:
		return constant.MakeBool(v.Bool()), true
	case ctfe.KString:
		return constant.MakeString(v.Str()), true
	case ctfe.KInt, ctfe.KInt8, ctfe.KInt16, ctfe.KInt32, ctfe.KInt64,
		ctfe.KUint, ctfe.KUint8, ctfe.KUint16, ctfe.KUint32, ctfe.KUint64, ctfe.KUintptr,
		ctfe.KUntypedInt, ctfe.KUntypedRune:
		return constant.MakeFromLiteral(v.BigInt().String(), token.INT, 0), true
	case ctfe.KFloat32, ctfe.KFloat64, ctfe.KUntypedFloat:
		return constant.MakeFromLiteral(strconv.FormatFloat(v.Float64(), 'g', -1, 64), token.FLOAT, 0), true
	}
	return nil, false
}
