// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

// This file implements M5-S2 calls, loops, switches, and builtins
// (SPEC Sec 2.1, T-201).

import (
	"cmd/compile/internal/syntax"
	"math/big"
)

// evalCallMulti evaluates a call, reporting whether it produced several
// values (a multi-valued call). Single-valued calls return one value.
func (e *evaluator) evalCallMulti(fr *frame, c *syntax.CallExpr) ([]Value, bool, error) {
	if err := e.step(posOf(c)); err != nil {
		return nil, false, err
	}
	// Method calls never evaluate in const context.
	if sel, ok := syntax.Unparen(c.Fun).(*syntax.SelectorExpr); ok {
		if name, ok := sel.X.(*syntax.Name); ok {
			if ent := e.h.Resolve(name); ent.Kind == EntPackage {
				return e.entityCall(fr, e.h.ResolveSel(sel), c)
			}
		}
		return nil, false, e.impurity(fr, posOf(c), "method call in const func")
	}
	// Conversions carry a type in function position.
	if name, ok := syntax.Unparen(c.Fun).(*syntax.Name); ok {
		ent := e.h.Resolve(name)
		if ent.Kind == EntType {
			v, err := e.evalConversion(fr, c)
			if err != nil {
				return nil, false, err
			}
			return []Value{v}, false, nil
		}
		return e.entityCall(fr, ent, c)
	}
	if _, ok := syntax.Unparen(c.Fun).(*syntax.IndexExpr); ok {
		return nil, false, e.impurity(fr, posOf(c), "generic call in const func")
	}
	if par, ok := syntax.Unparen(c.Fun).(*syntax.ParenExpr); ok {
		inner := *c
		inner.Fun = par.X
		return e.evalCallMulti(fr, &inner)
	}
	if lit, ok := syntax.Unparen(c.Fun).(*syntax.FuncLit); ok {
		_ = lit
		return nil, false, e.impurity(fr, posOf(c), "function literal in const func")
	}
	return nil, false, e.fail(posOf(c), "unsupported call in const evaluation")
}

// entityCall dispatches a resolved call target.
func (e *evaluator) entityCall(fr *frame, ent Entity, c *syntax.CallExpr) ([]Value, bool, error) {
	switch ent.Kind {
	case EntBuiltin:
		v, err := e.evalBuiltin(fr, ent.Builtin, c)
		if err != nil {
			return nil, false, err
		}
		return []Value{v}, false, nil
	case EntConstFunc, EntPlainFunc:
		return e.evalUserCall(fr, ent.Fn, c)
	case EntImport:
		return nil, false, e.impurity(fr, posOf(c), "call to imported function %s in const func", ent.Name)
	case EntPackageVar:
		return nil, false, e.impurity(fr, posOf(c), "call to package variable %s in const func", ent.Name)
	}
	return nil, false, e.fail(posOf(c), "unsupported call in const evaluation")
}

// evalUserCall binds arguments and runs a function body.
func (e *evaluator) evalUserCall(fr *frame, fn *Func, c *syntax.CallExpr) ([]Value, bool, error) {
	if fn.Generic {
		return nil, false, e.impurity(fr, posOf(c), "generic call in const func")
	}
	if fn.Body == nil {
		return nil, false, e.fail(posOf(c), "call to function %s without body in const evaluation", fn.Name)
	}
	args, err := e.evalArgs(fr, c)
	if err != nil {
		return nil, false, err
	}
	params := fn.Params
	if fn.Variadic {
		if len(args) < len(params)-1 {
			return nil, false, e.fail(posOf(c), "too few arguments in call to %s", fn.Name)
		}
	} else if len(args) != len(params) {
		return nil, false, e.fail(posOf(c), "wrong number of arguments in call to %s", fn.Name)
	}
	nfr := &frame{fn: fn, scopes: []map[string]Value{{}}}
	e.frames = append(e.frames, nfr)
	defer func() { e.frames = e.frames[:len(e.frames)-1] }()
	for i, p := range params {
		if fn.Variadic && i == len(params)-1 {
			elems := make([]Value, 0, len(args)-i)
			for _, a := range args[i:] {
				elems = append(elems, copyArg(a))
			}
			var elemT *Type
			if p.Type != nil && p.Type.Kind == KSlice {
				elemT = p.Type.Elem
			}
			e.declare(nfr, p.Decl.Value, sliceValue(elemT, elems, len(elems)))
			break
		}
		v := copyArg(args[i])
		if p.Type != nil {
			if v, err = e.convert(v, p.Type, posOf(c)); err != nil {
				return nil, false, err
			}
		}
		e.declare(nfr, p.Decl.Value, withVar(v))
	}
	// Named results start as zero values.
	for i, name := range fn.ResultNames {
		if name == "" || i >= len(fn.Results) || fn.Results[i] == nil {
			continue
		}
		e.declare(nfr, name, zeroValue(fn.Results[i]))
	}
	if err := e.step(posOf(c)); err != nil {
		return nil, false, err
	}
	e.depth++
	f, err := e.execBlock(nfr, fn.Body)
	e.depth--
	if err != nil {
		return nil, false, err
	}
	if !f.returned {
		if len(fn.Results) == 0 {
			return nil, false, nil
		}
		// Fell off the end with named results: read them back.
		vals := make([]Value, 0, len(fn.Results))
		for _, name := range fn.ResultNames {
			if name == "" {
				return nil, false, e.fail(posOf(c), "missing return values in const evaluation")
			}
			v, ok := e.lookup(nfr, name)
			if !ok {
				return nil, false, e.fail(posOf(c), "missing return values in const evaluation")
			}
			vals = append(vals, v)
		}
		return vals, len(vals) > 1, nil
	}
	return f.values, len(f.values) > 1, nil
}

// evalArgs evaluates call arguments, spreading a trailing slice or
// string over ... .
func (e *evaluator) evalArgs(fr *frame, c *syntax.CallExpr) ([]Value, error) {
	var out []Value
	for i, a := range c.ArgList {
		v, err := e.evalExpr(fr, a)
		if err != nil {
			return nil, err
		}
		if c.HasDots && i == len(c.ArgList)-1 {
			if v.typ != nil && v.typ.Kind == KString {
				for _, r := range []byte(v.sval) {
					out = append(out, byteValue(r))
				}
				continue
			}
			if v.slice == nil {
				return nil, e.fail(posOf(a), "cannot spread non-slice in const evaluation")
			}
			out = append(out, sliceElems(v)...)
			continue
		}
		out = append(out, v)
	}
	return out, nil
}

// execFor runs classic, condition-only, infinite, and range loops.
func (e *evaluator) execFor(fr *frame, s *syntax.ForStmt) (flow, error) {
	if rc, ok := s.Init.(*syntax.RangeClause); ok {
		return e.execRange(fr, s, rc)
	}
	e.pushScope(fr)
	defer e.popScope(fr)
	if s.Init != nil {
		f, err := e.execSimple(fr, s.Init)
		if err != nil {
			return flow{}, err
		}
		if f.returned || f.brk != "" || f.cont != "" {
			return f, nil
		}
	}
	for {
		if s.Cond != nil {
			cond, err := e.evalExpr(fr, s.Cond)
			if err != nil {
				return flow{}, err
			}
			b, err := e.truth(cond, posOf(s.Cond))
			if err != nil {
				return flow{}, err
			}
			if !b {
				return flow{}, nil
			}
		}
		if err := e.step(posOf(s)); err != nil {
			return flow{}, err
		}
		f, err := e.execBlock(fr, s.Body)
		if err != nil {
			return flow{}, err
		}
		if f.returned {
			return f, nil
		}
		if f.brk != "" {
			if f.brk == unlabeled {
				return flow{}, nil
			}
			return f, nil
		}
		if f.cont != "" && f.cont != unlabeled {
			return f, nil
		}
		if s.Post != nil {
			pf, err := e.execSimple(fr, s.Post)
			if err != nil {
				return flow{}, err
			}
			if pf.returned || pf.brk != "" || pf.cont != "" {
				return pf, nil
			}
		}
	}
}

// execRange runs for-range over slices, arrays, strings, and integers.
// Ranging over a map is impure (nondeterministic order).
func (e *evaluator) execRange(fr *frame, s *syntax.ForStmt, rc *syntax.RangeClause) (flow, error) {
	xv, err := e.evalExpr(fr, rc.X)
	if err != nil {
		return flow{}, err
	}
	if xv.typ != nil && xv.typ.Kind == KMap {
		return flow{}, e.impurity(fr, posOf(rc.X), "range over map in const func")
	}
	e.pushScope(fr)
	defer e.popScope(fr)
	lhs := unpackList(rc.Lhs)
	for _, le := range lhs {
		if _, ok := syntax.Unparen(le).(*syntax.Name); !ok {
			return flow{}, e.fail(posOf(le), "unsupported range variables in const evaluation")
		}
	}
	setVar := func(name string, val Value) {
		if name == "_" {
			return
		}
		if rc.Def {
			fr.scopes[len(fr.scopes)-1][name] = withVar(val)
		} else {
			e.assignName(fr, name, val)
		}
	}
	// loop runs body once per element with the given index/value.
	// A second loop variable binds only when the clause names two.
	loop := func(idx func(i int) Value, val func(i int) Value, n int) (flow, error) {
		for i := range n {
			if err := e.step(posOf(s)); err != nil {
				return flow{}, err
			}
			if len(lhs) > 0 {
				if n0, ok := syntax.Unparen(lhs[0]).(*syntax.Name); ok {
					setVar(n0.Value, withVar(idx(i)))
				}
			}
			if len(lhs) > 1 {
				if n1, ok := syntax.Unparen(lhs[1]).(*syntax.Name); ok {
					setVar(n1.Value, withVar(val(i)))
				}
			}
			f, err := e.execBlock(fr, s.Body)
			if err != nil {
				return flow{}, err
			}
			if f.returned {
				return f, nil
			}
			if f.brk != "" {
				if f.brk == unlabeled {
					return flow{}, nil
				}
				return f, nil
			}
			if f.cont != "" && f.cont != unlabeled {
				return f, nil
			}
		}
		return flow{}, nil
	}
	intIdx := func(i int) Value { return intValue(big.NewInt(int64(i)), &Type{Kind: KInt}, false) }
	switch {
	case xv.slice != nil:
		elems := sliceElems(xv)
		return loop(intIdx, func(i int) Value { return elems[i] }, len(elems))
	case xv.array != nil:
		return loop(intIdx, func(i int) Value { return xv.array[i] }, len(xv.array))
	case xv.typ != nil && xv.typ.Kind == KString:
		runes := []rune(xv.sval)
		offs := make([]int, len(runes))
		off := 0
		for i, r := range runes {
			offs[i] = off
			off += len(string(r))
		}
		return loop(func(i int) Value {
			return intValue(big.NewInt(int64(offs[i])), &Type{Kind: KInt}, false)
		}, func(i int) Value {
			return runeValue(runes[i])
		}, len(runes))
	case xv.typ != nil && isInteger(xv.typ.Kind):
		n := xv.ival.Int64()
		if n < 0 {
			return flow{}, e.fail(posOf(rc.X), "range over negative integer in const evaluation")
		}
		return loop(intIdx, intIdx, int(n))
	}
	return flow{}, e.fail(posOf(rc.X), "cannot range in const evaluation")
}

// assignName assigns to a visible binding.
func (e *evaluator) assignName(fr *frame, name string, v Value) {
	for i := len(fr.scopes) - 1; i >= 0; i-- {
		if _, ok := fr.scopes[i][name]; ok {
			fr.scopes[i][name] = v
			return
		}
	}
	fr.scopes[len(fr.scopes)-1][name] = v
}

// execSwitch runs expression switches.
func (e *evaluator) execSwitch(fr *frame, s *syntax.SwitchStmt) (flow, error) {
	e.pushScope(fr)
	defer e.popScope(fr)
	if s.Init != nil {
		f, err := e.execSimple(fr, s.Init)
		if err != nil {
			return flow{}, err
		}
		if f.returned || f.brk != "" || f.cont != "" {
			return f, nil
		}
	}
	var tag Value
	hasTag := s.Tag != nil
	if hasTag {
		if _, ok := syntax.Unparen(s.Tag).(*syntax.TypeSwitchGuard); ok {
			return flow{}, e.fail(posOf(s.Tag), "type switch in const func not supported")
		}
		var err error
		tag, err = e.evalExpr(fr, s.Tag)
		if err != nil {
			return flow{}, err
		}
	} else {
		tag = Value{typ: &Type{Kind: KBool}, bval: true, constant: true}
	}
	falling := false
	for _, cc := range s.Body {
		matched := falling
		falling = false
		if !matched {
			if cc.Cases == nil {
				matched = true
			} else {
				for _, cx := range unpackList(cc.Cases) {
					cv, err := e.evalExpr(fr, cx)
					if err != nil {
						return flow{}, err
					}
					eq, err := e.equal(tag, cv, posOf(cx))
					if err != nil {
						return flow{}, err
					}
					if eq {
						matched = true
						break
					}
				}
			}
		}
		if !matched {
			continue
		}
		f, err := e.execBlock(fr, &syntax.BlockStmt{List: cc.Body})
		if err != nil {
			return flow{}, err
		}
		if f.returned {
			return f, nil
		}
		if f.fall {
			falling = true
			continue
		}
		if f.brk != "" {
			if f.brk == unlabeled {
				return flow{}, nil
			}
			return f, nil
		}
		if f.cont != "" {
			return f, nil
		}
		return flow{}, nil
	}
	return flow{}, nil
}
