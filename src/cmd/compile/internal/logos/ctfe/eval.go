// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"math/big"
)

type frame struct {
	fn     *Func
	scopes []map[string]Value
}

type flow struct {
	returned bool
	values   []Value
	brk      string
	cont     string
	fall     bool
}

const unlabeled = "\x00"

type evaluator struct {
	h        Host
	cfg      Config
	steps    int
	depth    int
	frames   []*frame
	inSwitch int
}

func (e *evaluator) top() *frame {
	return &frame{scopes: []map[string]Value{{}}}
}

func (e *evaluator) step(pos syntax.Pos) error {
	e.steps++
	if e.steps > e.cfg.MaxSteps {
		return &Error{Code: CodeStepLimit, Pos: pos}
	}
	total, err := e.checkSize()
	if err != nil {
		err.Pos = pos
		return err
	}
	if q := total / 4096; q > 0 {
		e.steps += int(q)
		if e.steps > e.cfg.MaxSteps {
			return &Error{Code: CodeStepLimit, Pos: pos}
		}
	}
	return nil
}

func (e *evaluator) chargeSteps(n int, pos syntax.Pos) error {
	e.steps += n
	if e.steps > e.cfg.MaxSteps {
		return &Error{Code: CodeStepLimit, Pos: pos}
	}
	return nil
}

func (e *evaluator) impurity(fr *frame, pos syntax.Pos, format string, args ...any) *Error {
	detail := fmt.Sprintf(format, args...)
	if fr != nil && fr.fn != nil && fr.fn.Const {
		return &Error{Code: CodeSilent, Detail: detail, Pos: pos}
	}
	return &Error{Code: CodeEval, Detail: detail, Pos: pos}
}

func (e *evaluator) fail(pos syntax.Pos, format string, args ...any) *Error {
	return &Error{Code: CodeEval, Detail: fmt.Sprintf(format, args...), Pos: pos}
}

func (e *evaluator) lookup(fr *frame, name string) (Value, bool) {
	for i := len(fr.scopes) - 1; i >= 0; i-- {
		if v, ok := fr.scopes[i][name]; ok {
			return v, true
		}
	}
	return Value{}, false
}

func (e *evaluator) declare(fr *frame, name string, v Value) {
	fr.scopes[len(fr.scopes)-1][name] = v
}

func (e *evaluator) pushScope(fr *frame) {
	fr.scopes = append(fr.scopes, map[string]Value{})
}

func (e *evaluator) popScope(fr *frame) {
	fr.scopes = fr.scopes[:len(fr.scopes)-1]
}

func (e *evaluator) execBlock(fr *frame, b *syntax.BlockStmt) (flow, error) {
	e.pushScope(fr)
	defer e.popScope(fr)
	for _, s := range b.List {
		f, err := e.execStmt(fr, s)
		if err != nil {
			return flow{}, err
		}
		if f.returned || f.brk != "" || f.cont != "" || f.fall {
			return f, nil
		}
	}
	return flow{}, nil
}

func (e *evaluator) execStmt(fr *frame, s syntax.Stmt) (flow, error) {
	if err := e.step(posOf(s)); err != nil {
		return flow{}, err
	}
	switch s := s.(type) {
	case *syntax.EmptyStmt:
		return flow{}, nil
	case *syntax.BlockStmt:
		savedSwitch := e.inSwitch
		e.inSwitch = 0
		f, err := e.execBlock(fr, s)
		e.inSwitch = savedSwitch
		return f, err
	case *syntax.LabeledStmt:
		if inner, ok := s.Stmt.(*syntax.ForStmt); ok {
			return e.execFor(fr, inner, s.Label.Value)
		}
		f, err := e.execStmt(fr, s.Stmt)
		if err != nil {
			return flow{}, err
		}
		if f.brk == s.Label.Value || f.cont == s.Label.Value {
			return flow{}, nil
		}
		return f, nil
	case *syntax.ExprStmt:
		if call, ok := syntax.Unparen(s.X).(*syntax.CallExpr); ok {
			_, _, err := e.evalCallMulti(fr, call)
			return flow{}, err
		}
		_, err := e.evalExpr(fr, s.X)
		return flow{}, err
	case *syntax.SendStmt:
		return flow{}, e.impurity(fr, posOf(s), "channel send in const evaluation")
	case *syntax.DeclStmt:
		return flow{}, e.execDecl(fr, s.DeclList)
	case *syntax.AssignStmt:
		return flow{}, e.execAssign(fr, s)
	case *syntax.BranchStmt:
		return e.execBranch(fr, s)
	case *syntax.CallStmt:
		if s.Tok == syntax.Go {
			return flow{}, e.impurity(fr, posOf(s), "goroutine in const func")
		}
		return flow{}, e.impurity(fr, posOf(s), "defer in const func")
	case *syntax.ReturnStmt:
		return e.execReturn(fr, s)
	case *syntax.IfStmt:
		return e.execIf(fr, s)
	case *syntax.ForStmt:
		return e.execFor(fr, s, "")
	case *syntax.SwitchStmt:
		return e.execSwitch(fr, s)
	case *syntax.SelectStmt:
		return flow{}, e.impurity(fr, posOf(s), "select in const func")
	}
	return flow{}, e.fail(posOf(s), "unsupported statement %T in const evaluation", s)
}

func posOf(n syntax.Node) syntax.Pos {
	if n == nil {
		return syntax.Pos{}
	}
	return syntax.StartPos(n)
}

func (e *evaluator) execBranch(fr *frame, s *syntax.BranchStmt) (flow, error) {
	label := unlabeled
	if s.Label != nil {
		label = s.Label.Value
	}
	switch s.Tok {
	case syntax.Break:
		return flow{brk: label}, nil
	case syntax.Continue:
		return flow{cont: label}, nil
	case syntax.Goto:
		return flow{}, e.impurity(fr, posOf(s), "goto in const func")
	}
	if s.Tok == syntax.Fallthrough && e.inSwitch > 0 {
		return flow{fall: true}, nil
	}
	return flow{}, e.fail(posOf(s), "fallthrough outside switch")
}

func (e *evaluator) execDecl(fr *frame, list []syntax.Decl) error {
	for _, d := range list {
		switch d := d.(type) {
		case *syntax.VarDecl:
			if err := e.execVar(fr, d.NameList, d.Type, d.Values); err != nil {
				return err
			}
		case *syntax.ConstDecl:
			if err := e.execVar(fr, d.NameList, d.Type, d.Values); err != nil {
				return err
			}
		default:
			return e.fail(posOf(d), "type declarations in const func not supported")
		}
	}
	return nil
}

func (e *evaluator) execVar(fr *frame, names []*syntax.Name, typ, vals syntax.Expr) error {
	var rhs []Value
	if vals != nil {
		var err error
		rhs, err = e.evalList(fr, vals)
		if err != nil {
			return err
		}
	}
	var t *Type
	if typ != nil {
		var err error
		if t, err = e.resolveType(fr, typ); err != nil {
			return err
		}
	}
	if len(rhs) == 0 {
		if t == nil {
			return e.fail(posOf(names[0]), "missing initializer in const evaluation")
		}
		for _, n := range names {
			e.declare(fr, n.Value, zeroValue(t))
		}
		return nil
	}
	if len(rhs) == 1 && len(names) > 1 {
		return e.fail(posOf(names[0]), "single value in multi-name declaration")
	}
	if len(rhs) != len(names) {
		return e.fail(posOf(names[0]), "assignment mismatch in const evaluation")
	}
	for i, n := range names {
		v := rhs[i]
		if t != nil {
			var err error
			v, err = e.convert(v, t, posOf(n))
			if err != nil {
				return err
			}
		} else {
			v = defaulted(v)
		}
		v = copyValue(v)
		v.constant = false
		e.declare(fr, n.Value, v)
	}
	return nil
}

func (e *evaluator) evalList(fr *frame, e2 syntax.Expr) ([]Value, error) {
	if l, ok := e2.(*syntax.ListExpr); ok {
		var out []Value
		for _, x := range l.ElemList {
			v, err := e.evalExpr(fr, x)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	}
	if c, ok := e2.(*syntax.CallExpr); ok {
		vs, _, err := e.evalCallMulti(fr, c)
		if err != nil {
			return nil, err
		}
		return vs, nil
	}
	v, err := e.evalExpr(fr, e2)
	if err != nil {
		return nil, err
	}
	return []Value{v}, nil
}

func unitValue(t *Type) Value {
	switch {
	case t != nil && isFloat(t.Kind):
		return FloatValue(1.0, t)
	case t != nil && isComplex(t.Kind):
		return complexConst(t, 1)
	}
	return intValue(big.NewInt(1), t, true)
}

func (e *evaluator) execAssign(fr *frame, s *syntax.AssignStmt) error {
	if s.Rhs == nil {
		p, err := e.resolvePlace(fr, s.Lhs)
		if err != nil {
			return err
		}
		v := p.get()
		one := unitValue(v.typ)
		var r Value
		if s.Op == syntax.Add {
			r, err = e.add(v, one, posOf(s))
		} else {
			r, err = e.sub(v, one, posOf(s))
		}
		if err != nil {
			return err
		}
		p.set(r)
		return nil
	}
	lhs := unpackList(s.Lhs)
	if len(lhs) == 2 {
		if done, err := e.execMapCommaOk(fr, s, lhs); err != nil {
			return err
		} else if done {
			return nil
		}
	}
	rhs, err := e.evalList(fr, s.Rhs)
	if err != nil {
		return err
	}
	if len(rhs) == 1 && len(lhs) > 1 {
		return e.fail(posOf(s), "single value in multi-name assignment")
	}
	if len(rhs) != len(lhs) {
		return e.fail(posOf(s), "assignment mismatch in const evaluation")
	}
	type pending struct {
		pl      place
		val     Value
		newName string
		isNew   bool
		discard bool
	}
	pend := make([]pending, len(lhs))
	curScope := fr.scopes[len(fr.scopes)-1]
	for i, le := range lhs {
		v := rhs[i]
		if s.Op != 0 && s.Op != syntax.Def {
			if nm, ok := nameOf(le); ok && nm == "_" {
				pend[i] = pending{discard: true}
				continue
			}
			p, err := e.resolvePlace(fr, le)
			if err != nil {
				return err
			}
			v, err = e.applyOp(s.Op, p.get(), v, posOf(s))
			if err != nil {
				return err
			}
			dest := p.get().typ
			if dest != nil {
				v, err = e.convert(v, dest, posOf(le))
				if err != nil {
					return err
				}
			}
			pend[i] = pending{pl: p, val: withVar(copyValue(v))}
			continue
		}
		if isDefine(s) {
			nm, isName := nameOf(le)
			if isName && nm == "_" {
				pend[i] = pending{discard: true}
				continue
			}
			if isName {
				if _, exists := curScope[nm]; exists {
					p, err := e.resolvePlace(fr, le)
					if err != nil {
						return err
					}
					dest := p.get().typ
					if dest != nil {
						var err error
						v, err = e.convert(v, dest, posOf(le))
						if err != nil {
							return err
						}
					}
					pend[i] = pending{pl: p, val: withVar(copyValue(v))}
					continue
				}
				pend[i] = pending{newName: nm, val: withVar(copyValue(defaulted(v))), isNew: true}
				continue
			}
			return e.fail(posOf(le), "cannot define non-variable in const evaluation")
		}
		if nm, ok := nameOf(le); ok && nm == "_" {
			pend[i] = pending{discard: true}
			continue
		}
		p, err := e.resolvePlace(fr, le)
		if err != nil {
			return err
		}
		dest := p.get().typ
		if dest != nil {
			var err error
			v, err = e.convert(v, dest, posOf(le))
			if err != nil {
				return err
			}
		}
		pend[i] = pending{pl: p, val: withVar(copyValue(v))}
	}
	for _, pd := range pend {
		if pd.discard {
			continue
		}
		if pd.isNew {
			curScope[pd.newName] = pd.val
			continue
		}
		pd.pl.set(pd.val)
	}
	return nil
}

func (e *evaluator) execMapCommaOk(fr *frame, s *syntax.AssignStmt, lhs []syntax.Expr) (bool, error) {
	ix, ok := syntax.Unparen(s.Rhs).(*syntax.IndexExpr)
	if !ok {
		return false, nil
	}
	base, err := e.evalExpr(fr, ix.X)
	if err != nil {
		return false, err
	}
	if base.mval == nil {
		return false, nil
	}
	key, err := e.evalExpr(fr, ix.Index)
	if err != nil {
		return false, err
	}
	val, present, err := e.mapLookup(base, key, posOf(ix))
	if err != nil {
		return false, err
	}
	okVal := Value{typ: UntypedBool(), constant: false, bval: present}
	if !present {
		val = zeroValue(base.typ.Elem)
	}
	if isDefine(s) {
		cur := fr.scopes[len(fr.scopes)-1]
		for i, le := range lhs {
			nm, isName := nameOf(le)
			if !isName || nm == "_" {
				continue
			}
			if i == 0 {
				cur[nm] = withVar(copyValue(val))
			} else {
				cur[nm] = withVar(copyValue(okVal))
			}
		}
		return true, nil
	}
	for i, le := range lhs {
		if nm, ok := nameOf(le); ok && nm == "_" {
			continue
		}
		p, err := e.resolvePlace(fr, le)
		if err != nil {
			return false, err
		}
		if i == 0 {
			p.set(withVar(copyValue(val)))
		} else {
			p.set(withVar(copyValue(okVal)))
		}
	}
	return true, nil
}

func isDefine(s *syntax.AssignStmt) bool {
	return s.Op == syntax.Def
}

func nameOf(e syntax.Expr) (string, bool) {
	n, ok := syntax.Unparen(e).(*syntax.Name)
	if !ok {
		return "", false
	}
	return n.Value, true
}

func unpackList(e syntax.Expr) []syntax.Expr {
	if l, ok := e.(*syntax.ListExpr); ok {
		return l.ElemList
	}
	return []syntax.Expr{e}
}

func withVar(v Value) Value {
	v.constant = false
	return v
}

type place struct {
	get func() Value
	set func(Value)
}

func (e *evaluator) resolvePlace(fr *frame, lhs syntax.Expr) (place, error) {
	switch lhs := syntax.Unparen(lhs).(type) {
	case *syntax.Name:
		for i := len(fr.scopes) - 1; i >= 0; i-- {
			if _, ok := fr.scopes[i][lhs.Value]; ok {
				idx := i
				name := lhs.Value
				return place{
					get: func() Value { return fr.scopes[idx][name] },
					set: func(v Value) { fr.scopes[idx][name] = v },
				}, nil
			}
		}
		return place{}, e.fail(posOf(lhs), "undefined variable in const evaluation")
	case *syntax.IndexExpr:
		return e.indexPlace(fr, lhs)
	case *syntax.SelectorExpr:
		return e.fieldPlace(fr, lhs)
	case *syntax.Operation:
		if lhs.Op == syntax.Mul && lhs.Y == nil {
			return place{}, e.impurity(fr, posOf(lhs), "pointer dereference in const func")
		}
	}
	return place{}, e.fail(posOf(lhs), "cannot assign in const evaluation")
}

func (e *evaluator) execReturn(fr *frame, s *syntax.ReturnStmt) (flow, error) {
	if s.Results == nil {
		if fr.fn == nil || len(fr.fn.ResultNames) == 0 {
			return flow{returned: true}, nil
		}
		vals := make([]Value, 0, len(fr.fn.ResultNames))
		for _, name := range fr.fn.ResultNames {
			if name == "" {
				return flow{}, e.fail(posOf(s), "missing return values in const evaluation")
			}
			v, ok := e.lookup(fr, name)
			if !ok {
				return flow{}, e.fail(posOf(s), "missing return values in const evaluation")
			}
			vals = append(vals, v)
		}
		return flow{returned: true, values: vals}, nil
	}
	vals, err := e.evalList(fr, s.Results)
	if err != nil {
		return flow{}, err
	}
	if fr.fn != nil && len(fr.fn.Results) > 0 && len(vals) == len(fr.fn.Results) {
		for i := range vals {
			if fr.fn.Results[i] == nil {
				continue
			}
			var err error
			vals[i], err = e.convert(vals[i], fr.fn.Results[i], posOf(s))
			if err != nil {
				return flow{}, err
			}
		}
	}
	return flow{returned: true, values: vals}, nil
}

func (e *evaluator) execIf(fr *frame, s *syntax.IfStmt) (flow, error) {
	e.pushScope(fr)
	defer e.popScope(fr)
	savedSwitch := e.inSwitch
	e.inSwitch = 0
	defer func() { e.inSwitch = savedSwitch }()
	if s.Init != nil {
		f, err := e.execSimple(fr, s.Init)
		if err != nil {
			return flow{}, err
		}
		if f.returned || f.brk != "" || f.cont != "" {
			return f, nil
		}
	}
	cond, err := e.evalExpr(fr, s.Cond)
	if err != nil {
		return flow{}, err
	}
	b, err := e.truth(cond, posOf(s.Cond))
	if err != nil {
		return flow{}, err
	}
	if b {
		return e.execBlock(fr, s.Then)
	}
	if s.Else != nil {
		switch el := s.Else.(type) {
		case *syntax.BlockStmt:
			return e.execBlock(fr, el)
		case *syntax.IfStmt:
			return e.execStmt(fr, el)
		}
	}
	return flow{}, nil
}

func (e *evaluator) execSimple(fr *frame, s syntax.SimpleStmt) (flow, error) {
	if s == nil {
		return flow{}, nil
	}
	return e.execStmt(fr, s.(syntax.Stmt))
}
