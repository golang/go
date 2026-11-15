// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"slices"

	"cmd/compile/internal/syntax"
)

type logosWriteStep struct {
	field *Var
	ref   bool
}

type logosWritePath struct {
	root  Object
	steps []logosWriteStep
}

func (check *Checker) logosWriteBase(e syntax.Expr) (base syntax.Expr, steps []logosWriteStep, viaReference bool) {
	finish := func(base syntax.Expr) (syntax.Expr, []logosWriteStep, bool) {
		slices.Reverse(steps)
		return base, steps, viaReference
	}
	for {
		e = syntax.Unparen(e)
		switch n := e.(type) {
		case *syntax.SelectorExpr:
			if check.logosQualified(n) {
				return finish(n)
			}
			steps = append(steps, logosWriteStep{field: check.logosSelectedField(n)})
			if _, ok := check.logosUnderlying(n.X).(*Pointer); ok {
				steps = append(steps, logosWriteStep{ref: true})
				viaReference = true
			}
			e = n.X
		case *syntax.IndexExpr:
			var crossed bool
			steps, crossed = check.logosElementStep(steps, n.X)
			viaReference = viaReference || crossed
			e = n.X
		case *syntax.SliceExpr:
			steps, _ = check.logosElementStep(steps, n.X)
			viaReference = true
			e = n.X
		case *syntax.Operation:
			if n.Op != syntax.Mul || n.Y != nil {
				return finish(n)
			}
			steps = append(steps, logosWriteStep{ref: true})
			viaReference = true
			e = n.X
		default:
			return finish(n)
		}
	}
}

func (check *Checker) logosElementStep(steps []logosWriteStep, operand syntax.Expr) ([]logosWriteStep, bool) {
	switch check.logosUnderlying(operand).(type) {
	case *Pointer:
		return append(steps, logosWriteStep{}, logosWriteStep{ref: true}), true
	case *Slice, *Map:
		return append(steps, logosWriteStep{ref: true}), true
	}
	return append(steps, logosWriteStep{}), false
}

func (check *Checker) logosSelectedField(sel *syntax.SelectorExpr) *Var {
	if check.Info == nil {
		return nil
	}
	s := check.Info.Selections[sel]
	if s == nil || s.Kind() != FieldVal {
		return nil
	}
	field, _ := s.Obj().(*Var)
	return field
}

func (check *Checker) logosUnderlying(e syntax.Expr) Type {
	if check.Info == nil {
		return nil
	}
	t := check.Info.TypeOf(e)
	if t == nil {
		return nil
	}
	t, _, _ = logosReadonlyBase(t)
	return Unalias(t).Underlying()
}

func (check *Checker) logosQualified(sel *syntax.SelectorExpr) bool {
	pkg, ok := sel.X.(*syntax.Name)
	if !ok || check.Info == nil {
		return false
	}
	_, ok = check.Info.Uses[pkg].(*PkgName)
	return ok
}

func (check *Checker) logosWritePath(e syntax.Expr) logosWritePath {
	base, steps, _ := check.logosWriteBase(e)
	return logosWritePath{root: check.logosWriteRoot(base), steps: steps}
}

func (check *Checker) logosContentsPath(e syntax.Expr) logosWritePath {
	path := check.logosWritePath(e)
	path.steps = append(path.steps, logosWriteStep{ref: true})
	return path
}

func (check *Checker) logosWriteRoot(base syntax.Expr) Object {
	if check.Info == nil {
		return nil
	}
	switch n := base.(type) {
	case *syntax.Name:
		return check.Info.Uses[n]
	case *syntax.SelectorExpr:
		if check.logosQualified(n) {
			return check.Info.Uses[n.Sel]
		}
	}
	return nil
}

func logosWriteHits(protected, target logosWritePath) bool {
	if protected.root == nil || protected.root != target.root {
		return false
	}
	shared := min(len(protected.steps), len(target.steps))
	if !slices.Equal(protected.steps[:shared], target.steps[:shared]) {
		return false
	}
	if len(target.steps) >= len(protected.steps) {
		return true
	}
	return !protected.steps[len(target.steps)].ref
}
