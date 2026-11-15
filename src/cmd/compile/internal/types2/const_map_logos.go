// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

var logosStdMutatedParams = map[string]uint64{
	"fmt.Print":       0,
	"fmt.Printf":      0,
	"fmt.Println":     0,
	"fmt.Sprint":      0,
	"fmt.Sprintf":     0,
	"fmt.Sprintln":    0,
	"fmt.Append":      0,
	"fmt.Appendf":     0,
	"fmt.Appendln":    0,
	"maps.All":        0,
	"maps.Clone":      0,
	"maps.Collect":    0,
	"maps.Copy":       1,
	"maps.DeleteFunc": 1,
	"maps.Equal":      0,
	"maps.EqualFunc":  0,
	"maps.Insert":     1,
	"maps.Keys":       0,
	"maps.Values":     0,
}

func logosImportedCallMask(check *Checker, call *syntax.CallExpr) (mask uint64, imported, known bool) {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok {
		return 0, false, false
	}
	name, ok := sel.X.(*syntax.Name)
	if !ok {
		return 0, false, false
	}
	pkg, ok := check.lookup(name.Value).(*PkgName)
	if !ok {
		return 0, false, false
	}
	mask, known = logosStdMutatedParams[pkg.Imported().Path()+"."+sel.Sel.Value]
	return mask, true, known
}

func logosImportedCallee(check *Checker, call *syntax.CallExpr) (path, name string, ok bool) {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok {
		return "", "", false
	}
	qual, ok := sel.X.(*syntax.Name)
	if !ok {
		return "", "", false
	}
	pkg, ok := check.lookup(qual.Value).(*PkgName)
	if !ok || pkg.Imported() == nil {
		return "", "", false
	}
	return pkg.Imported().Path(), sel.Sel.Value, true
}

func logosConstMapKeyOK(t Type) bool {
	switch u := t.Underlying().(type) {
	case *Basic:
		return u.info&(IsString|IsBoolean|IsInteger|IsFloat|IsComplex) != 0
	case *Struct:
		for i := range u.NumFields() {
			if f := u.Field(i); f.Name() != "_" && !logosConstMapKeyOK(f.Type()) {
				return false
			}
		}
		return true
	case *Array:
		return logosConstMapKeyOK(u.Elem())
	}
	return false
}

func (check *Checker) logosConstMapReadOnly(call *syntax.CallExpr, imported, known bool, mask uint64, i int) bool {
	if imported {
		if known && i < 64 && mask&(1<<uint(i)) == 0 {
			return true
		}
		if path, name, ok := logosImportedCallee(check, call); ok {
			if admitted, _ := check.logosAdmitImportedConstFunc(path, name); admitted {
				return true
			}
		}
		return false
	}
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return false
	}
	fn, ok := check.lookup(name.Value).(*Func)
	if !ok || fn.Pkg() != check.pkg {
		return false
	}
	bodies, consts := check.logosConstFuncBodies()
	fd, ok := bodies[name.Value]
	return ok && consts[fd]
}

func (check *Checker) logosConstMapArgs(call *syntax.CallExpr, sigParams *Tuple, args []*operand) {
	if !check.conf.Logos || sigParams == nil {
		return
	}
	mask, imported, known := logosImportedCallMask(check, call)
	for i, a := range args {
		if !a.isValid() || i >= sigParams.Len() {
			continue
		}
		c := logosAsConstView(a.typ())
		if c == nil || !c.ConstMap {
			continue
		}
		if logosAsConstView(sigParams.At(i).Type()) != nil {
			continue
		}
		readOnly := check.logosConstMapReadOnly(call, imported, known, mask, i)
		if !readOnly {
			if imported {
				if path, name, ok := logosImportedCallee(check, call); ok {
					if _, detail := check.logosAdmitImportedConstFunc(path, name); detail != "" {
						check.errorf(a, InvalidConstInit, "%s %s", logosConstEvalRule, detail)
						a.invalidate()
						continue
					}
				}
			}
			check.errorf(a, IncompatibleAssign, "%s cannot pass argument through const view %s in argument to %s", logosConstMutation, ExprString(a.expr), ExprString(call.Fun))
			a.invalidate()
			continue
		}
		a.typ_ = c.Elem
		check.updateExprType(a.expr, c.Elem, false)
	}
}
