// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

// This file implements the M5-S2 static purity check for const funcs
// (SPEC Sec 2.1, T-201). Every const func body is scanned after it is
// type-checked; the first impurity reports
// [correctness::const_func_impure] with its reason. Dynamic evaluation
// mirrors the same impurity class (silently inside const bodies), so
// each construct reports exactly once.

import (
	"cmd/compile/internal/syntax"
	"fmt"
	. "internal/types/errors"
)

// logosSchedulePurity enqueues a const func purity check behind the
// body check the caller just scheduled, so Info maps resolve precisely.
// Non-const functions return without scheduling anything.
func (check *Checker) logosSchedulePurity(obj *Func, fdecl *syntax.FuncDecl) {
	if !check.conf.Logos {
		return
	}
	if !check.logosConstSyntax[fdecl] {
		return
	}
	check.later(func() {
		check.logosCheckConstFunc(obj, fdecl)
	}).describef(obj, "const func %s purity", obj.name)
}

// logosCheckConstFunc scans one const func for purity violations. It
// runs after the body is checked, so Info maps resolve precisely. Only
// the first violation in source order reports.
func (check *Checker) logosCheckConstFunc(obj *Func, fdecl *syntax.FuncDecl) {
	if !check.conf.Logos {
		return
	}
	s := &logosPurity{check: check}
	s.scanSignature(fdecl)
	if !s.done {
		syntax.Inspect(fdecl.Body, s.visit)
	}
	if s.reason != "" {
		check.errorf(s.at, InvalidConstInit, "%s %s", logosImpureRule, s.reason)
	}
}

// logosPurity holds scanner state.
type logosPurity struct {
	check  *Checker
	at     syntax.Node
	reason string
	done   bool
}

func (s *logosPurity) fail(at syntax.Node, format string, args ...any) bool {
	if s.done {
		return false
	}
	s.at = at
	s.reason = fmt.Sprintf(format, args...)
	s.done = true
	return false
}

// scanSignature checks receiver, parameter, and result types.
func (s *logosPurity) scanSignature(fdecl *syntax.FuncDecl) {
	if fdecl.Recv != nil {
		s.scanType(fdecl.Recv.Type)
	}
	for _, f := range fdecl.Type.ParamList {
		s.scanType(f.Type)
	}
	for _, f := range fdecl.Type.ResultList {
		s.scanType(f.Type)
	}
}

// scanType flags pointer, channel, uintptr, interface, function, and
// option shapes in type positions, resolving names through scope.
func (s *logosPurity) scanType(te syntax.Expr) {
	if s.done || te == nil {
		return
	}
	switch te := syntax.Unparen(te).(type) {
	case *syntax.Operation:
		if te.Op == syntax.Mul && te.Y == nil {
			s.fail(te, "pointer type in const func")
		}
	case *syntax.ChanType:
		s.fail(te, "channel type in const func")
	case *syntax.FuncType:
		s.fail(te, "function type in const func")
	case *syntax.InterfaceType:
		s.fail(te, "interface type in const func")
	case *syntax.OptionType:
		s.fail(te, "option type in const func")
	case *syntax.Name:
		switch te.Value {
		case "uintptr":
			s.fail(te, "uintptr type in const func")
		case "any", "error":
			s.fail(te, "interface type in const func")
		default:
			s.scanNamed(te)
		}
	case *syntax.SelectorExpr:
		if x, ok := syntax.Unparen(te.X).(*syntax.Name); ok && x.Value == "unsafe" && te.Sel.Value == "Pointer" {
			s.fail(te, "pointer type in const func")
		}
	case *syntax.SliceType:
		s.scanType(te.Elem)
	case *syntax.ArrayType:
		s.scanType(te.Elem)
	case *syntax.MapType:
		s.scanType(te.Key)
		if !s.done {
			s.scanType(te.Value)
		}
	case *syntax.StructType:
		for _, f := range te.FieldList {
			s.scanType(f.Type)
			if s.done {
				return
			}
		}
	case *syntax.DotsType:
		s.scanType(te.Elem)
	case *syntax.ConstType:
		s.scanType(te.Elem)
	}
}

// scanNamed resolves a type name and flags impurity behind it.
func (s *logosPurity) scanNamed(id *syntax.Name) {
	obj := s.lookup(id.Value)
	tn, ok := obj.(*TypeName)
	if !ok {
		return
	}
	switch u := tn.typ.(type) {
	case *Pointer:
		s.fail(id, "pointer type in const func")
	case *Chan:
		s.fail(id, "channel type in const func")
	case *Basic:
		if u.Kind() == Uintptr {
			s.fail(id, "uintptr type in const func")
		}
	case *Interface:
		s.fail(id, "interface type in const func")
	case *Signature:
		s.fail(id, "function type in const func")
	}
}

// lookup resolves a package-level or universe name.
func (s *logosPurity) lookup(name string) Object {
	check := s.check
	if obj := check.pkg.scope.Lookup(name); obj != nil {
		return obj
	}
	return Universe.Lookup(name)
}

// visit inspects one node; returning false prunes reported subtrees.
// Declarations and composite literals carry type positions, so they
// descend manually: types through scanType, values through Inspect.
func (s *logosPurity) visit(n syntax.Node) bool {
	if s.done || n == nil {
		return false
	}
	switch n := n.(type) {
	case *syntax.VarDecl:
		s.scanType(n.Type)
		for _, v := range syntax.UnpackListExpr(n.Values) {
			syntax.Inspect(v, s.visit)
			if s.done {
				return false
			}
		}
		return false
	case *syntax.ConstDecl:
		s.scanType(n.Type)
		for _, v := range syntax.UnpackListExpr(n.Values) {
			syntax.Inspect(v, s.visit)
			if s.done {
				return false
			}
		}
		return false
	case *syntax.CompositeLit:
		if n.Type != nil {
			s.scanType(n.Type)
			if s.done {
				return false
			}
		}
		for _, el := range n.ElemList {
			syntax.Inspect(el, s.visit)
			if s.done {
				return false
			}
		}
		return false
	case *syntax.CallExpr:
		return s.visitCall(n)
	case *syntax.AssignStmt:
		if s.visitStore(n.Lhs) {
			return false
		}
		return true
	case *syntax.RangeClause:
		if s.visitStore(n.Lhs) {
			return false
		}
		s.visitRange(n)
		return !s.done
	case *syntax.FuncLit:
		s.fail(n, "function literal in const func")
		return false
	case *syntax.CallStmt:
		if n.Tok == syntax.Go {
			s.fail(n, "goroutine in const func")
		} else {
			s.fail(n, "defer in const func")
		}
		return false
	case *syntax.BranchStmt:
		if n.Tok == syntax.Goto {
			s.fail(n, "goto in const func")
			return false
		}
		return true
	case *syntax.SendStmt:
		s.fail(n, "channel send in const func")
		return false
	case *syntax.SelectStmt:
		s.fail(n, "select in const func")
		return false
	case *syntax.Name:
		s.visitName(n, false)
		return true
	case *syntax.Operation:
		if n.Y == nil && (n.Op == syntax.Mul || n.Op == syntax.And) {
			// Type positions never reach here (scanType owns
			// them), so a bare & or * operates on a value.
			if n.Op == syntax.Mul {
				s.fail(n, "pointer dereference in const func")
			} else {
				s.fail(n, "address-of in const func")
			}
			return false
		}
		return true
	}
	return true
}

// visitStore flags package-variable stores in an assignment target:
// bare names and the bases of selectors and indexes. It reports
// whether it failed so the caller prunes the subtree.
func (s *logosPurity) visitStore(lhs syntax.Expr) bool {
	found := false
	var walk func(e syntax.Expr)
	walk = func(e syntax.Expr) {
		if s.done || found {
			return
		}
		switch e := syntax.Unparen(e).(type) {
		case *syntax.Name:
			if s.visitName(e, true) {
				found = true
			}
		case *syntax.ListExpr:
			for _, x := range e.ElemList {
				walk(x)
			}
		case *syntax.SelectorExpr:
			walk(e.X)
		case *syntax.IndexExpr:
			walk(e.X)
		}
	}
	for _, le := range unpackLogosList(lhs) {
		walk(le)
	}
	return found || s.done
}

// visitRange flags ranging over maps.
func (s *logosPurity) visitRange(rc *syntax.RangeClause) {
	if s.done {
		return
	}
	if tv, ok := s.check.Info.Types[rc.X]; ok && tv.Type != nil {
		if _, isMap := Unalias(tv.Type).(*Map); isMap {
			s.fail(rc.X, "range over map in const func")
		}
	}
}

// visitName flags package-variable uses: stores report writes, all
// other positions report reads. Declarations never reach here (Defs,
// not Uses, records them). It reports whether it failed.
func (s *logosPurity) visitName(id *syntax.Name, store bool) bool {
	if s.done {
		return false
	}
	obj, ok := s.check.Info.Uses[id].(*Var)
	if !ok {
		return false
	}
	if obj.parent == s.check.pkg.scope {
		if store {
			s.fail(id, "write to package variable %s in const func", obj.name)
		} else {
			s.fail(id, "read of package variable %s in const func", obj.name)
		}
		return true
	}
	if obj.pkg != nil && obj.pkg != s.check.pkg {
		s.fail(id, "read of imported variable %s in const func", obj.name)
		return true
	}
	return false
}

// visitCall checks call targets.
func (s *logosPurity) visitCall(c *syntax.CallExpr) bool {
	check := s.check
	switch fun := syntax.Unparen(c.Fun).(type) {
	case *syntax.Name:
		obj := check.Info.Uses[fun]
		if obj == nil {
			obj = s.lookup(fun.Value)
		}
		switch obj := obj.(type) {
		case *Func:
			if obj.pkg != check.pkg {
				path := ""
				if obj.pkg != nil {
					path = obj.pkg.path + "."
				}
				s.fail(c, "call to imported function %s%s in const func", path, obj.name)
				return false
			}
			if sig, ok := obj.typ.(*Signature); ok && sig.TypeParams() != nil && sig.TypeParams().Len() > 0 {
				s.fail(c, "generic call in const func")
				return false
			}
		case *Builtin:
			switch obj.name {
			case "print", "println":
				s.fail(c, "I/O: call to %s in const func", obj.name)
				return false
			case "recover":
				s.fail(c, "call to recover in const func")
				return false
			case "new":
				s.fail(c, "pointer allocation in const func")
				return false
			case "make":
				if len(c.ArgList) > 0 {
					s.scanType(c.ArgList[0])
					if s.done {
						return false
					}
				}
			}
		case *TypeName:
			if u, ok := obj.typ.(*Basic); ok && u.Kind() == Uintptr {
				s.fail(c, "conversion to uintptr in const func")
				return false
			}
		}
	case *syntax.SelectorExpr:
		if x, ok := syntax.Unparen(fun.X).(*syntax.Name); ok {
			if _, ok := s.pkgName(x); ok {
				s.fail(c, "call to imported function %s in const func", selText(fun))
				return false
			}
		}
		s.fail(c, "method call in const func")
		return false
	case *syntax.IndexExpr:
		s.fail(c, "generic call in const func")
		return false
	case *syntax.FuncLit:
		s.fail(c, "function literal in const func")
		return false
	default:
		// A composite type in function position converts; its
		// shapes belong to scanType rather than the walk.
		s.scanType(c.Fun)
		if s.done {
			return false
		}
		for _, a := range c.ArgList {
			syntax.Inspect(a, s.visit)
			if s.done {
				return false
			}
		}
		return false
	}
	return true
}

// pkgName reports whether x names an imported package.
func (s *logosPurity) pkgName(x *syntax.Name) (*PkgName, bool) {
	check := s.check
	if obj, ok := check.Info.Uses[x].(*PkgName); ok {
		return obj, true
	}
	if obj, ok := check.pkg.scope.Lookup(x.Value).(*PkgName); ok {
		return obj, true
	}
	return nil, false
}

// unpackLogosList splits an assignment left-hand side.
func unpackLogosList(e syntax.Expr) []syntax.Expr {
	return syntax.UnpackListExpr(e)
}

// selText renders pkg.Func for messages.
func selText(sel *syntax.SelectorExpr) string {
	if x, ok := syntax.Unparen(sel.X).(*syntax.Name); ok {
		return x.Value + "." + sel.Sel.Value
	}
	return sel.Sel.Value
}
