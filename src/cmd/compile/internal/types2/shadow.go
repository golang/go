// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"strings"
)

// shadowedNames collects predeclared names hidden by declarations referenced
// in an error. It does not diagnose shadowing in otherwise valid code.
type shadowedNames struct {
	check *Checker
	scope shadowingScope
	names []string
	seen  map[Type]bool
}

func (s *shadowedNames) add(arg any) {
	switch arg := arg.(type) {
	case *operand:
		if arg != nil {
			s.expr(arg.expr)
			s.typ(arg.typ())
		}
	case syntax.Expr:
		s.expr(arg)
	case Object:
		s.object(arg)
	case Type:
		s.typ(arg)
	case *Term:
		if arg != nil {
			s.typ(arg.typ)
		}
	case []*operand:
		for _, x := range arg {
			s.add(x)
		}
	case []syntax.Expr:
		for _, e := range arg {
			s.expr(e)
		}
	case []Type:
		for _, t := range arg {
			s.typ(t)
		}
	case []*TypeParam:
		for _, t := range arg {
			s.typ(t)
		}
	}
}

func (s *shadowedNames) object(obj Object) {
	if obj == nil || obj.Parent() == Universe || obj.Pkg() != s.check.pkg {
		return
	}
	if obj.Parent() == nil {
		// Renamed type parameters retain their declaration but have no
		// parent scope (see Checker.renameTParams).
		tname, ok := obj.(*TypeName)
		if !ok {
			return
		}
		if _, ok := tname.Type().(*TypeParam); !ok {
			return
		}
	}
	if v, ok := obj.(*Var); ok && v.IsField() {
		return
	}
	if f, ok := obj.(*Func); ok && f.Signature().Recv() != nil {
		return
	}
	name := obj.Name()
	if Universe.Lookup(name) == nil {
		return
	}
	for _, old := range s.names {
		if old == name {
			return
		}
	}
	s.names = append(s.names, name)
}

func (s *shadowedNames) expr(e syntax.Expr) {
	if e == nil {
		return
	}
	syntax.Inspect(e, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.Name:
			s.object(s.lookup(n))
		case *syntax.SelectorExpr:
			// Member names are not lexical references.
			if ident, _ := n.X.(*syntax.Name); ident != nil {
				// Package qualifiers do not affect the selected member's meaning.
				obj := s.lookup(ident)
				if _, ok := obj.(*PkgName); !ok {
					s.object(obj)
				}
			} else {
				s.expr(n.X)
			}
			return false
		case *syntax.KeyValueExpr:
			// A key may name a struct field. Invalid map keys are diagnosed
			// separately, with the key expression as an operand.
			s.expr(n.Value)
			return false
		case *syntax.Field:
			// Parameter, field, and method names are declarations.
			s.expr(n.Type)
			return false
		case *syntax.FuncLit:
			// Its body has a different lexical environment. The signature's
			// types are handled by typ when the operand is formatted.
			return false
		}
		return true
	})
}

func (s *shadowedNames) lookup(ident *syntax.Name) Object {
	if s.scope.scope != nil {
		return s.scope.lookup(ident)
	}
	return s.check.lookup(ident.Value)
}

func (s *shadowedNames) typ(t Type) {
	if t == nil {
		return
	}
	if _, ok := t.(*Basic); ok {
		return // an implicit basic type is not a reference to a shadowed name
	}
	if s.seen == nil {
		s.seen = make(map[Type]bool)
	}
	if s.seen[t] {
		return
	}
	s.seen[t] = true
	switch t := t.(type) {
	case *Named:
		s.object(t.Obj())
		if t.TypeArgs().Len() == 0 {
			for _, param := range t.TypeParams().list() {
				s.typ(param)
			}
		}
		for _, arg := range t.TypeArgs().list() {
			s.typ(arg)
		}
	case *Alias:
		s.object(t.Obj())
		if t.TypeArgs().Len() == 0 {
			for _, param := range t.TypeParams().list() {
				s.typ(param)
			}
		}
		for _, arg := range t.TypeArgs().list() {
			s.typ(arg)
		}
	case *TypeParam:
		s.object(t.Obj())
		s.typ(t.bound)
	case *Pointer:
		s.typ(t.base)
	case *Array:
		s.typ(t.elem)
	case *Slice:
		s.typ(t.elem)
	case *Map:
		s.typ(t.key)
		s.typ(t.elem)
	case *Chan:
		s.typ(t.elem)
	case *Tuple:
		if t != nil {
			for _, v := range t.vars {
				s.typ(v.typ)
			}
		}
	case *Signature:
		for _, param := range t.TypeParams().list() {
			s.typ(param)
		}
		s.typ(t.params)
		s.typ(t.results)
	case *Struct:
		for _, f := range t.fields {
			s.typ(f.typ)
		}
	case *Interface:
		for _, m := range t.methods {
			s.typ(m.typ)
		}
		for _, e := range t.embeddeds {
			s.typ(e)
		}
	case *Union:
		for _, term := range t.terms {
			s.typ(term.typ)
		}
	}
}

func (s *shadowedNames) notes() string {
	var notes strings.Builder
	for _, name := range s.names {
		kind := "predeclared "
		if _, ok := Universe.Lookup(name).(*Builtin); ok {
			kind = "built-in "
		}
		notes.WriteString(" (" + kind + name + " is shadowed)")
	}
	return notes.String()
}

func (err *error_) addShadowed(arg any) {
	s := shadowedNames{check: err.check, names: err.shadowed}
	s.add(arg)
	err.shadowed = s.names
}

func (err *error_) addShadowedInScope(scope shadowingScope, expr syntax.Expr) {
	s := shadowedNames{check: err.check, scope: scope, names: err.shadowed}
	s.expr(expr)
	err.shadowed = s.names
}

func (err *error_) addShadowingNotes() {
	s := shadowedNames{names: err.shadowed}
	err.desc[0].msg += s.notes()
}
