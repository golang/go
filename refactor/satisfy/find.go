// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package satisfy inspects the type-checked ASTs of Go packages and
// reports the set of discovered type constraints of the form (lhs, rhs
// Type) where lhs is a non-trivial interface, rhs satisfies this
// interface, and this fact is necessary for the package to be
// well-typed.
//
// THIS PACKAGE IS EXPERIMENTAL AND MAY CHANGE AT ANY TIME.
//
// It is provided only for the gorename tool.  Ideally this
// functionality will become part of the type-checker in due course,
// since it is computing it anyway, and it is robust for ill-typed
// inputs, which this package is not.
//
package satisfy // import "golang.org/x/tools/refactor/satisfy"

// NOTES:
//
// We don't care about numeric conversions, so we don't descend into
// types or constant expressions.  This is unsound because
// constant expressions can contain arbitrary statements, e.g.
//   const x = len([1]func(){func() {
//     ...
//   }})
//
// TODO(adonovan): make this robust against ill-typed input.
// Or move it into the type-checker.
//
// Assignability conversions are possible in the following places:
// - in assignments y = x, y := x, var y = x.
// - from call argument types to formal parameter types
// - in append and delete calls
// - from return operands to result parameter types
// - in composite literal T{k:v}, from k and v to T's field/element/key type
// - in map[key] from key to the map's key type
// - in comparisons x==y and switch x { case y: }.
// - in explicit conversions T(x)
// - in sends ch <- x, from x to the channel element type
// - in type assertions x.(T) and switch x.(type) { case T: }
//
// The results of this pass provide information equivalent to the
// ssa.MakeInterface and ssa.ChangeInterface instructions.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
)

// A Constraint records the fact that the RHS type does and must
// satisify the LHS type, which is an interface.
// The names are suggestive of an assignment statement LHS = RHS.
type Constraint struct {
	LHS, RHS types.Type
}

// A Finder inspects the type-checked ASTs of Go packages and
// accumulates the set of type constraints (x, y) such that x is
// assignable to y, y is an interface, and both x and y have methods.
//
// In other words, it returns the subset of the "implements" relation
// that is checked during compilation of a package.  Refactoring tools
// will need to preserve at least this part of the relation to ensure
// continued compilation.
//
type Finder struct {
	Result    map[Constraint]bool
	msetcache typeutil.MethodSetCache

	// per-Find state
	info *types.Info
	sig  *types.Signature
}

// Find inspects a single package, populating Result with its pairs of
// constrained types.
//
// The result is non-canonical and thus may contain duplicates (but this
// tends to preserves names of interface types better).
//
// The package must be free of type errors, and
// info.{Defs,Uses,Selections,Types} must have been populated by the
// type-checker.
//
func (f *Finder) Find(info *types.Info, files []*ast.File) {
	if f.Result == nil {
		f.Result = make(map[Constraint]bool)
	}

	f.info = info
	for _, file := range files {
		for _, d := range file.Decls {
			switch d := d.(type) {
			case *ast.GenDecl:
				if d.Tok == token.VAR { // ignore consts
					for _, spec := range d.Specs {
						f.valueSpec(spec.(*ast.ValueSpec))
					}
				}

			case *ast.FuncDecl:
				if d.Body != nil {
					f.sig = f.info.Defs[d.Name].Type().(*types.Signature)
					f.stmt(d.Body)
					f.sig = nil
				}
			}
		}
	}
	f.info = nil
}

var (
	tInvalid     = types.Typ[types.Invalid]
	tUntypedBool = types.Typ[types.UntypedBool]
	tUntypedNil  = types.Typ[types.UntypedNil]
)

// exprN visits an expression in a multi-value context.
func (f *Finder) exprN(e ast.Expr) types.Type {
	typ := f.info.Types[e].Type.(*types.Tuple)
	switch e := e.(type) {
	case *ast.ParenExpr:
		return f.exprN(e.X)

	case *ast.CallExpr:
		// x, err := f(args)
		sig := f.expr(e.Fun).Underlying().(*types.Signature)
		f.call(sig, e.Args)

	case *ast.IndexExpr:
		// y, ok := x[i]
		x := f.expr(e.X)
		f.assign(f.expr(e.Index), x.Underlying().(*types.Map).Key())

	case *ast.TypeAssertExpr:
		// y, ok := x.(T)
		f.typeAssert(f.expr(e.X), typ.At(0).Type())

	case *ast.UnaryExpr: // must be receive <-
		// y, ok := <-x
		f.expr(e.X)

	default:
		panic(e)
	}
	return typ
}

func (f *Finder) call(sig *types.Signature, args []ast.Expr) {
	if len(args) == 0 {
		return
	}

	// Ellipsis call?  e.g. f(x, y, z...)
	if _, ok := args[len(args)-1].(*ast.Ellipsis); ok {
		for i, arg := range args {
			// The final arg is a slice, and so is the final param.
			f.assign(sig.Params().At(i).Type(), f.expr(arg))
		}
		return
	}

	var argtypes []types.Type

	// Gather the effective actual parameter types.
	if tuple, ok := f.info.Types[args[0]].Type.(*types.Tuple); ok {
		// f(g()) call where g has multiple results?
		f.expr(args[0])
		// unpack the tuple
		for i := 0; i < tuple.Len(); i++ {
			argtypes = append(argtypes, tuple.At(i).Type())
		}
	} else {
		for _, arg := range args {
			argtypes = append(argtypes, f.expr(arg))
		}
	}

	// Assign the actuals to the formals.
	if !sig.Variadic() {
		for i, argtype := range argtypes {
			f.assign(sig.Params().At(i).Type(), argtype)
		}
	} else {
		// The first n-1 parameters are assigned normally.
		nnormals := sig.Params().Len() - 1
		for i, argtype := range argtypes[:nnormals] {
			f.assign(sig.Params().At(i).Type(), argtype)
		}
		// Remaining args are assigned to elements of varargs slice.
		tElem := sig.Params().At(nnormals).Type().(*types.Slice).Elem()
		for i := nnormals; i < len(argtypes); i++ {
			f.assign(tElem, argtypes[i])
		}
	}
}

func (f *Finder) builtin(obj *types.Builtin, sig *types.Signature, args []ast.Expr, T types.Type) types.Type {
	switch obj.Name() {
	case "make", "new":
		// skip the type operand
		for _, arg := range args[1:] {
			f.expr(arg)
		}

	case "append":
		s := f.expr(args[0])
		if _, ok := args[len(args)-1].(*ast.Ellipsis); ok && len(args) == 2 {
			// append(x, y...)   including append([]byte, "foo"...)
			f.expr(args[1])
		} else {
			// append(x, y, z)
			tElem := s.Underlying().(*types.Slice).Elem()
			for _, arg := range args[1:] {
				f.assign(tElem, f.expr(arg))
			}
		}

	case "delete":
		m := f.expr(args[0])
		k := f.expr(args[1])
		f.assign(m.Underlying().(*types.Map).Key(), k)

	default:
		// ordinary call
		f.call(sig, args)
	}

	return T
}

func (f *Finder) extract(tuple types.Type, i int) types.Type {
	if tuple, ok := tuple.(*types.Tuple); ok && i < tuple.Len() {
		return tuple.At(i).Type()
	}
	return tInvalid
}

func (f *Finder) valueSpec(spec *ast.ValueSpec) {
	var T types.Type
	if spec.Type != nil {
		T = f.info.Types[spec.Type].Type
	}
	switch len(spec.Values) {
	case len(spec.Names): // e.g. var x, y = f(), g()
		for _, value := range spec.Values {
			v := f.expr(value)
			if T != nil {
				f.assign(T, v)
			}
		}

	case 1: // e.g. var x, y = f()
		tuple := f.exprN(spec.Values[0])
		for i := range spec.Names {
			if T != nil {
				f.assign(T, f.extract(tuple, i))
			}
		}
	}
}

// assign records pairs of distinct types that are related by
// assignability, where the left-hand side is an interface and both
// sides have methods.
//
// It should be called for all assignability checks, type assertions,
// explicit conversions and comparisons between two types, unless the
// types are uninteresting (e.g. lhs is a concrete type, or the empty
// interface; rhs has no methods).
//
func (f *Finder) assign(lhs, rhs types.Type) {
	if types.Identical(lhs, rhs) {
		return
	}
	if !isInterface(lhs) {
		return
	}

	if f.msetcache.MethodSet(lhs).Len() == 0 {
		return
	}
	if f.msetcache.MethodSet(rhs).Len() == 0 {
		return
	}
	// record the pair
	f.Result[Constraint{lhs, rhs}] = true
}

// typeAssert must be called for each type assertion x.(T) where x has
// interface type I.
func (f *Finder) typeAssert(I, T types.Type) {
	// Type assertions are slightly subtle, because they are allowed
	// to be "impossible", e.g.
	//
	// 	var x interface{f()}
	//	_ = x.(interface{f()int}) // legal
	//
	// (In hindsight, the language spec should probably not have
	// allowed this, but it's too late to fix now.)
	//
	// This means that a type assert from I to T isn't exactly a
	// constraint that T is assignable to I, but for a refactoring
	// tool it is a conditional constraint that, if T is assignable
	// to I before a refactoring, it should remain so after.

	if types.AssignableTo(T, I) {
		f.assign(I, T)
	}
}

// compare must be called for each comparison x==y.
func (f *Finder) compare(x, y types.Type) {
	if types.AssignableTo(x, y) {
		f.assign(y, x)
	} else if types.AssignableTo(y, x) {
		f.assign(x, y)
	}
}

// expr visits a true expression (not a type or defining ident)
// and returns its type.
func (f *Finder) expr(e ast.Expr) types.Type {
	tv := f.info.Types[e]
	if tv.Value != nil {
		return tv.Type // prune the descent for constants
	}

	// tv.Type may be nil for an ast.Ident.

	switch e := e.(type) {
	case *ast.BadExpr, *ast.BasicLit:
		// no-op

	case *ast.Ident:
		// (referring idents only)
		if obj, ok := f.info.Uses[e]; ok {
			return obj.Type()
		}
		if e.Name == "_" { // e.g. "for _ = range x"
			return tInvalid
		}
		panic("undefined ident: " + e.Name)

	case *ast.Ellipsis:
		if e.Elt != nil {
			f.expr(e.Elt)
		}

	case *ast.FuncLit:
		saved := f.sig
		f.sig = tv.Type.(*types.Signature)
		f.stmt(e.Body)
		f.sig = saved

	case *ast.CompositeLit:
		switch T := deref(tv.Type).Underlying().(type) {
		case *types.Struct:
			for i, elem := range e.Elts {
				if kv, ok := elem.(*ast.KeyValueExpr); ok {
					f.assign(f.info.Uses[kv.Key.(*ast.Ident)].Type(), f.expr(kv.Value))
				} else {
					f.assign(T.Field(i).Type(), f.expr(elem))
				}
			}

		case *types.Map:
			for _, elem := range e.Elts {
				elem := elem.(*ast.KeyValueExpr)
				f.assign(T.Key(), f.expr(elem.Key))
				f.assign(T.Elem(), f.expr(elem.Value))
			}

		case *types.Array, *types.Slice:
			tElem := T.(interface {
				Elem() types.Type
			}).Elem()
			for _, elem := range e.Elts {
				if kv, ok := elem.(*ast.KeyValueExpr); ok {
					// ignore the key
					f.assign(tElem, f.expr(kv.Value))
				} else {
					f.assign(tElem, f.expr(elem))
				}
			}

		default:
			panic("unexpected composite literal type: " + tv.Type.String())
		}

	case *ast.ParenExpr:
		f.expr(e.X)

	case *ast.SelectorExpr:
		if _, ok := f.info.Selections[e]; ok {
			f.expr(e.X) // selection
		} else {
			return f.info.Uses[e.Sel].Type() // qualified identifier
		}

	case *ast.IndexExpr:
		x := f.expr(e.X)
		i := f.expr(e.Index)
		if ux, ok := x.Underlying().(*types.Map); ok {
			f.assign(ux.Key(), i)
		}

	case *ast.SliceExpr:
		f.expr(e.X)
		if e.Low != nil {
			f.expr(e.Low)
		}
		if e.High != nil {
			f.expr(e.High)
		}
		if e.Max != nil {
			f.expr(e.Max)
		}

	case *ast.TypeAssertExpr:
		x := f.expr(e.X)
		f.typeAssert(x, f.info.Types[e.Type].Type)

	case *ast.CallExpr:
		if tvFun := f.info.Types[e.Fun]; tvFun.IsType() {
			// conversion
			arg0 := f.expr(e.Args[0])
			f.assign(tvFun.Type, arg0)
		} else {
			// function call
			if id, ok := unparen(e.Fun).(*ast.Ident); ok {
				if obj, ok := f.info.Uses[id].(*types.Builtin); ok {
					sig := f.info.Types[id].Type.(*types.Signature)
					return f.builtin(obj, sig, e.Args, tv.Type)
				}
			}
			// ordinary call
			f.call(f.expr(e.Fun).Underlying().(*types.Signature), e.Args)
		}

	case *ast.StarExpr:
		f.expr(e.X)

	case *ast.UnaryExpr:
		f.expr(e.X)

	case *ast.BinaryExpr:
		x := f.expr(e.X)
		y := f.expr(e.Y)
		if e.Op == token.EQL || e.Op == token.NEQ {
			f.compare(x, y)
		}

	case *ast.KeyValueExpr:
		f.expr(e.Key)
		f.expr(e.Value)

	case *ast.ArrayType,
		*ast.StructType,
		*ast.FuncType,
		*ast.InterfaceType,
		*ast.MapType,
		*ast.ChanType:
		panic(e)
	}

	if tv.Type == nil {
		panic(fmt.Sprintf("no type for %T", e))
	}

	return tv.Type
}

func (f *Finder) stmt(s ast.Stmt) {
	switch s := s.(type) {
	case *ast.BadStmt,
		*ast.EmptyStmt,
		*ast.BranchStmt:
		// no-op

	case *ast.DeclStmt:
		d := s.Decl.(*ast.GenDecl)
		if d.Tok == token.VAR { // ignore consts
			for _, spec := range d.Specs {
				f.valueSpec(spec.(*ast.ValueSpec))
			}
		}

	case *ast.LabeledStmt:
		f.stmt(s.Stmt)

	case *ast.ExprStmt:
		f.expr(s.X)

	case *ast.SendStmt:
		ch := f.expr(s.Chan)
		val := f.expr(s.Value)
		f.assign(ch.Underlying().(*types.Chan).Elem(), val)

	case *ast.IncDecStmt:
		f.expr(s.X)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			// y := x   or   y = x
			var rhsTuple types.Type
			if len(s.Lhs) != len(s.Rhs) {
				rhsTuple = f.exprN(s.Rhs[0])
			}
			for i := range s.Lhs {
				var lhs, rhs types.Type
				if rhsTuple == nil {
					rhs = f.expr(s.Rhs[i]) // 1:1 assignment
				} else {
					rhs = f.extract(rhsTuple, i) // n:1 assignment
				}

				if id, ok := s.Lhs[i].(*ast.Ident); ok {
					if id.Name != "_" {
						if obj, ok := f.info.Defs[id]; ok {
							lhs = obj.Type() // definition
						}
					}
				}
				if lhs == nil {
					lhs = f.expr(s.Lhs[i]) // assignment
				}
				f.assign(lhs, rhs)
			}

		default:
			// y op= x
			f.expr(s.Lhs[0])
			f.expr(s.Rhs[0])
		}

	case *ast.GoStmt:
		f.expr(s.Call)

	case *ast.DeferStmt:
		f.expr(s.Call)

	case *ast.ReturnStmt:
		formals := f.sig.Results()
		switch len(s.Results) {
		case formals.Len(): // 1:1
			for i, result := range s.Results {
				f.assign(formals.At(i).Type(), f.expr(result))
			}

		case 1: // n:1
			tuple := f.exprN(s.Results[0])
			for i := 0; i < formals.Len(); i++ {
				f.assign(formals.At(i).Type(), f.extract(tuple, i))
			}
		}

	case *ast.SelectStmt:
		f.stmt(s.Body)

	case *ast.BlockStmt:
		for _, s := range s.List {
			f.stmt(s)
		}

	case *ast.IfStmt:
		if s.Init != nil {
			f.stmt(s.Init)
		}
		f.expr(s.Cond)
		f.stmt(s.Body)
		if s.Else != nil {
			f.stmt(s.Else)
		}

	case *ast.SwitchStmt:
		if s.Init != nil {
			f.stmt(s.Init)
		}
		var tag types.Type = tUntypedBool
		if s.Tag != nil {
			tag = f.expr(s.Tag)
		}
		for _, cc := range s.Body.List {
			cc := cc.(*ast.CaseClause)
			for _, cond := range cc.List {
				f.compare(tag, f.info.Types[cond].Type)
			}
			for _, s := range cc.Body {
				f.stmt(s)
			}
		}

	case *ast.TypeSwitchStmt:
		if s.Init != nil {
			f.stmt(s.Init)
		}
		var I types.Type
		switch ass := s.Assign.(type) {
		case *ast.ExprStmt: // x.(type)
			I = f.expr(unparen(ass.X).(*ast.TypeAssertExpr).X)
		case *ast.AssignStmt: // y := x.(type)
			I = f.expr(unparen(ass.Rhs[0]).(*ast.TypeAssertExpr).X)
		}
		for _, cc := range s.Body.List {
			cc := cc.(*ast.CaseClause)
			for _, cond := range cc.List {
				tCase := f.info.Types[cond].Type
				if tCase != tUntypedNil {
					f.typeAssert(I, tCase)
				}
			}
			for _, s := range cc.Body {
				f.stmt(s)
			}
		}

	case *ast.CommClause:
		if s.Comm != nil {
			f.stmt(s.Comm)
		}
		for _, s := range s.Body {
			f.stmt(s)
		}

	case *ast.ForStmt:
		if s.Init != nil {
			f.stmt(s.Init)
		}
		if s.Cond != nil {
			f.expr(s.Cond)
		}
		if s.Post != nil {
			f.stmt(s.Post)
		}
		f.stmt(s.Body)

	case *ast.RangeStmt:
		x := f.expr(s.X)
		// No conversions are involved when Tok==DEFINE.
		if s.Tok == token.ASSIGN {
			if s.Key != nil {
				k := f.expr(s.Key)
				var xelem types.Type
				// keys of array, *array, slice, string aren't interesting
				switch ux := x.Underlying().(type) {
				case *types.Chan:
					xelem = ux.Elem()
				case *types.Map:
					xelem = ux.Key()
				}
				if xelem != nil {
					f.assign(xelem, k)
				}
			}
			if s.Value != nil {
				val := f.expr(s.Value)
				var xelem types.Type
				// values of strings aren't interesting
				switch ux := x.Underlying().(type) {
				case *types.Array:
					xelem = ux.Elem()
				case *types.Chan:
					xelem = ux.Elem()
				case *types.Map:
					xelem = ux.Elem()
				case *types.Pointer: // *array
					xelem = deref(ux).(*types.Array).Elem()
				case *types.Slice:
					xelem = ux.Elem()
				}
				if xelem != nil {
					f.assign(xelem, val)
				}
			}
		}
		f.stmt(s.Body)

	default:
		panic(s)
	}
}

// -- Plundered from golang.org/x/tools/go/ssa -----------------

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

func unparen(e ast.Expr) ast.Expr { return astutil.Unparen(e) }

func isInterface(T types.Type) bool { return types.IsInterface(T) }
