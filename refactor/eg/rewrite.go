// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eg

// This file defines the AST rewriting pass.
// Most of it was plundered directly from
// $GOROOT/src/cmd/gofmt/rewrite.go (after convergent evolution).

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

// Transform applies the transformation to the specified parsed file,
// whose type information is supplied in info, and returns the number
// of replacements that were made.
//
// It mutates the AST in place (the identity of the root node is
// unchanged), and may add nodes for which no type information is
// available in info.
//
// Derived from rewriteFile in $GOROOT/src/cmd/gofmt/rewrite.go.
//
func (tr *Transformer) Transform(info *types.Info, pkg *types.Package, file *ast.File) int {
	if !tr.seenInfos[info] {
		tr.seenInfos[info] = true
		mergeTypeInfo(tr.info, info)
	}
	tr.currentPkg = pkg
	tr.nsubsts = 0

	if tr.verbose {
		fmt.Fprintf(os.Stderr, "before: %s\n", astString(tr.fset, tr.before))
		fmt.Fprintf(os.Stderr, "after: %s\n", astString(tr.fset, tr.after))
	}

	var f func(rv reflect.Value) reflect.Value
	f = func(rv reflect.Value) reflect.Value {
		// don't bother if val is invalid to start with
		if !rv.IsValid() {
			return reflect.Value{}
		}

		rv = apply(f, rv)

		e := rvToExpr(rv)
		if e != nil {
			savedEnv := tr.env
			tr.env = make(map[string]ast.Expr) // inefficient!  Use a slice of k/v pairs

			if tr.matchExpr(tr.before, e) {
				if tr.verbose {
					fmt.Fprintf(os.Stderr, "%s matches %s",
						astString(tr.fset, tr.before), astString(tr.fset, e))
					if len(tr.env) > 0 {
						fmt.Fprintf(os.Stderr, " with:")
						for name, ast := range tr.env {
							fmt.Fprintf(os.Stderr, " %s->%s",
								name, astString(tr.fset, ast))
						}
					}
					fmt.Fprintf(os.Stderr, "\n")
				}
				tr.nsubsts++

				// Clone the replacement tree, performing parameter substitution.
				// We update all positions to n.Pos() to aid comment placement.
				rv = tr.subst(tr.env, reflect.ValueOf(tr.after),
					reflect.ValueOf(e.Pos()))
			}
			tr.env = savedEnv
		}

		return rv
	}
	file2 := apply(f, reflect.ValueOf(file)).Interface().(*ast.File)

	// By construction, the root node is unchanged.
	if file != file2 {
		panic("BUG")
	}

	// Add any necessary imports.
	// TODO(adonovan): remove no-longer needed imports too.
	if tr.nsubsts > 0 {
		pkgs := make(map[string]*types.Package)
		for obj := range tr.importedObjs {
			pkgs[obj.Pkg().Path()] = obj.Pkg()
		}

		for _, imp := range file.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			delete(pkgs, path)
		}
		delete(pkgs, pkg.Path()) // don't import self

		// NB: AddImport may completely replace the AST!
		// It thus renders info and tr.info no longer relevant to file.
		var paths []string
		for path := range pkgs {
			paths = append(paths, path)
		}
		sort.Strings(paths)
		for _, path := range paths {
			astutil.AddImport(tr.fset, file, path)
		}
	}

	tr.currentPkg = nil

	return tr.nsubsts
}

// setValue is a wrapper for x.SetValue(y); it protects
// the caller from panics if x cannot be changed to y.
func setValue(x, y reflect.Value) {
	// don't bother if y is invalid to start with
	if !y.IsValid() {
		return
	}
	defer func() {
		if x := recover(); x != nil {
			if s, ok := x.(string); ok &&
				(strings.Contains(s, "type mismatch") || strings.Contains(s, "not assignable")) {
				// x cannot be set to y - ignore this rewrite
				return
			}
			panic(x)
		}
	}()
	x.Set(y)
}

// Values/types for special cases.
var (
	objectPtrNil = reflect.ValueOf((*ast.Object)(nil))
	scopePtrNil  = reflect.ValueOf((*ast.Scope)(nil))

	identType        = reflect.TypeOf((*ast.Ident)(nil))
	selectorExprType = reflect.TypeOf((*ast.SelectorExpr)(nil))
	objectPtrType    = reflect.TypeOf((*ast.Object)(nil))
	positionType     = reflect.TypeOf(token.NoPos)
	scopePtrType     = reflect.TypeOf((*ast.Scope)(nil))
)

// apply replaces each AST field x in val with f(x), returning val.
// To avoid extra conversions, f operates on the reflect.Value form.
func apply(f func(reflect.Value) reflect.Value, val reflect.Value) reflect.Value {
	if !val.IsValid() {
		return reflect.Value{}
	}

	// *ast.Objects introduce cycles and are likely incorrect after
	// rewrite; don't follow them but replace with nil instead
	if val.Type() == objectPtrType {
		return objectPtrNil
	}

	// similarly for scopes: they are likely incorrect after a rewrite;
	// replace them with nil
	if val.Type() == scopePtrType {
		return scopePtrNil
	}

	switch v := reflect.Indirect(val); v.Kind() {
	case reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			e := v.Index(i)
			setValue(e, f(e))
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			e := v.Field(i)
			setValue(e, f(e))
		}
	case reflect.Interface:
		e := v.Elem()
		setValue(v, f(e))
	}
	return val
}

// subst returns a copy of (replacement) pattern with values from env
// substituted in place of wildcards and pos used as the position of
// tokens from the pattern.  if env == nil, subst returns a copy of
// pattern and doesn't change the line number information.
func (tr *Transformer) subst(env map[string]ast.Expr, pattern, pos reflect.Value) reflect.Value {
	if !pattern.IsValid() {
		return reflect.Value{}
	}

	// *ast.Objects introduce cycles and are likely incorrect after
	// rewrite; don't follow them but replace with nil instead
	if pattern.Type() == objectPtrType {
		return objectPtrNil
	}

	// similarly for scopes: they are likely incorrect after a rewrite;
	// replace them with nil
	if pattern.Type() == scopePtrType {
		return scopePtrNil
	}

	// Wildcard gets replaced with map value.
	if env != nil && pattern.Type() == identType {
		id := pattern.Interface().(*ast.Ident)
		if old, ok := env[id.Name]; ok {
			return tr.subst(nil, reflect.ValueOf(old), reflect.Value{})
		}
	}

	// Emit qualified identifiers in the pattern by appropriate
	// (possibly qualified) identifier in the input.
	//
	// The template cannot contain dot imports, so all identifiers
	// for imported objects are explicitly qualified.
	//
	// We assume (unsoundly) that there are no dot or named
	// imports in the input code, nor are any imported package
	// names shadowed, so the usual normal qualified identifier
	// syntax may be used.
	// TODO(adonovan): fix: avoid this assumption.
	//
	// A refactoring may be applied to a package referenced by the
	// template.  Objects belonging to the current package are
	// denoted by unqualified identifiers.
	//
	if tr.importedObjs != nil && pattern.Type() == selectorExprType {
		obj := isRef(pattern.Interface().(*ast.SelectorExpr), tr.info)
		if obj != nil {
			if sel, ok := tr.importedObjs[obj]; ok {
				var id ast.Expr
				if obj.Pkg() == tr.currentPkg {
					id = sel.Sel // unqualified
				} else {
					id = sel // pkg-qualified
				}

				// Return a clone of id.
				saved := tr.importedObjs
				tr.importedObjs = nil // break cycle
				r := tr.subst(nil, reflect.ValueOf(id), pos)
				tr.importedObjs = saved
				return r
			}
		}
	}

	if pos.IsValid() && pattern.Type() == positionType {
		// use new position only if old position was valid in the first place
		if old := pattern.Interface().(token.Pos); !old.IsValid() {
			return pattern
		}
		return pos
	}

	// Otherwise copy.
	switch p := pattern; p.Kind() {
	case reflect.Slice:
		v := reflect.MakeSlice(p.Type(), p.Len(), p.Len())
		for i := 0; i < p.Len(); i++ {
			v.Index(i).Set(tr.subst(env, p.Index(i), pos))
		}
		return v

	case reflect.Struct:
		v := reflect.New(p.Type()).Elem()
		for i := 0; i < p.NumField(); i++ {
			v.Field(i).Set(tr.subst(env, p.Field(i), pos))
		}
		return v

	case reflect.Ptr:
		v := reflect.New(p.Type()).Elem()
		if elem := p.Elem(); elem.IsValid() {
			v.Set(tr.subst(env, elem, pos).Addr())
		}

		// Duplicate type information for duplicated ast.Expr.
		// All ast.Node implementations are *structs,
		// so this case catches them all.
		if e := rvToExpr(v); e != nil {
			updateTypeInfo(tr.info, e, p.Interface().(ast.Expr))
		}
		return v

	case reflect.Interface:
		v := reflect.New(p.Type()).Elem()
		if elem := p.Elem(); elem.IsValid() {
			v.Set(tr.subst(env, elem, pos))
		}
		return v
	}

	return pattern
}

// -- utilities -------------------------------------------------------

func rvToExpr(rv reflect.Value) ast.Expr {
	if rv.CanInterface() {
		if e, ok := rv.Interface().(ast.Expr); ok {
			return e
		}
	}
	return nil
}

// updateTypeInfo duplicates type information for the existing AST old
// so that it also applies to duplicated AST new.
func updateTypeInfo(info *types.Info, new, old ast.Expr) {
	switch new := new.(type) {
	case *ast.Ident:
		orig := old.(*ast.Ident)
		if obj, ok := info.Defs[orig]; ok {
			info.Defs[new] = obj
		}
		if obj, ok := info.Uses[orig]; ok {
			info.Uses[new] = obj
		}

	case *ast.SelectorExpr:
		orig := old.(*ast.SelectorExpr)
		if sel, ok := info.Selections[orig]; ok {
			info.Selections[new] = sel
		}
	}

	if tv, ok := info.Types[old]; ok {
		info.Types[new] = tv
	}
}
