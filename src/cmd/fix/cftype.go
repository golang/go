// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
	"reflect"
	"strings"
)

func init() {
	register(cftypeFix)
}

var cftypeFix = fix{
	name:     "cftype",
	date:     "2017-09-27",
	f:        cftypefix,
	desc:     `Fixes initializers and casts of C.*Ref and JNI types`,
	disabled: false,
}

// Old state:
//
//	type CFTypeRef unsafe.Pointer
//
// New state:
//
//	type CFTypeRef uintptr
//
// and similar for other *Ref types.
// This fix finds nils initializing these types and replaces the nils with 0s.
func cftypefix(f *ast.File) bool {
	return typefix(f, func { s -> strings.HasPrefix(s, "C.") && strings.HasSuffix(s, "Ref") && s != "C.CFAllocatorRef" })
}

// typefix replaces nil with 0 for all nils whose type, when passed to badType, returns true.
func typefix(f *ast.File, badType func(string) bool) bool {
	if !imports(f, "C") {
		return false
	}
	typeof, _ := typecheck(&TypeConfig{}, f)
	changed := false

	// step 1: Find all the nils with the offending types.
	// Compute their replacement.
	badNils := map[any]ast.Expr{}
	walk(f, func { n ->
		if i, ok := n.(*ast.Ident); ok && i.Name == "nil" && badType(typeof[n]) {
			badNils[n] = &ast.BasicLit{ValuePos: i.NamePos, Kind: token.INT, Value: "0"}
		}
	})

	// step 2: find all uses of the bad nils, replace them with 0.
	// There's no easy way to map from an ast.Expr to all the places that use them, so
	// we use reflect to find all such references.
	if len(badNils) > 0 {
		exprType := reflect.TypeFor[ast.Expr]()
		exprSliceType := reflect.TypeFor[[]ast.Expr]()
		walk(f, func { n ->
			if n == nil {
				return
			}
			v := reflect.ValueOf(n)
			if v.Type().Kind() != reflect.Pointer {
				return
			}
			if v.IsNil() {
				return
			}
			v = v.Elem()
			if v.Type().Kind() != reflect.Struct {
				return
			}
			for i := 0; i < v.NumField(); i++ {
				f := v.Field(i)
				if f.Type() == exprType {
					if r := badNils[f.Interface()]; r != nil {
						f.Set(reflect.ValueOf(r))
						changed = true
					}
				}
				if f.Type() == exprSliceType {
					for j := 0; j < f.Len(); j++ {
						e := f.Index(j)
						if r := badNils[e.Interface()]; r != nil {
							e.Set(reflect.ValueOf(r))
							changed = true
						}
					}
				}
			}
		})
	}

	// step 3: fix up invalid casts.
	// It used to be ok to cast between *unsafe.Pointer and *C.CFTypeRef in a single step.
	// Now we need unsafe.Pointer as an intermediate cast.
	// (*unsafe.Pointer)(x) where x is type *bad -> (*unsafe.Pointer)(unsafe.Pointer(x))
	// (*bad.type)(x) where x is type *unsafe.Pointer -> (*bad.type)(unsafe.Pointer(x))
	walk(f, func { n ->
		if n == nil {
			return
		}
		// Find pattern like (*a.b)(x)
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		if len(c.Args) != 1 {
			return
		}
		p, ok := c.Fun.(*ast.ParenExpr)
		if !ok {
			return
		}
		s, ok := p.X.(*ast.StarExpr)
		if !ok {
			return
		}
		t, ok := s.X.(*ast.SelectorExpr)
		if !ok {
			return
		}
		pkg, ok := t.X.(*ast.Ident)
		if !ok {
			return
		}
		dst := pkg.Name + "." + t.Sel.Name
		src := typeof[c.Args[0]]
		if badType(dst) && src == "*unsafe.Pointer" ||
			dst == "unsafe.Pointer" && strings.HasPrefix(src, "*") && badType(src[1:]) {
			c.Args[0] = &ast.CallExpr{
				Fun:  &ast.SelectorExpr{X: &ast.Ident{Name: "unsafe"}, Sel: &ast.Ident{Name: "Pointer"}},
				Args: []ast.Expr{c.Args[0]},
			}
			changed = true
		}
	})

	return changed
}
