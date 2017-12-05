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
	desc:     `Fixes initializers of C.CF*Ptr types`,
	disabled: false,
}

// Old state:
//   type CFTypeRef unsafe.Pointer
// New state:
//   type CFTypeRef uintptr
// and similar for other CF*Ref types.
// This fix finds nils initializing these types and replaces the nils with 0s.
func cftypefix(f *ast.File) bool {
	return typefix(f, func(s string) bool {
		return strings.HasPrefix(s, "C.CF") && strings.HasSuffix(s, "Ref")
	})
}

// typefix replaces nil with 0 for all nils whose type, when passed to badType, returns true.
func typefix(f *ast.File, badType func(string) bool) bool {
	if !imports(f, "C") {
		return false
	}
	typeof, _ := typecheck(&TypeConfig{}, f)

	// step 1: Find all the nils with the offending types.
	// Compute their replacement.
	badNils := map[interface{}]ast.Expr{}
	walk(f, func(n interface{}) {
		if i, ok := n.(*ast.Ident); ok && i.Name == "nil" && badType(typeof[n]) {
			badNils[n] = &ast.BasicLit{ValuePos: i.NamePos, Kind: token.INT, Value: "0"}
		}
	})
	if len(badNils) == 0 {
		return false
	}

	// step 2: find all uses of the bad nils, replace them with 0.
	// There's no easy way to map from an ast.Expr to all the places that use them, so
	// we use reflect to find all such references.
	exprType := reflect.TypeOf((*ast.Expr)(nil)).Elem()
	exprSliceType := reflect.TypeOf(([]ast.Expr)(nil))
	walk(f, func(n interface{}) {
		if n == nil {
			return
		}
		v := reflect.ValueOf(n)
		if v.Type().Kind() != reflect.Ptr {
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
				}
			}
			if f.Type() == exprSliceType {
				for j := 0; j < f.Len(); j++ {
					e := f.Index(j)
					if r := badNils[e.Interface()]; r != nil {
						e.Set(reflect.ValueOf(r))
					}
				}
			}
		}
	})

	return true
}
