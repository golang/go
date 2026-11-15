// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/types2"
	"cmd/internal/objabi"
)

func staticTypeName(t types2.Type) (string, bool) {
	var w typeNameWriter
	w.typ(t)
	return w.b.String(), !w.dynamic
}

func staticTypeNameLit(t types2.Type) string {
	if name, ok := staticTypeName(t); ok {
		return strconv.Quote(name)
	}
	return `""`
}

type typeNameWriter struct {
	b       strings.Builder
	link    bool
	dynamic bool
}

func (w *typeNameWriter) typ(t types2.Type) {
	switch t := types2.Unalias(t).(type) {
	case *types2.Basic:
		w.basic(t)
	case *types2.Named:
		w.named(t)
	case *types2.Pointer, *types2.Slice, *types2.Array, *types2.Map, *types2.Chan:
		w.composite(t)
	case *types2.Signature, *types2.Struct, *types2.Interface:
		w.literal(t)
	default:
		w.dynamic = true
	}
}

func (w *typeNameWriter) basic(t *types2.Basic) {
	switch {
	case t.Info()&types2.IsUntyped != 0 || t.Kind() == types2.Invalid:
		w.dynamic = true
	case t.Kind() == types2.UnsafePointer:
		w.b.WriteString("unsafe.Pointer")
	default:
		w.b.WriteString(types2.Typ[t.Kind()].Name())
	}
}

func (w *typeNameWriter) named(t *types2.Named) {
	obj := t.Obj()
	pkg := obj.Pkg()
	if pkg == nil {
		w.b.WriteString(obj.Name())
		return
	}
	if w.unspellable(obj) {
		w.dynamic = true
		return
	}
	w.b.WriteString(w.pkgName(pkg) + "." + obj.Name())
	w.typeArgs(t.TypeArgs())
}

func (w *typeNameWriter) unspellable(obj *types2.TypeName) bool {
	if obj.Parent() == obj.Pkg().Scope() {
		return false
	}
	return w.link || declaredInGeneric(obj)
}

func (w *typeNameWriter) pkgName(pkg *types2.Package) string {
	if w.link {
		return objabi.PathToPrefix(pkg.Path())
	}
	return pkg.Name()
}

func (w *typeNameWriter) typeArgs(args *types2.TypeList) {
	if args.Len() == 0 {
		return
	}
	outer := w.link
	w.link = true
	w.b.WriteByte('[')
	for i := range args.Len() {
		if i > 0 {
			w.b.WriteByte(',')
		}
		w.typ(args.At(i))
	}
	w.b.WriteByte(']')
	w.link = outer
}

func declaredInGeneric(obj *types2.TypeName) bool {
	pkgScope := obj.Pkg().Scope()
	for s := obj.Parent(); s != nil && s != pkgScope; s = s.Parent() {
		if scopeDeclaresTypeParam(s) {
			return true
		}
	}
	return false
}

func scopeDeclaresTypeParam(s *types2.Scope) bool {
	return slices.ContainsFunc(s.Names(), func(name string) bool {
		tn, ok := s.Lookup(name).(*types2.TypeName)
		if !ok {
			return false
		}
		_, ok = tn.Type().(*types2.TypeParam)
		return ok
	})
}

func (w *typeNameWriter) composite(t types2.Type) {
	switch t := t.(type) {
	case *types2.Pointer:
		w.b.WriteByte('*')
		w.typ(t.Elem())
	case *types2.Slice:
		w.b.WriteString("[]")
		w.typ(t.Elem())
	case *types2.Array:
		w.b.WriteString("[" + strconv.FormatInt(t.Len(), 10) + "]")
		w.typ(t.Elem())
	case *types2.Map:
		w.b.WriteString("map[")
		w.typ(t.Key())
		w.b.WriteByte(']')
		w.typ(t.Elem())
	case *types2.Chan:
		w.channel(t)
	}
}

func (w *typeNameWriter) channel(t *types2.Chan) {
	switch t.Dir() {
	case types2.RecvOnly:
		w.b.WriteString("<-chan ")
	case types2.SendOnly:
		w.b.WriteString("chan<- ")
	default:
		w.b.WriteString("chan ")
		if isRecvChan(t.Elem()) {
			w.b.WriteByte('(')
			w.typ(t.Elem())
			w.b.WriteByte(')')
			return
		}
	}
	w.typ(t.Elem())
}

func isRecvChan(t types2.Type) bool {
	c, ok := types2.Unalias(t).(*types2.Chan)
	return ok && c.Dir() == types2.RecvOnly
}

func (w *typeNameWriter) literal(t types2.Type) {
	switch t := t.(type) {
	case *types2.Signature:
		w.b.WriteString("func")
		w.signature(t)
	case *types2.Struct:
		w.structure(t)
	case *types2.Interface:
		w.iface(t)
	}
}

func (w *typeNameWriter) signature(s *types2.Signature) {
	w.params(s.Params(), variadicIndex(s))
	results := s.Results()
	switch results.Len() {
	case 0:
	case 1:
		w.b.WriteByte(' ')
		w.typ(results.At(0).Type())
	default:
		w.b.WriteByte(' ')
		w.params(results, -1)
	}
}

func variadicIndex(s *types2.Signature) int {
	if s.Variadic() {
		return s.Params().Len() - 1
	}
	return -1
}

func (w *typeNameWriter) params(tuple *types2.Tuple, ellipsis int) {
	w.b.WriteByte('(')
	for i := range tuple.Len() {
		if i > 0 {
			w.b.WriteString(", ")
		}
		t := tuple.At(i).Type()
		if i == ellipsis {
			w.b.WriteString("...")
			t = sliceElem(t)
		}
		w.typ(t)
	}
	w.b.WriteByte(')')
}

func sliceElem(t types2.Type) types2.Type {
	if s, ok := types2.Unalias(t).(*types2.Slice); ok {
		return s.Elem()
	}
	return nil
}

func (w *typeNameWriter) structure(s *types2.Struct) {
	w.b.WriteString("struct {")
	for i := range s.NumFields() {
		if i > 0 {
			w.b.WriteByte(';')
		}
		w.b.WriteByte(' ')
		w.field(s.Field(i), s.Tag(i))
	}
	if s.NumFields() > 0 {
		w.b.WriteByte(' ')
	}
	w.b.WriteByte('}')
}

func (w *typeNameWriter) field(f *types2.Var, tag string) {
	if w.link && (f.Embedded() || !f.Exported()) {
		w.dynamic = true
		return
	}
	if !f.Embedded() {
		w.b.WriteString(f.Name() + " ")
	}
	w.typ(f.Type())
	if tag != "" {
		w.b.WriteString(" " + strconv.Quote(tag))
	}
}

func (w *typeNameWriter) iface(t *types2.Interface) {
	if !t.IsMethodSet() || (w.link && t.NumMethods() > 0) {
		w.dynamic = true
		return
	}
	if t.NumMethods() == 0 {
		w.b.WriteString("interface {}")
		return
	}
	methods := make([]*types2.Func, t.NumMethods())
	for i := range methods {
		methods[i] = t.Method(i)
	}
	if slices.ContainsFunc(methods, func(m *types2.Func) bool { return !m.Exported() }) {
		w.dynamic = true
		return
	}
	slices.SortFunc(methods, func(a, b *types2.Func) int {
		return strings.Compare(a.Name(), b.Name())
	})
	w.b.WriteString("interface {")
	for i, m := range methods {
		if i > 0 {
			w.b.WriteByte(';')
		}
		w.b.WriteString(" " + m.Name())
		w.signature(m.Type().(*types2.Signature))
	}
	w.b.WriteString(" }")
}
