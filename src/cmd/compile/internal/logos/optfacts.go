// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"
	"strings"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

// This file carries the M3-S7 option facts (SPEC T-Sec6, T-202). The
// writer emits lowered Go-facing types in __.PKGDEF; EmitOptionFacts
// records one opt line per option position for .lgo importers, and
// CloneWithOverlay rebuilds those positions as Option for a .lgo view
// using only the public types2 API. Go importers keep the lowered types.

func optAsOption(t types2.Type) *types2.Option {
	if t == nil {
		return nil
	}
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		return o
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if u := n.Underlying(); u != nil {
			if o, ok := u.(*types2.Option); ok {
				return o
			}
		}
	}
	return nil
}

// EmitOptionFacts returns one opt record per option position in the
// exported declarations of pkg, or nil when there are none. Objects are
// func names, type names, and T.M or (*T).M methods; paths are R<i>,
// P<i>, F<name>, E, K, M<name> steps. A nested option extends its
// parent with E, so (T?)? yields R0 and R0.E.
func EmitOptionFacts(pkg *types2.Package, _ *types2.Info) *facts.Facts {
	if pkg == nil || pkg.Scope() == nil {
		return nil
	}
	var opts []facts.PathFact
	emit := func(object, path string) {
		opts = append(opts, facts.PathFact{Object: object, Path: path})
	}
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if obj == nil || !obj.Exported() {
			continue
		}
		switch obj := obj.(type) {
		case *types2.Func:
			sig, ok := obj.Type().(*types2.Signature)
			if !ok || sig.Recv() != nil {
				continue
			}
			if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
				continue
			}
			emitFuncOpts(name, sig, emit)
		case *types2.TypeName:
			if !obj.IsAlias() {
				emitTypeOpts(name, obj.Type(), emit)
			}
		}
	}
	for _, name := range scope.Names() {
		obj, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		named, ok := obj.Type().(*types2.Named)
		if !ok {
			continue
		}
		for i := range named.NumMethods() {
			m := named.Method(i)
			if m == nil || !m.Exported() {
				continue
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
				continue
			}
			emitFuncOpts(methodObject(name, sig.Recv(), m.Name()), sig, emit)
		}
	}
	if len(opts) == 0 {
		return nil
	}
	return &facts.Facts{Opts: opts}
}

func methodObject(typeName string, recv *types2.Var, method string) string {
	if recv == nil {
		return typeName + "." + method
	}
	if _, ok := types2.Unalias(recv.Type()).(*types2.Pointer); ok {
		return "(*" + typeName + ")." + method
	}
	return typeName + "." + method
}

func emitFuncOpts(object string, sig *types2.Signature, emit func(string, string)) {
	if params := sig.Params(); params != nil {
		for i := range params.Len() {
			v := params.At(i)
			if v == nil {
				continue
			}
			walkOptType(v.Type(), []string{fmt.Sprintf("P%d", i)}, object, emit, make(map[types2.Type]bool))
		}
	}
	if results := sig.Results(); results != nil {
		for i := range results.Len() {
			v := results.At(i)
			if v == nil {
				continue
			}
			walkOptType(v.Type(), []string{fmt.Sprintf("R%d", i)}, object, emit, make(map[types2.Type]bool))
		}
	}
}

func emitTypeOpts(name string, t types2.Type, emit func(string, string)) {
	u, ok := types2.Unalias(t).(*types2.Named)
	if !ok {
		return
	}
	if tp := u.TypeParams(); tp != nil && tp.Len() > 0 {
		return
	}
	switch u := u.Underlying().(type) {
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkOptType(f.Type(), []string{"F" + f.Name()}, name, emit, make(map[types2.Type]bool))
		}
	case *types2.Interface:
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil || !m.Exported() {
				continue
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			base := []string{"M" + m.Name()}
			if params := sig.Params(); params != nil {
				for j := range params.Len() {
					v := params.At(j)
					if v == nil {
						continue
					}
					walkOptType(v.Type(), append(slices.Clone(base), fmt.Sprintf("P%d", j)), name, emit, make(map[types2.Type]bool))
				}
			}
			if results := sig.Results(); results != nil {
				for j := range results.Len() {
					v := results.At(j)
					if v == nil {
						continue
					}
					walkOptType(v.Type(), append(slices.Clone(base), fmt.Sprintf("R%d", j)), name, emit, make(map[types2.Type]bool))
				}
			}
		}
	}
}

func walkOptType(t types2.Type, path []string, object string, emit func(string, string), seen map[types2.Type]bool) {
	if t == nil {
		return
	}
	if opt := optAsOption(t); opt != nil {
		emit(object, joinPath(path))
		walkOptType(opt.Elem, append(slices.Clone(path), "E"), object, emit, seen)
		return
	}
	u := types2.Unalias(t)
	switch u := u.(type) {
	case *types2.Slice:
		walkOptType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Array:
		walkOptType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Map:
		walkOptType(u.Key(), append(slices.Clone(path), "K"), object, emit, seen)
		walkOptType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Pointer:
		walkOptType(u.Elem(), path, object, emit, seen)
	case *types2.Chan:
		walkOptType(u.Elem(), path, object, emit, seen)
	case *types2.Signature:
		if params := u.Params(); params != nil {
			for i := range params.Len() {
				v := params.At(i)
				if v == nil {
					continue
				}
				walkOptType(v.Type(), append(slices.Clone(path), fmt.Sprintf("P%d", i)), object, emit, seen)
			}
		}
		if results := u.Results(); results != nil {
			for i := range results.Len() {
				v := results.At(i)
				if v == nil {
					continue
				}
				walkOptType(v.Type(), append(slices.Clone(path), fmt.Sprintf("R%d", i)), object, emit, seen)
			}
		}
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkOptType(f.Type(), append(slices.Clone(path), "F"+f.Name()), object, emit, seen)
		}
	case *types2.Interface:
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil || !m.Exported() {
				continue
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			base := append(slices.Clone(path), "M"+m.Name())
			if params := sig.Params(); params != nil {
				for j := range params.Len() {
					v := params.At(j)
					if v == nil {
						continue
					}
					walkOptType(v.Type(), append(slices.Clone(base), fmt.Sprintf("P%d", j)), object, emit, seen)
				}
			}
			if results := sig.Results(); results != nil {
				for j := range results.Len() {
					v := results.At(j)
					if v == nil {
						continue
					}
					walkOptType(v.Type(), append(slices.Clone(base), fmt.Sprintf("R%d", j)), object, emit, seen)
				}
			}
		}
	case *types2.Named:
		if seen[u] {
			return
		}
		seen[u] = true
		walkOptType(u.Underlying(), path, object, emit, seen)
		delete(seen, u)
	}
}

func joinPath(path []string) string {
	return strings.Join(path, ".")
}

// CloneWithOverlay rebuilds the exported declarations named by opts as
// Option for a .lgo view and returns the new package. The input package
// is untouched, so Go importers sharing nothing keep the lowered types.
// Only func results/params, struct fields, and method results/params at
// the recorded paths are converted; everything else shares the input
// objects. A lowered bare payload wraps as Option; an instantiated
// logos/rt.Option[T] becomes Option with Elem T.
func CloneWithOverlay(orig *types2.Package, opts []facts.PathFact) *types2.Package {
	if orig == nil || len(opts) == 0 {
		return orig
	}
	byObj := make(map[string][]string)
	for _, o := range opts {
		if o.Object != "" && o.Path != "" {
			byObj[o.Object] = append(byObj[o.Object], o.Path)
		}
	}
	if len(byObj) == 0 {
		return orig
	}
	out := types2.NewPackage(orig.Path(), orig.Name())
	out.SetImports(orig.Imports())
	needClone := func(name string) bool {
		if _, ok := byObj[name]; ok {
			return true
		}
		for obj := range byObj {
			if recv, mname, ok := splitMethodObject(obj); ok && mname != "" {
				if recv == name || recv == "(*"+name+")" {
					return true
				}
			}
		}
		return false
	}
	for _, name := range orig.Scope().Names() {
		obj := orig.Scope().Lookup(name)
		if obj == nil {
			continue
		}
		if !needClone(name) {
			out.Scope().Insert(obj)
			continue
		}
		switch obj := obj.(type) {
		case *types2.Func:
			sig, ok := obj.Type().(*types2.Signature)
			if !ok || sig.Recv() != nil {
				out.Scope().Insert(obj)
				continue
			}
			out.Scope().Insert(cloneFunc(out, obj, sig, byObj[name]))
		case *types2.TypeName:
			if obj.IsAlias() {
				out.Scope().Insert(obj)
				continue
			}
			out.Scope().Insert(cloneTypeName(out, obj, byObj))
		default:
			out.Scope().Insert(obj)
		}
	}
	out.MarkComplete()
	return out
}

func splitMethodObject(s string) (recv, name string, ok bool) {
	recv, name, ok = strings.CutLast(s, ".")
	if !ok || recv == "" || name == "" {
		return "", "", false
	}
	if strings.Contains(name, ".") {
		return "", "", false
	}
	return recv, name, true
}

func cloneFunc(pkg *types2.Package, fn *types2.Func, sig *types2.Signature, paths []string) *types2.Func {
	params := cloneTuple(pkg, sig.Params(), paths, "P")
	results := cloneTuple(pkg, sig.Results(), paths, "R")
	nsig := types2.NewSignatureType(nil, nil, nil, params, results, sig.Variadic())
	nf := types2.NewFunc(fn.Pos(), pkg, fn.Name(), nsig)
	return nf
}

func cloneTuple(pkg *types2.Package, tup *types2.Tuple, paths []string, prefix string) *types2.Tuple {
	if tup == nil {
		return nil
	}
	vars := make([]*types2.Var, tup.Len())
	for i := range tup.Len() {
		v := tup.At(i)
		t := v.Type()
		for _, p := range paths {
			steps := strings.Split(p, ".")
			if len(steps) == 0 {
				continue
			}
			head, rest := steps[0], steps[1:]
			var idx int
			if _, err := fmt.Sscanf(head, prefix+"%d", &idx); err != nil || idx != i {
				continue
			}
			t = wrapAt(t, rest)
		}
		nv := types2.NewVar(v.Pos(), pkg, v.Name(), t)
		nv.SetKind(v.Kind())
		vars[i] = nv
	}
	if len(vars) == 0 {
		return nil
	}
	return types2.NewTuple(vars...)
}

func cloneTypeName(pkg *types2.Package, tn *types2.TypeName, byObj map[string][]string) *types2.TypeName {
	named, ok := tn.Type().(*types2.Named)
	if !ok {
		return tn
	}
	methodPaths := map[string][]string{}
	for obj, paths := range byObj {
		if recv, mname, ok := splitMethodObject(obj); ok {
			if recv == tn.Name() || recv == "(*"+tn.Name()+")" {
				methodPaths[mname] = append(methodPaths[mname], paths...)
			}
		}
	}
	nt := types2.NewTypeName(tn.Pos(), pkg, tn.Name(), nil)
	switch u := named.Underlying().(type) {
	case *types2.Struct:
		fieldPaths := map[string][]string{}
		for _, p := range byObj[tn.Name()] {
			steps := strings.Split(p, ".")
			if len(steps) == 0 || steps[0][0] != 'F' {
				continue
			}
			fieldPaths[steps[0][1:]] = append(fieldPaths[steps[0][1:]], strings.Join(steps[1:], "."))
		}
		var fields []*types2.Var
		var tags []string
		for i := range u.NumFields() {
			f := u.Field(i)
			t := f.Type()
			if rests, ok := fieldPaths[f.Name()]; ok {
				for _, r := range rests {
					if r == "" {
						t = toOption(t)
					} else {
						t = wrapAt(t, strings.Split(r, "."))
					}
				}
			}
			nf := types2.NewField(f.Pos(), pkg, f.Name(), t, f.Embedded())
			fields = append(fields, nf)
			tags = append(tags, u.Tag(i))
		}
		under := types2.NewStruct(fields, tags)
		named2 := types2.NewNamed(nt, under, nil)
		for i := range named.NumMethods() {
			m := named.Method(i)
			named2.AddMethod(cloneMethod(pkg, named2, m, methodPaths[m.Name()]))
		}
		return nt
	default:
		if len(methodPaths) == 0 && len(byObj[tn.Name()]) == 0 {
			return tn
		}
		named2 := types2.NewNamed(nt, u, nil)
		for i := range named.NumMethods() {
			m := named.Method(i)
			named2.AddMethod(cloneMethod(pkg, named2, m, methodPaths[m.Name()]))
		}
		return nt
	}
}

func cloneMethod(pkg *types2.Package, named2 *types2.Named, m *types2.Func, paths []string) *types2.Func {
	sig, ok := m.Type().(*types2.Signature)
	if !ok {
		return m
	}
	recv := sig.Recv()
	var nrecv *types2.Var
	if recv != nil {
		rt := recv.Type()
		if _, ok := types2.Unalias(rt).(*types2.Pointer); ok {
			rt = types2.NewPointer(named2)
		} else {
			rt = named2
		}
		nrecv = types2.NewVar(recv.Pos(), pkg, recv.Name(), rt)
		nrecv.SetKind(recv.Kind())
	}
	params := cloneTuple(pkg, sig.Params(), paths, "P")
	results := cloneTuple(pkg, sig.Results(), paths, "R")
	nsig := types2.NewSignatureType(nrecv, nil, nil, params, results, sig.Variadic())
	nm := types2.NewFunc(m.Pos(), pkg, m.Name(), nsig)
	return nm
}

// wrapAt converts t at the remaining steps: empty steps wrap the
// position itself; E steps descend into option/slice/map elems; F/K/M
// steps descend into structs/maps/interfaces.
func wrapAt(t types2.Type, steps []string) types2.Type {
	if len(steps) == 0 {
		return toOption(t)
	}
	if t == nil {
		return t
	}
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		elem := wrapAt(o.Elem, steps)
		if elem == o.Elem {
			return t
		}
		return &types2.Option{Elem: elem}
	}
	u := types2.Unalias(t)
	switch u := u.(type) {
	case *types2.Slice:
		if steps[0] != "E" {
			return t
		}
		return types2.NewSlice(wrapAt(u.Elem(), steps[1:]))
	case *types2.Array:
		if steps[0] != "E" {
			return t
		}
		return types2.NewArray(wrapAt(u.Elem(), steps[1:]), u.Len())
	case *types2.Map:
		if steps[0] == "K" {
			return types2.NewMap(wrapAt(u.Key(), steps[1:]), u.Elem())
		}
		if steps[0] == "E" {
			return types2.NewMap(u.Key(), wrapAt(u.Elem(), steps[1:]))
		}
		return t
	case *types2.Pointer:
		return wrapAt(u.Elem(), steps)
	case *types2.Chan:
		return wrapAt(u.Elem(), steps)
	case *types2.Struct:
		if steps[0][0] != 'F' {
			return t
		}
		return t
	case *types2.Named:
		return t
	default:
		return t
	}
}

// toOption converts a lowered type to Option: rt.Option[T] becomes
// Option with Elem T; any other type wraps as Option.
func toOption(t types2.Type) types2.Type {
	if t == nil {
		return &types2.Option{}
	}
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		return o
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if obj := n.Obj(); obj != nil && obj.Name() == "Option" && obj.Pkg() != nil && obj.Pkg().Path() == "logos/rt" {
			if targs := n.TypeArgs(); targs != nil && targs.Len() == 1 {
				return &types2.Option{Elem: targs.At(0)}
			}
		}
	}
	return &types2.Option{Elem: t}
}
