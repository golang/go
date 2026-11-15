// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"fmt"
	"slices"

	"cmd/compile/internal/types2"
)

func writeOpts(pkg *types2.Package, f *Facts) {
	if pkg == nil || pkg.Scope() == nil {
		return
	}
	emit := func(object, path string) {
		f.Opts = appendUnique(f.Opts, PathFact{Object: object, Path: path})
	}
	eachExportedSig(pkg, func(object string, sig *types2.Signature) {
		walkSigQuals(sig, object, emit, nil)
	})
	eachExportedNamed(pkg, func(name string, named *types2.Named) {
		walkNamedQuals(name, named, emit, nil)
	})
}

func walkSigQuals(sig *types2.Signature, object string, emitOpt, emitConst func(string, string)) {
	if params := sig.Params(); params != nil {
		for i := range params.Len() {
			v := params.At(i)
			if v == nil {
				continue
			}
			walkQualType(v.Type(), []string{fmt.Sprintf("P%d", i)}, object, emitOpt, emitConst, make(map[types2.Type]bool))
		}
	}
	if results := sig.Results(); results != nil {
		for i := range results.Len() {
			v := results.At(i)
			if v == nil {
				continue
			}
			walkQualType(v.Type(), []string{fmt.Sprintf("R%d", i)}, object, emitOpt, emitConst, make(map[types2.Type]bool))
		}
	}
}

func walkNamedQuals(name string, named *types2.Named, emitOpt, emitConst func(string, string)) {
	u := named.Underlying()
	if tuple, ok := u.(*types2.LogosTuple); ok {
		u = tuple.Underlying()
	}
	switch u := u.(type) {
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkQualType(f.Type(), []string{"F" + f.Name()}, name, emitOpt, emitConst, make(map[types2.Type]bool))
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
					walkQualType(v.Type(), append(slices.Clone(base), fmt.Sprintf("P%d", j)), name, emitOpt, emitConst, make(map[types2.Type]bool))
				}
			}
			if results := sig.Results(); results != nil {
				for j := range results.Len() {
					v := results.At(j)
					if v == nil {
						continue
					}
					walkQualType(v.Type(), append(slices.Clone(base), fmt.Sprintf("R%d", j)), name, emitOpt, emitConst, make(map[types2.Type]bool))
				}
			}
		}
	}
}

func walkQualType(t types2.Type, path []string, object string, emitOpt, emitConst func(string, string), seen map[types2.Type]bool) {
	if t == nil {
		return
	}
	if opt, ok := unaliasAs[*types2.Option](t); ok {
		if emitOpt != nil {
			emitOpt(object, joinPath(path))
		}
		walkQualType(opt.Elem, append(slices.Clone(path), "E"), object, emitOpt, emitConst, seen)
		return
	}
	if c, ok := unaliasAs[*types2.ConstView](t); ok {
		if emitConst != nil {
			emitConst(object, joinPath(path))
		}
		walkQualType(c.Elem, append(slices.Clone(path), "E"), object, emitOpt, emitConst, seen)
		return
	}
	u := types2.Unalias(t)
	if tuple, ok := u.(*types2.LogosTuple); ok {
		u = tuple.Underlying()
	}
	switch u := u.(type) {
	case *types2.Slice:
		walkQualType(u.Elem(), append(slices.Clone(path), "E"), object, emitOpt, emitConst, seen)
	case *types2.Array:
		walkQualType(u.Elem(), append(slices.Clone(path), "E"), object, emitOpt, emitConst, seen)
	case *types2.Map:
		walkQualType(u.Key(), append(slices.Clone(path), "K"), object, emitOpt, emitConst, seen)
		walkQualType(u.Elem(), append(slices.Clone(path), "E"), object, emitOpt, emitConst, seen)
	case *types2.Pointer:
		walkQualType(u.Elem(), append(slices.Clone(path), "D"), object, emitOpt, emitConst, seen)
	case *types2.Chan:
		walkQualType(u.Elem(), append(slices.Clone(path), "D"), object, emitOpt, emitConst, seen)
	case *types2.Signature:
		if params := u.Params(); params != nil {
			for i := range params.Len() {
				v := params.At(i)
				if v == nil {
					continue
				}
				walkQualType(v.Type(), append(slices.Clone(path), fmt.Sprintf("P%d", i)), object, emitOpt, emitConst, seen)
			}
		}
		if results := u.Results(); results != nil {
			for i := range results.Len() {
				v := results.At(i)
				if v == nil {
					continue
				}
				walkQualType(v.Type(), append(slices.Clone(path), fmt.Sprintf("R%d", i)), object, emitOpt, emitConst, seen)
			}
		}
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkQualType(f.Type(), append(slices.Clone(path), "F"+f.Name()), object, emitOpt, emitConst, seen)
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
					walkQualType(v.Type(), append(slices.Clone(base), fmt.Sprintf("P%d", j)), object, emitOpt, emitConst, seen)
				}
			}
			if results := sig.Results(); results != nil {
				for j := range results.Len() {
					v := results.At(j)
					if v == nil {
						continue
					}
					walkQualType(v.Type(), append(slices.Clone(base), fmt.Sprintf("R%d", j)), object, emitOpt, emitConst, seen)
				}
			}
		}
	case *types2.Named:
		if seen[u] {
			return
		}
		seen[u] = true
		walkQualType(u.Underlying(), path, object, emitOpt, emitConst, seen)
		delete(seen, u)
	}
}
