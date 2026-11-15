// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"internal/logosfile"
	"internal/pkgbits"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func logosExportHasLogosFile(pr *pkgbits.PkgDecoder) bool {
	n := pr.NumElems(pkgbits.SectionPosBase)
	for i := range n {
		r := pr.TempDecoder(pkgbits.SectionPosBase, pkgbits.RelElemIdx(i), pkgbits.SyncPosBase)
		name := r.String()
		pr.RetireDecoder(&r)
		if logosfile.Is(name) {
			return true
		}
	}
	return false
}

func checkLogosGoImport(pkg *types2.Package) {
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	cands := make(map[*types2.TypeName]bool)
	for _, name := range scope.Names() {
		tn, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok || tn.IsAlias() {
			continue
		}
		iface, ok := tn.Type().Underlying().(*types2.Interface)
		if !ok || !logos.HasUnionTerms(iface) {
			continue
		}
		cands[tn] = true
	}
	if len(cands) == 0 {
		return
	}
	v := &unionMentionWalker{cands: cands, seen: make(map[types2.Type]bool)}
	for _, name := range scope.Names() {
		switch obj := scope.Lookup(name).(type) {
		case *types2.Func:
			v.signature(obj.Type())
		case *types2.TypeName:
			if obj.IsAlias() {
				continue
			}
			if n, ok := obj.Type().(*types2.Named); ok {
				for i := range n.NumMethods() {
					v.signature(n.Method(i).Type())
				}
			}
			v.value(obj.Type().Underlying())
		case *types2.Var:
			v.value(obj.Type())
		}
		if v.hit != nil {
			base.Fatalf("logos build: %s", logos.DualUseMessage(v.hit.Name()))
		}
	}
}

type unionMentionWalker struct {
	cands map[*types2.TypeName]bool
	seen  map[types2.Type]bool
	hit   *types2.TypeName
}

func (v *unionMentionWalker) signature(sig types2.Type) {
	s, ok := sig.(*types2.Signature)
	if !ok || s == nil {
		return
	}
	v.tuple(s.Params())
	v.tuple(s.Results())
}

func (v *unionMentionWalker) tuple(t *types2.Tuple) {
	if t == nil {
		return
	}
	for i := range t.Len() {
		v.value(t.At(i).Type())
	}
}

func (v *unionMentionWalker) value(t types2.Type) {
	if t == nil || v.hit != nil || v.seen[t] {
		return
	}
	v.seen[t] = true
	switch t := t.(type) {
	case *types2.Basic:
		return
	case *types2.TypeParam:
		return
	case *types2.Union:
		return
	case *types2.Alias:
		v.value(types2.Unalias(t))
	case *types2.Named:
		if n, ok := types2.Unalias(t).(*types2.Named); ok && v.cands[n.Obj()] {
			v.hit = n.Obj()
			return
		}
		if args := t.TypeArgs(); args != nil {
			for i := range args.Len() {
				v.value(args.At(i))
			}
		}
		v.value(t.Underlying())
	case *types2.Interface:
		for i := range t.NumExplicitMethods() {
			v.signature(t.ExplicitMethod(i).Type())
		}
	case *types2.Signature:
		v.signature(t)
	case *types2.Tuple:
		v.tuple(t)
	case *types2.Struct:
		for i := range t.NumFields() {
			v.value(t.Field(i).Type())
		}
	case *types2.Pointer:
		v.value(t.Elem())
	case *types2.Slice:
		v.value(t.Elem())
	case *types2.Array:
		v.value(t.Elem())
	case *types2.Map:
		v.value(t.Key())
		v.value(t.Elem())
	case *types2.Chan:
		v.value(t.Elem())
	}
}

var logosUnions = make(map[*pkgWriter]map[*types2.TypeName]logos.UnionUse)

func (w *writer) logosUnderlying(obj *types2.TypeName, under types2.Type) {
	iface, ok := under.(*types2.Interface)
	if !ok || logosUnions[w.p][obj] != logos.UnionValueOnly {
		w.typ(under)
		return
	}
	methods := make([]*types2.Func, 0, iface.NumExplicitMethods())
	for i := range iface.NumExplicitMethods() {
		methods = append(methods, iface.ExplicitMethod(i))
	}
	var kept []types2.Type
	for i := range iface.NumEmbeddeds() {
		if e := iface.EmbeddedType(i); !logos.IsUnionTerm(e) {
			kept = append(kept, e)
		}
	}
	w.typ(types2.NewInterfaceType(methods, kept))
}

func (pw *pkgWriter) logosClassifyUnionUses(info *types2.Info, noders []*noder) {
	if !base.Logos {
		return
	}
	files := make([]*syntax.File, 0, len(noders))
	for _, p := range noders {
		if p.file != nil {
			files = append(files, p.file)
		}
	}
	logosUnions[pw] = logos.ClassifyDeclaredUnions(info, files)
}

func logosSkipUnionDecl(p *pkgWriter, name *types2.TypeName) bool {
	return !logos.KeepUnionDecl(logosUnions[p], name)
}
