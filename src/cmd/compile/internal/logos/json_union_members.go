// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"fmt"
	"strings"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

type unionMemberClass struct {
	spell   string
	code    string
	bare    string
	keyFunc string
	keys    []string
	hasJSON bool
	keyed   bool
	foreign bool
	typ     types2.Type
}

type unionNames struct {
	display string
	code    string
}

func classifyUnionMembers(holderPkg *types2.Package, staticJSON map[*types2.Named]bool, unionType types2.Type, codeQual func(*types2.Package) string) ([]unionMemberClass, unionNames, bool) {
	typ := unionType
	if cv, ok := types2.Unalias(typ).(*types2.ConstView); ok {
		typ = cv.Elem
	}
	names := unionNames{display: types2.TypeString(typ, qualifierFor(holderPkg)), code: types2.TypeString(typ, codeQual)}
	members, ok := types2.LogosUnionMembers(typ)
	if !ok {
		return nil, names, false
	}
	qual := qualifierFor(holderPkg)
	out := make([]unionMemberClass, 0, len(members))
	for _, m := range members {
		var c unionMemberClass
		var classified bool
		if m.Type != nil {
			c, classified = classifyOneMember(holderPkg, staticJSON, m.Type, qual)
			c.code = types2.TypeString(m.Type, codeQual)
		} else {
			c, classified = classifyFactMember(holderPkg, m.Text(), codeQual)
		}
		if !classified {
			return nil, names, false
		}
		out = append(out, c)
	}
	return out, names, true
}

func qualifierFor(holderPkg *types2.Package) func(*types2.Package) string {
	return func(p *types2.Package) string {
		if p == nil {
			return ""
		}
		if holderPkg != nil && p.Path() == holderPkg.Path() {
			return ""
		}
		return p.Name()
	}
}

func classifyFactMember(holderPkg *types2.Package, member string, codeQual func(*types2.Package) string) (unionMemberClass, bool) {
	trimmed, isPtr := strings.CutPrefix(member, "*")
	pkgPath, typeName, ok := strings.CutLast(trimmed, ".")
	if !ok || pkgPath == "" || typeName == "" {
		return unionMemberClass{}, false
	}
	if strings.Contains(typeName, "/") {
		return unionMemberClass{}, false
	}
	pkg := memberPackageFor(holderPkg, pkgPath)
	if pkg == nil {
		return unionMemberClass{}, false
	}
	spell := pkg.Name() + "." + typeName
	code := typeName
	if q := codeQual(pkg); q != "" {
		code = q + "." + typeName
	}
	if isPtr {
		spell = "*" + spell
		code = "*" + code
	}
	keys, eligible, hasJSON := foreignUnionKeysByPath(pkgPath, typeName)
	keyFunc := foreignKeyFunc(pkgPath, typeName)
	foreign := true
	if holderPkg != nil && holderPkg.Path() == pkgPath {
		foreign = false
		keyFunc = "_logos_jsonkey_" + typeName
	}
	if !eligible || len(keys) == 0 {
		return unionMemberClass{spell: spell, code: code, bare: typeName, keyFunc: keyFunc, keyed: false, hasJSON: hasJSON, foreign: foreign}, hasJSON
	}
	return unionMemberClass{spell: spell, code: code, bare: typeName, keyFunc: keyFunc, keys: keys, hasJSON: true, keyed: true, foreign: foreign}, true
}

func memberPackageFor(holderPkg *types2.Package, path string) *types2.Package {
	if holderPkg != nil {
		for _, imp := range holderPkg.Imports() {
			if imp.Path() == path {
				return imp
			}
		}
		if holderPkg.Path() == path {
			return nil
		}
	}
	_, name, ok := strings.CutLast(path, "/")
	if !ok {
		name = path
	}
	return types2.NewPackage(path, name)
}

func foreignUnionKeysByPath(pkgPath, typ string) ([]string, bool, bool) {
	rec, ok := facts.Lookup(pkgPath)
	if !ok || rec == nil {
		return nil, false, false
	}
	hasJSON := false
	eligible := false
	if meta := jsonMetaFor(rec, typ); meta != nil {
		hasJSON = meta.HasDecoder
		eligible = meta.Eligible
	}
	keys := jsonKeysFor(rec, typ)
	if !eligible || len(keys) == 0 {
		return nil, false, hasJSON
	}
	return keys, true, true
}

func classifyOneMember(holderPkg *types2.Package, staticJSON map[*types2.Named]bool, m types2.Type, qual func(*types2.Package) string) (unionMemberClass, bool) {
	spell := types2.TypeString(m, qual)
	elem := m
	if ptr, ok := types2.Unalias(elem).(*types2.Pointer); ok {
		elem = ptr.Elem()
	}
	named, ok := unaliasConstView(elem).(*types2.Named)
	if !ok || named.Obj() == nil || named.Obj().Pkg() == nil {
		return unionMemberClass{spell: spell, keyed: false, hasJSON: hasUnmarshalMethods(m), typ: m}, hasUnmarshalMethods(m)
	}
	if args := named.TypeArgs(); args != nil && args.Len() > 0 {
		for i := range args.Len() {
			if _, ok := types2.Unalias(args.At(i)).(*types2.TypeParam); ok {
				return unionMemberClass{}, false
			}
		}
	}
	bare := named.Obj().Name()
	memberPkg := named.Obj().Pkg()
	samePkg := holderPkg != nil && memberPkg != nil && holderPkg.Path() == memberPkg.Path()
	concreteArgs := named.TypeArgs() != nil && named.TypeArgs().Len() > 0
	hasJSON := hasUnmarshalMethods(m)
	var keys []string
	var eligible bool
	if samePkg || concreteArgs {
		keys, eligible = localUnionKeys(staticJSON, named)
	} else {
		keys, eligible, hasJSON = foreignUnionKeys(memberPkg, bare, hasJSON)
	}
	keyFunc := keyFuncFor(bare, memberPkg, holderPkg)
	foreign := !samePkg
	if !eligible || len(keys) == 0 {
		return unionMemberClass{spell: spell, bare: bare, keyFunc: keyFunc, keyed: false, hasJSON: hasJSON, foreign: foreign, typ: m}, true
	}
	return unionMemberClass{spell: spell, bare: bare, keyFunc: keyFunc, keys: keys, hasJSON: true, keyed: true, foreign: foreign, typ: m}, true
}

func localUnionKeys(staticJSON map[*types2.Named]bool, named *types2.Named) ([]string, bool) {
	st, ok := types2.Unalias(named).Underlying().(*types2.Struct)
	if !ok {
		return nil, false
	}
	if staticJSON == nil {
		return nil, false
	}
	origin := cmp.Or(named.Origin(), named)
	if !staticJSON[origin] {
		return nil, false
	}
	return structKeys(st), true
}

func structKeys(st *types2.Struct) []string {
	var keys []string
	for i := range st.NumFields() {
		if field, ok := types2.LogosJSONFieldOf(st, i); ok {
			keys = append(keys, field.Name)
		}
	}
	return keys
}

func foreignUnionKeys(memberPkg *types2.Package, bare string, baseJSON bool) ([]string, bool, bool) {
	rec, ok := facts.Lookup(memberPkg.Path())
	if !ok || rec == nil {
		return nil, false, baseJSON
	}
	hasJSON := baseJSON
	eligible := false
	if meta := jsonMetaFor(rec, bare); meta != nil {
		hasJSON = meta.HasDecoder
		eligible = meta.Eligible
	}
	keys := jsonKeysFor(rec, bare)
	if !eligible || len(keys) == 0 {
		return nil, false, hasJSON
	}
	return keys, true, true
}

func keyFuncFor(bare string, memberPkg, holderPkg *types2.Package) string {
	if holderPkg != nil && memberPkg != nil && holderPkg.Path() == memberPkg.Path() {
		return "_logos_jsonkey_" + bare
	}
	return foreignKeyFunc(memberPkg.Path(), bare)
}

func foreignKeyFunc(pkgPath, bare string) string {
	return fmt.Sprintf("_logos_jsonkey_%016x_%s", constHashString(pkgPath), bare)
}

func jsonMetaFor(rec *facts.Facts, typ string) *facts.JsonMetaFact {
	for i := range rec.JsonMetas {
		if rec.JsonMetas[i].Type == typ {
			return &rec.JsonMetas[i]
		}
	}
	return nil
}

func jsonKeysFor(rec *facts.Facts, typ string) []string {
	var indexed []facts.JsonKeyFact
	for _, k := range rec.JsonKeys {
		if k.Type == typ {
			indexed = append(indexed, k)
		}
	}
	for i := 1; i < len(indexed); i++ {
		for j := i; j > 0 && indexed[j].Index < indexed[j-1].Index; j-- {
			indexed[j], indexed[j-1] = indexed[j-1], indexed[j]
		}
	}
	out := make([]string, 0, len(indexed))
	for _, k := range indexed {
		out = append(out, k.Key)
	}
	return out
}
