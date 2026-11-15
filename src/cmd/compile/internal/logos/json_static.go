// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const jsonAlias = "_logos_json"

const jsonImportPath = "logos/rt/json"

const jsonRecv = "_logos_v"

type scalarKind int

const (
	scalarBool scalarKind = iota
	scalarInt
	scalarUint
	scalarFloat32
	scalarFloat64
	scalarString
)

type shapeKind int

const (
	shapeFallback shapeKind = iota
	shapeScalar
	shapePointer
	shapeSlice
	shapeUnion
)

type jsonShape struct {
	kind        shapeKind
	scalar      scalarKind
	nonNil      bool
	needsIsZero bool
}

type jsonField struct {
	goName    string
	key       string
	omitEmpty bool
	omitZero  bool
	quoted    bool
	shape     jsonShape
	pos       int
	required  int
	typ       types2.Type
}

type jsonStruct struct {
	name       string
	fields     []*jsonField
	defaults   []string
	required   []string
	staticJSON map[*types2.Named]bool
	holderPkg  *types2.Package
	pos        syntax.Pos
	qualify    func(*types2.Package) string
}

func (fl *fileLower) genQualifier(p *types2.Package) string {
	if p == fl.pkg.tpkg {
		return ""
	}
	return fl.synthQualifier(p)
}

func (fl *fileLower) lowerStaticJSON() {
	for _, decl := range fl.file.DeclList {
		if td, ok := decl.(*syntax.TypeDecl); ok {
			fl.lowerStaticJSONType(td)
		}
	}
}

func (fl *fileLower) lowerStaticJSONType(td *syntax.TypeDecl) {
	named := fl.declaredNamed(td)
	if named == nil {
		return
	}
	fields, jsonOK := types2.LogosStaticStruct(fl.pkg.info, named, td.Type)
	binaryFields, binaryOK := types2.LogosBinaryStruct(fl.pkg.info, named, td.Type)
	if !jsonOK && !binaryOK {
		return
	}
	if !jsonOK {
		fields = binaryFields
	}
	js := fl.buildJSONStruct(td.Name.Value, named.Underlying().(*types2.Struct), fields)
	js.pos = td.Pos()
	var b strings.Builder
	if jsonOK && !userDeclaresMethod(named, "MarshalJSON") {
		plan := fl.planJSONWalk(js, named)
		vname := ""
		if plan.stateful {
			if key := fl.ensureJSONHelper(plan, named); key != "" {
				vname = fl.pkg.jsonHelpers[key].vname
			}
		}
		if vname != "" {
			fl.emitJSONHelpers(td.Pos(), plan)
			js.marshalTextWalk(&b, plan, vname)
		} else {
			js.marshalText(&b)
		}
	}
	if jsonOK {
		if !userDeclaresMethod(named, "UnmarshalJSON") {
			js.unmarshalText(&b)
		}
		js.keyFuncs(&b)
		js.missingFunc(&b)
		if fl.pkg.jsonUnionKeys == nil {
			fl.pkg.jsonUnionKeys = make(map[string]bool)
		}
		js.unionMemberKeys(&b, fl.pkg.jsonUnionKeys)
	}
	if binaryOK {
		fl.lowerStaticBinary(td, named, js, &b)
	}
	if b.Len() == 0 {
		return
	}
	text := b.String()
	fl.addGeneratedDecls(td.Pos(), text)
	fl.needJSON = fl.needJSON || strings.Contains(text, jsonAlias+".")
	fl.needRT = fl.needRT || strings.Contains(text, rtAlias+".")
	fl.needBinary = fl.needBinary || strings.Contains(text, "_logos_b")
}

func (fl *fileLower) declaredNamed(td *syntax.TypeDecl) *types2.Named {
	if td.Alias || len(td.TParamList) > 0 {
		return nil
	}
	tn, ok := fl.pkg.info.Defs[td.Name].(*types2.TypeName)
	if !ok {
		return nil
	}
	named, _ := tn.Type().(*types2.Named)
	return named
}

func userDeclaresMethod(named *types2.Named, name string) bool {
	for i := range named.NumMethods() {
		m := named.Method(i)
		if m.Name() == name && m.Pos().IsKnown() {
			return true
		}
	}
	return false
}

func (fl *fileLower) addGeneratedDecls(pos syntax.Pos, text string) {
	decls := parseGeneratedDecls(text, pos)
	for _, d := range decls {
		if vd, ok := d.(*syntax.VarDecl); ok && strings.Contains(text, rodataMarker+"\nvar "+vd.NameList[0].Value) {
			fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd})
		}
	}
	fl.constDecls = append(fl.constDecls, decls...)
	fl.constText += "\n" + text
}

func parseGeneratedDecls(text string, pos syntax.Pos) []syntax.Decl {
	decls := parseConstMapDecls(text, pos)
	for _, d := range decls {
		syntax.Inspect(d, func(n syntax.Node) bool {
			if cc, ok := n.(*syntax.CaseClause); ok {
				cc.Colon = pos
			}
			return true
		})
	}
	return decls
}

func (fl *fileLower) buildJSONStruct(name string, st *types2.Struct, fields []*types2.Var) *jsonStruct {
	js := &jsonStruct{name: name, staticJSON: fl.pkg.info.LogosStaticJSON, holderPkg: fl.pkg.tpkg, qualify: fl.genQualifier}

	for _, fld := range fields {
		i := fieldIndexOf(st, fld)
		if i < 0 {
			continue
		}
		info, ok := types2.LogosJSONFieldOf(st, i)
		if !ok {
			continue
		}
		f := &jsonField{
			goName:    fld.Name(),
			key:       info.Name,
			omitEmpty: info.OmitEmpty,
			omitZero:  info.OmitZero,
			pos:       len(js.fields),
			required:  -1,
			typ:       fld.Type(),
		}
		f.shape = fl.classifyJSONField(fld.Type())
		f.quoted = info.StringOpt && isQuotableForString(f.typ, f.shape)
		if f.shape.nonNil {
			f.required = len(js.required)
			js.required = append(js.required, f.goName)
		}
		js.fields = append(js.fields, f)
	}
	for _, d := range fl.pkg.info.LogosFieldDefaults[st] {
		js.defaults = append(js.defaults, fmt.Sprintf("%s.%s = %s", jsonRecv, d.Field, renderExpr(d.Default)))
	}
	return js
}

func fieldIndexOf(st *types2.Struct, fld *types2.Var) int {
	for i := range st.NumFields() {
		if st.Field(i) == fld {
			return i
		}
	}
	return -1
}

func (fl *fileLower) classifyJSONField(t types2.Type) jsonShape {
	if cv, ok := types2.Unalias(t).(*types2.ConstView); ok {
		t = cv.Elem
	}
	_, bare := types2.Unalias(t).(*types2.Pointer)
	if o := fl.asOption(t); o != nil {
		t = o.Elem
		if _, ok := types2.Unalias(t).(*types2.Pointer); !ok {
			return jsonShape{kind: shapeFallback}
		}
	}
	shape := fl.classifyShape(t)
	shape.nonNil = bare
	return shape
}

func (fl *fileLower) classifyShape(t types2.Type) jsonShape {
	if _, ok := types2.LogosUnionMembers(t); ok {
		return jsonShape{kind: shapeUnion}
	}
	switch c := types2.Unalias(t).(type) {
	case *types2.Pointer:
		return fl.scalarShape(c.Elem(), shapePointer)
	case *types2.Slice:
		return fl.scalarShape(c.Elem(), shapeSlice)
	}
	return fl.scalarShape(t, shapeScalar)
}

func (fl *fileLower) scalarShape(elem types2.Type, k shapeKind) jsonShape {
	kind, ok := scalarOf(elem)
	if !ok || (k == shapeSlice && kind == scalarUint && isByteSized(elem)) {
		return jsonShape{kind: shapeFallback}
	}
	shape := jsonShape{kind: k, scalar: kind}
	if k == shapeScalar || k == shapePointer {
		shape.needsIsZero = needsIsZero(elem)
	}
	return shape
}

func isByteSized(t types2.Type) bool {
	b, ok := t.Underlying().(*types2.Basic)
	return ok && b.Kind() == types2.Uint8
}

var jsonMethodNames = append(slices.Clone(marshalMethodNames), unmarshalMethodNames...)

func scalarOf(t types2.Type) (scalarKind, bool) {
	switch c := types2.Unalias(t).(type) {
	case *types2.Basic:
		return basicScalar(c)
	case *types2.Named:
		b, ok := c.Underlying().(*types2.Basic)
		if !ok || hasJSONMethods(c) {
			return 0, false
		}
		return basicScalar(b)
	}
	return 0, false
}

func hasJSONMethods(named *types2.Named) bool {
	obj := named.Obj()
	if obj.Pkg() != nil && obj.Pkg().Path() == "encoding/json" {
		return true
	}
	return hasMethodNamed(named, jsonMethodNames)
}

var marshalMethodNames = []string{
	"MarshalJSON", "MarshalJSONTo", "MarshalText", "AppendText",
}

var unmarshalMethodNames = []string{
	"UnmarshalJSON", "UnmarshalJSONFrom", "UnmarshalText",
}

func unaliasConstView(t types2.Type) types2.Type {
	u := types2.Unalias(t)
	if cv, ok := u.(*types2.ConstView); ok {
		return types2.Unalias(cv.Elem)
	}
	return u
}

func hasMethodNamed(t types2.Type, names []string) bool {
	u := unaliasConstView(t)
	if p, ok := u.(*types2.Pointer); ok {
		u = unaliasConstView(p.Elem())
	}
	named, ok := u.(*types2.Named)
	if !ok || named.Obj() == nil {
		return false
	}
	ptr := types2.NewPointer(named)
	for _, name := range names {
		if m, _, _ := types2.LookupFieldOrMethod(ptr, false, named.Obj().Pkg(), name); m != nil {
			return true
		}
	}
	return false
}

func hasMarshalMethods(t types2.Type) bool {
	if isExactNumber(t) {
		return true
	}
	return hasMethodNamed(t, marshalMethodNames)
}

func hasUnmarshalMethods(t types2.Type) bool {
	if isExactNumber(t) {
		return true
	}
	return hasMethodNamed(t, unmarshalMethodNames)
}

func isExactNumber(t types2.Type) bool {
	u := unaliasConstView(t)
	if p, ok := u.(*types2.Pointer); ok {
		u = unaliasConstView(p.Elem())
	}
	named, ok := u.(*types2.Named)
	if !ok || named.Obj() == nil || named.Obj().Name() != "Number" {
		return false
	}
	b, ok := named.Underlying().(*types2.Basic)
	if !ok || b.Kind() != types2.String {
		return false
	}
	pkg := named.Obj().Pkg()
	return pkg != nil && pkg.Path() == "encoding/json"
}

func fallbackScalarKind(t types2.Type) (scalarKind, bool) {
	u := unaliasConstView(t)
	if p, ok := u.(*types2.Pointer); ok {
		u = unaliasConstView(p.Elem())
	}
	return scalarOfNamedOrBasic(u)
}

func scalarOfNamedOrBasic(t types2.Type) (scalarKind, bool) {
	if named, ok := t.(*types2.Named); ok && named.Obj() != nil && named.Obj().Pkg() != nil && named.Obj().Pkg().Path() == "encoding/json" && named.Obj().Name() == "Number" {
		return scalarString, true
	}
	switch c := t.(type) {
	case *types2.Basic:
		return basicScalar(c)
	case *types2.Named:
		b, ok := c.Underlying().(*types2.Basic)
		if !ok {
			return 0, false
		}
		return basicScalar(b)
	}
	return 0, false
}

func isQuotableForString(t types2.Type, shape jsonShape) bool {
	if shape.kind == shapeSlice || shape.kind == shapeUnion {
		return false
	}
	if shape.kind != shapeFallback {
		return true
	}
	_, ok := fallbackScalarKind(t)
	return ok
}

func isIsZeroSignature(obj types2.Object) bool {
	fn, ok := obj.(*types2.Func)
	if !ok {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		return false
	}
	if sig.Params().Len() != 0 || sig.Results().Len() != 1 {
		return false
	}
	b, ok := types2.Unalias(sig.Results().At(0).Type()).(*types2.Basic)
	return ok && b.Kind() == types2.Bool
}

func needsIsZero(t types2.Type) bool {
	u := unaliasConstView(t)
	if o, ok := u.(*types2.Option); ok {
		u = unaliasConstView(o.Elem)
	} else if n, ok := u.(*types2.Named); ok {
		if o, ok := n.Underlying().(*types2.Option); ok {
			u = types2.Unalias(o.Elem)
		}
	}
	if p, ok := u.(*types2.Pointer); ok {
		u = unaliasConstView(p.Elem())
	}
	named, ok := u.(*types2.Named)
	if !ok || named.Obj() == nil {
		return false
	}
	ptr := types2.NewPointer(named)
	obj, _, _ := types2.LookupFieldOrMethod(ptr, false, named.Obj().Pkg(), "IsZero")
	return obj != nil && isIsZeroSignature(obj)
}

func basicScalar(b *types2.Basic) (scalarKind, bool) {
	switch b.Kind() {
	case types2.Bool:
		return scalarBool, true
	case types2.Int, types2.Int8, types2.Int16, types2.Int32, types2.Int64:
		return scalarInt, true
	case types2.Uint, types2.Uint8, types2.Uint16, types2.Uint32, types2.Uint64, types2.Uintptr:
		return scalarUint, true
	case types2.Float32:
		return scalarFloat32, true
	case types2.Float64:
		return scalarFloat64, true
	case types2.String:
		return scalarString, true
	}
	return 0, false
}

func jsonImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, jsonAlias),
		Path:         litOf(pos, fmt.Sprintf("%q", jsonImportPath), syntax.StringLit),
	}, pos)
}

const reflectAlias = "_logos_reflect"

func reflectImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, reflectAlias),
		Path:         litOf(pos, `"reflect"`, syntax.StringLit),
	}, pos)
}

func (fl *fileLower) helperEligibleType(t types2.Type) *types2.Named {
	u := types2.Unalias(t)
	if o, isOpt := u.(*types2.Option); isOpt {
		u = o.Elem
	}
	if p, ok := u.(*types2.Pointer); ok {
		u = types2.Unalias(p.Elem())
	}
	named, ok := u.(*types2.Named)
	if !ok || named.TypeArgs().Len() != 0 {
		return nil
	}
	if _, ok := named.Underlying().(*types2.Struct); !ok {
		return nil
	}
	if named.Obj().Pkg() != fl.pkg.tpkg {
		return nil
	}
	for i := range named.NumMethods() {
		m := named.Method(i)
		if m.Name() != "MarshalJSON" {
			continue
		}
		recv := m.Signature().Recv().Type()
		if p, ok := recv.(*types2.Pointer); ok {
			recv = p.Elem()
		}
		if recv != named {
			continue
		}
		if m.Pos().IsKnown() {
			return nil
		}
		return named
	}
	return nil
}

func (fl *fileLower) jsonTypeName(named *types2.Named) (string, bool) {
	if named.Obj().Pkg() != fl.pkg.tpkg {
		return "", false
	}
	return named.Obj().Name(), true
}
