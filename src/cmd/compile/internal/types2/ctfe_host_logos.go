// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"go/constant"
	"math/big"
	"strconv"
)

type ctfeHost struct {
	check *Checker
}

func (h *ctfeHost) Resolve(id *syntax.Name) ctfe.Entity {
	check := h.check
	if u := check.logosUnitForNode(id); u != nil {
		if ent, ok := h.resolveUnitName(u, id); ok {
			return ent
		}
		return ctfe.Entity{Kind: ctfe.EntNone, Name: id.Value}
	}
	if obj := check.Info.Uses[id]; obj != nil {
		if ent, ok := h.entityOf(obj, id.Value); ok {
			return ent
		}
	}
	if obj := check.pkg.scope.Lookup(id.Value); obj != nil {
		if ent, ok := h.entityOf(obj, id.Value); ok {
			return ent
		}
	}
	if obj := Universe.Lookup(id.Value); obj != nil {
		if ent, ok := h.entityOf(obj, id.Value); ok {
			return ent
		}
	}
	return ctfe.Entity{Kind: ctfe.EntNone, Name: id.Value}
}

func (h *ctfeHost) entityOf(obj Object, name string) (ctfe.Entity, bool) {
	check := h.check
	switch obj := obj.(type) {
	case *Const:
		if obj == universeIota {
			if check.iota == nil {
				return ctfe.Entity{}, false
			}
			v, ok := h.constToValue(check.iota, nil)
			if !ok {
				return ctfe.Entity{}, false
			}
			return ctfe.Entity{Kind: ctfe.EntConst, Const: v}, true
		}
		if obj.parent == check.pkg.scope {
			if obj.typ == nil || obj.val == nil {
				check.objDecl(obj)
			}
			check.addDeclDep(obj)
		}
		if obj.typ == Typ[Invalid] || obj.val == nil {
			return ctfe.Entity{}, false
		}
		if obj.val.Kind() != constant.Unknown {
			v, ok := h.constToValue(obj.val, h.fromType(obj.typ))
			if !ok {
				return ctfe.Entity{}, false
			}
			return ctfe.Entity{Kind: ctfe.EntConst, Const: v}, true
		}
		if v, ok := check.logosRecorded(obj); ok {
			return ctfe.Entity{Kind: ctfe.EntConst, Const: v}, true
		}
		return ctfe.Entity{}, false
	case *Var:
		if obj.parent == check.pkg.scope {
			return ctfe.Entity{Kind: ctfe.EntPackageVar, Name: obj.name}, true
		}
		if obj.pkg != nil && obj.pkg != check.pkg {
			return ctfe.Entity{Kind: ctfe.EntPackageVar, Name: obj.pkg.path + "." + obj.name}, true
		}
		return ctfe.Entity{Kind: ctfe.EntNone, Name: obj.name}, true
	case *Func:
		if obj.pkg == check.pkg {
			if body, ok := check.logosBodies[obj.name]; ok && body != nil {
				fn := h.buildFunc(obj, body)
				if fn == nil {
					return ctfe.Entity{}, false
				}
				_, isConst := check.logosConstSyntax[body]
				h.logosRecordConstFunc(obj.name, body, isConst)
				if rec, ok := h.logosInfoConstFunc(obj.name); ok && rec.Const {
					isConst = true
				}
				if isConst {
					fn.Const = true
					return ctfe.Entity{Kind: ctfe.EntConstFunc, Fn: fn}, true
				}
				return ctfe.Entity{Kind: ctfe.EntPlainFunc, Fn: fn}, true
			}
			if rec, ok := h.logosInfoConstFunc(obj.name); ok {
				fn := h.buildFunc(obj, rec.Decl)
				if fn == nil {
					return ctfe.Entity{}, false
				}
				if rec.Const {
					fn.Const = true
					return ctfe.Entity{Kind: ctfe.EntConstFunc, Fn: fn}, true
				}
				return ctfe.Entity{Kind: ctfe.EntPlainFunc, Fn: fn}, true
			}
			return ctfe.Entity{}, false
		}
		path := ""
		if obj.pkg != nil {
			path = obj.pkg.path
		}
		return ctfe.Entity{Kind: ctfe.EntImport, Name: path + "." + obj.name}, true
	case *Builtin:
		return ctfe.Entity{Kind: ctfe.EntBuiltin, Builtin: obj.name}, true
	case *TypeName:
		return ctfe.Entity{Kind: ctfe.EntType}, true
	case *PkgName:
		return ctfe.Entity{Kind: ctfe.EntPackage, Name: name}, true
	}
	return ctfe.Entity{}, false
}

func (h *ctfeHost) ResolveSel(sel *syntax.SelectorExpr) ctfe.Entity {
	if u := h.check.logosUnitForNode(sel); u != nil {
		return h.resolveUnitSel(u, sel)
	}
	x, ok := syntax.Unparen(sel.X).(*syntax.Name)
	if !ok {
		return ctfe.Entity{}
	}
	pkgName := logosPkgName(h.check, x)
	if pkgName == nil {
		return ctfe.Entity{}
	}
	imp := pkgName.Imported()
	if imp == nil {
		return ctfe.Entity{}
	}
	obj := imp.scope.Lookup(sel.Sel.Value)
	if obj == nil {
		return ctfe.Entity{}
	}
	switch obj := obj.(type) {
	case *Const:
		if obj.val == nil || obj.val.Kind() == constant.Unknown {
			return ctfe.Entity{}
		}
		v, ok := h.constToValue(obj.val, h.fromType(obj.typ))
		if !ok {
			return ctfe.Entity{}
		}
		return ctfe.Entity{Kind: ctfe.EntConst, Const: v}
	case *Var:
		return ctfe.Entity{Kind: ctfe.EntPackageVar, Name: pkgName.name + "." + obj.name}
	case *Func:
		if u, fn, body, err := h.check.logosResolveImportedConstFunc(imp.path, obj.name); err == nil && u != nil {
			if callee := h.buildFunc(fn, body); callee != nil {
				callee.Const = true
				return ctfe.Entity{Kind: ctfe.EntConstFunc, Fn: callee}
			}
		}
		return ctfe.Entity{Kind: ctfe.EntImport, Name: imp.path + "." + obj.name}
	case *TypeName:
		return ctfe.Entity{Kind: ctfe.EntType}
	}
	return ctfe.Entity{}
}

func logosPkgName(check *Checker, x *syntax.Name) *PkgName {
	if obj, ok := check.Info.Uses[x].(*PkgName); ok {
		return obj
	}
	if obj, ok := check.pkg.scope.Lookup(x.Value).(*PkgName); ok {
		return obj
	}
	return nil
}

func (h *ctfeHost) constToValue(c constant.Value, t *ctfe.Type) (ctfe.Value, bool) {
	switch c.Kind() {
	case constant.Bool:
		typ := ctfe.UntypedBool()
		if t != nil && (t.Kind == ctfe.KBool) {
			typ = t
		}
		return ctfe.BoolValue(constant.BoolVal(c), typ), true
	case constant.String:
		typ := ctfe.UntypedString()
		if t != nil && t.Kind == ctfe.KString {
			typ = t
		}
		return ctfe.StringValue(constant.StringVal(c), typ), true
	case constant.Int:
		s := c.ExactString()
		n, ok := parseConstInt(s)
		if !ok {
			return ctfe.Value{}, false
		}
		typ := ctfe.UntypedInt()
		if t != nil && isCTFEIntKind(t.Kind) {
			typ = t
		}
		return ctfe.IntValue(n, typ), true
	case constant.Float:
		if t != nil && t.Kind == ctfe.KFloat32 {
			f32, _ := constant.Float32Val(c)
			return ctfe.FloatValue(float64(f32), t), true
		}
		if t != nil && t.Kind == ctfe.KFloat64 {
			f, _ := constant.Float64Val(c)
			return ctfe.FloatValue(f, t), true
		}
		typ := ctfe.UntypedFloat()
		if t != nil && t.Kind == ctfe.KUntypedFloat {
			typ = t
		}
		switch v := constant.Val(c).(type) {
		case *big.Rat:
			return ctfe.ExactFloatValue(v, typ), true
		case *big.Float:
			r, _ := v.Rat(nil)
			if r != nil {
				return ctfe.ExactFloatValue(r, typ), true
			}
		}
		if r, ok := new(big.Rat).SetString(c.ExactString()); ok {
			return ctfe.ExactFloatValue(r, typ), true
		}
		f, _ := constant.Float64Val(c)
		return ctfe.FloatValue(f, typ), true
	case constant.Complex:
		typ := ctfe.UntypedComplex()
		if t != nil && (t.Kind == ctfe.KComplex64 || t.Kind == ctfe.KComplex128) {
			typ = t
		}
		re, _ := constant.Float64Val(constant.Real(c))
		im, _ := constant.Float64Val(constant.Imag(c))
		return ctfe.ComplexValue(complex(re, im), typ), true
	}
	return ctfe.Value{}, false
}

func (h *ctfeHost) TypeOf(e syntax.Expr) *ctfe.Type {
	check := h.check
	if u := check.logosUnitForNode(e); u != nil {
		return h.resolveUnitTypeExpr(u, e)
	}
	if tv, ok := check.Info.Types[e]; ok && tv.Type != nil {
		if t := h.fromType(tv.Type); t != nil {
			return t
		}
	}
	return h.resolveTypeExpr(e)
}

func (h *ctfeHost) resolveTypeExpr(e syntax.Expr) *ctfe.Type {
	check := h.check
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if t := ctfe.BasicType(e.Value); t != nil {
			return t
		}
		if e.Value == "byte" {
			return ctfe.BasicType("uint8")
		}
		if e.Value == "rune" {
			return ctfe.BasicType("int32")
		}
		if e.Value == "any" || e.Value == "error" {
			return nil
		}
		if obj, ok := check.pkg.scope.Lookup(e.Value).(*TypeName); ok {
			return h.fromType(obj.typ)
		}
		if obj := Universe.Lookup(e.Value); obj != nil {
			if tn, ok := obj.(*TypeName); ok {
				return h.fromType(tn.typ)
			}
		}
	case *syntax.SelectorExpr:
		return h.resolveQualified(e)
	case *syntax.SliceType:
		if el := h.resolveTypeExpr(e.Elem); el != nil {
			return &ctfe.Type{Kind: ctfe.KSlice, Elem: el}
		}
	case *syntax.ArrayType:
		el := h.resolveTypeExpr(e.Elem)
		if el == nil {
			return nil
		}
		if e.Len == nil {
			return &ctfe.Type{Kind: ctfe.KArray, Len: -1, Elem: el}
		}
		if n, ok := h.arrayLength(e.Len); ok {
			return &ctfe.Type{Kind: ctfe.KArray, Len: n, Elem: el}
		}
	case *syntax.MapType:
		k := h.resolveTypeExpr(e.Key)
		v := h.resolveTypeExpr(e.Value)
		if k != nil && v != nil {
			return &ctfe.Type{Kind: ctfe.KMap, Key: k, Elem: v}
		}
	case *syntax.StructType:
		fields := make([]ctfe.Field, 0, len(e.FieldList))
		for _, f := range e.FieldList {
			ft := h.resolveTypeExpr(f.Type)
			if ft == nil {
				return nil
			}
			name := ""
			if f.Name != nil {
				name = f.Name.Value
			}
			fields = append(fields, ctfe.Field{Name: name, Type: ft})
		}
		return &ctfe.Type{Kind: ctfe.KStruct, Fields: fields}
	case *syntax.DotsType:
		return h.resolveTypeExpr(e.Elem)
	case *syntax.ConstType:
		return h.resolveTypeExpr(e.Elem)
	case *syntax.CallExpr:
		if _, ok := syntax.Unparen(e.Fun).(*syntax.Name); ok {
			return h.resolveTypeExpr(e.Fun)
		}
		if sel, ok := syntax.Unparen(e.Fun).(*syntax.SelectorExpr); ok {
			return h.resolveQualified(sel)
		}
	case *syntax.CompositeLit:
		return h.resolveTypeExpr(e.Type)
	}
	return nil
}

func (h *ctfeHost) resolveQualified(sel *syntax.SelectorExpr) *ctfe.Type {
	check := h.check
	x, ok := syntax.Unparen(sel.X).(*syntax.Name)
	if !ok {
		return nil
	}
	for _, imp := range check.imports {
		if imp.name != x.Value {
			continue
		}
		pkg := imp.Imported()
		if pkg == nil {
			continue
		}
		if obj, ok := pkg.scope.Lookup(sel.Sel.Value).(*TypeName); ok {
			t := h.fromType(obj.typ)
			if t == nil {
				return nil
			}
			t.Opaque = true
			return t
		}
		return nil
	}
	return nil
}

func (h *ctfeHost) arrayLength(e syntax.Expr) (int64, bool) {
	if lit, ok := syntax.Unparen(e).(*syntax.BasicLit); ok && lit.Kind == syntax.IntLit {
		var n int64
		if _, err := parseDecimal(lit.Value, &n); err == nil {
			return n, true
		}
		return 0, false
	}
	if name, ok := syntax.Unparen(e).(*syntax.Name); ok {
		ent := h.Resolve(name)
		if ent.Kind == ctfe.EntConst && ent.Const.Type() != nil && isCTFEIntKind(ent.Const.Type().Kind) {
			return ent.Const.Int64(), true
		}
	}
	return 0, false
}

func (h *ctfeHost) fromType(t Type) *ctfe.Type {
	switch t := Unalias(t).(type) {
	case *Basic:
		switch t.Kind() {
		case Bool:
			return &ctfe.Type{Kind: ctfe.KBool}
		case String:
			return &ctfe.Type{Kind: ctfe.KString}
		case Int:
			return &ctfe.Type{Kind: ctfe.KInt}
		case Int8:
			return &ctfe.Type{Kind: ctfe.KInt8}
		case Int16:
			return &ctfe.Type{Kind: ctfe.KInt16}
		case Int32:
			return &ctfe.Type{Kind: ctfe.KInt32}
		case Int64:
			return &ctfe.Type{Kind: ctfe.KInt64}
		case Uint:
			return &ctfe.Type{Kind: ctfe.KUint}
		case Uint8:
			return &ctfe.Type{Kind: ctfe.KUint8}
		case Uint16:
			return &ctfe.Type{Kind: ctfe.KUint16}
		case Uint32:
			return &ctfe.Type{Kind: ctfe.KUint32}
		case Uint64:
			return &ctfe.Type{Kind: ctfe.KUint64}
		case Uintptr:
			return &ctfe.Type{Kind: ctfe.KUintptr}
		case Float32:
			return &ctfe.Type{Kind: ctfe.KFloat32}
		case Float64:
			return &ctfe.Type{Kind: ctfe.KFloat64}
		case Complex64:
			return &ctfe.Type{Kind: ctfe.KComplex64}
		case Complex128:
			return &ctfe.Type{Kind: ctfe.KComplex128}
		case UnsafePointer:
			return &ctfe.Type{Impure: "pointer type in const func"}
		case UntypedBool:
			return ctfe.UntypedBool()
		case UntypedInt:
			return ctfe.UntypedInt()
		case UntypedRune:
			return ctfe.UntypedRune()
		case UntypedFloat:
			return ctfe.UntypedFloat()
		case UntypedComplex:
			return ctfe.UntypedComplex()
		case UntypedString:
			return ctfe.UntypedString()
		case UntypedNil:
			return &ctfe.Type{Kind: ctfe.KUntypedNil}
		}
	case *Array:
		if el := h.fromType(t.Elem()); el != nil {
			return &ctfe.Type{Kind: ctfe.KArray, Len: t.Len(), Elem: el}
		}
	case *Slice:
		if el := h.fromType(t.Elem()); el != nil {
			return &ctfe.Type{Kind: ctfe.KSlice, Elem: el}
		}
	case *Map:
		k := h.fromType(t.Key())
		v := h.fromType(t.Elem())
		if k != nil && v != nil {
			return &ctfe.Type{Kind: ctfe.KMap, Key: k, Elem: v}
		}
	case *Struct:
		fields := make([]ctfe.Field, 0, t.NumFields())
		for i := range t.NumFields() {
			f := t.Field(i)
			ft := h.fromType(f.Type())
			if ft == nil {
				return nil
			}
			fields = append(fields, ctfe.Field{Name: f.Name(), Type: ft, Embedded: f.Embedded()})
		}
		return &ctfe.Type{Kind: ctfe.KStruct, Fields: fields}
	case *Named:
		obj := t.Obj()
		under := h.fromType(t.Underlying())
		if under == nil {
			return nil
		}
		if under.Impure != "" {
			return under
		}
		under.Named = obj.Name()
		under.Opaque = obj.Pkg() != h.check.pkg
		if obj.Pkg() != nil {
			under.PkgPath = obj.Pkg().Path()
		}
		return under
	case *Pointer:
		return &ctfe.Type{Impure: "pointer type in const func"}
	case *Chan:
		return &ctfe.Type{Impure: "channel type in const func"}
	case *ConstView:
		return h.fromType(t.Elem)
	case *ReadonlyDescendants:
		return &ctfe.Type{Impure: "readonly-descendant capability in const func"}
	}
	return nil
}

type LogosConstFunc struct {
	Decl  *syntax.FuncDecl
	Const bool
}

func (h *ctfeHost) logosRecordConstFunc(name string, body *syntax.FuncDecl, isConst bool) {
	if body == nil || h.check.Info == nil {
		return
	}
	if h.check.Info.LogosConstFuncs == nil {
		h.check.Info.LogosConstFuncs = map[string]LogosConstFunc{}
	}
	if _, ok := h.check.Info.LogosConstFuncs[name]; ok {
		return
	}
	h.check.Info.LogosConstFuncs[name] = LogosConstFunc{Decl: body, Const: isConst}
}

func (h *ctfeHost) logosInfoConstFunc(name string) (LogosConstFunc, bool) {
	if h.check.Info == nil {
		return LogosConstFunc{}, false
	}
	rec, ok := h.check.Info.LogosConstFuncs[name]
	if !ok || rec.Decl == nil {
		return LogosConstFunc{}, false
	}
	return rec, true
}

func (h *ctfeHost) buildFunc(obj *Func, body *syntax.FuncDecl) *ctfe.Func {
	sig, ok := obj.typ.(*Signature)
	if !ok || body.Body == nil {
		return nil
	}
	fn := &ctfe.Func{Name: obj.name, Body: body.Body}
	if sig.TypeParams() != nil && sig.TypeParams().Len() > 0 {
		fn.Generic = true
		return fn
	}
	n := 0
	for _, f := range body.Type.ParamList {
		name := f.Name
		if name == nil {
			name = syntax.NewName(syntax.StartPos(f.Type), "")
		}
		var pt *ctfe.Type
		if params := sig.Params(); n < params.Len() {
			pt = h.fromType(params.At(n).Type())
		}
		fn.Params = append(fn.Params, ctfe.Param{Decl: name, Type: pt})
		n++
	}
	if sig.Variadic() && len(fn.Params) > 0 {
		fn.Variadic = true
	}
	results := sig.Results()
	names := resultNames(body)
	for i := range results.Len() {
		r := results.At(i)
		fn.Results = append(fn.Results, h.fromType(r.Type()))
		name := ""
		if i < len(names) {
			name = names[i]
		}
		if name == "_" {
			name = ""
		}
		fn.ResultNames = append(fn.ResultNames, name)
	}
	return fn
}

func resultNames(body *syntax.FuncDecl) []string {
	var out []string
	for _, f := range body.Type.ResultList {
		if f.Name != nil {
			out = append(out, f.Name.Value)
		} else {
			out = append(out, "")
		}
	}
	return out
}

func isCTFEIntKind(k ctfe.Kind) bool {
	switch k {
	case ctfe.KInt, ctfe.KInt8, ctfe.KInt16, ctfe.KInt32, ctfe.KInt64,
		ctfe.KUint, ctfe.KUint8, ctfe.KUint16, ctfe.KUint32, ctfe.KUint64, ctfe.KUintptr,
		ctfe.KUntypedInt, ctfe.KUntypedRune:
		return true
	}
	return false
}

func parseConstInt(s string) (*big.Int, bool) {
	n, ok := new(big.Int).SetString(s, 0)
	if !ok {
		return nil, false
	}
	return n, true
}

func parseDecimal(s string, out *int64) (int64, error) {
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, err
	}
	*out = n
	return n, nil
}
