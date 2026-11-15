// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"internal/logosfile"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerStaticBinary(_ *syntax.TypeDecl, named *types2.Named, js *jsonStruct, out *strings.Builder) {
	if !types2.LogosBinaryTypeOK(named) {
		return
	}
	for _, f := range js.fields {
		for _, m := range types2.LogosBinaryMethodNames() {
			if f.goName == m {
				return
			}
		}
	}
	needMarshal := !userDeclaresMethod(named, "MarshalBinary")
	needAppend := !userDeclaresMethod(named, "AppendBinary")
	needUnmarshal := !userDeclaresMethod(named, "UnmarshalBinary")
	if !needMarshal && !needAppend && !needUnmarshal {
		return
	}
	g := binaryGenerator{fl: fl, prefix: "_logos_bin_" + js.name, ids: make(map[types2.Type]int)}
	id, ok := g.codec(named)
	if !ok {
		return
	}
	out.WriteString(g.text.String())
	if needMarshal {
		fmt.Fprintf(out, "func (_logos_v %s) MarshalBinary() ([]byte, error) { return %s_e%d([]byte{1}, _logos_v, 1) }\n", js.name, g.prefix, id)
	}
	if needAppend {
		fmt.Fprintf(out, "func (_logos_v %s) AppendBinary(_logos_dst []byte) ([]byte, error) { return %s_e%d(append(_logos_dst, 1), _logos_v, 1) }\n", js.name, g.prefix, id)
	}
	if needUnmarshal {
		fmt.Fprintf(out, "func (_logos_v *%s) UnmarshalBinary(_logos_src []byte) error { _logos_off := 0; if len(_logos_src) == 0 { return _logos_bfmt.Errorf(\"binary field $version at offset 0: truncated input\") }; if _logos_src[0] != 1 { return _logos_bfmt.Errorf(\"binary field $version at offset 0: wrong version %%d\", _logos_src[0]) }; _logos_off++; if _logos_err := %s_d%d(_logos_src, &_logos_off, _logos_v, %q, 1); _logos_err != nil { return _logos_err }; if _logos_off != len(_logos_src) { return _logos_bfmt.Errorf(\"binary field $trailing at offset %%d: trailing bytes\", _logos_off) }; return nil }\n", js.name, g.prefix, id, js.name)
	}
}

type binaryGenerator struct {
	fl     *fileLower
	prefix string
	ids    map[types2.Type]int
	text   strings.Builder
}

func (g *binaryGenerator) codec(t types2.Type) (int, bool) {
	t = types2.Unalias(t)
	if id, ok := g.ids[t]; ok {
		return id, true
	}
	id := len(g.ids)
	g.ids[t] = id
	name, ok := g.fl.spellLowered(t)
	if !ok {
		return 0, false
	}
	var enc, dec strings.Builder
	e := func(format string, args ...any) { fmt.Fprintf(&enc, format+"\n", args...) }
	d := func(format string, args ...any) { fmt.Fprintf(&dec, format+"\n", args...) }
	child := func(et types2.Type, value, target, field string) bool {
		c, ok := g.codec(et)
		if !ok {
			return false
		}
		if g.isDepthComposite(et) {
			e("if _logos_depth+1 > 10000 { return _logos_dst, _logos_bfmt.Errorf(\"binary nesting depth exceeds 10000\") }")
			e("_logos_dst, _logos_err = %s_e%d(_logos_dst, %s, _logos_depth+1); if _logos_err != nil { return _logos_dst, _logos_err }", g.prefix, c, value)
			d("if _logos_depth+1 > 10000 { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: nesting depth exceeds 10000\", %s, *_logos_off) }", field)
			d("if _logos_err := %s_d%d(_logos_src, _logos_off, %s, %s, _logos_depth+1); _logos_err != nil { return _logos_err }", g.prefix, c, target, field)
		} else {
			e("_logos_dst, _logos_err = %s_e%d(_logos_dst, %s, _logos_depth); if _logos_err != nil { return _logos_dst, _logos_err }", g.prefix, c, value)
			d("if _logos_err := %s_d%d(_logos_src, _logos_off, %s, %s, _logos_depth); _logos_err != nil { return _logos_err }", g.prefix, c, target, field)
		}
		return true
	}
	var tup *types2.LogosTuple
	if x, ok := types2.Unalias(t).(*types2.LogosTuple); ok {
		tup = x
	} else if x, ok := t.Underlying().(*types2.LogosTuple); ok {
		tup = x
	}
	fail := "return _logos_bfmt.Errorf(\"binary field %s at offset %d: truncated input\", _logos_field, *_logos_off)"
	if opt := g.fl.asOption(t); opt != nil && types2.LogosOptionIsBare(opt.Elem) {
		payload, ok := g.codec(opt.Elem)
		if !ok {
			return 0, false
		}
		fmt.Fprintf(&g.text, "func %s_e%d(_logos_dst []byte, _logos_v %s, _logos_depth int) ([]byte, error) { return %s_e%d(_logos_dst, _logos_v, _logos_depth) }\n", g.prefix, id, name, g.prefix, payload)
		fmt.Fprintf(&g.text, "func %s_d%d(_logos_src []byte, _logos_off *int, _logos_v *%s, _logos_field string, _logos_depth int) error { return %s_d%d(_logos_src, _logos_off, _logos_v, _logos_field, _logos_depth) }\n", g.prefix, id, name, g.prefix, payload)
		return id, true
	}
	if opt := g.fl.asOption(t); opt != nil {
		e("if !_logos_v.Present { return append(_logos_dst, 0), nil }; _logos_dst = append(_logos_dst, 1)")
		d("if *_logos_off >= len(_logos_src) { %s }; _logos_presence := _logos_src[*_logos_off]; *_logos_off++; if _logos_presence > 1 { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: bad presence byte %%d\", _logos_field, *_logos_off-1, _logos_presence) }; *_logos_v = %s{}; if _logos_presence == 0 { return nil }; _logos_v.Present = true", fail, name)
		if !child(opt.Elem, "_logos_v.Value", "&_logos_v.Value", "_logos_field") {
			return 0, false
		}
	} else if tup != nil {
		for i := range tup.Len() {
			fname := tupleLoweredName(tup.Name(i), i)
			field := "_logos_field + " + fmt.Sprintf("%q", "."+fname)
			if !child(tup.Elem(i), "_logos_v."+fname, "&_logos_v."+fname, field) {
				return 0, false
			}
		}
	} else {
		u := t.Underlying()
		switch x := u.(type) {
		case *types2.Basic:
			s, _ := basicScalar(x)
			switch s {
			case scalarBool:
				e("if _logos_v { _logos_dst = append(_logos_dst, 1) } else { _logos_dst = append(_logos_dst, 0) }")
				d("if *_logos_off >= len(_logos_src) { %s }; _logos_b := _logos_src[*_logos_off]; *_logos_off++; if _logos_b > 1 { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: bad bool byte %%d\", _logos_field, *_logos_off-1, _logos_b) }; *_logos_v = %s(_logos_b == 1)", fail, name)
			case scalarString:
				e("_logos_dst = _logos_bbinary.AppendUvarint(_logos_dst, uint64(len(_logos_v))); _logos_dst = append(_logos_dst, _logos_v...)")
				d("_logos_n, _logos_used := _logos_bbinary.Uvarint(_logos_src[*_logos_off:]); if _logos_used <= 0 { %s }; *_logos_off += _logos_used; if _logos_n > uint64(len(_logos_src)-*_logos_off) { %s }; *_logos_v = %s(string(_logos_src[*_logos_off:*_logos_off+int(_logos_n)])); *_logos_off += int(_logos_n)", fail, fail, name)
			default:
				width, bits, decode := 8, "uint64(_logos_v)", "_logos_n"
				if s == scalarFloat32 {
					width, bits, decode = 4, "uint64(_logos_bmath.Float32bits(float32(_logos_v)))", "_logos_bmath.Float32frombits(uint32(_logos_n))"
				}
				if s == scalarFloat64 {
					bits, decode = "_logos_bmath.Float64bits(float64(_logos_v))", "_logos_bmath.Float64frombits(_logos_n)"
				}
				tail := fmt.Sprintf("*_logos_off += %d; *_logos_v = %s(%s)", width, name, decode)
				if s != scalarFloat32 && s != scalarFloat64 {
					tail = fmt.Sprintf("_logos_x := %s(_logos_n); if uint64(_logos_x) != _logos_n { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: value %%d out of range\", _logos_field, *_logos_off, _logos_n) }; *_logos_off += %d; *_logos_v = _logos_x", name, width)
				}
				e("_logos_n := %s; for _logos_i := uint(0); _logos_i < %d; _logos_i++ { _logos_dst = append(_logos_dst, byte(_logos_n >> (8*_logos_i))) }", bits, width)
				d("if len(_logos_src)-*_logos_off < %d { %s }; var _logos_n uint64; for _logos_i := uint(0); _logos_i < %d; _logos_i++ { _logos_n |= uint64(_logos_src[*_logos_off+int(_logos_i)]) << (8*_logos_i) }; %s", width, fail, width, tail)
			}
		case *types2.Pointer:
			pointed, ok := g.fl.spellLowered(x.Elem())
			if !ok {
				return 0, false
			}
			e("if _logos_v == nil { return append(_logos_dst, 0), nil }; _logos_dst = append(_logos_dst, 1)")
			d("if *_logos_off >= len(_logos_src) { %s }; _logos_p := _logos_src[*_logos_off]; *_logos_off++; if _logos_p > 1 { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: bad presence byte %%d\", _logos_field, *_logos_off-1, _logos_p) }; if _logos_p == 0 { *_logos_v = nil; return nil }; *_logos_v = new(%s)", fail, pointed)
			if !child(x.Elem(), "*_logos_v", "*_logos_v", "_logos_field") {
				return 0, false
			}
		case *types2.Struct:
			for i := range x.NumFields() {
				if _, ok := types2.LogosJSONFieldOf(x, i); !ok {
					continue
				}
				f := x.Field(i)
				field := "_logos_field + " + fmt.Sprintf("%q", "."+f.Name())
				ft := f.Type()
				if cv, ok := types2.Unalias(ft).(*types2.ConstView); ok {
					ft = cv.Elem
				}
				_, bare := types2.Unalias(ft).(*types2.Pointer)
				nonNil := bare && logosfile.Is(f.Pos().RelFilename())
				if nonNil {
					d("{ _logos_at := *_logos_off")
				}
				if !child(f.Type(), "_logos_v."+f.Name(), "&_logos_v."+f.Name(), field) {
					return 0, false
				}
				if nonNil {
					d("if _logos_v.%s == nil { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: missing required non-nil value\", %s, _logos_at) } }", f.Name(), field)
				}
			}
		case *types2.Slice:
			if b, ok := x.Elem().Underlying().(*types2.Basic); ok && b.Kind() == types2.Uint8 {
				head := "_logos_n, _logos_used := _logos_bbinary.Uvarint(_logos_src[*_logos_off:]); if _logos_used <= 0 { " + fail + " }; *_logos_off += _logos_used; if _logos_n > uint64(len(_logos_src)-*_logos_off) { " + fail + " }; _logos_cap64 := min(_logos_n, uint64(len(_logos_src)-*_logos_off)/2, (1<<20)); "
				count := "_logos_dst = _logos_bbinary.AppendUvarint(_logos_dst, uint64(len(_logos_v))); "
				if _, plain := types2.Unalias(x.Elem()).(*types2.Basic); plain {
					e(count + "_logos_dst = append(_logos_dst, _logos_v...)")
					d("%s*_logos_v = append(make(%s, 0, int(_logos_cap64)), _logos_src[*_logos_off:*_logos_off+int(_logos_n)]...); *_logos_off += int(_logos_n)", head, name)
					break
				}
				elemType, ok := g.fl.spellLowered(x.Elem())
				if !ok {
					return 0, false
				}
				e(count + "for _, _logos_b := range _logos_v { _logos_dst = append(_logos_dst, byte(_logos_b)) }")
				d("%s*_logos_v = make(%s, 0, int(_logos_cap64)); for _, _logos_b := range _logos_src[*_logos_off:*_logos_off+int(_logos_n)] { *_logos_v = append(*_logos_v, %s(_logos_b)) }; *_logos_off += int(_logos_n)", head, name, elemType)
				break
			}
			if g.isZeroWire(x.Elem()) {
				elemType, ok := g.fl.spellLowered(x.Elem())
				if !ok {
					return 0, false
				}
				e("if uint64(len(_logos_v)) > (1 << 20) { return _logos_dst, _logos_bfmt.Errorf(\"binary count %%d out of range\", len(_logos_v)) }")
				e("_logos_zsize := uint64(_logos_bunsafe.Sizeof(*new(%s))); if _logos_zsize == 0 { _logos_zsize = 1 }", elemType)
				e("if uint64(len(_logos_v)) > (1 << 23)/_logos_zsize { return _logos_dst, _logos_bfmt.Errorf(\"binary count %%d out of range\", len(_logos_v)) }")
				e("_logos_dst = _logos_bbinary.AppendUvarint(_logos_dst, uint64(len(_logos_v)))")
				d("_logos_n, _logos_used := _logos_bbinary.Uvarint(_logos_src[*_logos_off:]); if _logos_used <= 0 { %s }; *_logos_off += _logos_used", fail)
				d("if _logos_n > uint64(int(^uint(0) >> 1)) { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: count %%d out of range\", _logos_field, *_logos_off-_logos_used, _logos_n) }")
				d("if _logos_n > (1 << 20) { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: count %%d out of range\", _logos_field, *_logos_off-_logos_used, _logos_n) }")
				d("_logos_zsize := uint64(_logos_bunsafe.Sizeof(*new(%s))); if _logos_zsize == 0 { _logos_zsize = 1 }", elemType)
				d("if _logos_n > (1 << 23)/_logos_zsize { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: count %%d out of range\", _logos_field, *_logos_off-_logos_used, _logos_n) }")
				d("*_logos_v = make(%s, int(_logos_n))", name)
				break
			}
			elemType, ok := g.fl.spellLowered(x.Elem())
			if !ok {
				return 0, false
			}
			e("_logos_dst = _logos_bbinary.AppendUvarint(_logos_dst, uint64(len(_logos_v))); for _, _logos_elem := range _logos_v {")
			d("_logos_n, _logos_used := _logos_bbinary.Uvarint(_logos_src[*_logos_off:]); if _logos_used <= 0 { %s }; *_logos_off += _logos_used; if _logos_n > uint64(len(_logos_src)-*_logos_off) { %s }; var _logos_elem %s; _logos_size := uint64(_logos_bunsafe.Sizeof(_logos_elem)); if _logos_size == 0 { _logos_size = 1 }; _logos_cap64 := min(_logos_n, uint64(len(_logos_src)-*_logos_off)/2, (1<<20)/_logos_size); *_logos_v = make(%s, 0, int(_logos_cap64)); for _logos_i := uint64(0); _logos_i < _logos_n; _logos_i++ {", fail, fail, elemType, name)
			if !child(x.Elem(), "_logos_elem", "&_logos_elem", "_logos_field") {
				return 0, false
			}
			e("}")
			d("*_logos_v = append(*_logos_v, _logos_elem) }")
		case *types2.Array:
			elemType, ok := g.fl.spellLowered(x.Elem())
			if !ok {
				return 0, false
			}
			e("_logos_dst = _logos_bbinary.AppendUvarint(_logos_dst, uint64(len(_logos_v))); for _, _logos_elem := range _logos_v {")
			d("_logos_n, _logos_used := _logos_bbinary.Uvarint(_logos_src[*_logos_off:]); if _logos_used <= 0 { %s }; *_logos_off += _logos_used; if _logos_n != %d { return _logos_bfmt.Errorf(\"binary field %%s at offset %%d: array count %%d, want %d\", _logos_field, *_logos_off-_logos_used, _logos_n) }; for _logos_i := range *_logos_v { var _logos_elem %s", fail, x.Len(), x.Len(), elemType)
			if !child(x.Elem(), "_logos_elem", "&_logos_elem", "_logos_field") {
				return 0, false
			}
			e("}")
			d("(*_logos_v)[_logos_i] = _logos_elem }")
		case *types2.Map:
			keyType, keyOK := g.fl.spellLowered(x.Key())
			valueType, valueOK := g.fl.spellLowered(x.Elem())
			if !keyOK || !valueOK {
				return 0, false
			}
			e("_logos_dst = _logos_bbinary.AppendUvarint(_logos_dst, uint64(len(_logos_v))); _logos_keys := make([]string, 0, len(_logos_v)); for _logos_key := range _logos_v { _logos_keys = append(_logos_keys, string(_logos_key)) }; _logos_bslices.Sort(_logos_keys); for _, _logos_raw := range _logos_keys { _logos_key := %s(_logos_raw); _logos_elem := _logos_v[_logos_key]", keyType)
			d("_logos_n, _logos_used := _logos_bbinary.Uvarint(_logos_src[*_logos_off:]); if _logos_used <= 0 { %s }; *_logos_off += _logos_used; *_logos_v = make(%s); for _logos_i := uint64(0); _logos_i < _logos_n; _logos_i++ { var _logos_key %s; var _logos_elem %s", fail, name, keyType, valueType)
			if !child(x.Key(), "_logos_key", "&_logos_key", "_logos_field") || !child(x.Elem(), "_logos_elem", "&_logos_elem", "_logos_field") {
				return 0, false
			}
			e("}")
			d("(*_logos_v)[_logos_key] = _logos_elem }")
		default:
			return 0, false
		}
	}
	fmt.Fprintf(&g.text, "func %s_e%d(_logos_dst []byte, _logos_v %s, _logos_depth int) ([]byte, error) { var _logos_err error; _ = _logos_err\n%sreturn _logos_dst, nil\n}\n", g.prefix, id, name, enc.String())
	fmt.Fprintf(&g.text, "func %s_d%d(_logos_src []byte, _logos_off *int, _logos_v *%s, _logos_field string, _logos_depth int) error {\n%sreturn nil\n}\n", g.prefix, id, name, dec.String())
	return id, true
}

func (g *binaryGenerator) isZeroWire(t types2.Type) bool {
	return g.zeroWire(types2.Unalias(t), make(map[types2.Type]bool))
}
func (g *binaryGenerator) isDepthComposite(t types2.Type) bool {
	t = types2.Unalias(t)
	if g.fl.asOption(t) != nil {
		return false
	}
	if cv, ok := t.(*types2.ConstView); ok {
		t = cv.Elem
	}
	u := t.Underlying()
	if cv, ok := u.(*types2.ConstView); ok {
		u = cv.Elem
	}
	if lt, ok := u.(*types2.LogosTuple); ok {
		u = lt.Underlying()
	}
	switch u.(type) {
	case *types2.Struct, *types2.Slice, *types2.Array, *types2.Map:
		return true
	default:
		return false
	}
}

func (g *binaryGenerator) zeroWire(t types2.Type, seen map[types2.Type]bool) bool {
	t = types2.Unalias(t)
	if seen[t] {
		return false
	}
	seen[t] = true
	defer delete(seen, t)
	if opt := g.fl.asOption(t); opt != nil {
		if types2.LogosOptionIsBare(opt.Elem) {
			return g.zeroWire(opt.Elem, seen)
		}
		return false
	}
	if cv, ok := t.(*types2.ConstView); ok {
		return g.zeroWire(cv.Elem, seen)
	}
	u := t.Underlying()
	if cv, ok := u.(*types2.ConstView); ok {
		return g.zeroWire(cv.Elem, seen)
	}
	switch x := u.(type) {
	case *types2.Basic:
		return false
	case *types2.Pointer:
		return false
	case *types2.Slice:
		return false
	case *types2.Array:
		return false
	case *types2.Map:
		return false
	case *types2.LogosTuple:
		for i := range x.Len() {
			if !g.zeroWire(x.Elem(i), seen) {
				return false
			}
		}
		return true
	case *types2.Struct:
		for i := range x.NumFields() {
			if _, ok := types2.LogosJSONFieldOf(x, i); !ok {
				continue
			}
			if !g.zeroWire(x.Field(i).Type(), seen) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

type binaryImport struct{ alias, path string }

var binaryImportTable = []binaryImport{
	{"_logos_bfmt", "fmt"},
	{"_logos_bbinary", "encoding/binary"},
	{"_logos_bmath", "math"},
	{"_logos_bunsafe", "unsafe"},
	{"_logos_bslices", "slices"},
}

func (b binaryImport) decl(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{LocalPkgName: nameOf(pos, b.alias), Path: litOf(pos, fmt.Sprintf("%q", b.path), syntax.StringLit)}, pos)
}
