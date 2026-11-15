// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"internal/buildcfg"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func jsonLowered(t types2.Type) types2.Type {
	for {
		switch c := types2.Unalias(t).(type) {
		case *types2.ConstView:
			t = c.Elem
		case *types2.ReadonlyDescendants:
			t = c.Elem
		default:
			o := jsonOptionOf(c)
			if o == nil || !types2.LogosOptionIsBare(o.Elem) {
				return c
			}
			t = o.Elem
		}
	}
}

func jsonOptionOf(t types2.Type) *types2.Option {
	switch c := types2.Unalias(t).(type) {
	case *types2.Option:
		return c
	case *types2.Named:
		o, _ := c.Underlying().(*types2.Option)
		return o
	}
	return nil
}

func jsonOptionStruct(o *types2.Option) *types2.Struct {
	return types2.NewStruct([]*types2.Var{
		types2.NewField(syntax.Pos{}, nil, "Value", o.Elem, false),
		types2.NewField(syntax.Pos{}, nil, "Present", types2.Typ[types2.Bool], false),
	}, nil)
}

func jsonStructOf(t types2.Type) (*types2.Struct, bool) {
	if o := jsonOptionOf(t); o != nil {
		return jsonOptionStruct(o), true
	}
	st, ok := t.Underlying().(*types2.Struct)
	return st, ok
}

func jsonMethod(t types2.Type, name string) *types2.Func {
	obj, _, _ := types2.LookupFieldOrMethod(t, false, nil, name)
	fn, _ := obj.(*types2.Func)
	return fn
}

func jsonHasIsZero(t types2.Type) bool {
	fn := jsonMethod(t, "IsZero")
	return fn != nil && isIsZeroSignature(fn)
}

func joinOmit(tests ...string) string {
	kept := []string{"false"}
	for _, t := range tests {
		if t != "" {
			kept = append(kept, "("+t+")")
		}
	}
	return strings.Join(kept, " || ")
}

func jsonEmptyTest(t types2.Type, x string) string {
	lt := jsonLowered(t)
	if jsonOptionOf(lt) != nil {
		return ""
	}
	switch u := lt.Underlying().(type) {
	case *types2.Array, *types2.Map, *types2.Slice:
		return "len(" + x + ") == 0"
	case *types2.Pointer, *types2.Interface:
		return x + " == nil"
	case *types2.Basic:
		return basicEmptyTest(u, x)
	}
	return ""
}

func basicEmptyTest(b *types2.Basic, x string) string {
	switch info := b.Info(); {
	case info&types2.IsString != 0:
		return "len(" + x + ") == 0"
	case info&types2.IsBoolean != 0:
		return "!(" + x + ")"
	case info&(types2.IsInteger|types2.IsFloat) != 0:
		return x + " == 0"
	}
	return ""
}

type jsonZero struct {
	qualify func(*types2.Package) string
	pkg     *types2.Package
	depth   int
}

func (z *jsonZero) zeroTest(t types2.Type, x, addr string) string {
	lt := jsonLowered(t)
	if _, isIface := lt.Underlying().(*types2.Interface); isIface {
		return z.interfaceZero(lt, x)
	}
	if _, isPtr := lt.Underlying().(*types2.Pointer); isPtr && jsonHasIsZero(lt) {
		return x + " == nil || " + x + ".IsZero()"
	}
	if jsonHasIsZero(lt) {
		return x + ".IsZero()"
	}
	if jsonHasIsZero(types2.NewPointer(lt)) {
		if addr != "" {
			return x + ".IsZero()"
		}
		return "(" + jsonAlias + ".JSONV2 && " + x + ".IsZero()) || (!" + jsonAlias + ".JSONV2 && func() bool { _logos_iz := " + x + "; return _logos_iz.IsZero() }())"
	}
	return z.plainZero(lt, x)
}

func (z *jsonZero) interfaceZero(lt types2.Type, x string) string {
	if !jsonHasIsZero(lt) {
		return x + " == nil"
	}
	members, ok := types2.LogosUnionMembers(lt)
	if !ok {
		return x + " == nil || " + jsonAlias + ".IsNilPointer(" + x + ") || " + x + ".IsZero()"
	}
	tests := []string{x + " == nil"}
	for _, m := range members {
		if m.Type != nil {
			if _, isPtr := types2.Unalias(m.Type).(*types2.Pointer); isPtr {
				tests = append(tests, x+" == ("+types2.TypeString(m.Type, z.qualify)+")(nil)")
			}
			continue
		}
		if !m.Ptr {
			continue
		}
		tests = append(tests, x+" == ("+z.importedMemberSpell(m)+")(nil)")
	}
	return strings.Join(append(tests, x+".IsZero()"), " || ")
}

func (z *jsonZero) importedMemberSpell(m types2.LogosUnionMember) string {
	name := m.Name
	if pkg := memberPackageFor(z.pkg, m.Pkg); pkg != nil {
		if q := z.qualify(pkg); q != "" {
			name = q + "." + m.Name
		}
	}
	return "*" + name
}

func (z *jsonZero) plainZero(lt types2.Type, x string) string {
	if st, ok := jsonStructOf(lt); ok {
		return z.structZero(st, x)
	}
	switch u := lt.Underlying().(type) {
	case *types2.Basic:
		return basicZero(u, x)
	case *types2.Array:
		return z.arrayZero(u, x)
	}
	return x + " == nil"
}

func basicZero(b *types2.Basic, x string) string {
	switch info := b.Info(); {
	case info&types2.IsBoolean != 0:
		return "!(" + x + ")"
	case info&types2.IsString != 0:
		return x + ` == ""`
	case b.Kind() == types2.UnsafePointer:
		return x + " == nil"
	}
	return x + " == 0"
}

func (z *jsonZero) arrayZero(a *types2.Array, x string) string {
	if types2.Comparable(a) {
		return jsonAlias + ".IsZero(" + x + ")"
	}
	i := fmt.Sprintf("_logos_z%d", z.depth)
	z.depth++
	elem := z.plainZero(jsonLowered(a.Elem()), x+"["+i+"]")
	return "func() bool {\nfor " + i + " := range len(" + x + ") {\nif !(" + elem + ") { return false }\n}\nreturn true\n}()"
}

func (z *jsonZero) structZero(st *types2.Struct, x string) string {
	if types2.Comparable(st) {
		return jsonAlias + ".IsZero(" + x + ")"
	}
	if !z.fieldsAccessible(st) {
		return z.layoutZero(st, x)
	}
	parts := []string{"true"}
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() != "_" {
			parts = append(parts, "("+z.plainZero(jsonLowered(f.Type()), x+"."+f.Name())+")")
		}
	}
	return strings.Join(parts, " && ")
}

func (z *jsonZero) fieldsAccessible(st *types2.Struct) bool {
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() != "_" && !f.Exported() && f.Pkg() != z.pkg {
			return false
		}
	}
	return true
}

func (z *jsonZero) layoutZero(st *types2.Struct, x string) string {
	sizes := types2.SizesFor("gc", buildcfg.GOARCH)
	u := z.qualify(types2.Unsafe)
	body := z.layoutStruct(sizes, st, "0")
	return "func(_logos_p " + u + ".Pointer) bool {\nreturn " + body + "\n}(" + u + ".Pointer(&" + x + "))"
}

func (z *jsonZero) layoutAt(off, typ string) string {
	return "*(*" + typ + ")(" + z.qualify(types2.Unsafe) + ".Add(_logos_p, " + off + "))"
}

func (z *jsonZero) layoutCheck(sizes types2.Sizes, t types2.Type, off string) string {
	switch u := t.Underlying().(type) {
	case *types2.Basic:
		return z.layoutBasic(u, off)
	case *types2.Array:
		i := fmt.Sprintf("_logos_z%d", z.depth)
		z.depth++
		elem := z.layoutCheck(sizes, u.Elem(), fmt.Sprintf("%s+%s*%d", off, i, sizes.Sizeof(u.Elem())))
		return fmt.Sprintf("func() bool {\nfor %s := range %d {\nif !(%s) { return false }\n}\nreturn true\n}()", i, u.Len(), elem)
	case *types2.Struct:
		return z.layoutStruct(sizes, u, off)
	}
	return z.layoutAt(off, z.qualify(types2.Unsafe)+".Pointer") + " == nil"
}

func (z *jsonZero) layoutBasic(b *types2.Basic, off string) string {
	if b.Kind() == types2.UnsafePointer {
		return z.layoutAt(off, z.qualify(types2.Unsafe)+".Pointer") + " == nil"
	}
	return basicZero(types2.Typ[b.Kind()], z.layoutAt(off, types2.Typ[b.Kind()].Name()))
}

func (z *jsonZero) layoutStruct(sizes types2.Sizes, st *types2.Struct, off string) string {
	fields := make([]*types2.Var, st.NumFields())
	for i := range fields {
		fields[i] = st.Field(i)
	}
	offsets := sizes.Offsetsof(fields)
	parts := []string{"true"}
	for i, f := range fields {
		if f.Name() != "_" {
			parts = append(parts, "("+z.layoutCheck(sizes, f.Type(), fmt.Sprintf("%s+%d", off, offsets[i]))+")")
		}
	}
	return strings.Join(parts, " && ")
}

func (g *marshalGen) zero() *jsonZero {
	return &jsonZero{qualify: g.js.qualify, pkg: g.js.holderPkg}
}

type jsonOmitOpts struct {
	empty bool
	zero  bool
	addr  string
}

func (z *jsonZero) omitted(t types2.Type, x string, opts jsonOmitOpts) string {
	var empty, zero string
	if opts.empty {
		empty = jsonEmptyTest(t, x)
	}
	if opts.zero {
		zero = z.zeroTest(t, x, opts.addr)
	}
	return joinOmit(empty, zero)
}

func (g *marshalGen) omitted(f *jsonField, x string) string {
	return g.zero().omitted(f.typ, x, jsonOmitOpts{empty: f.omitEmpty, zero: f.omitZero})
}
