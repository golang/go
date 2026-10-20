// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file generates the binary encoding of SPEC Sec 12.3 (M11-S3,
// T-Sec12.1): for every struct type that types2.LogosStaticStruct
// accepts, the methods MarshalBinary, AppendBinary, and UnmarshalBinary
// that implement encoding.BinaryMarshaler, encoding.BinaryAppender, and
// encoding.BinaryUnmarshaler. The type checker exposes the same three
// methods in its first pass, so a call site type-checks before this pass
// emits the declarations. A method the program declares itself is never
// generated. The text is plain Go over the logos/rt helpers (binary.go
// there) and is parsed in place, like the const map code of M5-S4: the
// native tree and the eject text carry the same declarations.
//
// A struct with a field that has no layout (complex numbers,
// unsafe.Pointer, a struct of another package without its own
// MarshalBinary and UnmarshalBinary, a type that reaches itself) gets no
// binary methods, as the JSON-ineligible kinds get no JSON methods: a
// method that always fails would still satisfy encoding.BinaryMarshaler
// and hijack encoding/gob, which prefers it to the reflective path. The
// first checker pass has already exposed the methods, so only a call of a
// method on such a struct fails, and it fails in the recheck.

const (
	binRecv    = "_logos_v"
	binBuf     = "_logos_b"
	binBuf0    = "_logos_b0"
	binReader  = "_logos_r"
	binData    = "_logos_data"
	binScratch = "_logos_t"
)

// genText accumulates indented generated source.
type genText struct {
	b   strings.Builder
	ind int
}

func (t *genText) linef(format string, args ...any) {
	for range t.ind {
		t.b.WriteByte('\t')
	}
	fmt.Fprintf(&t.b, format, args...)
	t.b.WriteByte('\n')
}

func (t *genText) open(format string, args ...any) {
	t.linef(format, args...)
	t.ind++
}

func (t *genText) close(line string) {
	t.ind--
	t.linef("%s", line)
}

func (t *genText) elseBlock() {
	t.ind--
	t.linef("} else {")
	t.ind++
}

// binaryGen emits the three methods of one struct type.
type binaryGen struct {
	fl      *fileLower
	layout  *binaryLayout
	typ     string
	recv    string
	seq     int
	failing bool
	out     genText
}

// userMethodSet maps a package-level type name to the names of the
// methods the program declares on it.
type userMethodSet map[string]map[string]bool

// collectUserMethods records every method declaration of the package, so
// generation skips the names a program wrote itself.
func collectUserMethods(files []*syntax.File) userMethodSet {
	out := userMethodSet{}
	for _, file := range files {
		for _, decl := range file.DeclList {
			fn, ok := decl.(*syntax.FuncDecl)
			if !ok || fn.Recv == nil {
				continue
			}
			name := receiverBaseName(fn.Recv.Type)
			if out[name] == nil {
				out[name] = map[string]bool{}
			}
			out[name][fn.Name.Value] = true
		}
	}
	return out
}

// receiverBaseName returns the type name of a receiver type expression,
// looking through parentheses, one pointer, and type arguments.
func receiverBaseName(e syntax.Expr) string {
	for e != nil {
		if n, ok := e.(*syntax.Name); ok {
			return n.Value
		}
		e = receiverInner(e)
	}
	return ""
}

// receiverInner peels one parenthesis, pointer, or index level off a
// receiver type expression, or returns nil.
func receiverInner(e syntax.Expr) syntax.Expr {
	switch x := e.(type) {
	case *syntax.ParenExpr:
		return x.X
	case *syntax.IndexExpr:
		return x.X
	case *syntax.Operation:
		if x.Op == syntax.Mul && x.Y == nil {
			return x.X
		}
	}
	return nil
}

// lowerStaticBinary generates the binary methods of every eligible
// struct type declared in the file.
func (fl *fileLower) lowerStaticBinary() {
	for _, decl := range fl.file.DeclList {
		if td, ok := decl.(*syntax.TypeDecl); ok {
			fl.lowerBinaryDecl(td)
		}
	}
}

// lowerBinaryDecl generates the methods of one type declaration when the
// checker found it eligible.
func (fl *fileLower) lowerBinaryDecl(td *syntax.TypeDecl) {
	if td.Alias || len(td.TParamList) > 0 {
		return
	}
	obj, ok := fl.pkg.info.Defs[td.Name].(*types2.TypeName)
	if !ok {
		return
	}
	named, ok := obj.Type().(*types2.Named)
	if !ok {
		return
	}
	if _, ok := types2.LogosStaticStruct(fl.pkg.info, named, td.Type); ok {
		fl.generateBinary(td, named)
	}
}

// generateBinary emits and records the methods of one struct type.
func (fl *fileLower) generateBinary(td *syntax.TypeDecl, named *types2.Named) {
	g := &binaryGen{fl: fl, layout: fl.newBinaryLayout(named), typ: td.Name.Value, recv: td.Name.Value}
	st := named.Underlying().(*types2.Struct)
	if !g.layout.fieldsSupported(st) {
		return
	}
	declared := fl.pkg.userMethods[g.typ]
	methods := []struct {
		name string
		emit func(*types2.Struct) string
	}{
		{"AppendBinary", g.appendMethod},
		{"MarshalBinary", g.marshalMethod},
		{"UnmarshalBinary", g.unmarshalMethod},
	}
	var text strings.Builder
	for _, m := range methods {
		if !declared[m.name] {
			text.WriteString(m.emit(st))
		}
	}
	if text.Len() == 0 {
		return
	}
	fl.needRT = true
	fl.constDecls = append(fl.constDecls, parseConstMapDecls(text.String(), td.Pos())...)
	fl.constText += "\n" + text.String()
}

// tmp returns a fresh temporary name.
func (g *binaryGen) tmp(prefix string) string {
	g.seq++
	return fmt.Sprintf("_logos_%s%d", prefix, g.seq)
}

// spell renders t as lowered Go source in the file.
func (g *binaryGen) spell(t types2.Type) string {
	s, ok := g.fl.spellLowered(t)
	if !ok {
		base.Fatalf("logos: cannot spell binary field type %s of %s", t, g.typ)
	}
	return s
}

// wrap converts a helper result to a named type t; an unnamed type takes
// the helper result as is.
func (g *binaryGen) wrap(t types2.Type, expr string) string {
	if _, ok := types2.Unalias(t).(*types2.Named); ok {
		return g.spell(t) + "(" + expr + ")"
	}
	return expr
}

// binaryErrPath extends an error path with a field name.
func binaryErrPath(path, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}

// marshalMethod spells MarshalBinary over AppendBinary, reserving the
// version byte and the minimum width of every field.
func (g *binaryGen) marshalMethod(st *types2.Struct) string {
	return fmt.Sprintf("\nfunc (%s %s) MarshalBinary() ([]byte, error) {\n\treturn %s.AppendBinary(make([]byte, 0, %d))\n}\n",
		binRecv, g.recv, binRecv, 1+g.layout.structWidth(st))
}

// appendMethod spells AppendBinary: the version byte, then every field in
// declaration order.
func (g *binaryGen) appendMethod(st *types2.Struct) string {
	g.out = genText{ind: 1}
	g.failing = false
	g.encodeStruct(binRecv, st, "")
	var b strings.Builder
	fmt.Fprintf(&b, "\nfunc (%s %s) AppendBinary(%s []byte) ([]byte, error) {\n", binRecv, g.recv, binBuf)
	if g.failing {
		fmt.Fprintf(&b, "\t%s := %s\n", binBuf0, binBuf)
	}
	fmt.Fprintf(&b, "\t%s = append(%s, %s.BinaryVersion)\n", binBuf, binBuf, rtAlias)
	b.WriteString(g.out.b.String())
	fmt.Fprintf(&b, "\treturn %s, nil\n}\n", binBuf)
	return b.String()
}

// unmarshalMethod spells UnmarshalBinary: decode into a scratch value and
// store it only when the whole input decoded.
func (g *binaryGen) unmarshalMethod(st *types2.Struct) string {
	g.out = genText{ind: 1}
	g.decodeStruct(binScratch, st, "")
	var b strings.Builder
	fmt.Fprintf(&b, "\nfunc (%s *%s) UnmarshalBinary(%s []byte) error {\n", binRecv, g.recv, binData)
	fmt.Fprintf(&b, "\t%s := %s.BinaryReader{Type: %q, Buf: %s}\n", binReader, rtAlias, g.typ, binData)
	fmt.Fprintf(&b, "\t%s.Version()\n\tvar %s %s\n", binReader, binScratch, g.recv)
	b.WriteString(g.out.b.String())
	fmt.Fprintf(&b, "\tif err := %s.Done(); err != nil {\n\t\treturn err\n\t}\n\t*%s = %s\n\treturn nil\n}\n", binReader, binRecv, binScratch)
	return b.String()
}

// encode emits the statements that append expr, a value of type t.
func (g *binaryGen) encode(expr string, t types2.Type, field string) {
	shape := g.layout.shape(t)
	if g.encodeScalar(expr, shape) {
		return
	}
	g.encodeComposite(expr, t, shape, field)
}

// encodeScalar emits the append of a scalar shape and reports whether
// shape was one.
func (g *binaryGen) encodeScalar(expr string, shape binShape) bool {
	var call string
	switch shape {
	case binBool:
		call = fmt.Sprintf("BinAppendBool(%s, bool(%s))", binBuf, expr)
	case binSigned:
		call = fmt.Sprintf("BinAppendUint64(%s, uint64(int64(%s)))", binBuf, expr)
	case binUnsigned:
		call = fmt.Sprintf("BinAppendUint64(%s, uint64(%s))", binBuf, expr)
	case binFloat32:
		call = fmt.Sprintf("BinAppendFloat32(%s, float32(%s))", binBuf, expr)
	case binFloat64:
		call = fmt.Sprintf("BinAppendFloat64(%s, float64(%s))", binBuf, expr)
	case binString:
		call = fmt.Sprintf("BinAppendString(%s, string(%s))", binBuf, expr)
	case binBytes:
		call = fmt.Sprintf("BinAppendBytes(%s, []byte(%s))", binBuf, expr)
	default:
		return false
	}
	g.out.linef("%s = %s.%s", binBuf, rtAlias, call)
	return true
}

// encodeComposite emits the append of a non-scalar shape.
func (g *binaryGen) encodeComposite(expr string, t types2.Type, shape binShape, field string) {
	switch shape {
	case binSlice:
		g.encodeSlice(expr, t.Underlying().(*types2.Slice).Elem(), field)
	case binArray:
		g.encodeArray(expr, t.Underlying().(*types2.Array), field)
	case binMap:
		g.encodeMap(expr, t.Underlying().(*types2.Map).Elem(), field)
	case binStruct:
		g.encodeStruct(expr, t.Underlying().(*types2.Struct), field)
	case binPointer:
		g.encodePointer(expr, t.Underlying().(*types2.Pointer).Elem(), field)
	case binOption:
		g.encodeOption(expr, g.fl.asOption(t), field)
	case binMarshaler:
		g.encodeMarshaler(expr, field)
	}
}

func (g *binaryGen) encodeSlice(expr string, elem types2.Type, field string) {
	i := g.tmp("i")
	g.out.linef("%s = %s.BinAppendUvarint(%s, uint64(len(%s)))", binBuf, rtAlias, binBuf, expr)
	g.out.open("for %s := 0; %s < len(%s); %s++ {", i, i, expr, i)
	g.encode(fmt.Sprintf("%s[%s]", expr, i), elem, field+"[]")
	g.out.close("}")
}

func (g *binaryGen) encodeArray(expr string, a *types2.Array, field string) {
	i := g.tmp("i")
	g.out.linef("%s = %s.BinAppendUvarint(%s, %d)", binBuf, rtAlias, binBuf, a.Len())
	g.out.open("for %s := 0; %s < %d; %s++ {", i, i, a.Len(), i)
	g.encode(fmt.Sprintf("%s[%s]", expr, i), a.Elem(), field+"[]")
	g.out.close("}")
}

func (g *binaryGen) encodeMap(expr string, elem types2.Type, field string) {
	keys, key, val := g.tmp("keys"), g.tmp("key"), g.tmp("e")
	g.out.linef("%s := %s.BinSortedKeys(%s)", keys, rtAlias, expr)
	g.out.linef("%s = %s.BinAppendUvarint(%s, uint64(len(%s)))", binBuf, rtAlias, binBuf, keys)
	g.out.open("for _, %s := range %s {", key, keys)
	g.out.linef("%s = %s.BinAppendString(%s, string(%s))", binBuf, rtAlias, binBuf, key)
	g.out.linef("%s := %s[%s]", val, expr, key)
	g.out.linef("_ = %s", val)
	g.encode(val, elem, field+"[]")
	g.out.close("}")
}

func (g *binaryGen) encodeStruct(expr string, st *types2.Struct, field string) {
	for i := range st.NumFields() {
		if f := st.Field(i); f.Name() != "_" {
			g.encode(expr+"."+f.Name(), f.Type(), binaryErrPath(field, f.Name()))
		}
	}
}

func (g *binaryGen) encodePointer(expr string, elem types2.Type, field string) {
	g.failing = true
	g.out.open("if %s == nil {", expr)
	g.out.linef("return %s, %s.BinNil(%q, %q, len(%s))", binBuf0, rtAlias, g.typ, field, binBuf)
	g.out.close("}")
	g.encode("(*"+expr+")", elem, field)
}

func (g *binaryGen) encodeOption(expr string, o *types2.Option, field string) {
	if keepMapRead(o.Elem) {
		g.out.open("if %s != nil {", expr)
		g.out.linef("%s = append(%s, 1)", binBuf, binBuf)
		g.encode("(*"+expr+")", o.Elem.Underlying().(*types2.Pointer).Elem(), field)
	} else {
		g.out.open("if %s.Present {", expr)
		g.out.linef("%s = append(%s, 1)", binBuf, binBuf)
		g.encode(expr+".Value", o.Elem, field)
	}
	g.out.elseBlock()
	g.out.linef("%s = append(%s, 0)", binBuf, binBuf)
	g.out.close("}")
}

func (g *binaryGen) encodeMarshaler(expr, field string) {
	data, err := g.tmp("p"), g.tmp("err")
	g.failing = true
	g.out.linef("%s, %s := %s.MarshalBinary()", data, err, expr)
	g.out.open("if %s != nil {", err)
	g.out.linef("return %s, %s.BinInvalid(%q, %q, len(%s), %s)", binBuf0, rtAlias, g.typ, field, binBuf, err)
	g.out.close("}")
	g.out.linef("%s = %s.BinAppendBytes(%s, %s)", binBuf, rtAlias, binBuf, data)
}

// decode emits the statements that read a value of type t into lhs.
func (g *binaryGen) decode(lhs string, t types2.Type, field string) {
	shape := g.layout.shape(t)
	if g.decodeScalar(lhs, t, shape, field) {
		return
	}
	g.decodeComposite(lhs, t, shape, field)
}

// decodeScalar emits the read of a scalar shape and reports whether
// shape was one.
func (g *binaryGen) decodeScalar(lhs string, t types2.Type, shape binShape, field string) bool {
	var call string
	switch shape {
	case binBool:
		call = fmt.Sprintf("%s.Bool(%q)", binReader, field)
	case binSigned:
		call = fmt.Sprintf("%s.BinSigned[%s](&%s, %q)", rtAlias, basicName(t), binReader, field)
	case binUnsigned:
		call = fmt.Sprintf("%s.BinUnsigned[%s](&%s, %q)", rtAlias, basicName(t), binReader, field)
	case binFloat32:
		call = fmt.Sprintf("%s.Float32(%q)", binReader, field)
	case binFloat64:
		call = fmt.Sprintf("%s.Float64(%q)", binReader, field)
	case binString:
		call = fmt.Sprintf("%s.String(%q)", binReader, field)
	case binBytes:
		call = fmt.Sprintf("%s.Bytes(%q)", binReader, field)
	default:
		return false
	}
	g.out.linef("%s = %s", lhs, g.wrap(t, call))
	return true
}

// basicName spells the basic type under t.
func basicName(t types2.Type) string {
	return types2.Typ[t.Underlying().(*types2.Basic).Kind()].Name()
}

// decodeComposite emits the read of a non-scalar shape.
func (g *binaryGen) decodeComposite(lhs string, t types2.Type, shape binShape, field string) {
	switch shape {
	case binSlice:
		g.decodeSlice(lhs, t, field)
	case binArray:
		g.decodeArray(lhs, t.Underlying().(*types2.Array), field)
	case binMap:
		g.decodeMap(lhs, t, field)
	case binStruct:
		g.decodeStruct(lhs, t.Underlying().(*types2.Struct), field)
	case binPointer:
		g.decodePointer(lhs, t.Underlying().(*types2.Pointer).Elem(), field)
	case binOption:
		g.decodeOption(lhs, t, field)
	case binMarshaler:
		g.out.linef("%s.Unmarshal(%q, &%s)", binReader, field, lhs)
	}
}

func (g *binaryGen) decodeSlice(lhs string, t types2.Type, field string) {
	elem := t.Underlying().(*types2.Slice).Elem()
	n, s, i, e := g.tmp("n"), g.tmp("s"), g.tmp("i"), g.tmp("e")
	elemType := g.spell(elem)
	g.out.open("{")
	g.out.linef("%s := %s.Count(%q, %d)", n, binReader, field, g.layout.width(elem))
	g.out.linef("%s := %s.BinSlice[%s](&%s, %s)", s, rtAlias, elemType, binReader, n)
	g.out.open("for %s := 0; %s < %s && %s.Err == nil; %s++ {", i, i, n, binReader, i)
	g.out.linef("var %s %s", e, elemType)
	g.decode(e, elem, field+"[]")
	g.out.linef("%s = append(%s, %s)", s, s, e)
	g.out.close("}")
	g.out.linef("%s = %s", lhs, g.wrap(t, s))
	g.out.close("}")
}

func (g *binaryGen) decodeArray(lhs string, a *types2.Array, field string) {
	i := g.tmp("i")
	g.out.linef("%s.ArrayLen(%q, %d)", binReader, field, a.Len())
	g.out.open("for %s := 0; %s < %d && %s.Err == nil; %s++ {", i, i, a.Len(), binReader, i)
	g.decode(fmt.Sprintf("%s[%s]", lhs, i), a.Elem(), field+"[]")
	g.out.close("}")
}

func (g *binaryGen) decodeMap(lhs string, t types2.Type, field string) {
	m := t.Underlying().(*types2.Map)
	n, out, i, k, e := g.tmp("n"), g.tmp("m"), g.tmp("i"), g.tmp("key"), g.tmp("e")
	g.out.open("{")
	g.out.linef("%s := %s.Count(%q, %d)", n, binReader, field, 1+g.layout.width(m.Elem()))
	g.out.linef("%s := %s.BinMap[%s](&%s, %s)", out, rtAlias, g.spell(t), binReader, n)
	g.out.open("for %s := 0; %s < %s && %s.Err == nil; %s++ {", i, i, n, binReader, i)
	g.out.linef("%s := %s", k, g.wrap(m.Key(), fmt.Sprintf("%s.String(%q)", binReader, field+"[key]")))
	g.out.linef("var %s %s", e, g.spell(m.Elem()))
	g.decode(e, m.Elem(), field+"[]")
	g.out.linef("%s[%s] = %s", out, k, e)
	g.out.close("}")
	g.out.linef("%s = %s", lhs, out)
	g.out.close("}")
}

func (g *binaryGen) decodeStruct(lhs string, st *types2.Struct, field string) {
	for i := range st.NumFields() {
		if f := st.Field(i); f.Name() != "_" {
			g.decode(lhs+"."+f.Name(), f.Type(), binaryErrPath(field, f.Name()))
		}
	}
}

func (g *binaryGen) decodePointer(lhs string, elem types2.Type, field string) {
	p := g.tmp("p")
	g.out.open("{")
	g.out.linef("%s := new(%s)", p, g.spell(elem))
	g.decode("(*"+p+")", elem, field)
	g.out.linef("%s = %s", lhs, p)
	g.out.close("}")
}

func (g *binaryGen) decodeOption(lhs string, t types2.Type, field string) {
	o := g.fl.asOption(t)
	g.out.open("if %s.Presence(%q) {", binReader, field)
	if keepMapRead(o.Elem) {
		g.decode(lhs, o.Elem, field)
	} else {
		x := g.tmp("x")
		g.out.linef("var %s %s", x, g.spell(o.Elem))
		g.decode(x, o.Elem, field)
		g.out.linef("%s = %s{Value: %s, Present: true}", lhs, g.spell(t), x)
	}
	g.out.close("}")
}
