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

func indentLines(lines []string) []string {
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = "\t" + l
	}
	return out
}

func blockLines(head string, body []string) []string {
	out := []string{head + " {"}
	out = append(out, indentLines(body)...)
	return append(out, "}")
}

func ifLines(cond string, body []string) []string {
	return blockLines("if "+cond, body)
}

func ifElseLines(cond string, yes, no []string) []string {
	out := []string{"if " + cond + " {"}
	out = append(out, indentLines(yes)...)
	out = append(out, "} else {")
	out = append(out, indentLines(no)...)
	return append(out, "}")
}

type valueParts struct {
	konst int
	size  []string
	emit  []string
}

func (p valueParts) inlineSize() []string {
	if p.konst == 0 {
		return p.size
	}
	return append([]string{fmt.Sprintf("_logos_n += %d", p.konst)}, p.size...)
}

type fieldCode struct {
	pins  []string
	pre   []string
	size  []string
	emit  []string
	fixed int
}

type marshalGen struct {
	js          *jsonStruct
	firstFixed  int
	runtimeSep  bool
	recv        string
	edges       map[*jsonField]jsonEdge
	calls       map[*jsonField]string
	errorPrefix []string
	snapshot    bool
}

func (f *jsonField) optional() bool {
	return f.omitEmpty || f.omitZero
}

func newMarshalGen(js *jsonStruct) *marshalGen {
	g := &marshalGen{js: js, firstFixed: -1, recv: jsonRecv}
	for i, f := range js.fields {
		if !f.optional() {
			g.firstFixed = i
			break
		}
	}
	g.runtimeSep = len(js.fields) == 0 || js.fields[0].optional()
	return g
}

func (g *marshalGen) sepRuntimeAt(p int) bool {
	return g.runtimeSep && (g.firstFixed < 0 || p <= g.firstFixed)
}

func (g *marshalGen) litStmts(f *jsonField, open string) []string {
	key := keyLiteral(f.key) + open
	if g.sepRuntimeAt(f.pos) {
		return []string{"_logos_buf = append(_logos_buf, _logos_sep)", "_logos_buf = append(_logos_buf, `" + key + "`...)", "_logos_sep = ','"}
	}
	sep := ","
	if f.pos == 0 {
		sep = "{"
	}
	return []string{"_logos_buf = append(_logos_buf, `" + sep + key + "`...)"}
}

func (g *marshalGen) litLen(f *jsonField, open string) int {
	return 1 + len(keyLiteral(f.key)) + len(open)
}

func (g *marshalGen) scalarParts(f *jsonField, c scalarCodec, x string) valueParts {
	open := c.open()
	p := valueParts{konst: g.litLen(f, open) + len(open)}
	p.size = []string{c.sizeStmt(x)}
	p.emit = append(g.litStmts(f, open), c.emitStmt(x))
	if open != "" {
		p.emit = append(p.emit, `_logos_buf = append(_logos_buf, '"')`)
	}
	return p
}

func (g *marshalGen) nullParts(f *jsonField) valueParts {
	return valueParts{
		konst: g.litLen(f, "") + 4,
		emit:  append(g.litStmts(f, ""), `_logos_buf = append(_logos_buf, "null"...)`),
	}
}

func (f *jsonField) local() string {
	return fmt.Sprintf("_logos_f%d", f.pos)
}

func (f *jsonField) presenceName() string {
	return fmt.Sprintf("_logos_p%d", f.pos)
}

func (g *marshalGen) code(f *jsonField) fieldCode {
	if g.edges != nil {
		if e, ok := g.edges[f]; ok && e != edgeLegacy {
			return g.edgeCode(f, e)
		}
	}
	var c fieldCode
	switch f.shape.kind {
	case shapeFallback:
		if f.quoted {
			c = g.fallbackQuotedCode(f)
			return c
		}
		c = g.fallbackCode(f)
	case shapeScalar:
		c = g.scalarCode(f)
	case shapePointer:
		c = g.pointerCode(f)
	case shapeUnion:
		c = g.unionCode(f)
	default:
		c = g.sliceCode(f)
	}
	if f.shape.kind != shapeFallback && f.shape.kind != shapeUnion && f.optional() {
		c = g.guardPresence(f, c)
	}
	return c
}

func (g *marshalGen) guardPresence(f *jsonField, c fieldCode) fieldCode {
	p := f.presenceName()
	out := fieldCode{pins: c.pins}
	out.pre = []string{p + " := " + g.presence(f)}
	if len(c.pre) > 0 {
		out.pre = append(out.pre, ifLines(p, c.pre)...)
	}
	out.size = ifLines(p, append(constLine(c.fixed), c.size...))
	out.emit = ifLines(p, c.emit)
	return out
}

func constLine(n int) []string {
	if n == 0 {
		return nil
	}
	return []string{fmt.Sprintf("_logos_n += %d", n)}
}

func (g *marshalGen) presence(f *jsonField) string {
	x := f.local()
	switch f.shape.kind {
	case shapeScalar:
		if f.omitZero && f.shape.needsIsZero {
			if f.omitEmpty {
				return "!(" + f.codec().isZeroCond(x) + ") && !(" + g.zero().zeroTest(f.typ, x, "") + ")"
			}
			return "!(" + g.zero().zeroTest(f.typ, x, "") + ")"
		}
		return "!(" + f.codec().isZeroCond(x) + ")"
	case shapePointer:
		if f.omitZero && f.shape.needsIsZero {
			return x + " != nil && !(" + g.zero().zeroTest(f.typ, x, "") + ")"
		}
		return x + " != nil"
	case shapeUnion:
		if f.omitZero {
			return "!(" + g.zero().zeroTest(f.typ, x, "") + ")"
		}
		return x + " != nil"
	}
	var conds []string
	if f.omitEmpty {
		conds = append(conds, "len("+x+") != 0")
	}
	if f.omitZero {
		conds = append(conds, x+" != nil")
	}
	return strings.Join(conds, " && ")
}

func (g *marshalGen) scalarCode(f *jsonField) fieldCode {
	x := f.local()
	c := f.codec()
	p := g.scalarParts(f, c, x)
	out := fieldCode{pins: []string{x + " := " + g.recv + "." + f.goName}}
	if chk := c.checkStmt(x); chk != "" {
		out.pre = []string{chk}
	}
	out.size = p.size
	out.emit = p.emit
	out.fixed = p.konst
	return out
}

func (g *marshalGen) pointerCode(f *jsonField) fieldCode {
	x := f.local()
	c := f.codec()
	p := g.scalarParts(f, c, "*"+x)
	out := fieldCode{pins: []string{x + " := " + g.recv + "." + f.goName}}
	if chk := c.checkStmt("*" + x); chk != "" {
		out.pre = ifLines(x+" != nil", []string{chk})
	}
	if g.snapshot {
		capture := ifLines(x+" != nil", []string{"_logos_value := *" + x, x + " = &_logos_value"})
		out.pre = append(capture, out.pre...)
	}
	if f.optional() {
		out.size = p.inlineSize()
		out.emit = p.emit
		return out
	}
	n := g.nullParts(f)
	out.size = ifElseLines(x+" != nil", p.inlineSize(), n.inlineSize())
	out.emit = ifElseLines(x+" != nil", p.emit, n.emit)
	return out
}

func (g *marshalGen) sliceCode(f *jsonField) fieldCode {
	x := f.local()
	c := f.elemCodec()
	out := fieldCode{pins: []string{x + " := " + g.recv + "." + f.goName}}
	if g.snapshot {
		out.pre = []string{x + " = append(" + x + "[:0:0], " + x + "...)"}
	}
	if g.edges != nil {
		frame := append(ifLines("_logos_err := _logos_s.EnterElems(0, len("+x+")); _logos_err != nil", g.depthErrorBody(f, "_logos_err")), "_logos_s.LeaveFrame()")
		out.pre = append(out.pre, frame...)
	}
	if chk := c.checkStmt("_logos_x"); chk != "" {
		loop := append([]string{"for _, _logos_x := range " + x + " {"}, indentLines([]string{chk})...)
		out.pre = append(out.pre, append(loop, "}")...)
	}
	n := g.nullParts(f)
	out.size = ifElseLines(x+" == nil", n.inlineSize(), g.sliceSize(f, c, x))
	out.emit = ifElseLines(x+" == nil", n.emit, g.sliceEmit(f, c, x))
	return out
}

func (g *marshalGen) sliceSize(f *jsonField, c scalarCodec, x string) []string {
	lines := []string{fmt.Sprintf("_logos_n += %d", g.litLen(f, "[")+1)}
	if c.isFloat() {
		return append(lines, fmt.Sprintf("_logos_n += len(%s) * (%s.MaxFloatLen%d + 1)", x, jsonAlias, c.floatBits()))
	}
	body := append([]string{"if _logos_i > 0 { _logos_n++ }"}, elemSize(c, "_logos_x")...)
	lines = append(lines, "for _logos_i, _logos_x := range "+x+" {")
	lines = append(lines, indentLines(body)...)
	return append(lines, "}")
}

func (g *marshalGen) sliceEmit(f *jsonField, c scalarCodec, x string) []string {
	lines := append(g.litStmts(f, "["), "for _logos_i, _logos_x := range "+x+" {")
	body := append([]string{"if _logos_i > 0 { _logos_buf = append(_logos_buf, ',') }"}, elemEmit(c, "_logos_x")...)
	lines = append(lines, indentLines(body)...)
	return append(lines, "}", "_logos_buf = append(_logos_buf, ']')")
}

func elemSize(c scalarCodec, x string) []string {
	lines := []string{c.sizeStmt(x)}
	if n := 2 * len(c.open()); n > 0 {
		lines = append(lines, fmt.Sprintf("_logos_n += %d", n))
	}
	return lines
}

func elemEmit(c scalarCodec, x string) []string {
	var lines []string
	if c.open() != "" {
		lines = append(lines, `_logos_buf = append(_logos_buf, '"')`)
	}
	lines = append(lines, c.emitStmt(x))
	if c.open() != "" {
		lines = append(lines, `_logos_buf = append(_logos_buf, '"')`)
	}
	return lines
}

func (g *marshalGen) fallbackCode(f *jsonField) fieldCode {
	field := g.recv + "." + f.goName
	b := fmt.Sprintf("_logos_b%d", f.pos)
	e := fmt.Sprintf("_logos_e%d", f.pos)
	call := []string{
		fmt.Sprintf("%s, %s := %s.MarshalField(%s)", b, e, jsonAlias, field),
	}
	check := ifLines(e+" != nil", []string{"return nil, " + e})
	var c fieldCode
	if !f.optional() {
		c.pre = append(call, check...)
		c.size = []string{fmt.Sprintf("_logos_n += %d + len(%s)", g.litLen(f, ""), b)}
		c.emit = append(g.litStmts(f, ""), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b))
		return c
	}
	p := f.presenceName()
	c.pre = []string{p + " := !(" + g.omitted(f, field) + ")", "var " + b + " []byte"}
	inner := []string{fmt.Sprintf("var %s error", e), fmt.Sprintf("%s, %s = %s.MarshalField(%s)", b, e, jsonAlias, field)}
	c.pre = append(c.pre, ifLines(p, append(inner, check...))...)
	c.size = ifLines(p, []string{fmt.Sprintf("_logos_n += %d + len(%s)", g.litLen(f, ""), b)})
	c.emit = ifLines(p, append(g.litStmts(f, ""), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b)))
	return c
}

func (g *marshalGen) fallbackQuotedCode(f *jsonField) fieldCode {
	if isExactNumber(f.typ) {
		return g.fallbackNumberQuotedCode(f)
	}
	if hasMarshalMethods(f.typ) {
		return g.fallbackCode(f)
	}
	if kind, ok := fallbackScalarKind(f.typ); ok {
		return g.fallbackScalarQuotedCode(f, kind)
	}
	return g.fallbackCode(f)
}

func (g *marshalGen) fallbackScalarQuotedCode(f *jsonField, kind scalarKind) fieldCode {
	u := unaliasConstView(f.typ)
	_, isPtr := u.(*types2.Pointer)
	var elem types2.Type
	if isPtr {
		p := u.(*types2.Pointer)
		elem = unaliasConstView(p.Elem())
	} else {
		elem = u
		if _, ok := elem.(*types2.Option); ok {
			return g.fallbackCode(f)
		}
	}
	tmp := &jsonField{
		goName:    f.goName,
		key:       f.key,
		omitEmpty: f.omitEmpty,
		omitZero:  f.omitZero,
		quoted:    true,
		pos:       f.pos,
		required:  f.required,
		typ:       f.typ,
	}
	if isPtr {
		tmp.shape = jsonShape{kind: shapePointer, scalar: kind, nonNil: f.shape.nonNil, needsIsZero: needsIsZero(elem)}
		c := g.pointerCode(tmp)
		if tmp.optional() {
			c = g.guardPresence(tmp, c)
		}
		return c
	}
	tmp.shape = jsonShape{kind: shapeScalar, scalar: kind, nonNil: false, needsIsZero: needsIsZero(elem)}
	c := g.scalarCode(tmp)
	if tmp.optional() {
		c = g.guardPresence(tmp, c)
	}
	return c
}

func (g *marshalGen) fallbackNumberQuotedCode(f *jsonField) fieldCode {
	field := g.recv + "." + f.goName
	b := fmt.Sprintf("_logos_b%d", f.pos)
	e := fmt.Sprintf("_logos_e%d", f.pos)
	u := unaliasConstView(f.typ)
	_, isPtr := u.(*types2.Pointer)
	call := []string{
		fmt.Sprintf("%s, %s := %s.MarshalField(%s)", b, e, jsonAlias, field),
	}
	check := ifLines(e+" != nil", []string{"return nil, " + e})
	quotedSize := fmt.Sprintf("_logos_n += %d + len(%s) + 1", g.litLen(f, `"`), b)
	quotedEmit := append(g.litStmts(f, `"`), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b), `_logos_buf = append(_logos_buf, '"')`)
	nullSize := fmt.Sprintf("_logos_n += %d + 4", g.litLen(f, ""))
	nullEmit := append(g.litStmts(f, ""), `_logos_buf = append(_logos_buf, "null"...)`)
	var c fieldCode
	if !f.optional() {
		if !isPtr {
			c.pre = append(call, check...)
			c.size = []string{quotedSize}
			c.emit = quotedEmit
			return c
		}
		c.pre = append(call, check...)
		c.size = ifElseLines(field+" == nil", []string{nullSize}, []string{quotedSize})
		c.emit = ifElseLines(field+" == nil", nullEmit, quotedEmit)
		return c
	}
	p := f.presenceName()
	c.pre = []string{p + " := !(" + g.omitted(f, field) + ")", "var " + b + " []byte"}
	inner := []string{fmt.Sprintf("var %s error", e), fmt.Sprintf("%s, %s = %s.MarshalField(%s)", b, e, jsonAlias, field)}
	c.pre = append(c.pre, ifLines(p, append(inner, check...))...)
	if !isPtr {
		c.size = ifLines(p, []string{quotedSize})
		c.emit = ifLines(p, quotedEmit)
		return c
	}
	c.size = ifLines(p, ifElseLines(field+" == nil", []string{nullSize}, []string{quotedSize}))
	c.emit = ifLines(p, ifElseLines(field+" == nil", nullEmit, quotedEmit))
	return c
}

func unionPackage(t types2.Type) *types2.Package {
	switch u := types2.Unalias(t).(type) {
	case *types2.Named:
		if obj := u.Obj(); obj != nil {
			return obj.Pkg()
		}
	}
	if terms, ok := types2.LogosUnionTerms(t); ok {
		for _, m := range terms {
			switch mu := types2.Unalias(m).(type) {
			case *types2.Named:
				if obj := mu.Obj(); obj != nil {
					return obj.Pkg()
				}
			case *types2.Pointer:
				if named, ok := types2.Unalias(mu.Elem()).(*types2.Named); ok && named.Obj() != nil {
					return named.Obj().Pkg()
				}
			}
		}
	}
	return nil
}

func isMarshalJSONSignature(obj types2.Object) bool {
	fn, ok := obj.(*types2.Func)
	if !ok {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		return false
	}
	if sig.Params().Len() != 0 || sig.Results().Len() != 2 {
		return false
	}
	res0 := sig.Results().At(0).Type()
	res1 := sig.Results().At(1).Type()
	sl, ok := types2.Unalias(res0).(*types2.Slice)
	if !ok {
		return false
	}
	b, ok := types2.Unalias(sl.Elem()).(*types2.Basic)
	if !ok || b.Kind() != types2.Uint8 {
		return false
	}
	named, ok := types2.Unalias(res1).(*types2.Named)
	if ok && named.Obj() != nil && named.Obj().Name() == "error" && named.Obj().Pkg() == nil {
		return true
	}
	return res1.String() == "error"
}

func scalarStaticMarshal(kind scalarKind, val, b string) string {
	switch kind {
	case scalarBool:
		return fmt.Sprintf("if %s { %s = []byte(\"true\") } else { %s = []byte(\"false\") }", val, b, b)
	case scalarInt:
		return fmt.Sprintf("%s = %s.AppendInt(nil, int64(%s))", b, jsonAlias, val)
	case scalarUint:
		return fmt.Sprintf("%s = %s.AppendUint(nil, uint64(%s))", b, jsonAlias, val)
	case scalarFloat32:
		return fmt.Sprintf("if %s-%s != 0 { return nil, %s.FloatErr(%s, float64(%s), 32) }; %s = %s.AppendFloat(nil, float64(%s), 32)", val, val, jsonAlias, val, val, b, jsonAlias, val)
	case scalarFloat64:
		return fmt.Sprintf("if %s-%s != 0 { return nil, %s.FloatErr(%s, float64(%s), 64) }; %s = %s.AppendFloat(nil, float64(%s), 64)", val, val, jsonAlias, val, val, b, jsonAlias, val)
	case scalarString:
		return fmt.Sprintf("%s = append(%s.AppendEscaped(append([]byte(nil), '\"'), string(%s)), '\"')", b, jsonAlias, val)
	}
	return ""
}

func unionMemberStaticCall(m types2.Type, pkg *types2.Package, b, e string) string {
	if ptr, ok := types2.Unalias(m).(*types2.Pointer); ok {
		obj, _, indirect := types2.LookupFieldOrMethod(ptr, false, pkg, "MarshalJSON")
		if obj != nil && !indirect && isMarshalJSONSignature(obj) {
			return fmt.Sprintf("\tif _logos_m != nil { %s, %s = _logos_m.MarshalJSON() } else { %s = []byte(\"null\") }", b, e, b)
		}
		elemObj, _, elemInd := types2.LookupFieldOrMethod(ptr.Elem(), false, pkg, "MarshalJSON")
		if elemObj != nil && !elemInd && isMarshalJSONSignature(elemObj) {
			return fmt.Sprintf("\tif _logos_m != nil { %s, %s = _logos_m.MarshalJSON() } else { %s = []byte(\"null\") }", b, e, b)
		}
		if sk, ok := scalarOf(ptr.Elem()); ok {
			return fmt.Sprintf("\tif _logos_m == nil { %s = []byte(\"null\") } else { %s }", b, scalarStaticMarshal(sk, "*_logos_m", b))
		}
		return fmt.Sprintf("\t%s, %s = %s.MarshalField(_logos_m)", b, e, jsonAlias)
	}

	obj, _, indirect := types2.LookupFieldOrMethod(m, false, pkg, "MarshalJSON")
	if obj != nil && !indirect && isMarshalJSONSignature(obj) {
		return fmt.Sprintf("\t%s, %s = _logos_m.MarshalJSON()", b, e)
	}
	if obj != nil && indirect && isMarshalJSONSignature(obj) {
		return fmt.Sprintf("\t_logos_tmp := _logos_m; %s, %s = _logos_tmp.MarshalJSON()", b, e)
	}
	if sk, ok := scalarOf(m); ok {
		return "\t" + scalarStaticMarshal(sk, "_logos_m", b)
	}
	return fmt.Sprintf("\t%s, %s = %s.MarshalField(_logos_m)", b, e, jsonAlias)
}

func (g *marshalGen) unionCode(f *jsonField) fieldCode {
	x := f.local()
	b := fmt.Sprintf("_logos_b%d", f.pos)
	e := fmt.Sprintf("_logos_e%d", f.pos)
	pins := []string{x + " := " + g.recv + "." + f.goName}

	members, ok := types2.LogosUnionMembers(f.typ)
	if !ok {
		base.Fatalf("logos: union field %s has no union terms", f.goName)
	}
	pkg := unionPackage(f.typ)

	sw := []string{"switch _logos_m := " + x + ".(type) {"}
	for _, m := range members {
		if m.Type == nil {
			continue
		}
		sw = append(sw, "case "+sourceTypeString(m.Type, g.js.qualify, g.js.holderPkg, g.js.cgoSource)+":", unionMemberStaticCall(m.Type, pkg, b, e))
	}
	sw = append(sw, "case nil:")
	sw = append(sw, fmt.Sprintf("\t%s = []byte(\"null\")", b))
	sw = append(sw, "default:")
	sw = append(sw, fmt.Sprintf("\t%s, %s = %s.MarshalField(_logos_m)", b, e, jsonAlias))
	sw = append(sw, "}")

	check := ifLines(e+" != nil", []string{"return nil, " + e})

	var c fieldCode
	c.pins = pins
	if !f.optional() {
		c.pre = append([]string{"var " + b + " []byte", "var " + e + " error"}, sw...)
		c.pre = append(c.pre, check...)
		c.size = []string{fmt.Sprintf("_logos_n += %d + len(%s)", g.litLen(f, ""), b)}
		c.emit = append(g.litStmts(f, ""), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b))
		return c
	}

	p := f.presenceName()
	c.pre = []string{p + " := " + g.presence(f), "var " + b + " []byte"}
	inner := append([]string{fmt.Sprintf("var %s error", e)}, sw...)
	inner = append(inner, check...)
	c.pre = append(c.pre, ifLines(p, inner)...)
	c.size = ifLines(p, []string{fmt.Sprintf("_logos_n += %d + len(%s)", g.litLen(f, ""), b)})
	c.emit = ifLines(p, append(g.litStmts(f, ""), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b)))
	return c
}

type jsonEdge int

const (
	edgeLegacy jsonEdge = iota
	edgeHelper
	edgePtrHelper
	edgeEncode
)

type jsonHelperDef struct {
	key   string
	spell string
	vname string
	pname string
	js    *jsonStruct
	edges map[*jsonField]jsonEdge
	calls map[*jsonField]string
	text  string
	made  bool
}

type jsonWalkPlan struct {
	stateful    bool
	edges       map[*jsonField]jsonEdge
	calls       map[*jsonField]string
	order       []string
	table       []*types2.Named
	tableSeen   map[string]bool
	descendSeen map[*types2.Named]bool
}

func (fl *fileLower) jsonEdgeOf(t types2.Type) jsonEdge {
	if fl.classifyJSONField(t).kind != shapeFallback {
		return edgeLegacy
	}
	core := jsonEdgeCore(fl, t)
	if p, ok := core.(*types2.Pointer); ok {
		if key, _, _ := fl.helperModelFor(p.Elem()); key != "" {
			return edgePtrHelper
		}
		return edgeEncode
	}
	if key, _, _ := fl.helperModelFor(core); key != "" {
		return edgeHelper
	}
	switch types2.Unalias(core).Underlying().(type) {
	case *types2.Slice, *types2.Array, *types2.Map, *types2.Struct, *types2.Interface:
		return edgeEncode
	}
	return edgeLegacy
}

func jsonEdgeCore(fl *fileLower, t types2.Type) types2.Type {
	u := types2.Unalias(t)
	for {
		if o := fl.asOption(u); o != nil {
			u = types2.Unalias(o.Elem)
			continue
		}
		if cv, ok := u.(*types2.ConstView); ok {
			u = types2.Unalias(cv.Elem)
			continue
		}
		return u
	}
}

func (fl *fileLower) planJSONWalk(js *jsonStruct, rootNamed *types2.Named) *jsonWalkPlan {
	plan := &jsonWalkPlan{
		edges:       make(map[*jsonField]jsonEdge),
		calls:       make(map[*jsonField]string),
		tableSeen:   make(map[string]bool),
		descendSeen: make(map[*types2.Named]bool),
	}
	for _, f := range js.fields {
		e := fl.jsonEdgeOf(f.typ)
		if e == edgeLegacy {
			continue
		}
		plan.stateful = true
		plan.edges[f] = e
		plan.calls[f] = fl.planEdgeCall(plan, f, e)
	}
	if !plan.stateful {
		return plan
	}
	fl.visitJSONTable(plan, rootNamed)
	for _, f := range js.fields {
		fl.descendJSONTable(plan, f.typ)
	}
	return plan
}

func (fl *fileLower) planEdgeCall(plan *jsonWalkPlan, f *jsonField, e jsonEdge) string {
	call := fl.helperEdgeCall(plan, f, e)
	if optionJSONPath(f.typ, make(map[types2.Type]bool)) {
		return fl.optionJSONCall(plan, f.typ, f.local())
	}
	if call != "" {
		return call
	}
	enc := &jsonEnc{fl: fl, plan: plan}
	if f.quoted {
		return enc.quotedCall(f.typ, f.local())
	}
	return enc.freshCall(f.typ, f.local())
}

func (fl *fileLower) helperEdgeCall(plan *jsonWalkPlan, f *jsonField, e jsonEdge) string {
	if e != edgeHelper && e != edgePtrHelper {
		return ""
	}
	key := fl.ensureJSONHelper(plan, f.typ)
	if key == "" || fl.pkg.jsonHelpers[key].vname == "" {
		return ""
	}
	name := fl.pkg.jsonHelpers[key].vname
	if e == edgePtrHelper {
		name = fl.pkg.jsonHelpers[key].pname
	}
	return fmt.Sprintf("%s(%s, _logos_s)", name, f.local())
}

func (fl *fileLower) ensureJSONHelper(plan *jsonWalkPlan, t types2.Type) string {
	key, spell, js := fl.helperModelFor(t)
	if key == "" {
		return ""
	}
	if fl.pkg.jsonHelpers == nil {
		fl.pkg.jsonHelpers = make(map[string]*jsonHelperDef)
	}
	if _, ok := fl.pkg.jsonHelpers[key]; ok {
		fl.visitJSONTable(plan, t)
		return key
	}
	def := &jsonHelperDef{key: key, spell: spell, js: js}
	fl.pkg.jsonHelpers[key] = def
	plan.order = append(plan.order, key)
	fl.pkg.jsonHelperSeq++
	def.vname = fmt.Sprintf("_logos_h%d", fl.pkg.jsonHelperSeq)
	fl.pkg.jsonHelperSeq++
	def.pname = fmt.Sprintf("_logos_h%d", fl.pkg.jsonHelperSeq)
	def.edges = make(map[*jsonField]jsonEdge)
	def.calls = make(map[*jsonField]string)
	for _, f := range js.fields {
		e := fl.jsonEdgeOf(f.typ)
		if e == edgeLegacy {
			continue
		}
		def.edges[f] = e
		def.calls[f] = fl.planEdgeCall(plan, f, e)
	}
	fl.visitJSONTable(plan, t)
	return key
}

func (fl *fileLower) visitJSONTable(plan *jsonWalkPlan, t types2.Type) {
	key, spell, js := fl.helperModelFor(t)
	if key == "" || spell == "" || plan.tableSeen[key] {
		return
	}
	plan.tableSeen[key] = true
	plan.table = append(plan.table, helperNamed(fl, t))
	for _, f := range js.fields {
		fl.descendJSONTable(plan, f.typ)
	}
}

func helperNamed(fl *fileLower, t types2.Type) *types2.Named {
	core := jsonEdgeCore(fl, t)
	if p, ok := core.(*types2.Pointer); ok {
		core = types2.Unalias(p.Elem())
	}
	return core.(*types2.Named)
}

func (fl *fileLower) descendJSONTable(plan *jsonWalkPlan, t types2.Type) {
	core := jsonEdgeCore(fl, t)
	if key, _, _ := fl.helperModelFor(core); key != "" {
		fl.visitJSONTable(plan, core)
		return
	}
	switch u := types2.Unalias(core).(type) {
	case *types2.Pointer:
		fl.descendJSONTable(plan, u.Elem())
	case *types2.Slice:
		fl.descendJSONTable(plan, u.Elem())
	case *types2.Array:
		fl.descendJSONTable(plan, u.Elem())
	case *types2.Map:
		fl.descendJSONTable(plan, u.Elem())
	case *types2.Named:
		fl.addPromotedJSONDispatchType(plan, u)
		fl.descendNamedJSONTable(plan, u)
	case *types2.Struct:
		fl.descendStructJSONTable(plan, u)
	}
}

func (fl *fileLower) addPromotedJSONDispatchType(plan *jsonWalkPlan, named *types2.Named) {
	if named == nil || named.Obj() == nil || named.Obj().Pkg() != fl.pkg.tpkg {
		return
	}
	method, index, _ := types2.LookupFieldOrMethod(named, true, fl.pkg.tpkg, "MarshalJSON")
	if len(index) <= 1 {
		return
	}
	fn, ok := method.(*types2.Func)
	if !ok || fn.Pkg() != fl.pkg.tpkg || fn.Pos().IsKnown() {
		return
	}
	recv := fn.Signature().Recv().Type()
	if p, ok := recv.(*types2.Pointer); ok {
		recv = p.Elem()
	}
	promoted, ok := types2.Unalias(recv).(*types2.Named)
	if !ok || promoted.Obj() == nil || promoted.Obj().Pkg() != fl.pkg.tpkg || promoted == named || fl.helperEligibleType(promoted) == nil {
		return
	}
	if _, ok := fl.jsonTypeName(named); !ok || named.TypeArgs().Len() != 0 || named.TypeParams().Len() != 0 {
		return
	}
	key := named.Obj().Pkg().Path() + "." + named.Obj().Name()
	if !plan.tableSeen[key] {
		plan.tableSeen[key] = true
		plan.table = append(plan.table, named)
	}
}

func (fl *fileLower) descendNamedJSONTable(plan *jsonWalkPlan, named *types2.Named) {
	if plan.descendSeen[named] {
		return
	}
	plan.descendSeen[named] = true
	st, ok := named.Underlying().(*types2.Struct)
	if !ok {
		fl.descendJSONTable(plan, named.Underlying())
		return
	}
	fl.descendStructJSONTable(plan, st)
}

func (fl *fileLower) descendStructJSONTable(plan *jsonWalkPlan, st *types2.Struct) {
	for i := range st.NumFields() {
		sf := st.Field(i)
		if !sf.Exported() {
			continue
		}
		fl.descendJSONTable(plan, sf.Type())
	}
}

func (fl *fileLower) helperModelFor(t types2.Type) (string, string, *jsonStruct) {
	core := jsonEdgeCore(fl, t)
	if p, ok := core.(*types2.Pointer); ok {
		core = types2.Unalias(p.Elem())
	}
	if named := fl.helperEligibleType(core); named != nil {
		key := named.Obj().Pkg().Path() + "." + named.Obj().Name()
		spell, ok := fl.jsonTypeName(named)
		if !ok {
			spell = ""
		}
		return key, spell, fl.modelNamedStruct(named)
	}
	return fl.instHelperModel(core)
}

func (fl *fileLower) modelNamedStruct(named *types2.Named) *jsonStruct {
	st := named.Underlying().(*types2.Struct)
	vars := make([]*types2.Var, st.NumFields())
	for i := range vars {
		vars[i] = st.Field(i)
	}
	return fl.buildJSONStruct(named.Obj().Name(), st, vars)
}

type jsonInstType struct {
	origin *types2.Named
	args   []types2.Type
	spell  string
}

func (fl *fileLower) instHelperModel(core types2.Type) (string, string, *jsonStruct) {
	inst, ok := fl.spellableInstType(core)
	if !ok {
		return "", "", nil
	}
	st, ok := inst.origin.Underlying().(*types2.Struct)
	if !ok {
		return "", "", nil
	}
	for i := range st.NumFields() {
		if st.Field(i).Embedded() {
			return "", "", nil
		}
	}
	if hasOriginJSONMethods(inst.origin) {
		return "", "", nil
	}
	params := inst.origin.TypeParams()
	js := &jsonStruct{name: inst.spell, staticJSON: fl.pkg.info.LogosStaticJSON, holderPkg: fl.pkg.tpkg, qualify: fl.genQualifier, cgoSource: fl.cgoSource}
	for i := range st.NumFields() {
		sf := st.Field(i)
		ft, ok := substJSONType(sf.Type(), params, inst.args)
		if !ok {
			ft = sf.Type()
		}
		info, ok := types2.LogosJSONFieldOf(st, i)
		if !ok {
			continue
		}
		f := &jsonField{
			goName:    sf.Name(),
			key:       info.Name,
			omitEmpty: info.OmitEmpty,
			omitZero:  info.OmitZero,
			pos:       len(js.fields),
			required:  -1,
			typ:       ft,
		}
		f.shape = fl.classifyJSONField(ft)
		f.quoted = info.StringOpt && isQuotableForString(ft, f.shape)
		if f.shape.nonNil {
			f.required = len(js.required)
			js.required = append(js.required, f.goName)
		}
		js.fields = append(js.fields, f)
	}
	return "inst:" + inst.spell, inst.spell, js
}

func hasOriginJSONMethods(origin *types2.Named) bool {
	for i := range origin.NumMethods() {
		m := origin.Method(i)
		for _, name := range jsonMethodNames {
			if m.Name() == name {
				return true
			}
		}
	}
	return false
}

func (fl *fileLower) spellableInstType(core types2.Type) (jsonInstType, bool) {
	named, ok := core.(*types2.Named)
	if !ok || named.TypeArgs().Len() == 0 {
		return jsonInstType{}, false
	}
	origin := named.Origin()
	ospell, ok := fl.jsonTypeName(origin)
	if !ok {
		return jsonInstType{}, false
	}
	targs := named.TypeArgs()
	args := make([]types2.Type, targs.Len())
	parts := make([]string, targs.Len())
	for i := range args {
		args[i] = targs.At(i)
		s, ok := fl.spellJSONType(args[i])
		if !ok {
			return jsonInstType{}, false
		}
		parts[i] = s
	}
	spell := ospell + "[" + strings.Join(parts, ", ") + "]"
	return jsonInstType{origin: origin, args: args, spell: spell}, true
}

func (fl *fileLower) spellJSONType(t types2.Type) (string, bool) {
	switch u := types2.Unalias(t).(type) {
	case *types2.Basic:
		return u.Name(), true
	case *types2.Named:
		if u.TypeArgs().Len() != 0 {
			inst, ok := fl.spellableInstType(u)
			if !ok {
				return "", false
			}
			return inst.spell, true
		}
		return fl.jsonTypeName(u)
	case *types2.Pointer:
		s, ok := fl.spellJSONType(u.Elem())
		if !ok {
			return "", false
		}
		return "*" + s, true
	case *types2.Slice:
		s, ok := fl.spellJSONType(u.Elem())
		if !ok {
			return "", false
		}
		return "[]" + s, true
	case *types2.Array:
		s, ok := fl.spellJSONType(u.Elem())
		if !ok {
			return "", false
		}
		return fmt.Sprintf("[%d]%s", u.Len(), s), true
	case *types2.Map:
		k, ok := fl.spellJSONType(u.Key())
		if !ok {
			return "", false
		}
		v, ok := fl.spellJSONType(u.Elem())
		if !ok {
			return "", false
		}
		return "map[" + k + "]" + v, true
	}
	return "", false
}

func substJSONType(t types2.Type, params *types2.TypeParamList, args []types2.Type) (types2.Type, bool) {
	switch u := t.(type) {
	case *types2.TypeParam:
		for i := range params.Len() {
			if params.At(i) == u {
				return args[i], true
			}
		}
		return nil, false
	case *types2.Pointer:
		e, ok := substJSONType(u.Elem(), params, args)
		if !ok {
			return nil, false
		}
		return types2.NewPointer(e), true
	case *types2.Slice:
		e, ok := substJSONType(u.Elem(), params, args)
		if !ok {
			return nil, false
		}
		return types2.NewSlice(e), true
	case *types2.Array:
		e, ok := substJSONType(u.Elem(), params, args)
		if !ok {
			return nil, false
		}
		return types2.NewArray(e, u.Len()), true
	case *types2.Map:
		k, ok := substJSONType(u.Key(), params, args)
		if !ok {
			return nil, false
		}
		v, ok := substJSONType(u.Elem(), params, args)
		if !ok {
			return nil, false
		}
		return types2.NewMap(k, v), true
	case *types2.Named:
		if u.TypeArgs().Len() == 0 {
			return t, true
		}
		return nil, false
	case *types2.Basic:
		return t, true
	case *types2.Interface:
		return t, true
	}
	return nil, false
}

func (js *jsonStruct) marshalText(b *strings.Builder) {
	g := newMarshalGen(js)
	var pins, pre, size, emit []string
	fixed := 1
	for _, f := range js.fields {
		c := g.code(f)
		pins = append(pins, c.pins...)
		pre = append(pre, c.pre...)
		size = append(size, c.size...)
		emit = append(emit, c.emit...)
		fixed += c.fixed
	}
	var body []string
	body = append(body, pins...)
	body = append(body, pre...)
	body = append(body, fmt.Sprintf("_logos_n := %d", fixed))
	body = append(body, size...)
	if g.firstFixed < 0 {
		body = append(body, ifLines("_logos_n == 1", []string{"_logos_n = 2"})...)
	}
	body = append(body, "_logos_buf := make([]byte, 0, _logos_n)")
	if g.runtimeSep {
		body = append(body, "_logos_sep := byte('{')")
	}
	body = append(body, emit...)
	if g.firstFixed < 0 {
		body = append(body, ifLines("_logos_sep == '{'", []string{"_logos_buf = append(_logos_buf, '{')"})...)
	}
	body = append(body, "_logos_buf = append(_logos_buf, '}')", "return _logos_buf, nil")
	fmt.Fprintf(b, "\nfunc (%s %s) MarshalJSON() ([]byte, error) {\n", jsonRecv, js.name)
	for _, l := range indentLines(body) {
		b.WriteString(l)
		b.WriteByte('\n')
	}
	b.WriteString("}\n")
}

func optionJSONPath(t types2.Type, seen map[types2.Type]bool) bool {
	wrapped := false
	for {
		t = unaliasConstView(t)
		if seen[t] {
			return false
		}
		seen[t] = true
		if _, ok := types2.LogosUnionMembers(t); ok {
			return wrapped
		}
		switch u := t.(type) {
		case *types2.Option:
			wrapped = wrapped || !types2.LogosOptionIsBare(u.Elem)
			t = u.Elem
		case *types2.Pointer:
			t = u.Elem()
		case *types2.Named:
			if obj, _, _ := types2.LookupFieldOrMethod(u, true, u.Obj().Pkg(), "MarshalJSON"); isMarshalJSONSignature(obj) {
				return false
			}
			t = u.Underlying()
		case *types2.Slice:
			t = u.Elem()
		case *types2.Array:
			t = u.Elem()
		case *types2.Map:
			key, ok := types2.Unalias(u.Key()).Underlying().(*types2.Basic)
			if !ok || key.Kind() != types2.String {
				return false
			}
			t = u.Elem()
		default:
			return false
		}
	}
}

func (fl *fileLower) optionJSONCall(plan *jsonWalkPlan, t types2.Type, x string) string {
	if !optionJSONPath(t, make(map[types2.Type]bool)) {
		return (&jsonEnc{fl: fl, plan: plan}).freshCall(t, x)
	}
	u := unaliasConstView(t)
	if n, ok := u.(*types2.Named); ok {
		u = n.Underlying()
	}
	var body []string
	switch v := u.(type) {
	case *types2.Option:
		if types2.LogosOptionIsBare(v.Elem) {
			return fl.optionJSONCall(plan, v.Elem, x)
		}
		body = []string{
			"if !" + x + ".Present { return []byte(\"null\"), nil }",
			"return " + fl.optionJSONCall(plan, v.Elem, x+".Value"),
		}
	case *types2.Pointer:
		body = []string{
			"if " + x + " == nil { return []byte(\"null\"), nil }",
			"if _logos_err := " + jsonAlias + ".EnterPointer(_logos_s, " + x + "); _logos_err != nil { return nil, _logos_err }",
			"defer " + jsonAlias + ".LeavePointer(_logos_s, " + x + ")",
			"return " + fl.optionJSONCall(plan, v.Elem(), "(*"+x+")"),
		}
	default:
		return fl.optionJSONContainerCall(plan, u, x)
	}
	return "func() ([]byte, error) {\n" + strings.Join(indentLines(body), "\n") + "\n}()"
}

func optionJSONVisit(u types2.Type, x string) []string {
	kind := "Slice"
	switch u.(type) {
	case *types2.Array:
		return nil
	case *types2.Map:
		kind = "Map"
	}
	return []string{
		"if " + x + " == nil { return []byte(\"null\"), nil }",
		"if _logos_err := " + jsonAlias + ".Enter" + kind + "(_logos_s, " + x + "); _logos_err != nil { return nil, _logos_err }",
		"defer " + jsonAlias + ".Leave" + kind + "(_logos_s, " + x + ")",
	}
}

func (fl *fileLower) optionJSONContainerCall(plan *jsonWalkPlan, u types2.Type, x string) string {
	var elem types2.Type
	enter := "_logos_s.EnterElems(0, len(" + x + "))"
	mapType, isMap := u.(*types2.Map)
	switch c := u.(type) {
	case *types2.Slice:
		elem = c.Elem()
	case *types2.Array:
		elem = c.Elem()
		enter = "_logos_s.EnterFrame(0)"
	case *types2.Map:
		elem = c.Elem()
	default:
		base.Fatalf("logos: option JSON path has non-container type %s", u)
	}
	body := optionJSONVisit(u, x)
	body = append(body,
		"if _logos_err := "+enter+"; _logos_err != nil { return nil, _logos_err }",
		"defer _logos_s.LeaveFrame()")
	open, close := "'['", "']'"
	loop := "for _, _logos_item := range " + x
	var prefix []string
	if isMap {
		open, close = "'{'", "'}'"
		body = append(body, sortedJSONMapKeys(x)...)
		loop = "for _, _logos_key := range _logos_keys"
		prefix = []string{
			"_logos_buf = append(_logos_buf, '\"')",
			"_logos_buf = " + jsonAlias + ".AppendEscaped(_logos_buf, _logos_key)",
			"_logos_buf = append(_logos_buf, '\"', ':')",
			"_logos_item := " + x + "[" + fl.sourceTypeString(mapType.Key(), fl.genQualifier) + "(_logos_key)]",
		}
	}
	body = append(body, "_logos_buf := []byte{"+open+"}")
	inner := []string{"if len(_logos_buf) > 1 { _logos_buf = append(_logos_buf, ',') }"}
	inner = append(inner, prefix...)
	inner = append(inner,
		"_logos_part, _logos_err := "+fl.optionJSONCall(plan, elem, "_logos_item"),
		"if _logos_err != nil { return nil, _logos_s.RebaseDepthError(_logos_err, int64(len(_logos_buf))) }",
		"_logos_buf = append(_logos_buf, _logos_part...)")
	body = append(body, blockLines(loop, inner)...)
	body = append(body, "_logos_buf = append(_logos_buf, "+close+")", "return _logos_buf, nil")
	return "func() ([]byte, error) {\n" + strings.Join(indentLines(body), "\n") + "\n}()"
}

func sortedJSONMapKeys(x string) []string {
	return []string{
		"_logos_keys := make([]string, 0, len(" + x + "))",
		"for _logos_key := range " + x + " { _logos_keys = append(_logos_keys, string(_logos_key)) }",
		"_logos_down := func(_logos_root, _logos_end int) {",
		"for 2*_logos_root+1 < _logos_end {",
		"_logos_child := 2*_logos_root+1",
		"if _logos_child+1 < _logos_end && _logos_keys[_logos_child] < _logos_keys[_logos_child+1] { _logos_child++ }",
		"if _logos_keys[_logos_root] >= _logos_keys[_logos_child] { break }",
		"_logos_keys[_logos_root], _logos_keys[_logos_child] = _logos_keys[_logos_child], _logos_keys[_logos_root]",
		"_logos_root = _logos_child",
		"}",
		"}",
		"for _logos_i := len(_logos_keys)/2-1; _logos_i >= 0; _logos_i-- { _logos_down(_logos_i, len(_logos_keys)) }",
		"for _logos_i := len(_logos_keys)-1; _logos_i > 0; _logos_i-- { _logos_keys[0], _logos_keys[_logos_i] = _logos_keys[_logos_i], _logos_keys[0]; _logos_down(0, _logos_i) }",
	}
}

func (g *marshalGen) depthErrorBody(f *jsonField, err string) []string {
	body := []string{"var _logos_buf []byte"}
	if g.runtimeSep && len(g.errorPrefix) != 0 {
		body = append(body, "_logos_sep := byte('{')")
	}
	body = append(body, g.errorPrefix...)
	return append(body, fmt.Sprintf("return nil, _logos_s.RebaseDepthError(%s, int64(len(_logos_buf) + %d))", err, g.litLen(f, "")))
}

func (g *marshalGen) edgeCode(f *jsonField, e jsonEdge) fieldCode {
	local := f.local()
	call := g.calls[f]
	b := fmt.Sprintf("_logos_b%d", f.pos)
	ec := fmt.Sprintf("_logos_e%d", f.pos)
	stmt := []string{fmt.Sprintf("%s, %s := %s", b, ec, call)}
	check := ifLines(ec+" != nil", g.depthErrorBody(f, ec))
	var c fieldCode
	pins := []string{local + " := " + g.recv + "." + f.goName}
	guard, ok := g.edgePresence(f, e, local)
	if !ok {
		c.pins = pins
		c.pre = append(stmt, check...)
		c.size = []string{fmt.Sprintf("_logos_n += %d + len(%s)", g.litLen(f, ""), b)}
		c.emit = append(g.litStmts(f, ""), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b))
		return c
	}
	p := f.presenceName()
	c.pins = pins
	c.pre = []string{p + " := " + guard, "var " + b + " []byte"}
	inner := []string{fmt.Sprintf("var %s error", ec), fmt.Sprintf("%s, %s = %s", b, ec, call)}
	c.pre = append(c.pre, ifLines(p, append(inner, check...))...)
	c.size = ifLines(p, []string{fmt.Sprintf("_logos_n += %d + len(%s)", g.litLen(f, ""), b)})
	c.emit = ifLines(p, append(g.litStmts(f, ""), fmt.Sprintf("_logos_buf = append(_logos_buf, %s...)", b)))
	return c
}

func (g *marshalGen) edgePresence(f *jsonField, e jsonEdge, local string) (string, bool) {
	if !f.optional() {
		return "", false
	}
	if e == edgeHelper && !f.omitZero {
		return "", false
	}
	return "!(" + g.omitted(f, local) + ")", true
}

func (js *jsonStruct) marshalTextWalk(b *strings.Builder, vname, gen, genAddr string) {
	body := []string{"var _logos_s " + jsonAlias + ".Walk"}
	if gen != "" {
		body = append(body, "_logos_s.Gen = "+gen)
	}
	if genAddr != "" {
		body = append(body, "_logos_s.GenAddress = "+genAddr)
	}
	body = append(body,
		"if _logos_err := "+jsonAlias+".EnterPointer(&_logos_s, &"+jsonRecv+"); _logos_err != nil { return nil, _logos_err }",
		"_logos_b, _logos_e := "+vname+"("+jsonRecv+", &_logos_s)",
		jsonAlias+".LeavePointer(&_logos_s, &"+jsonRecv+")",
		"return _logos_b, _logos_e")
	fmt.Fprintf(b, "\nfunc (%s %s) MarshalJSON() ([]byte, error) {\n", jsonRecv, js.name)
	for _, l := range indentLines(body) {
		b.WriteString(l)
		b.WriteByte('\n')
	}
	b.WriteString("}\n")
}

func (fl *fileLower) emitJSONHelpers(pos syntax.Pos, plan *jsonWalkPlan) {
	for _, key := range plan.order {
		def := fl.pkg.jsonHelpers[key]
		if def.made || def.vname == "" {
			continue
		}
		def.made = true
		text := def.text
		if text == "" {
			var b strings.Builder
			def.emitInto(&b)
			text = b.String()
		}
		fl.addGeneratedDecls(pos, text)
	}
	fl.needJSON = true
}

func (def *jsonHelperDef) emitInto(b *strings.Builder) {
	g := newMarshalGen(def.js)
	g.recv = "_logos_w"
	g.edges = def.edges
	g.calls = def.calls
	prelude := []string{
		"if _logos_err := _logos_s.EnterFrame(0); _logos_err != nil { return nil, _logos_err }",
		"defer _logos_s.LeaveFrame()",
	}
	body := sequentialJSONBody(g, prelude)
	fmt.Fprintf(b, "\nfunc %s(_logos_w %s, _logos_s *%s.Walk) ([]byte, error) {\n", def.vname, def.spell, jsonAlias)
	for _, l := range indentLines(body) {
		b.WriteString(l)
		b.WriteByte('\n')
	}
	b.WriteString("}\n")
	fmt.Fprintf(b, "\nfunc %s(_logos_w *%s, _logos_s *%s.Walk) ([]byte, error) {\n", def.pname, def.spell, jsonAlias)
	inner := []string{
		"if _logos_w == nil { return []byte(\"null\"), nil }",
		"if _logos_err := " + jsonAlias + ".EnterPointer(_logos_s, _logos_w); _logos_err != nil { return nil, _logos_err }",
		"_logos_b, _logos_e := " + def.vname + "(*_logos_w, _logos_s)",
		jsonAlias + ".LeavePointer(_logos_s, _logos_w)",
		"if _logos_e != nil { return nil, _logos_e }",
		"return _logos_b, nil",
	}
	for _, l := range indentLines(inner) {
		b.WriteString(l)
		b.WriteByte('\n')
	}
	b.WriteString("}\n")
}

func sequentialJSONBody(g *marshalGen, prelude []string) []string {
	g.snapshot = true
	var pins, pre, size, emit []string
	fixed := 1
	for _, f := range g.js.fields {
		g.errorPrefix = emit
		c := g.code(f)
		pins = append(pins, c.pins...)
		pre = append(pre, c.pre...)
		size = append(size, c.size...)
		emit = append(emit, c.emit...)
		fixed += c.fixed
	}
	body := pins
	body = append(body, prelude...)
	body = append(body, pre...)
	body = append(body, fmt.Sprintf("_logos_n := %d", fixed))
	body = append(body, size...)
	if g.firstFixed < 0 {
		body = append(body, ifLines("_logos_n == 1", []string{"_logos_n = 2"})...)
	}
	body = append(body, "_logos_buf := make([]byte, 0, _logos_n)")
	if g.runtimeSep {
		body = append(body, "_logos_sep := byte('{')")
	}
	body = append(body, emit...)
	if g.firstFixed < 0 {
		body = append(body, ifLines("_logos_sep == '{'", []string{"_logos_buf = append(_logos_buf, '{')"})...)
	}
	return append(body, "_logos_buf = append(_logos_buf, '}')", "return _logos_buf, nil")
}
