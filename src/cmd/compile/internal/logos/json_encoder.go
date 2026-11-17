// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"fmt"
	"internal/buildcfg"
	"reflect"
	"slices"
	"strings"
	"unicode"

	"cmd/compile/internal/types2"
)

type jsonEnc struct {
	fl   *fileLower
	plan *jsonWalkPlan
}

type jsonEncKind int

const (
	encValue jsonEncKind = iota
	encAddr
	encQuoted
)

func jsonTypeKey(t types2.Type) string {
	var b strings.Builder
	writeJSONTypeKey(&b, t)
	return b.String()
}

func jsonPkgPath(p *types2.Package) string {
	return p.Path()
}

func writeJSONTypeKey(b *strings.Builder, t types2.Type) {
	switch c := types2.Unalias(t).(type) {
	case *types2.Option:
		b.WriteByte('(')
		writeJSONTypeKey(b, c.Elem)
		b.WriteString(")?")
	case *types2.Pointer:
		b.WriteByte('*')
		writeJSONTypeKey(b, c.Elem())
	case *types2.Slice:
		b.WriteString("[]")
		writeJSONTypeKey(b, c.Elem())
	case *types2.Array:
		fmt.Fprintf(b, "[%d]", c.Len())
		writeJSONTypeKey(b, c.Elem())
	case *types2.Map:
		b.WriteString("map[")
		writeJSONTypeKey(b, c.Key())
		b.WriteByte(']')
		writeJSONTypeKey(b, c.Elem())
	case *types2.Named:
		writeJSONNamedKey(b, c)
	default:
		b.WriteString(types2.TypeString(c, jsonPkgPath))
	}
}

func writeJSONNamedKey(b *strings.Builder, n *types2.Named) {
	args := n.TypeArgs()
	if args.Len() == 0 {
		b.WriteString(types2.TypeString(n, jsonPkgPath))
		return
	}
	b.WriteString(n.Obj().Pkg().Path() + "." + n.Obj().Name() + "[")
	for i := range args.Len() {
		if i > 0 {
			b.WriteString(", ")
		}
		writeJSONTypeKey(b, args.At(i))
	}
	b.WriteByte(']')
}

func jsonMarshalOrder() []string {
	if buildcfg.Experiment.JSONv2 {
		return []string{"MarshalJSONTo", "MarshalJSON", "AppendText", "MarshalText"}
	}
	return []string{"MarshalJSON", "MarshalText"}
}

func jsonIsNumber(t types2.Type) bool {
	named, ok := types2.Unalias(t).(*types2.Named)
	if !ok || named.Obj() == nil || named.Obj().Pkg() == nil {
		return false
	}
	return named.Obj().Name() == "Number" && named.Obj().Pkg().Path() == "encoding/json"
}

func jsonIsByteSliceType(t types2.Type) bool {
	s, ok := types2.Unalias(t).(*types2.Slice)
	if !ok {
		return false
	}
	b, ok := types2.Unalias(s.Elem()).(*types2.Basic)
	return ok && b.Kind() == types2.Uint8
}

func jsonIsErrorType(t types2.Type) bool {
	return types2.Identical(t, types2.Universe.Lookup("error").Type())
}

func jsonIsEncoderPointer(t types2.Type) bool {
	p, ok := types2.Unalias(t).(*types2.Pointer)
	if !ok {
		return false
	}
	named, ok := types2.Unalias(p.Elem()).(*types2.Named)
	if !ok || named.Obj() == nil || named.Obj().Pkg() == nil {
		return false
	}
	return named.Obj().Name() == "Encoder" && named.Obj().Pkg().Path() == "encoding/json/jsontext"
}

func tupleMatches(t *types2.Tuple, preds ...func(types2.Type) bool) bool {
	if t.Len() != len(preds) {
		return false
	}
	for i, pred := range preds {
		if !pred(t.At(i).Type()) {
			return false
		}
	}
	return true
}

func jsonMarshalSignature(name string, fn *types2.Func) bool {
	sig := fn.Signature()
	switch name {
	case "MarshalJSON", "MarshalText":
		return isMarshalJSONSignature(fn)
	case "AppendText":
		return tupleMatches(sig.Params(), jsonIsByteSliceType) && tupleMatches(sig.Results(), jsonIsByteSliceType, jsonIsErrorType)
	}
	return tupleMatches(sig.Params(), jsonIsEncoderPointer) && tupleMatches(sig.Results(), jsonIsErrorType)
}

func jsonImplements(t types2.Type, name string) *types2.Func {
	fn := jsonMethod(t, name)
	if fn == nil || !jsonMarshalSignature(name, fn) {
		return nil
	}
	return fn
}

func jsonHasMarshal(t types2.Type, name string) bool {
	fn := jsonImplements(t, name)
	return fn != nil && (types2.IsInterface(t) || fn.Pos().IsKnown())
}

func jsonValueMarshals(t types2.Type) bool {
	if jsonIsNumber(t) {
		return true
	}
	return slices.ContainsFunc(jsonMarshalOrder(), func(name string) bool { return jsonHasMarshal(t, name) })
}

func jsonAddrMarshals(t types2.Type) (found, viaPtr bool) {
	_, isPtr := t.Underlying().(*types2.Pointer)
	if jsonIsNumber(t) || isPtr || types2.IsInterface(t) {
		return jsonValueMarshals(t), false
	}
	for _, name := range jsonMarshalOrder() {
		if jsonHasMarshal(t, name) {
			return true, false
		}
		if jsonHasMarshal(types2.NewPointer(t), name) {
			return true, true
		}
	}
	return false, false
}

func jsonMapEntries(k types2.Type) (string, bool) {
	lk := jsonLowered(k)
	b, isBasic := lk.Underlying().(*types2.Basic)
	if isBasic && b.Info()&types2.IsString != 0 {
		return "StringEntries", true
	}
	if jsonImplements(lk, "MarshalText") != nil {
		return jsonTextEntries(lk), true
	}
	if !isBasic {
		return "", false
	}
	return jsonIntegerEntries(b)
}

func jsonTextEntries(lk types2.Type) string {
	if _, isPtr := types2.Unalias(lk).(*types2.Pointer); isPtr {
		return "PointerTextEntries"
	}
	return "TextEntries"
}

func jsonIntegerEntries(b *types2.Basic) (string, bool) {
	switch info := b.Info(); {
	case info&types2.IsUnsigned != 0:
		return "UintEntries", true
	case info&types2.IsInteger != 0:
		return "IntEntries", true
	}
	return "", false
}

func (e *jsonEnc) isGen(lt types2.Type) bool {
	if p, ok := types2.Unalias(lt).(*types2.Pointer); ok {
		lt = jsonLowered(p.Elem())
	}
	named, ok := types2.Unalias(lt).(*types2.Named)
	if !ok || jsonOptionOf(named) != nil {
		return false
	}
	key, _, _ := e.fl.helperModelFor(named)
	return key != ""
}

func (e *jsonEnc) methodArg(lt types2.Type, x, p string) (string, bool) {
	if e.isGen(lt) {
		return "", false
	}
	if p == "" {
		return x, jsonValueMarshals(lt)
	}
	found, viaPtr := jsonAddrMarshals(lt)
	if viaPtr {
		return p, found
	}
	return x, found
}

func (e *jsonEnc) zero() *jsonZero {
	return &jsonZero{qualify: e.fl.genQualifier, pkg: e.fl.pkg.tpkg, cgoSource: e.fl.cgoSource}
}

func (e *jsonEnc) spell(t types2.Type) (string, bool) {
	if !hoistableType(t, e.fl.pkg.tpkg) {
		return "", false
	}
	return e.fl.spellLowered(t)
}

func (e *jsonEnc) pathAccessible(path []*types2.Var) bool {
	return !slices.ContainsFunc(path, func(v *types2.Var) bool { return !v.Exported() && v.Pkg() != e.fl.pkg.tpkg })
}

func (e *jsonEnc) encodable(lt types2.Type) bool {
	if _, ok := jsonStructOf(lt); !ok {
		return true
	}
	return !slices.ContainsFunc(jsonEncodedFields(lt), func(f jsonEncField) bool { return !e.pathAccessible(f.path) })
}

func (e *jsonEnc) register(key, prefix string) *jsonHelperDef {
	if e.fl.pkg.jsonHelpers == nil {
		e.fl.pkg.jsonHelpers = make(map[string]*jsonHelperDef)
	}
	e.fl.pkg.jsonHelperSeq++
	def := &jsonHelperDef{key: key, vname: fmt.Sprintf("%s%d", prefix, e.fl.pkg.jsonHelperSeq)}
	e.fl.pkg.jsonHelpers[key] = def
	e.plan.order = append(e.plan.order, key)
	return def
}

func (e *jsonEnc) encoder(lt types2.Type, kind jsonEncKind) (string, bool) {
	key := fmt.Sprintf("enc%d:%s", kind, jsonTypeKey(lt))
	if def, ok := e.fl.pkg.jsonHelpers[key]; ok {
		return def.vname, true
	}
	if !e.encodable(lt) {
		return "", false
	}
	spell, ok := e.spell(lt)
	if !ok {
		return "", false
	}
	def := e.register(key, "_logos_j")
	def.text = e.encoderText(def.vname, spell, lt, kind)
	return def.vname, true
}

func (e *jsonEnc) encoderText(name, spell string, lt types2.Type, kind jsonEncKind) string {
	param, x, p, a := spell, "_logos_x", "", ""
	if kind == encAddr {
		param, x, p, a = "*"+spell, "(*_logos_x)", "_logos_x", "_logos_x"
	}
	var body []string
	switch kind {
	case encQuoted:
		body = e.quotedStmts(lt, x, p)
	default:
		body = e.compositeBody(lt, x, p, a)
	}
	lines := []string{fmt.Sprintf("func %s(_logos_buf []byte, _logos_x %s, _logos_s *%s.Walk) ([]byte, error) {", name, param, jsonAlias), "\tvar _logos_err error"}
	lines = append(lines, indentLines(body)...)
	lines = append(lines, "\treturn _logos_buf, _logos_err", "}")
	return "\n" + strings.Join(lines, "\n") + "\n"
}

func (e *jsonEnc) freshCall(t types2.Type, x string) string {
	return e.freshEncoderCall(jsonLowered(t), x, encValue)
}

func (e *jsonEnc) quotedCall(t types2.Type, x string) string {
	return e.freshEncoderCall(jsonLowered(t), x, encQuoted)
}

func (e *jsonEnc) freshEncoderCall(lt types2.Type, x string, kind jsonEncKind) string {
	name, ok := e.encoder(lt, kind)
	if !ok {
		return "_logos_s.AppendAny(nil, " + x + ")"
	}
	return name + "(nil, " + x + ", _logos_s)"
}

func errStmts(call string) []string {
	return ifLines("_logos_buf, _logos_err = "+call+"; _logos_err != nil", []string{"return _logos_buf, _logos_err"})
}

func checkStmts(call string) []string {
	return ifLines("_logos_err = "+call+"; _logos_err != nil", []string{"return _logos_buf, _logos_err"})
}

func nullIfNil(x string) []string {
	return ifLines(x+" == nil", []string{`return append(_logos_buf, "null"...), nil`})
}

func nullStmts() []string {
	return []string{`_logos_buf = append(_logos_buf, "null"...)`}
}

func enterFrameStmts() []string {
	return append(checkStmts("_logos_s.EnterFrame(int64(len(_logos_buf)))"), "defer _logos_s.LeaveFrame()")
}

func enterElemsStmts(x string) []string {
	return append(checkStmts("_logos_s.EnterElems(int64(len(_logos_buf)), len("+x+"))"), "defer _logos_s.LeaveFrame()")
}

func quoteLines(lines ...string) []string {
	out := append([]string{`_logos_buf = append(_logos_buf, '"')`}, lines...)
	return append(out, `_logos_buf = append(_logos_buf, '"')`)
}

func marshalFieldStmts(arg string) []string {
	return ifElseLines("_logos_mb, _logos_me := "+jsonAlias+".MarshalField("+arg+"); _logos_me != nil",
		[]string{"return _logos_buf, _logos_me"},
		[]string{"_logos_buf = append(_logos_buf, _logos_mb...)"})
}

func quotedNumberStmts(x string) []string {
	return ifElseLines("_logos_mb, _logos_me := "+jsonAlias+".MarshalField("+x+"); _logos_me != nil",
		[]string{"return _logos_buf, _logos_me"},
		quoteLines("_logos_buf = append(_logos_buf, _logos_mb...)"))
}

func jsonFloatBits(b *types2.Basic) int {
	if b.Kind() == types2.Float32 {
		return 32
	}
	return 64
}

func floatCheck(x string, bits int, ok []string) []string {
	return ifElseLines(fmt.Sprintf("_logos_fv := float64(%s); _logos_fv-_logos_fv != 0", x),
		[]string{fmt.Sprintf("return _logos_buf, %s.FloatErr(%s, _logos_fv, %d)", jsonAlias, x, bits)}, ok)
}

func appendFloatLine(bits int) string {
	return fmt.Sprintf("_logos_buf = %s.AppendFloat(_logos_buf, _logos_fv, %d)", jsonAlias, bits)
}

func basicJSONStmts(b *types2.Basic, x string) []string {
	switch info := b.Info(); {
	case info&types2.IsBoolean != 0:
		return ifElseLines(x, []string{`_logos_buf = append(_logos_buf, "true"...)`}, []string{`_logos_buf = append(_logos_buf, "false"...)`})
	case info&types2.IsUnsigned != 0:
		return []string{"_logos_buf = " + jsonAlias + ".AppendUint(_logos_buf, uint64(" + x + "))"}
	case info&types2.IsInteger != 0:
		return []string{"_logos_buf = " + jsonAlias + ".AppendInt(_logos_buf, int64(" + x + "))"}
	case info&types2.IsFloat != 0:
		return floatCheck(x, jsonFloatBits(b), []string{appendFloatLine(jsonFloatBits(b))})
	case info&types2.IsString != 0:
		return quoteLines("_logos_buf = " + jsonAlias + ".AppendEscaped(_logos_buf, string(" + x + "))")
	}
	return marshalFieldStmts(x)
}

func quotedBasicStmts(b *types2.Basic, x string) []string {
	switch info := b.Info(); {
	case info&types2.IsBoolean != 0:
		return ifElseLines(x, []string{"_logos_buf = append(_logos_buf, `\"true\"`...)"}, []string{"_logos_buf = append(_logos_buf, `\"false\"`...)"})
	case info&types2.IsUnsigned != 0:
		return quoteLines("_logos_buf = " + jsonAlias + ".AppendUint(_logos_buf, uint64(" + x + "))")
	case info&types2.IsInteger != 0:
		return quoteLines("_logos_buf = " + jsonAlias + ".AppendInt(_logos_buf, int64(" + x + "))")
	case info&types2.IsFloat != 0:
		return floatCheck(x, jsonFloatBits(b), quoteLines(appendFloatLine(jsonFloatBits(b))))
	case info&types2.IsString != 0:
		return quoteLines("_logos_buf = " + jsonAlias + ".AppendQuoted(_logos_buf, string(" + x + "))")
	}
	return nil
}

func (e *jsonEnc) valueStmts(t types2.Type, x, p string) []string {
	lt := jsonLowered(t)
	if arg, ok := e.methodArg(lt, x, p); ok {
		return marshalFieldStmts(arg)
	}
	if jsonOptionOf(lt) != nil {
		return e.callStmts(lt, x, p)
	}
	return e.kindStmts(lt, x, p)
}

func (e *jsonEnc) kindStmts(lt types2.Type, x, p string) []string {
	switch u := lt.Underlying().(type) {
	case *types2.Basic:
		return basicJSONStmts(u, x)
	case *types2.Interface:
		return errStmts("_logos_s.AppendAny(_logos_buf, " + x + ")")
	case *types2.Map:
		if _, ok := jsonMapEntries(u.Key()); !ok {
			return marshalFieldStmts(x)
		}
	case *types2.Pointer, *types2.Slice, *types2.Array, *types2.Struct:
	default:
		return marshalFieldStmts(x)
	}
	return e.callStmts(lt, x, p)
}

func (e *jsonEnc) callStmts(lt types2.Type, x, p string) []string {
	kind, arg := encValue, x
	if p != "" {
		kind, arg = encAddr, p
	}
	name, ok := e.encoder(lt, kind)
	if !ok {
		if kind == encAddr {
			return errStmts("_logos_s.AppendAddress(_logos_buf, " + arg + ")")
		}
		return errStmts("_logos_s.AppendAny(_logos_buf, " + x + ")")
	}
	return errStmts(name + "(_logos_buf, " + arg + ", _logos_s)")
}

func (e *jsonEnc) compositeBody(lt types2.Type, x, p, a string) []string {
	if _, ok := e.methodArg(lt, x, p); ok {
		return e.valueStmts(lt, x, p)
	}
	if jsonOptionOf(lt) != nil && optionJSONPath(lt, make(map[types2.Type]bool)) {
		return ifElseLines("_logos_part, _logos_err := "+e.fl.optionJSONCall(e.plan, lt, x)+"; _logos_err != nil",
			[]string{"return _logos_buf, _logos_s.RebaseDepthError(_logos_err, int64(len(_logos_buf)))"},
			[]string{"_logos_buf = append(_logos_buf, _logos_part...)"})
	}
	if _, ok := jsonStructOf(lt); ok {
		return e.structBody(lt, x, p, a)
	}
	switch u := lt.Underlying().(type) {
	case *types2.Pointer:
		return e.pointerBody(u, x)
	case *types2.Slice:
		return e.sliceBody(u, x)
	case *types2.Array:
		return e.arrayBody(u, x, p)
	case *types2.Map:
		if _, ok := jsonMapEntries(u.Key()); ok {
			return e.mapBody(lt, x)
		}
	}
	return e.valueStmts(lt, x, p)
}

func (e *jsonEnc) pointerBody(u *types2.Pointer, x string) []string {
	body := nullIfNil(x)
	body = append(body, checkStmts(jsonAlias+".EnterPointer(_logos_s, "+x+")")...)
	body = append(body, "defer "+jsonAlias+".LeavePointer(_logos_s, "+x+")")
	return append(body, e.valueStmts(u.Elem(), "(*"+x+")", x)...)
}

func jsonIsByteSlice(u *types2.Slice) bool {
	elem := jsonLowered(u.Elem())
	b, ok := elem.Underlying().(*types2.Basic)
	if !ok || b.Kind() != types2.Uint8 {
		return false
	}
	found, _ := jsonAddrMarshals(elem)
	return !found
}

func elemSeparator() string {
	return "if _logos_i > 0 { _logos_buf = append(_logos_buf, ',') }"
}

func (e *jsonEnc) sliceBody(u *types2.Slice, x string) []string {
	body := nullIfNil(x)
	if jsonIsByteSlice(u) {
		return append(body, "_logos_buf = "+jsonAlias+".AppendBase64(_logos_buf, "+x+")")
	}
	body = append(body, enterElemsStmts(x)...)
	body = append(body, checkStmts(jsonAlias+".EnterSlice(_logos_s, "+x+")")...)
	body = append(body, "defer "+jsonAlias+".LeaveSlice(_logos_s, "+x+")", "_logos_buf = append(_logos_buf, '[')")
	loop := append([]string{elemSeparator()}, e.valueStmts(u.Elem(), x+"[_logos_i]", "&"+x+"[_logos_i]")...)
	body = append(body, blockLines("for _logos_i := range "+x, loop)...)
	return append(body, "_logos_buf = append(_logos_buf, ']')")
}

func (e *jsonEnc) arrayBody(u *types2.Array, x, p string) []string {
	body := append(enterFrameStmts(), "_logos_buf = append(_logos_buf, '[')")
	ep := ""
	if p != "" {
		ep = "&" + x + "[_logos_i]"
	}
	loop := append([]string{elemSeparator()}, e.valueStmts(u.Elem(), x+"[_logos_i]", ep)...)
	body = append(body, blockLines(fmt.Sprintf("for _logos_i := range %d", u.Len()), loop)...)
	return append(body, "_logos_buf = append(_logos_buf, ']')")
}

func (e *jsonEnc) mapBody(lt types2.Type, x string) []string {
	u, _ := lt.Underlying().(*types2.Map)
	entries, _ := jsonMapEntries(u.Key())
	args := x
	if entries == "TextEntries" || entries == "PointerTextEntries" {
		args += ", " + staticTypeNameLit(lt)
	}
	body := nullIfNil(x)
	body = append(body, enterElemsStmts(x)...)
	body = append(body, checkStmts(jsonAlias+".EnterMap(_logos_s, "+x+")")...)
	body = append(body,
		"defer "+jsonAlias+".LeaveMap(_logos_s, "+x+")",
		"_logos_es, _logos_ke := "+jsonAlias+"."+entries+"("+args+")")
	body = append(body, ifLines("_logos_ke != nil", []string{"return _logos_buf, _logos_ke"})...)
	body = append(body, "_logos_buf = append(_logos_buf, '{')")
	loop := []string{
		elemSeparator(),
		`_logos_buf = append(_logos_buf, '"')`,
		"_logos_buf = " + jsonAlias + ".AppendEscaped(_logos_buf, _logos_es[_logos_i].Key)",
		`_logos_buf = append(_logos_buf, '"', ':')`,
	}
	loop = append(loop, e.valueStmts(u.Elem(), "_logos_es[_logos_i].Value", "")...)
	body = append(body, blockLines("for _logos_i := range _logos_es", loop)...)
	return append(body, "_logos_buf = append(_logos_buf, '}')")
}

func (e *jsonEnc) structBody(lt types2.Type, x, p, a string) []string {
	body := append(enterFrameStmts(), "_logos_next := byte('{')")
	for _, f := range jsonEncodedFields(lt) {
		body = append(body, e.fieldStmts(f, x, p, a)...)
	}
	body = append(body, ifLines("_logos_next == '{'", []string{"_logos_buf = append(_logos_buf, '{')"})...)
	return append(body, "_logos_buf = append(_logos_buf, '}')")
}

func jsonFieldAccess(f jsonEncField, x string) (string, []string, bool) {
	access := x
	var guards []string
	for i, v := range f.path {
		access += "." + v.Name()
		if _, isPtr := types2.Unalias(v.Type()).(*types2.Pointer); isPtr && i < len(f.path)-1 {
			guards = append(guards, access+" != nil")
		}
	}
	return access, guards, len(guards) > 0
}

func (e *jsonEnc) fieldStmts(f jsonEncField, x, p, a string) []string {
	access, guards, indirect := jsonFieldAccess(f, x)
	fp := ""
	if p != "" || indirect {
		fp = "&" + access
	}
	fa := ""
	if a != "" || indirect {
		fa = "&" + access
	}
	stmts := []string{
		"_logos_buf = append(_logos_buf, _logos_next)",
		"_logos_next = ','",
		"_logos_buf = append(_logos_buf, `" + keyLiteral(f.name) + "`...)",
	}
	if f.quoted {
		stmts = append(stmts, e.quotedStmts(f.typ, access, fp)...)
	} else {
		stmts = append(stmts, e.valueStmts(f.typ, access, fp)...)
	}
	stmts = ifLines("!("+e.zero().omitted(f.typ, access, jsonOmitOpts{empty: f.omitEmpty, zero: f.omitZero, addr: fa})+")", stmts)
	for i := len(guards) - 1; i >= 0; i-- {
		stmts = ifLines(guards[i], stmts)
	}
	return stmts
}

func (e *jsonEnc) quotedStmts(t types2.Type, x, p string) []string {
	lt := jsonLowered(t)
	if jsonIsNumber(lt) {
		return quotedNumberStmts(x)
	}
	if ptr, ok := lt.Underlying().(*types2.Pointer); ok && jsonIsNumber(jsonLowered(ptr.Elem())) {
		return ifElseLines(x+" != nil", quotedNumberStmts("(*"+x+")"), nullStmts())
	}
	if arg, ok := e.methodArg(lt, x, p); ok {
		return marshalFieldStmts(arg)
	}
	switch u := lt.Underlying().(type) {
	case *types2.Pointer:
		return ifElseLines(x+" == nil", nullStmts(), e.quotedStmts(u.Elem(), "(*"+x+")", x))
	case *types2.Basic:
		if s := quotedBasicStmts(u, x); s != nil {
			return s
		}
	}
	return e.valueStmts(lt, x, p)
}

func (fl *fileLower) jsonDispatcher(plan *jsonWalkPlan) (string, string) {
	keys := make([]string, len(plan.table))
	for i, named := range plan.table {
		keys[i] = jsonTypeKey(named)
	}
	key := "gen:" + strings.Join(keys, ",")
	addrKey := "genaddr:" + strings.Join(keys, ",")
	if def, ok := fl.pkg.jsonHelpers[key]; ok {
		if adef, ok := fl.pkg.jsonHelpers[addrKey]; ok {
			return def.vname, adef.vname
		}
	}
	e := &jsonEnc{fl: fl, plan: plan}
	var cases, addrCases []string
	for _, named := range plan.table {
		v, a := e.dispatchCases(named)
		cases = append(cases, v...)
		addrCases = append(addrCases, a...)
	}
	if len(cases) == 0 {
		return "", ""
	}
	def := e.register(key, "_logos_g")
	var lines []string
	lines = append(lines, "func "+def.vname+"(_logos_s *"+jsonAlias+".Walk, _logos_buf []byte, _logos_v any) ([]byte, bool, error) {",
		"\tswitch _logos_x := _logos_v.(type) {")
	lines = append(lines, indentLines(cases)...)
	lines = append(lines, "\t}", "\treturn _logos_buf, false, nil", "}")
	def.text = "\n" + strings.Join(lines, "\n") + "\n"
	adef := e.register(addrKey, "_logos_g")
	lines = []string{
		"func " + adef.vname + "(_logos_s *" + jsonAlias + ".Walk, _logos_buf []byte, _logos_v any) ([]byte, bool, error) {",
		"\tswitch _logos_x := _logos_v.(type) {",
	}
	lines = append(lines, indentLines(addrCases)...)
	lines = append(lines, "\t}", "\treturn _logos_buf, false, nil", "}")
	adef.text = "\n" + strings.Join(lines, "\n") + "\n"
	return def.vname, adef.vname
}

func (e *jsonEnc) dispatchCases(named *types2.Named) (values, addrs []string) {
	for _, t := range []types2.Type{named, types2.NewPointer(named)} {
		name, ok := e.encoder(t, encValue)
		if !ok {
			continue
		}
		spell, _ := e.spell(t)
		values = append(values, "case "+spell+":",
			"\t_logos_out, _logos_err := "+name+"(_logos_buf, _logos_x, _logos_s)",
			"\treturn _logos_out, true, _logos_err")
	}
	name, ok := e.encoder(named, encAddr)
	if ok {
		spell, _ := e.spell(types2.NewPointer(named))
		addrs = append(addrs, "case "+spell+":",
			"\t_logos_out, _logos_err := "+name+"(_logos_buf, _logos_x, _logos_s)",
			"\treturn _logos_out, true, _logos_err")
	}
	return values, addrs
}

type jsonEncField struct {
	name      string
	tag       bool
	index     []int
	path      []*types2.Var
	typ       types2.Type
	omitEmpty bool
	omitZero  bool
	quoted    bool
}

type jsonFieldLevel struct {
	st    *types2.Struct
	key   string
	index []int
	path  []*types2.Var
}

type jsonFieldScan struct {
	fields    []jsonEncField
	next      []jsonFieldLevel
	count     map[string]int
	nextCount map[string]int
}

func jsonEncodedFields(t types2.Type) []jsonEncField {
	st, _ := jsonStructOf(t)
	next := []jsonFieldLevel{{st: st, key: jsonTypeKey(t)}}
	visited := map[string]bool{}
	var count map[string]int
	var all []jsonEncField
	for len(next) > 0 {
		sc := &jsonFieldScan{count: count, nextCount: map[string]int{}}
		for _, lv := range next {
			if !visited[lv.key] {
				visited[lv.key] = true
				sc.scan(lv)
			}
		}
		all = append(all, sc.fields...)
		next, count = sc.next, sc.nextCount
	}
	return dominantJSONFields(all)
}

func (sc *jsonFieldScan) scan(lv jsonFieldLevel) {
	for i := range lv.st.NumFields() {
		if ft, ok := jsonFieldVisible(lv.st, i); ok {
			sc.consider(lv, i, ft)
		}
	}
}

func jsonDeref(t types2.Type) types2.Type {
	lt := jsonLowered(t)
	if p, ok := types2.Unalias(lt).(*types2.Pointer); ok {
		return jsonLowered(p.Elem())
	}
	return lt
}

func isJSONStruct(t types2.Type) bool {
	_, ok := t.Underlying().(*types2.Struct)
	return ok
}

func jsonFieldVisible(st *types2.Struct, i int) (types2.Type, bool) {
	sf := st.Field(i)
	ft := jsonDeref(sf.Type())
	if !sf.Exported() && (!sf.Embedded() || !isJSONStruct(ft)) {
		return nil, false
	}
	return ft, reflect.StructTag(st.Tag(i)).Get("json") != "-"
}

func (sc *jsonFieldScan) consider(lv jsonFieldLevel, i int, ft types2.Type) {
	sf := lv.st.Field(i)
	name, opts, _ := strings.Cut(reflect.StructTag(lv.st.Tag(i)).Get("json"), ",")
	if !isValidJSONTagName(name) {
		name = ""
	}
	index := append(slices.Clone(lv.index), i)
	path := append(slices.Clone(lv.path), sf)
	if name == "" && sf.Embedded() && isJSONStruct(ft) {
		sc.embed(jsonFieldLevel{st: ft.Underlying().(*types2.Struct), key: jsonTypeKey(ft), index: index, path: path})
		return
	}
	f := jsonEncField{
		name:      cmp.Or(name, sf.Name()),
		tag:       name != "",
		index:     index,
		path:      path,
		typ:       sf.Type(),
		omitEmpty: jsonTagHas(opts, "omitempty"),
		omitZero:  jsonTagHas(opts, "omitzero"),
		quoted:    isQuotedJSONKind(ft) && jsonTagHas(opts, "string"),
	}
	sc.fields = append(sc.fields, f)
	if sc.count[lv.key] > 1 {
		sc.fields = append(sc.fields, f)
	}
}

func (sc *jsonFieldScan) embed(lv jsonFieldLevel) {
	sc.nextCount[lv.key]++
	if sc.nextCount[lv.key] == 1 {
		sc.next = append(sc.next, lv)
	}
}

func jsonTagHas(opts, name string) bool {
	return slices.Contains(strings.Split(opts, ","), name)
}

func isQuotedJSONKind(t types2.Type) bool {
	b, ok := t.Underlying().(*types2.Basic)
	return ok && b.Info()&(types2.IsBoolean|types2.IsInteger|types2.IsFloat|types2.IsString) != 0
}

func isValidJSONTagName(s string) bool {
	for _, c := range s {
		if !strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", c) && !unicode.IsLetter(c) && !unicode.IsDigit(c) {
			return false
		}
	}
	return s != ""
}

func compareJSONFields(a, b jsonEncField) int {
	if c := strings.Compare(a.name, b.name); c != 0 {
		return c
	}
	if c := cmp.Compare(len(a.index), len(b.index)); c != 0 {
		return c
	}
	if a.tag != b.tag {
		if a.tag {
			return -1
		}
		return 1
	}
	return slices.Compare(a.index, b.index)
}

func dominantJSONFields(fields []jsonEncField) []jsonEncField {
	slices.SortFunc(fields, compareJSONFields)
	var out []jsonEncField
	for i := 0; i < len(fields); {
		n := 1
		for i+n < len(fields) && fields[i+n].name == fields[i].name {
			n++
		}
		if n == 1 || len(fields[i].index) != len(fields[i+1].index) || fields[i].tag != fields[i+1].tag {
			out = append(out, fields[i])
		}
		i += n
	}
	slices.SortFunc(out, func(a, b jsonEncField) int { return slices.Compare(a.index, b.index) })
	return out
}
