// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types2"
)

func foldJSONKey(k string) string {
	var b strings.Builder
	for i := 0; i < len(k); {
		c := k[i]
		if c < utf8.RuneSelf {
			if 'a' <= c && c <= 'z' {
				c -= 'a' - 'A'
			}
			b.WriteByte(c)
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(k[i:])
		b.WriteRune(foldJSONRune(r))
		i += n
	}
	return b.String()
}

func foldJSONRune(r rune) rune {
	for {
		r2 := unicode.SimpleFold(r)
		if r2 <= r {
			return r2
		}
		r = r2
	}
}

func seenWords(n int) int {
	return (n + 63) / 64
}

func (js *jsonStruct) unmarshalText(b *strings.Builder) {
	var body []string
	body = append(body, fmt.Sprintf("_logos_s := %s.NewScanner(_logos_data, (*%s)(nil))", jsonAlias, js.name))
	body = append(body, ifLines("!_logos_s.Object()", []string{"return _logos_s.End()"})...)
	body = append(body, "var _logos_container_saved error")
	if len(js.required) > 0 {
		body = append(body, fmt.Sprintf("var _logos_seen [%d]uint64", seenWords(len(js.required))))
		body = append(body, "var _logos_firstnulled = -1")
	}
	body = append(body, js.defaults...)
	body = append(body, js.loopLines()...)
	body = append(body, ifLines("_logos_e := _logos_s.Err(); _logos_e != nil", []string{"return _logos_e"})...)
	body = append(body, ifLines("_logos_container_saved != nil", []string{"return _logos_container_saved"})...)
	body = append(body, ifLines("_logos_v := _logos_s.Saved(); _logos_v != nil", []string{"return _logos_v"})...)
	body = append(body, js.requiredCheck()...)
	body = append(body, "return nil")
	fmt.Fprintf(b, "\nfunc (%s *%s) UnmarshalJSON(_logos_data []byte) error {\n", jsonRecv, js.name)
	for _, l := range indentLines(body) {
		b.WriteString(l)
		b.WriteByte('\n')
	}
	b.WriteString("}\n")
}

func (js *jsonStruct) loopLines() []string {
	body := []string{
		fmt.Sprintf("_logos_i := _logos_jsonkey_%s(_logos_s.Key())", js.name),
		"if _logos_i < 0 {",
		fmt.Sprintf("\t_logos_i = _logos_jsonfold_%s(_logos_s.FoldKey())", js.name),
		"}",
		"switch _logos_i {",
	}
	for _, f := range js.fields {
		body = append(body, fmt.Sprintf("case %d:", f.pos))
		body = append(body, indentLines(js.decodeLines(f))...)
	}
	body = append(body, "default:", "\t_logos_s.Skip()", "}")
	return blockLines("for _logos_s.Next()", body)
}

func (js *jsonStruct) decodeLines(f *jsonField) []string {
	if f.required < 0 {
		return js.valueLines(f)
	}
	w, b := f.required/64, f.required%64
	lines := []string{"if _logos_s.Null() {"}
	lines = append(lines, fmt.Sprintf("\t_logos_seen[%d] &^= 1 << %d", w, b))
	lines = append(lines, fmt.Sprintf("\tif _logos_firstnulled < 0 { _logos_firstnulled = %d }", f.required))
	lines = append(lines, "} else {")
	lines = append(lines, fmt.Sprintf("\t_logos_seen[%d] |= 1 << %d", w, b))
	for _, l := range js.valueLines(f) {
		lines = append(lines, "\t"+l)
	}
	return append(lines, "}")
}

func (js *jsonStruct) valueLines(f *jsonField) []string {
	field := jsonRecv + "." + f.goName
	switch f.shape.kind {
	case shapeScalar:
		return []string{fmt.Sprintf("%s(&_logos_s, &%s)", f.codec().decodeFn(), field)}
	case shapePointer:
		return f.pointerLines(field)
	case shapeSlice:
		return []string{fmt.Sprintf("%s.Slice(&_logos_s, &%s, %s)", jsonAlias, field, f.elemCodec().decodeFn())}
	case shapeUnion:
		return js.unionLines(f, field)
	case shapeFallback:
		if uc, ok := unionContainerOf(f.typ); ok {
			return js.unionContainerLines(f, uc, field)
		}
		if f.quoted {
			if isExactNumber(f.typ) {
				return []string{fmt.Sprintf("%s.FallbackQ(&_logos_s, &%s)", jsonAlias, field)}
			}
			if hasUnmarshalFrom(f.typ) {
				var fallback []string
				switch {
				case hasUnmarshalJSON(f.typ):
					fallback = js.quotedMethodCall(f, field, "QuotedJSON", "QuotedJSONPtr")
				case hasUnmarshalText(f.typ):
					fallback = js.quotedMethodCall(f, field, "QuotedText", "QuotedTextPtr")
				default:
					kind, ok := fallbackScalarKind(f.typ)
					if !ok {
						base.Fatalf("logos: ,string fallback field %s has no scalar kind", f.goName)
					}
					fallback = f.fallbackQuotedValueLines(field, kind)
				}
				return ifLines(js.quotedFromHead(f, field), fallback)
			}
			if hasUnmarshalJSON(f.typ) {
				return js.quotedMethodCall(f, field, "QuotedJSON", "QuotedJSONPtr")
			}
			if hasUnmarshalText(f.typ) {
				return js.quotedMethodCall(f, field, "QuotedText", "QuotedTextPtr")
			}
			if kind, ok := fallbackScalarKind(f.typ); ok {
				return f.fallbackQuotedValueLines(field, kind)
			}
		}
		return []string{fmt.Sprintf("_logos_s.Fallback(&%s)", field)}
	default:
		base.Fatalf("logos: unhandled shape kind %v for %s", f.shape.kind, f.goName)
		return nil
	}
}

func (js *jsonStruct) quotedMethodCall(f *jsonField, field, valueName, ptrName string) []string {
	if p, isPtr := unaliasConstView(f.typ).(*types2.Pointer); isPtr {
		elem := types2.TypeString(unaliasConstView(p.Elem()), js.qualify)
		return []string{fmt.Sprintf("%s.%s[%s](&_logos_s, &%s)", jsonAlias, ptrName, elem, field)}
	}
	return []string{fmt.Sprintf("%s.%s(&_logos_s, &%s)", jsonAlias, valueName, field)}
}

func (js *jsonStruct) quotedFromHead(f *jsonField, field string) string {
	if p, isPtr := unaliasConstView(f.typ).(*types2.Pointer); isPtr {
		elem := types2.TypeString(unaliasConstView(p.Elem()), js.qualify)
		return fmt.Sprintf("!%s.QuotedFromPtr[%s](&_logos_s, &%s)", jsonAlias, elem, field)
	}
	elem := types2.TypeString(unaliasConstView(f.typ), js.qualify)
	return fmt.Sprintf("!%s.QuotedFrom[%s](&_logos_s, &%s)", jsonAlias, elem, field)
}

func (f *jsonField) fallbackQuotedValueLines(field string, kind scalarKind) []string {
	scalar := *f
	if _, isPtr := unaliasConstView(f.typ).(*types2.Pointer); isPtr {
		scalar.shape = jsonShape{kind: shapePointer, scalar: kind, nonNil: f.shape.nonNil}
		return scalar.pointerLines(field)
	}
	scalar.shape = jsonShape{kind: shapeScalar, scalar: kind}
	return []string{fmt.Sprintf("%s(&_logos_s, &%s)", scalar.codec().decodeFn(), field)}
}

type trialMember struct {
	keyed   bool
	spell   string
	sample  string
	ptr     bool
	keyFunc string
}

type jsonUnionClass struct {
	members []trialMember
	keyed   []int
	name    string
	typ     string
	list    string
	id      string
}

func (js *jsonStruct) unionMembersOf(utyp types2.Type) (jsonUnionClass, bool) {
	members, names, ok := classifyUnionMembers(js.holderPkg, js.staticJSON, utyp, js.qualify)
	if !ok || len(members) == 0 {
		return jsonUnionClass{}, false
	}
	memberQuotes := make([]string, len(members))
	for i, c := range members {
		memberQuotes[i] = strconv.Quote(c.spell)
	}
	tid := names.display
	if js.holderPkg != nil {
		tid = js.holderPkg.Path() + "\x00" + names.display
	}
	uc := jsonUnionClass{name: names.display, typ: names.code, list: strings.Join(memberQuotes, ", "), id: tid}
	uc.members = make([]trialMember, len(members))
	for i, c := range members {
		sample, isPtr := strings.CutPrefix(c.code, "*")
		uc.members[i] = trialMember{
			keyed:   c.keyed,
			spell:   c.code,
			sample:  sample,
			ptr:     isPtr,
			keyFunc: c.keyFunc,
		}
		if c.keyed {
			uc.keyed = append(uc.keyed, i)
		}
	}
	return uc, true
}

func (js *jsonStruct) unionLines(f *jsonField, field string) []string {
	uc, ok := js.unionMembersOf(f.typ)
	if !ok {
		base.Fatalf("logos: union field %s has no union terms", f.goName)
	}
	return unionFieldWrap(f, field, js.unionTrial(uc, field, "_logos_s", ""))
}

func unionFieldWrap(f *jsonField, field string, trials []string) []string {
	if f.required < 0 {
		return ifElseLines("!_logos_s.Null()", trials, []string{field + " = nil"})
	}
	out := []string{"{"}
	out = append(out, indentLines(trials)...)
	return append(out, "}")
}

func (uc jsonUnionClass) store(kind, winner, matched string) string {
	return fmt.Sprintf("_logos_trial.Store(%s, _logos_raw, %s.UnionTrialOutcome{Kind: %s.UnionTrial%s, Winner: %s, Matched: %s})", strconv.Quote(uc.id), jsonAlias, jsonAlias, kind, winner, matched)
}

func (uc jsonUnionClass) errorExpr(matched string) string {
	return fmt.Sprintf("\t%s.UnionTrialError{Union: %q, Members: []string{%s}, Matched: %s, Total: %d}", rtAlias, uc.name, uc.list, matched, len(uc.members))
}

func (uc jsonUnionClass) needsClaim() bool {
	return len(uc.keyed) > 0 && len(uc.keyed) < len(uc.members)
}

func ownsKey(s trialMember) string {
	return fmt.Sprintf("%s(_logos_scan.Key()) >= 0 || %s(_logos_scan.FoldKey()) >= 0", s.keyFunc, unionFoldFunc(s.keyFunc))
}

func (js *jsonStruct) unionTrial(uc jsonUnionClass, target, scanner, errVar string) []string {
	errRet := func(e string) []string { return unionFailure("", e) }
	var trials []string
	trials = append(trials, "_logos_raw := "+scanner+".Value()")
	trials = append(trials, "if _logos_value_err := "+scanner+".Err(); _logos_value_err != nil {")
	trials = append(trials, errRet("\t_logos_value_err")...)
	trials = append(trials, "}")
	trials = append(trials, "_logos_trial := "+jsonAlias+".EnterUnionTrial(_logos_raw)")
	trials = append(trials, "defer _logos_trial.Exit()")
	trials = append(trials, fmt.Sprintf("_logos_hit, _logos_out := _logos_trial.Lookup(%s, _logos_raw)", strconv.Quote(uc.id)))
	trials = append(trials, "var _logos_match int")
	trials = append(trials, "var _logos_winidx int")
	trials = append(trials, fmt.Sprintf("var _logos_target %s", uc.typ))
	trials = append(trials, "var _logos_custom int")
	trials = append(trials, fmt.Sprintf("var _logos_custom_target %s", uc.typ))
	if len(uc.keyed) > 0 {
		trials = append(trials, unionKeyScanLines(uc)...)
	}
	for i := range uc.members {
		trials = append(trials, unionMemberTrialLines(uc, i)...)
	}
	trials = append(trials, fmt.Sprintf("if _logos_hit && _logos_out.Kind == %s.UnionTrialAmbiguous {", jsonAlias))
	trials = append(trials, errRet(uc.errorExpr("_logos_out.Matched"))...)
	trials = append(trials, "}")
	trials = append(trials, unionResultLines(uc, target, errRet)...)
	trials = append(trials, "return nil")
	out := []string{"if _logos_trial_err := func() error {"}
	out = append(out, indentLines(trials)...)
	out = append(out, "}(); _logos_trial_err != nil {")
	out = append(out, indentLines(unionFailure(errVar, "_logos_trial_err"))...)
	return append(out, "}")
}

func unionKeyScanLines(uc jsonUnionClass) []string {
	var out, scan, loop []string
	keyed := uc.keyed
	for _, i := range keyed {
		out = append(out, fmt.Sprintf("var _logos_keys_%d int", i), fmt.Sprintf("_logos_matched_%d := false", i))
		scan = append(scan, fmt.Sprintf("_logos_matched_%d = true", i))
	}
	if uc.needsClaim() {
		out = append(out, "var _logos_claimed bool")
		var conds []string
		for _, i := range keyed {
			conds = append(conds, ownsKey(uc.members[i]))
		}
		loop = ifLines(strings.Join(conds, " || "), []string{"_logos_claimed = true"})
	}
	for _, i := range keyed {
		var others []string
		for _, j := range keyed {
			if j != i {
				others = append(others, ifLines(ownsKey(uc.members[j]), []string{fmt.Sprintf("_logos_matched_%d = false", i)})...)
			}
		}
		loop = append(loop, ifElseLines(ownsKey(uc.members[i]), []string{fmt.Sprintf("_logos_keys_%d++", i)}, others)...)
	}
	loop = append(loop, "_logos_scan.Skip()")
	scan = append(scan, blockLines("for _logos_scan.Next()", loop)...)
	body := []string{fmt.Sprintf("_logos_scan := %s.NewScanner(_logos_raw, (*%s)(nil))", jsonAlias, uc.members[keyed[0]].sample)}
	body = append(body, ifLines("_logos_scan.Object()", scan)...)
	out = append(out, "{")
	out = append(out, indentLines(body)...)
	return append(out, "}")
}

func unionMemberTrialLines(uc jsonUnionClass, i int) []string {
	s := uc.members[i]
	scratch := fmt.Sprintf("_logos_scratch%d", i)
	decl := fmt.Sprintf("var %s %s", scratch, s.spell)
	if s.ptr {
		decl = fmt.Sprintf("%s := new(%s)", scratch, s.sample)
	}
	counter, dest := "_logos_custom", "_logos_custom_target"
	if s.keyed {
		counter, dest = "_logos_match", "_logos_target"
	}
	attempt := ifLines(scratch+".UnmarshalJSON(_logos_raw) == nil", []string{
		counter + "++",
		dest + " = " + scratch,
		fmt.Sprintf("_logos_winidx = %d", i),
	})
	inner := []string{decl}
	switch {
	case s.keyed:
		inner = append(inner, ifLines(fmt.Sprintf("_logos_keys_%d == 0", i), []string{fmt.Sprintf("_logos_matched_%d = false", i)})...)
		inner = append(inner, ifLines(fmt.Sprintf("_logos_matched_%d", i), attempt)...)
	case uc.needsClaim():
		inner = append(inner, ifLines("!_logos_claimed", attempt)...)
	default:
		inner = append(inner, attempt...)
	}
	out := []string{"{"}
	out = append(out, indentLines(ifLines(fmt.Sprintf("!_logos_hit || _logos_out.Winner == %d", i), inner))...)
	return append(out, "}")
}

func unionResultLines(uc jsonUnionClass, target string, errRet func(string) []string) []string {
	out := []string{"switch {"}
	out = append(out, "case _logos_match == 1:", uc.store("Single", "_logos_winidx", "0"), "\t"+target+" = _logos_target")
	out = append(out, "case _logos_match == 0 && _logos_custom == 1:", uc.store("Single", "_logos_winidx", "0"), "\t"+target+" = _logos_custom_target")
	out = append(out, "case _logos_match == 0 && _logos_custom == 0:", uc.store("NoMatch", "-1", "0"))
	out = append(out, errRet(uc.errorExpr("0"))...)
	out = append(out, "case _logos_match == 0:", uc.store("Ambiguous", "-1", "_logos_custom"))
	out = append(out, errRet(uc.errorExpr("_logos_custom"))...)
	out = append(out, "default:", uc.store("Ambiguous", "-1", "_logos_match"))
	out = append(out, errRet(uc.errorExpr("_logos_match"))...)
	return append(out, "}")
}

func unionFailure(errVar, err string) []string {
	if errVar == "" {
		return []string{"return " + err}
	}
	return []string{errVar + " = " + err, "return"}
}

type unionContainerKind int

const (
	containerSlice unionContainerKind = iota + 1
	containerMap
	containerPointer
	containerOption
	containerArray
)

type unionContainer struct {
	kind  unionContainerKind
	typ   types2.Type
	key   types2.Type
	elem  types2.Type
	next  *unionContainer
	union types2.Type
	depth int
}

func unionContainerOf(t types2.Type) (*unionContainer, bool) {
	return unionContainerAt(t, 0, make(map[*types2.Named]bool))
}

func unionContainerAt(t types2.Type, depth int, seen map[*types2.Named]bool) (*unionContainer, bool) {
	t = unaliasConstView(t)
	under := t
	if n, ok := t.(*types2.Named); ok {
		if seen[n] || hasUnmarshalMethods(n) {
			return nil, false
		}
		seen[n] = true
		under = n.Underlying()
	}
	uc := &unionContainer{typ: t, depth: depth}
	switch c := under.(type) {
	case *types2.Option:
		uc.kind, uc.elem = containerOption, c.Elem
	case *types2.Pointer:
		uc.kind, uc.elem = containerPointer, c.Elem()
	case *types2.Slice:
		uc.kind, uc.elem = containerSlice, c.Elem()
	case *types2.Array:
		uc.kind, uc.elem = containerArray, c.Elem()
	case *types2.Map:
		kb, ok := types2.Unalias(c.Key()).Underlying().(*types2.Basic)
		if !ok || kb.Kind() != types2.String {
			return nil, false
		}
		uc.kind, uc.key, uc.elem = containerMap, c.Key(), c.Elem()
	default:
		return nil, false
	}
	if e := unaliasConstView(uc.elem); hasUnionMembers(e) {
		uc.union = e
		return uc, true
	}
	next, ok := unionContainerAt(uc.elem, depth+1, seen)
	if !ok {
		return nil, false
	}
	uc.next, uc.union = next, next.union
	return uc, true
}

func hasUnionMembers(t types2.Type) bool {
	_, ok := types2.LogosUnionMembers(t)
	return ok
}

func (js *jsonStruct) unionContainerType(t types2.Type) string {
	t = unaliasConstView(t)
	switch u := t.(type) {
	case *types2.Option:
		elem := js.unionContainerType(u.Elem)
		if types2.LogosOptionIsBare(u.Elem) {
			return elem
		}
		return rtAlias + ".Option[" + elem + "]"
	case *types2.Pointer:
		return "*" + js.unionContainerType(u.Elem())
	case *types2.Slice:
		return "[]" + js.unionContainerType(u.Elem())
	case *types2.Array:
		return fmt.Sprintf("[%d]%s", u.Len(), js.unionContainerType(u.Elem()))
	case *types2.Map:
		return "map[" + js.unionContainerType(u.Key()) + "]" + js.unionContainerType(u.Elem())
	}
	return types2.TypeString(t, js.qualify)
}

func (js *jsonStruct) unionAddr(uc *unionContainer, target string) string {
	if n, ok := uc.typ.(*types2.Named); ok {
		return "(*" + js.unionContainerType(n.Underlying()) + ")(&" + target + ")"
	}
	return "&" + target
}

func (js *jsonStruct) unionContainerLines(f *jsonField, uc *unionContainer, field string) []string {
	cls, ok := js.unionMembersOf(uc.union)
	if !ok {
		base.Fatalf("logos: union field %s has no union terms", f.goName)
	}
	return js.unionLevelLines(f, cls, uc, field, "(&_logos_s)", "")
}

func (js *jsonStruct) unionLevelLines(f *jsonField, cls jsonUnionClass, uc *unionContainer, target, scanner, errVar string) []string {
	elem := js.unionContainerType(uc.elem)
	suffix := fmt.Sprintf("%d_%d", f.pos, uc.depth)
	element := func(t, s, e string) []string {
		if uc.next == nil {
			return js.unionTrial(cls, t, s, e)
		}
		return js.unionLevelLines(f, cls, uc.next, t, s, e)
	}
	switch uc.kind {
	case containerSlice, containerArray:
		errName := "_logos_uerr_" + suffix
		tmp := "_logos_tmp_" + suffix
		dst := js.unionAddr(uc, target)
		out := []string{"var " + errName + " error"}
		if uc.kind == containerArray {
			dst = "&" + tmp
			out = append(out, "var "+tmp+" []"+elem)
		}
		callback := []string{"if _logos_es.Null() {", "\tvar _logos_zero " + elem, "\t*_logos_ee = _logos_zero", "} else {"}
		callback = append(callback, indentLines(element("(*_logos_ee)", "_logos_es", errName))...)
		callback = append(callback, "}")
		out = append(out, fmt.Sprintf("%s.Slice(%s, %s, func(_logos_es *%s.Scanner, _logos_ee *%s) {", jsonAlias, scanner, dst, jsonAlias, elem))
		out = append(out, indentLines(callback)...)
		out = append(out, "})")
		out = append(out, ifLines(errName+" != nil", unionFailure(errVar, errName))...)
		if uc.kind == containerArray {
			out = append(out, ifLines(tmp+" != nil", []string{
				fmt.Sprintf("copy(%s[:], %s)", target, tmp),
				fmt.Sprintf("if len(%s) < len(%s) { clear(%s[len(%s):]) }", tmp, target, target, tmp),
			})...)
		}
		return out
	case containerMap:
		mv, mk, mm, ms := "_logos_mv_"+suffix, "_logos_mk_"+suffix, "_logos_m_"+suffix, "_logos_ms_"+suffix
		raw, valueErr := "_logos_rawm_"+suffix, "_logos_value_err_"+suffix
		key := js.unionContainerType(uc.key)
		mapType := "map[" + key + "]" + elem
		out := []string{
			"if _logos_container_saved == nil { _logos_container_saved = " + scanner + ".Saved() }",
			raw + " := " + scanner + ".Value()",
		}
		out = append(out, ifLines(valueErr+" := "+scanner+".Err(); "+valueErr+" != nil", unionFailure(errVar, valueErr))...)
		out = append(out, fmt.Sprintf("%s := %s.NewScanner(%s, (*%s)(nil))", ms, jsonAlias, raw, mapType))
		entry := []string{
			mk + " := string(append([]byte(nil), " + ms + ".Key()...))",
			"var " + mv + " " + elem,
		}
		entry = append(entry, ifLines("!"+ms+".Null()", element(mv, "(&"+ms+")", errVar))...)
		entry = append(entry, fmt.Sprintf("%s[%s(%s)] = %s", mm, key, mk, mv))
		object := []string{mm + " := make(" + mapType + ")"}
		object = append(object, blockLines("for "+ms+".Next()", entry)...)
		object = append(object, target+" = "+mm)
		out = append(out, "if "+ms+".Null() {", "\t"+target+" = nil", "} else if "+ms+".Object() {")
		out = append(out, indentLines(object)...)
		out = append(out, "}")
		out = append(out, ifLines(valueErr+" := "+ms+".Err(); "+valueErr+" != nil", unionFailure(errVar, valueErr))...)
		return append(out, ifLines("_logos_container_saved == nil", []string{"_logos_container_saved = " + ms + ".Saved()"})...)
	case containerPointer:
		alloc := "_logos_ap_" + suffix
		out := []string{fmt.Sprintf("%s := %s.Alloc(%s)", alloc, jsonAlias, js.unionAddr(uc, target))}
		return append(out, element("(*"+alloc+")", scanner, errVar)...)
	case containerOption:
		bare := types2.LogosOptionIsBare(uc.elem)
		_, named := uc.typ.(*types2.Named)
		if bare && !named {
			return ifElseLines("!"+scanner+".Null()", element(target, scanner, errVar), []string{target + " = nil"})
		}
		value := "_logos_ov_" + suffix
		result := value
		if !bare {
			result = fmt.Sprintf("%s.Option[%s]{Value: %s, Present: true}", rtAlias, elem, value)
		}
		if named {
			result = js.unionContainerType(uc.typ) + "(" + result + ")"
		}
		body := []string{"var " + value + " " + elem}
		body = append(body, element(value, scanner, errVar)...)
		body = append(body, target+" = "+result)
		if bare {
			return ifElseLines("!"+scanner+".Null()", body, []string{target + " = nil"})
		}
		return ifLines("!"+scanner+".Null()", body)
	default:
		base.Fatalf("logos: unhandled union container for %s", f.goName)
		return nil
	}
}

func (f *jsonField) pointerLines(field string) []string {
	fill := fmt.Sprintf("%s(&_logos_s, %s.Alloc(&%s))", f.codec().decodeFn(), jsonAlias, field)
	if f.required >= 0 {
		return []string{fill}
	}
	return ifElseLines("_logos_s.Null()", []string{field + " = nil"}, []string{fill})
}

func (js *jsonStruct) requiredCheck() []string {
	n := len(js.required)
	var lines, conds []string
	for w := range seenWords(n) {
		lines = append(lines, fmt.Sprintf("const _logos_required%d = 1<<%d - 1", w, min(64, n-64*w)))
		conds = append(conds, fmt.Sprintf("_logos_seen[%[1]d]&_logos_required%[1]d != _logos_required%[1]d", w))
	}
	if n == 0 {
		return nil
	}
	quoted := make([]string, len(js.required))
	for i, name := range js.required {
		quoted[i] = strconv.Quote(name)
	}
	nulled := fmt.Sprintf("return %s.MissingFieldError{Type: %q, Field: []string{%s}[_logos_firstnulled]}", rtAlias, js.name, strings.Join(quoted, ", "))
	nulled = fmt.Sprintf("if _logos_firstnulled >= 0 && _logos_seen[_logos_firstnulled/64]>>(_logos_firstnulled%%64)&1 == 0 { %s }", nulled)
	ret := fmt.Sprintf("return %s.MissingFieldError{Type: %q, Field: _logos_jsonmissing_%s(_logos_seen)}", rtAlias, js.name, js.name)
	return append(lines, ifLines(strings.Join(conds, " || "), []string{nulled, ret})...)
}

func (js *jsonStruct) missingFunc(b *strings.Builder) {
	if len(js.required) == 0 {
		return
	}
	quoted := make([]string, len(js.required))
	for i, name := range js.required {
		quoted[i] = strconv.Quote(name)
	}
	fmt.Fprintf(b, "\nfunc _logos_jsonmissing_%s(_logos_seen [%d]uint64) string {\n", js.name, seenWords(len(js.required)))
	fmt.Fprintf(b, "\tfor _logos_i, _logos_name := range [...]string{%s} {\n", strings.Join(quoted, ", "))
	b.WriteString("\t\tif _logos_seen[_logos_i/64]>>(_logos_i%64)&1 == 0 {\n\t\t\treturn _logos_name\n\t\t}\n\t}\n\treturn \"\"\n}\n")
}

type keyEntry struct {
	key string
	idx int
}

func (js *jsonStruct) keyFuncs(b *strings.Builder) {
	var exact, folded []keyEntry
	seen := make(map[string]bool)
	for _, f := range js.fields {
		exact = append(exact, keyEntry{f.key, f.pos})
		fk := foldJSONKey(f.key)
		if !seen[fk] {
			seen[fk] = true
			folded = append(folded, keyEntry{fk, f.pos})
		}
	}
	js.writeKeyFunc(b, js.name, "jsonkey", exact)
	js.writeKeyFunc(b, js.name, "jsonfold", folded)
}

func unionFoldFunc(keyFunc string) string {
	return "_logos_jsonfold_" + strings.TrimPrefix(keyFunc, "_logos_jsonkey_")
}

func (js *jsonStruct) unionMemberKeys(b *strings.Builder, seen map[string]bool) {
	for _, f := range js.fields {
		members := js.foreignKeyedMembers(f)
		for _, c := range members {
			if !c.keyed || !c.foreign || seen[c.keyFunc] {
				continue
			}
			seen[c.keyFunc] = true
			var exact, folded []keyEntry
			foldSeen := make(map[string]bool)
			for i, k := range c.keys {
				exact = append(exact, keyEntry{k, i})
				fk := foldJSONKey(k)
				if !foldSeen[fk] {
					foldSeen[fk] = true
					folded = append(folded, keyEntry{fk, i})
				}
			}
			name := strings.TrimPrefix(c.keyFunc, "_logos_jsonkey_")
			js.writeKeyFunc(b, name, "jsonkey", exact)
			js.writeKeyFunc(b, name, "jsonfold", folded)
		}
	}
}

func (js *jsonStruct) foreignKeyedMembers(f *jsonField) []unionMemberClass {
	typ := f.typ
	if f.shape.kind != shapeUnion {
		uc, ok := unionContainerOf(f.typ)
		if !ok {
			return nil
		}
		typ = uc.union
	}
	members, _, ok := classifyUnionMembers(js.holderPkg, js.staticJSON, typ, js.qualify)
	if !ok {
		return nil
	}
	return members
}

func (js *jsonStruct) writeKeyFunc(b *strings.Builder, name, kind string, keys []keyEntry) {
	if len(keys) <= constMapSwitchMax {
		fmt.Fprintf(b, "\nfunc _logos_%s_%s(_logos_k string) int {\n\tswitch _logos_k {\n", kind, name)
		for _, e := range keys {
			fmt.Fprintf(b, "\tcase %s:\n\t\treturn %d\n", strconv.Quote(e.key), e.idx)
		}
		b.WriteString("\t}\n\treturn -1\n}\n")
		return
	}
	js.writeKeyTable(b, name, kind, keys)
}

func (js *jsonStruct) writeKeyTable(b *strings.Builder, name, kind string, keys []keyEntry) {
	var placed, shadowed []keyEntry
	taken := make(map[uint64]bool, len(keys))
	for _, e := range keys {
		h := constHashString(e.key)
		if taken[h] {
			shadowed = append(shadowed, e)
			continue
		}
		taken[h] = true
		placed = append(placed, e)
	}
	entries := make([]constMapKey, len(placed))
	for i, e := range placed {
		entries[i] = constMapKey{hash: constHashString(e.key), canon: e.key}
	}
	layout, err := buildConstMap(entries, constMapSeedBudget(js.pos))
	if err != nil {
		base.Errorf("%s: %v", js.pos.String(), err)
		base.ExitIfErrors()
	}
	table := fmt.Sprintf("_logos_%ss_%s", kind, name)
	fmt.Fprintf(b, "\n%s\nvar %s = %s.ConstTable[string, int]{Salt: %d, Buckets: []%s.ConstBucket{", rodataMarker, table, rtAlias, layout.salt, rtAlias)
	for i, bk := range layout.buckets {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(b, "{Off: %d, Size: %d, Seed: %d}", bk.off, bk.size, bk.seed)
	}
	fmt.Fprintf(b, "}, Slots: []%s.ConstSlot[string, int]{", rtAlias)
	for s, ki := range layout.slots {
		if s > 0 {
			b.WriteString(", ")
		}
		if ki < 0 {
			b.WriteString("{}")
			continue
		}
		fmt.Fprintf(b, "{Key: %s, Val: %d, Occ: true}", strconv.Quote(placed[ki].key), placed[ki].idx)
	}
	b.WriteString("}}\n")
	fmt.Fprintf(b, "func _logos_%s_%s(_logos_k string) int {\n\tif _logos_i, _logos_ok := %s.Get(%s.HashString(_logos_k), _logos_k); _logos_ok {\n\t\treturn _logos_i\n\t}\n", kind, name, table, rtAlias)
	if len(shadowed) > 0 {
		b.WriteString("\tswitch _logos_k {\n")
		for _, e := range shadowed {
			fmt.Fprintf(b, "\tcase %s:\n\t\treturn %d\n", strconv.Quote(e.key), e.idx)
		}
		b.WriteString("\t}\n")
	}
	b.WriteString("\treturn -1\n}\n")
}
