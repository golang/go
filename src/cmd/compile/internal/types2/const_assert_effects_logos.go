// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"slices"
)

type logosAssertEffects struct {
	tp, rtp []byte
}

var logosAssertZero = &logosAssertEffects{}

type logosAssertDecl struct {
	sig         *Signature
	sites       []logosAssertSite
	edges       []logosAssertEdge
	sigEdgesRun bool
	tp          []byte
	rtp         []byte
}

type logosAssertSite struct {
	pos     syntax.Pos
	operand Type
	target  Type
}

type logosAssertEdge struct {
	pos   syntax.Pos
	fn    *Func
	fnN   int
	typ   *Named
	typN  int
	targs []Type
}

type logosAssertRoot = logosAssertEdge

type logosAssertEffectsCleaner struct {
	check *Checker
}

func (c *logosAssertEffectsCleaner) cleanup() {
	c.check.logosAssertFixpoint()
}

func logosDeclParams(sig *Signature) []*TypeParam {
	if sig == nil {
		return nil
	}
	var ps []*TypeParam
	if tp := sig.TypeParams(); tp != nil {
		ps = append(ps, tp.list()...)
	}
	if rp := sig.RecvTypeParams(); rp != nil {
		ps = append(ps, rp.list()...)
	}
	return ps
}

func logosAssertMaskSet(m []byte, i int) []byte {
	for len(m) < i/8+1 {
		m = append(m, 0)
	}
	m[i/8] |= 1 << uint(i%8)
	return m
}

func logosAssertMaskGet(m []byte, i int) bool {
	return i/8 < len(m) && m[i/8]&(1<<uint(i%8)) != 0
}

func logosAssertMaskOr(dst, src []byte) []byte {
	for len(dst) < len(src) {
		dst = append(dst, 0)
	}
	for i, b := range src {
		dst[i] |= b
	}
	return dst
}

func logosAssertMaskOnes(n int) []byte {
	m := make([]byte, (n+7)/8)
	for i := range m {
		m[i] = 0xff
	}
	if r := n % 8; r != 0 {
		m[len(m)-1] &= (1 << uint(r)) - 1
	}
	return m
}

func logosAssertMaskEqual(a, b []byte) bool {
	return LogosAssertMaskHex(a) == LogosAssertMaskHex(b)
}

func logosAssertMaskEmpty(m []byte) bool {
	for _, b := range m {
		if b != 0 {
			return false
		}
	}
	return true
}

func LogosAssertMaskHex(m []byte) string {
	trim := len(m)
	for trim > 0 && m[trim-1] == 0 {
		trim--
	}
	if trim == 0 {
		return "0"
	}
	be := slices.Clone(m[:trim])
	slices.Reverse(be)
	const hexd = "0123456789abcdef"
	out := make([]byte, 0, 2*len(be))
	for _, b := range be {
		out = append(out, hexd[b>>4], hexd[b&0xf])
	}
	if out[0] == '0' {
		out = out[1:]
	}
	return string(out)
}

func LogosAssertMaskParse(s string) ([]byte, bool) {
	if s == "" {
		return nil, false
	}
	for i := range len(s) {
		c := s[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return nil, false
		}
	}
	for len(s) > 0 && s[0] == '0' {
		s = s[1:]
	}
	if s == "" {
		return nil, true
	}
	if len(s)%2 == 1 {
		s = "0" + s
	}
	m := make([]byte, len(s)/2)
	for i := range m {
		m[i] = hexNibble(s[2*i])<<4 | hexNibble(s[2*i+1])
	}
	slices.Reverse(m)
	return m, true
}

func hexNibble(c byte) byte {
	if c > '9' {
		return c - 'a' + 10
	}
	return c - '0'
}

func logosBearingIface(t Type) bool {
	u, _ := Unalias(t).Underlying().(*Interface)
	return u != nil && logosInterfaceHasReadonly(u)
}

func logosSigErasableQualified(t Type, seen map[Type]bool) bool {
	sig, _ := Unalias(t).(*Signature)
	if sig == nil {
		return false
	}
	if sig.params != nil {
		for i := range sig.params.Len() {
			if logosErasableQualified(sig.params.At(i).typ, seen) {
				return true
			}
		}
	}
	if sig.results != nil {
		for i := range sig.results.Len() {
			if logosErasableQualified(sig.results.At(i).typ, seen) {
				return true
			}
		}
	}
	return false
}

func logosErasableQualified(t Type, seen map[Type]bool) bool {
	if t == nil || !isValid(t) {
		return false
	}
	if seen == nil {
		seen = make(map[Type]bool)
	}
	if seen[t] {
		return false
	}
	seen[t] = true
	switch u := Unalias(t).(type) {
	case *Interface:
		return logosInterfaceHasReadonly(u) ||
			logosIfaceMethodsErasable(u, seen) ||
			logosEmbeddedsErasable(u, seen)
	case *Named:
		if logosTypeArgsErasable(u, seen) {
			return true
		}
		if _, ok := u.Underlying().(*Interface); ok {
			for i := range u.NumMethods() {
				if logosSigErasableQualified(u.Method(i).typ, seen) {
					return true
				}
			}
		}
		return false
	case *Slice, *Array, *Pointer, *Chan:
		return logosErasableQualified(u.(interface{ Elem() Type }).Elem(), seen)
	case *Map:
		return logosErasableQualified(u.Key(), seen) || logosErasableQualified(u.Elem(), seen)
	case *Struct:
		for i := range u.NumFields() {
			if logosErasableQualified(u.Field(i).typ, seen) {
				return true
			}
		}
	case *Tuple:
		if u != nil {
			for i := range u.Len() {
				if logosErasableQualified(u.At(i).Type(), seen) {
					return true
				}
			}
		}
	case *Option:
		return logosErasableQualified(u.Elem, seen)
	case *ReadonlyDescendants:
		return true
	case *ConstView:
		return logosErasableQualified(u.Elem, seen)
	case *Union:
		for i := range u.Len() {
			if tm := u.Term(i); tm != nil {
				if logosErasableQualified(tm.Type(), seen) {
					return true
				}
			}
		}
	case *Signature:
		return logosSigErasableQualified(u, seen)
	}
	return false
}

func logosIfaceMethodsErasable(u *Interface, seen map[Type]bool) bool {
	for _, m := range u.typeSet().methods {
		if logosSigErasableQualified(m.typ, seen) {
			return true
		}
	}
	return false
}

func logosEmbeddedsErasable(u *Interface, seen map[Type]bool) bool {
	for i := range u.NumEmbeddeds() {
		if logosErasableQualified(u.EmbeddedType(i), seen) {
			return true
		}
	}
	return false
}

func logosTypeArgsErasable(u *Named, seen map[Type]bool) bool {
	args := u.TypeArgs()
	if args == nil {
		return false
	}
	for i := range args.Len() {
		if logosErasableQualified(args.At(i), seen) {
			return true
		}
	}
	return false
}

func logosAssertManufacturable(t Type) bool {
	if t == nil {
		return false
	}
	if _, ok := Unalias(t).(*TypeParam); ok {
		return false
	}
	return logosBearingIface(t) || logosErasableQualified(t, nil)
}

func logosConcreteAssertErasable(T Type, cause *string) bool {
	if T == nil || IsInterface(T) || !logosErasableQualified(T, nil) {
		return false
	}
	if logosContainsTypeParam(T, nil) {
		return false
	}
	if cause != nil {
		*cause = "(runtime type identity erases the readonly qualifier)"
	}
	return true
}

func logosParamBits(t Type, tpars []*TypeParam, into *[]byte, seen map[Type]bool) {
	if t == nil || len(tpars) == 0 {
		return
	}
	if seen == nil {
		seen = make(map[Type]bool)
	}
	if seen[t] {
		return
	}
	seen[t] = true
	if p, ok := Unalias(t).(*TypeParam); ok {
		for i, q := range tpars {
			if p == q {
				*into = logosAssertMaskSet(*into, i)
				return
			}
		}
		return
	}
	logosParamBitsWalk(t, tpars, into, seen)
}

func logosParamBitsWalk(t Type, tpars []*TypeParam, into *[]byte, seen map[Type]bool) {
	switch u := Unalias(t).(type) {
	case *Slice, *Array, *Pointer, *Chan:
		logosParamBits(u.(interface{ Elem() Type }).Elem(), tpars, into, seen)
	case *Map:
		logosParamBits(u.Key(), tpars, into, seen)
		logosParamBits(u.Elem(), tpars, into, seen)
	case *Struct:
		for i := range u.NumFields() {
			logosParamBits(u.Field(i).typ, tpars, into, seen)
		}
	case *Tuple:
		if u != nil {
			for i := range u.Len() {
				logosParamBits(u.At(i).Type(), tpars, into, seen)
			}
		}
	case *Option:
		logosParamBits(u.Elem, tpars, into, seen)
	case *ReadonlyDescendants:
		logosParamBits(u.Elem, tpars, into, seen)
	case *ConstView:
		logosParamBits(u.Elem, tpars, into, seen)
	case *Union:
		for i := range u.Len() {
			if tm := u.Term(i); tm != nil {
				logosParamBits(tm.Type(), tpars, into, seen)
			}
		}
	case *Signature:
		logosParamBitsTuple(u.params, tpars, into, seen)
		logosParamBitsTuple(u.results, tpars, into, seen)
	case *Interface:
		for _, m := range u.typeSet().methods {
			logosParamBits(m.typ, tpars, into, seen)
		}
		for i := range u.NumEmbeddeds() {
			logosParamBits(u.EmbeddedType(i), tpars, into, seen)
		}
	case *Named:
		args := u.TypeArgs()
		if args != nil && args.Len() > 0 {
			for i := range args.Len() {
				logosParamBits(args.At(i), tpars, into, seen)
			}
			return
		}
		logosParamBits(u.Underlying(), tpars, into, seen)
	}
}

func logosParamBitsTuple(tup *Tuple, tpars []*TypeParam, into *[]byte, seen map[Type]bool) {
	if tup == nil {
		return
	}
	for i := range tup.Len() {
		logosParamBits(tup.At(i).typ, tpars, into, seen)
	}
}

func logosParamBitsOf(t Type, tpars []*TypeParam) []byte {
	var m []byte
	logosParamBits(t, tpars, &m, nil)
	return m
}

func (check *Checker) logosPushFuncSig(sig *Signature) {
	check.logosAssertEnsureCleaner()
	check.logosFuncSigs = append(check.logosFuncSigs, sig)
}

func (check *Checker) logosPopFuncSig() {
	if n := len(check.logosFuncSigs); n > 0 {
		check.logosFuncSigs = check.logosFuncSigs[:n-1]
	}
}

func (check *Checker) logosAssertEnclosing() *logosAssertDecl {
	for i := len(check.logosFuncSigs) - 1; i >= 0; i-- {
		if sig := check.logosFuncSigs[i]; sig != nil && len(logosDeclParams(sig)) > 0 {
			return check.logosAssertDeclState(sig)
		}
	}
	return nil
}

func (check *Checker) logosAssertEnsureCleaner() {
	if check.logosAssertDecls == nil {
		check.logosAssertDecls = make(map[*Signature]*logosAssertDecl)
		check.needsCleanup(&logosAssertEffectsCleaner{check: check})
	}
}

func (check *Checker) logosAssertDeclState(sig *Signature) *logosAssertDecl {
	if sig == nil {
		return nil
	}
	check.logosAssertEnsureCleaner()
	st := check.logosAssertDecls[sig]
	if st == nil {
		st = &logosAssertDecl{sig: sig}
		check.logosAssertDecls[sig] = st
	}
	return st
}

func (check *Checker) logosRecordAssertTarget(e syntax.Expr, x *operand, T Type) {
	if x == nil || !isValid(T) {
		return
	}
	st := check.logosAssertEnclosing()
	if st == nil {
		return
	}
	st.sites = append(st.sites, logosAssertSite{pos: e.Pos(), operand: x.typ(), target: T})
}

func logosBitsInto(tp, rtp *[]byte, bits []byte, nTP int) {
	for i := range len(bits) * 8 {
		if !logosAssertMaskGet(bits, i) {
			continue
		}
		if i < nTP {
			*tp = logosAssertMaskSet(*tp, i)
		} else {
			*rtp = logosAssertMaskSet(*rtp, i-nTP)
		}
	}
}

func (check *Checker) logosRecordFuncInst(pos syntax.Pos, expr syntax.Expr, typ *Signature, targs []Type) {
	if typ == nil {
		return
	}
	var fn *Func
	if ident := instantiatedIdent(expr); ident != nil && check.Uses != nil {
		if obj, _ := check.Uses[ident].(*Func); obj != nil {
			fn = obj.Origin()
			if recv := obj.Signature().Recv(); recv != nil {
				rtyp := Unalias(recv.Type())
				if ptr, ok := rtyp.(*Pointer); ok {
					rtyp = Unalias(ptr.Elem())
				}
				if named, ok := rtyp.(*Named); ok && named.TypeArgs().Len() != 0 {
					targs = slices.Concat(targs, argsSlice(named.TypeArgs()))
				}
			}
		}
	}
	if len(targs) == 0 {
		return
	}
	e := logosAssertEdge{pos: pos, fn: fn, fnN: len(targs), targs: targs}
	if enc := check.logosAssertEnclosing(); enc != nil {
		if logosTargsConcrete(targs) {
			check.logosAssertEnsureCleaner()
			check.logosAssertRoots = append(check.logosAssertRoots, e)
			return
		}
		enc.edges = append(enc.edges, e)
		return
	}
	check.logosAssertEnsureCleaner()
	check.logosAssertRoots = append(check.logosAssertRoots, e)
}

func (check *Checker) logosRecordTypeInst(enclosing *logosAssertDecl, inst Type, targs []Type, pos syntax.Pos) {
	if len(targs) == 0 || inst == nil {
		return
	}
	n := asNamed(Unalias(inst))
	if n == nil {
		return
	}
	if o := n.Origin(); o != nil {
		n = o
	}
	if n.TypeParams() == nil || n.TypeParams().Len() == 0 {
		return
	}
	e := logosAssertEdge{pos: pos, typ: n, typN: n.TypeParams().Len(), targs: targs}
	if enclosing == nil {
		return
	}
	if logosTargsConcrete(targs) {
		return
	}
	enclosing.edges = append(enclosing.edges, e)
}

func (check *Checker) logosTypeAssertMask(n *Named) []byte {
	g := check.logosTypeAggGraph(n)
	for changed := true; changed; {
		changed = false
		for node, mentions := range g.mentions {
			cand := check.logosTypeMethodsAgg(node)
			for _, m := range mentions {
				sub := g.agg[m.target]
				for j := range m.targs {
					if !logosAssertMaskGet(sub, j) {
						continue
					}
					cand = logosAssertMaskOr(cand, logosParamBitsOf(m.targs[j], namedParams(node)))
				}
			}
			if !logosAssertMaskEqual(cand, g.agg[node]) {
				g.agg[node] = logosAssertMaskOr(g.agg[node], cand)
				changed = true
			}
		}
	}
	return g.agg[n]
}

type logosTypeAggMention struct {
	target *Named
	targs  []Type
}

type logosTypeAggGraph struct {
	agg      map[*Named][]byte
	mentions map[*Named][]logosTypeAggMention
}

func (check *Checker) logosTypeAggGraph(n *Named) *logosTypeAggGraph {
	g := &logosTypeAggGraph{
		agg:      make(map[*Named][]byte),
		mentions: make(map[*Named][]logosTypeAggMention),
	}
	queue := []*Named{n}
	seenNodes := map[*Named]bool{n: true}
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		g.agg[node] = check.logosTypeMethodsAgg(node)
		var add func(Type)
		add = func(t Type) {
			ins, ok := Unalias(t).(*Named)
			if !ok {
				return
			}
			args := ins.TypeArgs()
			if args == nil || args.Len() == 0 {
				return
			}
			o := ins.Origin()
			if o == nil {
				o = ins
			}
			g.mentions[node] = append(g.mentions[node], logosTypeAggMention{target: o, targs: argsSlice(args)})
			if !seenNodes[o] {
				seenNodes[o] = true
				queue = append(queue, o)
			}
			for i := range args.Len() {
				logosWalkType(args.At(i), add, make(map[Type]bool))
			}
		}
		logosWalkType(node.Underlying(), add, make(map[Type]bool))
		for i := range node.NumMethods() {
			sig := node.Method(i).Signature()
			params, results := Type(sig.Params()), Type(sig.Results())
			if sig.RecvTypeParams().Len() != 0 {
				args := make([]Type, len(namedParams(node)))
				for j, p := range namedParams(node) {
					args[j] = p
				}
				if len(args) == sig.RecvTypeParams().Len() {
					smap := makeSubstMap(sig.RecvTypeParams().list(), args)
					params = check.subst(nopos, params, smap, nil, check.context())
					results = check.subst(nopos, results, smap, nil, check.context())
				}
			}
			logosWalkType(params, add, make(map[Type]bool))
			logosWalkType(results, add, make(map[Type]bool))
		}
	}
	return g
}

func (check *Checker) logosTypeMethodsAgg(n *Named) []byte {
	np := 0
	if tp := n.TypeParams(); tp != nil {
		np = tp.Len()
	}
	local := check.logosAssertLocalType(n)
	var agg []byte
	for i := range n.NumMethods() {
		m := n.Method(i)
		rtp := []byte{}
		if local {
			if st := check.logosAssertLocalDecl(m); st != nil {
				rtp = st.rtp
			}
		} else {
			_, r, known := check.logosAssertEffectsFor(m)
			if !known {
				r = logosAssertMaskOnes(np)
			}
			rtp = r
		}
		agg = logosAssertMaskOr(agg, rtp)
	}
	return agg
}

func namedParams(n *Named) []*TypeParam {
	if tp := n.TypeParams(); tp != nil {
		return tp.list()
	}
	return nil
}

func logosWalkType(t Type, add func(Type), seen map[Type]bool) {
	if t == nil || seen[t] {
		return
	}
	seen[t] = true
	add(t)
	switch u := Unalias(t).(type) {
	case *Slice, *Array, *Pointer, *Chan:
		logosWalkType(u.(interface{ Elem() Type }).Elem(), add, seen)
	case *Map:
		logosWalkType(u.Key(), add, seen)
		logosWalkType(u.Elem(), add, seen)
	case *Struct:
		for i := range u.NumFields() {
			logosWalkType(u.Field(i).typ, add, seen)
		}
	case *Tuple:
		if u != nil {
			for i := range u.Len() {
				logosWalkType(u.At(i).Type(), add, seen)
			}
		}
	case *Option:
		logosWalkType(u.Elem, add, seen)
	case *ReadonlyDescendants:
		logosWalkType(u.Elem, add, seen)
	case *ConstView:
		logosWalkType(u.Elem, add, seen)
	case *Union:
		for i := range u.Len() {
			if tm := u.Term(i); tm != nil {
				logosWalkType(tm.Type(), add, seen)
			}
		}
	case *Signature:
		logosSigTupleTypes(u.params, func(t Type) { logosWalkType(t, add, seen) })
		logosSigTupleTypes(u.results, func(t Type) { logosWalkType(t, add, seen) })
	case *Interface:
		for _, m := range u.typeSet().methods {
			logosWalkType(m.typ, add, seen)
		}
		for i := range u.NumEmbeddeds() {
			logosWalkType(u.EmbeddedType(i), add, seen)
		}
	case *Named:
		if args := u.TypeArgs(); args != nil {
			for i := range args.Len() {
				logosWalkType(args.At(i), add, seen)
			}
		}
	}
}

func (check *Checker) logosAssertSigEdges(st *logosAssertDecl) {
	if st.sig == nil || st.sigEdgesRun {
		return
	}
	st.sigEdgesRun = true
	tpars := logosDeclParams(st.sig)
	var edges []logosAssertEdge
	seen := make(map[Type]bool)
	add := func(t Type) { logosCollectInstances(t, tpars, &edges, seen) }
	logosSigTupleTypes(st.sig.Params(), add)
	logosSigTupleTypes(st.sig.Results(), add)
	st.edges = append(st.edges, edges...)
}

func logosSigTupleTypes(tup *Tuple, add func(Type)) {
	if tup == nil {
		return
	}
	for i := range tup.Len() {
		add(tup.At(i).typ)
	}
}

func logosCollectInstances(t Type, tpars []*TypeParam, edges *[]logosAssertEdge, seen map[Type]bool) {
	if t == nil || seen[t] {
		return
	}
	seen[t] = true
	if u, ok := Unalias(t).(*Named); ok {
		args := u.TypeArgs()
		if args != nil && args.Len() > 0 {
			mentions := false
			for i := range args.Len() {
				if len(logosParamBitsOf(args.At(i), tpars)) > 0 {
					mentions = true
				}
				logosWalkType(args.At(i), func(a Type) { logosCollectInstances(a, tpars, edges, seen) }, make(map[Type]bool))
			}
			if mentions {
				o := u.Origin()
				if o == nil {
					o = u
				}
				*edges = append(*edges, logosAssertEdge{typ: o, typN: o.TypeParams().Len(), targs: argsSlice(args)})
			}
			return
		}
		logosWalkType(u.Underlying(), func(a Type) { logosCollectInstances(a, tpars, edges, seen) }, make(map[Type]bool))
		return
	}
	logosWalkTypeChildren(t, func(c Type) { logosCollectInstances(c, tpars, edges, seen) })
}

func logosWalkTypeChildren(t Type, visit func(Type)) {
	switch u := Unalias(t).(type) {
	case *Slice, *Array, *Pointer, *Chan:
		visit(u.(interface{ Elem() Type }).Elem())
	case *Map:
		visit(u.Key())
		visit(u.Elem())
	case *Struct:
		for i := range u.NumFields() {
			visit(u.Field(i).typ)
		}
	case *Tuple:
		if u != nil {
			for i := range u.Len() {
				visit(u.At(i).Type())
			}
		}
	case *Option:
		visit(u.Elem)
	case *ReadonlyDescendants:
		visit(u.Elem)
	case *ConstView:
		visit(u.Elem)
	case *Union:
		for i := range u.Len() {
			if tm := u.Term(i); tm != nil {
				visit(tm.Type())
			}
		}
	case *Signature:
		logosSigTupleTypes(u.params, visit)
		logosSigTupleTypes(u.results, visit)
	case *Interface:
		for _, m := range u.typeSet().methods {
			visit(m.typ)
		}
		for i := range u.NumEmbeddeds() {
			visit(u.EmbeddedType(i))
		}
	}
}

func argsSlice(args *TypeList) []Type {
	out := make([]Type, args.Len())
	for i := range args.Len() {
		out[i] = args.At(i)
	}
	return out
}

func logosTargsConcrete(targs []Type) bool {
	for _, t := range targs {
		if logosContainsTypeParam(t, nil) {
			return false
		}
	}
	return true
}

func logosContainsTypeParam(t Type, seen map[Type]bool) bool {
	if t == nil {
		return false
	}
	if seen == nil {
		seen = make(map[Type]bool)
	}
	if seen[t] {
		return false
	}
	seen[t] = true
	if _, ok := Unalias(t).(*TypeParam); ok {
		return true
	}
	switch u := Unalias(t).(type) {
	case *Slice, *Array, *Pointer, *Chan:
		return logosContainsTypeParam(u.(interface{ Elem() Type }).Elem(), seen)
	case *Map:
		return logosContainsTypeParam(u.Key(), seen) || logosContainsTypeParam(u.Elem(), seen)
	case *Struct:
		for i := range u.NumFields() {
			if logosContainsTypeParam(u.Field(i).typ, seen) {
				return true
			}
		}
	case *Tuple:
		if u != nil {
			for i := range u.Len() {
				if logosContainsTypeParam(u.At(i).Type(), seen) {
					return true
				}
			}
		}
	case *Option:
		return logosContainsTypeParam(u.Elem, seen)
	case *ReadonlyDescendants:
		return logosContainsTypeParam(u.Elem, seen)
	case *ConstView:
		return logosContainsTypeParam(u.Elem, seen)
	case *Union:
		for i := range u.Len() {
			if tm := u.Term(i); tm != nil {
				if logosContainsTypeParam(tm.Type(), seen) {
					return true
				}
			}
		}
	case *Signature:
		return logosTupleHasParam(u.params, seen) || logosTupleHasParam(u.results, seen)
	case *Named:
		if args := u.TypeArgs(); args != nil {
			for i := range args.Len() {
				if logosContainsTypeParam(args.At(i), seen) {
					return true
				}
			}
			return false
		}
		return logosContainsTypeParam(u.Underlying(), seen)
	case *Interface:
		for _, m := range u.typeSet().methods {
			if logosContainsTypeParam(m.typ, seen) {
				return true
			}
		}
		for i := range u.NumEmbeddeds() {
			if logosContainsTypeParam(u.EmbeddedType(i), seen) {
				return true
			}
		}
	}
	return false
}

func logosTupleHasParam(tup *Tuple, seen map[Type]bool) bool {
	if tup == nil {
		return false
	}
	for i := range tup.Len() {
		if logosContainsTypeParam(tup.At(i).typ, seen) {
			return true
		}
	}
	return false
}

func (check *Checker) logosAssertFixpoint() {
	if check.logosAssertDecls == nil && len(check.logosAssertRoots) == 0 {
		return
	}
	check.logosAssertEnsureStates()
	for _, st := range check.logosAssertDecls {
		check.logosAssertSigEdges(st)
	}
	check.logosAssertFixMasks()
	check.logosAssertExportMasks()
}

func (check *Checker) logosAssertFinalDrain() {
	if check.logosAssertDecls != nil {
		check.logosAssertFixpoint()
	}
}

func (check *Checker) logosAssertValidateRoots() {
	roots := check.logosAssertRoots
	for i := range roots {
		if !logosTargsConcrete(roots[i].targs) {
			continue
		}
		valid := true
		for _, t := range roots[i].targs {
			if !isValid(t) {
				valid = false
				break
			}
		}
		if !valid {
			continue
		}
		check.logosValidateRoot(&roots[i], newLogosAssertMemo())
	}
	check.logosAssertRoots = nil
}

func (check *Checker) logosAssertFixMasks() {
	for changed := true; changed; {
		changed = false
		for _, st := range check.logosAssertDecls {
			ntp, nrp := check.logosDeclMask(st)
			if !logosAssertMaskEqual(ntp, st.tp) || !logosAssertMaskEqual(nrp, st.rtp) {
				st.tp, st.rtp = ntp, nrp
				changed = true
			}
		}
	}
}

func (check *Checker) logosDeclMask(st *logosAssertDecl) (tp, rtp []byte) {
	tpars := logosDeclParams(st.sig)
	nTP := 0
	if tps := st.sig.TypeParams(); tps != nil {
		nTP = tps.Len()
	}
	tp, rtp = []byte{}, []byte{}
	for _, s := range st.sites {
		logosBitsInto(&tp, &rtp, logosParamBitsOf(s.target, tpars), nTP)
	}
	for _, e := range st.edges {
		cmask := check.logosEdgeCalleeMask(&e)
		for j := range e.targs {
			if logosAssertMaskGet(cmask, j) {
				logosBitsInto(&tp, &rtp, logosParamBitsOf(e.targs[j], tpars), nTP)
			}
		}
	}
	return tp, rtp
}

func (check *Checker) logosEdgeCalleeMask(e *logosAssertEdge) []byte {
	switch {
	case e.fn != nil:
		if st2 := check.logosAssertLocalDecl(e.fn); st2 != nil {
			return logosCombinedAssertMask(st2.sig, st2.tp, st2.rtp)
		}
		if tp, rtp, known := check.logosAssertEffectsFor(e.fn); known {
			return logosCombinedAssertMask(e.fn.Origin().Signature(), tp, rtp)
		}
		return logosAssertMaskOnes(e.fnN)
	case e.typ != nil:
		return check.logosTypeAssertMask(e.typ)
	}
	return logosAssertMaskOnes(e.fnN)
}

func (check *Checker) logosAssertEnsureStates() {
	scope := check.pkg.Scope()
	for _, name := range scope.Names() {
		switch obj := scope.Lookup(name).(type) {
		case *Func:
			if sig, ok := obj.Type().(*Signature); ok && len(logosDeclParams(sig)) > 0 {
				check.logosAssertDeclState(sig)
			}
		case *TypeName:
			check.logosAssertEnsureTypeStates(obj)
		}
	}
}

func (check *Checker) logosAssertEnsureTypeStates(obj *TypeName) {
	if obj.IsAlias() {
		return
	}
	n, ok := obj.Type().(*Named)
	if !ok {
		return
	}
	for i := range n.NumMethods() {
		if sig, ok := n.Method(i).Type().(*Signature); ok && len(logosDeclParams(sig)) != 0 {
			check.logosAssertDeclState(sig)
		}
	}
}

func (check *Checker) logosAssertExportMasks() {
	scope := check.pkg.Scope()
	for _, name := range scope.Names() {
		switch obj := scope.Lookup(name).(type) {
		case *Func:
			check.logosExportFuncMask(obj)
		case *TypeName:
			if n, ok := obj.Type().(*Named); ok {
				for i := range n.NumMethods() {
					check.logosExportFuncMask(n.Method(i))
				}
			}
		}
	}
}

func (check *Checker) logosExportFuncMask(fn *Func) {
	sig, ok := fn.Type().(*Signature)
	if !ok {
		return
	}
	if st := check.logosAssertDecls[sig]; st != nil {
		fn.SetLogosAssertEffects(st.tp, st.rtp, true)
	}
}

func (check *Checker) logosAssertLocalDecl(fn *Func) *logosAssertDecl {
	if fn == nil || fn.Pkg() != check.pkg {
		return nil
	}
	if sig, ok := fn.Type().(*Signature); ok {
		return check.logosAssertDecls[sig]
	}
	return nil
}

func (check *Checker) logosAssertLocalType(n *Named) bool {
	return n != nil && n.Obj() != nil && n.Obj().Pkg() == check.pkg
}

func (check *Checker) logosAssertGapOK(V, T Type) bool {
	if !isValid(V) || !isValid(T) {
		return false
	}
	if logosContainsTypeParam(T, nil) {
		return true
	}
	if IsInterface(T) {
		m, _ := logosAssertReadonlyGap(V, T)
		if m != nil {
			return false
		}
	}
	if logosBearingIface(T) && !logosBearingIface(V) || logosErasableQualified(T, nil) {
		return false
	}
	return true
}

type logosAssertProofState uint8

const (
	logosAssertPending logosAssertProofState = iota
	logosAssertPassed
	logosAssertFailed
)

type logosAssertProof struct {
	args  []Type
	state logosAssertProofState
}

type logosAssertMemo struct {
	exact map[*logosAssertDecl][]*logosAssertProof
	types map[*Named][]*logosAssertProof
}

func newLogosAssertMemo() *logosAssertMemo {
	return &logosAssertMemo{
		exact: make(map[*logosAssertDecl][]*logosAssertProof),
		types: make(map[*Named][]*logosAssertProof),
	}
}

func logosAssertSameArgs(a, b []Type) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !Identical(a[i], b[i]) {
			return false
		}
	}
	return true
}

func logosTargsStrictlyGrow(from, to []Type) bool {
	if len(from) != len(to) || logosAssertSameArgs(from, to) {
		return false
	}
	for i := range from {
		if Identical(from[i], to[i]) {
			continue
		}
		if !logosTypeContains(to[i], from[i], nil) {
			return false
		}
	}
	return true
}

func logosTypeContains(t, sub Type, seen map[Type]bool) bool {
	if t == nil || sub == nil {
		return false
	}
	if Identical(t, sub) {
		return true
	}
	if seen == nil {
		seen = make(map[Type]bool)
	}
	if seen[t] {
		return false
	}
	seen[t] = true
	switch u := Unalias(t).(type) {
	case *Pointer:
		return logosTypeContains(u.Elem(), sub, seen)
	case *Slice:
		return logosTypeContains(u.Elem(), sub, seen)
	case *Array:
		return logosTypeContains(u.Elem(), sub, seen)
	case *Chan:
		return logosTypeContains(u.Elem(), sub, seen)
	case *Map:
		return logosTypeContains(u.Key(), sub, seen) || logosTypeContains(u.Elem(), sub, seen)
	case *Struct:
		for i := range u.NumFields() {
			if logosTypeContains(u.Field(i).Type(), sub, seen) {
				return true
			}
		}
	case *Tuple:
		if u != nil {
			for i := range u.Len() {
				if logosTypeContains(u.At(i).Type(), sub, seen) {
					return true
				}
			}
		}
	case *Signature:
		if u.Params() != nil {
			for i := range u.Params().Len() {
				if logosTypeContains(u.Params().At(i).Type(), sub, seen) {
					return true
				}
			}
		}
		if u.Results() != nil {
			for i := range u.Results().Len() {
				if logosTypeContains(u.Results().At(i).Type(), sub, seen) {
					return true
				}
			}
		}
	case *Interface:
		for i := range u.NumMethods() {
			if logosTypeContains(u.Method(i).Type(), sub, seen) {
				return true
			}
		}
		for i := range u.NumEmbeddeds() {
			if logosTypeContains(u.EmbeddedType(i), sub, seen) {
				return true
			}
		}
	case *Named:
		if args := u.TypeArgs(); args != nil {
			for i := range args.Len() {
				if logosTypeContains(args.At(i), sub, seen) {
					return true
				}
			}
		}
	case *Option:
		return logosTypeContains(u.Elem, sub, seen)
	case *ReadonlyDescendants:
		return logosTypeContains(u.Elem, sub, seen)
	case *ConstView:
		return logosTypeContains(u.Elem, sub, seen)
	case *Union:
		for i := range u.Len() {
			if tm := u.Term(i); tm != nil {
				if logosTypeContains(tm.Type(), sub, seen) {
					return true
				}
			}
		}
	}
	return false
}

func (check *Checker) logosAssertProven(st *logosAssertDecl, targs []Type, memo *logosAssertMemo) bool {
	if st == nil || (len(st.sites) == 0 && len(st.edges) == 0) {
		return true
	}
	if st.sig == nil || len(targs) != len(logosDeclParams(st.sig)) {
		return false
	}
	for _, proof := range memo.exact[st] {
		if logosAssertSameArgs(proof.args, targs) {
			return proof.state != logosAssertFailed
		}
	}
	for _, proof := range memo.exact[st] {
		if proof.state == logosAssertPending && !logosAssertSameArgs(proof.args, targs) {
			_ = check.logosMonoCyclicDecl(logosDeclParams(st.sig))
			if logosTargsStrictlyGrow(proof.args, targs) {
				mask := logosCombinedAssertMask(st.sig, st.tp, st.rtp)
				return logosAssertTargsSafe(mask, targs)
			}
		}
	}
	proof := &logosAssertProof{args: slices.Clone(targs)}
	memo.exact[st] = append(memo.exact[st], proof)
	if !check.logosSitesProven(st, targs) {
		proof.state = logosAssertFailed
		return false
	}
	for i := range st.edges {
		if !check.logosAssertEdgeProven(&st.edges[i], st, targs, memo) {
			proof.state = logosAssertFailed
			return false
		}
	}
	proof.state = logosAssertPassed
	return true
}

func (check *Checker) logosSitesProven(st *logosAssertDecl, targs []Type) bool {
	tpars := logosDeclParams(st.sig)
	if len(targs) != len(tpars) {
		return false
	}
	smap := makeSubstMap(tpars, targs)
	for _, s := range st.sites {
		V := check.subst(nopos, s.operand, smap, nil, check.context())
		T := check.subst(nopos, s.target, smap, nil, check.context())
		if !check.logosAssertGapOK(V, T) {
			return false
		}
	}
	return true
}

func (check *Checker) logosAssertEdgeProven(e *logosAssertEdge, owner *logosAssertDecl, targs []Type, memo *logosAssertMemo) bool {
	tpars := logosDeclParams(owner.sig)
	smap := makeSubstMap(tpars, targs)
	sub := make([]Type, len(e.targs))
	for i, t := range e.targs {
		sub[i] = check.subst(nopos, t, smap, nil, check.context())
	}
	switch {
	case e.fn != nil:
		return check.logosFuncTargsProven(e.fn, sub, e.fnN, memo)
	case e.typ != nil:
		return check.logosTypeTargsProven(e.typ, sub, memo)
	}
	return logosAssertTargsSafe(logosAssertMaskOnes(e.fnN), sub)
}

func (check *Checker) logosFuncTargsProven(fn *Func, targs []Type, fnN int, memo *logosAssertMemo) bool {
	if st := check.logosAssertLocalDecl(fn); st != nil {
		return check.logosAssertProven(st, targs, memo)
	}
	tp, rtp, known := check.logosAssertEffectsFor(fn)
	if !known {
		return logosAssertTargsSafe(logosAssertMaskOnes(fnN), targs)
	}
	return logosAssertTargsSafe(logosCombinedAssertMask(fn.Origin().Signature(), tp, rtp), targs)
}

func (check *Checker) logosTypeTargsProven(n *Named, targs []Type, memo *logosAssertMemo) bool {
	if !check.logosAssertLocalType(n) {
		return logosAssertTargsSafe(check.logosTypeAssertMask(n), targs)
	}
	if len(targs) != len(namedParams(n)) {
		return false
	}
	for _, proof := range memo.types[n] {
		if logosAssertSameArgs(proof.args, targs) {
			return proof.state != logosAssertFailed
		}
	}
	for _, proof := range memo.types[n] {
		if proof.state == logosAssertPending && !logosAssertSameArgs(proof.args, targs) {
			_ = check.logosMonoCyclicDecl(namedParams(n))
			_ = check.logosMonoCyclicNamed(n)
			if logosTargsStrictlyGrow(proof.args, targs) {
				return logosAssertTargsSafe(check.logosTypeAssertMask(n), targs)
			}
		}
	}
	proof := &logosAssertProof{args: slices.Clone(targs)}
	memo.types[n] = append(memo.types[n], proof)
	for i := range n.NumMethods() {
		st := check.logosAssertLocalDecl(n.Method(i))
		if st == nil {
			continue
		}
		args := make([]Type, st.sig.TypeParams().Len(), st.sig.TypeParams().Len()+len(targs))
		for j := range args {
			args[j] = st.sig.TypeParams().At(j)
		}
		args = append(args, targs...)
		if !check.logosAssertProven(st, args, memo) {
			proof.state = logosAssertFailed
			return false
		}
	}
	smap := makeSubstMap(namedParams(n), targs)
	for _, edge := range check.logosTypeAggGraph(n).mentions[n] {
		sub := make([]Type, len(edge.targs))
		for i, t := range edge.targs {
			sub[i] = check.subst(nopos, t, smap, nil, check.context())
		}
		if !check.logosTypeTargsProven(edge.target, sub, memo) {
			proof.state = logosAssertFailed
			return false
		}
	}
	proof.state = logosAssertPassed
	return true
}

func (check *Checker) logosAssertEffectsFor(fn *Func) (tp, rtp []byte, known bool) {
	tp, rtp, known = fn.LogosAssertEffects()
	if !known || fn.Pkg() == check.pkg {
		return tp, rtp, known
	}
	sig := fn.Origin().Signature()
	if !logosAssertMaskFits(tp, sig.TypeParams().Len()) || !logosAssertMaskFits(rtp, sig.RecvTypeParams().Len()) {
		return nil, nil, false
	}
	return tp, rtp, known
}

func (check *Checker) logosRecordRecvCall(pos syntax.Pos, expr syntax.Expr, fn *Func, recvTargs []Type) {
	if fn == nil || len(recvTargs) == 0 {
		return
	}
	if o := fn.Origin(); o != nil {
		fn = o
	}
	e := logosAssertEdge{pos: pos, fn: fn, fnN: len(recvTargs), targs: slices.Clone(recvTargs)}
	if enc := check.logosAssertEnclosing(); enc != nil {
		if logosTargsConcrete(recvTargs) {
			check.logosAssertEnsureCleaner()
			check.logosAssertRoots = append(check.logosAssertRoots, e)
			return
		}
		enc.edges = append(enc.edges, e)
		return
	}
	check.logosAssertEnsureCleaner()
	check.logosAssertRoots = append(check.logosAssertRoots, e)
}

func (check *Checker) logosRecordRecvMethodCall(call *syntax.CallExpr) {
	if call == nil || call.Fun == nil {
		return
	}
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok {
		return
	}
	s := check.Selections[sel]
	if s == nil {
		return
	}
	if s.Kind() != MethodVal && s.Kind() != MethodExpr {
		return
	}
	fn, ok := s.Obj().(*Func)
	if !ok || fn == nil {
		return
	}
	origin := fn.Origin()
	if origin == nil {
		origin = fn
	}
	osig, _ := origin.Type().(*Signature)
	if osig == nil {
		return
	}
	if osig.TypeParams().Len() != 0 {
		return
	}
	if osig.RecvTypeParams().Len() == 0 {
		return
	}
	recv := s.Recv()
	if recv == nil || !isValid(recv) {
		return
	}
	rt := Unalias(recv)
	if ptr, ok := rt.(*Pointer); ok {
		rt = Unalias(ptr.Elem())
	}
	named, ok := rt.(*Named)
	if !ok {
		return
	}
	args := named.TypeArgs()
	if args == nil || args.Len() == 0 {
		return
	}
	recvTargs := argsSlice(args)
	if len(recvTargs) != osig.RecvTypeParams().Len() {
		return
	}
	check.logosRecordRecvCall(call.Pos(), call.Fun, origin, recvTargs)
}

func logosAssertMaskFits(mask []byte, arity int) bool {
	full, remaining := arity/8, uint(arity%8)
	for i, b := range mask {
		if i < full {
			continue
		}
		if i == full {
			if b>>remaining != 0 {
				return false
			}
		} else if b != 0 {
			return false
		}
	}
	return true
}

func logosCombinedAssertMask(sig *Signature, tp, rtp []byte) []byte {
	if logosAssertMaskEmpty(rtp) {
		return tp
	}
	out := slices.Clone(tp)
	for i := range sig.RecvTypeParams().Len() {
		if logosAssertMaskGet(rtp, i) {
			out = logosAssertMaskSet(out, sig.TypeParams().Len()+i)
		}
	}
	return out
}

func logosAssertTargsSafe(mask []byte, targs []Type) bool {
	for j := range targs {
		if logosAssertMaskGet(mask, j) && logosAssertManufacturable(targs[j]) {
			return false
		}
	}
	return true
}

func (check *Checker) logosValidateRoot(r *logosAssertRoot, memo *logosAssertMemo) {
	switch {
	case r.fn != nil:
		check.logosValidateFuncRoot(r, memo)
	case r.typ != nil:
		return
	default:
		check.logosValidateUnknownRoot(r)
	}
}

func (check *Checker) logosValidateFuncRoot(r *logosAssertRoot, memo *logosAssertMemo) {
	if r.fn != nil && r.fn.Pkg() != nil && r.fn.Pkg().Path() == "logos/rt" && r.fn.Name() == "CovSlice" {
		return
	}
	if st := check.logosAssertLocalDecl(r.fn); st != nil {
		if st.sig == nil || len(r.targs) != len(logosDeclParams(st.sig)) {
			return
		}
		if check.logosAssertProven(st, r.targs, memo) {
			return
		}
		check.logosReportRoot(r, r.fn.Name(), "", "")
		return
	}
	tp, rtp, known := check.logosAssertEffectsFor(r.fn)
	mask := logosCombinedAssertMask(r.fn.Origin().Signature(), tp, rtp)
	if !known {
		mask = logosAssertMaskOnes(r.fnN)
	}
	for j := range r.targs {
		if logosAssertMaskGet(mask, j) && logosAssertManufacturable(r.targs[j]) {
			check.logosReportRoot(r, r.fn.Name(), "", logosUnverifiedPath(r.fn.Pkg(), known))
			return
		}
	}
}

func (check *Checker) logosValidateUnknownRoot(r *logosAssertRoot) {
	mask := logosAssertMaskOnes(r.fnN)
	for j := range r.targs {
		if logosAssertMaskGet(mask, j) && logosAssertManufacturable(r.targs[j]) {
			check.logosReportRoot(r, "", "", check.pkg.Path())
			return
		}
	}
}

func logosUnverifiedPath(pkg *Package, known bool) string {
	if known {
		return ""
	}
	if pkg == nil {
		return "?"
	}
	return pkg.Path()
}

func (check *Checker) logosReportRoot(r *logosAssertRoot, name string, method string, unverified string) {
	targ := ""
	for _, t := range r.targs {
		if logosAssertManufacturable(t) {
			targ = TypeString(t, nil)
			break
		}
	}
	if targ == "" {
		return
	}
	suffix := ""
	if unverified != "" {
		suffix = " (assertion effects of " + unverified + " unverified)"
	}
	at := noposer{pos: r.pos}
	if method != "" {
		check.errorf(at, IncompatibleAssign, "%s cannot instantiate %s with %s: method %s may manufacture a qualified %s%s", logosConstMutation, name, targ, method, targ, suffix)
		return
	}
	if name == "" {
		check.errorf(at, IncompatibleAssign, "%s cannot instantiate generic with %s: type assertion may manufacture a qualified %s%s", logosConstMutation, targ, targ, suffix)
		return
	}
	check.errorf(at, IncompatibleAssign, "%s cannot instantiate %s with %s: type assertion in %s may manufacture a qualified %s%s", logosConstMutation, name, targ, name, targ, suffix)
}
