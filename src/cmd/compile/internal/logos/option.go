// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"
	"sort"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M3 option nodes (SPEC Sec 1.4, T-104.3 and Sec
// 2.2, T-202): explicit `T?` declarations, `?.`, and `??`, plus the
// single-value map reads `m[k]` that already yielded `V?` since M3-S6.
// After this pass the backend sees no *types2.Option: every option type
// becomes either the bare payload (a non-nil `*T` stays `*T`, an
// interface-backed union with no nil-capable member stays the bare
// interface, nil as none in both) or the presence-bearing
// `_logos_rt.Option[T]` (every other T, so a present nil stays distinct
// from absent), and every `?.` and `??` becomes an immediately invoked
// function literal that tests presence once.
//
// Table N (IMPLEMENTATION.md) defines no OptSelectorExpr lowering
// entity: `?.` and `??` lower via the pexpr-shaped nodes plus generated
// func wrappers here, with no dedicated node field.
//
// The pass runs after lowerUnions and before the statement lowering, so
// `try` templates only ever copy already-lowered expressions. Every
// rewrite builds the native tree and the eject text together. Wrappers
// use boundary inserts (a head insert, mid inserts, operator deletions,
// a tail append), so nested splices strictly inside compose in input
// order; value wraps (`T` into `T?`) use one textOf replacement per
// site, which consumes the already-recorded inner splices. Only the end
// of a `??` fallback and of a wrapped value is heuristic (a scan to the
// next token that cannot continue an expression); every other boundary
// is an exact token. A composite-literal fallback whose type is not a
// composite (for example `T{...}.Field` with a scalar field type) may
// place the closing `}()` early in ejected text only; compiled output
// always comes from the native tree.
//
// Representation decisions reuse keepMapRead: a bare payload is a
// pointer or an interface-backed union with no nil-capable member.
// Nil-capable means nil is a valid payload: slice, map, func, chan, or
// interface kinds (plus type parameters, nested unions, and unsafe
// pointers, conservatively). A `nil` literal aimed at a wrapped option
// becomes the zero `Option[T]{}`; a plain `T` value becomes
// `Option[T]{Value: v, Present: true}` spelled from the checked type.
// Bare targets need no wrap: Go assignability already covers them.
//
// Narrowed reads (M3-S4) need no narrowing logic here: a read whose
// recorded type is the element of its declared option is unwrapped with
// `.Value` when wrapped (bare stays as is). The `?.` and `??` operands
// the checker un-narrowed keep the declared option, and write targets
// are excluded, so the rule keys off recorded types only. A `!= nil` or
// `== nil` test on a wrapped option becomes `v.Present` or `!v.Present`;
// bare tests stay as they are.

const (
	// optTemp is the value temporary of the lowered `?.` and `??`
	// wrappers. Each wrapper is a fresh function literal, so one
	// spelling cannot capture.
	optTemp = "v"
	// optOk is the presence temporary of the comma-ok `??` wrapper.
	optOk = "ok"
	// optWrapTemp is the per-result temporary prefix of a multi-result
	// optional call wrapper (`_logos_w0`, ...). Each wrapper is a fresh
	// function literal scope, so the indices restart every time.
	optWrapTemp = "_logos_w%d"
)

// exprType is the static type of expression slots the option pass
// rewrites: plain, listed, and fixed-array expression fields.
var exprType = reflect.TypeFor[syntax.Expr]()

// lowerOptions rewrites every option form of one file: option types in
// every type position, `?.` selections and optional calls, `??`
// expressions, single-value map reads, nil comparisons on options,
// narrowed reads, and the value wraps that option assigns, returns,
// arguments, and composite elements need. It runs after lowerUnions,
// then drains the synthesized imports its result-type spellings may
// have recorded. The walk is syntax.Inspect, which follows true child
// edges only, so type info, positions, and scopes attached to nodes
// never lead the pass. Builders lower embedded originals eagerly, so a
// later textOf always finds the inner splices already recorded, and the
// descend into replacements only meets idempotent no-ops.
func (fl *fileLower) lowerOptions() {
	fl.optMemo = make(map[*syntax.OptionType]syntax.Expr)
	fl.narrowDone = make(map[*syntax.Name]bool)
	fl.wrapSpan = make(map[syntax.Expr]span)
	fl.wrapOwner = make(map[syntax.Expr]int)
	fl.collectOptReturns()
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		fl.lowerOptNode(n)
		return true
	})
	fl.flushTypeImports()
}

// lowerOptNode handles one node of the option walk: statements with
// conversion targets (assignments, returns, declarations, sends, range
// clauses) get their site handling first, then every expression slot of
// every node goes through lowerOptExpr. Rewrites are idempotent, so the
// field pass may revisit already-handled subtrees.
func (fl *fileLower) lowerOptNode(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		fl.lowerOptAssign(n)
		fl.lowerOptFieldsExcept(n, "Lhs")
		return
	case *syntax.RangeClause:
		fl.lowerOptRange(n)
		fl.lowerOptFieldsExcept(n, "Lhs")
		return
	case *syntax.ReturnStmt:
		fl.lowerOptReturn(n)
	case *syntax.VarDecl:
		n.Values = fl.lowerOptVarValues(n.NameList, n.Type, n.Values)
	case *syntax.ConstDecl:
		n.Values = fl.lowerOptVarValues(n.NameList, n.Type, n.Values)
	case *syntax.SendStmt:
		fl.lowerOptSend(n)
	case *syntax.LambdaExpr:
		// M6-S6 (D4): an expression-lambda body is a syntax.Node
		// field, so the field pass below never lowers a top-level
		// `??` or `?.` in it. Lower it here and stash the checked
		// type: chains capture body text before reparse, and
		// prodType reads the live node afterwards.
		fl.lowerOptLambdaBody(n)
	}
	fl.lowerOptFields(n)
}

// lowerOptLambdaBody lowers a top-level `??` or `?.` that sits
// directly as an expression-lambda body. Interior option forms lower
// through the field pass and the Inspect descent as usual; only the
// top form needs the explicit expression call, mirroring what an
// assignment right-hand side gets from lowerOptAssign. A replaced body
// keeps its checked type in optBodyType: the replacement is synthetic
// and carries no recorded type, but chain prodType still needs the
// produced type afterwards.
func (fl *fileLower) lowerOptLambdaBody(lam *syntax.LambdaExpr) {
	body, ok := lam.Body.(syntax.Expr)
	if !ok {
		return
	}
	switch syntax.Unparen(body).(type) {
	case *syntax.CoalesceExpr, *syntax.OptSelectExpr:
	default:
		return
	}
	orig := fl.pkg.info.TypeOf(body)
	lowered := fl.lowerOptExpr(body)
	lam.Body = lowered
	if lowered != body && orig != nil {
		if fl.optBodyType == nil {
			fl.optBodyType = make(map[syntax.Expr]types2.Type)
		}
		fl.optBodyType[lowered] = orig
	}
}

// lowerOptFields rewrites the option forms held directly by n: every
// field of static type syntax.Expr, []syntax.Expr, or [N]syntax.Expr
// goes through lowerOptExpr. All other fields are left alone; Inspect
// visits the nodes they hold.
func (fl *fileLower) lowerOptFields(n syntax.Node) {
	fl.lowerOptFieldsExcept(n, "")
}

// lowerOptFieldsExcept is lowerOptFields skipping the named struct
// field. Assignment and range write targets go through lowerOptLHS
// instead, so a plain write target never gains a narrowed `.Value`.
func (fl *fileLower) lowerOptFieldsExcept(n syntax.Node, except string) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	t := v.Type()
	for i := range v.NumField() {
		if !t.Field(i).IsExported() {
			continue
		}
		if except != "" && t.Field(i).Name == except {
			continue
		}
		fl.lowerOptField(v.Field(i), t.Field(i).Type)
	}
}

// lowerOptField rewrites one field of static type ft through
// lowerOptExpr when it holds expressions.
func (fl *fileLower) lowerOptField(v reflect.Value, ft reflect.Type) {
	if ft == exprType {
		if v.IsNil() {
			return
		}
		v.Set(reflect.ValueOf(fl.lowerOptExpr(v.Interface().(syntax.Expr))))
		return
	}
	k := ft.Kind()
	if (k == reflect.Slice || k == reflect.Array) && ft.Elem() == exprType {
		for i := range v.Len() {
			ev := v.Index(i)
			if ev.IsNil() {
				continue
			}
			ev.Set(reflect.ValueOf(fl.lowerOptExpr(ev.Interface().(syntax.Expr))))
		}
	}
}

// lowerOptExpr lowers the option forms of one expression and returns
// the expression to use in its place, often the same node mutated in
// place. Transparent nodes recurse explicitly; transformed nodes
// (OptionType, CoalesceExpr, OptSelectExpr, optional calls, map reads
// needing Lookup, composites, ordinary calls) are handled whole. The
// result is safe to revisit: transformed tops are no longer option
// nodes, wraps consult recorded types that only originals carry, and
// narrowed names are pinned in narrowDone.
func (fl *fileLower) lowerOptExpr(e syntax.Expr) syntax.Expr {
	switch e := e.(type) {
	case nil:
		return nil
	case *syntax.Name:
		if out, ok := fl.lowerConstMapName(e); ok { // M5-S4 hook: plain const map uses materialize.
			return out
		}
		return fl.narrowAccess(e)
	case *syntax.BasicLit:
		return e
	case *syntax.OptionType:
		return fl.lowerOptionType(e)
	case *syntax.CoalesceExpr:
		return fl.lowerCoalesce(e)
	case *syntax.OptSelectExpr:
		return fl.lowerOptSelect(e)
	case *syntax.CallExpr:
		if out, ok := fl.lowerConstMapLen(e); ok { // M5-S4 hook: len of a const map is a constant.
			return out
		}
		return fl.lowerOptCallExpr(e)
	case *syntax.IndexExpr:
		return fl.lowerMapIndex(e)
	case *syntax.SelectorExpr:
		e.X = fl.lowerOptExpr(e.X)
		return e
	case *syntax.Operation:
		if out, ok := fl.lowerNilCompare(e); ok {
			return out
		}
		e.X = fl.lowerOptExpr(e.X)
		if e.Y != nil {
			e.Y = fl.lowerOptExpr(e.Y)
		}
		return e
	case *syntax.CompositeLit:
		return fl.lowerOptComposite(e)
	case *syntax.ParenExpr:
		e.X = fl.lowerOptExpr(e.X)
		return e
	case *syntax.ListExpr:
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
		return e
	case *syntax.SliceExpr:
		e.X = fl.lowerOptExpr(e.X)
		for i, x := range e.Index {
			if x != nil {
				e.Index[i] = fl.lowerOptExpr(x)
			}
		}
		return e
	case *syntax.AssertExpr:
		e.X = fl.lowerOptExpr(e.X)
		if e.Type != nil {
			e.Type = fl.lowerTypeExpr(e.Type)
		}
		return e
	case *syntax.TypeSwitchGuard:
		e.X = fl.lowerOptExpr(e.X)
		return e
	case *syntax.KeyValueExpr:
		e.Key = fl.lowerOptExpr(e.Key)
		e.Value = fl.lowerOptExpr(e.Value)
		return e
	case *syntax.FuncLit:
		// Signature types lower when the field walk reaches them and
		// returns lower through the return map on descend.
		return e
	default:
		return e
	}
}

// nextWrapID mints one wrapper ID for the boundary inserts of one
// wrapper built below.
func (fl *fileLower) nextWrapID() int {
	fl.wrapSeq++
	return fl.wrapSeq
}

// optType reports the checked type of e from the pre-lowering check, or
// nil when e is synthetic (already lowered) or untyped.
func (fl *fileLower) optType(e syntax.Expr) types2.Type {
	if e == nil {
		return nil
	}
	return fl.pkg.info.TypeOf(e)
}

// asOptionThroughConst returns the `*types2.Option` denoted by t, looking
// through const views as well as aliases and defined types. A `??` or `?.`
// operand that reads through a const view (SPEC T-201) keeps the
// view in its recorded type, so coeType must see through it; the
// qualifier is erased before code generation.
func (fl *fileLower) asOptionThroughConst(t types2.Type) *types2.Option {
	if c := constAsView(t); c != nil {
		return fl.asOptionThroughConst(c.Elem)
	}
	return fl.asOption(t)
}

// asOption returns the `*types2.Option` denoted by t, mirroring the
// checker's option match through aliases and defined types, or nil.
func (fl *fileLower) asOption(t types2.Type) *types2.Option {
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		return o
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if u := n.Underlying(); u != nil {
			if o, ok := u.(*types2.Option); ok {
				return o
			}
		}
	}
	return nil
}

// keepMapRead reports whether an option with element type V keeps the
// bare payload form (SPEC T-202): V is a pointer, or V is an
// interface-backed union with no nil-capable member. Every other V
// lowers to `_logos_rt.Option[V]`. Nil-capable means nil is a valid
// payload: slice, map, func, chan, or interface kinds (plus type
// parameters, nested unions, and unsafe pointers, conservatively).
func keepMapRead(V types2.Type) bool {
	if V == nil {
		return false
	}
	if _, ok := V.Underlying().(*types2.Pointer); ok {
		return true
	}
	members, isUnion := unionTerms(V)
	if !isUnion {
		return false
	}
	for _, m := range members {
		mu := types2.Unalias(m)
		if n, ok := mu.(*types2.Named); ok {
			mu = n.Underlying()
		}
		switch mu := mu.(type) {
		case *types2.Slice, *types2.Map, *types2.Signature, *types2.Chan, *types2.Interface, *types2.Union:
			return false
		case *types2.Basic:
			if mu.Kind() == types2.UnsafePointer {
				return false
			}
		case *types2.TypeParam:
			return false
		}
	}
	return true
}

// lowerOptionType lowers one `T?` type occurrence to its Go-facing form
// (SPEC T-202): a bare payload keeps the element type with the `?`
// deleted, every other element becomes `_logos_rt.Option[...]` with the
// `?` replaced by `]`. Shared type nodes (one `int?` for several names)
// lower once; later uses clone the result so no node is shared. The
// decision keys off the checked element; an unrecorded type falls back
// to the wrapped form, which is always safe.
func (fl *fileLower) lowerOptionType(e *syntax.OptionType) syntax.Expr {
	if prev, ok := fl.optMemo[e]; ok {
		return cloneType(prev, e.Pos())
	}
	pos := e.Pos()
	var elemT types2.Type
	if ot := fl.asOption(fl.optType(e)); ot != nil {
		elemT = ot.Elem
	} else if et := fl.optType(e.Elem); et != nil {
		elemT = et
	}
	qoff := fl.off(pos)
	var out syntax.Expr
	if elemT != nil && keepMapRead(elemT) {
		out = fl.lowerTypeExpr(e.Elem)
		fl.add(qoff, qoff+1, "")
	} else {
		// The head anchor comes from the original tree: lowering
		// first would descend into already-lowered nodes whose
		// synthetic positions sit mid-type.
		es := fl.optTypeHead(e.Elem)
		le := fl.lowerTypeExpr(e.Elem)
		fl.needRT = true
		fl.add(es, es, rtAlias+".Option[")
		fl.add(qoff, qoff+1, "]")
		out = at(&syntax.IndexExpr{X: selOf(pos, nameOf(pos, rtAlias), "Option"), Index: le}, pos)
		fl.deleteGroupingParens(es, qoff)
	}
	fl.optMemo[e] = out
	return out
}

// optTypeHead returns the offset of the first token of the option
// payload type. A node position covers every type but one: a FuncType
// sits at its parameter parenthesis (`func` has no stored position), so
// the wrapper would land between `func` and its parameters and split
// the type. The grammar puts nothing but trivia between the keyword and
// the parameters, so the nearest preceding `func` token is always the
// payload's own keyword. Keyword offsets come from one forward
// tokenization per file, looked up with a binary search.
func (fl *fileLower) optTypeHead(e syntax.Expr) int {
	es := fl.off(fl.optLeftmost(e))
	if _, ok := syntax.Unparen(e).(*syntax.FuncType); !ok {
		return es
	}
	if fl.funcHeadOffs == nil {
		fl.funcHeadOffs = funcHeadOffsets(fl.src)
	}
	i := sort.SearchInts(fl.funcHeadOffs, es)
	if i == 0 {
		base.Fatalf("logos: function type at offset %d has no func keyword", es)
	}
	return fl.funcHeadOffs[i-1]
}

// funcHeadOffsets returns the offsets of every `func` keyword in src,
// in source order, from a single forward tokenization starting at the
// file beginning, which is always a lexically valid position.
func funcHeadOffsets(src []byte) []int {
	var out []int
	scanSegment(src, 0, len(src), func(t tok) bool {
		if t.kind == token.FUNC {
			out = append(out, t.start)
		}
		return true
	})
	return out
}

// deleteGroupingParens deletes the parenthesis pairs that group the
// option payload starting at es and ending before the `?` at qoff, as in
// `(T?)?`, `(T)?`, or a parenthesized result `func() (T)?`. The parser
// drops these parentheses, and kept parentheses would split the
// `Option[...]` brackets (`Option[(int])` never parses). Each `)` right
// before the `?` closes one grouping pair, so the same number of `(`
// right before the payload is deleted.
func (fl *fileLower) deleteGroupingParens(es, qoff int) {
	n := 0
	for off := skipSpaceBack(fl.src, qoff); off > es && fl.src[off-1] == ')'; off = skipSpaceBack(fl.src, off-1) {
		fl.add(off-1, off, "")
		n++
	}
	for off := skipSpaceBack(fl.src, es); n > 0 && off > 0 && fl.src[off-1] == '('; off = skipSpaceBack(fl.src, off-1) {
		fl.add(off-1, off, "")
		n--
	}
}

// optLeftmost returns the position of the leftmost token of e,
// descending through option forms to the payload or receiver: a wrapper
// head anchored here fuses with nested wrapper heads at the same
// offset instead of splitting their spans. It mirrors leftmostOf for
// plain nodes (selector, index, call, and the like all start at their
// operand) and keeps descending through option nodes, which leftmostOf
// would stop at.
func (fl *fileLower) optLeftmost(e syntax.Expr) syntax.Pos {
	for {
		switch n := syntax.Unparen(e).(type) {
		case *syntax.OptSelectExpr:
			e = n.X
		case *syntax.CoalesceExpr:
			e = n.X
		case *syntax.OptionType:
			e = n.Elem
		case *syntax.SelectorExpr:
			e = n.X
		case *syntax.IndexExpr:
			e = n.X
		case *syntax.SliceExpr:
			e = n.X
		case *syntax.AssertExpr:
			e = n.X
		case *syntax.CallExpr:
			e = n.Fun
		case *syntax.Operation:
			if n.Y == nil {
				return e.Pos()
			}
			e = n.X
		default:
			return e.Pos()
		}
	}
}

// optHeadStart returns the position of the leftmost token of e keeping
// parentheses whole, for wrapper head placement: a `??` head anchored
// here starts its generated func literal before an opening paren
// instead of inside it, where it would split the receiver the chain
// pass reuses. It matches optLeftmost on paren-free shapes, so nested
// wrapper heads still fuse at a shared offset there.
func (fl *fileLower) optHeadStart(e syntax.Expr) syntax.Pos {
	for {
		switch n := e.(type) {
		case *syntax.ParenExpr:
			return e.Pos()
		case *syntax.OptSelectExpr:
			e = n.X
		case *syntax.CoalesceExpr:
			e = n.X
		case *syntax.OptionType:
			e = n.Elem
		case *syntax.SelectorExpr:
			e = n.X
		case *syntax.IndexExpr:
			e = n.X
		case *syntax.SliceExpr:
			e = n.X
		case *syntax.AssertExpr:
			e = n.X
		case *syntax.CallExpr:
			e = n.Fun
		case *syntax.Operation:
			if n.Y == nil {
				return e.Pos()
			}
			e = n.X
		default:
			return e.Pos()
		}
	}
}

// spanStart returns the eject offset where the text of e begins: the
// recorded span for a synthetic wrapper top, otherwise the leftmost
// payload position. Builders lower embedded originals eagerly, so a
// synthetic top always has an entry.
func (fl *fileLower) spanStart(e syntax.Expr) int {
	if s, ok := fl.wrapSpan[e]; ok {
		return s.start
	}
	return fl.off(fl.optLeftmost(e))
}

// lowerTypeExpr lowers the option types inside the type expression e,
// returning the expression to use in its place. Composite type
// constructors recurse; every other node stands as is (the Inspect
// backstop reaches option types nested under nodes not listed here).
func (fl *fileLower) lowerTypeExpr(e syntax.Expr) syntax.Expr {
	switch t := e.(type) {
	case *syntax.OptionType:
		return fl.lowerOptionType(t)
	case *syntax.ParenExpr:
		t.X = fl.lowerTypeExpr(t.X)
		return e
	case *syntax.Operation:
		if t.Op == syntax.Mul && t.Y == nil {
			t.X = fl.lowerTypeExpr(t.X)
		}
		return e
	case *syntax.SliceType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.ArrayType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.DotsType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.MapType:
		t.Key = fl.lowerTypeExpr(t.Key)
		t.Value = fl.lowerTypeExpr(t.Value)
		return e
	case *syntax.ChanType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.FuncType:
		for _, f := range t.ParamList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		for _, f := range t.ResultList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		return e
	case *syntax.StructType:
		for _, f := range t.FieldList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		return e
	case *syntax.InterfaceType:
		for _, f := range t.MethodList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		return e
	default:
		return e
	}
}

// spellLowered renders the checked type t as lowered Go source in this
// file: nested options become the named support instantiation when
// wrapped (bare payloads spell their element), anonymous composites
// spell structurally so options nested inside them never leak `?`
// syntax, and named types spell through spellType with a synthesized
// import recorded for flushTypeImports. It reports false when t mentions
// an unexported name of another package.
func (fl *fileLower) spellLowered(t types2.Type) (string, bool) {
	if o := fl.asOption(t); o != nil {
		if keepMapRead(o.Elem) {
			return fl.spellLowered(o.Elem)
		}
		inner, ok := fl.spellLowered(o.Elem)
		if !ok {
			return "", false
		}
		fl.needRT = true
		return rtAlias + ".Option[" + inner + "]", true
	}
	if t == nil {
		return "", false
	}
	switch c := types2.Unalias(t).(type) {
	case *types2.ConstView: // M5-S1: lowering erases const views.
		return fl.spellLowered(c.Elem)
	case *types2.Named, *types2.Alias, *types2.Basic, *types2.TypeParam:
		return fl.spellType(t)
	case *types2.Pointer:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return "*" + e, true
	case *types2.Slice:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return "[]" + e, true
	case *types2.Array:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return fmt.Sprintf("[%d]", c.Len()) + e, true
	case *types2.Map:
		k, ok := fl.spellLowered(c.Key())
		if !ok {
			return "", false
		}
		v, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return "map[" + k + "]" + v, true
	case *types2.Chan:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		switch c.Dir() {
		case types2.SendOnly:
			return "chan<- " + e, true
		case types2.RecvOnly:
			return "<-chan " + e, true
		default:
			return "chan " + e, true
		}
	case *types2.Signature:
		return fl.spellLoweredSig(c)
	case *types2.Struct:
		return fl.spellLoweredStruct(c)
	case *types2.Interface:
		return fl.spellLoweredInterface(c)
	default:
		return "", false
	}
}

// spellLoweredSig renders a signature with lowered parameter and result
// types. Parameter and result names are dropped: generated signatures
// never bind them.
func (fl *fileLower) spellLoweredSig(sig *types2.Signature) (string, bool) {
	var b strings.Builder
	b.WriteString("func(")
	for i := range sig.Params().Len() {
		if i > 0 {
			b.WriteString(", ")
		}
		p, ok := fl.spellLowered(sig.Params().At(i).Type())
		if !ok {
			return "", false
		}
		b.WriteString(p)
	}
	b.WriteString(")")
	switch n := sig.Results().Len(); n {
	case 0:
	case 1:
		r, ok := fl.spellLowered(sig.Results().At(0).Type())
		if !ok {
			return "", false
		}
		b.WriteString(" " + r)
	default:
		b.WriteString(" (")
		for i := range n {
			if i > 0 {
				b.WriteString(", ")
			}
			r, ok := fl.spellLowered(sig.Results().At(i).Type())
			if !ok {
				return "", false
			}
			b.WriteString(r)
		}
		b.WriteString(")")
	}
	return b.String(), true
}

// spellLoweredStruct renders an anonymous struct type with lowered
// field types, keeping tags. Unexported fields of another package make
// the type unspellable.
func (fl *fileLower) spellLoweredStruct(st *types2.Struct) (string, bool) {
	var b strings.Builder
	b.WriteString("struct{ ")
	for i := range st.NumFields() {
		if i > 0 {
			b.WriteString("; ")
		}
		f := st.Field(i)
		if !f.Exported() && f.Pkg() != fl.pkg.tpkg {
			return "", false
		}
		if f.Embedded() {
			t, ok := fl.spellLowered(f.Type())
			if !ok {
				return "", false
			}
			b.WriteString(t)
		} else {
			t, ok := fl.spellLowered(f.Type())
			if !ok {
				return "", false
			}
			b.WriteString(f.Name() + " " + t)
		}
		if tag := st.Tag(i); tag != "" {
			b.WriteString(" `" + tag + "`")
		}
	}
	b.WriteString(" }")
	return b.String(), true
}

// spellLoweredInterface renders an interface type with lowered method
// signatures. Empty interfaces and type names spell through spellType;
// only interfaces with methods reaching options need this path.
func (fl *fileLower) spellLoweredInterface(it *types2.Interface) (string, bool) {
	if it.NumMethods() == 0 && it.NumEmbeddeds() == 0 {
		return fl.spellType(it)
	}
	var b strings.Builder
	b.WriteString("interface{ ")
	sep := ""
	for i := range it.NumMethods() {
		m := it.Method(i)
		if !m.Exported() && m.Pkg() != fl.pkg.tpkg {
			return "", false
		}
		sig, ok := types2.Unalias(m.Type()).(*types2.Signature)
		if !ok {
			return "", false
		}
		s, ok := fl.spellLoweredSig(sig)
		if !ok {
			return "", false
		}
		b.WriteString(sep + m.Name() + s)
		sep = "; "
	}
	for i := range it.NumEmbeddeds() {
		e, ok := fl.spellLowered(it.EmbeddedType(i))
		if !ok {
			return "", false
		}
		b.WriteString(sep + e)
		sep = "; "
	}
	b.WriteString(" }")
	return b.String(), true
}

// isNilName reports whether e is the universe nil identifier.
func isNilName(e syntax.Expr) bool {
	n, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && n.Value == "nil"
}

// narrowTarget returns the declared option of the variable read by the
// plain name e when the read is narrowed (its recorded type is the
// option's element), or nil. Write targets never reach this: assigning
// statements exclude their left-hand side and the checker disables
// narrowing for address-taken and closure-assigned variables, so a hit
// is always a value read the checker proved present.
func (fl *fileLower) narrowTarget(e *syntax.Name) *types2.Option {
	obj, ok := fl.pkg.info.Uses[e].(*types2.Var)
	if !ok {
		return nil
	}
	dOpt := fl.asOption(obj.Type())
	if dOpt == nil {
		return nil
	}
	if rec := fl.optType(e); rec == nil || !types2.Identical(rec, dOpt.Elem) {
		return nil
	}
	return dOpt
}

// narrowAccess lowers one narrowed variable read: a bare payload reads
// as is, a wrapped payload reads its `.Value`, which the guard proved
// present. The eject splice is a pure insertion, so it composes with
// every enclosing rewrite.
func (fl *fileLower) narrowAccess(e *syntax.Name) syntax.Expr {
	if fl.narrowDone[e] {
		return e
	}
	fl.narrowDone[e] = true
	dOpt := fl.narrowTarget(e)
	if dOpt == nil || keepMapRead(dOpt.Elem) {
		return e
	}
	pos := e.Pos()
	off := fl.off(pos) + len(e.Value)
	acc := selOf(pos, e, "Value")
	fl.wrapOwner[acc] = fl.nextWrapID()
	fl.addOwned(fl.wrapOwner[acc], off, off, ".Value")
	return acc
}

// narrowForWrite lowers a narrowed variable in a write position (`++`,
// `--`, compound assignment): a wrapped payload updates its `.Value`
// in place, a bare payload stays.
func (fl *fileLower) narrowForWrite(e *syntax.Name) syntax.Expr {
	if fl.narrowDone[e] {
		return e
	}
	fl.narrowDone[e] = true
	dOpt := fl.narrowTarget(e)
	if dOpt == nil || keepMapRead(dOpt.Elem) {
		return e
	}
	pos := e.Pos()
	off := fl.off(pos) + len(e.Value)
	acc := selOf(pos, e, "Value")
	fl.wrapOwner[acc] = fl.nextWrapID()
	fl.addOwned(fl.wrapOwner[acc], off, off, ".Value")
	return acc
}

// coeType returns the option of a `?.` or `??` operand: usually the
// recorded type, but a narrowed plain name records its element (the
// checker's un-narrow restores the operand, not typed nodes), so a
// narrowed receiver recovers its declared option.
func (fl *fileLower) coeType(e syntax.Expr) *types2.Option {
	if o := fl.asOptionThroughConst(fl.optType(e)); o != nil {
		return o
	}
	if name, ok := syntax.Unparen(e).(*syntax.Name); ok {
		return fl.narrowTarget(name)
	}
	return nil
}

// lowerOptReceiver lowers a `?.` or `??` receiver: a narrowed plain
// name stays the option itself (the wrapper tests presence), every
// other shape lowers normally.
func (fl *fileLower) lowerOptReceiver(x syntax.Expr) syntax.Expr {
	if name, ok := syntax.Unparen(x).(*syntax.Name); ok {
		if fl.narrowTarget(name) != nil {
			fl.narrowDone[name] = true
			return x
		}
	}
	return fl.lowerOptExpr(x)
}

// needsWrap reports whether the value v of recorded type rec needs a
// wrap to meet the target type, and whether the wrap is the zero value.
// A nil aimed at a wrapped option becomes the zero `Option[T]{}`; any
// other non-identical value aimed at a wrapped option gains presence.
// Bare targets and identical types need nothing: Go assignability
// already covers them.
func (fl *fileLower) needsWrap(v syntax.Expr, rec, target types2.Type) (wrap, zero bool) {
	if target == nil {
		return false, false
	}
	tOpt := fl.asOption(target)
	if tOpt == nil {
		return false, false
	}
	if isNilName(v) {
		return !keepMapRead(tOpt.Elem), !keepMapRead(tOpt.Elem)
	}
	if rec == nil || types2.Identical(rec, target) {
		return false, false
	}
	if keepMapRead(tOpt.Elem) {
		return false, false
	}
	return true, false
}

// wrapKnown converts the (already lowered) value v of recorded type rec
// to the target type, returning the expression to use in its place. A
// plain value aimed at a nested option (for example `3` at `(int?)?`)
// wraps once per layer. The eject edit is one replacement of v's
// original span with the innermost wrap, consuming the inner splices
// through textOf, plus boundary inserts for the outer layers; the
// native tree nests the composites around v. Callers must pass the
// original node (mutated in place at most), never a synthetic one,
// except through wrapSpan.
func (fl *fileLower) wrapKnown(v syntax.Expr, rec, target types2.Type) syntax.Expr {
	wrap, zero := fl.needsWrap(v, rec, target)
	if !wrap {
		return v
	}
	pos := v.Pos()
	tText, ok := fl.spellLowered(target)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the option wrap at %s", target, pos.String())
	}
	if zero {
		n := syntax.Unparen(v).(*syntax.Name)
		off := fl.off(n.Pos())
		fl.add(off, off+3, tText+"{}")
		typ, ok := typeExprAt(tText, pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q of the option wrap at %s", tText, pos.String())
		}
		out := at(&syntax.CompositeLit{Type: typ, Rbrace: pos}, pos)
		fl.wrapSpan[out] = span{start: off, end: off + 3}
		fl.wrapOwner[out] = fl.nextWrapID()
		return out
	}
	texts := []string{tText}
	for cur := target; ; {
		elem := fl.asOption(cur).Elem
		if types2.Identical(rec, elem) {
			break
		}
		inner := fl.asOption(elem)
		if inner == nil || keepMapRead(inner.Elem) {
			break
		}
		t, ok := fl.spellLowered(elem)
		if !ok {
			base.Fatalf("logos: cannot spell type %s of the option wrap at %s", elem, pos.String())
		}
		texts = append(texts, t)
		cur = elem
	}
	vs := fl.wrapSpanOf(v)
	text := fl.textOf(vs)
	// Drain this wrapper's own boundary inserts (an IIFE head and its
	// `}()` tail) into the fused text: left outside they would split
	// the replacement from its operand. Outer wrappers' inserts at the
	// same offsets carry other owners and stay.
	id, ok := fl.wrapOwner[v]
	if !ok {
		id = fl.nextWrapID()
	} else {
		head, tail := fl.drainOwned(vs, id)
		text = head + text + tail
	}
	// One fused replacement for the whole tower: boundary inserts for
	// outer layers would interleave with an enclosing wrapper's head at
	// a shared offset, while a single span composes by position.
	fused := text
	for i := len(texts) - 1; i >= 0; i-- {
		fused = texts[i] + "{Value: " + fused + ", Present: true}"
	}
	fl.add(vs.start, vs.end, fused)
	native := v
	for i := len(texts) - 1; i >= 0; i-- {
		typ, ok := typeExprAt(texts[i], pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q of the option wrap at %s", texts[i], pos.String())
		}
		native = at(&syntax.CompositeLit{
			Type: typ,
			ElemList: []syntax.Expr{
				at(&syntax.KeyValueExpr{Key: nameOf(pos, "Value"), Value: native}, pos),
				at(&syntax.KeyValueExpr{Key: nameOf(pos, "Present"), Value: nameOf(pos, "true")}, pos),
			},
			// The unified IR writer reads NKeys == 0 as a positional
			// literal; both elements here are keyed.
			NKeys:  2,
			Rbrace: pos,
		}, pos)
	}
	fl.wrapSpan[native] = vs
	fl.wrapOwner[native] = id
	return native
}

// wrapSpanOf returns the original span of v: the recorded span for a
// synthetic wrapper top, otherwise the source extent of the node.
func (fl *fileLower) wrapSpanOf(v syntax.Expr) span {
	if s, ok := fl.wrapSpan[v]; ok {
		return s
	}
	from := fl.off(leftmostOf(v))
	return span{start: from, end: fl.exprEndNested(from)}
}

// drainOwned removes the boundary inserts owned by id at the ends of
// vs from the pending splices and returns their texts, head then tail.
// Inserts owned by other wrappers at the same offsets stay pending. A
// span inside a const for copy drains the splices of its region, where
// appendSplice recorded them.
func (fl *fileLower) drainOwned(vs span, id int) (head, tail string) {
	// Each end drains from the list holding insertions at it: the
	// narrowest moved region strictly holding the endpoint, else its
	// const for region, else the file. Ends in different lists drain
	// separately; ends sharing one list drain in one pass so a
	// zero-width span keeps HEAD's head-first behavior.
	ls := fl.ownerOfInsertion(vs.start)
	if le := fl.ownerOfInsertion(vs.end); le != ls {
		head = drainOwnedAt(ls, vs.start, id)
		tail = drainOwnedAt(le, vs.end, id)
		return head, tail
	}
	var rest []splice
	for _, sp := range *ls {
		if sp.owner != id || sp.start != sp.end || (sp.start != vs.start && sp.start != vs.end) {
			rest = append(rest, sp)
			continue
		}
		if sp.start == vs.start {
			head += sp.text
		} else {
			tail += sp.text
		}
	}
	*ls = rest
	return head, tail
}

// drainOwnedAt removes the boundary inserts owned by id at off from list
// and returns their texts.
func drainOwnedAt(list *[]splice, off, id int) string {
	var text string
	var rest []splice
	for _, sp := range *list {
		if sp.owner != id || sp.start != sp.end || sp.start != off {
			rest = append(rest, sp)
			continue
		}
		text += sp.text
	}
	*list = rest
	return text
}

// drainBodyWrapper composes the owned boundary inserts of a top-level
// `??` or `?.` wrapper that a lambda-body capture left outside its
// textOf span (D4, D7): the head insert sits exactly on the capture
// start and the tail exactly on its end, so the plain textOf keeps them
// pending. A wrapper strictly inside the span already drained through
// textOf, so drainOwned finds nothing and the text passes through
// unchanged. Bodies that are not wrapper tops pass through too.
func (fl *fileLower) drainBodyWrapper(body syntax.Node, text string) string {
	expr, ok := body.(syntax.Expr)
	if !ok {
		return text
	}
	top, ok := syntax.Unparen(expr).(*syntax.CallExpr)
	if !ok {
		return text
	}
	id, ok := fl.wrapOwner[top]
	if !ok {
		return text
	}
	vs, ok := fl.wrapSpan[top]
	if !ok {
		return text
	}
	head, tail := fl.drainOwned(vs, id)
	return head + text + tail
}

// lowerMapIndex lowers one single-value map read: a kept read stays,
// every other option-typed read becomes a Lookup call. Reads the checker
// re-recorded with the element type (comma-ok, store targets, `++`) carry
// no option and are left alone. Children lower first so embedded forms
// record their splices before the Lookup splices surround them.
func (fl *fileLower) lowerMapIndex(e *syntax.IndexExpr) syntax.Expr {
	if g, ok := fl.constMapUse(e.X); ok { // M5-S4 hook: reads of a const map use its lookup.
		return fl.lowerConstMapRead(e, g)
	}
	opt := fl.asOption(fl.optType(e))
	if opt == nil || keepMapRead(opt.Elem) {
		e.X = fl.lowerOptExpr(e.X)
		e.Index = fl.lowerOptExpr(e.Index)
		return e
	}
	id := fl.nextWrapID()
	mStart, _, lbrack, rbrack := fl.lookupSpans(e)
	fl.addOwned(id, mStart, mStart, rtAlias+".Lookup(")
	e.X = fl.lowerOptExpr(e.X)
	e.Index = fl.lowerOptExpr(e.Index)
	out := fl.lookupFinish(e, lbrack, rbrack, id)
	fl.wrapOwner[out] = id
	return out
}

// lookupSpans returns the eject offsets of a map-read rewrite from the
// original tree: the map start for the head, the key start for the
// brackets, and both bracket offsets.
func (fl *fileLower) lookupSpans(e *syntax.IndexExpr) (mStart, kStart, lbrack, rbrack int) {
	mStart = fl.off(fl.optLeftmost(e.X))
	kStart = fl.off(leftmostOf(e.Index))
	lbrack = fl.findLbrack(mStart, kStart)
	rbrack = fl.findRbrack(kStart)
	return mStart, kStart, lbrack, rbrack
}

// lookupFinish builds `_logos_rt.Lookup(m, k)` for the read e whose
// head is already recorded and records the bracket splices. head
// carries the insertion text so a fused `??` or `?.` wrapper can extend
// it. The result span is recorded for wraps of the whole read, with the
// boundary head owned by id.
func (fl *fileLower) lookupFinish(e *syntax.IndexExpr, lbrack, rbrack int, id int) syntax.Expr {
	pos := e.Pos()
	fl.needRT = true
	call := callOf(pos, selOf(pos, nameOf(pos, rtAlias), "Lookup"), []syntax.Expr{e.X, e.Index})
	fl.add(lbrack, lbrack+1, ", ")
	fl.add(rbrack, rbrack+1, ")")
	mStart := fl.spanStart(e.X)
	fl.wrapSpan[call] = span{start: mStart, end: rbrack + 1}
	return call
}

// lowerCoalesce lowers `x ?? y` to an immediately invoked function
// literal of the element type: a kept map read fuses with a comma-ok
// test, a Lookup-lowered read tests `.Present`, a bare payload tests
// `!= nil`, and any other option tests `.Present`. The fallback lowers
// and wraps to the element type. Nested forms inside the operands lower
// eagerly, so the recorded wrapper span covers finished splices.
func (fl *fileLower) lowerCoalesce(e *syntax.CoalesceExpr) syntax.Expr {
	opt := fl.coeType(e.X)
	ux, isIndex := syntax.Unparen(e.X).(*syntax.IndexExpr)
	if isIndex {
		if _, ok := fl.constMapUse(ux.X); ok { // M5-S4: a const map read takes the generic option path.
			isIndex = false
		}
	}
	if opt == nil && isIndex {
		opt = fl.asOption(fl.optType(ux))
	}
	if opt == nil {
		// Not an option left side; checked code never lands here.
		e.X = fl.lowerOptExpr(e.X)
		e.Y = fl.lowerOptExpr(e.Y)
		return e
	}
	pos := e.Pos()
	id := fl.nextWrapID()
	etext, ok := fl.spellLowered(opt.Elem)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the ?? expression at %s", opt.Elem, pos.String())
	}
	resultType, ok := typeExprAt(etext, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the ?? expression at %s", etext, pos.String())
	}
	signature := at(&syntax.FuncType{
		ParamList:  []*syntax.Field{},
		ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
	}, pos)
	qoff := fl.off(pos)
	yStart := fl.off(leftmostOf(e.Y))
	yEnd := fl.exprEnd(yStart)
	if isCompositeType(fl.optType(e.Y)) {
		yEnd = fl.exprEndNested(yStart)
	}
	fl.add(qoff, qoff+2, "")
	fl.addOwned(id, yEnd, yEnd, " }()")
	if isIndex {
		mStart, _, lbrack, rbrack := fl.lookupSpans(ux)
		// The head anchors before any grouping parens around the read,
		// and the pair is deleted so the fused init stays valid Go.
		// Without parens both are no-ops: the anchor equals mStart and
		// no `)` stands right before `??`.
		fl.deleteGroupingParens(mStart, qoff)
		xStart := fl.off(fl.optHeadStart(e.X))
		xEnd := rbrack + 1
		wy := fl.wrapKnown(fl.lowerOptExpr(e.Y), fl.optType(e.Y), opt.Elem)
		if keepMapRead(opt.Elem) {
			fl.addOwned(id, xStart, xStart, "func() "+etext+" { if "+optTemp+", "+optOk+" := ")
			ux.X = fl.lowerOptExpr(ux.X)
			ux.Index = fl.lowerOptExpr(ux.Index)
			// func() E { if v, ok := m[k]; ok { return v }; return y }()
			init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp), nameOf(pos, optOk)}, []syntax.Expr{ux})
			body := []syntax.Stmt{
				ifOf(pos, init, nameOf(pos, optOk), []syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, optTemp)})}),
				returnOf(pos, []syntax.Expr{wy}),
			}
			lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
			fl.addOwned(id, xEnd, xEnd, "; "+optOk+" { return "+optTemp+" }; return ")
			out := callOf(pos, lit, nil)
			fl.wrapSpan[out] = span{start: xStart, end: yEnd}
			fl.wrapOwner[out] = id
			return out
		}
		// func() E { if v := Lookup(m, k); v.Present { return v.Value }; return y }()
		// The fused head goes first so nested map/key heads follow it.
		fl.addOwned(id, xStart, xStart, "func() "+etext+" { if "+optTemp+" := "+rtAlias+".Lookup(")
		ux.X = fl.lowerOptExpr(ux.X)
		ux.Index = fl.lowerOptExpr(ux.Index)
		call := fl.lookupFinish(ux, lbrack, rbrack, id)
		init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{call})
		cond := selOf(pos, nameOf(pos, optTemp), "Present")
		then := []syntax.Stmt{returnOf(pos, []syntax.Expr{selOf(pos, nameOf(pos, optTemp), "Value")})}
		body := []syntax.Stmt{
			ifOf(pos, init, cond, then),
			returnOf(pos, []syntax.Expr{wy}),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		// The fused Lookup head left its call open and lookupFinish
		// recorded the closing `)`; the mid only adds the presence test.
		fl.addOwned(id, xEnd, xEnd, "; "+optTemp+".Present { return "+optTemp+".Value }; return ")
		out := callOf(pos, lit, nil)
		fl.wrapSpan[out] = span{start: xStart, end: yEnd}
		fl.wrapOwner[out] = id
		return out
	}
	bare := keepMapRead(opt.Elem)
	xStart := fl.off(fl.optHeadStart(e.X))
	fl.addOwned(id, xStart, xStart, "func() "+etext+" { if "+optTemp+" := ")
	nx := fl.lowerOptReceiver(e.X)
	wy := fl.wrapKnown(fl.lowerOptExpr(e.Y), fl.optType(e.Y), opt.Elem)
	xEnd := lastTokenEnd(fl.src, xStart, qoff)
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{nx})
	var cond syntax.Expr
	var mid string
	if bare {
		cond = notNil(pos, nameOf(pos, optTemp))
		mid = "; " + optTemp + " != nil { return " + optTemp + " }; return "
	} else {
		cond = selOf(pos, nameOf(pos, optTemp), "Present")
		mid = "; " + optTemp + ".Present { return " + optTemp + ".Value }; return "
	}
	body := []syntax.Stmt{
		ifOf(pos, init, cond, []syntax.Stmt{returnOf(pos, []syntax.Expr{thenValue(pos, bare)})}),
		returnOf(pos, []syntax.Expr{wy}),
	}
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
	fl.addOwned(id, xEnd, xEnd, mid)
	out := callOf(pos, lit, nil)
	fl.wrapSpan[out] = span{start: xStart, end: yEnd}
	fl.wrapOwner[out] = id
	return out
}

// thenValue builds the `??` wrapper's taken-branch value: the bare
// temporary itself, or its `.Value`.
func thenValue(pos syntax.Pos, bare bool) syntax.Expr {
	if bare {
		return nameOf(pos, optTemp)
	}
	return selOf(pos, nameOf(pos, optTemp), "Value")
}

// isCompositeType reports whether t's underlying type opens a brace in
// source: struct, array, slice, map, or pointer (address-of) literals.
// The `??` fallback end scan nests braces only then, so an `if` body
// brace after a plain fallback still ends the scan.
func isCompositeType(t types2.Type) bool {
	switch types2.Unalias(t).(type) {
	case *types2.Struct, *types2.Array, *types2.Slice, *types2.Map, *types2.Pointer:
		return true
	}
	return false
}

// fieldTypeOf returns the type a `?.` selection yields before option
// wrapping: a field's type, or a method's signature (never an option
// itself). It reports nil for anything else.
func fieldTypeOf(obj types2.Object) types2.Type {
	switch obj := obj.(type) {
	case *types2.Var:
		return obj.Type()
	case *types2.Func:
		return obj.Type()
	}
	return nil
}

// lowerOptSelect lowers `x?.Sel` to an immediately invoked function
// literal of the checked (flattened) result option: none yields the
// zero, some yields the selected field or method value wrapped. The
// receiver evaluates once into the temporary; a bare receiver tests
// `!= nil` and selects directly, a wrapped one tests `.Present` and
// selects through `.Value`. The receiver lowers eagerly, so chains
// (`x?.y?.z`) nest finished wrappers and map reads fuse their Lookup
// heads in input order.
func (fl *fileLower) lowerOptSelect(e *syntax.OptSelectExpr) syntax.Expr {
	pos := e.Pos()
	id := fl.nextWrapID()
	ox := fl.coeType(e.X)
	ro := fl.asOption(fl.optType(e))
	if ox == nil || ro == nil {
		// Not option-typed; checked code never lands here.
		e.X = fl.lowerOptExpr(e.X)
		return e
	}
	// The head goes first so nested receiver heads follow it at a
	// shared offset.
	xStart := fl.off(fl.optLeftmost(e.X))
	rText, ok := fl.spellLowered(ro)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the ?. expression at %s", ro, pos.String())
	}
	fl.addOwned(id, xStart, xStart, "func() "+rText+" { if "+optTemp+" := ")
	nx := fl.lowerOptReceiver(e.X)
	bareRecv := keepMapRead(ox.Elem)
	accessRec := ro.Elem
	if obj, _, _ := types2.LookupFieldOrMethod(ox.Elem, true, fl.pkg.tpkg, e.Sel.Value); obj != nil {
		if ft := fieldTypeOf(obj); ft != nil && fl.asOption(ft) != nil {
			accessRec = ro
		}
	}
	recv := nameOf(pos, optTemp)
	acc := selOf(pos, recv, e.Sel.Value)
	accSuffix := "." + e.Sel.Value
	if !bareRecv {
		acc = selOf(pos, selOf(pos, recv, "Value"), e.Sel.Value)
		accSuffix = ".Value." + e.Sel.Value
	}
	some, someHead, someTail := fl.wrapParts(acc, accessRec, ro, pos)
	var none syntax.Expr
	var noneT string
	if keepMapRead(ro.Elem) {
		none = nilOf(pos)
		noneT = "nil"
	} else {
		noneT = rText + "{}"
		typ, ok := typeExprAt(rText, pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q of the ?. expression at %s", rText, pos.String())
		}
		none = at(&syntax.CompositeLit{Type: typ, Rbrace: pos}, pos)
	}
	var cond syntax.Expr
	var condT string
	if bareRecv {
		cond = notNil(pos, nameOf(pos, optTemp))
		condT = optTemp + " != nil"
	} else {
		cond = selOf(pos, nameOf(pos, optTemp), "Present")
		condT = optTemp + ".Present"
	}
	resultType, ok := typeExprAt(rText, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the ?. expression at %s", rText, pos.String())
	}
	signature := at(&syntax.FuncType{
		ParamList:  []*syntax.Field{},
		ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
	}, pos)
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{nx})
	body := []syntax.Stmt{
		ifOf(pos, init, cond, []syntax.Stmt{returnOf(pos, []syntax.Expr{some})}),
		returnOf(pos, []syntax.Expr{none}),
	}
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
	xEnd := lastTokenEnd(fl.src, xStart, fl.off(pos))
	selEnd := fl.off(e.Sel.Pos()) + len(e.Sel.Value)
	fl.add(xEnd, selEnd, "; "+condT+" { return "+someHead+"v"+accSuffix+someTail+" }; return "+noneT+" }()")
	out := callOf(pos, lit, nil)
	fl.wrapSpan[out] = span{start: xStart, end: selEnd}
	fl.wrapOwner[out] = id
	return out
}

// wrapParts builds the present-branch value of a `?.` selection or
// optional call: the access expression wrapped to the result option
// when needed, plus the head and tail text that spells the same wrap
// in ejected output. Empty head and tail mean no wrap.
func (fl *fileLower) wrapParts(acc syntax.Expr, rec, target types2.Type, pos syntax.Pos) (some syntax.Expr, head, tail string) {
	wrap, _ := fl.needsWrap(acc, rec, target)
	if !wrap {
		return acc, "", ""
	}
	tText, ok := fl.spellLowered(target)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the ?. value at %s", target, pos.String())
	}
	typ, ok := typeExprAt(tText, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the ?. value at %s", tText, pos.String())
	}
	return at(&syntax.CompositeLit{
		Type: typ,
		ElemList: []syntax.Expr{
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Value"), Value: acc}, pos),
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Present"), Value: nameOf(pos, "true")}, pos),
		},
		// The unified IR writer reads NKeys == 0 as a positional
		// literal; both elements here are keyed.
		NKeys:  2,
		Rbrace: pos,
	}, pos), tText + "{Value: ", ", Present: true}"
}

// lowerOptCallExpr lowers an ordinary call: the function lowers, every
// argument lowers and wraps to its parameter type. A call whose Fun is
// a `?.` selection takes the optional-call path instead, which tests
// presence around the whole call.
func (fl *fileLower) lowerOptCallExpr(e *syntax.CallExpr) syntax.Expr {
	if osel, ok := syntax.Unparen(e.Fun).(*syntax.OptSelectExpr); ok {
		return fl.lowerOptCall(e, osel)
	}
	e.Fun = fl.lowerOptExpr(e.Fun)
	var targets []types2.Type
	if sig := fl.callSig(e); sig != nil {
		targets = fl.callTargets(e, sig)
	}
	for i, a := range e.ArgList {
		la := fl.lowerOptExpr(a)
		if targets != nil && i < len(targets) {
			la = fl.wrapKnown(la, fl.optType(a), targets[i])
		}
		e.ArgList[i] = la
	}
	return e
}

// callSig returns the signature of a call's function operand, or nil
// for builtins and unknown shapes (arguments then lower as reads).
func (fl *fileLower) callSig(e *syntax.CallExpr) *types2.Signature {
	switch fun := syntax.Unparen(e.Fun).(type) {
	case *syntax.Name:
		if obj := fl.pkg.info.Uses[fun]; obj != nil {
			if fn, ok := obj.(*types2.Func); ok {
				if sig, ok := types2.Unalias(fn.Type()).(*types2.Signature); ok {
					return sig
				}
			}
		}
		return nil
	case *syntax.SelectorExpr:
		if sel := fl.pkg.info.Selections[fun]; sel != nil {
			if fn, ok := sel.Obj().(*types2.Func); ok {
				if sig, ok := types2.Unalias(fn.Type()).(*types2.Signature); ok {
					return sig
				}
			}
		}
		if sig, ok := types2.Unalias(fl.optType(fun)).(*types2.Signature); ok {
			return sig
		}
		return nil
	default:
		if sig, ok := types2.Unalias(fl.optType(fun)).(*types2.Signature); ok {
			return sig
		}
		return nil
	}
}

// callTargets maps each argument to its parameter type: variadic tails
// take the element type, a `...` spread takes the slice type, and a
// method expression prepends the receiver. A nil entry means reads only.
func (fl *fileLower) callTargets(e *syntax.CallExpr, sig *types2.Signature) []types2.Type {
	params := sig.Params()
	n := len(e.ArgList)
	targets := make([]types2.Type, n)
	shift := 0
	if fun, ok := syntax.Unparen(e.Fun).(*syntax.SelectorExpr); ok {
		isType := fun.X.GetTypeInfo().IsType()
		if tv, ok := fl.pkg.info.Types[fun.X]; ok {
			isType = tv.IsType()
		}
		if isType {
			if recv := sig.Recv(); recv != nil && n > 0 {
				targets[0] = recv.Type()
				shift = 1
			}
		}
	}
	for i := range targets {
		targets[i] = fl.paramTarget(params, sig.Variadic(), e.HasDots, n, i-shift)
	}
	return targets
}

// paramTarget returns the parameter type for the argument at index i
// (already shifted past a method-expression receiver), or nil.
func (fl *fileLower) paramTarget(params *types2.Tuple, variadic, dots bool, nargs, i int) types2.Type {
	if i < 0 {
		return nil
	}
	np := params.Len()
	if !variadic {
		if i < np {
			return params.At(i).Type()
		}
		return nil
	}
	if np == 0 {
		return nil
	}
	if dots {
		if i < np {
			return params.At(i).Type()
		}
		return nil
	}
	if i < np-1 {
		return params.At(i).Type()
	}
	if s, ok := types2.Unalias(params.At(np - 1).Type()).(*types2.Slice); ok {
		return s.Elem()
	}
	return nil
}

// lowerOptCall lowers `x?.M(args)` to an immediately invoked function
// literal: none skips the call, some runs it on the unwrapped receiver
// and wraps each result. A void method yields a bare closure
// statement. Arguments lower eagerly and wrap to the parameter types;
// their eject text composes through textOf.
func (fl *fileLower) lowerOptCall(e *syntax.CallExpr, osel *syntax.OptSelectExpr) syntax.Expr {
	pos := osel.Pos()
	id := fl.nextWrapID()
	ox := fl.coeType(osel.X)
	if ox == nil {
		e.Fun = fl.lowerOptExpr(e.Fun)
		for i, a := range e.ArgList {
			e.ArgList[i] = fl.lowerOptExpr(a)
		}
		return e
	}
	obj, _, _ := types2.LookupFieldOrMethod(ox.Elem, true, fl.pkg.tpkg, osel.Sel.Value)
	fn, ok := obj.(*types2.Func)
	var sig *types2.Signature
	if ok {
		sig, ok = types2.Unalias(fn.Type()).(*types2.Signature)
	}
	if !ok {
		osel.X = fl.lowerOptReceiver(osel.X)
		e.Fun = osel
		for i, a := range e.ArgList {
			e.ArgList[i] = fl.lowerOptExpr(a)
		}
		return e
	}
	// The head goes first so nested receiver heads follow it at a
	// shared offset. Result spellings come before any lowering.
	xStart := fl.off(fl.optLeftmost(osel.X))
	rs := sig.Results()
	var headText string
	var rText string
	var rTexts []string
	switch rs.Len() {
	case 0:
		headText = "func() { if " + optTemp + " := "
	case 1:
		r := rs.At(0).Type()
		ro := r
		if fl.asOption(r) == nil {
			ro = &types2.Option{Elem: r}
		}
		var ok bool
		rText, ok = fl.spellLowered(ro)
		if !ok {
			base.Fatalf("logos: cannot spell type %s of the ?. call at %s", ro, pos.String())
		}
		headText = "func() " + rText + " { if " + optTemp + " := "
	default:
		n := rs.Len()
		rTexts = make([]string, n)
		for i := range n {
			r := rs.At(i).Type()
			ro := r
			if fl.asOption(r) == nil {
				ro = &types2.Option{Elem: r}
			}
			t, ok := fl.spellLowered(ro)
			if !ok {
				base.Fatalf("logos: cannot spell type %s of the ?. call at %s", ro, pos.String())
			}
			rTexts[i] = t
		}
		headText = "func() (" + strings.Join(rTexts, ", ") + ") { if " + optTemp + " := "
	}
	fl.addOwned(id, xStart, xStart, headText)
	nx := fl.lowerOptReceiver(osel.X)
	params := sig.Params()
	for i, a := range e.ArgList {
		la := fl.lowerOptExpr(a)
		e.ArgList[i] = fl.wrapKnown(la, fl.optType(a), fl.paramTarget(params, sig.Variadic(), e.HasDots, len(e.ArgList), i))
	}
	bareRecv := keepMapRead(ox.Elem)
	recv := nameOf(pos, optTemp)
	accSuffix := "." + osel.Sel.Value
	var racc syntax.Expr = selOf(pos, recv, osel.Sel.Value)
	if !bareRecv {
		racc = selOf(pos, selOf(pos, recv, "Value"), osel.Sel.Value)
		accSuffix = ".Value." + osel.Sel.Value
	}
	var cond syntax.Expr
	var condT string
	if bareRecv {
		cond = notNil(pos, nameOf(pos, optTemp))
		condT = optTemp + " != nil"
	} else {
		cond = selOf(pos, nameOf(pos, optTemp), "Present")
		condT = optTemp + ".Present"
	}
	xEnd := lastTokenEnd(fl.src, xStart, fl.off(pos))
	selEnd := fl.off(osel.Sel.Pos()) + len(osel.Sel.Value)
	lparen, rparen := fl.findCallParens(selEnd)
	argsText := fl.textOf(span{start: lparen, end: rparen + 1})
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{nx})
	call := callOf(pos, racc, e.ArgList)
	switch rs.Len() {
	case 0:
		lit := at(&syntax.FuncLit{
			Type: at(&syntax.FuncType{}, pos),
			Body: blockOf(pos, []syntax.Stmt{
				ifOf(pos, init, cond, []syntax.Stmt{exprStmtOf(pos, call)}),
			}),
		}, pos)
		fl.add(xEnd, rparen+1, "; "+condT+" { v"+accSuffix+argsText+" } }()")
		out := callOf(pos, lit, nil)
		fl.wrapSpan[out] = span{start: xStart, end: rparen + 1}
		fl.wrapOwner[out] = id
		return out
	case 1:
		r := rs.At(0).Type()
		ro := r
		if fl.asOption(r) == nil {
			ro = &types2.Option{Elem: r}
		}
		some, someHead, someTail := fl.wrapParts(call, r, ro, pos)
		var none syntax.Expr
		var noneT string
		if bare, ok := fl.bareResult(ro); ok && bare {
			none = nilOf(pos)
			noneT = "nil"
		} else {
			noneT = rText + "{}"
			typ, ok := typeExprAt(rText, pos)
			if !ok {
				base.Fatalf("logos: cannot respell type %q of the ?. call at %s", rText, pos.String())
			}
			none = at(&syntax.CompositeLit{Type: typ, Rbrace: pos}, pos)
		}
		resultType, ok := typeExprAt(rText, pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q of the ?. call at %s", rText, pos.String())
		}
		signature := at(&syntax.FuncType{
			ParamList:  []*syntax.Field{},
			ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
		}, pos)
		body := []syntax.Stmt{
			ifOf(pos, init, cond, []syntax.Stmt{returnOf(pos, []syntax.Expr{some})}),
			returnOf(pos, []syntax.Expr{none}),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		fl.add(xEnd, rparen+1, "; "+condT+" { return "+someHead+"v"+accSuffix+argsText+someTail+" }; return "+noneT+" }()")
		out := callOf(pos, lit, nil)
		fl.wrapSpan[out] = span{start: xStart, end: rparen + 1}
		fl.wrapOwner[out] = id
		return out
	default:
		n := rs.Len()
		rTexts := make([]string, n)
		roList := make([]types2.Type, n)
		recList := make([]types2.Type, n)
		for i := range n {
			r := rs.At(i).Type()
			ro := r
			if fl.asOption(r) == nil {
				ro = &types2.Option{Elem: r}
			}
			roList[i] = ro
			recList[i] = r
			t, ok := fl.spellLowered(ro)
			if !ok {
				base.Fatalf("logos: cannot spell type %s of the ?. call at %s", ro, pos.String())
			}
			rTexts[i] = t
		}
		fields := make([]*syntax.Field, n)
		for i := range n {
			typ, ok := typeExprAt(rTexts[i], pos)
			if !ok {
				base.Fatalf("logos: cannot respell type %q of the ?. call at %s", rTexts[i], pos.String())
			}
			fields[i] = at(&syntax.Field{Type: typ}, pos)
		}
		signature := at(&syntax.FuncType{
			ParamList:  []*syntax.Field{},
			ResultList: fields,
		}, pos)
		tmps := make([]syntax.Expr, n)
		rets := make([]syntax.Expr, n)
		var headNames, retWraps, noneList []string
		for i := range n {
			tmps[i] = nameOf(pos, fmt.Sprintf(optWrapTemp, i))
			some, sh, st := fl.wrapParts(tmps[i], recList[i], roList[i], pos)
			rets[i] = some
			headNames = append(headNames, fmt.Sprintf(optWrapTemp, i))
			retWraps = append(retWraps, sh+fmt.Sprintf(optWrapTemp, i)+st)
			if bare, ok := fl.bareResult(roList[i]); ok && bare {
				noneList = append(noneList, "nil")
			} else {
				noneList = append(noneList, rTexts[i]+"{}")
			}
		}
		then := []syntax.Stmt{
			assignOf(pos, tmps, []syntax.Expr{call}),
			returnOf(pos, rets),
		}
		body := []syntax.Stmt{
			ifOf(pos, init, cond, then),
			returnOf(pos, fl.noneExprs(pos, rs, rTexts)),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		fl.add(xEnd, rparen+1, "; "+condT+" { "+strings.Join(headNames, ", ")+" := v"+accSuffix+argsText+"; return "+strings.Join(retWraps, ", ")+" }; return "+strings.Join(noneList, ", ")+" }()")
		out := callOf(pos, lit, nil)
		fl.wrapSpan[out] = span{start: xStart, end: rparen + 1}
		fl.wrapOwner[out] = id
		return out
	}
}

// bareResult reports whether the `?.` result option uses the bare
// payload form.
func (fl *fileLower) bareResult(ro types2.Type) (bool, bool) {
	o := fl.asOption(ro)
	if o == nil {
		return false, false
	}
	return keepMapRead(o.Elem), true
}

// noneExprs builds the none-branch values of a multi-result optional
// call.
func (fl *fileLower) noneExprs(pos syntax.Pos, rs *types2.Tuple, rTexts []string) []syntax.Expr {
	out := make([]syntax.Expr, rs.Len())
	for i := range rs.Len() {
		r := rs.At(i).Type()
		ro := r
		if fl.asOption(r) == nil {
			ro = &types2.Option{Elem: r}
		}
		if bare, ok := fl.bareResult(ro); ok && bare {
			out[i] = nilOf(pos)
			continue
		}
		typ, ok := typeExprAt(rTexts[i], pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q of the ?. call at %s", rTexts[i], pos.String())
		}
		out[i] = at(&syntax.CompositeLit{Type: typ, Rbrace: pos}, pos)
	}
	return out
}

// lowerNilCompare lowers `x == nil` and `x != nil` (either operand
// order) when the other side is a wrapped option: none tests become
// presence tests. Bare payloads and non-options stay as they are. It
// reports whether it rewrote the operation.
func (fl *fileLower) lowerNilCompare(e *syntax.Operation) (syntax.Expr, bool) {
	if e.Op != syntax.Eql && e.Op != syntax.Neq {
		return nil, false
	}
	if e.X == nil || e.Y == nil {
		return nil, false
	}
	lx, rx := syntax.Unparen(e.X), syntax.Unparen(e.Y)
	lnil := isNilName(lx)
	rnil := isNilName(rx)
	if lnil == rnil {
		return nil, false
	}
	var xside, nilside syntax.Expr
	if rnil {
		xside, nilside = e.X, rx
	} else {
		xside, nilside = e.Y, lx
	}
	ox := fl.asOption(fl.optType(xside))
	if ox == nil || keepMapRead(ox.Elem) {
		return nil, false
	}
	pos := e.Pos()
	present := selOf(pos, xside, "Present")
	var out syntax.Expr = present
	if e.Op == syntax.Eql {
		out = at(&syntax.Operation{Op: syntax.Not, X: present}, pos)
	}
	bang := ""
	if e.Op == syntax.Eql {
		bang = "!"
	}
	// StartPos descends to the operand surface like leftmostOf but
	// keeps parentheses whole, so the bang lands before an opening
	// paren instead of inside it, where it would invert the operand
	// instead of the presence test.
	xs := fl.off(syntax.StartPos(xside))
	if rnil {
		// The nil name ends exactly; unmatched group parentheses end
		// at their closers. The replace spans the operator through
		// the nil, keeping the spaces around the whole comparison and
		// any comment between the operand and the operator.
		op := fl.findCmpOp(xs, fl.off(leftmostOf(nilside)))
		fl.add(xs, xs, bang)
		fl.add(op, fl.nilEnd(op, fl.off(syntax.EndPos(nilside))), ".Present")
	} else {
		// The replace spans the whole `nil` prefix through the
		// operator, so the bang lands directly on the operand, and
		// the operand end (makelen.operandEnd, exact past the last
		// operand token, comments skipped not consumed) places
		// `.Present` without a stray space.
		fl.add(fl.off(e.X.Pos()), xs, bang)
		fl.add(fl.operandEnd(xs), fl.operandEnd(xs), ".Present")
	}
	return out, true
}

// nilEnd returns the offset just past the nil operand whose Name ends
// at inner, including any parentheses around it: each unmatched opening
// parenthesis between the comparison operator op and the nil takes its
// closing paren too, so the replacement never leaves a stray `)`. The
// range holds only the operator, trivia, grouping parentheses, and the
// nil itself, so counting `(` there is exact with no tree dependence.
func (fl *fileLower) nilEnd(op, inner int) int {
	end := inner
	depth := 0
	scanSegment(fl.src, op, end, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.LPAREN {
			depth++
		}
		return true
	})
	for range depth {
		matched := false
		scanSegment(fl.src, end, len(fl.src), func(t tok) bool {
			if t.comment || t.auto {
				return true
			}
			if t.kind == token.RPAREN {
				end = t.span.end
				matched = true
			}
			return false
		})
		if !matched {
			base.Fatalf("logos: parenthesized nil at offset %d has no )", op)
		}
	}
	return end
}

// findCmpOp returns the offset of the `==` or `!=` operator ending the
// expression starting at from: the first depth-zero equality token
// before to. A last-token scan would swallow the operator into the
// operand end; the comparison operand never holds a depth-zero
// equality itself (it would type as bool, never an option).
func (fl *fileLower) findCmpOp(from, to int) int {
	op := -1
	depth := 0
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth > 0 {
				depth--
			}
		case token.EQL, token.NEQ:
			if depth == 0 {
				op = t.start
				return false
			}
		}
		return true
	})
	if op < 0 {
		base.Fatalf("logos: nil comparison at offset %d has no == or !=", from)
	}
	return op
}

// unlist splits a possibly-list expression for pairwise handling.
func unlist(e syntax.Expr) ([]syntax.Expr, bool) {
	if l, ok := syntax.Unparen(e).(*syntax.ListExpr); ok {
		return l.ElemList, true
	}
	return []syntax.Expr{e}, false
}

// lowerOptAssign lowers an assignment: `:=` defines (reads only), plain
// `=` wraps each right-hand side to its target, and `++`, `--`, and
// compound forms narrow-rewrite a wrapped write target in place.
func (fl *fileLower) lowerOptAssign(s *syntax.AssignStmt) {
	if s.Op == syntax.Def {
		fl.lowerOptLHS(s.Lhs)
		if s.Rhs != nil {
			s.Rhs = fl.lowerOptExpr(s.Rhs)
		}
		return
	}
	if s.Rhs == nil {
		if name, ok := syntax.Unparen(s.Lhs).(*syntax.Name); ok {
			s.Lhs = fl.narrowForWrite(name)
		} else {
			fl.lowerOptLHS(s.Lhs)
		}
		return
	}
	if s.Op != 0 {
		if name, ok := syntax.Unparen(s.Lhs).(*syntax.Name); ok {
			s.Lhs = fl.narrowForWrite(name)
		} else {
			fl.lowerOptLHS(s.Lhs)
		}
		s.Rhs = fl.lowerOptExpr(s.Rhs)
		return
	}
	lhs, _ := unlist(s.Lhs)
	rhs, rhsList := unlist(s.Rhs)
	if len(lhs) > 1 && len(rhs) == 1 {
		fl.lowerOptLHS(s.Lhs)
		s.Rhs = fl.lowerOptExpr(s.Rhs)
		return
	}
	if len(lhs) == len(rhs) {
		for i := range lhs {
			nr := fl.lowerOptExpr(rhs[i])
			nr = fl.wrapKnown(nr, fl.optType(rhs[i]), fl.targetOfLHS(lhs[i]))
			if rhsList {
				s.Rhs.(*syntax.ListExpr).ElemList[i] = nr
			} else {
				s.Rhs = nr
			}
		}
		fl.lowerOptLHS(s.Lhs)
		return
	}
	fl.lowerOptLHS(s.Lhs)
	s.Rhs = fl.lowerOptExpr(s.Rhs)
}

// lowerOptLHS lowers an assignment write target: the top name stands
// (writes never narrow), every other position lowers as a read.
func (fl *fileLower) lowerOptLHS(e syntax.Expr) {
	switch e := e.(type) {
	case *syntax.Name:
	case *syntax.ListExpr:
		for _, x := range e.ElemList {
			fl.lowerOptLHS(x)
		}
	case *syntax.ParenExpr:
		fl.lowerOptLHS(e.X)
	case *syntax.IndexExpr:
		e.X = fl.lowerOptExpr(e.X)
		e.Index = fl.lowerOptExpr(e.Index)
	case *syntax.SelectorExpr:
		e.X = fl.lowerOptExpr(e.X)
	case *syntax.Operation:
		if e.Op == syntax.Mul && e.Y == nil {
			e.X = fl.lowerOptExpr(e.X)
		}
	}
}

// targetOfLHS returns the declared type an assignment target converts
// to, or nil when unknown. Names resolve through Uses, fields through
// Selections, index and dereference targets through their container.
func (fl *fileLower) targetOfLHS(e syntax.Expr) types2.Type {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if obj, ok := fl.pkg.info.Uses[e].(*types2.Var); ok {
			return obj.Type()
		}
		return nil
	case *syntax.SelectorExpr:
		if sel := fl.pkg.info.Selections[e]; sel != nil {
			return sel.Obj().Type()
		}
		return nil
	case *syntax.IndexExpr:
		switch c := types2.Unalias(fl.optType(e.X)).(type) {
		case *types2.Slice:
			return c.Elem()
		case *types2.Array:
			return c.Elem()
		case *types2.Map:
			return c.Elem()
		}
		return nil
	case *syntax.Operation:
		if e.Op == syntax.Mul && e.Y == nil {
			if p, ok := types2.Unalias(fl.optType(e.X)).(*types2.Pointer); ok {
				return p.Elem()
			}
		}
		return nil
	}
	return nil
}

// collectOptReturns maps every return statement to its enclosing
// function's result types. Nested function literals carry their own
// signatures, so each body walk prunes literals and attributes each
// return to exactly one function.
func (fl *fileLower) collectOptReturns() {
	fl.retResults = make(map[*syntax.ReturnStmt][]types2.Type)
	type fb struct {
		body    *syntax.BlockStmt
		results []types2.Type
	}
	var list []fb
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncDecl:
			if n.Body != nil {
				list = append(list, fb{n.Body, fl.funcDeclResults(n)})
			}
		case *syntax.FuncLit:
			if sig, ok := types2.Unalias(fl.optType(n)).(*types2.Signature); ok {
				list = append(list, fb{n.Body, tupleTypes(sig.Results())})
			}
		}
		return true
	})
	for _, f := range list {
		if f.body == nil {
			continue
		}
		results := f.results
		syntax.Inspect(f.body, func(n syntax.Node) bool {
			if _, ok := n.(*syntax.FuncLit); ok {
				return false
			}
			if rs, ok := n.(*syntax.ReturnStmt); ok {
				fl.retResults[rs] = results
			}
			return true
		})
	}
}

// funcDeclResults returns the checked result types of a function
// declaration.
func (fl *fileLower) funcDeclResults(fn *syntax.FuncDecl) []types2.Type {
	if obj := fl.pkg.info.Defs[fn.Name]; obj != nil {
		if f, ok := obj.(*types2.Func); ok {
			if sig, ok := types2.Unalias(f.Type()).(*types2.Signature); ok {
				return tupleTypes(sig.Results())
			}
		}
	}
	return nil
}

// tupleTypes flattens a result tuple to a slice.
func tupleTypes(t *types2.Tuple) []types2.Type {
	if t == nil {
		return nil
	}
	out := make([]types2.Type, t.Len())
	for i := range t.Len() {
		out[i] = t.At(i).Type()
	}
	return out
}

// lowerOptReturn lowers a return statement's results and wraps each to
// its function result type. A single multi-valued call passes through;
// synthetic wrapper returns carry no entry and lower as reads.
func (fl *fileLower) lowerOptReturn(s *syntax.ReturnStmt) {
	if s.Results == nil {
		return
	}
	orig := s.Results
	low := fl.lowerOptExpr(orig)
	s.Results = low
	res := fl.retResults[s]
	if len(res) == 0 {
		return
	}
	ll, lowList := unlist(low)
	ol, _ := unlist(orig)
	if len(ll) > 1 && len(ol) == 1 {
		return
	}
	if lowList && len(ll) == len(res) && len(ol) == len(res) {
		for i := range ll {
			ll[i] = fl.wrapKnown(ll[i], fl.optType(ol[i]), res[i])
		}
		return
	}
	if len(ll) == 1 && len(res) == 1 && len(ol) == 1 {
		s.Results = fl.wrapKnown(low, fl.optType(orig), res[0])
	}
}

// lowerOptVarValues lowers a variable or constant declaration's values,
// wrapping each to its declared type, and returns the values to use.
// Typeless declarations infer and need nothing; tuple calls pass
// through.
func (fl *fileLower) lowerOptVarValues(names []*syntax.Name, typ, vals syntax.Expr) syntax.Expr {
	if vals == nil || typ == nil {
		return vals
	}
	vl, isList := unlist(vals)
	if len(vl) != len(names) {
		return fl.lowerOptExpr(vals)
	}
	targets := make([]types2.Type, len(names))
	for i, name := range names {
		if obj := fl.pkg.info.Defs[name]; obj != nil {
			switch obj := obj.(type) {
			case *types2.Var:
				targets[i] = obj.Type()
			case *types2.Const:
				targets[i] = obj.Type()
			}
		}
	}
	for i := range vl {
		nv := fl.lowerOptExpr(vl[i])
		nv = fl.wrapKnown(nv, fl.optType(vl[i]), targets[i])
		if isList {
			vals.(*syntax.ListExpr).ElemList[i] = nv
		} else {
			vals = nv
		}
	}
	return vals
}

// lowerOptSend lowers a channel send, wrapping the value to the channel
// element type.
func (fl *fileLower) lowerOptSend(s *syntax.SendStmt) {
	s.Chan = fl.lowerOptExpr(s.Chan)
	var elem types2.Type
	if ch, ok := types2.Unalias(fl.optType(s.Chan)).(*types2.Chan); ok {
		elem = ch.Elem()
	}
	s.Value = fl.wrapKnown(fl.lowerOptExpr(s.Value), fl.optType(s.Value), elem)
}

// structFieldType returns the type of the struct field named by key, or
// nil.
func structFieldType(st *types2.Struct, key syntax.Expr) types2.Type {
	n, ok := syntax.Unparen(key).(*syntax.Name)
	if !ok {
		return nil
	}
	for i := range st.NumFields() {
		if st.Field(i).Name() == n.Value {
			return st.Field(i).Type()
		}
	}
	return nil
}

// lowerOptComposite lowers a composite literal: elements lower and wrap
// to their field, key, or element type. Literals without recorded types
// (synthetic wraps) lower as reads.
func (fl *fileLower) lowerOptComposite(e *syntax.CompositeLit) syntax.Expr {
	t := fl.optType(e)
	if t == nil {
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
		return e
	}
	// A literal of a named type (or of a pointer to one, `&P{...}`)
	// carries the defined type, not the underlying composite: unwrap to
	// it so elements of named structs, maps, slices, and arrays wrap to
	// their option field, key, or element types (SPEC T-202).
	u := types2.Unalias(t)
	if p, ok := u.(*types2.Pointer); ok {
		u = types2.Unalias(p.Elem())
	}
	if n, ok := u.(*types2.Named); ok {
		u = n.Underlying()
	}
	switch c := u.(type) {
	case *types2.Struct:
		for i, x := range e.ElemList {
			if kv, ok := syntax.Unparen(x).(*syntax.KeyValueExpr); ok {
				if _, isName := syntax.Unparen(kv.Key).(*syntax.Name); isName {
					nv := fl.lowerOptExpr(kv.Value)
					kv.Value = fl.wrapKnown(nv, fl.optType(kv.Value), structFieldType(c, kv.Key))
					continue
				}
			}
			nv := fl.lowerOptExpr(x)
			var ft types2.Type
			if i < c.NumFields() {
				ft = c.Field(i).Type()
			}
			e.ElemList[i] = fl.wrapKnown(nv, fl.optType(x), ft)
		}
	case *types2.Map:
		for i, x := range e.ElemList {
			if kv, ok := syntax.Unparen(x).(*syntax.KeyValueExpr); ok {
				nk := fl.lowerOptExpr(kv.Key)
				nv := fl.lowerOptExpr(kv.Value)
				kv.Key = fl.wrapKnown(nk, fl.optType(kv.Key), c.Key())
				kv.Value = fl.wrapKnown(nv, fl.optType(kv.Value), c.Elem())
				continue
			}
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
	case *types2.Slice, *types2.Array:
		var elem types2.Type
		switch c := c.(type) {
		case *types2.Slice:
			elem = c.Elem()
		case *types2.Array:
			elem = c.Elem()
		}
		for i, x := range e.ElemList {
			nv := fl.lowerOptExpr(x)
			e.ElemList[i] = fl.wrapKnown(nv, fl.optType(x), elem)
		}
	default:
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
	}
	return e
}

// lowerOptRange lowers a range clause: the range operand and the write
// targets lower as reads. Element assignment needs no wrap: matching
// types already agree, and mismatches are clean recheck errors.
func (fl *fileLower) lowerOptRange(s *syntax.RangeClause) {
	s.X = fl.lowerOptExpr(s.X)
	if s.Lhs != nil {
		fl.lowerOptLHS(s.Lhs)
	}
}

// findCallParens returns the offsets of the parenthesis pair opening
// the call whose function name ends at selEnd: the first `(` after it
// and its match. Comments between the name and the parenthesis are
// skipped.
func (fl *fileLower) findCallParens(selEnd int) (lparen, rparen int) {
	lparen = -1
	scanSegment(fl.src, selEnd, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind != token.LPAREN {
			base.Fatalf("logos: optional call at offset %d has no (", selEnd)
		}
		lparen = t.start
		return false
	})
	if lparen < 0 {
		base.Fatalf("logos: optional call at offset %d has no (", selEnd)
	}
	depth := 0
	scanSegment(fl.src, lparen, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
			if depth == 0 {
				rparen = t.start
				return false
			}
		case token.SEMICOLON:
			if depth == 0 {
				base.Fatalf("logos: optional call at offset %d has no )", lparen)
			}
		}
		return true
	})
	if rparen <= lparen {
		base.Fatalf("logos: optional call at offset %d has no )", lparen)
	}
	return lparen, rparen
}

// findLbrack returns the offset of the `[` opening the index expression
// whose map starts at from and whose key starts at kStart: the last `[`
// token before kStart. Brackets inside the map operand stay balanced
// before it, and strings and comments never scan as brackets.
func (fl *fileLower) findLbrack(from, kStart int) int {
	lbrack := -1
	scanSegment(fl.src, from, kStart, func(t tok) bool {
		if !t.comment && t.kind == token.LBRACK {
			lbrack = t.start
		}
		return true
	})
	if lbrack < 0 {
		base.Fatalf("logos: index expression at offset %d has no [", from)
	}
	return lbrack
}

// findRbrack returns the offset of the `]` closing the index expression
// whose key starts at kStart: the first depth-zero closer. The key is
// complete and balanced, so any depth-zero closer must be that bracket.
func (fl *fileLower) findRbrack(kStart int) int {
	rbrack := -1
	depth := 0
	scanSegment(fl.src, kStart, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				if t.kind != token.RBRACK {
					base.Fatalf("logos: index expression at offset %d ends in %s", kStart, fl.src[t.start:t.end])
				}
				rbrack = t.start
				return false
			}
			depth--
		case token.SEMICOLON, token.EOF:
			base.Fatalf("logos: index expression at offset %d has no ]", kStart)
		}
		return true
	})
	if rbrack < 0 {
		base.Fatalf("logos: index expression at offset %d has no ]", kStart)
	}
	return rbrack
}

// exprEnd returns the offset just past the expression starting at from:
// the start of the first depth-zero token that cannot continue an
// expression. Brackets of any kind nest; `{` at depth zero ends (a
// composite literal directly under `??` spells past it in ejected text
// only; compiled output comes from the native tree).
func (fl *fileLower) exprEnd(from int) int {
	end := len(fl.src)
	depth := 0
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment {
			return true
		}
		if t.auto {
			end = t.start
			return false
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK:
			depth++
		case token.LBRACE:
			if depth == 0 {
				end = t.start
				return false
			}
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				end = t.start
				return false
			}
			depth--
		case token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
			if depth == 0 {
				end = t.start
				return false
			}
		case token.EOF:
			end = len(fl.src)
			return false
		}
		return true
	})
	return end
}

// exprEndNested is exprEnd with `{` nesting at depth zero: it ends past
// a composite literal in a wrap or fallback tail. Wrap sites never
// precede a block brace, so every depth-zero `{` there opens a literal.
func (fl *fileLower) exprEndNested(from int) int {
	end := len(fl.src)
	depth := 0
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment {
			return true
		}
		if t.auto {
			end = t.start
			return false
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				end = t.start
				return false
			}
			depth--
		case token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
			if depth == 0 {
				end = t.start
				return false
			}
		case token.EOF:
			end = len(fl.src)
			return false
		}
		return true
	})
	return end
}
