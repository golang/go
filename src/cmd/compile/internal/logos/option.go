// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"reflect"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M3-S6 nodes (SPEC Sec 1.4, T-104.3 and Sec 2.2,
// T-202): a single-value map read `m[k]` typed `V?` and `??` over such a
// read. Every other option form (`?.`, explicit `T?` declarations and
// results, `??` over non-map options) is left in place; like any option
// program outside this slice it still fails downstream, exactly as M3-S2
// left it.
//
// A single-value read keeps the plain `m[k]` form when V is a pointer or
// an interface-backed union with no nil-capable member (nil as none, zero
// overhead); every other V becomes `_logos_rt.Lookup(m, k)`, whose
// `Option[V]` result carries presence apart from the payload so a
// present nil stays distinguishable from absent. `m[k] ?? d` becomes an
// immediately invoked function literal that tests presence once: the
// `.Present`/`.Value` form over a Lookup call, or a comma-ok read for a
// kept map read (comma-ok keeps Go types through the recheck, where a
// bare kept read would rewrap as `V?` and no longer assign to `V`).
//
// The checker re-records comma-ok reads, store targets, and `++`/`--`
// and compound-store operands with the plain element type, so this pass
// only ever sees the single-value readable form: `info.TypeOf` on any
// surviving `*syntax.IndexExpr` is `*types2.Option` exactly for the reads
// to lower. The pass runs after lowerUnions and before the statement
// lowering, so `try` templates only ever copy already-lowered
// expressions.
//
// Every rewrite builds the native tree and the eject text together from
// small token-anchored splices, like the union lowering: an insertion
// before the map, a replacement of `[` with `, `, and a replacement of
// `]` with `)` for Lookup, plus wrapper insertions and the `??` deletion
// for `??`. A
// consumed map read emits no splice of its own; the outer `??` splices
// fuse the Lookup head. Nested rewrites sit strictly inside the outer
// brackets, so no two splices overlap. Only the end of the `??` fallback
// is heuristic (a scan to the next token that cannot continue an
// expression); every other boundary is an exact token. Exotic fallbacks
// (a composite literal or a `:` slice bound directly under `??`) may
// place the closing `}()` early in ejected text only; compiled output
// always comes from the native tree.

const (
	// optTemp is the value temporary of the lowered `??` wrappers. Each
	// wrapper is a fresh function literal, and the lookup form nests any
	// fallback outside the `if` scope, so one spelling cannot capture.
	optTemp = "v"
	// optOk is the presence temporary of the comma-ok `??` wrapper.
	optOk = "ok"
)

// exprType is the static type of expression slots the option pass
// rewrites: plain, listed, and fixed-array expression fields.
var exprType = reflect.TypeFor[syntax.Expr]()

// lowerOptions rewrites the single-value map reads and map-read `??`
// forms of one file. It runs after lowerUnions, then drains the
// synthesized imports its result-type spellings may have recorded. The
// walk is syntax.Inspect, which follows true child edges only, so type
// info, positions, and scopes attached to nodes never lead the pass;
// each visited node rewrites the option forms in its own expression
// slots, and Inspect descends into the replacements afterwards, which
// lowers nested forms inside already-lowered parents.
func (fl *fileLower) lowerOptions() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		fl.lowerOptFields(n)
		return true
	})
	fl.flushTypeImports()
}

// lowerOptFields rewrites the option forms held directly by n: every
// field of static type syntax.Expr, []syntax.Expr, or [N]syntax.Expr is
// checked for a *syntax.CoalesceExpr or *syntax.IndexExpr and replaced
// with its lowering. All other fields are left alone; Inspect visits the
// nodes they hold.
func (fl *fileLower) lowerOptFields(n syntax.Node) {
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
		fl.lowerOptField(v.Field(i), t.Field(i).Type)
	}
}

// lowerOptField rewrites one field of static type ft when it holds a
// *syntax.CoalesceExpr or *syntax.IndexExpr.
func (fl *fileLower) lowerOptField(v reflect.Value, ft reflect.Type) {
	if ft == exprType {
		if v.IsNil() {
			return
		}
		switch e := v.Interface().(syntax.Expr).(type) {
		case *syntax.CoalesceExpr:
			v.Set(reflect.ValueOf(fl.lowerCoalesce(e)))
		case *syntax.IndexExpr:
			v.Set(reflect.ValueOf(fl.lowerMapIndex(e)))
		}
		return
	}
	k := ft.Kind()
	if (k == reflect.Slice || k == reflect.Array) && ft.Elem() == exprType {
		for i := range v.Len() {
			ev := v.Index(i)
			if ev.IsNil() {
				continue
			}
			switch e := ev.Interface().(syntax.Expr).(type) {
			case *syntax.CoalesceExpr:
				ev.Set(reflect.ValueOf(fl.lowerCoalesce(e)))
			case *syntax.IndexExpr:
				ev.Set(reflect.ValueOf(fl.lowerMapIndex(e)))
			}
		}
	}
}

// optType reports the checked type of e from the pre-lowering check, or
// nil when e is synthetic (already lowered) or untyped.
func (fl *fileLower) optType(e syntax.Expr) types2.Type {
	if e == nil {
		return nil
	}
	return fl.pkg.info.TypeOf(e)
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

// keepMapRead reports whether a single-value read of element type V keeps
// the plain `m[k]` form (SPEC T-202): V is a pointer, or V is an
// interface-backed union with no nil-capable member. Every other V lowers
// to a Lookup call. Nil-capable means nil is a valid payload: slice, map,
// func, chan, or interface kinds (plus type parameters, nested unions,
// and unsafe pointers, conservatively).
func keepMapRead(V types2.Type) bool {
	if V == nil {
		return false
	}
	u := types2.Unalias(V)
	if _, ok := u.(*types2.Pointer); ok {
		return true
	}
	if _, ok := u.(*types2.Interface); !ok {
		return false
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

// lowerMapIndex lowers one standalone single-value map read: a kept read
// stays, every other option-typed read becomes a Lookup call. Reads the
// checker re-recorded with the element type (comma-ok, store targets,
// `++`) carry no option and are left alone. Nested forms inside the map
// and key lower when Inspect descends into the result.
func (fl *fileLower) lowerMapIndex(e *syntax.IndexExpr) syntax.Expr {
	opt := fl.asOption(fl.optType(e))
	if opt == nil || keepMapRead(opt.Elem) {
		return e
	}
	return fl.lookupCall(e, rtAlias+".Lookup(")
}

// lookupCall builds `_logos_rt.Lookup(m, k)` for the read e and records
// its eject splices: head inserts before the map, `[` becomes `, `, and
// `]` becomes `)`. head carries the insertion text so a fused `??`
// wrapper can extend it.
func (fl *fileLower) lookupCall(e *syntax.IndexExpr, head string) syntax.Expr {
	pos := e.Pos()
	fl.needRT = true
	call := callOf(pos, selOf(pos, nameOf(pos, rtAlias), "Lookup"), []syntax.Expr{e.X, e.Index})
	mStart := fl.off(leftmostOf(e.X))
	kStart := fl.off(leftmostOf(e.Index))
	lbrack := fl.findLbrack(mStart, kStart)
	rbrack := fl.findRbrack(kStart)
	fl.add(mStart, mStart, head)
	fl.add(lbrack, lbrack+1, ", ")
	fl.add(rbrack, rbrack+1, ")")
	return call
}

// lowerCoalesce lowers `x ?? y`. A single-value map read on the left fuses
// with its own lowering: over a Lookup-lowered read the wrapper tests
// `.Present`, over a kept read it reads comma-ok. Any other left side of
// option type with a non-pointer element lowers to the presence-testing
// wrapper, which the recheck sees as the named support struct (a variable
// bound to a lowered read, transitively). Nested forms inside the
// operands lower when Inspect descends into the result. A pointer-element
// left side that is not a map read is left in place, like every option
// form outside this slice.
func (fl *fileLower) lowerCoalesce(e *syntax.CoalesceExpr) syntax.Expr {
	opt := fl.asOption(fl.optType(e.X))
	ux, isIndex := syntax.Unparen(e.X).(*syntax.IndexExpr)
	if opt == nil && isIndex {
		opt = fl.asOption(fl.optType(ux))
	}
	if opt == nil || (!isIndex && keepMapRead(opt.Elem)) {
		return e
	}
	pos := e.Pos()
	etext, ok := fl.spellResultType(opt.Elem, pos)
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
	yEnd := fl.exprEnd(fl.off(leftmostOf(e.Y)))
	fl.add(qoff, qoff+2, "")
	fl.add(yEnd, yEnd, " }()")
	if isIndex {
		xStart := fl.off(leftmostOf(ux.X))
		xEnd := fl.findRbrack(fl.off(leftmostOf(ux.Index))) + 1
		if keepMapRead(opt.Elem) {
			// func() E { if v, ok := m[k]; ok { return v }; return y }()
			init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp), nameOf(pos, optOk)}, []syntax.Expr{ux})
			body := []syntax.Stmt{
				ifOf(pos, init, nameOf(pos, optOk), []syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, optTemp)})}),
				returnOf(pos, []syntax.Expr{e.Y}),
			}
			lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
			fl.add(xStart, xStart, "func() "+etext+" { if "+optTemp+", "+optOk+" := ")
			fl.add(xEnd, xEnd, "; "+optOk+" { return "+optTemp+" }; return ")
			return callOf(pos, lit, nil)
		}
		// func() E { if v := Lookup(m, k); v.Present { return v.Value }; return y }()
		call := fl.lookupCall(ux, "func() "+etext+" { if "+optTemp+" := "+rtAlias+".Lookup(")
		init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{call})
		cond := selOf(pos, nameOf(pos, optTemp), "Present")
		then := []syntax.Stmt{returnOf(pos, []syntax.Expr{selOf(pos, nameOf(pos, optTemp), "Value")})}
		body := []syntax.Stmt{
			ifOf(pos, init, cond, then),
			returnOf(pos, []syntax.Expr{e.Y}),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		// The fused Lookup head left its call open and lookupCall recorded
		// the closing `)`; the mid only adds the presence test.
		fl.add(xEnd, xEnd, "; "+optTemp+".Present { return "+optTemp+".Value }; return ")
		return callOf(pos, lit, nil)
	}
	// A non-map option left side with a non-pointer element: the recheck
	// sees the lowered operand as the named support struct.
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{e.X})
	cond := selOf(pos, nameOf(pos, optTemp), "Present")
	then := []syntax.Stmt{returnOf(pos, []syntax.Expr{selOf(pos, nameOf(pos, optTemp), "Value")})}
	body := []syntax.Stmt{
		ifOf(pos, init, cond, then),
		returnOf(pos, []syntax.Expr{e.Y}),
	}
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
	xStart := fl.off(leftmostOf(e.X))
	xEnd := lastTokenEnd(fl.src, xStart, qoff)
	fl.add(xStart, xStart, "func() "+etext+" { if "+optTemp+" := ")
	fl.add(xEnd, xEnd, "; "+optTemp+".Present { return "+optTemp+".Value }; return ")
	return callOf(pos, lit, nil)
}

// spellResultType renders the `??` result type E as Go source in this
// file: a nested option becomes the named support instantiation, every
// other type spells directly.
func (fl *fileLower) spellResultType(E types2.Type, pos syntax.Pos) (string, bool) {
	if o := fl.asOption(E); o != nil {
		inner, ok := fl.spellResultType(o.Elem, pos)
		if !ok {
			return "", false
		}
		fl.needRT = true
		return rtAlias + ".Option[" + inner + "]", true
	}
	return fl.spellType(E)
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
