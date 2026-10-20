// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"internal/buildcfg"
)

// LoGo/s M14-S1 lowering for fixed-array SIMD (SPEC Sec 2.4,
// T-204.2): element-wise `+ - * / & | ^` on qualifying `[N]T` arrays
// and the `.sum() .min() .max() .dot()` reductions become immediately
// invoked function literals over plain Go, so native tree and eject
// text agree by construction.
//
// When the build has GOEXPERIMENT=simd on amd64 or arm64, element-wise
// operators lower through simd/archsimd (one load per operand, one
// method call, one store); otherwise they unroll to scalar lane
// expressions. On amd64 the vector body runs only when the archsimd X86
// feature check its methods need is true, with the scalar lane
// expressions as the else branch; one check guards each lowered
// operator, never a lane or a chunk. A GOAMD64 level that guarantees
// the feature drops the check, and arm64 has none because every arm64
// method is baseline NEON. Reductions always use the same fixed
// pairwise tree in both forms, giving bit-identical float results:
// `.dot()` composes the element-wise `*` lowering with a pairwise sum.
// Integer `/` has no archsimd method on either arch, and `*` has none
// on amd64 8-bit lanes or arm64 64-bit integer lanes (verified by
// direct audit of src/simd/archsimd/ops_amd64.go and ops_arm64.go);
// those operators take the scalar form.
//
// The pass runs after the statement lowering (no TryExpr may remain in
// an operand) and before tuples and lambdas (their text drains this
// pass's splices). Sites are collected up front and lowered innermost
// first, so an outer operand's text composes the inner replacement
// through textOf.

const logosSimdAlias = "_logos_simd"
const logosSimdPath = "simd/archsimd"

// logosSimdMethodNames mirrors the types2 reduction set.
var logosSimdMethodNames = map[string]bool{
	"sum": true,
	"min": true,
	"max": true,
	"dot": true,
}

// logosSimdBinOps is the set of element-wise operators.
var logosSimdBinOps = map[syntax.Operator]bool{
	syntax.Add: true,
	syntax.Sub: true,
	syntax.Mul: true,
	syntax.Div: true,
	syntax.And: true,
	syntax.Or:  true,
	syntax.Xor: true,
}

// simdOpText spells one element-wise operator.
func simdOpText(op syntax.Operator) string {
	switch op {
	case syntax.Add:
		return "+"
	case syntax.Sub:
		return "-"
	case syntax.Mul:
		return "*"
	case syntax.Div:
		return "/"
	case syntax.And:
		return "&"
	case syntax.Or:
		return "|"
	case syntax.Xor:
		return "^"
	}
	return ""
}

// simdOpMethod spells the archsimd method for one element-wise
// operator.
func simdOpMethod(op syntax.Operator) string {
	switch op {
	case syntax.Add:
		return "Add"
	case syntax.Sub:
		return "Sub"
	case syntax.Mul:
		return "Mul"
	case syntax.Div:
		return "Div"
	case syntax.And:
		return "And"
	case syntax.Or:
		return "Or"
	case syntax.Xor:
		return "Xor"
	}
	return ""
}

// simdLane describes one vector lane kind.
type simdLane struct {
	name    string // archsimd lane prefix: Float32, Int8, ...
	width   int64  // lane width in bytes
	isFloat bool
}

// simdLaneOf maps an element basic kind to its lane. int, uint, and
// uintptr ride the 64-bit lanes; the vector form only fires on amd64
// and arm64, which are both 64-bit targets.
func simdLaneOf(kind types2.BasicKind) (simdLane, bool) {
	switch kind {
	case types2.Int8:
		return simdLane{"Int8", 1, false}, true
	case types2.Uint8:
		return simdLane{"Uint8", 1, false}, true
	case types2.Int16:
		return simdLane{"Int16", 2, false}, true
	case types2.Uint16:
		return simdLane{"Uint16", 2, false}, true
	case types2.Int32:
		return simdLane{"Int32", 4, false}, true
	case types2.Uint32:
		return simdLane{"Uint32", 4, false}, true
	case types2.Int64, types2.Int:
		return simdLane{"Int64", 8, false}, true
	case types2.Uint64, types2.Uint, types2.Uintptr:
		return simdLane{"Uint64", 8, false}, true
	case types2.Float32:
		return simdLane{"Float32", 4, true}, true
	case types2.Float64:
		return simdLane{"Float64", 8, true}, true
	}
	return simdLane{}, false
}

// simdWordKind reports whether kind is `int`, `uint`, or `uintptr`,
// whose pointers are not the *[N]int64 or *[N]uint64 the archsimd
// loaders take.
func simdWordKind(kind types2.BasicKind) bool {
	return kind == types2.Int || kind == types2.Uint || kind == types2.Uintptr
}

// simdShape is a lowering-side array shape.
type simdShape struct {
	arr       *types2.Array
	lane      simdLane
	len       int64
	spell     string // spelling of the (possibly named) array type
	under     string // spelling of the underlying array type
	named     bool   // spell differs from under
	elem      string // spelling of the underlying element kind
	elemSpell string // spelling of the element type as written
	exact     bool   // element pointer is already *[N]<lane kind>
}

// simdShapeOf validates the lowering shape of at: a numeric lane, a
// power-of-two length of at least 1, and at most 64 bytes. The checker
// guarantees every accepted site qualifies; anything else is skipped
// so the recheck fails loudly instead of miscompiling.
func (fl *fileLower) simdShapeOf(at *types2.Array, typ types2.Type) *simdShape {
	b, ok := at.Elem().Underlying().(*types2.Basic)
	if !ok {
		return nil
	}
	lane, ok := simdLaneOf(b.Kind())
	if !ok {
		return nil
	}
	n := at.Len()
	if n < 1 || n&(n-1) != 0 || n*lane.width > 64 {
		return nil
	}
	spell, ok := fl.spellType(typ)
	if !ok {
		return nil
	}
	elem, ok := fl.spellType(b)
	if !ok {
		return nil
	}
	elemSpell, ok := fl.spellType(at.Elem())
	if !ok {
		return nil
	}
	under := fmt.Sprintf("[%d]%s", n, elem)
	// archsimd loaders take *[N]int64-style pointers: int, uint, uintptr,
	// and defined (non-alias) element types are distinct types of the same
	// size, so their pointers would not match and they keep the scalar
	// form. An alias is the type it names, so it resolves first (byte is
	// uint8; `type f32 = float32` is float32).
	_, predeclared := types2.Unalias(at.Elem()).(*types2.Basic)
	exact := predeclared && !simdWordKind(b.Kind())
	return &simdShape{arr: at, lane: lane, len: n, spell: spell, under: under, named: spell != under, elem: elem, elemSpell: elemSpell, exact: exact}
}

// simdVectorEnabled reports whether this build lowers through
// simd/archsimd.
func simdVectorEnabled() bool {
	return buildcfg.Experiment.SIMD && (buildcfg.GOARCH == "amd64" || buildcfg.GOARCH == "arm64")
}

// simdFeature is an amd64 archsimd feature level for one vector form:
// the X86 method the emitted calls require, and the GOAMD64 level that
// guarantees it, so the guard can be elided.
type simdFeature struct {
	check   string
	goamd64 int
}

// The three amd64 archsimd feature levels the element-wise methods
// carry (`CPU Feature:` lines of ops_amd64.go) and the GOAMD64 level
// that guarantees each: v3 implies AVX and AVX2, v4 implies exactly the
// F+CD+BW+DQ+VL set that X86.AVX512 reports (internal/cpu cpu_x86.go).
// The baseline feature is every arm64 form: each archsimd method there
// is NEON (ops_arm64.go `CPU Feature:` lines), which the arch always
// has, so its level 0 is below every GOAMD64 value.
var (
	simdFeatBaseline = simdFeature{}
	simdFeatAVX      = simdFeature{"AVX", 3}
	simdFeatAVX2     = simdFeature{"AVX2", 3}
	simdFeatAVX512   = simdFeature{"AVX512", 4}
)

// amd64Feature selects the feature one emitted archsimd call set
// needs, from the vector byte width, the lane class, and the operator.
// A 64-byte vector needs AVX512 for every lane and operator. Below
// that, float vectors need AVX at 16 and 32 bytes; integer vectors need
// AVX at 16 bytes and AVX2 at 32 bytes. A 64-bit integer Mul has no
// AVX or AVX2 form at any width and needs AVX512, while 64-bit float
// Mul stays on the width rule. The array loads and stores carry no
// `CPU Feature:` line and need no more than the width feature.
func amd64Feature(total int64, lane simdLane, op syntax.Operator) simdFeature {
	switch {
	case total == 64, op == syntax.Mul && lane.width == 8 && !lane.isFloat:
		return simdFeatAVX512
	case total == 32 && !lane.isFloat:
		return simdFeatAVX2
	}
	return simdFeatAVX
}

// simdNoIntMulWidth is the integer lane width in bytes that has no
// archsimd Mul method on each arch (ops_amd64.go and ops_arm64.go
// audit): 8-bit lanes on amd64, 64-bit lanes on arm64.
var simdNoIntMulWidth = map[string]int64{"amd64": 1, "arm64": 8}

// simdMethodExists reports whether archsimd has an element-wise method
// for op on the lane. Float lanes have every operator the lowering
// emits; integer lanes lack `/` everywhere and `*` at one width per
// arch.
func simdMethodExists(lane simdLane, op syntax.Operator) bool {
	if lane.isFloat {
		return true
	}
	return op != syntax.Div && (op != syntax.Mul || lane.width != simdNoIntMulWidth[buildcfg.GOARCH])
}

// simdAmd64Form is the amd64 vector form of a shape of total bytes: one
// full-width vector for 16, 32, and 64 byte shapes, with the feature its
// methods need.
func simdAmd64Form(sh *simdShape, op syntax.Operator, total int64) (vec string, chunks int64, feat simdFeature, ok bool) {
	if total != 16 && total != 32 && total != 64 {
		return "", 0, simdFeature{}, false
	}
	return fmt.Sprintf("%sx%d", sh.lane.name, sh.len), 1, amd64Feature(total, sh.lane, op), true
}

// simdArm64Form is the arm64 vector form of a shape of total bytes:
// 128-bit chunks, all baseline NEON.
func simdArm64Form(sh *simdShape, total int64) (vec string, chunks int64, feat simdFeature, ok bool) {
	if total < 16 || total%16 != 0 {
		return "", 0, simdFeature{}, false
	}
	per := int64(16 / sh.lane.width)
	return fmt.Sprintf("%sx%d", sh.lane.name, per), total / 16, simdFeatBaseline, true
}

// simdVecForm selects the vector form for one element-wise operator:
// the archsimd vector type name, chunk count, and the amd64 feature
// its methods need, or ok == false for the scalar form.
func simdVecForm(sh *simdShape, op syntax.Operator) (vec string, chunks int64, feat simdFeature, ok bool) {
	if !simdVectorEnabled() || !sh.exact || !simdMethodExists(sh.lane, op) {
		return "", 0, simdFeature{}, false
	}
	total := sh.len * sh.lane.width
	if buildcfg.GOARCH == "amd64" {
		return simdAmd64Form(sh, op, total)
	}
	return simdArm64Form(sh, total)
}

// simdGuardCond spells the runtime guard condition of one vector-form
// operator, or "" when no guard is needed: the build's GOAMD64 level
// guarantees the feature, or the feature is the baseline.
func (fl *fileLower) simdGuardCond(feat simdFeature) string {
	if buildcfg.GOAMD64 >= feat.goamd64 {
		return ""
	}
	return logosSimdAlias + ".X86." + feat.check + "()"
}

// simdScalarBody spells the unrolled scalar form of one element-wise
// operator over _logos_a and _logos_b.
func simdScalarBody(sh *simdShape, op syntax.Operator) string {
	var lanes []string
	for i := int64(0); i < sh.len; i++ {
		lanes = append(lanes, fmt.Sprintf("_logos_a[%d] %s _logos_b[%d]", i, simdOpText(op), i))
	}
	return "return " + sh.spell + "{" + strings.Join(lanes, ", ") + "}"
}

// simdSite is one collected array operator or reduction with its source
// spans: the whole replacement span and one span per operand. Spans are
// recorded before any lowering replaces nodes with normalized fragments,
// so an outer operand's text composes the inner replacement through
// textOf.
type simdSite struct {
	expr  syntax.Expr
	whole span
	ops   []span
}

// lowerSimd rewrites every array operator and reduction of one file,
// innermost first.
func (fl *fileLower) lowerSimd() {
	var sites []simdSite
	kept := fl.simdKeptParens()
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.Operation:
			if fl.isSimdBinary(n) {
				sites = append(sites, simdSite{n, fl.simdWhole(n, kept), []span{fl.simdSpan(n.X), fl.simdSpan(n.Y)}})
			}
		case *syntax.CallExpr:
			if fl.isSimdCall(n) {
				sites = append(sites, fl.simdCallSite(n, kept))
			}
		}
		return true
	})
	var done []span
	for i := len(sites) - 1; i >= 0; i-- {
		s := &sites[i]
		for j := range s.ops {
			s.ops[j] = simdHull(s.ops[j], done)
		}
		s.whole = simdHull(s.whole, done)
		switch n := s.expr.(type) {
		case *syntax.Operation:
			fl.lowerSimdBinary(n, *s)
		case *syntax.CallExpr:
			fl.lowerSimdCall(n, *s)
		}
		done = append(done, s.whole)
	}
}

// simdHull expands s to cover every completed whole span overlapping
// it, so an outer operand's text composes inner replacements through
// textOf. Completed spans run innermost first, so only descendants fuse;
// disjoint siblings never overlap.
func simdHull(s span, done []span) span {
	for _, w := range done {
		if w.start < s.end && w.end > s.start {
			if w.start < s.start {
				s.start = w.start
			}
			if w.end > s.end {
				s.end = w.end
			}
		}
	}
	return s
}

// simdCallSite collects one array reduction with its operand spans: the
// receiver, plus the argument of dot.
func (fl *fileLower) simdCallSite(c *syntax.CallExpr, kept map[int]bool) simdSite {
	sel := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
	ops := []span{fl.simdSpan(sel.X)}
	if sel.Sel.Value == "dot" && len(c.ArgList) == 1 {
		ops = append(ops, fl.simdSpan(c.ArgList[0]))
	}
	return simdSite{c, fl.simdWhole(c, kept), ops}
}

// simdKeptParens records the offsets of parentheses the parser kept as
// ParenExpr nodes: conditions, go and defer calls, and composite literal
// types keep theirs, so spans must leave them in place to agree with the
// native tree.
func (fl *fileLower) simdKeptParens() map[int]bool {
	kept := make(map[int]bool)
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if p, ok := n.(*syntax.ParenExpr); ok {
			kept[fl.off(p.Pos())] = true
		}
		return true
	})
	return kept
}

// simdSpan reports the exact source span of a simd operand: the node
// span closed by delimiter scan, with no wrapper extension. Dropped
// wrappers stay in the enclosing whole span; hulling the operand with
// completed inner wholes at lowering time keeps nested replacements
// composable through textOf.
func (fl *fileLower) simdSpan(e syntax.Expr) span {
	return span{fl.off(syntax.StartPos(syntax.Unparen(e))), fl.simdExprEnd(e)}
}

// simdWhole reports the balanced replacement span of a simd site: the
// exact node span extended over grouping parentheses the parser
// dropped, so the eject splice stays balanced. Kept parentheses stay
// out: the native tree carries them, so the text must too. A call's own
// parentheses never join the span: the start stops at call openers, and
// only closers whose opener sits inside join the end.
func (fl *fileLower) simdWhole(e syntax.Expr, kept map[int]bool) span {
	start := simdBackStart(fl.src, fl.off(syntax.StartPos(e)), kept)
	return span{start, simdFwdEnd(fl.src, start, fl.simdExprEnd(e))}
}

// simdBackStart extends start backward over grouping parentheses the
// parser dropped: a `(` following a value opens a call, conversion, or
// index, never a dropped wrapper, so the walk stops there, as it does
// at every parenthesis the parser kept.
func simdBackStart(src []byte, start int, kept map[int]bool) int {
	for {
		prev, ok := simdPrevTok(src, start)
		if !ok || prev.kind != token.LPAREN || kept[prev.start] || simdCallOpen(src, prev.start) {
			return start
		}
		start = prev.start
	}
}

// simdCallOpen reports whether the `(` at open opens a call or
// conversion: it follows a value, so it cannot be a dropped wrapper.
func simdCallOpen(src []byte, open int) bool {
	prev, ok := simdPrevTok(src, open)
	if !ok {
		return false
	}
	switch prev.kind {
	case token.IDENT, token.RPAREN, token.RBRACK, token.RBRACE,
		token.INT, token.FLOAT, token.IMAG, token.CHAR, token.STRING:
		return true
	}
	return false
}

// simdFwdEnd extends end forward over the `)` closers of wrappers
// simdBackStart consumed: a closer belongs to the span exactly when its
// opener sits at or after start.
func simdFwdEnd(src []byte, start, end int) int {
	for {
		next, ok := simdNextTok(src, end)
		if !ok || next.kind != token.RPAREN || simdMatchParen(src, next.end) < start {
			return end
		}
		end = next.end
	}
}

// simdMatchParen returns the offset of the `(` matching the closer
// ending at closeEnd: the last unmatched opener before it.
func simdMatchParen(src []byte, closeEnd int) int {
	var opens []int
	match := -1
	scanSegment(src, 0, closeEnd, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN:
			opens = append(opens, t.start)
		case token.RPAREN:
			if len(opens) > 0 {
				last := opens[len(opens)-1]
				opens = opens[:len(opens)-1]
				if t.end == closeEnd {
					match = last
				}
			}
		}
		return true
	})
	if match < 0 {
		base.Fatalf("logos: parenthesis closing at offset %d has no (", closeEnd)
	}
	return match
}

// simdPrevTok returns the last code token before off.
func simdPrevTok(src []byte, off int) (tok, bool) {
	var prev tok
	found := false
	scanSegment(src, 0, off, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		prev, found = t, true
		return true
	})
	return prev, found
}

// simdNextTok returns the first code token at or after off.
func simdNextTok(src []byte, off int) (tok, bool) {
	var next tok
	found := false
	scanSegment(src, off, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		next, found = t, true
		return false
	})
	return next, found
}

// simdExprEnd reports the offset just past e: a binary operation ends at
// its right operand, a call at its closing parenthesis, a composite
// literal past its closing brace, an index past its closing bracket, and
// an assertion past the closing parenthesis of its type.
func (fl *fileLower) simdExprEnd(e syntax.Expr) int {
	switch n := syntax.Unparen(e).(type) {
	case *syntax.Operation:
		if n.Y != nil {
			return fl.simdExprEnd(n.Y)
		}
	case *syntax.CallExpr:
		return fl.chainCallEnd(e)
	case *syntax.CompositeLit:
		return fl.off(n.Rbrace) + 1
	case *syntax.IndexExpr:
		return fl.findRbrack(fl.off(syntax.StartPos(n.Index))) + 1
	case *syntax.AssertExpr:
		return fl.simdAssertEnd(n)
	}
	return fl.off(syntax.EndPos(e))
}

// simdAssertEnd reports the offset just past a type assertion: the exact
// end of its asserted expression, then `.`, `(`, the asserted type, and
// its closing parenthesis.
func (fl *fileLower) simdAssertEnd(a *syntax.AssertExpr) int {
	end := fl.simdExprEnd(a.X)
	seen := 0
	depth := 0
	scanSegment(fl.src, end, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch seen {
		case 0:
			if t.kind != token.PERIOD {
				base.Fatalf("logos: assertion at offset %d has no .", end)
			}
			seen = 1
		case 1:
			if t.kind != token.LPAREN {
				base.Fatalf("logos: assertion at offset %d has no (", end)
			}
			seen = 2
			depth = 1
		default:
			switch t.kind {
			case token.LPAREN, token.LBRACK, token.LBRACE:
				depth++
			case token.RPAREN, token.RBRACK, token.RBRACE:
				depth--
				if depth == 0 {
					end = t.end
					return false
				}
			case token.SEMICOLON, token.EOF:
				base.Fatalf("logos: assertion at offset %d has no )", end)
			}
		}
		return true
	})
	return end
}

// isSimdBinary reports whether e is a checker-accepted array operator.
// No upstream binary operator produces an array, so an array-typed
// element-wise binary expression is always ours.
func (fl *fileLower) isSimdBinary(e *syntax.Operation) bool {
	if !logosSimdBinOps[e.Op] || e.Y == nil {
		return false
	}
	t := fl.pkg.info.TypeOf(e)
	if t == nil {
		return false
	}
	at, ok := t.Underlying().(*types2.Array)
	if !ok {
		return false
	}
	return fl.simdShapeOf(at, t) != nil
}

// isSimdCall reports whether c is a checker-accepted array reduction.
// A user-defined method of the same name wins in the checker, so the
// method sets distinguish it here.
func (fl *fileLower) isSimdCall(c *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil || !logosSimdMethodNames[sel.Sel.Value] {
		return false
	}
	t := fl.pkg.info.TypeOf(sel.X)
	if t == nil {
		return false
	}
	at, ok := t.Underlying().(*types2.Array)
	if !ok || fl.simdShapeOf(at, t) == nil {
		return false
	}
	// A user-defined method of the same name wins in the checker, which
	// records it in info.Selections; a reduction records nothing.
	if fl.pkg.info.Selections != nil {
		if _, ok := fl.pkg.info.Selections[sel]; ok {
			return false
		}
	} else if obj, _, _ := types2.LookupFieldOrMethod(t, true, fl.pkg.tpkg, sel.Sel.Value); obj != nil {
		return false
	}
	return true
}

// simdAddr spells a pointer to the array value v of shape sh for a
// vector load or store: a plain address, or a conversion when the
// value's type is named.
func simdAddr(sh *simdShape, v string) string {
	if !sh.named {
		return "&" + v
	}
	return "(*" + sh.under + ")(&" + v + ")"
}

// simdChunkAddr spells a pointer to chunk c (k lanes) of the array
// value v for the arm64 chunked form, converting the addressable
// slice to an array pointer.
func simdChunkAddr(sh *simdShape, v string, c, k int64) string {
	lo, hi := c*k, (c+1)*k
	return fmt.Sprintf("(*[%d]%s)(%s[%d:%d])", k, sh.elem, v, lo, hi)
}

// simdElemwiseText builds the IIFE text computing aText op bText over
// shape sh, in vector form when the build selects it.
func (fl *fileLower) simdElemwiseText(sh *simdShape, op syntax.Operator, aText, bText string) string {
	params := "_logos_a, _logos_b " + sh.spell
	if vec, chunks, feat, ok := simdVecForm(sh, op); ok {
		load := logosSimdAlias + ".Load" + vec + "Array"
		method := simdOpMethod(op)
		var stmts []string
		stmts = append(stmts, "var _logos_r "+sh.spell)
		if chunks == 1 {
			stmts = append(stmts, load+"("+simdAddr(sh, "_logos_a")+")."+method+"("+load+"("+simdAddr(sh, "_logos_b")+")).StoreArray("+simdAddr(sh, "_logos_r")+")")
		} else {
			k := sh.len / chunks
			chunkVec := fmt.Sprintf("%sx%d", sh.lane.name, 16/sh.lane.width)
			chunkLoad := logosSimdAlias + ".Load" + chunkVec + "Array"
			for c := int64(0); c < chunks; c++ {
				stmts = append(stmts, chunkLoad+"("+simdChunkAddr(sh, "_logos_a", c, k)+")."+method+"("+chunkLoad+"("+simdChunkAddr(sh, "_logos_b", c, k)+")).StoreArray("+simdChunkAddr(sh, "_logos_r", c, k)+")")
			}
		}
		stmts = append(stmts, "return _logos_r")
		fl.needSimd = true
		vecBody := strings.Join(stmts, "; ")
		if cond := fl.simdGuardCond(feat); cond != "" {
			return "func(" + params + ") " + sh.spell + " { if " + cond + " { " + vecBody + " } else { " + simdScalarBody(sh, op) + " } }(" + aText + ", " + bText + ")"
		}
		return "func(" + params + ") " + sh.spell + " { " + vecBody + " }(" + aText + ", " + bText + ")"
	}
	return "func(" + params + ") " + sh.spell + " { " + simdScalarBody(sh, op) + " }(" + aText + ", " + bText + ")"
}

// simdReduceBody folds the lanes of v in the fixed pairwise tree, one
// temporary per combined pair, so the order is the same in every form.
// The body has no parentheses: the logos tuple parser reads
// `(a + b) + (c + d)` as a tuple, so the tree is spelled as statements.
func simdReduceBody(v string, n int64, combine func(a, b string) string) string {
	level := make([]string, 0, n)
	for i := int64(0); i < n; i++ {
		level = append(level, fmt.Sprintf("%s[%d]", v, i))
	}
	var stmts []string
	for depth := 0; len(level) > 1; depth++ {
		next := make([]string, 0, len(level)/2)
		for i := 0; i < len(level); i += 2 {
			name := fmt.Sprintf("_logos_t%d_%d", depth, i/2)
			stmts = append(stmts, name+" := "+combine(level[i], level[i+1]))
			next = append(next, name)
		}
		level = next
	}
	stmts = append(stmts, "return "+level[0])
	return strings.Join(stmts, "; ")
}

// simdSumBody is the pairwise sum over the lanes of v.
func simdSumBody(v string, n int64) string {
	return simdReduceBody(v, n, func(a, b string) string { return a + " + " + b })
}

// simdMinMaxBody is the pairwise min or max over the lanes of v.
func simdMinMaxBody(v string, n int64, method string) string {
	return simdReduceBody(v, n, func(a, b string) string { return method + "(" + a + ", " + b + ")" })
}

// lowerSimdBinary rewrites one array operator to an IIFE.
func (fl *fileLower) lowerSimdBinary(e *syntax.Operation, s simdSite) {
	t := fl.pkg.info.TypeOf(e)
	if t == nil {
		return
	}
	at, ok := t.Underlying().(*types2.Array)
	if !ok {
		return
	}
	sh := fl.simdShapeOf(at, t)
	if sh == nil {
		return
	}
	aText := fl.textOf(s.ops[0])
	bText := fl.textOf(s.ops[1])
	text := fl.simdElemwiseText(sh, e.Op, aText, bText)
	fl.replaceSimd(e, s.whole, text)
}

// lowerSimdCall rewrites one array reduction to an IIFE.
func (fl *fileLower) lowerSimdCall(c *syntax.CallExpr, s simdSite) {
	sel := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
	t := fl.pkg.info.TypeOf(sel.X)
	if t == nil {
		return
	}
	at, ok := t.Underlying().(*types2.Array)
	if !ok {
		return
	}
	sh := fl.simdShapeOf(at, t)
	if sh == nil {
		return
	}
	method := sel.Sel.Value
	recvText := fl.textOf(s.ops[0])
	var text string
	switch method {
	case "sum":
		text = "func(_logos_a " + sh.spell + ") " + sh.elemSpell + " { " + simdSumBody("_logos_a", sh.len) + " }(" + recvText + ")"
	case "min", "max":
		text = "func(_logos_a " + sh.spell + ") " + sh.elemSpell + " { " + simdMinMaxBody("_logos_a", sh.len, method) + " }(" + recvText + ")"
	case "dot":
		if len(c.ArgList) != 1 {
			return
		}
		argText := fl.textOf(s.ops[1])
		inner := fl.simdElemwiseText(sh, syntax.Mul, recvText, argText)
		text = "func(_logos_p " + sh.spell + ") " + sh.elemSpell + " { " + simdSumBody("_logos_p", sh.len) + " }(" + inner + ")"
	default:
		return
	}
	fl.replaceSimd(c, s.whole, text)
}

// replaceSimd parses text as the replacement expression and swaps it
// for old, recording the matching eject splice over whole.
func (fl *fileLower) replaceSimd(old syntax.Expr, whole span, text string) {
	frag := parseSimdExprFrag(text, old.Pos())
	replaceTupleExpr(fl.file, old, frag)
	fl.add(whole.start, whole.end, text)
}

// parseSimdExprFrag parses text as one expression and normalizes every
// position to pos, mirroring the chain fragment parser. A block's
// closing brace is a bare position, not a node, so the guard's nested
// blocks keep fragment coordinates after the node walk; the DWARF scope
// marker pops each block at that position and rejects one that sorts
// before the block start, so every closing brace moves to pos too.
func parseSimdExprFrag(text string, pos syntax.Pos) syntax.Expr {
	f, err := syntax.Parse(syntax.NewFileBase("logos-simd"), strings.NewReader("package logossimd\nvar _ = "+text+"\n"), nil, nil, 0)
	if err != nil || len(f.DeclList) != 1 {
		base.Fatalf("logos: cannot parse simd fragment %q: %v", text, err)
	}
	vd, ok := f.DeclList[0].(*syntax.VarDecl)
	if !ok {
		base.Fatalf("logos: cannot parse simd fragment %q", text)
	}
	vals := syntax.UnpackListExpr(vd.Values)
	if len(vals) != 1 {
		base.Fatalf("logos: cannot parse simd fragment %q", text)
	}
	e := vals[0]
	normalizeChainPos(e, pos)
	return e
}

// simdImport builds the implicit simd/archsimd import of lowered files.
func simdImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, logosSimdAlias),
		Path:         litOf(pos, fmt.Sprintf("%q", logosSimdPath), syntax.StringLit),
	}, pos)
}
