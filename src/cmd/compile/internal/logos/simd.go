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

const logosSimdAlias = "_logos_simd"
const logosSimdPath = "simd/archsimd"
const logosSimdMathAlias = "_logos_math"
const logosSimdMathPath = "math"

type simdOp struct {
	text   string
	method string
}

var simdOps = map[syntax.Operator]simdOp{
	syntax.Add: {"+", "Add"},
	syntax.Sub: {"-", "Sub"},
	syntax.Mul: {"*", "Mul"},
	syntax.Div: {"/", "Div"},
	syntax.And: {"&", "And"},
	syntax.Or:  {"|", "Or"},
	syntax.Xor: {"^", "Xor"},
}

type simdLane struct {
	name    string
	width   int64
	isFloat bool
}

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

func simdWordKind(kind types2.BasicKind) bool {
	return kind == types2.Int || kind == types2.Uint || kind == types2.Uintptr
}

type simdShape struct {
	arr       *types2.Array
	lane      simdLane
	len       int64
	spell     string
	under     string
	named     bool
	elem      string
	elemSpell string
	exact     bool
}

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
	_, predeclared := types2.Unalias(at.Elem()).(*types2.Basic)
	exact := predeclared && !simdWordKind(b.Kind())
	return &simdShape{arr: at, lane: lane, len: n, spell: spell, under: under, named: spell != under, elem: elem, elemSpell: elemSpell, exact: exact}
}

func (fl *fileLower) simdArrayShape(t types2.Type) *simdShape {
	if t == nil {
		return nil
	}
	t = unaliasConstView(t)
	at, ok := t.Underlying().(*types2.Array)
	if !ok {
		return nil
	}
	return fl.simdShapeOf(at, t)
}

func simdVectorEnabled() bool {
	return buildcfg.Experiment.SIMD && (buildcfg.GOARCH == "amd64" || buildcfg.GOARCH == "arm64")
}

type simdFeature struct {
	check   string
	goamd64 int
}

var (
	simdFeatBaseline = simdFeature{}
	simdFeatAVX      = simdFeature{"AVX", 3}
	simdFeatAVX2     = simdFeature{"AVX2", 3}
	simdFeatAVX512   = simdFeature{"AVX512", 4}
)

func amd64Feature(total int64, lane simdLane, op syntax.Operator) simdFeature {
	switch {
	case total == 64, op == syntax.Mul && lane.width == 8 && !lane.isFloat:
		return simdFeatAVX512
	case total == 32 && (!lane.isFloat || simdFloatBitwiseOp(op)):
		return simdFeatAVX2
	}
	return simdFeatAVX
}

var simdNoIntMulWidth = map[string]int64{"amd64": 1, "arm64": 8}

func simdMethodExists(lane simdLane, op syntax.Operator) bool {
	if lane.isFloat {
		return true
	}
	return op != syntax.Div && (op != syntax.Mul || lane.width != simdNoIntMulWidth[buildcfg.GOARCH])
}

func simdAmd64Form(sh *simdShape, op syntax.Operator, total int64) (vec string, chunks int64, feat simdFeature, ok bool) {
	if total != 16 && total != 32 && total != 64 {
		return "", 0, simdFeature{}, false
	}
	return fmt.Sprintf("%sx%d", sh.lane.name, sh.len), 1, amd64Feature(total, sh.lane, op), true
}

func simdArm64Form(sh *simdShape, total int64) (vec string, chunks int64, feat simdFeature, ok bool) {
	if total < 16 || total%16 != 0 {
		return "", 0, simdFeature{}, false
	}
	per := int64(16 / sh.lane.width)
	return fmt.Sprintf("%sx%d", sh.lane.name, per), total / 16, simdFeatBaseline, true
}

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

func (fl *fileLower) simdGuardCond(feat simdFeature) string {
	if buildcfg.GOAMD64 >= feat.goamd64 {
		return ""
	}
	return logosSimdAlias + ".X86." + feat.check + "()"
}

func simdFloatBitwiseOp(op syntax.Operator) bool {
	return op == syntax.And || op == syntax.Or || op == syntax.Xor
}

func simdFloatBitsTo(sh *simdShape) string {
	if sh.lane.width == 4 {
		return "BitsToFloat32"
	}
	return "BitsToFloat64"
}

func (fl *fileLower) simdScalarBody(sh *simdShape, op syntax.Operator) string {
	laneText := func(i int64) string {
		return fmt.Sprintf("_logos_a[%d] %s _logos_b[%d]", i, simdOps[op].text, i)
	}
	if sh.lane.isFloat && op == syntax.Mul {
		laneText = func(i int64) string {
			return fmt.Sprintf("%s(_logos_a[%d] * _logos_b[%d])", sh.elemSpell, i, i)
		}
	}
	if sh.lane.isFloat && simdFloatBitwiseOp(op) {
		fl.needSimdMath = true
		bits := "Float64bits"
		from := "Float64frombits"
		if sh.lane.width == 4 {
			bits = "Float32bits"
			from = "Float32frombits"
		}
		laneText = func(i int64) string {
			return fmt.Sprintf("%s.%s(%s.%s(_logos_a[%d]) %s %s.%s(_logos_b[%d]))", logosSimdMathAlias, from, logosSimdMathAlias, bits, i, simdOps[op].text, logosSimdMathAlias, bits, i)
		}
	}
	var lanes []string
	for i := int64(0); i < sh.len; i++ {
		lanes = append(lanes, laneText(i))
	}
	return "return " + sh.spell + "{" + strings.Join(lanes, ", ") + "}"
}

func simdVecSuffix(sh *simdShape, op syntax.Operator) (string, string) {
	if sh.lane.isFloat && simdFloatBitwiseOp(op) {
		return ".ToBits()", "." + simdFloatBitsTo(sh) + "().StoreArray"
	}
	return "", ".StoreArray"
}

type simdSite struct {
	expr  syntax.Expr
	whole span
	ops   []span
}

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

func (fl *fileLower) simdCallSite(c *syntax.CallExpr, kept map[int]bool) simdSite {
	sel := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
	ops := []span{fl.simdSpan(sel.X)}
	if sel.Sel.Value == "dot" && len(c.ArgList) == 1 {
		ops = append(ops, fl.simdSpan(c.ArgList[0]))
	}
	return simdSite{c, fl.simdWhole(c, kept), ops}
}

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

func (fl *fileLower) simdSpan(e syntax.Expr) span {
	return span{fl.off(syntax.StartPos(syntax.Unparen(e))), fl.simdExprEnd(e)}
}

func (fl *fileLower) simdWhole(e syntax.Expr, kept map[int]bool) span {
	start := simdBackStart(fl.src, fl.off(syntax.StartPos(e)), kept)
	return span{start, simdFwdEnd(fl.src, start, fl.simdExprEnd(e))}
}

func simdBackStart(src []byte, start int, kept map[int]bool) int {
	for {
		prev, ok := simdPrevTok(src, start)
		if !ok || prev.kind != token.LPAREN || kept[prev.start] || simdCallOpen(src, prev.start) {
			return start
		}
		start = prev.start
	}
}

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

func simdFwdEnd(src []byte, start, end int) int {
	for {
		next, ok := simdNextTok(src, end)
		if !ok || next.kind != token.RPAREN || simdMatchParen(src, next.end) < start {
			return end
		}
		end = next.end
	}
}

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

func (fl *fileLower) simdExprEnd(e syntax.Expr) int {
	switch n := syntax.Unparen(e).(type) {
	case *syntax.Operation:
		if n.Y != nil {
			return fl.simdExprEnd(n.Y)
		}
		return simdFwdEnd(fl.src, fl.off(syntax.StartPos(n)), fl.simdExprEnd(n.X))
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

func (fl *fileLower) isSimdBinary(e *syntax.Operation) bool {
	if _, ok := simdOps[e.Op]; !ok || e.Y == nil {
		return false
	}
	return fl.simdArrayShape(fl.pkg.info.TypeOf(e)) != nil
}

func (fl *fileLower) isSimdCall(c *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil || !types2.LogosSimdMethod(sel.Sel.Value) {
		return false
	}
	t := fl.pkg.info.TypeOf(sel.X)
	if fl.simdArrayShape(t) == nil {
		return false
	}
	if fl.pkg.info.Selections != nil {
		if _, ok := fl.pkg.info.Selections[sel]; ok {
			return false
		}
	} else if obj, _, _ := types2.LookupFieldOrMethod(t, true, fl.pkg.tpkg, sel.Sel.Value); obj != nil {
		return false
	}
	return true
}

func simdAddr(sh *simdShape, v string) string {
	if !sh.named {
		return "&" + v
	}
	return "(*" + sh.under + ")(&" + v + ")"
}

func simdChunkAddr(sh *simdShape, v string, c, k int64) string {
	lo, hi := c*k, (c+1)*k
	return fmt.Sprintf("(*[%d]%s)(%s[%d:%d])", k, sh.elem, v, lo, hi)
}

func simdCallText(sh *simdShape, body string, aText, bText otext) otext {
	params := "_logos_a, _logos_b " + sh.spell
	return cat(gen("func("+params+") "+sh.spell+" { "+body+" }("), aText, gen(", "), bText, gen(")"))
}

func (fl *fileLower) simdElemwiseText(sh *simdShape, op syntax.Operator, aText, bText otext) otext {
	if vec, chunks, feat, ok := simdVecForm(sh, op); ok {
		load := logosSimdAlias + ".Load" + vec + "Array"
		method := simdOps[op].method
		tis, store := simdVecSuffix(sh, op)
		var stmts []string
		stmts = append(stmts, "var _logos_r "+sh.spell)
		if chunks == 1 {
			stmts = append(stmts, load+"("+simdAddr(sh, "_logos_a")+")"+tis+"."+method+"("+load+"("+simdAddr(sh, "_logos_b")+")"+tis+")"+store+"("+simdAddr(sh, "_logos_r")+")")
		} else {
			k := sh.len / chunks
			chunkVec := fmt.Sprintf("%sx%d", sh.lane.name, 16/sh.lane.width)
			chunkLoad := logosSimdAlias + ".Load" + chunkVec + "Array"
			for c := int64(0); c < chunks; c++ {
				stmts = append(stmts, chunkLoad+"("+simdChunkAddr(sh, "_logos_a", c, k)+")"+tis+"."+method+"("+chunkLoad+"("+simdChunkAddr(sh, "_logos_b", c, k)+")"+tis+")"+store+"("+simdChunkAddr(sh, "_logos_r", c, k)+")")
			}
		}
		stmts = append(stmts, "return _logos_r")
		fl.needSimd = true
		vecBody := strings.Join(stmts, "; ")
		if cond := fl.simdGuardCond(feat); cond != "" {
			return simdCallText(sh, "if "+cond+" { "+vecBody+" } else { "+fl.simdScalarBody(sh, op)+" }", aText, bText)
		}
		return simdCallText(sh, vecBody, aText, bText)
	}
	return simdCallText(sh, fl.simdScalarBody(sh, op), aText, bText)
}

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

func simdSumBody(v string, n int64) string {
	return simdReduceBody(v, n, func(a, b string) string { return a + " + " + b })
}

func (fl *fileLower) lowerSimdBinary(e *syntax.Operation, s simdSite) {
	sh := fl.simdArrayShape(fl.pkg.info.TypeOf(e))
	if sh == nil {
		return
	}
	aText := fl.textOf(s.ops[0])
	bText := fl.textOf(s.ops[1])
	text := fl.simdElemwiseText(sh, e.Op, aText, bText)
	fl.replaceSimd(e, s.whole, text)
}

func (fl *fileLower) lowerSimdCall(c *syntax.CallExpr, s simdSite) {
	sel := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
	sh := fl.simdArrayShape(fl.pkg.info.TypeOf(sel.X))
	if sh == nil {
		return
	}
	method := sel.Sel.Value
	recvText := fl.textOf(s.ops[0])
	var text otext
	switch method {
	case "sum":
		text = cat(gen("func(_logos_a "+sh.spell+") "+sh.elemSpell+" { "+simdSumBody("_logos_a", sh.len)+" }("), recvText, gen(")"))
	case "min", "max":
		text = cat(gen("func(_logos_a "+sh.spell+") "+sh.elemSpell+" { "+simdReduceBody("_logos_a", sh.len, func(a, b string) string { return method + "(" + a + ", " + b + ")" })+" }("), recvText, gen(")"))
	case "dot":
		if len(c.ArgList) != 1 {
			return
		}
		argText := fl.textOf(s.ops[1])
		inner := fl.simdElemwiseText(sh, syntax.Mul, recvText, argText)
		text = cat(gen("func(_logos_p "+sh.spell+") "+sh.elemSpell+" { "+simdSumBody("_logos_p", sh.len)+" }("), inner, gen(")"))
	default:
		return
	}
	fl.replaceSimd(c, s.whole, text)
}

func (fl *fileLower) replaceSimd(old syntax.Expr, whole span, text otext) {
	frag := parseSimdExprFrag(text.s, old.Pos())
	replaceTupleExpr(fl.file, old, frag)
	fl.addT(whole.start, whole.end, text)
}

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

func simdImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, logosSimdAlias),
		Path:         litOf(pos, fmt.Sprintf("%q", logosSimdPath), syntax.StringLit),
	}, pos)
}

func simdMathImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, logosSimdMathAlias),
		Path:         litOf(pos, fmt.Sprintf("%q", logosSimdMathPath), syntax.StringLit),
	}, pos)
}
