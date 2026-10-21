// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// isGeneratedFile reports whether name is a compiler-generated input:
// a cgo output. Only generated inputs go through //line mapping below,
// so user //line directives in hand-written files keep their physical
// names exactly as before.
func isGeneratedFile(name string) bool {
	bn := filepath.Base(name)
	return strings.HasPrefix(bn, "_cgo_") || strings.HasSuffix(bn, ".cgo1.go")
}

// siteDisplay reports the file and line a generated check site names:
// for a check under a //line directive in a generated input (M20-S0: the
// rewritten cgo user file, whose checks lowering adds after cmd/cgo
// mapped the positions back to the .lgo source), the directive's file
// and line, so the panic names the source the user wrote (SPEC T-202).
// Everywhere else it keeps the physical file and line, so no existing
// site moves.
func siteDisplay(pos syntax.Pos, base string, line uint) (string, uint) {
	if b := pos.Base(); b != nil && !b.IsFileBase() {
		if phys := pos.FileBase().Filename(); isGeneratedFile(phys) {
			return filepath.Base(pos.RelFilename()), pos.RelLine()
		}
	}
	return base, line
}

// Lower rewrites the loGo/s nodes of files to plain Go syntax and records
// the matching eject edits. pkg and info come from a successful types2
// check with Config.Logos; src holds the original bytes of the files Lower
// may rewrite. Files without an entry in src are left alone. Lower mutates
// the files in place and reports whether anything changed. A true result
// also covers a semantic-only change: const views that live purely in
// checker info (M5-S6 rt.Freeze results, with no ConstType node to erase)
// still need the plain recheck, so the export writer and the backend only
// ever see Go-facing types. A surviving
// loGo/s node (TryExpr, ErrdeferStmt, SwitchExpr, ArmClause,
// AssertedStruct, OptionType, CoalesceExpr, OptSelectExpr, ConstType,
// YieldStmt, ScopeDeferStmt, or InterpExpr) is an internal error.
func Lower(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) (bool, map[*syntax.File][]Edit) {
	clearRodataPackage(pkg.Path()) // M5-S3: fresh read-only marks per compile.
	if f := EmitOptionFacts(pkg, info); f != nil {
		facts.SetCurrent(f)
	}
	if f := EmitConstFacts(pkg, info, files, facts.Current()); f != nil { // M5-S1: const lines and mutation masks.
		facts.SetCurrent(f)
	}
	if f := EmitDefaultFacts(pkg, info, facts.Current()); f != nil { // M6-S3: struct field defaults.
		facts.SetCurrent(f)
	}
	shared := &packageLower{info: info, tpkg: pkg, constMaps: collectConstMaps(files, info)}
	shared.ctExp = newCfExpander(shared, files, src) // M7-S3 hook: comptime generics of every file, registered before any file lowers.
	edits := make(map[*syntax.File][]Edit)
	changed := false
	for _, file := range files {
		data, ok := src[file]
		if !ok {
			continue
		}
		fl := &fileLower{
			pkg:           shared,
			file:          file,
			src:           slices.Clone(data),
			orig:          len(data),
			base:          file.Pos().FileBase(),
			start:         lineStarts(data),
			loweredChains: make(map[*syntax.CallExpr]bool),
		}
		fl.baseName = filepath.Base(fl.base.Filename())
		fl.lowerFile()
		if fl.changed {
			changed = true
			edits[file] = editsOf(data, fl.base, fl.splices)
		}
	}
	clear(interpKinds) // Kinds of originals a const for replaced are never consumed; drop them with the rest.
	if !changed {
		changed = infoHasConstView(info, files) // M5-S6: pure-info consts (rt.Freeze results) need a plain recheck.
	}
	base.ExitIfErrors() // M7-S3 hook: comptime diagnostics end lowering before survivors are rejected.
	for _, file := range files {
		rejectSurvivors(file)
	}
	return changed, edits
}

// packageLower carries the state shared by every file of one package.
type packageLower struct {
	info      *types2.Info
	tpkg      *types2.Package
	siteSeq   int
	constMaps map[*types2.Const]*constMapGen // M5-S4: every const map of the package.
	ctExp     *cfExpander                    // M7-S3: definitions, memo, and cycle stack across files.
}

// fileLower carries the state of lowering one file.
type fileLower struct {
	pkg       *packageLower
	file      *syntax.File
	src       []byte
	orig      int
	start     []int
	base      *syntax.PosBase
	baseName  string
	splices   []splice
	regions   []*cfRegion
	hosts     []*cfHost
	moves     []*chainMove
	changed   bool
	needRT    bool
	needSimd  bool
	siteDecls []*syntax.VarDecl
	// constDecls and constText hold the M5-S4 generated declarations of
	// this file, as parsed nodes and as eject text.
	constDecls []syntax.Decl
	constText  string
	names      map[*types2.Package]string
	// Option lowering state (M3): optMemo pins one lowering per shared
	// `T?` type node, narrowDone pins rewritten narrowed reads,
	// wrapSpan remembers original spans of synthetic wrapper tops for
	// wraps, wrapOwner maps a wrapped node to its wrapper ID for
	// boundary-insert drains, wrapSeq mints wrapper IDs, and
	// retResults maps each return to its function results.
	spreadSeq  int
	rodataVars []rodataVar
	optMemo    map[*syntax.OptionType]syntax.Expr
	narrowDone map[*syntax.Name]bool
	wrapSpan   map[syntax.Expr]span
	wrapOwner  map[syntax.Expr]int
	wrapSeq    int
	retResults map[*syntax.ReturnStmt][]types2.Type
	// optBodyType stashes the checked type of a replaced top-level
	// lambda body (`??`/`?.`/property-read/interpolation): the
	// replacement is synthetic and carries no recorded type, but chain
	// prodType still needs the produced type afterwards (D4).
	optBodyType map[syntax.Expr]types2.Type
	// optBodySpan stashes the source span of a lambda body before the
	// capture pre-pass replaces it: positions of synthetic replacements
	// cannot recover the span (tupleCallSpan reads real parens), but the
	// capture text must cover the original extent (D7).
	optBodySpan map[*syntax.LambdaExpr]span
	// funcHeadOffs caches the offsets of `func` keywords in source
	// order, built once per file by a single forward tokenization for
	// the option type-head lookup.
	funcHeadOffs []int
	// interpNeedConv/interpNeedUnsafe carry the strconv/unsafe import
	// needs of scoped interpolation pre-lowering (D4, D7) into the
	// final import flush, which runs once at the end of lowerInterp.
	interpNeedConv   bool
	interpNeedUnsafe bool
	// Chain lowering state (M6-S7): loweredChains pins one lowering per
	// chain call, so traversals over replaced trees never reprocess a
	// finished call with stale spans. transparent marks generated
	// option-wrapper literals, whose chains lower with the enclosing
	// function's results.
	loweredChains map[*syntax.CallExpr]bool
	transparent   map[*syntax.FuncLit]bool
	chainGrp      chainGroup               // M6-S7 hook: group-body scope for chain results.
	chainExit     func(err string) string  // M6-S7 hook: a header closure hands a failing hoist's error to its statement; nil returns it.
	groupDone     map[*syntax.FuncLit]bool // M8-S2 hook: finished group closures, pruned by the literal scan.
}

// off converts a valid file position to a byte offset.
func (fl *fileLower) off(pos syntax.Pos) int {
	return fl.start[pos.Line()-1] + int(pos.Col()) - 1
}

// appendSplice records one replacement splice in its narrowest owner: a
// moved region holding it, else its const for region, else the file. A
// zero-width splice on a move boundary stays outside the move, as textOf
// treats it, while region routing keeps the M7 boundary rule.
func (fl *fileLower) appendSplice(sp splice) {
	fl.changed = true
	if m := fl.moveContaining(span{start: sp.start, end: sp.end}); m != nil {
		m.inner = append(m.inner, sp)
		return
	}
	r := fl.regionAt(sp.start, sp.end)
	if r == nil {
		fl.splices = append(fl.splices, sp)
		return
	}
	if r.host != nil && r.host.retired {
		base.Fatalf("logos: edit [%d,%d) lands in a finished const for expansion", sp.start, sp.end)
	}
	r.splices = append(r.splices, sp)
}

// moveContaining returns the narrowest moved region holding s, or nil.
// A zero-width splice on the boundary stays outside, matching the
// partition rule textOf drains with.
func (fl *fileLower) moveContaining(s span) *chainMove {
	var best *chainMove
	for _, m := range fl.moves {
		r := m.region
		if s.start < r.start || s.end > r.end {
			continue
		}
		if s.start == s.end && (s.start <= r.start || s.end >= r.end) {
			continue
		}
		if best == nil || r.end-r.start < best.region.end-best.region.start {
			best = m
		}
	}
	return best
}

// moveAtInsertion returns the narrowest moved region strictly holding
// off, or nil. Insertions on the boundary belong to the enclosing
// region or file, never to the move.
func (fl *fileLower) moveAtInsertion(off int) *chainMove {
	return fl.moveContaining(span{start: off, end: off})
}

// ownerOfInsertion returns the list holding a zero-width splice at off:
// the narrowest moved region strictly holding it, else its const for
// region, else the file.
func (fl *fileLower) ownerOfInsertion(off int) *[]splice {
	if m := fl.moveAtInsertion(off); m != nil {
		return &m.inner
	}
	if r := fl.regionAt(off, off); r != nil {
		return &r.splices
	}
	return &fl.splices
}

// openMove registers r as a moved region and returns the list that held
// it, from which the edits inside r have left for the copy, and the mark
// that stands for the copy of r.
func (fl *fileLower) openMove(r span) (*[]splice, string) {
	outer := &fl.splices
	if m := fl.moveContaining(r); m != nil {
		outer = &m.inner
	} else if rg := fl.regionAt(r.start, r.end); rg != nil {
		outer = &rg.splices
	}
	kept, inner := partitionSplices(*outer, r)
	*outer = kept
	m := &chainMove{region: r, mark: fmt.Sprintf("\x00move%d\x00", len(fl.moves)), inner: inner}
	fl.moves = append(fl.moves, m)
	fl.changed = true
	return outer, m.mark
}

func (fl *fileLower) add(start, end int, text string) {
	fl.appendSplice(splice{start: start, end: end, text: text})
}

// addOwned records one splice owned by one wrapper (a wrap ID from
// nextWrapID): a later wrap of the same wrapper drains its boundary
// inserts into fused text instead of leaving them to interleave.
func (fl *fileLower) addOwned(id, start, end int, text string) {
	fl.appendSplice(splice{start: start, end: end, text: text, owner: id})
}

// chainMove is a source region that a hoist copies ahead of its statement
// while the region itself keeps a stub in place. The copy stands in the
// hoist text as mark; the edits inside the region, those already recorded
// and those that later passes record, belong to the copy, so the stub
// never overlaps them, and the copy is composed once every pass has run.
type chainMove struct {
	region span
	mark   string
	inner  []splice
}

// moveRegion turns r into a moved region whose place holds stub, and
// returns the mark that stands for the copy of r. The edits recorded inside
// r so far move to the copy.
func (fl *fileLower) moveRegion(r span, stub string) string {
	list, mark := fl.openMove(r)
	*list = append(*list, splice{start: r.start, end: r.end, text: stub})
	return mark
}

// copyRegion copies r into the text that carries the returned mark, for a
// caller that replaces r in place with a splice covering it. The edits
// inside r, recorded so far and recorded later, belong to the copy, so the
// covering splice never overlaps them.
func (fl *fileLower) copyRegion(r span) string {
	_, mark := fl.openMove(r)
	return mark
}

// resolveChainMoves composes the copy of every moved region, with the
// edits of every pass inside it, into the texts that carry its mark.
// Regions resolve in the order they were moved, which puts a region
// nested in another one ahead of it. Marks substitute in the file, in
// every const for region (a chain inside a const for hoists into its
// region copy), and in later moves; finalizeConstFor then composes the
// regions with the copies inside, so this runs before it.
func (fl *fileLower) resolveChainMoves() {
	for i, m := range fl.moves {
		r := m.region
		adj := make([]splice, len(m.inner))
		for j, sp := range m.inner {
			adj[j] = splice{start: sp.start - r.start, end: sp.end - r.start, text: sp.text, owner: sp.owner}
		}
		out, err := applySplices(fl.src[r.start:r.end], adj)
		if err != nil {
			base.Fatalf("logos: overlapping edits at byte offset %d", r.start)
		}
		for j := range fl.splices {
			fl.splices[j].text = strings.ReplaceAll(fl.splices[j].text, m.mark, string(out))
		}
		for _, rg := range fl.regions {
			for j := range rg.splices {
				rg.splices[j].text = strings.ReplaceAll(rg.splices[j].text, m.mark, string(out))
			}
		}
		for _, later := range fl.moves[i+1:] {
			for j := range later.inner {
				later.inner[j].text = strings.ReplaceAll(later.inner[j].text, m.mark, string(out))
			}
		}
	}
	fl.moves = nil
}

// textOf returns the source of s with nested splices substituted.
func (fl *fileLower) textOf(s span) string {
	var inner []splice
	if m := fl.moveContaining(s); m != nil {
		var kept []splice
		kept, inner = partitionSplices(m.inner, s)
		m.inner = kept
	} else {
		inner = fl.drainInner(s)
	}
	sub := fl.src[s.start:s.end]
	adj := make([]splice, len(inner))
	for i, sp := range inner {
		adj[i] = splice{start: sp.start - s.start, end: sp.end - s.start, text: sp.text, owner: sp.owner}
	}
	out, err := applySplices(sub, adj)
	if err != nil {
		base.Fatalf("logos: overlapping edits at byte offset %d", s.start)
	}
	return string(out)
}

func spanInside(sp splice, s span) bool {
	return sp.start >= s.start && sp.end <= s.end && (sp.end > sp.start || (sp.start > s.start && sp.end < s.end))
}

func partitionSplices(list []splice, s span) (kept, inner []splice) {
	for _, sp := range list {
		if spanInside(sp, s) {
			inner = append(inner, sp)
		} else {
			kept = append(kept, sp)
		}
	}
	return kept, inner
}

func (fl *fileLower) drainInner(s span) []splice {
	if r := fl.regionAt(s.start, s.end); r != nil {
		kept, inner := partitionSplices(r.splices, s)
		r.splices = kept
		return inner
	}
	var kept, inner []splice
	for _, sp := range fl.splices {
		switch {
		case sp.host != nil && spanInside(sp, s):
			inner = append(inner, splice{start: sp.start, end: sp.end, text: fl.retireHost(sp.host), owner: sp.owner})
		case sp.host != nil && hostOverlaps(sp, s):
			cutsHost(sp, s)
		case sp.host == nil && spanInside(sp, s):
			inner = append(inner, sp)
		default:
			kept = append(kept, sp)
		}
	}
	fl.splices = kept
	return inner
}

func hostOverlaps(sp splice, s span) bool {
	return sp.start < s.end && s.start < sp.end
}

func cutsHost(sp splice, s span) {
	base.Fatalf("logos: span [%d,%d) cuts the const for expansion at [%d,%d)", s.start, s.end, sp.start, sp.end)
}

func (fl *fileLower) dropInner(s span) []splice {
	if r := fl.regionAt(s.start, s.end); r != nil {
		kept, inner := partitionSplices(r.splices, s)
		r.splices = kept
		return inner
	}
	var kept, inner []splice
	for _, sp := range fl.splices {
		switch {
		case sp.host != nil && spanInside(sp, s):
			sp.host.retired = true
		case sp.host != nil && hostOverlaps(sp, s):
			cutsHost(sp, s)
		case sp.host == nil && spanInside(sp, s):
			inner = append(inner, sp)
		default:
			kept = append(kept, sp)
		}
	}
	fl.splices = kept
	return inner
}

// newSite registers one package-level trace site and returns its spelling.
func (fl *fileLower) newSite(pos syntax.Pos, line uint) string {
	if r := fl.regionFor(pos); r != nil {
		line = r.owner.Line()
	}
	name := fmt.Sprintf("_logos_site%d", fl.pkg.siteSeq)
	fl.pkg.siteSeq++
	node := nameOf(pos, name)
	base, ln := siteDisplay(pos, fl.baseName, line)
	fl.siteDecls = append(fl.siteDecls, siteDecl(pos, node, base, ln))
	fl.needRT = true
	return name
}

// lowerFile lowers every function body of one file.
func (fl *fileLower) lowerFile() {
	fl.collectInterpKinds()  // M6-S1 hook: interpolation operand kinds before any rewrite.
	fl.lowerComptime()       // M7-S2/S3 hook: comptime generic copies, then const for unrolling, before any rewrite.
	fl.scanRodataMarkers()   // M5-S3 hook: hand-written //logos:rodata lines.
	fl.lowerZeroCopy()       // M5-S5 hook: zero-copy string/bytes conversions.
	fl.lowerCTFE()           // M5-S2 hook: const composite literals and const func erasure.
	fl.lowerConstMapRanges() // M5-S4 hook: range over a const map reads its sorted entries.
	fl.eraseConstViews()     // M5-S1 hook: erase const views and bound recorded slices.
	fl.lowerPlatform()       // M3-S5 hook: MustNonNil wraps for platform-pointer stores.
	fl.lowerUnions()         // M2 hook: switch expressions and struct assertions.
	fl.lowerOptions()        // M3-S6 hook: single-value map reads and ?? over them.
	fl.lowerViewAs()         // M7-S5 hook: comptime.ViewAs calls to the rt helper.
	fl.lowerChains()         // M6-S7 hook: collection chains fused into loops.
	fl.lowerSpreads()        // M6-S4 hook: `..src` spread elements.
	fl.flushTypeImports()    // M6-S4 hook: imports for spelled elided spread literal types.
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if ok && fn.Body != nil {
			newFuncLower(fl, fn.Type, fn.Body).run()
			continue
		}
		scanForLiterals(fl, decl)
	}
	fl.endComptime()  // M7-S2/S3 hook: comptime import and erased-package imports, after pruning removed the dead branches.
	fl.lowerSimd()    // M14-S1 hook: array operators and reductions to IIFEs.
	fl.lowerTuples()  // M6-S5 hook: tuples to structs, pack/unpack.
	fl.lowerLambdas() // M6-S6 hook: lambdas to function literals.
	fl.lowerMakeLen() // M15-S2 hook: products in make lengths to MulLen; sets needRT before the import.
	if fl.needRT || fl.needSimd {
		var imps []syntax.Decl
		if fl.needRT {
			imps = append(imps, rtImport(fl.file.Pos()))
		}
		if fl.needSimd {
			imps = append(imps, simdImport(fl.file.Pos()))
		}
		fl.file.DeclList = append(imps, fl.file.DeclList...)
		fl.addHeaderSplice()
	}
	for _, decl := range fl.siteDecls {
		fl.file.DeclList = append(fl.file.DeclList, decl)
	}
	fl.file.DeclList = append(fl.file.DeclList, fl.constDecls...) // M5-S4: generated const map code.
	if fl.constText != "" {
		fl.add(fl.orig, fl.orig, fl.constText)
	}
	fl.lowerProperties()     // M4-S2 hook: property accessors, selector lowering, prop facts.
	fl.lowerStructDefaults() // M6-S3 hook: strip `= const`, fill composite literals.
	fl.lowerAddrFresh()      // M6-S2 hook: `&expr` to new(expr).
	fl.emitRodata()          // M5-S3 hook: read-only vars, after every pass that creates them.
	fl.lowerInterp()         // M6-S1 hook: interpolated strings (last: whole-span text drains inner splices).
	fl.resolveChainMoves()   // M6-S7 hook: copies of moved chain operands, after every edit inside them.
	fl.finalizeConstFor()
	fl.remapRegions()
}

// addHeaderSplice adds the eject-text edit that spells the rt import and
// any package-level site variables of a lowered file. The AST side
// carries the ImportDecl and VarDecls; the rendered text must match so
// the ejected file compiles standalone. The text is inserted right after
// the package name token: the scan starts at that token and stops at its
// end, skipping only comments and automatic semicolons. No statement
// splice can overlap the zero-width insertion, because every statement
// begins after the package clause.
func (fl *fileLower) addHeaderSplice() {
	off := fl.off(fl.file.PkgName.Pos())
	ins := -1
	scanSegment(fl.src, off, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		ins = t.end
		return false
	})
	if ins < 0 {
		ins = off
	}
	if fl.needRT {
		fl.add(ins, ins, "\n\nimport "+rtAlias+" "+fmt.Sprintf("%q", rtImportPath))
	}
	if fl.needSimd {
		fl.add(ins, ins, "\n\nimport "+logosSimdAlias+" "+fmt.Sprintf("%q", logosSimdPath))
	}
	if len(fl.siteDecls) > 0 {
		text := "\n\n"
		for i, decl := range fl.siteDecls {
			if i > 0 {
				text += "\n"
			}
			text += "var " + decl.NameList[0].Value + " = " + rtAlias + ".Site{" + siteText(decl) + "}\n"
		}
		fl.add(fl.orig, fl.orig, text)
	}
}

// siteText spells the keyed elements of a Site literal for ejected text.
func siteText(decl *syntax.VarDecl) string {
	lit, _ := decl.Values.(*syntax.CompositeLit)
	var file, ln string
	for _, e := range lit.ElemList {
		kv, ok := e.(*syntax.KeyValueExpr)
		if !ok {
			continue
		}
		key, _ := kv.Key.(*syntax.Name)
		val, _ := kv.Value.(*syntax.BasicLit)
		switch key.Value {
		case "File":
			file = "File: " + val.Value
		case "Line":
			ln = "Line: " + val.Value
		}
	}
	return file + ", " + ln
}

// scanForLiterals lowers the function literals under root, skipping none.
// Lowered literals are pruned so their bodies are visited exactly once.
func scanForLiterals(fl *fileLower, root syntax.Node) {
	if root == nil {
		return
	}
	syntax.Inspect(root, func(n syntax.Node) bool {
		lit, ok := n.(*syntax.FuncLit)
		if !ok {
			return true
		}
		if fl.groupDone[lit] { // M8-S2 hook: group closures lower eagerly.
			return false
		}
		newFuncLower(fl, lit.Type, lit.Body).run()
		return false
	})
}

// resultSpec describes one function result.
type resultSpec struct {
	field    *syntax.Field
	name     *syntax.Name
	typ      types2.Type
	typeExpr syntax.Expr
	isErr    bool
}

// funcLower carries the state of lowering one function body.
type funcLower struct {
	fl               *fileLower
	ft               *syntax.FuncType
	body             *syntax.BlockStmt
	specs            []resultSpec
	errName          string
	hasDeferTry      bool
	hasErrdefer      bool
	hasScopeDefer    bool
	okName           string
	errSeq           int
	funSeq           int
	argSeq           int
	tmpSeq           int
	scopeSeq         int
	groupSeq         int          // M8-S2 hook: group temporary counter.
	groupStack       []groupFrame // M8-S2 hook: group frames under lowering.
	scopeParents     map[syntax.Node]syntax.Node
	scopeLabels      map[string]*syntax.LabeledStmt
	scopeDeferLabels map[string]int
	scopeCtxs        []*scopeDeferCtx
}

// newFuncLower builds the lowering state of one function body.
func newFuncLower(fl *fileLower, ft *syntax.FuncType, body *syntax.BlockStmt) *funcLower {
	fc := &funcLower{fl: fl, ft: ft, body: body}
	n := len(ft.ResultList)
	for i, field := range ft.ResultList {
		fc.specs = append(fc.specs, resultSpec{
			field:    field,
			name:     field.Name,
			typ:      fl.pkg.info.TypeOf(field.Type),
			typeExpr: field.Type,
			isErr:    i == n-1,
		})
	}
	if n > 0 {
		fc.errName = namedValue(fc.specs[n-1].name)
	}
	return fc
}

// namedValue returns the spelling of a result name.
func namedValue(name *syntax.Name) string {
	if name == nil {
		return ""
	}
	return name.Value
}

// run prepares and lowers one function body.
func (fc *funcLower) run() {
	fc.lowerYield() // M6-S9 hook: wrap generator bodies first.
	fc.detectCleanups()
	if fc.hasDeferTry || fc.hasErrdefer || fc.hasScopeDefer {
		fc.ensureNamed()
	}
	if fc.hasErrdefer {
		fc.armOkFlag()
	}
	fc.lowerScopeDefers() // M6-S10 hook: scope-defer exits before statement lowering.
	fc.lowerConstIf()     // M7-S1 hook: prune dead `const if` branches first.
	fc.body.List = fc.lowerList(fc.body.List)
}

// detectCleanups records whether the body holds a defer? or an errdefer
// outside nested function literals.
func (fc *funcLower) detectCleanups() {
	syntax.Inspect(fc.body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			return n.Body == fc.body
		case *syntax.CallStmt:
			if n.Tok == syntax.Defer {
				if _, ok := syntax.Unparen(n.Call).(*syntax.TryExpr); ok {
					fc.hasDeferTry = true
				}
			}
		case *syntax.ErrdeferStmt:
			fc.hasErrdefer = true
		case *syntax.ScopeDeferStmt:
			fc.hasScopeDefer = true
		}
		return true
	})
}

// ensureNamed gives every result an assignable name when a defer? or an
// errdefer must reference the error result.
func (fc *funcLower) ensureNamed() {
	if len(fc.specs) == 0 {
		return
	}
	count := 0
	for i := range fc.specs {
		sp := &fc.specs[i]
		if sp.name != nil && sp.name.Value != "_" {
			continue
		}
		spelling := fc.errSpelling()
		if !sp.isErr {
			spelling = fmt.Sprintf("_logos_r%d", count)
			count++
		}
		fc.renameResult(sp, spelling)
	}
	fc.errName = namedValue(fc.specs[len(fc.specs)-1].name)
}

// errSpelling returns the synthetic error result name.
func (fc *funcLower) errSpelling() string {
	return "_logos_err"
}

// renameResult assigns a synthetic name to one result in the tree and text.
func (fc *funcLower) renameResult(sp *resultSpec, spelling string) {
	pos := sp.typeExpr.Pos()
	if sp.name != nil {
		off := fc.fl.off(sp.name.Pos())
		sp.name.Value = spelling
		fc.fl.add(off, off+1, spelling)
		return
	}
	sp.name = nameOf(pos, spelling)
	sp.field.Name = sp.name
	typeOff := fc.fl.off(leftmostOf(sp.typeExpr))
	if len(fc.specs) > 1 || openParenBefore(fc.fl.src, typeOff) {
		fc.fl.add(typeOff, typeOff, spelling+" ")
		return
	}
	braceOff := fc.fl.off(fc.body.Pos())
	fc.fl.add(typeOff, typeOff, "("+spelling+" ")
	fc.fl.add(braceOff, braceOff, ") ")
}

// openParenBefore reports whether the previous token before off is "(".
func openParenBefore(src []byte, off int) bool {
	for i := off - 1; i >= 0; i-- {
		switch src[i] {
		case ' ', '\t', '\n', '\r':
			continue
		case '(':
			return true
		}
		return false
	}
	return false
}

// armOkFlag prepends _logos_ok := false to the function body.
func (fc *funcLower) armOkFlag() {
	fc.okName = "_logos_ok"
	ok := nameOf(fc.body.Pos(), fc.okName)
	fc.body.List = append([]syntax.Stmt{okInit(fc.body.Pos(), ok)}, fc.body.List...)
	braceOff := fc.fl.off(fc.body.Pos())
	fc.fl.add(braceOff+1, braceOff+1, " "+fc.okName+" := false; ")
}

// rejectSurvivors reports any loGo/s node that lowering failed to remove.
func rejectSurvivors(file *syntax.File) {
	syntax.Inspect(file, func(n syntax.Node) bool {
		switch n.(type) {
		case *syntax.TryExpr, *syntax.ErrdeferStmt, *syntax.SwitchExpr, *syntax.ArmClause, *syntax.AssertedStruct,
			*syntax.OptionType, *syntax.CoalesceExpr, *syntax.OptSelectExpr, *syntax.ConstType, *syntax.ConstFuncDecl,
			*syntax.YieldStmt, *syntax.ScopeDeferStmt, *syntax.FieldDefault, *syntax.SpreadExpr, *syntax.InterpExpr,
			*syntax.TupleType, *syntax.TupleLit, *syntax.LambdaExpr, *syntax.ConstIfStmt, *syntax.ConstSwitchStmt,
			*syntax.ConstForStmt, *syntax.FieldProjExpr, *syntax.GroupStmt, *syntax.GroupChild:
			base.Fatalf("logos: %T survived lowering at %s", n, n.Pos().String())
		}
		return true
	})
}
