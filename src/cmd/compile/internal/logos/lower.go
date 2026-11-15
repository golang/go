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
// for a check under a //line directive in a generated input (the
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
// checker info (rt.Freeze results, with no ConstType node to erase)
// still need the plain recheck, so the export writer and the backend only
// ever see Go-facing types. A surviving
// loGo/s node (TryExpr, ErrdeferStmt, SwitchExpr, ArmClause,
// AssertedStruct, OptionType, CoalesceExpr, OptSelectExpr, ConstType,
// YieldStmt, ScopeDeferStmt, or InterpExpr) is an internal error.
func Lower(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) (bool, map[*syntax.File][]Edit) {
	clearRodataPackage(pkg.Path()) // fresh read-only marks per compile.
	clearOwnedConstants()
	f := facts.Write(pkg, info, files, src)
	f.BareUnions = bareUnions(pkg)
	facts.SetCurrent(f)
	shared := &packageLower{info: info, tpkg: pkg, constMaps: collectConstMaps(files, info)}
	shared.importedMaps = map[*types2.Var]*constMapGen{}
	shared.importedDone = map[string]bool{}
	shared.composites, shared.compositeSeq = collectComposites(files, info)
	shared.importedComposites = map[*types2.Var]*importedComposite{}
	base.ClearLogosCtClones(pkg.Path())
	shared.ctExp = newCfExpander(shared, files, src) // comptime generics of every file, registered before any file lowers.
	edits := make(map[*syntax.File][]Edit)
	changed := false
	for _, file := range files {
		data, ok := src[file]
		if !ok {
			continue
		}
		buf := slices.Clone(data)
		fl := &fileLower{
			pkg:           shared,
			file:          file,
			src:           buf,
			orig:          len(buf),
			base:          file.Pos().FileBase(),
			start:         lineStarts(buf),
			loweredChains: make(map[*syntax.CallExpr]bool),
			compositeDone: make(map[syntax.Expr]bool),
			parenClaims:   make(map[int]bool),
			wrapSpan:      make(map[syntax.Expr]span),
			wrapOwner:     make(map[syntax.Expr]int),
		}
		fl.baseName = filepath.Base(fl.base.Filename())
		fl.srcMap = origBuf(buf)
		fl.lowerFile()
		if fl.changed {
			changed = true
			edits[file] = editsOf(data, fl.base, fl.splices)
		}
	}
	clear(interpKinds) // Kinds of originals a const for replaced are never consumed; drop them with the rest.
	if !changed {
		changed = infoHasConstView(info, files) || hasSynthStaticMethods(pkg) // pure-info consts (rt.Freeze results) need a plain recheck.
	}
	base.ExitIfErrors()
	for _, file := range files {
		rejectSurvivors(file)
	}
	for _, name := range shared.ctExp.memo {
		base.RegisterLogosCtClone(pkg.Path(), name)
	}
	return changed, edits
}

// hasSynthStaticMethods reports whether pkg holds a checker-synthesized
// serializer method, which the lowering must replace with a generated
// body.
func hasSynthStaticMethods(pkg *types2.Package) bool {
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		tn, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok {
			continue
		}
		named, ok := tn.Type().(*types2.Named)
		if !ok {
			continue
		}
		for i := range named.NumMethods() {
			m := named.Method(i)
			if m.Pos().IsKnown() {
				continue
			}
			if m.Name() == "MarshalJSON" || m.Name() == "UnmarshalJSON" || m.Name() == "MarshalBinary" || m.Name() == "UnmarshalBinary" || m.Name() == "AppendBinary" {
				return true
			}
		}
	}
	return false
}

// packageLower carries the state shared by every file of one package.
type packageLower struct {
	info               *types2.Info
	tpkg               *types2.Package
	siteSeq            int
	constMaps          map[*types2.Const]*constMapGen
	importedMaps       map[*types2.Var]*constMapGen
	importedSeq        int
	importedDone       map[string]bool
	composites         map[*types2.Const]*compositeInfo
	compositeSeq       int
	localConstSeq      int
	importedComposites map[*types2.Var]*importedComposite
	ctExp              *cfExpander
	jsonHelpers        map[string]*jsonHelperDef
	jsonHelperSeq      int
	jsonUnionKeys      map[string]bool
}

// fileLower carries the state of lowering one file.
type fileLower struct {
	pkg          *packageLower
	file         *syntax.File
	src          []byte
	keywords     facts.KeywordIndex
	srcMap       srcBuf
	orig         int
	start        []int
	base         *syntax.PosBase
	baseName     string
	splices      []splice
	regions      []*cfRegion
	hosts        []*cfHost
	moves        []*chainMove
	changed      bool
	needRT       bool
	needRTText   bool // text-only rt need: eject output references rt, checked AST does not.
	needSimd     bool
	needSimdMath bool
	needJSON     bool
	needBinary   bool
	needReflect  bool
	siteDecls    []*syntax.VarDecl
	// constDecls and constText hold the generated declarations of
	// this file, as parsed nodes and as eject text.
	constDecls                 []syntax.Decl
	constText                  string
	names                      map[*types2.Package]string
	genImps                    *genImports
	importsPendingGeneratedUse []*syntax.ImportDecl
	// Option lowering state: optMemo pins one lowering per shared
	// `T?` type node, narrowDone pins rewritten narrowed reads,
	// wrapSpan remembers original spans of synthetic wrapper tops for
	// wraps, wrapOwner maps a wrapped node to its wrapper ID for
	// boundary-insert drains, wrapSeq mints wrapper IDs, and
	// retResults maps each return to its function results.
	spreadSeq      int
	rodataVars     []rodataVar
	optMemo        map[*syntax.OptionType]syntax.Expr
	parenClaims    map[int]bool
	narrowDone     map[*syntax.Name]bool
	optDeferGoPend map[*syntax.CallExpr]optDeferGoWork
	gotoTargets    map[*syntax.LabeledStmt]bool
	anchorBlocks   map[*syntax.BlockStmt]bool
	stubEnd        map[*syntax.Name]int
	armResult      map[*syntax.ReturnStmt]types2.Type
	wrapSpan       map[syntax.Expr]span
	wrapOwner      map[syntax.Expr]int
	wrapSeq        int
	retResults     map[*syntax.ReturnStmt][]types2.Type
	// optBodyType stashes the checked type of a replaced top-level
	// lambda body (`??`/`?.`/property-read/interpolation): the
	// replacement is synthetic and carries no recorded type, but chain
	// prodType still needs the produced type afterwards (D4).
	optBodyType map[syntax.Expr]types2.Type
	// optBodySpan stashes the source span of a lambda body before the
	// capture pre-pass replaces it: positions of synthetic replacements
	// cannot recover the span (callBodySpan reads real parens), but the
	// capture text must cover the original extent (D7).
	optBodySpan map[*syntax.LambdaExpr]span
	erasedConst map[*syntax.ConstType]bool
	tupleArity  int
	tupleScope  syntax.Node
	propEmitted map[int]bool
	stmtLimit   map[syntax.Stmt]int
	// funcHeadOffs caches the offsets of `func` keywords in source
	// order, built once per file by a single forward tokenization for
	// the option type-head lookup.
	funcHeadOffs     []int
	interpNeedConv   bool
	interpNeedUnsafe bool
	interpNeedFmt    bool
	// Chain lowering state: loweredChains pins one lowering per
	// chain call, so traversals over replaced trees never reprocess a
	// finished call with stale spans. transparent marks generated
	// option-wrapper literals, whose chains lower with the enclosing
	// function's results.
	loweredChains map[*syntax.CallExpr]bool
	transparent   map[*syntax.FuncLit]bool
	// stagedReplacements batches whole-tree expression swaps: passes
	// record old-to-new pairs while lowering and flush them with one
	// walk at the end of the pass, instead of rescanning the whole file
	// per replaced node.
	stagedReplacements map[syntax.Expr]syntax.Expr
	chainGrp           chainGroup               // group-body scope for chain results.
	chainExit          func(err string) string  // a header closure hands a failing hoist's error to its statement; nil returns it.
	groupDone          map[*syntax.FuncLit]bool // finished group closures, pruned by the literal scan.
	groupPlans         map[*syntax.BlockStmt]groupFrame
	groupRecvSelects   map[*syntax.SelectStmt]bool
	groupClosures      *groupClosureRegistry
	groupJoins         map[syntax.Stmt]bool
	funcs              map[*syntax.BlockStmt]*funcLower
	compositeDone      map[syntax.Expr]bool
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
	s := span{start: sp.start, end: sp.end}
	fl.placeSplice(sp, fl.narrowestMove(s, fl.coversOwnCopy(sp)))
}

func (fl *fileLower) clearMoved(s span) {
	fl.placeSplice(splice{start: s.start, end: s.end}, fl.moveAbove(s))
}

func (fl *fileLower) placeSplice(sp splice, m *chainMove) {
	fl.changed = true
	if m != nil {
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
	return fl.narrowestMove(s, false)
}

func (fl *fileLower) moveAbove(s span) *chainMove {
	return fl.narrowestMove(s, true)
}

func (fl *fileLower) narrowestMove(s span, properly bool) *chainMove {
	var best *chainMove
	for _, m := range fl.moves {
		r := m.region
		if s.start < r.start || s.end > r.end || (properly && s == r) {
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

func (fl *fileLower) coversOwnCopy(sp splice) bool {
	for _, m := range fl.moves {
		if m.region.start == sp.start && m.region.end == sp.end && strings.Contains(sp.text, m.mark) {
			return true
		}
	}
	return false
}

// ownerOfInsertion returns the list holding a zero-width splice at off:
// the narrowest moved region strictly holding it, else its const for
// region, else the file.
func (fl *fileLower) ownerOfInsertion(off int) *[]splice {
	if m := fl.moveContaining(span{start: off, end: off}); m != nil {
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
	if m := fl.moveAbove(r); m != nil {
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

func (fl *fileLower) addFirst(off int, text string) {
	list := fl.ownerOfInsertion(off)
	*list = slices.Insert(*list, 0, splice{start: off, end: off, text: text})
	fl.changed = true
}

func (fl *fileLower) addAfterExpression(off int, text string) {
	fl.appendSplice(splice{start: off, end: off, text: text, placement: placementAfterExpression})
}

// addOwned records one splice owned by one wrapper (a wrap ID from
// nextWrapID): a later wrap of the same wrapper drains its boundary
// inserts into fused text instead of leaving them to interleave.
func (fl *fileLower) addOwned(id, start, end int, text string) {
	fl.appendSplice(splice{start: start, end: end, text: text, owner: id})
}

// addT records one replacement whose text carries origin chunks.
func (fl *fileLower) addT(start, end int, t otext) {
	fl.appendSplice(splice{start: start, end: end, text: t.s, chunks: t.chunks})
}

// addOwnedT is addOwned for text that carries origin chunks.
func (fl *fileLower) addOwnedT(id, start, end int, t otext) {
	fl.appendSplice(splice{start: start, end: end, text: t.s, chunks: t.chunks, owner: id})
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
	stub   *string
	host   *[]splice
}

// moveRegion turns r into a moved region whose place holds stub, and
// returns the mark that stands for the copy of r. The edits recorded inside
// r so far move to the copy.
func (fl *fileLower) moveRegion(r span, stub string) string {
	list, mark := fl.openMove(r)
	*list = append(*list, splice{start: r.start, end: r.end, text: stub})
	moved := fl.moves[len(fl.moves)-1]
	moved.stub, moved.host = &stub, list
	return mark
}

// copyRegion copies r into the text that carries the returned mark, for a
// caller that replaces r in place with a splice covering it. The edits
// inside r, recorded so far and recorded later, belong to the copy, so the
// covering splice never overlaps them.
func (fl *fileLower) copyRegion(r span) string {
	for _, m := range fl.moves {
		if m.stub != nil && m.region == r {
			return fl.consumeStub(m)
		}
	}
	_, mark := fl.openMove(r)
	return mark
}

func (fl *fileLower) consumeStub(m *chainMove) string {
	text := *m.stub
	r := m.region
	*m.host = slices.DeleteFunc(*m.host, func(sp splice) bool {
		return sp.start == r.start && sp.end == r.end && sp.text == text
	})
	m.stub = nil
	return text
}

func (fl *fileLower) lazyText(r span) otext {
	return gen(fl.copyRegion(r))
}

func (fl *fileLower) composeMove(m *chainMove) otext {
	r := m.region
	adj := make([]splice, len(m.inner))
	for j, sp := range m.inner {
		adj[j] = sp
		adj[j].start -= r.start
		adj[j].end -= r.start
	}
	out, err := applyChunks(fl.srcMap.sliceO(r.start, r.end), adj)
	if err != nil {
		base.Fatalf("logos: %v", err)
	}
	return out
}

func (fl *fileLower) settleMarks(t otext) otext {
	return substNul(t, func(mark string) (otext, bool) {
		if text, ok := fl.groupClosureText(mark); ok {
			return fl.settleMarks(text), true
		}
		for _, m := range fl.moves {
			if m.mark == mark {
				return fl.settleMarks(fl.composeMove(m)), true
			}
		}
		return otext{}, false
	})
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
		out := fl.composeMove(m)
		subst := func(list []splice) {
			for j := range list {
				t := substNul(otext{s: list[j].text, chunks: list[j].chunks}, func(mark string) (otext, bool) {
					if mark == m.mark {
						return out, true
					}
					return otext{}, false
				})
				list[j].text, list[j].chunks = t.s, t.chunks
			}
		}
		subst(fl.splices)
		for _, rg := range fl.regions {
			subst(rg.splices)
		}
		for _, later := range fl.moves[i+1:] {
			subst(later.inner)
		}
	}
	fl.moves = nil
}

// textOf returns the source of s with nested splices substituted,
// carrying origin: the untouched bytes keep their source offsets and
// every nested splice embeds its own chunks.
func (fl *fileLower) textOf(s span) otext {
	var inner []splice
	if m := fl.moveAbove(s); m != nil {
		var kept []splice
		kept, inner = partitionSplices(m.inner, s)
		m.inner = kept
	} else {
		inner = fl.drainInner(s)
	}
	adj := make([]splice, len(inner))
	for i, sp := range inner {
		adj[i] = sp
		adj[i].start -= s.start
		adj[i].end -= s.start
	}
	out, err := applyChunks(fl.srcMap.sliceO(s.start, s.end), adj)
	if err != nil {
		base.Fatalf("logos: %v", err)
	}
	return out
}

// srcUser returns the verbatim source of [start, end) with its origin:
// the bytes map to themselves in the original file.
func (fl *fileLower) srcUser(start, end int) otext {
	return fl.srcMap.sliceO(start, end)
}

func spanInside(sp splice, s span) bool {
	return sp.start >= s.start && sp.end <= s.end && (sp.end > sp.start || (sp.start > s.start && sp.end < s.end))
}

func partitionSplices(list []splice, s span) (kept, inner []splice) {
	kept = list[:0]
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
	kept := fl.splices[:0]
	var inner []splice
	for _, sp := range fl.splices {
		switch {
		case sp.host != nil && spanInside(sp, s):
			t := fl.retireHost(sp.host)
			inner = append(inner, splice{start: sp.start, end: sp.end, text: t.s, chunks: t.chunks, owner: sp.owner, placement: sp.placement})
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
	kept := fl.splices[:0]
	var inner []splice
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
	fl.collectInterpKinds() // interpolation operand kinds before any rewrite.
	fl.lowerComptime()      // comptime generic copies, then const for unrolling, before any rewrite.
	fl.lowerStructDefaults()
	fl.scanRodataMarkers() // hand-written //logos:rodata lines.
	fl.lowerArenaEject()
	fl.lowerZeroCopy() // zero-copy string/bytes conversions.
	fl.lowerCTFE()
	fl.lowerConstMapRanges()
	fl.eraseConstViews()  // erase const views and bound recorded slices; also erases readonly interface `const` qualifiers.
	fl.lowerConstSlices() // []C arguments to `const []I` parameters convert through rt.CovSlice.
	fl.lowerPlatform()    // MustNonNil wraps for platform-pointer stores.
	fl.hoistTryInits()    // switch-expression try inits out of IIFE closures.
	fl.lowerUnions()      // switch expressions and struct assertions.
	fl.lowerOptions()     // single-value map reads and ?? over them.
	fl.lowerViewAs()      // comptime.ViewAs calls to the rt helper.
	fl.lowerSpreads()     // `..src` spread elements.
	fl.lowerChains()      // collection chains fused into loops.
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if ok && fn.Body != nil {
			newFuncLower(fl, fn.Type, fl.cfFuncSig(fn), fn.Body).run()
			continue
		}
		scanForLiterals(fl, decl)
	}
	fl.endComptime()     // comptime import and erased-package imports, after pruning removed the dead branches.
	fl.lowerSimd()       // array operators and reductions to IIFEs.
	fl.lowerTuples()     // tuples to structs, pack/unpack.
	fl.lowerLambdas()    // lambdas to function literals.
	fl.lowerProperties() // property accessors, selector lowering, prop facts.
	fl.lowerMakeLen()    // products in make lengths to MulLen; sets needRT before the import.

	fl.lowerStaticJSON() // generated MarshalJSON and UnmarshalJSON; sets needRT and needJSON before the imports.
	fl.flushTypeImports()
	if fl.needRT || fl.needRTText || fl.needSimd || fl.needSimdMath || fl.needJSON || fl.needReflect || fl.needBinary {
		var imps []syntax.Decl
		pos := fl.file.Pos()
		for _, spec := range fl.logosImportSpecs() {
			if spec.need {
				imps = append(imps, spec.make(pos))
			}
		}
		fl.file.DeclList = append(imps, fl.file.DeclList...)
		fl.addHeaderSplice()
	}
	for _, decl := range fl.siteDecls {
		fl.file.DeclList = append(fl.file.DeclList, decl)
	}
	fl.file.DeclList = append(fl.file.DeclList, fl.constDecls...) // generated const map code.
	fl.cfDemoteImportsUnusedByGenerated()
	if fl.constText != "" {
		fl.add(fl.orig, fl.orig, fl.constText)
	}
	fl.lowerAddrFresh() // `&expr` to new(expr).
	fl.emitRodata()     // read-only vars, after every pass that creates them.
	fl.lowerInterp()    // interpolated strings (last: whole-span text drains inner splices).
	fl.resolveGroupClosureText()
	fl.resolveChainMoves() // copies of moved chain operands, after every edit inside them.
	fl.finalizeConstFor()
	fl.remapRegions()
}

// genImports tracks synthesized and reused imports for one file being lowered.
type genImports struct {
	qual  map[string]string          // pkg path -> alias, monotone, never deleted
	pkgs  map[string]*types2.Package // pkg path -> package object (if known)
	used  map[string]bool            // every alias bound in this file, source and generated
	order []string                   // pkg paths in allocation order
}

type logosImportSpec struct {
	need  bool
	alias string
	path  string
	make  func(syntax.Pos) *syntax.ImportDecl
}

func (fl *fileLower) logosImportSpecs() []logosImportSpec {
	specs := []logosImportSpec{
		{fl.needRT, rtAlias, rtImportPath, rtImport},
		{fl.needJSON, jsonAlias, jsonImportPath, jsonImport},
		{fl.needReflect, reflectAlias, "reflect", reflectImport},
	}
	for _, item := range binaryImportTable {
		need := fl.needBinary && strings.Contains(fl.constText, item.alias+".")
		specs = append(specs, logosImportSpec{need, item.alias, item.path, item.decl})
	}
	return append(specs,
		logosImportSpec{fl.needSimd, logosSimdAlias, logosSimdPath, simdImport},
		logosImportSpec{fl.needSimdMath, logosSimdMathAlias, logosSimdMathPath, simdMathImport},
	)
}

func (fl *fileLower) getGenImports() *genImports {
	if fl.genImps != nil {
		return fl.genImps
	}
	used := map[string]bool{
		interpConvPkg:   true,
		interpUnsafePkg: true,
		interpFmtPkg:    true,
	}
	for _, spec := range fl.logosImportSpecs() {
		used[spec.alias] = true
	}
	for _, name := range fl.importNames() {
		if name != "" {
			used[name] = true
		}
	}
	fl.genImps = &genImports{
		qual: make(map[string]string),
		pkgs: make(map[string]*types2.Package),
		used: used,
	}
	return fl.genImps
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
	for _, spec := range fl.logosImportSpecs() {
		if spec.need || (spec.path == rtImportPath && fl.needRTText) {
			fl.add(ins, ins, "\n\nimport "+spec.alias+" "+fmt.Sprintf("%q", spec.path))
		}
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
		if fl.groupDone[lit] { // group closures lower eagerly.
			return false
		}
		if fl.groupClosurePending(lit) {
			return false
		}
		newFuncLower(fl, lit.Type, fl.cfLitSig(lit), lit.Body).run()
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

type hoistedVar struct {
	name   string
	native syntax.Expr
	eject  string
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
	groupSeq         int          // group temporary counter.
	groupStack       []groupFrame // group frames under lowering.
	elementFrame     bool
	scopeParents     map[syntax.Node]syntax.Node
	scopeLabels      map[string]*syntax.LabeledStmt
	scopeDeferLabels map[string]int
	scopeCtxs        []*scopeDeferCtx
	groupCtxs        map[syntax.Node]*scopeDeferCtx
	scopeHoists      []string
	hoisted          []hoistedVar
	hoistSubs        []hoistSub
	label            *syntax.LabeledStmt
}

func (fl *fileLower) resultSpecs(ft *syntax.FuncType, sig *types2.Signature) []resultSpec {
	n := len(ft.ResultList)
	specs := make([]resultSpec, n)
	for i, field := range ft.ResultList {
		typ := fl.pkg.info.TypeOf(field.Type)
		if sig != nil {
			typ = sig.Results().At(i).Type()
		}
		specs[i] = resultSpec{field: field, name: field.Name, typ: typ, typeExpr: field.Type, isErr: i == n-1}
	}
	return specs
}

// newFuncLower builds the lowering state of one function body.
func newFuncLower(fl *fileLower, ft *syntax.FuncType, sig *types2.Signature, body *syntax.BlockStmt) *funcLower {
	if fc, ok := fl.funcs[body]; ok {
		return fc
	}
	fc := &funcLower{fl: fl, ft: ft, body: body, specs: fl.resultSpecs(ft, sig)}
	if fl.funcs == nil {
		fl.funcs = make(map[*syntax.BlockStmt]*funcLower)
	}
	fl.funcs[body] = fc
	if n := len(fc.specs); n > 0 {
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
	defer fc.groupClosureOwner()()
	fc.lower()
	fc.flushHoists()
}

func (fc *funcLower) lower() {
	fc.lowerYield()
	fc.detectCleanups()
	if fc.hasDeferTry || fc.hasErrdefer || fc.hasScopeDefer {
		fc.ensureNamed()
	}
	if fc.hasErrdefer {
		fc.armOkFlag()
	}
	fc.lowerScopeDefers()
	fc.lowerConstIf()
	fc.body.List = fc.lowerList(fc.body.List)
}

// detectCleanups records whether the body holds a defer? or an errdefer
// outside nested function literals and lambdas.
func (fc *funcLower) detectCleanups() {
	fc.hasDeferTry, fc.hasErrdefer, fc.hasScopeDefer = cleanupsIn(fc.body)
}

func cleanupsIn(body *syntax.BlockStmt) (deferTry, errdefer, scopeDefer bool) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			return n.Body == body
		case *syntax.GroupChild:
			return n.Body == body
		case *syntax.LambdaExpr:
			return false
		case *syntax.CallStmt:
			if n.Tok == syntax.Defer {
				if _, ok := syntax.Unparen(n.Call).(*syntax.TryExpr); ok {
					deferTry = true
				}
			}
		case *syntax.ErrdeferStmt:
			errdefer = true
		case *syntax.ScopeDeferStmt:
			scopeDefer = true
		}
		return true
	})
	return
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

func (fc *funcLower) armOkFlag() {
	fc.okName = "_logos_ok"
	fc.hoistTemp(fc.okName, nameOf(fc.body.Pos(), "bool"), "bool")
}

func (fc *funcLower) hoistTemp(name string, native syntax.Expr, eject string) {
	if slices.ContainsFunc(fc.hoisted, func(h hoistedVar) bool { return h.name == name }) {
		return
	}
	fc.hoisted = append(fc.hoisted, hoistedVar{name: name, native: native, eject: eject})
}

func (fc *funcLower) prependHoists() string {
	pos := fc.body.Pos()
	decls := make([]syntax.Stmt, 0, len(fc.hoisted))
	var text strings.Builder
	for _, h := range fc.hoisted {
		decls = append(decls, typedVarDecl(pos, nameOf(pos, h.name), h.native))
		text.WriteString(" var " + h.name + " " + h.eject + ";")
	}
	fc.body.List = append(decls, fc.body.List...)
	return text.String()
}

func (fc *funcLower) flushHoists() {
	if len(fc.hoisted) == 0 {
		return
	}
	braceOff := fc.fl.off(fc.body.Pos())
	fc.fl.add(braceOff+1, braceOff+1, fc.prependHoists())
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
