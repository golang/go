// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers M7-S2 `const for` over concrete types (SPEC Sec 2.5,
// T-205): each const for unrolls once per field of its struct type in
// declaration order. Lowering is text-primary: the expansion text is
// built from source slices with leaf substitutions, the native tree is
// re-parsed from that text, and every new node takes the const for
// position so traces name the .lgo line. S3 (comptime.go) reuses the
// core for per-instantiation copies.

// cfField is one unrolled struct field.
type cfField struct {
	name     string
	typ      types2.Type
	tag      string
	index    int
	exported bool
}

// cfFieldsOf lists the fields of st in declaration order: exported
// fields plus unexported fields of the current package (#65).
func cfFieldsOf(fl *fileLower, st *types2.Struct) []cfField {
	var out []cfField
	for i := range st.NumFields() {
		f := st.Field(i)
		if !f.Exported() && f.Pkg() != fl.pkg.tpkg {
			continue
		}
		out = append(out, cfField{
			name:     f.Name(),
			typ:      f.Type(),
			tag:      st.Tag(i),
			index:    i,
			exported: f.Exported(),
		})
	}
	return out
}

// cfRegion is one virtual copy of expansion text. It lives at
// [start, end) of the working buffer after the original file text, on
// the virtual lines from line on, so every position inside it names a
// byte of its own text. Splices recorded inside the region apply to
// text only; the host composes the finished region texts. owner is the
// source position every node of the region takes after lowering.
type cfRegion struct {
	start   int
	end     int
	line    int
	owner   syntax.Pos
	text    []byte
	splices []splice
	host    *cfHost
}

// cfHost is the outer splice that stands for the regions of one
// expansion: its text is head, then each region's finished text followed
// by sep, then tail. A host is retired once its text is consumed
// (composed into an enclosing span) or discarded (pruned with its span),
// and its regions accept no further edit.
type cfHost struct {
	regions []*cfRegion
	head    string
	sep     string
	tail    string
	retired bool
}

type cfFuncCtx struct {
	sig  *types2.Signature
	body *syntax.BlockStmt
}

// lowerConstFor unrolls every concrete const for of one file. It runs
// before the other file passes so leaf offsets stay pristine; generic
// const fors (M7-S3) are left for the expansion phase.
func (fl *fileLower) lowerConstFor() {
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if !ok || fn.Body == nil {
			fl.lowerConstForLits(decl)
			continue
		}
		fl.lowerConstForBody(fn.Body, cfFuncCtx{sig: fl.cfFuncSig(fn), body: fn.Body})
		fl.lowerConstForLits(fn.Body)
	}
	fl.collectInterpKinds()
}

// lowerConstForLits unrolls the const fors of every function literal
// under root, each literal under its own signature, outer literals
// before the literals their copies and bodies hold: a literal's body is
// expanded first, so the literals found inside it include those of its
// fresh copies, whose types the copy checks recorded.
func (fl *fileLower) lowerConstForLits(root syntax.Node) {
	for _, lit := range collectFuncLits(root) {
		fl.lowerConstForBody(lit.Body, cfFuncCtx{sig: fl.cfLitSig(lit), body: lit.Body})
		fl.lowerConstForLits(lit.Body)
	}
}

func (fl *fileLower) cfFuncSig(fn *syntax.FuncDecl) *types2.Signature {
	if obj, ok := fl.pkg.info.Defs[fn.Name]; ok {
		if fnObj, ok := obj.(*types2.Func); ok {
			if sig, ok := fnObj.Type().(*types2.Signature); ok {
				return sig
			}
		}
	}
	return nil
}

// cfLitSig returns the signature the checker recorded for a function
// literal. TypeOf reads the Types map or the syntax node, whichever the
// check populated: the compiler stores types in the nodes only, and a
// missing signature skips the check of every copy inside the literal.
func (fl *fileLower) cfLitSig(lit *syntax.FuncLit) *types2.Signature {
	sig, _ := fl.pkg.info.TypeOf(lit).(*types2.Signature)
	return sig
}

// collectFuncLits returns the function literals under root, outermost
// first, without descending into nested ones twice.
func collectFuncLits(root syntax.Node) []*syntax.FuncLit {
	var out []*syntax.FuncLit
	syntax.Inspect(root, func(n syntax.Node) bool {
		if lit, ok := n.(*syntax.FuncLit); ok {
			out = append(out, lit)
			return false
		}
		return true
	})
	return out
}

// lowerConstForBody expands the ancestorless const fors of one body in
// place: each is replaced by its unrolled statements with one splice
// carrying the expansion text.
func (fl *fileLower) lowerConstForBody(body *syntax.BlockStmt, fctx cfFuncCtx) {
	fl.lowerConstForList(&body.List, fctx)
}

// lowerConstForList expands the const fors of one statement list in
// place, recursing into child lists of the statements it keeps. A
// statement of a dead `const if` branch is left alone: the pruning of
// that statement deletes it with its text.
func (fl *fileLower) lowerConstForList(list *[]syntax.Stmt, fctx cfFuncCtx) {
	for i := 0; i < len(*list); i++ {
		if fl.pkg.info.LogosConstIfDead[(*list)[i]] {
			continue
		}
		cf, ok := (*list)[i].(*syntax.ConstForStmt)
		if !ok {
			fl.lowerConstForNested((*list)[i], fctx)
			continue
		}
		info := fl.pkg.info.LogosConstFor[cf]
		if info == nil || info.Struct == nil {
			continue
		}
		stmts, host := fl.unrollConstFor(cf, info, fl.cfRunFor(cf), fctx)
		*list = append((*list)[:i], append(stmts, (*list)[i+1:]...)...)
		fl.addHost(fl.off(syntax.StartPos(cf)), fl.off(syntax.EndPos(cf))+1, host)
		i += len(stmts) - 1
	}
}

// lowerConstForNested expands const fors nested inside one statement's
// child lists (if, for, switch bodies and plain blocks).
func (fl *fileLower) lowerConstForNested(s syntax.Stmt, fctx cfFuncCtx) {
	eachStmtList(s, func(list *[]syntax.Stmt) {
		fl.lowerConstForList(list, fctx)
	})
}

// cfFieldsFunc resolves the fields one const for iterates, given the
// field variables active around it.
type cfFieldsFunc func(*syntax.ConstForStmt, cfChain) []cfField

// cfFieldsResolver builds the field resolver of one expansion context:
// the range type argument resolves through the active field variables,
// the copy context, and the checker's recorded types, in that order.
func (fl *fileLower) cfFieldsResolver(sites *cfSites) cfFieldsFunc {
	return func(inner *syntax.ConstForStmt, active cfChain) []cfField {
		pos := sites.at(inner.Pos())
		typ, ok := fl.cfRangeType(inner, active, sites)
		if !ok {
			return fl.cfRecordedFields(inner, pos)
		}
		st, ok := types2.Unalias(typ).Underlying().(*types2.Struct)
		if !ok {
			fl.cfError(pos, "[comptime::const_for_fields] const for range must be a struct type, have %s", typ)
			return nil
		}
		return cfFieldsOf(fl, st)
	}
}

// cfRecordedFields returns the fields of a const for from the struct
// the checker recorded for it: the fallback when the range type
// expression has no recorded type of its own.
func (fl *fileLower) cfRecordedFields(cf *syntax.ConstForStmt, pos syntax.Pos) []cfField {
	info := fl.pkg.info.LogosConstFor[cf]
	if info == nil || info.Struct == nil {
		fl.cfError(pos, "[comptime::const_for_fields] cannot resolve the const for range type")
		return nil
	}
	return cfFieldsOf(fl, info.Struct)
}

// eachStmtList applies f to every child statement list of s.
func eachStmtList(s syntax.Stmt, f func(*[]syntax.Stmt)) {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		f(&s.List)
	case *syntax.IfStmt:
		eachStmtList(s.Then, f)
		if s.Else != nil {
			if b, ok := s.Else.(*syntax.BlockStmt); ok {
				f(&b.List)
			} else {
				eachStmtList(s.Else, f)
			}
		}
	case *syntax.ConstIfStmt:
		eachStmtList(s.Then, f)
		if s.Else != nil {
			eachStmtList(s.Else, f)
		}
	case *syntax.GroupStmt:
		eachStmtList(s.Body, f)
	case *syntax.GroupChild:
		eachStmtList(s.Body, f)
	case *syntax.ForStmt:
		eachStmtList(s.Body, f)
	case *syntax.SwitchStmt:
		for _, c := range s.Body {
			f(&c.Body)
		}
	case *syntax.SelectStmt:
		for _, c := range s.Body {
			f(&c.Body)
		}
	case *syntax.LabeledStmt:
		eachStmtList(s.Stmt, f)
	case *syntax.RangeClause:
	}
}

// cfChain is the active field-variable chain, innermost last.
type cfChain []cfActive

// cfActive is one active loop variable binding: a field variable (with
// its field) or an index variable (with its iteration ordinal).
type cfActive struct {
	name  string
	field cfField
	index bool
	idx   int
	body  *syntax.BlockStmt
}

// unrollConstFor expands one const for over the fields of info with the
// outer chain applied inside. It returns the replacement statements and
// their text.
func (fl *fileLower) unrollConstFor(cf *syntax.ConstForStmt, info *types2.LogosConstFor, run *cfRun, fctx cfFuncCtx) ([]syntax.Stmt, *cfHost) {
	return fl.unrollConstForTop(cf, cfFieldsOf(fl, info.Struct), run, fctx)
}

func (fl *fileLower) unrollConstForTop(cf *syntax.ConstForStmt, fields []cfField, run *cfRun, fctx cfFuncCtx) ([]syntax.Stmt, *cfHost) {
	bodyOff := offAt(run.ctx.starts, syntax.StartPos(cf.Body))
	bodyEnd := offAt(run.ctx.starts, syntax.EndPos(cf.Body)) + 1
	bodySrc := string(run.ctx.src[bodyOff:bodyEnd])
	var out []syntax.Stmt
	host := &cfHost{sep: "\n"}
	for ord, fd := range fields {
		active := cfChain{{name: cf.Field.Value, field: fd, body: cf.Body}}
		if cf.Index != nil && cf.Index.Value != "_" {
			active = append(active, cfActive{name: cf.Index.Value, index: true, idx: ord, body: cf.Body})
		}
		text := fl.expandLevel(cf.Body, active, bodySrc, 0, &cfMapper{start: bodyOff}, run)
		stmts, region := fl.reparseTopCopy(text, cf, fctx)
		out = append(out, stmts...)
		host.regions = append(host.regions, region)
	}
	fl.cfCountNodes(out, cf.Pos())
	return out, host
}

// addHost records the outer splice of one expansion over [start, end).
func (fl *fileLower) addHost(start, end int, host *cfHost) {
	for _, r := range host.regions {
		r.host = host
	}
	fl.splices = append(fl.splices, splice{start: start, end: end, host: host})
	fl.hosts = append(fl.hosts, host)
	fl.changed = true
}

// regionAt returns the region holding the span [start, end), boundaries
// included, or nil for a span of the original text. Every offset past
// the original text belongs to exactly one region, so an edit never
// lands in the outer list at a buffer offset the original text lacks.
func (fl *fileLower) regionAt(start, end int) *cfRegion {
	for _, r := range fl.regions {
		if start >= r.start && end <= r.end {
			return r
		}
	}
	return nil
}

// reparseTopCopy parses one unrolled iteration as statements, checks
// them on their own, and moves them into a region of their own. The
// parse wraps the text so that it begins on line 3.
func (fl *fileLower) reparseTopCopy(text string, cf *syntax.ConstForStmt, fctx cfFuncCtx) ([]syntax.Stmt, *cfRegion) {
	fbase := syntax.NewFileBase(fl.baseName + ".cf")
	var errs []error
	f, _ := syntax.Parse(fbase, strings.NewReader("package cfreparse\nfunc cfreparse() {\n"+text+"\n}\n"), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.CheckBranches|syntax.LogosMode)
	fn := cfFirstFunc(f)
	if len(errs) > 0 || fn == nil || fn.Body == nil {
		base.Fatalf("logos: cannot parse const for expansion at %s: %v", cf.Pos().String(), errs)
	}
	var blocks []*syntax.BlockStmt
	var roots []syntax.Node
	for _, s := range fn.Body.List {
		b, ok := s.(*syntax.BlockStmt)
		if !ok {
			base.Fatalf("logos: cannot parse const for expansion at %s", cf.Pos().String())
		}
		blocks = append(blocks, b)
		roots = append(roots, s)
	}
	fl.miniCheckCopies(blocks, cf, fctx)
	return fn.Body.List, fl.adoptRegion(text, cf.Pos(), fbase, 3, roots)
}

// miniCheckCopies types the copies of one const for under the scope and
// signature the const for sits in, so the lowering of each copy reads
// the types of its own nodes.
func (fl *fileLower) miniCheckCopies(blocks []*syntax.BlockStmt, cf *syntax.ConstForStmt, fctx cfFuncCtx) {
	ctx := types2.LogosCopyContext{
		Sig:     fctx.sig,
		Disqual: fl.pkg.info.LogosDisqual[fctx.body],
		Funcs:   fl.pkg.ctExp.copyFuncs,
		Aliases: fl.cfAliases(),
	}
	if enclosing := fl.pkg.info.Scopes[cf]; enclosing != nil {
		ctx.Scope = enclosing.Parent()
	}
	if recorded := fl.pkg.info.LogosConstFor[cf]; recorded != nil {
		ctx.Version = recorded.GoVersion
	}
	types2.LogosCheckCopy(fl.pkg.tpkg, fl.pkg.info, ctx, blocks)
}

// cfAliases returns the synthesized package aliases of the file so far,
// each keyed by the package object it names.
func (fl *fileLower) cfAliases() map[*types2.Package]string {
	if set := logosTypeImports[fl.file]; set != nil {
		return set.qual
	}
	return nil
}

// adoptRegion appends text to the working buffer as a new region and
// moves every position of roots, parsed from a wrapper in which text
// starts on line textLine of the parsed file base, onto the region's own
// virtual lines. owner is the position the region's nodes take back
// after lowering.
func (fl *fileLower) adoptRegion(text string, owner syntax.Pos, parsed *syntax.PosBase, textLine int, roots []syntax.Node) *cfRegion {
	fl.src = append(fl.src, '\n')
	r := &cfRegion{start: len(fl.src), line: len(lineStarts(fl.src)), owner: owner, text: []byte(text)}
	fl.src = append(fl.src, text...)
	r.end = len(fl.src)
	fl.src = append(fl.src, '\n')
	fl.start = lineStarts(fl.src)
	fl.regions = append(fl.regions, r)
	virtual := func(p syntax.Pos) syntax.Pos {
		if !p.IsKnown() || p.FileBase() != parsed {
			return p
		}
		return syntax.MakePos(fl.base, uint(r.line)+p.Line()-uint(textLine), p.Col())
	}
	for _, root := range roots {
		cfMapPositions(root, virtual)
	}
	return r
}

var (
	cfPosType     = reflect.TypeFor[syntax.Pos]()
	cfPosListType = reflect.TypeFor[[]syntax.Pos]()
)

// cfMapPositions maps every position held by a node under root through
// fn: the node position and every exported Pos or []Pos field (block and
// switch closers, clause colons, interpolation operand braces, lambda
// arrows, and every other delimiter a node records outside SetPos).
func cfMapPositions(root syntax.Node, fn func(syntax.Pos) syntax.Pos) {
	syntax.Inspect(root, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		n.SetPos(fn(n.Pos()))
		v := reflect.ValueOf(n).Elem()
		for i := range v.NumField() {
			f := v.Field(i)
			if !f.CanSet() {
				continue
			}
			switch f.Type() {
			case cfPosType:
				f.Set(reflect.ValueOf(fn(f.Interface().(syntax.Pos))))
			case cfPosListType:
				for j := range f.Len() {
					e := f.Index(j)
					e.Set(reflect.ValueOf(fn(e.Interface().(syntax.Pos))))
				}
			}
		}
		return true
	})
}

// composeHost returns the finished text of a host: its head, each
// region's text with the region's edits applied followed by sep, and its
// tail.
func (fl *fileLower) composeHost(h *cfHost) string {
	var b strings.Builder
	b.WriteString(h.head)
	for _, r := range h.regions {
		adj := make([]splice, len(r.splices))
		for i, sp := range r.splices {
			adj[i] = splice{start: sp.start - r.start, end: sp.end - r.start, text: sp.text, owner: sp.owner}
		}
		out, err := applySplices(r.text, adj)
		if err != nil {
			base.Fatalf("logos: overlapping edits at %s", r.owner.String())
		}
		b.Write(out)
		b.WriteString(h.sep)
	}
	b.WriteString(h.tail)
	return b.String()
}

// retireHost composes the text of h for the enclosing span that
// consumes it and closes its regions to further edits.
func (fl *fileLower) retireHost(h *cfHost) string {
	h.retired = true
	return fl.composeHost(h)
}

// finalizeConstFor stores the finished text in every host still live.
func (fl *fileLower) finalizeConstFor() {
	for _, h := range fl.hosts {
		if !h.retired {
			fl.setHostText(h, fl.composeHost(h))
		}
	}
}

// setHostText sets the text of the outer splice that stands for h.
func (fl *fileLower) setHostText(h *cfHost, text string) {
	for i := range fl.splices {
		if fl.splices[i].host == h {
			fl.splices[i].text = text
			return
		}
	}
	base.Fatalf("logos: host splice missing at %s", h.regions[0].owner.String())
}

// remapRegions gives every node that lowering left on a virtual line the
// position of the source construct its region expands, so the lowered
// tree, the recheck, and the line tables see the original file.
func (fl *fileLower) remapRegions() {
	if len(fl.regions) == 0 {
		return
	}
	cfMapPositions(fl.file, fl.remapPos)
}

// remapPos returns the owner position of the region p lies in, or p.
func (fl *fileLower) remapPos(p syntax.Pos) syntax.Pos {
	if r := fl.regionFor(p); r != nil {
		return r.owner
	}
	return p
}

// regionFor returns the region that holds the virtual position p, or nil
// for a position of the original text.
func (fl *fileLower) regionFor(p syntax.Pos) *cfRegion {
	if !p.IsKnown() || p.FileBase() != fl.base {
		return nil
	}
	line := int(p.Line())
	if line > len(fl.start) {
		base.Fatalf("logos: copy position outside any region at %s", p.String())
	}
	off := fl.start[line-1] + int(p.Col()) - 1
	if off <= fl.orig {
		return nil
	}
	for _, r := range fl.regions {
		if off >= r.start && off <= r.end {
			return r
		}
	}
	base.Fatalf("logos: copy position outside any region at %s", p.String())
	return nil
}

// cfRun bundles what one expansion carries: the source coordinates, the
// field resolver for nested loops, and the substitution context.
type cfRun struct {
	ctx      cfCtx
	fieldsOf cfFieldsFunc
	sites    *cfSites
}

// cfRunFor builds the run of one const for of the file being lowered.
func (fl *fileLower) cfRunFor(cf *syntax.ConstForStmt) *cfRun {
	sites := fl.cfSitesFor()
	return &cfRun{
		ctx:      cfCtx{src: fl.src, starts: fl.start, stamp: cf.Pos()},
		fieldsOf: fl.cfFieldsResolver(sites),
		sites:    sites,
	}
}

// unrollConstForFields expands cf over fields. src/starts locate the cf
// positions; stamp is the position every new node takes so traces name
// the .lgo const for line. chain carries outer bindings for nesting.
func (fl *fileLower) unrollConstForFields(cf *syntax.ConstForStmt, fields []cfField, run *cfRun) ([]syntax.Stmt, string) {
	bodyOff := offAt(run.ctx.starts, syntax.StartPos(cf.Body))
	bodyEnd := offAt(run.ctx.starts, syntax.EndPos(cf.Body)) + 1
	bodySrc := string(run.ctx.src[bodyOff:bodyEnd])
	var out []syntax.Stmt
	var text strings.Builder
	for ord, fd := range fields {
		active := cfChain{{name: cf.Field.Value, field: fd, body: cf.Body}}
		if cf.Index != nil && cf.Index.Value != "_" {
			active = append(active, cfActive{name: cf.Index.Value, index: true, idx: ord, body: cf.Body})
		}
		iter, itext := fl.unrollIteration(cf, active, bodySrc, bodyOff, run)
		out = append(out, iter...)
		text.WriteString(itext)
		text.WriteByte('\n')
	}
	return out, text.String()
}

// offAt converts a valid file position to a byte offset.
func offAt(starts []int, pos syntax.Pos) int {
	return starts[pos.Line()-1] + int(pos.Col()) - 1
}

// cfCtx bundles the source coordinates one expansion works with.
type cfCtx struct {
	src    []byte
	starts []int
	stamp  syntax.Pos
}

// cfSub is one leaf replacement in base coordinates.
type cfSub struct {
	start, end int
	text       string
}

// cfMapper maps base offsets to current-text offsets through the
// applied substitutions.
type cfMapper struct {
	start int
	subs  []cfSub
}

// mapPos maps a base offset to the current text.
func (m *cfMapper) mapPos(p int) int {
	out := p - m.start
	for _, s := range m.subs {
		if s.end <= p {
			out += len(s.text) - (s.end - s.start)
		}
	}
	return out
}

// mapSpan maps a base span to the current text.
func (m *cfMapper) mapSpan(s, e int) (int, int) {
	return m.mapPos(s), m.mapPos(e)
}

// cfSpan is a base-coordinate span.
type cfSpan struct{ start, end int }

// cfInSpans reports whether [s,e) lies strictly inside any span.
func cfInSpans(spans []cfSpan, s, e int) bool {
	for _, sp := range spans {
		if sp.start <= s && e <= sp.end {
			return true
		}
	}
	return false
}

// cfParents maps every node under root to its parent.
func cfParents(root syntax.Node) map[syntax.Node]syntax.Node {
	parents := make(map[syntax.Node]syntax.Node)
	var stack []syntax.Node
	syntax.Walk(root, cfParentVisitor{parents: parents, stack: &stack})
	return parents
}

// cfParentVisitor tracks the node stack through Visit(nil) exits.
type cfParentVisitor struct {
	parents map[syntax.Node]syntax.Node
	stack   *[]syntax.Node
}

// Visit records parents on entry and pops on exit.
func (v cfParentVisitor) Visit(n syntax.Node) syntax.Visitor {
	if n == nil {
		*v.stack = (*v.stack)[:len(*v.stack)-1]
		return nil
	}
	if len(*v.stack) > 0 {
		v.parents[n] = (*v.stack)[len(*v.stack)-1]
	}
	*v.stack = append(*v.stack, n)
	return v
}

// unrollIteration expands one field iteration: substitute the active
// bindings in the body source, recursively unroll nested const fors in
// the result, and re-parse the final text for the native tree.
func (fl *fileLower) unrollIteration(cf *syntax.ConstForStmt, active cfChain, bodySrc string, bodyOff int, run *cfRun) ([]syntax.Stmt, string) {
	m := &cfMapper{start: bodyOff}
	text := fl.expandLevel(cf.Body, active, bodySrc, 0, m, run)
	stmts := fl.reparseStmts(text, run.ctx.stamp)
	return stmts, text
}

// expandLevel substitutes the active bindings in the body text slice and
// recursively expands nested const fors right to left. sliceStart is the
// mapped offset of text[0]; m maps base offsets to the evolving text.
func (fl *fileLower) expandLevel(body *syntax.BlockStmt, active cfChain, text string, sliceStart int, m *cfMapper, run *cfRun) string {
	ctx := run.ctx
	ls := fl.cfLeafSubs(body, active, m.covered(), run)
	rel := make([]splice, 0, len(ls))
	for _, x := range ls {
		ms, me := m.mapSpan(x.start, x.end)
		rel = append(rel, splice{start: ms - sliceStart, end: me - sliceStart, text: x.text})
	}
	out, err := applySplices([]byte(text), rel)
	if err != nil {
		base.Fatalf("logos: overlapping const for substitutions")
	}
	text = string(out)
	m.subs = append(m.subs, ls...)
	inners := fl.cfInnerConstFors(body, active, run.sites)
	slices.Reverse(inners)
	for _, inner := range inners {
		istart := offAt(ctx.starts, syntax.StartPos(inner))
		iend := offAt(ctx.starts, syntax.EndPos(inner)) + 1
		if cfInSpans(m.covered(), istart, iend) {
			continue
		}
		fields := run.fieldsOf(inner, active)
		s, e := m.mapSpan(istart, iend)
		bstart := offAt(ctx.starts, syntax.StartPos(inner.Body))
		bend := offAt(ctx.starts, syntax.EndPos(inner.Body)) + 1
		bs, be := m.mapSpan(bstart, bend)
		var innerText strings.Builder
		for ord, fd := range fields {
			sub := append(slices.Clone(active), cfActive{name: inner.Field.Value, field: fd, body: inner.Body})
			if inner.Index != nil && inner.Index.Value != "_" {
				sub = append(sub, cfActive{name: inner.Index.Value, index: true, idx: ord, body: inner.Body})
			}
			iterMap := &cfMapper{start: m.start, subs: slices.Clone(m.subs)}
			one := fl.expandLevel(inner.Body, sub, text[bs-sliceStart:be-sliceStart], bs, iterMap, run)
			innerText.WriteString(one)
			innerText.WriteByte('\n')
		}
		text = text[:s-sliceStart] + innerText.String() + text[e-sliceStart:]
	}
	return text
}

// cfCovered converts applied subs to covered spans.
func cfCovered(subs []cfSub) []cfSpan {
	out := make([]cfSpan, 0, len(subs))
	for _, s := range subs {
		out = append(out, cfSpan{s.start, s.end})
	}
	return out
}

// covered reports the covered spans of a mapper.
func (m *cfMapper) covered() []cfSpan {
	return cfCovered(m.subs)
}

// unquote strips the quotes of an import path literal.
func unquote(s string) string {
	if len(s) >= 2 && s[0] == '"' {
		if u, err := strconv.Unquote(s); err == nil {
			return u
		}
	}
	return s
}

// reparseStmts parses text as a statement list for the native tree and
// stamps every node and every scope boundary with stamp so traces name
// the .lgo const for line and debug scopes stay ordered.
func (fl *fileLower) reparseStmts(text string, stamp syntax.Pos) []syntax.Stmt {
	src := "package cfreparse\nfunc cfreparse() {\n" + text + "\n}\n"
	fbase := syntax.NewFileBase(fl.baseName + ".cf")
	var errs []error
	f, _ := syntax.Parse(fbase, strings.NewReader(src), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.CheckBranches|syntax.LogosMode)
	if len(errs) > 0 || f == nil {
		base.Fatalf("logos: cannot parse const for expansion at %s: %v", stamp.String(), errs)
	}
	var body *syntax.BlockStmt
	for _, decl := range f.DeclList {
		if fn, ok := decl.(*syntax.FuncDecl); ok && fn.Name.Value == "cfreparse" {
			body = fn.Body
		}
	}
	if body == nil {
		base.Fatalf("logos: cannot parse const for expansion at %s", stamp.String())
	}
	for _, s := range body.List {
		normalizeChainPos(s, stamp)
	}
	return body.List
}

// cfCountNodes enforces the expanded-node limit on one expansion: a
// count above the selected limit fails, a count at the limit passes.
func (fl *fileLower) cfCountNodes(stmts []syntax.Stmt, pos syntax.Pos) {
	n := 0
	for _, s := range stmts {
		syntax.Inspect(s, func(x syntax.Node) bool {
			n++
			return true
		})
	}
	if n > base.LogosComptimeNodes {
		fl.cfError(pos, "[comptime::node_limit] expansion has %d nodes above the selected limit %d", n, base.LogosComptimeNodes)
	}
}

// cfBareSkip reports whether a bare Name occurrence is a declaration,
// label, or selector element rather than a value use. Assignment
// targets are not skipped: writing a loop variable is meaningless and
// fails loudly at substitution.
func cfBareSkip(n *syntax.Name, p syntax.Node) bool {
	switch p := p.(type) {
	case *syntax.SelectorExpr:
		return p.Sel == n
	case *syntax.Field:
		return p.Name == n
	case *syntax.FuncDecl:
		return p.Name == n
	case *syntax.LabeledStmt:
		return p.Label == n
	case *syntax.KeyValueExpr:
		return p.Key == n
	case *syntax.AssignStmt:
		if p.Op == syntax.Def {
			for _, e := range syntax.UnpackListExpr(p.Lhs) {
				if en, ok := e.(*syntax.Name); ok && en == n {
					return true
				}
			}
		}
		return false
	case *syntax.VarDecl:
		for _, name := range p.NameList {
			if name == n {
				return true
			}
		}
		return false
	case *syntax.RangeClause:
		for _, e := range syntax.UnpackListExpr(p.Lhs) {
			if en, ok := e.(*syntax.Name); ok && en == n {
				return true
			}
		}
		return false
	}
	return false
}

// cfAssignTarget reports whether n is an assignment target under p.
func cfAssignTarget(n *syntax.Name, p syntax.Node) bool {
	as, ok := p.(*syntax.AssignStmt)
	if !ok || as.Op == syntax.Def {
		return false
	}
	for _, e := range syntax.UnpackListExpr(as.Lhs) {
		if en, ok := e.(*syntax.Name); ok && en == n {
			return true
		}
	}
	if as.Rhs == nil {
		if lhs, ok := as.Lhs.(*syntax.Name); ok && lhs == n {
			return true
		}
	}
	return false
}

// cfSites carries the substitution context of one expansion: the local
// name of the comptime import for live Error detection, and, for S3
// copies built from record text, the marker spellings, the marker
// types, the scope that resolves package-level names of the record, and
// the position errors report at (the instantiation site).
type cfSites struct {
	comptimeName string
	spell        map[string]func() (string, bool)
	types        map[string]types2.Type
	scope        *types2.Scope
	errPos       syntax.Pos
}

// at returns the position a diagnostic about pos reports: the
// instantiation site for copies, pos itself otherwise.
func (s *cfSites) at(pos syntax.Pos) syntax.Pos {
	if s != nil && s.errPos.IsKnown() {
		return s.errPos
	}
	return pos
}

// cfError reports one user-facing comptime diagnostic at pos. The run
// fails once lowering finishes, so one defect never cascades.
func (fl *fileLower) cfError(pos syntax.Pos, format string, args ...any) {
	base.Errorf("%s: %s", pos.String(), fmt.Sprintf(format, args...))
}

// cfRangeType resolves the type argument of a const for range
// `comptime.Fields[X]()`.
func (fl *fileLower) cfRangeType(cf *syntax.ConstForStmt, active cfChain, sites *cfSites) (types2.Type, bool) {
	call, ok := cf.Range.(*syntax.CallExpr)
	if !ok {
		return nil, false
	}
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok {
		return nil, false
	}
	return fl.cfResolveTypeExpr(idx.Index, active, sites)
}

// cfResolveTypeExpr evaluates a type expression of const for context:
// `f.Type` of an active field variable, a marker of the copy context, a
// package-level name of the record's scope, a pointer or slice of those,
// or any type the checker recorded without type parameters.
func (fl *fileLower) cfResolveTypeExpr(e syntax.Expr, active cfChain, sites *cfSites) (types2.Type, bool) {
	switch e := e.(type) {
	case *syntax.ParenExpr:
		return fl.cfResolveTypeExpr(e.X, active, sites)
	case *syntax.SelectorExpr:
		if t, ok := cfSelectorType(e, active, sites); ok {
			return t, true
		}
	case *syntax.Name:
		if t, ok := cfNameType(e.Value, sites); ok {
			return t, true
		}
	case *syntax.Operation:
		if t, ok := fl.cfElemType(e.X, active, sites); ok && e.Op == syntax.Mul && e.Y == nil {
			return types2.NewPointer(t), true
		}
	case *syntax.SliceType:
		if t, ok := fl.cfElemType(e.Elem, active, sites); ok {
			return types2.NewSlice(t), true
		}
	}
	if t := fl.cfRecordedType(e); t != nil && !cfHasTypeParam(t) {
		return t, true
	}
	return nil, false
}

// cfRecordedType returns the type the checker recorded for the type
// expression e, from the Types map or, as in the compiler, from the
// syntax node. It returns nil for an expression that is not a type or
// that the checker did not visit.
func (fl *fileLower) cfRecordedType(e syntax.Expr) types2.Type {
	if m := fl.pkg.info.Types; m != nil {
		if tv, ok := m[e]; ok && tv.IsType() {
			return tv.Type
		}
		return nil
	}
	if tv := e.GetTypeInfo(); tv.IsType() {
		return tv.Type
	}
	return nil
}

// cfElemType resolves the element expression of a composite type.
func (fl *fileLower) cfElemType(e syntax.Expr, active cfChain, sites *cfSites) (types2.Type, bool) {
	return fl.cfResolveTypeExpr(e, active, sites)
}

// cfSelectorType resolves `f.Type` and `<pkg marker>.Name`.
func cfSelectorType(e *syntax.SelectorExpr, active cfChain, sites *cfSites) (types2.Type, bool) {
	x, ok := e.X.(*syntax.Name)
	if !ok {
		return nil, false
	}
	if e.Sel.Value == "Type" {
		if a := cfActiveFor(active, x.Value); a != nil && !a.index {
			return a.field.typ, true
		}
	}
	if sites != nil && sites.scope != nil && x.Value == cfPkgMarker {
		return cfScopeType(sites.scope, e.Sel.Value)
	}
	return nil, false
}

// cfNameType resolves a marker or a package-level name of the record.
func cfNameType(name string, sites *cfSites) (types2.Type, bool) {
	if sites == nil {
		return nil, false
	}
	if t, ok := sites.types[name]; ok {
		return t, true
	}
	if sites.scope != nil {
		return cfScopeType(sites.scope, name)
	}
	return nil, false
}

// cfScopeType looks up a type name in scope.
func cfScopeType(scope *types2.Scope, name string) (types2.Type, bool) {
	tn, ok := scope.Lookup(name).(*types2.TypeName)
	if !ok {
		return nil, false
	}
	return tn.Type(), true
}

// cfExactEnd returns the exclusive end offset of node n in ctx. EndPos
// stops short of the closing brackets of calls, indexes, and literals
// (and of the brackets enclosing them): the tokens between the node
// start and its approximate end are balanced up, and the closers still
// open are consumed after it, skipping commas and comments.
func cfExactEnd(ctx cfCtx, n syntax.Node) int {
	start := offAt(ctx.starts, syntax.StartPos(n))
	end := offAt(ctx.starts, syntax.EndPos(n))
	open := 0
	scanSegment(ctx.src, start, end, func(t tok) bool {
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			open++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			open--
		}
		return true
	})
	if open <= 0 {
		return end
	}
	scanSegment(ctx.src, end, len(ctx.src), func(t tok) bool {
		switch t.kind {
		case token.RPAREN, token.RBRACK, token.RBRACE:
			open--
			end = t.end
		case token.COMMA, token.COMMENT, token.SEMICOLON:
		default:
			return false
		}
		return open > 0
	})
	return end
}

// cfSitesFor builds the S2 substitution context: the local name of the
// comptime import.
func (fl *fileLower) cfSitesFor() *cfSites {
	sites := &cfSites{}
	for _, decl := range fl.file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok {
			continue
		}
		if imp.Path == nil || unquote(imp.Path.Value) != "comptime" {
			continue
		}
		if imp.LocalPkgName != nil {
			sites.comptimeName = imp.LocalPkgName.Value
		} else {
			sites.comptimeName = "comptime"
		}
	}
	return sites
}

// cfActiveFor returns the innermost chain entry with the given name.
func cfActiveFor(active cfChain, name string) *cfActive {
	for i := len(active) - 1; i >= 0; i-- {
		if active[i].name == name {
			return &active[i]
		}
	}
	return nil
}

// cfListOf returns the child statement list of p, if any.
func cfListOf(p syntax.Node) *[]syntax.Stmt {
	switch p := p.(type) {
	case *syntax.BlockStmt:
		return &p.List
	case *syntax.CaseClause:
		return &p.Body
	case *syntax.CommClause:
		return &p.Body
	}
	return nil
}

// cfStmtBinds reports whether s declares name as a value.
func cfStmtBinds(s syntax.Stmt, name string) bool {
	switch s := s.(type) {
	case *syntax.AssignStmt:
		if s.Op != syntax.Def {
			return false
		}
		for _, e := range syntax.UnpackListExpr(s.Lhs) {
			if en, ok := e.(*syntax.Name); ok && en.Value == name {
				return true
			}
		}
	case *syntax.DeclStmt:
		for _, d := range s.DeclList {
			if vd, ok := d.(*syntax.VarDecl); ok {
				for _, n := range vd.NameList {
					if n.Value == name {
						return true
					}
				}
			}
		}
	case *syntax.RangeClause:
		if s.Def {
			for _, e := range syntax.UnpackListExpr(s.Lhs) {
				if en, ok := e.(*syntax.Name); ok && en.Value == name {
					return true
				}
			}
		}
	}
	return false
}

// cfHasBinderBefore reports whether a value binder of name sits
// flow-before ref inside body: preceding siblings in each enclosing
// list, plus function literal parameters on the path.
func cfHasBinderBefore(body *syntax.BlockStmt, ref syntax.Node, name string, parents map[syntax.Node]syntax.Node) bool {
	prev := ref
	for p := parents[prev]; p != nil; prev, p = p, parents[p] {
		if lit, ok := p.(*syntax.FuncLit); ok && prev == syntax.Node(lit.Body) {
			for _, f := range lit.Type.ParamList {
				if f.Name != nil && f.Name.Value == name {
					return true
				}
			}
		}
		if list := cfListOf(p); list != nil {
			idx := -1
			for j, s := range *list {
				if syntax.Node(s) == prev {
					idx = j
					break
				}
			}
			if idx < 0 {
				return false
			}
			for _, s := range (*list)[:idx] {
				if cfStmtBinds(s, name) {
					return true
				}
			}
		}
		if p == syntax.Node(body) {
			return false
		}
	}
	return false
}

// cfLeafSubs collects the leaf substitutions for active bindings in
// body. Spans are base coordinates of ctx.
func (fl *fileLower) cfLeafSubs(root syntax.Node, active cfChain, covered []cfSpan, run *cfRun) []cfSub {
	var out []cfSub
	ctx, sites := run.ctx, run.sites
	parents := cfParents(root)
	at := func(n syntax.Node) (int, int) {
		return offAt(ctx.starts, syntax.StartPos(n)), cfExactEnd(ctx, n)
	}
	add := func(s, e int, text string) {
		if e <= s || cfInSpans(covered, s, e) {
			return
		}
		for _, o := range out {
			if o.start <= s && e <= o.end {
				return
			}
		}
		out = append(out, cfSub{s, e, text})
	}
	quoted := strconv.Quote
	var visit func(syntax.Node) bool
	visit = func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.ConstIfStmt:
			if cfInConstFor(n, parents) {
				return true
			}
			if live := fl.cfPruneConstIf(n, active, run, add); live != nil {
				syntax.Inspect(live, visit)
			}
			return false
		case *syntax.FieldProjExpr:
			a := cfBindingFor(n.Field, active, parents)
			if a == nil || a.index {
				return true
			}
			_, xe := at(n.X)
			_, e := at(n)
			add(xe, e, "."+a.field.name)
			return true
		case *syntax.SelectorExpr:
			xn, ok := n.X.(*syntax.Name)
			if !ok {
				return true
			}
			a := cfBindingFor(xn, active, parents)
			if a == nil || a.index {
				return true
			}
			s, e := at(n)
			switch n.Sel.Value {
			case "Name":
				add(s, e, quoted(a.field.name))
			case "Index":
				add(s, e, strconv.Itoa(a.field.index))
			case "Exported":
				if a.field.exported {
					add(s, e, "true")
				} else {
					add(s, e, "false")
				}
			case "Tag":
				add(s, e, quoted(a.field.tag))
			case "Type":
				spelled, ok := fl.spellLowered(a.field.typ)
				if !ok {
					base.Fatalf("logos: cannot spell field type at %s", n.Pos().String())
				}
				add(s, e, spelled)
			}
			return true
		case *syntax.CallExpr:
			if fl.cfFoldTagGet(n, active, parents, at, add) {
				return false
			}
			if sites != nil && sites.comptimeName != "" {
				fl.cfCheckError(n, sites)
			}
			if s, e, text, ok := fl.cfNestedSite(n, active, run); ok {
				add(s, e, text)
			}
			return true
		case *syntax.Name:
			if s, e := at(n); cfInSpans(covered, s, e) {
				return true
			}
			if cfMarkerSub(n, parents, sites, at, add) {
				return true
			}
			a := cfBindingFor(n, active, parents)
			if a == nil || !a.index {
				return true
			}
			p, ok := parents[n]
			if !ok || cfBareSkip(n, p) {
				return true
			}
			if cfAssignTarget(n, p) {
				fl.cfError(sites.at(n.Pos()), "[comptime::const_for_assign] cannot assign to const for variable %s", n.Value)
				return true
			}
			s, e := at(n)
			add(s, e, strconv.Itoa(a.idx))
			return true
		case *syntax.AssignStmt:
			fl.cfBlankAssign(n, active, parents, at, add)
			return true
		}
		return true
	}
	syntax.Inspect(root, visit)
	return out
}

// cfMarkerSub substitutes one marker name of a copy context by its
// spelling and reports whether n is a marker.
func cfMarkerSub(n *syntax.Name, parents map[syntax.Node]syntax.Node, sites *cfSites, at func(syntax.Node) (int, int), add func(int, int, string)) bool {
	if sites == nil {
		return false
	}
	resolve, ok := sites.spell[n.Value]
	if !ok {
		return false
	}
	if p, ok := parents[n]; ok && cfBareSkip(n, p) {
		return true
	}
	if spelling, ok := resolve(); ok {
		s, e := at(n)
		add(s, e, spelling)
	}
	return true
}

// cfBindingFor resolves a loop-variable use to its active entry: the
// innermost entry whose body contains the use wins unless a value
// binder intervenes flow-before it.
func cfBindingFor(ref *syntax.Name, active cfChain, parents map[syntax.Node]syntax.Node) *cfActive {
	for i := len(active) - 1; i >= 0; i-- {
		a := &active[i]
		if a.name != ref.Value || a.body == nil {
			continue
		}
		if !cfWithin(ref, a.body, parents) {
			continue
		}
		if cfHasBinderBefore(a.body, ref, ref.Value, parents) {
			return nil
		}
		return a
	}
	return nil
}

// cfWithin reports whether node lies under root.
func cfWithin(node, root syntax.Node, parents map[syntax.Node]syntax.Node) bool {
	for p := parents[node]; p != nil; p = parents[p] {
		if p == root {
			return true
		}
	}
	return false
}

// cfFoldTagGet folds f.Tag.Get("k") with a literal key to the result.
// It reports whether the call was folded.
func (fl *fileLower) cfFoldTagGet(call *syntax.CallExpr, active cfChain, parents map[syntax.Node]syntax.Node, at func(syntax.Node) (int, int), add func(int, int, string)) bool {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Get" || len(call.ArgList) != 1 {
		return false
	}
	tx, ok := sel.X.(*syntax.SelectorExpr)
	if !ok || tx.Sel.Value != "Tag" {
		return false
	}
	xn, ok := tx.X.(*syntax.Name)
	if !ok {
		return false
	}
	a := cfBindingFor(xn, active, parents)
	if a == nil || a.index {
		return false
	}
	lit, ok := call.ArgList[0].(*syntax.BasicLit)
	if !ok || lit.Kind != syntax.StringLit {
		return false
	}
	key, err := strconv.Unquote(lit.Value)
	if err != nil {
		return false
	}
	s, e := at(call)
	add(s, e, strconv.Quote(reflect.StructTag(a.field.tag).Get(key)))
	return true
}

// cfCheckError reports a live comptime.Error call with its message.
func (fl *fileLower) cfCheckError(call *syntax.CallExpr, sites *cfSites) {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Error" {
		return
	}
	xn, ok := sel.X.(*syntax.Name)
	if !ok || xn.Value != sites.comptimeName {
		return
	}
	msg := ""
	if len(call.ArgList) == 1 {
		if lit, ok := call.ArgList[0].(*syntax.BasicLit); ok {
			if u, err := strconv.Unquote(lit.Value); err == nil {
				msg = u
			}
		} else if text, ok := fl.cfRecordedConst(call.ArgList[0]); ok {
			msg = text
		}
	}
	fl.cfError(sites.at(call.Pos()), "[comptime::error] %s", msg)
}

// cfRecordedConst spells the constant value the checker recorded for e,
// from the Types map or, as in the compiler, from the syntax node.
func (fl *fileLower) cfRecordedConst(e syntax.Expr) (string, bool) {
	if m := fl.pkg.info.Types; m != nil {
		if tv, ok := m[e]; ok && tv.Value != nil {
			return tv.Value.String(), true
		}
		return "", false
	}
	if tv := e.GetTypeInfo(); tv.Value != nil {
		return tv.Value.String(), true
	}
	return "", false
}

// cfBlankAssign deletes `_ = f` blank assignments over field variables.
func (fl *fileLower) cfBlankAssign(s *syntax.AssignStmt, active cfChain, parents map[syntax.Node]syntax.Node, at func(syntax.Node) (int, int), add func(int, int, string)) {
	if s.Op != 0 {
		return
	}
	lhs := syntax.UnpackListExpr(s.Lhs)
	rhs := syntax.UnpackListExpr(s.Rhs)
	if len(lhs) != 1 || len(rhs) != 1 {
		return
	}
	ln, ok := lhs[0].(*syntax.Name)
	if !ok || ln.Value != "_" {
		return
	}
	rn, ok := rhs[0].(*syntax.Name)
	if !ok {
		return
	}
	if un, ok := rhs[0].(*syntax.ParenExpr); ok {
		_ = un
	}
	a := cfBindingFor(rn, active, parents)
	if a == nil || a.index {
		return
	}
	ss, se := at(s)
	add(ss, se, "")
}
