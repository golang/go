// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/constant"
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
			fl.lowerConstForLits(decl, nil)
			continue
		}
		sig := fl.cfFuncSig(fn)
		fl.lowerConstForList(&fn.Body.List, cfFuncCtx{sig: sig, body: fn.Body})
		fl.lowerConstForLits(fn.Body, sig)
	}
	fl.collectInterpKinds()
}

// lowerConstForLits unrolls the const fors of every function literal
// and block lambda under root, outer bodies before the bodies their
// copies and bodies hold: a body is expanded first, so the bodies found
// inside it include those of its fresh copies, whose types the copy
// checks recorded.
func (fl *fileLower) lowerConstForLits(root syntax.Node, enclosing *types2.Signature) {
	for _, lb := range collectLitBodies(root) {
		sig := enclosing
		if lb.funcLit != nil {
			sig = fl.cfLitSig(lb.funcLit)
		}
		fl.lowerConstForList(&lb.body.List, cfFuncCtx{sig: sig, body: lb.body})
		fl.lowerConstForLits(lb.body, sig)
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

type cfLitBody struct {
	body    *syntax.BlockStmt
	funcLit *syntax.FuncLit
}

func collectLitBodies(root syntax.Node) []cfLitBody {
	var out []cfLitBody
	syntax.Inspect(root, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			out = append(out, cfLitBody{body: n.Body, funcLit: n})
			return false
		case *syntax.LambdaExpr:
			if block, ok := n.Body.(*syntax.BlockStmt); ok {
				out = append(out, cfLitBody{body: block})
				return false
			}
		}
		return true
	})
	return out
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
			eachStmtList((*list)[i], func(child *[]syntax.Stmt) {
				fl.lowerConstForList(child, fctx)
			})
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
	bodyOff := offAt(run.ctx.starts, syntax.StartPos(cf.Body))
	bodyEnd := offAt(run.ctx.starts, syntax.EndPos(cf.Body)) + 1
	bodySrc := run.ctx.buf.sliceO(bodyOff, bodyEnd)
	var out []syntax.Stmt
	host := &cfHost{sep: "\n"}
	outerNodes := cfBodyNodes(cf.Body)
	for ord, fd := range cfFieldsOf(fl, info.Struct) {
		if !fl.cfCharge(run, outerNodes) {
			break
		}
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
func (fl *fileLower) reparseTopCopy(text otext, cf *syntax.ConstForStmt, fctx cfFuncCtx) ([]syntax.Stmt, *cfRegion) {
	fbase := syntax.NewFileBase(fl.baseName + ".cf")
	var errs []error
	f, _ := syntax.Parse(fbase, strings.NewReader("package cfreparse\nfunc cfreparse() {\n"+text.s+"\n}\n"), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.LogosMode)
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
		Sig:       fctx.sig,
		Disqual:   fl.pkg.info.LogosDisqual[fctx.body],
		Funcs:     fl.pkg.ctExp.copyFuncs,
		Aliases:   fl.cfAliases(),
		MaxSteps:  fl.pkg.info.LogosCTFEMaxSteps,
		NodeLimit: base.LogosComptimeNodes,
		Importer:  fl.pkg.info.LogosImporter,
		Context:   fl.pkg.info.LogosContext,
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
	reg := fl.getGenImports()
	if reg == nil {
		return nil
	}
	out := make(map[*types2.Package]string, len(reg.qual))
	for path, q := range reg.qual {
		if pkg := reg.pkgs[path]; pkg != nil && q != "" {
			out[pkg] = q
		}
	}
	return out
}

// adoptRegion appends text to the working buffer as a new region and
// moves every position of roots, parsed from a wrapper in which text
// starts on line textLine of the parsed file base, onto the region's own
// virtual lines. owner is the position the region's nodes take back
// after lowering.
func (fl *fileLower) adoptRegion(text otext, owner syntax.Pos, parsed *syntax.PosBase, textLine int, roots []syntax.Node) *cfRegion {
	fl.src = append(fl.src, '\n')
	r := &cfRegion{start: len(fl.src), line: len(lineStarts(fl.src)), owner: owner}
	fl.src = append(fl.src, text.s...)
	r.end = len(fl.src)
	fl.src = append(fl.src, '\n')
	fl.keywords.Append(fl.src, r.start-1)
	fl.srcMap.b = fl.src
	for _, c := range text.chunks {
		if c.n == 0 {
			continue
		}
		fl.srcMap.chunks = append(fl.srcMap.chunks, chunk{out: r.start + c.out, orig: c.orig, n: c.n})
	}
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
		if s, ok := n.(*syntax.ConstSwitchStmt); ok {
			for _, c := range s.Cases {
				c.Case = fn(c.Case)
				c.Colon = fn(c.Colon)
				c.TildePos = fn(c.TildePos)
			}
		}
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
// tail. The region bytes live in the working buffer with their origin
// mapping, so the composed text keeps the origin of every copied byte.
func (fl *fileLower) composeHost(h *cfHost) otext {
	parts := []otext{gen(h.head)}
	for _, r := range h.regions {
		adj := make([]splice, len(r.splices))
		for i, sp := range r.splices {
			adj[i] = sp
			adj[i].start -= r.start
			adj[i].end -= r.start
		}
		out, err := applyChunks(fl.srcMap.sliceO(r.start, r.end), adj)
		if err != nil {
			base.Fatalf("logos: overlapping edits at %s", r.owner.String())
		}
		parts = append(parts, out, gen(h.sep))
	}
	parts = append(parts, gen(h.tail))
	return cat(parts...)
}

// retireHost composes the text of h for the enclosing span that
// consumes it and closes its regions to further edits.
func (fl *fileLower) retireHost(h *cfHost) otext {
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
func (fl *fileLower) setHostText(h *cfHost, text otext) {
	for i := range fl.splices {
		if fl.splices[i].host == h {
			fl.splices[i].text, fl.splices[i].chunks = text.s, text.chunks
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
	budget   int
	labels   map[string]bool
	labelID  int
}

// cfRunFor builds the run of one const for of the file being lowered.
func (fl *fileLower) cfRunFor(cf *syntax.ConstForStmt) *cfRun {
	sites := fl.cfSitesFor()
	return &cfRun{
		ctx:      cfCtx{buf: fl.srcMap, starts: fl.start, stamp: cf.Pos()},
		fieldsOf: fl.cfFieldsResolver(sites),
		sites:    sites,
	}
}

// unrollConstForFields expands cf over fields. src/starts locate the cf
// positions; stamp is the position every new node takes so traces name
// the .lgo const for line. chain carries outer bindings for nesting.
func (fl *fileLower) unrollConstForFields(cf *syntax.ConstForStmt, fields []cfField, run *cfRun) ([]syntax.Stmt, otext) {
	bodyOff := offAt(run.ctx.starts, syntax.StartPos(cf.Body))
	bodyEnd := offAt(run.ctx.starts, syntax.EndPos(cf.Body)) + 1
	bodySrc := run.ctx.buf.sliceO(bodyOff, bodyEnd)
	var out []syntax.Stmt
	var parts []otext
	outerNodes := cfBodyNodes(cf.Body)
	for ord, fd := range fields {
		if !fl.cfCharge(run, outerNodes) {
			break
		}
		active := cfChain{{name: cf.Field.Value, field: fd, body: cf.Body}}
		if cf.Index != nil && cf.Index.Value != "_" {
			active = append(active, cfActive{name: cf.Index.Value, index: true, idx: ord, body: cf.Body})
		}
		m := &cfMapper{start: bodyOff}
		text := fl.expandLevel(cf.Body, active, bodySrc, 0, m, run)
		out = append(out, fl.reparseStmts(text.s, run.ctx.stamp)...)
		parts = append(parts, text, gen("\n"))
	}
	return out, cat(parts...)
}

// offAt converts a valid file position to a byte offset.
func offAt(starts []int, pos syntax.Pos) int {
	return starts[pos.Line()-1] + int(pos.Col()) - 1
}

// cfCtx bundles the source coordinates one expansion works with.
type cfCtx struct {
	buf    srcBuf
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

func (fl *fileLower) cfLabelSubs(body *syntax.BlockStmt, covered []span, run *cfRun) []cfSub {
	var declarations []*syntax.Name
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.ConstForStmt, *syntax.FuncLit:
			return false
		case *syntax.LabeledStmt:
			if n.Label.Value != "_" {
				declarations = append(declarations, n.Label)
			}
		}
		return true
	})
	if len(declarations) == 0 {
		return nil
	}
	if run.labels == nil {
		run.labels = make(map[string]bool)
		syntax.Inspect(fl.file, func(n syntax.Node) bool {
			if s, ok := n.(*syntax.LabeledStmt); ok {
				run.labels[s.Label.Value] = true
			}
			return true
		})
		scanSegment(run.ctx.buf.b, 0, len(run.ctx.buf.b), func(t tok) bool {
			if t.kind == token.IDENT {
				run.labels[string(run.ctx.buf.b[t.start:t.end])] = true
			}
			return true
		})
	}
	names := make(map[string]string)
	for _, label := range declarations {
		for {
			run.labelID++
			name := fmt.Sprintf("_logos_cf_label_%d_%d_%d", run.ctx.stamp.Line(), run.ctx.stamp.Col(), run.labelID)
			if !run.labels[name] {
				run.labels[name] = true
				names[label.Value] = name
				break
			}
		}
	}
	var subs []cfSub
	syntax.Inspect(body, func(n syntax.Node) bool {
		var label *syntax.Name
		switch n := n.(type) {
		case *syntax.FuncLit:
			return false
		case *syntax.LabeledStmt:
			label = n.Label
		case *syntax.BranchStmt:
			label = n.Label
		}
		if label != nil {
			if name, ok := names[label.Value]; ok {
				start := offAt(run.ctx.starts, label.Pos())
				end := start + len(label.Value)
				if !insideAny(span{start: start, end: end}, covered) {
					subs = append(subs, cfSub{start: start, end: end, text: name})
				}
			}
		}
		return true
	})
	return subs
}

// expandLevel substitutes the active bindings in the body text slice and
// recursively expands nested const fors right to left. sliceStart is the
// mapped offset of text[0]; m maps base offsets to the evolving text.
func (fl *fileLower) expandLevel(body *syntax.BlockStmt, active cfChain, text otext, sliceStart int, m *cfMapper, run *cfRun) otext {
	ctx := run.ctx
	ls := fl.cfLeafSubs(body, active, m.covered(), run)
	ls = append(ls, fl.cfLabelSubs(body, m.covered(), run)...)
	rel := make([]splice, 0, len(ls))
	for _, x := range ls {
		ms, me := m.mapSpan(x.start, x.end)
		rel = append(rel, splice{start: ms - sliceStart, end: me - sliceStart, text: x.text})
	}
	out, err := applyChunks(text, rel)
	if err != nil {
		base.Fatalf("logos: overlapping const for substitutions")
	}
	text = out
	m.subs = append(m.subs, ls...)
	inners := fl.cfInnerConstFors(body, active, run.sites)
	slices.Reverse(inners)
	for _, inner := range inners {
		istart := offAt(ctx.starts, syntax.StartPos(inner))
		iend := offAt(ctx.starts, syntax.EndPos(inner)) + 1
		if insideAny(span{start: istart, end: iend}, m.covered()) {
			continue
		}
		fields := run.fieldsOf(inner, active)
		s, e := m.mapSpan(istart, iend)
		bstart := offAt(ctx.starts, syntax.StartPos(inner.Body))
		bend := offAt(ctx.starts, syntax.EndPos(inner.Body)) + 1
		bs, be := m.mapSpan(bstart, bend)
		var innerParts []otext
		innerNodes := cfBodyNodes(inner.Body)
		for ord, fd := range fields {
			if !fl.cfCharge(run, innerNodes) {
				return gen("")
			}
			sub := append(slices.Clone(active), cfActive{name: inner.Field.Value, field: fd, body: inner.Body})
			if inner.Index != nil && inner.Index.Value != "_" {
				sub = append(sub, cfActive{name: inner.Index.Value, index: true, idx: ord, body: inner.Body})
			}
			iterMap := &cfMapper{start: m.start, subs: slices.Clone(m.subs)}
			one := fl.expandLevel(inner.Body, sub, text.slice(bs-sliceStart, be-sliceStart), bs, iterMap, run)
			innerParts = append(innerParts, one, gen("\n"))
		}
		text = cat(text.slice(0, s-sliceStart), cat(innerParts...), text.slice(e-sliceStart, len(text.s)))
	}
	return text
}

// covered reports the covered spans of a mapper.
func (m *cfMapper) covered() []span {
	out := make([]span, 0, len(m.subs))
	for _, s := range m.subs {
		out = append(out, span{s.start, s.end})
	}
	return out
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
	}, nil, syntax.LogosMode)
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

func (fl *fileLower) cfCharge(run *cfRun, n int) bool {
	if run.budget > 8*base.LogosComptimeNodes {
		return false
	}
	run.budget += n
	if run.budget > 8*base.LogosComptimeNodes {
		fl.cfError(run.ctx.stamp, "[comptime::node_limit] expansion has %d copies above the selected limit %d", run.budget, base.LogosComptimeNodes)
		return false
	}
	return true
}

func cfBodyNodes(root syntax.Node) int {
	n := 0
	syntax.Inspect(root, func(x syntax.Node) bool {
		n++
		return true
	})
	if n < 1 {
		return 1
	}
	return n
}

// cfBareSkip reports whether a bare Name occurrence is a declaration,
// label, or selector element rather than a value use. Assignment
// targets are not skipped: writing a loop variable is meaningless and
// fails loudly at substitution.
func (fl *fileLower) cfBareSkip(n *syntax.Name, p syntax.Node) bool {
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
		if p.Key == n {
			field, _ := fl.pkg.info.ObjectOf(n).(*types2.Var)
			return field != nil && field.IsField()
		}
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
	case *syntax.ConstDecl:
		for _, name := range p.NameList {
			if name == n {
				return true
			}
		}
		return false
	case *syntax.TypeDecl:
		return p.Name == n
	case *syntax.BranchStmt:
		return p.Label == n
	case *syntax.TypeSwitchGuard:
		return p.Lhs == n
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
func cfAssignTarget(n *syntax.Name, p syntax.Node, parents map[syntax.Node]syntax.Node) bool {
	var lhs syntax.Expr
	switch p := cfPastParens(p, parents).(type) {
	case *syntax.AssignStmt:
		if p.Op == syntax.Def {
			return false
		}
		lhs = p.Lhs
	case *syntax.RangeClause:
		if p.Def {
			return false
		}
		lhs = p.Lhs
	default:
		return false
	}
	for _, e := range syntax.UnpackListExpr(lhs) {
		if syntax.Unparen(e) == syntax.Expr(n) {
			return true
		}
	}
	return false
}

func cfPastParens(p syntax.Node, parents map[syntax.Node]syntax.Node) syntax.Node {
	for {
		switch p.(type) {
		case *syntax.ParenExpr, *syntax.ListExpr:
			p = parents[p]
		default:
			return p
		}
	}
}

func cfAddressTaken(p syntax.Node, parents map[syntax.Node]syntax.Node) bool {
	for {
		switch q := p.(type) {
		case *syntax.ParenExpr:
			p = parents[p]
		case *syntax.Operation:
			return q.Op == syntax.And && q.Y == nil
		default:
			return false
		}
	}
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
	pkg          *types2.Package
	imports      []cfImport
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
// a type the checker recorded without type parameters, or any other
// type expression of a copy, checked in the scope of the record's
// package.
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
		if t, ok := fl.cfResolveTypeExpr(e.X, active, sites); ok && e.Op == syntax.Mul && e.Y == nil {
			return types2.NewPointer(t), true
		}
	case *syntax.SliceType:
		if t, ok := fl.cfResolveTypeExpr(e.Elem, active, sites); ok {
			return types2.NewSlice(t), true
		}
	}
	if t := fl.cfRecordedType(e); t != nil && !cfHasTypeParam(t) {
		return t, true
	}
	return fl.cfEvalTypeExpr(e, active, sites)
}

func (fl *fileLower) cfEvalTypeExpr(e syntax.Expr, active cfChain, sites *cfSites) (types2.Type, bool) {
	if sites == nil || sites.scope == nil {
		return nil, false
	}
	bind := types2.LogosTypeBindings{
		Types:  sites.types,
		Pkgs:   map[string]*types2.Package{cfPkgMarker: sites.pkg},
		Fields: make(map[string]types2.Type),
	}
	for i, imp := range sites.imports {
		if pkg, err := fl.cfLoadPackage(imp.path); err == nil {
			bind.Pkgs[cfImportMarker+strconv.Itoa(i)] = pkg
		}
	}
	for _, a := range active {
		if !a.index {
			bind.Fields[a.name] = a.field.typ
		}
	}
	return types2.LogosResolveType(sites.pkg, fl.pkg.info, types2.LogosCopyContext{Scope: sites.scope, Bind: bind}, e)
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
	scanSegment(ctx.buf.b, start, end, func(t tok) bool {
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
	scanSegment(ctx.buf.b, end, len(ctx.buf.b), func(t tok) bool {
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

func cfFoldEnd(ctx cfCtx, e syntax.Expr) int {
	end := cfExactEnd(ctx, e)
	depth := 0
	scanSegment(ctx.buf.b, end, len(ctx.buf.b), func(t tok) bool {
		switch t.kind {
		case token.LPAREN:
			depth++
		case token.RPAREN:
			depth--
			if depth < 0 {
				return false
			}
			if depth == 0 {
				end = t.end
				return false
			}
		case token.COMMENT:
		default:
			return false
		}
		return true
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
		if imp.Path == nil {
			continue
		}
		if path, _ := strconv.Unquote(imp.Path.Value); path != "comptime" {
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
			if cfDeclBinds(d, name) {
				return true
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

func cfDeclBinds(d syntax.Decl, name string) bool {
	switch d := d.(type) {
	case *syntax.VarDecl:
		return cfNamesInclude(d.NameList, name)
	case *syntax.ConstDecl:
		return cfNamesInclude(d.NameList, name)
	case *syntax.TypeDecl:
		return d.Name.Value == name
	}
	return false
}

func cfNamesInclude(names []*syntax.Name, name string) bool {
	for _, n := range names {
		if n.Value == name {
			return true
		}
	}
	return false
}

func cfLitDeclares(lit *syntax.FuncLit, name string) bool {
	for _, fields := range [][]*syntax.Field{lit.Type.ParamList, lit.Type.ResultList} {
		for _, f := range fields {
			if f.Name != nil && f.Name.Value == name {
				return true
			}
		}
	}
	return false
}

func cfHeaderBinds(p, within syntax.Node, name string) bool {
	var header syntax.SimpleStmt
	switch p := p.(type) {
	case *syntax.ForStmt:
		header = p.Init
	case *syntax.IfStmt:
		header = p.Init
	case *syntax.SwitchStmt:
		header = p.Init
		if g, ok := p.Tag.(*syntax.TypeSwitchGuard); ok && within != syntax.Node(g) && g.Lhs != nil && g.Lhs.Value == name {
			return true
		}
	case *syntax.CommClause:
		header = p.Comm
	case *syntax.TypeDecl:
		return within != syntax.Node(p.Name) && p.Name.Value == name
	}
	return header != nil && within != syntax.Node(header) && cfStmtBinds(header, name)
}

// cfHasBinderBefore reports whether a value binder of name sits
// flow-before ref inside body: preceding siblings in each enclosing
// list, plus function literal parameters on the path.
func cfHasBinderBefore(body *syntax.BlockStmt, ref syntax.Node, name string, parents map[syntax.Node]syntax.Node) bool {
	prev := ref
	for p := parents[prev]; p != nil; prev, p = p, parents[p] {
		if lit, ok := p.(*syntax.FuncLit); ok && prev == syntax.Node(lit.Body) && cfLitDeclares(lit, name) {
			return true
		}
		if cfHeaderBinds(p, prev, name) {
			return true
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

type cfLeaf struct {
	fl      *fileLower
	run     *cfRun
	active  cfChain
	covered []span
	parents map[syntax.Node]syntax.Node
	out     []cfSub
}

// cfLeafSubs collects the leaf substitutions for active bindings in
// body. Spans are base coordinates of ctx.
func (fl *fileLower) cfLeafSubs(root syntax.Node, active cfChain, covered []span, run *cfRun) []cfSub {
	c := &cfLeaf{fl: fl, run: run, active: active, covered: covered, parents: cfParents(root)}
	syntax.Inspect(root, c.visit)
	return c.out
}

func (c *cfLeaf) at(n syntax.Node) (int, int) {
	return offAt(c.run.ctx.starts, syntax.StartPos(n)), cfExactEnd(c.run.ctx, n)
}

func (c *cfLeaf) add(s, e int, text string) {
	if e <= s || insideAny(span{start: s, end: e}, c.covered) {
		return
	}
	for _, o := range c.out {
		if o.start <= s && e <= o.end {
			return
		}
	}
	c.out = append(c.out, cfSub{s, e, text})
}

func (c *cfLeaf) visit(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.ConstIfStmt:
		if cfInConstFor(n, c.parents) {
			return true
		}
		if live := c.fl.cfPruneConstIf(n, c.active, c.run, c.add); live != nil {
			syntax.Inspect(live, c.visit)
		}
		return false
	case *syntax.FieldProjExpr:
		c.subFieldProj(n)
		return true
	case *syntax.SelectorExpr:
		c.subSelector(n)
		return true
	case *syntax.CallExpr:
		return c.subCall(n)
	case *syntax.Name:
		c.subName(n)
		return true
	case *syntax.AssignStmt:
		c.subBlankAssign(n)
		return true
	}
	return true
}

func (c *cfLeaf) subFieldProj(n *syntax.FieldProjExpr) {
	a := cfFieldBinding(n.Field, c.active, c.parents)
	if a == nil {
		return
	}
	_, xe := c.at(n.X)
	_, e := c.at(n)
	c.add(xe, e, "."+a.field.name)
}

func (c *cfLeaf) subSelector(n *syntax.SelectorExpr) {
	a := cfFieldBinding(n.X, c.active, c.parents)
	if a == nil {
		return
	}
	s, e := c.at(n)
	switch n.Sel.Value {
	case "Name":
		c.add(s, e, strconv.Quote(a.field.name))
	case "Index":
		c.add(s, e, strconv.Itoa(a.field.index))
	case "Exported":
		c.add(s, e, strconv.FormatBool(a.field.exported))
	case "Tag":
		c.add(s, e, strconv.Quote(a.field.tag))
	case "Type":
		spelled, ok := c.fl.spellLowered(a.field.typ)
		if !ok {
			base.Fatalf("logos: cannot spell field type at %s", n.Pos().String())
		}
		c.add(s, e, spelled)
	}
}

func (c *cfLeaf) subCall(n *syntax.CallExpr) bool {
	ctx, sites := c.run.ctx, c.run.sites
	if matched, descend := c.subTagGet(n); matched {
		return descend
	}
	if v, ok := c.fl.logosLayoutCallValue(n, c.active, sites); ok {
		s, e := cfCallSpan(ctx, n)
		c.add(s, e, keepComments(ctx.buf.b, s, e, logosLayoutText(v)))
		return false
	}
	if c.fl.cfFoldTypeOf(n, sites) {
		s, e := cfCallSpan(ctx, n)
		c.add(s, e, keepComments(ctx.buf.b, s, e, "struct{}{}"))
		return false
	}
	if s, e, text, ok := c.fl.cfNestedSite(n, c.active, c.run); ok {
		c.add(s, e, text)
	}
	return true
}

func (c *cfLeaf) subName(n *syntax.Name) {
	s, e := c.at(n)
	if insideAny(span{start: s, end: e}, c.covered) {
		return
	}
	if c.subMarker(n) {
		return
	}
	a := cfBindingFor(n, c.active, c.parents)
	if a == nil || !a.index {
		return
	}
	p, ok := c.parents[n]
	if !ok {
		return
	}
	if cfAssignTarget(n, p, c.parents) {
		c.fl.cfError(c.run.sites.at(n.Pos()), "[comptime::const_for_assign] cannot assign to const for variable %s", n.Value)
		return
	}
	if cfAddressTaken(p, c.parents) {
		c.fl.cfError(c.run.sites.at(n.Pos()), "[comptime::const_for_assign] cannot take the address of const for variable %s", n.Value)
		return
	}
	if c.fl.cfBareSkip(n, p) {
		return
	}
	c.add(s, e, strconv.Itoa(a.idx))
}

func (c *cfLeaf) subMarker(n *syntax.Name) bool {
	sites := c.run.sites
	if sites == nil {
		return false
	}
	resolve, ok := sites.spell[n.Value]
	if !ok {
		return false
	}
	if p, ok := c.parents[n]; ok && c.fl.cfBareSkip(n, p) {
		return true
	}
	if spelling, ok := resolve(); ok {
		s, e := c.at(n)
		c.add(s, e, spelling)
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

func cfFieldBinding(x syntax.Expr, active cfChain, parents map[syntax.Node]syntax.Node) *cfActive {
	xn, ok := x.(*syntax.Name)
	if !ok {
		return nil
	}
	a := cfBindingFor(xn, active, parents)
	if a == nil || a.index {
		return nil
	}
	return a
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

func (c *cfLeaf) subTagGet(call *syntax.CallExpr) (matched, descend bool) {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Get" || len(call.ArgList) != 1 {
		return false, false
	}
	tx, ok := sel.X.(*syntax.SelectorExpr)
	if !ok || tx.Sel.Value != "Tag" {
		return false, false
	}
	a := cfFieldBinding(tx.X, c.active, c.parents)
	if a == nil {
		return false, false
	}
	if key, ok := c.fl.cfTagGetKey(call.ArgList[0]); ok {
		s, e := c.at(call)
		c.add(s, e, strconv.Quote(reflect.StructTag(a.field.tag).Get(key)))
		return true, false
	}
	s, e := c.at(tx)
	c.add(s, e, reflectAlias+".StructTag("+strconv.Quote(a.field.tag)+")")
	c.fl.needReflect = true
	return true, true
}

func (fl *fileLower) cfTagGetKey(e syntax.Expr) (string, bool) {
	u := syntax.Unparen(e)
	if m := fl.pkg.info.Types; m != nil {
		for _, n := range []syntax.Expr{e, u} {
			if tv, ok := m[n]; ok && tv.Value != nil && tv.Value.Kind() == constant.String {
				return constant.StringVal(tv.Value), true
			}
		}
	} else if tv := u.GetTypeInfo(); tv.Value != nil && tv.Value.Kind() == constant.String {
		return constant.StringVal(tv.Value), true
	}
	if lit, ok := u.(*syntax.BasicLit); ok && lit.Kind == syntax.StringLit {
		if key, err := strconv.Unquote(lit.Value); err == nil {
			return key, true
		}
	}
	return "", false
}

func (fl *fileLower) cfFoldTypeOf(call *syntax.CallExpr, sites *cfSites) bool {
	if sites == nil || sites.comptimeName == "" {
		return false
	}
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok || len(call.ArgList) != 0 {
		return false
	}
	sel, ok := idx.X.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "TypeOf" {
		return false
	}
	qn, ok := sel.X.(*syntax.Name)
	if !ok || qn.Value != sites.comptimeName {
		return false
	}
	return true
}

func (c *cfLeaf) subBlankAssign(s *syntax.AssignStmt) {
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
	if cfFieldBinding(rhs[0], c.active, c.parents) == nil {
		return
	}
	ss, se := c.at(s)
	c.add(ss, se, "")
}
