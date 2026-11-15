// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"fmt"
	"go/token"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const (
	cfPkgMarker      = "_logos_ctpkg"
	cfImportMarker   = "_logos_ctimp"
	cfComptimeMarker = "_logos_ctcomptime"
	cfTParamMarker   = "_logos_tparam"
	cfCopyInfix      = "_logosct"
)

func cfMarker(n int) string {
	return cfTParamMarker + strconv.Itoa(n)
}

type cfImport struct {
	name string
	path string
}

type cfRecord struct {
	name    string
	tparams int
	foreign bool
	imports []cfImport
	source  string
	sourceO otext
}

func (r *cfRecord) text() string {
	var b strings.Builder
	b.WriteString("//logosct\n")
	for _, imp := range r.imports {
		fmt.Fprintf(&b, "//logosimp %s %s\n", imp.name, imp.path)
	}
	b.WriteString(r.source)
	return b.String()
}

func cfParseRecord(name, text string) (*cfRecord, error) {
	rec := &cfRecord{name: name}
	for strings.HasPrefix(text, "//logos") {
		line, rest, _ := strings.Cut(text, "\n")
		text = rest
		fields := strings.Fields(line)
		switch {
		case fields[0] == "//logosct":
			rec.foreign = true
		case fields[0] == "//logosimp" && len(fields) == 3:
			rec.imports = append(rec.imports, cfImport{name: fields[1], path: fields[2]})
		default:
			return nil, fmt.Errorf("unknown record header %q", line)
		}
	}
	rec.source = text
	n, err := cfCountTParams(text)
	if err != nil {
		return nil, err
	}
	rec.tparams = n
	return rec, nil
}

func cfCountTParams(source string) (int, error) {
	file, errs := cfParseSource("package cfrec\n" + source)
	if len(errs) > 0 {
		return 0, errs[0]
	}
	fn := cfFirstFunc(file)
	if fn == nil {
		return 0, fmt.Errorf("record has no function")
	}
	return len(fn.TParamList), nil
}

func cfParseSource(text string) (*syntax.File, []error) {
	var errs []error
	file, _ := syntax.Parse(syntax.NewFileBase("cfrecord.lgo"), strings.NewReader(text), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.CheckBranches|syntax.LogosMode)
	return file, errs
}

func cfFirstFunc(file *syntax.File) *syntax.FuncDecl {
	if file == nil {
		return nil
	}
	for _, decl := range file.DeclList {
		if fn, ok := decl.(*syntax.FuncDecl); ok {
			return fn
		}
	}
	return nil
}

type cfGenericDef struct {
	decl   *syntax.FuncDecl
	obj    *types2.Func
	file   *syntax.File
	record *cfRecord
}

type cfFrame struct {
	key   string
	label string
}

type cfExpander struct {
	defs      []*cfGenericDef
	byObj     map[*types2.Func]*cfGenericDef
	byDecl    map[*syntax.FuncDecl]*cfGenericDef
	foreign   map[string]*cfRecord
	memo      map[string]string
	stack     []cfFrame
	pkgs      map[string]*types2.Package
	roots     []*types2.Package
	seq       int
	copyFuncs []*types2.Func
}

func newCfExpander(shared *packageLower, files []*syntax.File, src map[*syntax.File][]byte) *cfExpander {
	exp := &cfExpander{
		byObj:   make(map[*types2.Func]*cfGenericDef),
		byDecl:  make(map[*syntax.FuncDecl]*cfGenericDef),
		foreign: make(map[string]*cfRecord),
		memo:    make(map[string]string),
		pkgs:    make(map[string]*types2.Package),
	}
	f := facts.Current()
	for _, file := range files {
		data, ok := src[file]
		if !ok {
			continue
		}
		fl := &fileLower{pkg: shared, file: file, src: data, orig: len(data), start: lineStarts(data)}
		fl.srcMap = origBuf(data)
		f = fl.registerComptime(exp, f)
	}
	if f != nil {
		facts.SetCurrent(f)
	}
	return exp
}

func (e *cfExpander) frameIndex(key string) int {
	return slices.IndexFunc(e.stack, func(f cfFrame) bool { return f.key == key })
}

func cfExpKey(name, pkgPath string, targs []types2.Type) string {
	var b strings.Builder
	b.WriteString(pkgPath)
	b.WriteByte(0)
	b.WriteString(name)
	for _, t := range targs {
		b.WriteByte(0)
		b.WriteString(types2.TypeString(t, func(p *types2.Package) string { return p.Path() }))
	}
	return b.String()
}

func cfShortKeys(targs []types2.Type) string {
	parts := make([]string, len(targs))
	for i, t := range targs {
		parts[i] = types2.TypeString(t, func(*types2.Package) string { return "" })
	}
	return strings.Join(parts, ", ")
}

func cfFactObject(obj *types2.Func) string {
	if obj == nil {
		return ""
	}
	if q := facts.ObjectName(obj); q != "" {
		return q
	}
	return obj.Name()
}

func (fl *fileLower) registerComptime(exp *cfExpander, f *facts.Facts) *facts.Facts {
	for _, def := range fl.cfCollectGenerics() {
		def.record = fl.cfRecordText(def, "")
		exp.defs = append(exp.defs, def)
		exp.byDecl[def.decl] = def
		if def.obj != nil {
			exp.byObj[def.obj] = def
		}
		if def.obj == nil || !def.obj.Exported() {
			continue
		}
		if f == nil {
			f = &facts.Facts{}
		}
		rec := fl.cfRecordText(def, cfPkgMarker)
		f.Comptimes = append(f.Comptimes, facts.ComptimeFact{Object: cfFactObject(def.obj), Source: []byte(rec.text())})
	}
	return f
}

func (fl *fileLower) cfCollectGenerics() []*cfGenericDef {
	var out []*cfGenericDef
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if !ok || fn.Body == nil || len(fn.TParamList) == 0 || !fl.cfUsesComptime(fn) {
			continue
		}
		var obj *types2.Func
		if o, ok := fl.pkg.info.Defs[fn.Name]; ok {
			obj, _ = o.(*types2.Func)
		}
		out = append(out, &cfGenericDef{decl: fn, obj: obj, file: fl.file})
	}
	return out
}

func (fl *fileLower) cfUsesComptime(fn *syntax.FuncDecl) bool {
	found := false
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.ConstForStmt, *syntax.FieldProjExpr:
			found = true
		case *syntax.SelectorExpr:
			found = found || fl.cfIsComptimePkg(n.X)
		}
		return !found
	})
	return found
}

func (fl *fileLower) cfIsComptimePkg(x syntax.Expr) bool {
	name, ok := x.(*syntax.Name)
	if !ok {
		return false
	}
	pn, ok := fl.pkg.info.Uses[name].(*types2.PkgName)
	return ok && pn.Imported().Path() == "comptime"
}

type cfRecordBuilder struct {
	fl        *fileLower
	qualifier string
	tpIndex   map[types2.Object]int
	imports   []cfImport
	subs      []splice
}

func (fl *fileLower) cfRecordText(def *cfGenericDef, qualifier string) *cfRecord {
	b := &cfRecordBuilder{fl: fl, qualifier: qualifier, tpIndex: make(map[types2.Object]int)}
	for i, f := range def.decl.TParamList {
		if o, ok := fl.pkg.info.Defs[f.Name]; ok {
			b.tpIndex[o] = i
		}
	}
	syntax.Inspect(def.decl, b.visit)
	start := fl.cfFuncStart(fl.off(syntax.StartPos(def.decl)))
	end := fl.off(syntax.EndPos(def.decl)) + 1
	out, err := applyChunks(fl.srcMap.sliceO(start, end), shiftSplices(b.subs, -start))
	if err != nil {
		base.Fatalf("logos: overlapping comptime record at %s", def.decl.Pos().String())
	}
	return &cfRecord{
		name:    def.decl.Name.Value,
		tparams: len(def.decl.TParamList),
		foreign: qualifier != "",
		imports: b.imports,
		source:  out.s,
		sourceO: out,
	}
}

func (b *cfRecordBuilder) visit(n syntax.Node) bool {
	if s, ok := n.(*syntax.ConstIfStmt); ok {
		b.foldCond(s.Cond)
		syntax.Inspect(s.Then, b.visit)
		if s.Else != nil {
			syntax.Inspect(s.Else, b.visit)
		}
		return false
	}
	name, ok := n.(*syntax.Name)
	if !ok {
		return true
	}
	text, ok := b.spelling(b.fl.pkg.info.Uses[name])
	if !ok {
		return true
	}
	s := b.fl.off(name.Pos())
	b.subs = append(b.subs, splice{start: s, end: s + len(name.Value), text: text})
	return true
}

func (b *cfRecordBuilder) spelling(obj types2.Object) (string, bool) {
	if obj == nil {
		return "", false
	}
	if pn, ok := obj.(*types2.PkgName); ok {
		return b.packageMarker(pn), true
	}
	if idx, ok := b.tpIndex[obj]; ok {
		return cfMarker(idx), true
	}
	if b.qualifier != "" && b.packageLevel(obj) {
		return b.qualifier + "." + obj.Name(), true
	}
	return "", false
}

func (b *cfRecordBuilder) packageLevel(obj types2.Object) bool {
	pkg := b.fl.pkg.tpkg
	return obj.Pkg() == pkg && pkg.Scope().Lookup(obj.Name()) == obj
}

func (b *cfRecordBuilder) packageMarker(pn *types2.PkgName) string {
	path := pn.Imported().Path()
	if path == "comptime" {
		return cfComptimeMarker
	}
	idx := slices.IndexFunc(b.imports, func(i cfImport) bool { return i.path == path })
	if idx < 0 {
		idx = len(b.imports)
		b.imports = append(b.imports, cfImport{name: pn.Imported().Name(), path: path})
	}
	return cfImportMarker + strconv.Itoa(idx)
}

func (fl *fileLower) cfFuncStart(off int) int {
	kw, ok := fl.keywords.Start(fl.src, token.FUNC, off)
	if !ok {
		base.Fatalf("logos: no func keyword before offset %d", off)
	}
	return kw
}

func shiftSplices(subs []splice, off int) []splice {
	out := make([]splice, 0, len(subs))
	for _, s := range subs {
		s.start += off
		s.end += off
		out = append(out, s)
	}
	return out
}

type cfSite struct {
	call  *syntax.CallExpr
	fun   syntax.Expr
	targs []types2.Type
	obj   *types2.Func
	pos   syntax.Pos
}

func (fl *fileLower) expandComptime() {
	fl.cfRejectGenericMethods()
	parents := cfParents(fl.file)
	for _, site := range fl.cfScanSites(parents) {
		fl.expandSite(site)
	}
	fl.cfRetireDefs()
}

func (fl *fileLower) lowerComptime() {
	fl.expandComptime()
	fl.lowerConstFor()
}

func (fl *fileLower) endComptime() {
	fl.dropComptimeImport(nil)
	if fl.changed {
		fl.cfDemoteErasedImports()
	}
}

func (fl *fileLower) cfDemoteErasedImports() {
	used := fl.cfNamesInUse()
	for _, decl := range fl.file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok || imp.Path == nil {
			continue
		}
		local, ok := fl.cfImportLocalName(imp)
		if !ok || used[local] {
			continue
		}
		path, _ := strconv.Unquote(imp.Path.Value)
		if fl.genImps != nil {
			if _, ok := fl.genImps.qual[path]; ok {
				fl.importsPendingGeneratedUse = append(fl.importsPendingGeneratedUse, imp)
				continue
			}
		}
		fl.cfDemoteImport(imp, path)
	}
}

func (fl *fileLower) cfDemoteImport(imp *syntax.ImportDecl, path string) {
	if _, ejectUsed := ownedEjectImports[fl.file][path]; !ejectUsed {
		fl.cfBlankImportText(imp)
	}
	imp.LocalPkgName = nameOf(imp.Path.Pos(), "_")
	fl.changed = true
}

func (fl *fileLower) cfDemoteImportsUnusedByGenerated() {
	used := fl.cfNamesInUse()
	for _, imp := range fl.importsPendingGeneratedUse {
		local, ok := fl.cfImportLocalName(imp)
		if !ok || used[local] {
			continue
		}
		path, _ := strconv.Unquote(imp.Path.Value)
		fl.cfDemoteImport(imp, path)
	}
	fl.importsPendingGeneratedUse = nil
}

func (fl *fileLower) cfNamesInUse() map[string]bool {
	used := make(map[string]bool)
	var visit func(syntax.Node) bool
	visit = func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.ImportDecl:
			return false
		case *syntax.SelectorExpr:
			syntax.Inspect(n.X, visit)
			return false
		case *syntax.Name:
			used[n.Value] = true
		}
		return true
	}
	syntax.Inspect(fl.file, visit)
	return used
}

func (fl *fileLower) cfImportLocalName(imp *syntax.ImportDecl) (string, bool) {
	var obj types2.Object
	if imp.LocalPkgName != nil {
		obj = fl.pkg.info.Defs[imp.LocalPkgName]
	} else {
		obj = fl.pkg.info.Implicits[imp]
	}
	pn, ok := obj.(*types2.PkgName)
	if ok && pn.Name() != "_" && pn.Name() != "." && pn.Imported().Path() != "comptime" {
		return pn.Name(), true
	}
	if imp.LocalPkgName != nil && imp.LocalPkgName.Value != "" && imp.LocalPkgName.Value != "_" && imp.LocalPkgName.Value != "." {
		return imp.LocalPkgName.Value, true
	}
	return "", false
}

func (fl *fileLower) cfBlankImportText(imp *syntax.ImportDecl) {
	if imp.LocalPkgName != nil {
		start := fl.off(imp.LocalPkgName.Pos())
		fl.add(start, start+len(imp.LocalPkgName.Value), "_")
	} else {
		start := fl.off(imp.Path.Pos())
		fl.add(start, start, "_ ")
	}
}

func (fl *fileLower) cfRejectGenericMethods() {
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if !ok || fn.Recv == nil || fn.Body == nil || !fl.cfHasGenericConstFor(fn.Body) {
			continue
		}
		fl.cfError(fn.Pos(), "[comptime::generic_method] method %s of a generic type cannot be a comptime generic", fn.Name.Value)
	}
}

func (fl *fileLower) cfHasGenericConstFor(body *syntax.BlockStmt) bool {
	found := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		if cf, ok := n.(*syntax.ConstForStmt); ok {
			info := fl.pkg.info.LogosConstFor[cf]
			found = found || (info != nil && info.Generic)
		}
		return !found
	})
	return found
}

func (fl *fileLower) cfScanSites(parents map[syntax.Node]syntax.Node) []*cfSite {
	var out []*cfSite
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok {
			if site := fl.cfSiteOf(call, parents); site != nil {
				out = append(out, site)
			}
		}
		return true
	})
	return out
}

func (fl *fileLower) cfSiteOf(call *syntax.CallExpr, parents map[syntax.Node]syntax.Node) *cfSite {
	name := cfCalleeName(call.Fun)
	if name == nil {
		return nil
	}
	inst, ok := fl.pkg.info.Instances[name]
	obj, isFunc := fl.pkg.info.Uses[name].(*types2.Func)
	if !ok || !isFunc || !fl.cfIsGeneric(obj) || fl.cfHandledElsewhere(call, parents) {
		return nil
	}
	targs := typeArgsOf(inst.TypeArgs)
	if slices.ContainsFunc(targs, cfHasTypeParam) {
		fl.cfError(call.Pos(), "[comptime::generic_targ] cannot instantiate %s with a type parameter", name.Value)
		return nil
	}
	return &cfSite{call: call, fun: call.Fun, targs: targs, obj: obj, pos: call.Pos()}
}

func (fl *fileLower) cfHandledElsewhere(call *syntax.CallExpr, parents map[syntax.Node]syntax.Node) bool {
	for p := parents[call]; p != nil; p = parents[p] {
		switch p := p.(type) {
		case *syntax.ConstForStmt:
			return true
		case *syntax.FuncDecl:
			_, isDef := fl.pkg.ctExp.byDecl[p]
			return isDef
		}
	}
	return false
}

func cfCalleeName(fun syntax.Expr) *syntax.Name {
	if idx, ok := fun.(*syntax.IndexExpr); ok {
		fun = idx.X
	}
	switch x := fun.(type) {
	case *syntax.Name:
		return x
	case *syntax.SelectorExpr:
		return x.Sel
	}
	return nil
}

func typeArgsOf(l *types2.TypeList) []types2.Type {
	out := make([]types2.Type, l.Len())
	for i := range l.Len() {
		out[i] = l.At(i)
	}
	return out
}

func (fl *fileLower) cfIsGeneric(obj *types2.Func) bool {
	if _, ok := fl.pkg.ctExp.byObj[obj]; ok {
		return true
	}
	if obj.Pkg() == nil || obj.Pkg() == fl.pkg.tpkg {
		return false
	}
	f, ok := facts.Lookup(obj.Pkg().Path())
	if !ok {
		return false
	}
	return slices.ContainsFunc(f.Comptimes, func(c facts.ComptimeFact) bool { return c.Object == cfFactObject(obj) })
}

func (fl *fileLower) cfRecordFor(obj *types2.Func) (*cfRecord, bool) {
	exp := fl.pkg.ctExp
	if def, ok := exp.byObj[obj]; ok {
		return def.record, true
	}
	key := obj.Pkg().Path() + "\x00" + cfFactObject(obj)
	if rec, ok := exp.foreign[key]; ok {
		return rec, true
	}
	f, ok := facts.Lookup(obj.Pkg().Path())
	if !ok {
		return nil, false
	}
	for _, c := range f.Comptimes {
		if c.Object != cfFactObject(obj) {
			continue
		}
		rec, err := cfParseRecord(obj.Name(), string(c.Source))
		if err != nil {
			base.Fatalf("logos: bad comptime record for %s: %v", obj.Name(), err)
		}
		exp.foreign[key] = rec
		return rec, true
	}
	return nil, false
}

func (fl *fileLower) expandSite(site *cfSite) {
	name := fl.cfCopyFor(site)
	ctx := cfCtx{buf: fl.srcMap, starts: fl.start}
	fl.add(offAt(fl.start, syntax.StartPos(site.fun)), cfExactEnd(ctx, site.fun), name)
	site.call.Fun = nameOf(syntax.StartPos(site.fun), name)
	fl.changed = true
}

func cfCopyName(obj *types2.Func, seq int) string {
	return obj.Name() + cfCopyInfix + strconv.Itoa(seq)
}

func (fl *fileLower) cfCopyFor(site *cfSite) string {
	exp := fl.pkg.ctExp
	key := cfExpKey(cfFactObject(site.obj), site.obj.Pkg().Path(), site.targs)
	if name, ok := exp.memo[key]; ok {
		return name
	}
	label := cfShortKeys(site.targs)
	if i := exp.frameIndex(key); i >= 0 {
		fl.cfError(site.pos, "[comptime::circular_instantiation] %s", cfChainText(exp.stack[i:], label))
		return cfCopyName(site.obj, exp.seq)
	}
	rec, ok := fl.cfRecordFor(site.obj)
	if !ok {
		fl.cfError(site.pos, "[comptime::generic_targ] no comptime source for %s", site.obj.Name())
		return cfCopyName(site.obj, exp.seq)
	}
	name := cfCopyName(site.obj, exp.seq)
	exp.seq++
	exp.stack = append(exp.stack, cfFrame{key: key, label: label})
	reported := base.Errors()
	text, ok := fl.cfBuildCopy(site, rec, name)
	exp.stack = exp.stack[:len(exp.stack)-1]
	exp.memo[key] = name
	if ok && base.Errors() == reported {
		fl.cfEmitCopy(text, site)
	}
	return name
}

func cfChainText(frames []cfFrame, label string) string {
	parts := make([]string, 0, len(frames)+1)
	for _, f := range frames {
		parts = append(parts, f.label)
	}
	return strings.Join(append(parts, label), " -> ")
}

func cfStripTParamsO(src otext) (otext, bool) {
	want := []token.Token{token.FUNC, token.IDENT, token.LBRACK}
	var open, depth, end int
	scanSegment([]byte(src.s), 0, len(src.s), func(t tok) bool {
		switch {
		case t.comment:
		case len(want) > 0:
			if t.kind != want[0] {
				return false
			}
			if want[0] == token.LBRACK {
				open, depth = t.start, 1
			}
			want = want[1:]
		case t.kind == token.LBRACK:
			depth++
		case t.kind == token.RBRACK:
			depth--
			if depth == 0 {
				end = t.end
				return false
			}
		}
		return true
	})
	if end == 0 {
		return otext{}, false
	}
	return cat(src.slice(0, open), src.slice(end, len(src.s))), true
}

func (fl *fileLower) cfBuildCopy(site *cfSite, rec *cfRecord, copyName string) (otext, bool) {
	srcO := rec.sourceO
	if srcO.s == "" {
		srcO = gen(rec.source)
	}
	body, ok := cfStripTParamsO(srcO)
	if !ok {
		fl.cfError(site.pos, "[comptime::generic_targ] malformed comptime source of %s", rec.name)
		return otext{}, false
	}
	sites, ok := fl.cfCopySites(site, rec)
	if !ok {
		return otext{}, false
	}
	const prefix = "package cfcopy\n"
	buf := cat(gen(prefix), body)
	src := []byte(buf.s)
	file, errs := cfParseSource(buf.s)
	fn := cfFirstFunc(file)
	if len(errs) > 0 || fn == nil {
		base.Fatalf("logos: cannot parse comptime source of %s: %v", rec.name, errs)
	}
	run := &cfRun{ctx: cfCtx{buf: srcBuf{b: src, chunks: buf.chunks}, starts: lineStarts(src), stamp: site.pos}, fieldsOf: fl.cfFieldsResolver(sites), sites: sites}
	out, err := applyChunks(buf, fl.cfCopySplices(fn, copyName, run))
	if err != nil {
		base.Fatalf("logos: overlapping comptime copy %s: %v", copyName, err)
	}
	return out.slice(len(prefix), len(out.s)), true
}

func (fl *fileLower) cfCopySplices(fn *syntax.FuncDecl, copyName string, run *cfRun) []splice {
	ctx := run.ctx
	nameStart := offAt(ctx.starts, fn.Name.Pos())
	subs := []splice{{start: nameStart, end: nameStart + len(fn.Name.Value), text: copyName}}
	var covered []span
	for _, cf := range fl.cfInnerConstFors(fn.Body, nil, run.sites) {
		s := offAt(ctx.starts, syntax.StartPos(cf))
		e := offAt(ctx.starts, syntax.EndPos(cf)) + 1
		stmts, text := fl.unrollConstForFields(cf, run.fieldsOf(cf, nil), run)
		fl.cfCountNodes(stmts, run.sites.at(cf.Pos()))
		subs = append(subs, splice{start: s, end: e, text: text.s, chunks: text.chunks})
		covered = append(covered, span{s, e})
	}
	for _, sub := range fl.cfLeafSubs(fn, nil, covered, run) {
		subs = append(subs, splice{start: sub.start, end: sub.end, text: sub.text})
	}
	return subs
}

func (fl *fileLower) cfCopySites(site *cfSite, rec *cfRecord) (*cfSites, bool) {
	if len(site.targs) != rec.tparams {
		fl.cfError(site.pos, "[comptime::generic_targ] %s needs %d type arguments, have %d", rec.name, rec.tparams, len(site.targs))
		return nil, false
	}
	fl.pkg.ctExp.roots = append(fl.pkg.ctExp.roots, site.obj.Pkg())
	sites := &cfSites{
		comptimeName: cfComptimeMarker,
		spell:        make(map[string]func() (string, bool)),
		types:        make(map[string]types2.Type),
		scope:        site.obj.Pkg().Scope(),
		pkg:          site.obj.Pkg(),
		imports:      rec.imports,
		errPos:       site.pos,
	}
	for i, t := range site.targs {
		sites.types[cfMarker(i)] = t
		sites.spell[cfMarker(i)] = fl.cfLazySpelling(func() (string, bool) { return fl.spellType(t) }, func() {
			fl.cfError(site.pos, "[comptime::generic_targ] cannot spell type argument %s of %s", t, rec.name)
		})
	}
	for i, imp := range rec.imports {
		sites.spell[cfImportMarker+strconv.Itoa(i)] = fl.cfLazySpelling(func() (string, bool) {
			return fl.cfImportQualifier(imp.path, site.pos)
		}, nil)
	}
	if rec.foreign {
		pkg := site.obj.Pkg()
		sites.spell[cfPkgMarker] = fl.cfLazySpelling(func() (string, bool) {
			return fl.cfImportQualifier(pkg.Path(), site.pos)
		}, nil)
	}
	return sites, true
}

func (fl *fileLower) cfLazySpelling(compute func() (string, bool), fail func()) func() (string, bool) {
	var text string
	var ok, done bool
	return func() (string, bool) {
		if !done {
			done = true
			if text, ok = compute(); !ok && fail != nil {
				fail()
			}
		}
		return text, ok
	}
}

func (fl *fileLower) cfImportQualifier(path string, pos syntax.Pos) (string, bool) {
	if reg := fl.getGenImports(); reg != nil {
		if q, ok := reg.qual[path]; ok && q != "" {
			return q, true
		}
	}
	var own []string
	for pkg, spelling := range fl.importNames() {
		if pkg.Path() == path && spelling != "" {
			own = append(own, spelling)
		}
	}
	if len(own) > 0 {
		slices.Sort(own)
		return own[0], true
	}
	exp := fl.pkg.ctExp
	pkg := exp.pkgs[path]
	if pkg == nil {
		var err error
		if pkg, err = fl.cfLoadPackage(path); err != nil {
			fl.cfError(pos, "[comptime::generic_import] cannot import %s for the copy: %v", path, err)
			return "", false
		}
		exp.pkgs[path] = pkg
	}
	return fl.synthQualifier(pkg), true
}

func (fl *fileLower) cfLoadPackage(path string) (*types2.Package, error) {
	if pkg := fl.cfKnownPackage(path); pkg != nil && pkg.Complete() {
		return pkg, nil
	}
	imp := fl.pkg.info.LogosImporter
	if imp == nil {
		return nil, errors.New("the check recorded no importer")
	}
	return imp.Import(path)
}

func (fl *fileLower) cfKnownPackage(path string) *types2.Package {
	roots := append([]*types2.Package{fl.pkg.tpkg}, fl.pkg.ctExp.roots...)
	for _, root := range roots {
		if root.Path() == path {
			return root
		}
		for _, imp := range root.Imports() {
			if imp.Path() == path {
				return imp
			}
		}
	}
	return nil
}

func (fl *fileLower) cfCalleeObj(x syntax.Expr, sites *cfSites) *types2.Func {
	name := cfCalleeName(x)
	if name == nil {
		return nil
	}
	if sites != nil && sites.scope != nil {
		return cfScopeFunc(sites.scope, x)
	}
	fn, _ := fl.pkg.info.Uses[name].(*types2.Func)
	return fn
}

func cfScopeFunc(scope *types2.Scope, x syntax.Expr) *types2.Func {
	var name string
	switch x := x.(type) {
	case *syntax.Name:
		name = x.Value
	case *syntax.SelectorExpr:
		if pkg, ok := x.X.(*syntax.Name); !ok || pkg.Value != cfPkgMarker {
			return nil
		}
		name = x.Sel.Value
	}
	fn, _ := scope.Lookup(name).(*types2.Func)
	return fn
}

func (fl *fileLower) cfNestedSite(call *syntax.CallExpr, active cfChain, run *cfRun) (int, int, string, bool) {
	fun, indexed := call.Fun.(*syntax.IndexExpr)
	callee := call.Fun
	if indexed {
		callee = fun.X
	}
	obj := fl.cfCalleeObj(callee, run.sites)
	if obj == nil || !fl.cfIsGeneric(obj) {
		return 0, 0, "", false
	}
	pos := run.sites.at(call.Pos())
	if !indexed {
		fl.cfError(pos, "[comptime::generic_targ] comptime generic %s needs explicit type arguments here", obj.Name())
		return 0, 0, "", false
	}
	var targs []types2.Type
	for _, arg := range syntax.UnpackListExpr(fun.Index) {
		t, ok := fl.cfResolveTypeExpr(arg, active, run.sites)
		if !ok {
			fl.cfError(pos, "[comptime::generic_targ] cannot resolve a type argument of %s", obj.Name())
			return 0, 0, "", false
		}
		targs = append(targs, t)
	}
	name := fl.cfCopyFor(&cfSite{call: call, fun: fun, targs: targs, obj: obj, pos: pos})
	return offAt(run.ctx.starts, syntax.StartPos(fun)), cfExactEnd(run.ctx, fun), name, true
}

func (fl *fileLower) cfEmitCopy(text otext, site *cfSite) {
	file, errs := cfParseSource("package cfcopy\n" + text.s)
	fn := cfFirstFunc(file)
	if len(errs) > 0 || fn == nil {
		base.Fatalf("logos: cannot parse comptime copy at %s: %v", site.pos.String(), errs)
	}
	if !fl.cfCheckCopyFunc(fn, site.pos) {
		return
	}
	region := fl.adoptRegion(text, site.pos, fn.Pos().FileBase(), 2, []syntax.Node{fn})
	fl.file.DeclList = append(fl.file.DeclList, fn)
	fl.addHost(fl.orig, fl.orig, &cfHost{regions: []*cfRegion{region}, head: "\n\n", tail: "\n"})
}

func (fl *fileLower) cfCheckCopyFunc(fn *syntax.FuncDecl, pos syntax.Pos) bool {
	exp := fl.pkg.ctExp
	ctx := types2.LogosCopyContext{
		Scope:     fl.pkg.info.Scopes[fl.file],
		Funcs:     exp.copyFuncs,
		Aliases:   fl.cfAliases(),
		MaxSteps:  fl.pkg.info.LogosCTFEMaxSteps,
		NodeLimit: base.LogosComptimeNodes,
		Importer:  fl.pkg.info.LogosImporter,
		Context:   fl.pkg.info.LogosContext,
	}
	obj, errs := types2.LogosCheckCopyFunc(fl.pkg.tpkg, fl.pkg.info, ctx, fn)
	if obj != nil {
		exp.copyFuncs = append(exp.copyFuncs, obj)
	}
	for _, err := range errs {
		fl.cfError(pos, "%s", err.Msg)
	}
	return len(errs) == 0
}

func (fl *fileLower) cfRetireDefs() {
	for _, def := range fl.pkg.ctExp.defs {
		if def.file != fl.file {
			continue
		}
		if def.obj != nil && def.obj.Exported() {
			fl.cfStubDef(def)
		} else {
			fl.cfDeleteDef(def.decl)
		}
	}
}

func (fl *fileLower) cfStubDef(def *cfGenericDef) {
	fn := def.decl
	call := "panic(" + strconv.Quote("logos: comptime generic "+fn.Name.Value+" runs only through loGo/s instantiation") + ")"
	fn.Body.List = fl.reparseStmts(call, fn.Pos())
	fl.add(fl.off(syntax.StartPos(fn.Body)), fl.off(syntax.EndPos(fn.Body))+1, "{\n\t"+call+"\n}")
	fl.changed = true
}

func (fl *fileLower) cfDeleteDef(decl *syntax.FuncDecl) {
	fl.file.DeclList = slices.DeleteFunc(fl.file.DeclList, func(d syntax.Decl) bool { return d == syntax.Decl(decl) })
	fl.add(fl.cfFuncStart(fl.off(syntax.StartPos(decl))), fl.off(syntax.EndPos(decl))+1, "")
	fl.changed = true
}

func cfHasTypeParam(t types2.Type) bool {
	switch t := t.(type) {
	case *types2.TypeParam:
		return true
	case *types2.Pointer:
		return cfHasTypeParam(t.Elem())
	case *types2.Slice:
		return cfHasTypeParam(t.Elem())
	case *types2.Array:
		return cfHasTypeParam(t.Elem())
	case *types2.Map:
		return cfHasTypeParam(t.Key()) || cfHasTypeParam(t.Elem())
	case *types2.Chan:
		return cfHasTypeParam(t.Elem())
	}
	return false
}
