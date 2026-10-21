// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file expands M7-S3 comptime generics (SPEC Sec 2.5, T-205): a
// generic function using const for, val.[f], or comptime calls expands
// per instantiation into a non-generic copy in the instantiating
// package. Every copy is built from record text: the definition source
// with its type parameters, imports, and (for packages other than the
// defining one) package-level names spelled as reserved markers. Copies
// are memoized per type-argument list; an instantiation stack detects
// cycles; exported generics travel as the same record text in __.LOGOS.
// The const for machinery of constfor.go unrolls each copy.

const (
	cfPkgMarker      = "_logos_ctpkg"
	cfImportMarker   = "_logos_ctimp"
	cfComptimeMarker = "_logos_ctcomptime"
	cfTParamMarker   = "_logos_tparam"
	cfCopyInfix      = "_logosct"
)

// cfMarker spells the Nth type parameter. The reserved prefix keeps
// user code from colliding with it.
func cfMarker(n int) string {
	return cfTParamMarker + strconv.Itoa(n)
}

// cfImport is one package a record names through an import marker.
type cfImport struct {
	name string
	path string
}

// cfRecord is one comptime generic source: the definition text starting
// at the func keyword, with the markers its expansion resolves.
type cfRecord struct {
	name    string
	tparams int
	foreign bool
	imports []cfImport
	source  string
}

// text serializes a record for __.LOGOS: header lines, then the source.
func (r *cfRecord) text() string {
	var b strings.Builder
	b.WriteString("//logosct\n")
	for _, imp := range r.imports {
		fmt.Fprintf(&b, "//logosimp %s %s\n", imp.name, imp.path)
	}
	b.WriteString(r.source)
	return b.String()
}

// cfParseRecord reads one record text of __.LOGOS.
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

// cfCountTParams counts the type parameters of a record's function.
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

// cfParseSource parses text in LogosMode and returns its errors.
func cfParseSource(text string) (*syntax.File, []error) {
	var errs []error
	file, _ := syntax.Parse(syntax.NewFileBase("cfrecord.lgo"), strings.NewReader(text), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.CheckBranches|syntax.LogosMode)
	return file, errs
}

// cfFirstFunc returns the first function declaration of file.
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

// cfGenericDef is one comptime generic definition of the package.
type cfGenericDef struct {
	decl   *syntax.FuncDecl
	obj    *types2.Func
	file   *syntax.File
	record *cfRecord
}

// cfFrame is one active instantiation on the cycle stack.
type cfFrame struct {
	key   string
	label string
}

// cfExpander carries S3 expansion state across one package.
type cfExpander struct {
	defs    []*cfGenericDef
	byObj   map[*types2.Func]*cfGenericDef
	byDecl  map[*syntax.FuncDecl]*cfGenericDef
	foreign map[string]*cfRecord
	memo    map[string]string
	stack   []cfFrame
	pkgs    map[string]*types2.Package
	roots   []*types2.Package
	seq     int
	// copyFuncs holds the function object of every copy emitted so far,
	// for the checks of copies that call them.
	copyFuncs []*types2.Func
}

// newCfExpander registers the comptime generics of every file of the
// package and records the exported ones in the facts. Registration
// precedes lowering, so a file can instantiate a generic that another
// file of its package defines.
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
		f = fl.registerComptime(exp, f)
	}
	if f != nil {
		facts.SetCurrent(f)
	}
	return exp
}

// register records one definition.
func (e *cfExpander) register(def *cfGenericDef) {
	e.defs = append(e.defs, def)
	e.byDecl[def.decl] = def
	if def.obj != nil {
		e.byObj[def.obj] = def
	}
}

// frameIndex returns the stack index of key, or -1.
func (e *cfExpander) frameIndex(key string) int {
	return slices.IndexFunc(e.stack, func(f cfFrame) bool { return f.key == key })
}

// cfExpKey identifies one instantiation for memoization and cycles.
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

// cfShortKeys spells targs with bare names for diagnostics.
func cfShortKeys(targs []types2.Type) string {
	parts := make([]string, len(targs))
	for i, t := range targs {
		parts[i] = types2.TypeString(t, func(*types2.Package) string { return "" })
	}
	return strings.Join(parts, ", ")
}

// registerComptime registers the comptime generics of one file with exp
// and records the exported ones in f.
func (fl *fileLower) registerComptime(exp *cfExpander, f *facts.Facts) *facts.Facts {
	for _, def := range fl.cfCollectGenerics() {
		def.record = fl.cfLocalRecord(def)
		exp.register(def)
		if def.obj == nil || !def.obj.Exported() {
			continue
		}
		if f == nil {
			f = &facts.Facts{}
		}
		rec := fl.cfForeignRecord(def)
		f.Comptimes = append(f.Comptimes, facts.ComptimeFact{Object: rec.name, Source: []byte(rec.text())})
	}
	return f
}

// cfCollectGenerics finds the comptime-generic definitions of one file.
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

// cfUsesComptime reports whether fn uses const for, a field projection,
// or a comptime package function.
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

// cfIsComptimePkg reports whether x names the comptime package.
func (fl *fileLower) cfIsComptimePkg(x syntax.Expr) bool {
	name, ok := x.(*syntax.Name)
	if !ok {
		return false
	}
	pn, ok := fl.pkg.info.Uses[name].(*types2.PkgName)
	return ok && pn.Imported().Path() == "comptime"
}

// cfLocalRecord builds the record used inside the defining package.
func (fl *fileLower) cfLocalRecord(def *cfGenericDef) *cfRecord {
	return fl.cfRecordText(def, "")
}

// cfForeignRecord builds the record that travels in __.LOGOS: names of
// the defining package carry the package marker.
func (fl *fileLower) cfForeignRecord(def *cfGenericDef) *cfRecord {
	return fl.cfRecordText(def, cfPkgMarker)
}

// cfRecordBuilder collects the marker splices of one definition.
type cfRecordBuilder struct {
	fl        *fileLower
	qualifier string
	tpIndex   map[types2.Object]int
	imports   []cfImport
	subs      []splice
}

// cfRecordText builds the record text of one definition. The definition
// AST is never modified.
func (fl *fileLower) cfRecordText(def *cfGenericDef, qualifier string) *cfRecord {
	b := &cfRecordBuilder{fl: fl, qualifier: qualifier, tpIndex: make(map[types2.Object]int)}
	for i, f := range def.decl.TParamList {
		if o, ok := fl.pkg.info.Defs[f.Name]; ok {
			b.tpIndex[o] = i
		}
	}
	syntax.Inspect(def.decl, b.visit)
	start := cfFuncStart(fl.src, fl.off(syntax.StartPos(def.decl)))
	end := fl.off(syntax.EndPos(def.decl)) + 1
	out, err := applySplices(fl.src[start:end], shiftSplices(b.subs, -start))
	if err != nil {
		base.Fatalf("logos: overlapping comptime record at %s", def.decl.Pos().String())
	}
	return &cfRecord{
		name:    def.decl.Name.Value,
		tparams: len(def.decl.TParamList),
		foreign: qualifier != "",
		imports: b.imports,
		source:  string(out),
	}
}

// visit records the splice of one name use. A `const if` folds its
// constant conditions and visits only its branches.
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

// spelling returns the marker text of one used object.
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

// packageLevel reports whether obj is a package-level object of the
// defining package.
func (b *cfRecordBuilder) packageLevel(obj types2.Object) bool {
	pkg := b.fl.pkg.tpkg
	return obj.Pkg() == pkg && pkg.Scope().Lookup(obj.Name()) == obj
}

// packageMarker returns the marker of one imported package.
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

// cfFuncStart moves a FuncDecl offset back over the func keyword: the
// parser positions the declaration after it.
func cfFuncStart(src []byte, off int) int {
	i := off
	for i > 0 && (src[i-1] == ' ' || src[i-1] == '\t' || src[i-1] == '\n' || src[i-1] == '\r') {
		i--
	}
	if i >= 4 && string(src[i-4:i]) == "func" {
		return i - 4
	}
	return off
}

// shiftSplices re-bases splices by off.
func shiftSplices(subs []splice, off int) []splice {
	out := make([]splice, 0, len(subs))
	for _, s := range subs {
		out = append(out, splice{start: s.start + off, end: s.end + off, text: s.text, owner: s.owner})
	}
	return out
}

// cfSite is one instantiation site: the expression naming the generic
// (an index expression or a bare callee) and its resolved arguments.
type cfSite struct {
	call  *syntax.CallExpr
	fun   syntax.Expr
	targs []types2.Type
	obj   *types2.Func
	pos   syntax.Pos
}

// expandComptime runs the S3 phase for one file: expand instantiation
// sites, then retire the definitions.
func (fl *fileLower) expandComptime() {
	fl.cfRejectGenericMethods()
	parents := cfParents(fl.file)
	for _, site := range fl.cfScanSites(parents) {
		fl.expandSite(site)
	}
	fl.cfRetireDefs()
}

// lowerComptime lowers the comptime features of one file: generic
// copies first, then const for unrolling. The imports go later, in
// endComptime, once pruning has removed the dead branches.
func (fl *fileLower) lowerComptime() {
	fl.expandComptime()
	fl.lowerConstFor()
}

// endComptime drops the import of the virtual package once no use of
// it remains in the lowered tree, then blanks the imports whose package
// the expansion erased. It runs after function lowering, because a dead
// `const if` branch still holds comptime uses until that pass prunes it.
func (fl *fileLower) endComptime() {
	fl.dropComptimeImport(nil)
	if fl.changed {
		fl.cfDemoteErasedImports()
	}
}

// cfDemoteErasedImports turns every import whose package no longer
// appears in the lowered file into a blank import, in the native tree
// and in the eject text. A package named only by an erased const for
// range (`comptime.Fields[pkg.T]()`) or only by the body of a retired
// comptime generic is a legal import of the loGo/s source, but plain Go
// rejects it as unused.
func (fl *fileLower) cfDemoteErasedImports() {
	used := fl.cfNamesInUse()
	for _, decl := range fl.file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok || imp.Path == nil {
			continue
		}
		if local, ok := fl.cfImportLocalName(imp); ok && !used[local] {
			fl.cfBlankImport(imp)
		}
	}
}

// cfNamesInUse collects the identifiers the lowered file still uses as
// values or types: selector members and the import declarations
// themselves do not count.
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

// cfImportLocalName returns the name an import binds in the file. Blank
// and dot imports bind none, and the virtual comptime package is left
// to dropComptimeImport.
func (fl *fileLower) cfImportLocalName(imp *syntax.ImportDecl) (string, bool) {
	var obj types2.Object
	if imp.LocalPkgName != nil {
		obj = fl.pkg.info.Defs[imp.LocalPkgName]
	} else {
		obj = fl.pkg.info.Implicits[imp]
	}
	pn, ok := obj.(*types2.PkgName)
	if !ok || pn.Name() == "_" || pn.Name() == "." || pn.Imported().Path() == "comptime" {
		return "", false
	}
	return pn.Name(), true
}

// cfBlankImport rewrites one import to bind the blank identifier.
func (fl *fileLower) cfBlankImport(imp *syntax.ImportDecl) {
	if imp.LocalPkgName != nil {
		start := fl.off(imp.LocalPkgName.Pos())
		fl.add(start, start+len(imp.LocalPkgName.Value), "_")
	} else {
		start := fl.off(imp.Path.Pos())
		fl.add(start, start, "_ ")
	}
	imp.LocalPkgName = nameOf(imp.Path.Pos(), "_")
	fl.changed = true
}

// cfRejectGenericMethods reports methods of generic types that use
// const for over a type parameter: a copy has no method to hang on.
func (fl *fileLower) cfRejectGenericMethods() {
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if !ok || fn.Recv == nil || fn.Body == nil || !fl.cfHasGenericConstFor(fn.Body) {
			continue
		}
		fl.cfError(fn.Pos(), "[comptime::generic_method] method %s of a generic type cannot be a comptime generic", fn.Name.Value)
	}
}

// cfHasGenericConstFor reports a const for over a type parameter in body.
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

// cfScanSites finds instantiation sites of the file: calls resolving to
// comptime generics outside definitions and outside const for bodies
// (those expand where the unroller substitutes their type arguments).
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

// cfSiteOf resolves one call to an instantiation site, or nil.
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

// cfHandledElsewhere reports whether a call sits inside a comptime
// definition or a const for body.
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

// cfCalleeName returns the name node of a callee expression: the
// function name of `f`, `pkg.f`, `f[T]`, or `pkg.f[T]`.
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

// typeArgsOf reads the substituted types of one recorded instance.
func typeArgsOf(l *types2.TypeList) []types2.Type {
	out := make([]types2.Type, l.Len())
	for i := range l.Len() {
		out[i] = l.At(i)
	}
	return out
}

// cfIsGeneric reports whether obj is a comptime generic: a definition
// of this package, or a function with a comptime record in the facts of
// its package.
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
	return slices.ContainsFunc(f.Comptimes, func(c facts.ComptimeFact) bool { return c.Object == obj.Name() })
}

// cfRecordFor resolves the expansion record of one generic.
func (fl *fileLower) cfRecordFor(obj *types2.Func) (*cfRecord, bool) {
	exp := fl.pkg.ctExp
	if def, ok := exp.byObj[obj]; ok {
		return def.record, true
	}
	key := obj.Pkg().Path() + "\x00" + obj.Name()
	if rec, ok := exp.foreign[key]; ok {
		return rec, true
	}
	f, ok := facts.Lookup(obj.Pkg().Path())
	if !ok {
		return nil, false
	}
	for _, c := range f.Comptimes {
		if c.Object != obj.Name() {
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

// expandSite rewrites one instantiation call to name its copy.
func (fl *fileLower) expandSite(site *cfSite) {
	name := fl.cfCopyFor(site)
	ctx := cfCtx{src: fl.src, starts: fl.start}
	fl.add(offAt(fl.start, syntax.StartPos(site.fun)), cfExactEnd(ctx, site.fun), name)
	site.call.Fun = nameOf(syntax.StartPos(site.fun), name)
	fl.changed = true
}

// cfCopyName returns the copy name of the Nth expansion of obj.
func cfCopyName(obj *types2.Func, seq int) string {
	return obj.Name() + cfCopyInfix + strconv.Itoa(seq)
}

// cfCopyFor returns the memoized copy name of one instantiation,
// expanding it once per type-argument list. The stack detects cycles
// (#129): a key re-entered while active fails with the chain.
func (fl *fileLower) cfCopyFor(site *cfSite) string {
	exp := fl.pkg.ctExp
	key := cfExpKey(site.obj.Name(), site.obj.Pkg().Path(), site.targs)
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
	if ok && base.Errors() == reported { // A copy built under a diagnostic names copies that never emitted: checking it would only cascade.
		fl.cfEmitCopy(text, site)
	}
	return name
}

// cfChainText spells an instantiation chain, closing it on label.
func cfChainText(frames []cfFrame, label string) string {
	parts := make([]string, 0, len(frames)+1)
	for _, f := range frames {
		parts = append(parts, f.label)
	}
	return strings.Join(append(parts, label), " -> ")
}

// cfStripTParams removes the type-parameter clause of a record head:
// `func Name[T any](...)` becomes `func Name(...)`.
func cfStripTParams(src, name string) (string, bool) {
	head := "func " + name
	if !strings.HasPrefix(src, head+"[") {
		return "", false
	}
	depth := 0
	for i := len(head); i < len(src); i++ {
		switch src[i] {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return head + src[i+1:], true
			}
		}
	}
	return "", false
}

// cfBuildCopy builds the plain-Go text of one instantiation: type
// parameters, imports, and package markers spelled out, const for
// statements unrolled, nested comptime generic calls pointed at their
// own copies. It reports false after a diagnostic.
func (fl *fileLower) cfBuildCopy(site *cfSite, rec *cfRecord, copyName string) (string, bool) {
	body, ok := cfStripTParams(rec.source, rec.name)
	if !ok {
		fl.cfError(site.pos, "[comptime::generic_targ] malformed comptime source of %s", rec.name)
		return "", false
	}
	sites, ok := fl.cfCopySites(site, rec)
	if !ok {
		return "", false
	}
	const prefix = "package cfcopy\n"
	src := []byte(prefix + body)
	file, errs := cfParseSource(string(src))
	fn := cfFirstFunc(file)
	if len(errs) > 0 || fn == nil {
		base.Fatalf("logos: cannot parse comptime source of %s: %v", rec.name, errs)
	}
	run := &cfRun{ctx: cfCtx{src: src, starts: lineStarts(src), stamp: site.pos}, fieldsOf: fl.cfFieldsResolver(sites), sites: sites}
	out, err := applySplices(src, fl.cfCopySplices(fn, copyName, run))
	if err != nil {
		base.Fatalf("logos: overlapping comptime copy %s: %v", copyName, err)
	}
	return string(out[len(prefix):]), true
}

// cfCopySplices collects the splices that turn one record function into
// its copy: the new name, the unrolled const fors, and the leaf
// substitutions everywhere else.
func (fl *fileLower) cfCopySplices(fn *syntax.FuncDecl, copyName string, run *cfRun) []splice {
	ctx := run.ctx
	nameStart := offAt(ctx.starts, fn.Name.Pos())
	subs := []splice{{start: nameStart, end: nameStart + len(fn.Name.Value), text: copyName}}
	var covered []cfSpan
	for _, cf := range fl.cfInnerConstFors(fn.Body, nil, run.sites) {
		s := offAt(ctx.starts, syntax.StartPos(cf))
		e := offAt(ctx.starts, syntax.EndPos(cf)) + 1
		stmts, text := fl.unrollConstForFields(cf, run.fieldsOf(cf, nil), run)
		fl.cfCountNodes(stmts, run.sites.at(cf.Pos()))
		subs = append(subs, splice{start: s, end: e, text: text})
		covered = append(covered, cfSpan{s, e})
	}
	for _, sub := range fl.cfLeafSubs(fn, nil, covered, run) {
		subs = append(subs, splice{start: sub.start, end: sub.end, text: sub.text})
	}
	return subs
}

// cfCopySites builds the substitution context of one copy: marker
// spellings and types, and the scope that resolves package-level names
// of the record's package.
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

// cfLazySpelling memoizes one marker spelling computed on first use.
// Spelling a package registers a synthesized import, so a marker the
// copy never reaches (a type argument that only appears in an erased
// const for range) must not spell at all. fail runs once when the
// spelling cannot be produced.
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

// cfImportQualifier spells the package at path in this file: its own
// import name when the file imports it, a synthesized import otherwise.
// The synthesized alias binds the complete package, because the copy
// check looks members up through it. It reports the diagnostic at pos and
// false when the package cannot load.
func (fl *fileLower) cfImportQualifier(path string, pos syntax.Pos) (string, bool) {
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

// cfLoadPackage returns the complete package at path. A package the
// compile already holds is complete once something imported it. The
// import list of an export-data package only names descriptors whose
// scopes hold the few objects that export data mentions, and a copy
// looks any member up through its alias, so a descriptor opens through
// the importer of the check, which fills the same package object in
// place.
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

// cfKnownPackage returns the package object at path that the compile
// already holds: the package under lowering, a package that defines a
// comptime generic being instantiated, or a package either of them
// imports. It returns nil when none of them names path.
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

// cfCalleeObj resolves a callee expression to its function: through the
// checker's uses for file code, through the record scope for copies.
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

// cfScopeFunc resolves `f` or `<pkg marker>.f` in the record scope.
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

// cfNestedSite expands a comptime generic call inside a const for body
// or a copy: it resolves the type arguments under the active field
// variables and returns the span of the generic expression with the name
// of its copy.
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

// cfEmitCopy appends one finished copy to the file. The copy is checked
// on its own, so its lowering reads the types of its own nodes, and it
// lives in a region of its own, so its edits never meet those of another
// copy. After lowering the native tree takes the instantiation position
// on every node and every scope boundary, and the eject text takes the
// finished region at the end of the file. A copy the check rejects
// reports its diagnostics at the instantiation and never joins the tree:
// its nodes lack the facts the lowering passes read.
func (fl *fileLower) cfEmitCopy(text string, site *cfSite) {
	file, errs := cfParseSource("package cfcopy\n" + text)
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

// cfCheckCopyFunc types one copy as a function of the package under the
// scope of the instantiating file, and remembers its object so copies
// emitted later can call it, even when the check rejects the copy: a
// caller then checks against the signature instead of cascading. It
// reports each hard diagnostic of the check at pos, the instantiation
// the copy answers for, and returns whether the copy is valid.
func (fl *fileLower) cfCheckCopyFunc(fn *syntax.FuncDecl, pos syntax.Pos) bool {
	exp := fl.pkg.ctExp
	ctx := types2.LogosCopyContext{Scope: fl.pkg.info.Scopes[fl.file], Funcs: exp.copyFuncs, Aliases: fl.cfAliases()}
	obj, errs := types2.LogosCheckCopyFunc(fl.pkg.tpkg, fl.pkg.info, ctx, fn)
	if obj != nil {
		exp.copyFuncs = append(exp.copyFuncs, obj)
	}
	for _, err := range errs {
		fl.cfError(pos, "%s", err.Msg)
	}
	return len(errs) == 0
}

// cfRetireDefs removes the comptime generics of this file from the
// output: exported ones keep their signature with a trapping body so
// importers still type-check, the others go away.
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

// cfStubDef replaces the body of an exported comptime generic.
func (fl *fileLower) cfStubDef(def *cfGenericDef) {
	fn := def.decl
	call := "panic(" + strconv.Quote("logos: comptime generic "+fn.Name.Value+" runs only through loGo/s instantiation") + ")"
	fn.Body.List = fl.reparseStmts(call, fn.Pos())
	fl.add(fl.off(syntax.StartPos(fn.Body)), fl.off(syntax.EndPos(fn.Body))+1, "{\n\t"+call+"\n}")
	fl.changed = true
}

// cfDeleteDef drops one comptime generic from the tree and the text.
func (fl *fileLower) cfDeleteDef(decl *syntax.FuncDecl) {
	fl.file.DeclList = slices.DeleteFunc(fl.file.DeclList, func(d syntax.Decl) bool { return d == syntax.Decl(decl) })
	fl.add(cfFuncStart(fl.src, fl.off(syntax.StartPos(decl))), fl.off(syntax.EndPos(decl))+1, "")
	fl.changed = true
}

// cfHasTypeParam reports whether t mentions a type parameter.
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
