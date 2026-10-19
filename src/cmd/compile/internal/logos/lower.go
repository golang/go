// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"path/filepath"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Lower rewrites the loGo/s nodes of files to plain Go syntax and records
// the matching eject edits. pkg and info come from a successful types2
// check with Config.Logos; src holds the original bytes of the files Lower
// may rewrite. Files without an entry in src are left alone. Lower mutates
// the files in place and reports whether anything changed. A surviving
// TryExpr or ErrdeferStmt is an internal error.
func Lower(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) (bool, map[*syntax.File][]Edit) {
	shared := &packageLower{info: info}
	edits := make(map[*syntax.File][]Edit)
	changed := false
	for _, file := range files {
		data, ok := src[file]
		if !ok {
			continue
		}
		fl := &fileLower{
			pkg:   shared,
			file:  file,
			src:   data,
			base:  file.Pos().FileBase(),
			start: lineStarts(data),
		}
		fl.baseName = filepath.Base(fl.base.Filename())
		fl.lowerFile()
		if fl.changed {
			changed = true
			edits[file] = editsOf(data, fl.base, fl.splices)
		}
	}
	for _, file := range files {
		rejectSurvivors(file)
	}
	return changed, edits
}

// packageLower carries the state shared by every file of one package.
type packageLower struct {
	info    *types2.Info
	siteSeq int
}

// fileLower carries the state of lowering one file.
type fileLower struct {
	pkg       *packageLower
	file      *syntax.File
	src       []byte
	start     []int
	base      *syntax.PosBase
	baseName  string
	splices   []splice
	changed   bool
	needRT    bool
	siteDecls []*syntax.VarDecl
}

// off converts a valid file position to a byte offset.
func (fl *fileLower) off(pos syntax.Pos) int {
	return fl.start[pos.Line()-1] + int(pos.Col()) - 1
}

// add records one replacement splice.
func (fl *fileLower) add(start, end int, text string) {
	fl.splices = append(fl.splices, splice{start: start, end: end, text: text})
	fl.changed = true
}

// textOf returns the source of s with nested splices substituted.
func (fl *fileLower) textOf(s span) string {
	kept, inner := fl.splitSplices(s)
	fl.splices = kept
	sub := fl.src[s.start:s.end]
	adj := make([]splice, len(inner))
	for i, sp := range inner {
		adj[i] = splice{start: sp.start - s.start, end: sp.end - s.start, text: sp.text}
	}
	out, err := applySplices(sub, adj)
	if err != nil {
		base.Fatalf("logos: overlapping edits at byte offset %d", s.start)
	}
	return string(out)
}

// splitSplices divides pending splices into those outside s and those
// strictly inside it. Zero-width splices on the boundary stay outside.
func (fl *fileLower) splitSplices(s span) ([]splice, []splice) {
	var kept, inner []splice
	for _, sp := range fl.splices {
		if sp.start >= s.start && sp.end <= s.end && (sp.end > sp.start || (sp.start > s.start && sp.end < s.end)) {
			inner = append(inner, sp)
		} else {
			kept = append(kept, sp)
		}
	}
	return kept, inner
}

// newSite registers one package-level trace site and returns its spelling.
func (fl *fileLower) newSite(pos syntax.Pos, line uint) string {
	name := fmt.Sprintf("_logos_site%d", fl.pkg.siteSeq)
	fl.pkg.siteSeq++
	node := nameOf(pos, name)
	fl.siteDecls = append(fl.siteDecls, siteDecl(pos, node, fl.baseName, line))
	fl.needRT = true
	return name
}

// lowerFile lowers every function body of one file.
func (fl *fileLower) lowerFile() {
	for _, decl := range fl.file.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if ok && fn.Body != nil {
			newFuncLower(fl, fn.Type, fn.Body).run()
			continue
		}
		scanForLiterals(fl, decl)
	}
	if fl.needRT {
		fl.file.DeclList = append([]syntax.Decl{rtImport(fl.file.Pos())}, fl.file.DeclList...)
		fl.addHeaderSplice()
	}
	for _, decl := range fl.siteDecls {
		fl.file.DeclList = append(fl.file.DeclList, decl)
	}
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
	fl.add(ins, ins, "\n\nimport "+rtAlias+" "+fmt.Sprintf("%q", rtImportPath))
	if len(fl.siteDecls) > 0 {
		text := "\n\n"
		for i, decl := range fl.siteDecls {
			if i > 0 {
				text += "\n"
			}
			text += "var " + decl.NameList[0].Value + " = " + rtAlias + ".Site{" + siteText(decl) + "}\n"
		}
		fl.add(len(fl.src), len(fl.src), text)
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
	fl          *fileLower
	ft          *syntax.FuncType
	body        *syntax.BlockStmt
	specs       []resultSpec
	errName     string
	hasDeferTry bool
	hasErrdefer bool
	okName      string
	errSeq      int
	funSeq      int
	argSeq      int
	tmpSeq      int
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
	fc.detectCleanups()
	if fc.hasDeferTry || fc.hasErrdefer {
		fc.ensureNamed()
	}
	if fc.hasErrdefer {
		fc.armOkFlag()
	}
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
		}
		return true
	})
}

// ensureNamed gives every result an assignable name when a defer? or an
// errdefer must reference the error result.
func (fc *funcLower) ensureNamed() {
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
		case *syntax.TryExpr, *syntax.ErrdeferStmt:
			base.Fatalf("logos: %T survived lowering at %s", n, n.Pos().String())
		}
		return true
	})
}
