// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"go/constant"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// parseFiles concurrently parses files into *syntax.File structures.
// Each declaration in every *syntax.File is converted to a syntax tree
// and its root represented by *Node is appended to xtop.
// Returns the total count of parsed lines.
func parseFiles(filenames []string) uint {
	noders := make([]*noder, 0, len(filenames))
	// Limit the number of simultaneously open files.
	sem := make(chan struct{}, runtime.GOMAXPROCS(0)+10)

	for _, filename := range filenames {
		p := &noder{
			basemap: make(map[*syntax.PosBase]*src.PosBase),
			err:     make(chan syntax.Error),
		}
		noders = append(noders, p)

		go func(filename string) {
			sem <- struct{}{}
			defer func() { <-sem }()
			defer close(p.err)
			base := syntax.NewFileBase(filename)

			f, err := os.Open(filename)
			if err != nil {
				p.error(syntax.Error{Msg: err.Error()})
				return
			}
			defer f.Close()

			p.file, _ = syntax.Parse(base, f, p.error, p.pragma, syntax.CheckBranches) // errors are tracked via p.error
		}(filename)
	}

	var lines uint
	for _, p := range noders {
		for e := range p.err {
			p.errorAt(e.Pos, "%s", e.Msg)
		}

		p.node()
		lines += p.file.Lines
		p.file = nil // release memory

		if base.SyntaxErrors() != 0 {
			base.ErrorExit()
		}
		// Always run testdclstack here, even when debug_dclstack is not set, as a sanity measure.
		testdclstack()
	}

	for _, p := range noders {
		p.processPragmas()
	}

	ir.LocalPkg.Height = myheight

	return lines
}

// makeSrcPosBase translates from a *syntax.PosBase to a *src.PosBase.
func (p *noder) makeSrcPosBase(b0 *syntax.PosBase) *src.PosBase {
	// fast path: most likely PosBase hasn't changed
	if p.basecache.last == b0 {
		return p.basecache.base
	}

	b1, ok := p.basemap[b0]
	if !ok {
		fn := b0.Filename()
		if b0.IsFileBase() {
			b1 = src.NewFileBase(fn, absFilename(fn))
		} else {
			// line directive base
			p0 := b0.Pos()
			p0b := p0.Base()
			if p0b == b0 {
				panic("infinite recursion in makeSrcPosBase")
			}
			p1 := src.MakePos(p.makeSrcPosBase(p0b), p0.Line(), p0.Col())
			b1 = src.NewLinePragmaBase(p1, fn, fileh(fn), b0.Line(), b0.Col())
		}
		p.basemap[b0] = b1
	}

	// update cache
	p.basecache.last = b0
	p.basecache.base = b1

	return b1
}

func (p *noder) makeXPos(pos syntax.Pos) (_ src.XPos) {
	return base.Ctxt.PosTable.XPos(src.MakePos(p.makeSrcPosBase(pos.Base()), pos.Line(), pos.Col()))
}

func (p *noder) errorAt(pos syntax.Pos, format string, args ...interface{}) {
	base.ErrorfAt(p.makeXPos(pos), format, args...)
}

// TODO(gri) Can we eliminate fileh in favor of absFilename?
func fileh(name string) string {
	return objabi.AbsFile("", name, base.Flag.TrimPath)
}

func absFilename(name string) string {
	return objabi.AbsFile(base.Ctxt.Pathname, name, base.Flag.TrimPath)
}

// noder transforms package syntax's AST into a Node tree.
type noder struct {
	basemap   map[*syntax.PosBase]*src.PosBase
	basecache struct {
		last *syntax.PosBase
		base *src.PosBase
	}

	file           *syntax.File
	linknames      []linkname
	pragcgobuf     [][]string
	err            chan syntax.Error
	scope          ir.ScopeID
	importedUnsafe bool
	importedEmbed  bool

	// scopeVars is a stack tracking the number of variables declared in the
	// current function at the moment each open scope was opened.
	scopeVars []int

	lastCloseScopePos syntax.Pos
}

func (p *noder) funcBody(fn *ir.Func, block *syntax.BlockStmt) {
	oldScope := p.scope
	p.scope = 0
	funchdr(fn)

	if block != nil {
		body := p.stmts(block.List)
		if body == nil {
			body = []ir.Node{ir.Nod(ir.OBLOCK, nil, nil)}
		}
		fn.PtrBody().Set(body)

		base.Pos = p.makeXPos(block.Rbrace)
		fn.Endlineno = base.Pos
	}

	funcbody()
	p.scope = oldScope
}

func (p *noder) openScope(pos syntax.Pos) {
	types.Markdcl()

	if trackScopes {
		Curfn.Parents = append(Curfn.Parents, p.scope)
		p.scopeVars = append(p.scopeVars, len(Curfn.Dcl))
		p.scope = ir.ScopeID(len(Curfn.Parents))

		p.markScope(pos)
	}
}

func (p *noder) closeScope(pos syntax.Pos) {
	p.lastCloseScopePos = pos
	types.Popdcl()

	if trackScopes {
		scopeVars := p.scopeVars[len(p.scopeVars)-1]
		p.scopeVars = p.scopeVars[:len(p.scopeVars)-1]
		if scopeVars == len(Curfn.Dcl) {
			// no variables were declared in this scope, so we can retract it.

			if int(p.scope) != len(Curfn.Parents) {
				base.Fatalf("scope tracking inconsistency, no variables declared but scopes were not retracted")
			}

			p.scope = Curfn.Parents[p.scope-1]
			Curfn.Parents = Curfn.Parents[:len(Curfn.Parents)-1]

			nmarks := len(Curfn.Marks)
			Curfn.Marks[nmarks-1].Scope = p.scope
			prevScope := ir.ScopeID(0)
			if nmarks >= 2 {
				prevScope = Curfn.Marks[nmarks-2].Scope
			}
			if Curfn.Marks[nmarks-1].Scope == prevScope {
				Curfn.Marks = Curfn.Marks[:nmarks-1]
			}
			return
		}

		p.scope = Curfn.Parents[p.scope-1]

		p.markScope(pos)
	}
}

func (p *noder) markScope(pos syntax.Pos) {
	xpos := p.makeXPos(pos)
	if i := len(Curfn.Marks); i > 0 && Curfn.Marks[i-1].Pos == xpos {
		Curfn.Marks[i-1].Scope = p.scope
	} else {
		Curfn.Marks = append(Curfn.Marks, ir.Mark{Pos: xpos, Scope: p.scope})
	}
}

// closeAnotherScope is like closeScope, but it reuses the same mark
// position as the last closeScope call. This is useful for "for" and
// "if" statements, as their implicit blocks always end at the same
// position as an explicit block.
func (p *noder) closeAnotherScope() {
	p.closeScope(p.lastCloseScopePos)
}

// linkname records a //go:linkname directive.
type linkname struct {
	pos    syntax.Pos
	local  string
	remote string
}

func (p *noder) node() {
	types.Block = 1
	p.importedUnsafe = false
	p.importedEmbed = false

	p.setlineno(p.file.PkgName)
	mkpackage(p.file.PkgName.Value)

	if pragma, ok := p.file.Pragma.(*Pragma); ok {
		pragma.Flag &^= ir.GoBuildPragma
		p.checkUnused(pragma)
	}

	xtop = append(xtop, p.decls(p.file.DeclList)...)

	base.Pos = src.NoXPos
	clearImports()
}

func (p *noder) processPragmas() {
	for _, l := range p.linknames {
		if !p.importedUnsafe {
			p.errorAt(l.pos, "//go:linkname only allowed in Go files that import \"unsafe\"")
			continue
		}
		n := ir.AsNode(lookup(l.local).Def)
		if n == nil || n.Op() != ir.ONAME {
			// TODO(mdempsky): Change to p.errorAt before Go 1.17 release.
			// base.WarnfAt(p.makeXPos(l.pos), "//go:linkname must refer to declared function or variable (will be an error in Go 1.17)")
			continue
		}
		if n.Sym().Linkname != "" {
			p.errorAt(l.pos, "duplicate //go:linkname for %s", l.local)
			continue
		}
		n.Sym().Linkname = l.remote
	}

	// The linker expects an ABI0 wrapper for all cgo-exported
	// functions.
	for _, prag := range p.pragcgobuf {
		switch prag[0] {
		case "cgo_export_static", "cgo_export_dynamic":
			if symabiRefs == nil {
				symabiRefs = make(map[string]obj.ABI)
			}
			symabiRefs[prag[1]] = obj.ABI0
		}
	}

	pragcgobuf = append(pragcgobuf, p.pragcgobuf...)
}

func (p *noder) decls(decls []syntax.Decl) (l []ir.Node) {
	var cs constState

	for _, decl := range decls {
		p.setlineno(decl)
		switch decl := decl.(type) {
		case *syntax.ImportDecl:
			p.importDecl(decl)

		case *syntax.VarDecl:
			l = append(l, p.varDecl(decl)...)

		case *syntax.ConstDecl:
			l = append(l, p.constDecl(decl, &cs)...)

		case *syntax.TypeDecl:
			l = append(l, p.typeDecl(decl))

		case *syntax.FuncDecl:
			l = append(l, p.funcDecl(decl))

		default:
			panic("unhandled Decl")
		}
	}

	return
}

func (p *noder) importDecl(imp *syntax.ImportDecl) {
	if imp.Path.Bad {
		return // avoid follow-on errors if there was a syntax error
	}

	if pragma, ok := imp.Pragma.(*Pragma); ok {
		p.checkUnused(pragma)
	}

	ipkg := importfile(p.basicLit(imp.Path))
	if ipkg == nil {
		if base.Errors() == 0 {
			base.Fatalf("phase error in import")
		}
		return
	}

	if ipkg == unsafepkg {
		p.importedUnsafe = true
	}
	if ipkg.Path == "embed" {
		p.importedEmbed = true
	}

	if !ipkg.Direct {
		sourceOrderImports = append(sourceOrderImports, ipkg)
	}
	ipkg.Direct = true

	var my *types.Sym
	if imp.LocalPkgName != nil {
		my = p.name(imp.LocalPkgName)
	} else {
		my = lookup(ipkg.Name)
	}

	pack := ir.NewPkgName(p.pos(imp), my, ipkg)

	switch my.Name {
	case ".":
		importdot(ipkg, pack)
		return
	case "init":
		base.ErrorfAt(pack.Pos(), "cannot import package as init - init must be a func")
		return
	case "_":
		return
	}
	if my.Def != nil {
		redeclare(pack.Pos(), my, "as imported package name")
	}
	my.Def = pack
	my.Lastlineno = pack.Pos()
	my.Block = 1 // at top level
}

func (p *noder) varDecl(decl *syntax.VarDecl) []ir.Node {
	names := p.declNames(decl.NameList)
	typ := p.typeExprOrNil(decl.Type)

	var exprs []ir.Node
	if decl.Values != nil {
		exprs = p.exprList(decl.Values)
	}

	if pragma, ok := decl.Pragma.(*Pragma); ok {
		if len(pragma.Embeds) > 0 {
			if !p.importedEmbed {
				// This check can't be done when building the list pragma.Embeds
				// because that list is created before the noder starts walking over the file,
				// so at that point it hasn't seen the imports.
				// We're left to check now, just before applying the //go:embed lines.
				for _, e := range pragma.Embeds {
					p.errorAt(e.Pos, "//go:embed only allowed in Go files that import \"embed\"")
				}
			} else {
				exprs = varEmbed(p, names, typ, exprs, pragma.Embeds)
			}
			pragma.Embeds = nil
		}
		p.checkUnused(pragma)
	}

	p.setlineno(decl)
	return variter(names, typ, exprs)
}

// constState tracks state between constant specifiers within a
// declaration group. This state is kept separate from noder so nested
// constant declarations are handled correctly (e.g., issue 15550).
type constState struct {
	group  *syntax.Group
	typ    ir.Ntype
	values []ir.Node
	iota   int64
}

func (p *noder) constDecl(decl *syntax.ConstDecl, cs *constState) []ir.Node {
	if decl.Group == nil || decl.Group != cs.group {
		*cs = constState{
			group: decl.Group,
		}
	}

	if pragma, ok := decl.Pragma.(*Pragma); ok {
		p.checkUnused(pragma)
	}

	names := p.declNames(decl.NameList)
	typ := p.typeExprOrNil(decl.Type)

	var values []ir.Node
	if decl.Values != nil {
		values = p.exprList(decl.Values)
		cs.typ, cs.values = typ, values
	} else {
		if typ != nil {
			base.Errorf("const declaration cannot have type without expression")
		}
		typ, values = cs.typ, cs.values
	}

	nn := make([]ir.Node, 0, len(names))
	for i, n := range names {
		n := n.(*ir.Name)
		if i >= len(values) {
			base.Errorf("missing value in const declaration")
			break
		}
		v := values[i]
		if decl.Values == nil {
			v = ir.DeepCopy(n.Pos(), v)
		}

		n.SetOp(ir.OLITERAL)
		declare(n, dclcontext)

		n.Ntype = typ
		n.Defn = v
		n.SetIota(cs.iota)

		nn = append(nn, p.nod(decl, ir.ODCLCONST, n, nil))
	}

	if len(values) > len(names) {
		base.Errorf("extra expression in const declaration")
	}

	cs.iota++

	return nn
}

func (p *noder) typeDecl(decl *syntax.TypeDecl) ir.Node {
	n := p.declName(decl.Name)
	n.SetOp(ir.OTYPE)
	declare(n, dclcontext)

	// decl.Type may be nil but in that case we got a syntax error during parsing
	typ := p.typeExprOrNil(decl.Type)

	n.Ntype = typ
	n.SetAlias(decl.Alias)
	if pragma, ok := decl.Pragma.(*Pragma); ok {
		if !decl.Alias {
			n.SetPragma(pragma.Flag & TypePragmas)
			pragma.Flag &^= TypePragmas
		}
		p.checkUnused(pragma)
	}

	nod := p.nod(decl, ir.ODCLTYPE, n, nil)
	if n.Alias() && !langSupported(1, 9, ir.LocalPkg) {
		base.ErrorfAt(nod.Pos(), "type aliases only supported as of -lang=go1.9")
	}
	return nod
}

func (p *noder) declNames(names []*syntax.Name) []ir.Node {
	nodes := make([]ir.Node, 0, len(names))
	for _, name := range names {
		nodes = append(nodes, p.declName(name))
	}
	return nodes
}

func (p *noder) declName(name *syntax.Name) *ir.Name {
	return ir.NewDeclNameAt(p.pos(name), p.name(name))
}

func (p *noder) funcDecl(fun *syntax.FuncDecl) ir.Node {
	name := p.name(fun.Name)
	t := p.signature(fun.Recv, fun.Type)
	f := ir.NewFunc(p.pos(fun))

	if fun.Recv == nil {
		if name.Name == "init" {
			name = renameinit()
			if t.List().Len() > 0 || t.Rlist().Len() > 0 {
				base.ErrorfAt(f.Pos(), "func init must have no arguments and no return values")
			}
		}

		if ir.LocalPkg.Name == "main" && name.Name == "main" {
			if t.List().Len() > 0 || t.Rlist().Len() > 0 {
				base.ErrorfAt(f.Pos(), "func main must have no arguments and no return values")
			}
		}
	} else {
		f.Shortname = name
		name = ir.BlankNode.Sym() // filled in by typecheckfunc
	}

	f.Nname = newFuncNameAt(p.pos(fun.Name), name, f)
	f.Nname.Defn = f
	f.Nname.Ntype = t

	if pragma, ok := fun.Pragma.(*Pragma); ok {
		f.Pragma = pragma.Flag & FuncPragmas
		if pragma.Flag&ir.Systemstack != 0 && pragma.Flag&ir.Nosplit != 0 {
			base.ErrorfAt(f.Pos(), "go:nosplit and go:systemstack cannot be combined")
		}
		pragma.Flag &^= FuncPragmas
		p.checkUnused(pragma)
	}

	if fun.Recv == nil {
		declare(f.Nname, ir.PFUNC)
	}

	p.funcBody(f, fun.Body)

	if fun.Body != nil {
		if f.Pragma&ir.Noescape != 0 {
			base.ErrorfAt(f.Pos(), "can only use //go:noescape with external func implementations")
		}
	} else {
		if base.Flag.Complete || strings.HasPrefix(ir.FuncName(f), "init.") {
			// Linknamed functions are allowed to have no body. Hopefully
			// the linkname target has a body. See issue 23311.
			isLinknamed := false
			for _, n := range p.linknames {
				if ir.FuncName(f) == n.local {
					isLinknamed = true
					break
				}
			}
			if !isLinknamed {
				base.ErrorfAt(f.Pos(), "missing function body")
			}
		}
	}

	return f
}

func (p *noder) signature(recv *syntax.Field, typ *syntax.FuncType) *ir.FuncType {
	var rcvr *ir.Field
	if recv != nil {
		rcvr = p.param(recv, false, false)
	}
	return ir.NewFuncType(p.pos(typ), rcvr,
		p.params(typ.ParamList, true),
		p.params(typ.ResultList, false))
}

func (p *noder) params(params []*syntax.Field, dddOk bool) []*ir.Field {
	nodes := make([]*ir.Field, 0, len(params))
	for i, param := range params {
		p.setlineno(param)
		nodes = append(nodes, p.param(param, dddOk, i+1 == len(params)))
	}
	return nodes
}

func (p *noder) param(param *syntax.Field, dddOk, final bool) *ir.Field {
	var name *types.Sym
	if param.Name != nil {
		name = p.name(param.Name)
	}

	typ := p.typeExpr(param.Type)
	n := ir.NewField(p.pos(param), name, typ, nil)

	// rewrite ...T parameter
	if typ, ok := typ.(*ir.SliceType); ok && typ.DDD {
		if !dddOk {
			// We mark these as syntax errors to get automatic elimination
			// of multiple such errors per line (see ErrorfAt in subr.go).
			base.Errorf("syntax error: cannot use ... in receiver or result parameter list")
		} else if !final {
			if param.Name == nil {
				base.Errorf("syntax error: cannot use ... with non-final parameter")
			} else {
				p.errorAt(param.Name.Pos(), "syntax error: cannot use ... with non-final parameter %s", param.Name.Value)
			}
		}
		typ.DDD = false
		n.IsDDD = true
	}

	return n
}

func (p *noder) exprList(expr syntax.Expr) []ir.Node {
	if list, ok := expr.(*syntax.ListExpr); ok {
		return p.exprs(list.ElemList)
	}
	return []ir.Node{p.expr(expr)}
}

func (p *noder) exprs(exprs []syntax.Expr) []ir.Node {
	nodes := make([]ir.Node, 0, len(exprs))
	for _, expr := range exprs {
		nodes = append(nodes, p.expr(expr))
	}
	return nodes
}

func (p *noder) expr(expr syntax.Expr) ir.Node {
	p.setlineno(expr)
	switch expr := expr.(type) {
	case nil, *syntax.BadExpr:
		return nil
	case *syntax.Name:
		return p.mkname(expr)
	case *syntax.BasicLit:
		n := ir.NewLiteral(p.basicLit(expr))
		if expr.Kind == syntax.RuneLit {
			n.SetType(types.UntypedRune)
		}
		n.SetDiag(expr.Bad) // avoid follow-on errors if there was a syntax error
		return n
	case *syntax.CompositeLit:
		n := p.nod(expr, ir.OCOMPLIT, nil, nil)
		if expr.Type != nil {
			n.SetRight(p.expr(expr.Type))
		}
		l := p.exprs(expr.ElemList)
		for i, e := range l {
			l[i] = p.wrapname(expr.ElemList[i], e)
		}
		n.PtrList().Set(l)
		base.Pos = p.makeXPos(expr.Rbrace)
		return n
	case *syntax.KeyValueExpr:
		// use position of expr.Key rather than of expr (which has position of ':')
		return p.nod(expr.Key, ir.OKEY, p.expr(expr.Key), p.wrapname(expr.Value, p.expr(expr.Value)))
	case *syntax.FuncLit:
		return p.funcLit(expr)
	case *syntax.ParenExpr:
		return p.nod(expr, ir.OPAREN, p.expr(expr.X), nil)
	case *syntax.SelectorExpr:
		// parser.new_dotname
		obj := p.expr(expr.X)
		if obj.Op() == ir.OPACK {
			pack := obj.(*ir.PkgName)
			pack.Used = true
			return importName(pack.Pkg.Lookup(expr.Sel.Value))
		}
		n := nodSym(ir.OXDOT, obj, p.name(expr.Sel))
		n.SetPos(p.pos(expr)) // lineno may have been changed by p.expr(expr.X)
		return n
	case *syntax.IndexExpr:
		return p.nod(expr, ir.OINDEX, p.expr(expr.X), p.expr(expr.Index))
	case *syntax.SliceExpr:
		op := ir.OSLICE
		if expr.Full {
			op = ir.OSLICE3
		}
		n := p.nod(expr, op, p.expr(expr.X), nil)
		var index [3]ir.Node
		for i, x := range &expr.Index {
			if x != nil {
				index[i] = p.expr(x)
			}
		}
		n.SetSliceBounds(index[0], index[1], index[2])
		return n
	case *syntax.AssertExpr:
		return p.nod(expr, ir.ODOTTYPE, p.expr(expr.X), p.typeExpr(expr.Type))
	case *syntax.Operation:
		if expr.Op == syntax.Add && expr.Y != nil {
			return p.sum(expr)
		}
		x := p.expr(expr.X)
		if expr.Y == nil {
			return p.nod(expr, p.unOp(expr.Op), x, nil)
		}
		return p.nod(expr, p.binOp(expr.Op), x, p.expr(expr.Y))
	case *syntax.CallExpr:
		n := p.nod(expr, ir.OCALL, p.expr(expr.Fun), nil)
		n.PtrList().Set(p.exprs(expr.ArgList))
		n.SetIsDDD(expr.HasDots)
		return n

	case *syntax.ArrayType:
		var len ir.Node
		if expr.Len != nil {
			len = p.expr(expr.Len)
		}
		return ir.NewArrayType(p.pos(expr), len, p.typeExpr(expr.Elem))
	case *syntax.SliceType:
		return ir.NewSliceType(p.pos(expr), p.typeExpr(expr.Elem))
	case *syntax.DotsType:
		t := ir.NewSliceType(p.pos(expr), p.typeExpr(expr.Elem))
		t.DDD = true
		return t
	case *syntax.StructType:
		return p.structType(expr)
	case *syntax.InterfaceType:
		return p.interfaceType(expr)
	case *syntax.FuncType:
		return p.signature(nil, expr)
	case *syntax.MapType:
		return ir.NewMapType(p.pos(expr),
			p.typeExpr(expr.Key), p.typeExpr(expr.Value))
	case *syntax.ChanType:
		return ir.NewChanType(p.pos(expr),
			p.typeExpr(expr.Elem), p.chanDir(expr.Dir))

	case *syntax.TypeSwitchGuard:
		n := p.nod(expr, ir.OTYPESW, nil, p.expr(expr.X))
		if expr.Lhs != nil {
			n.SetLeft(p.declName(expr.Lhs))
			if ir.IsBlank(n.Left()) {
				base.Errorf("invalid variable name %v in type switch", n.Left())
			}
		}
		return n
	}
	panic("unhandled Expr")
}

// sum efficiently handles very large summation expressions (such as
// in issue #16394). In particular, it avoids left recursion and
// collapses string literals.
func (p *noder) sum(x syntax.Expr) ir.Node {
	// While we need to handle long sums with asymptotic
	// efficiency, the vast majority of sums are very small: ~95%
	// have only 2 or 3 operands, and ~99% of string literals are
	// never concatenated.

	adds := make([]*syntax.Operation, 0, 2)
	for {
		add, ok := x.(*syntax.Operation)
		if !ok || add.Op != syntax.Add || add.Y == nil {
			break
		}
		adds = append(adds, add)
		x = add.X
	}

	// nstr is the current rightmost string literal in the
	// summation (if any), and chunks holds its accumulated
	// substrings.
	//
	// Consider the expression x + "a" + "b" + "c" + y. When we
	// reach the string literal "a", we assign nstr to point to
	// its corresponding Node and initialize chunks to {"a"}.
	// Visiting the subsequent string literals "b" and "c", we
	// simply append their values to chunks. Finally, when we
	// reach the non-constant operand y, we'll join chunks to form
	// "abc" and reassign the "a" string literal's value.
	//
	// N.B., we need to be careful about named string constants
	// (indicated by Sym != nil) because 1) we can't modify their
	// value, as doing so would affect other uses of the string
	// constant, and 2) they may have types, which we need to
	// handle correctly. For now, we avoid these problems by
	// treating named string constants the same as non-constant
	// operands.
	var nstr ir.Node
	chunks := make([]string, 0, 1)

	n := p.expr(x)
	if ir.IsConst(n, constant.String) && n.Sym() == nil {
		nstr = n
		chunks = append(chunks, ir.StringVal(nstr))
	}

	for i := len(adds) - 1; i >= 0; i-- {
		add := adds[i]

		r := p.expr(add.Y)
		if ir.IsConst(r, constant.String) && r.Sym() == nil {
			if nstr != nil {
				// Collapse r into nstr instead of adding to n.
				chunks = append(chunks, ir.StringVal(r))
				continue
			}

			nstr = r
			chunks = append(chunks, ir.StringVal(nstr))
		} else {
			if len(chunks) > 1 {
				nstr.SetVal(constant.MakeString(strings.Join(chunks, "")))
			}
			nstr = nil
			chunks = chunks[:0]
		}
		n = p.nod(add, ir.OADD, n, r)
	}
	if len(chunks) > 1 {
		nstr.SetVal(constant.MakeString(strings.Join(chunks, "")))
	}

	return n
}

func (p *noder) typeExpr(typ syntax.Expr) ir.Ntype {
	// TODO(mdempsky): Be stricter? typecheck should handle errors anyway.
	n := p.expr(typ)
	if n == nil {
		return nil
	}
	if _, ok := n.(ir.Ntype); !ok {
		ir.Dump("NOT NTYPE", n)
	}
	return n.(ir.Ntype)
}

func (p *noder) typeExprOrNil(typ syntax.Expr) ir.Ntype {
	if typ != nil {
		return p.typeExpr(typ)
	}
	return nil
}

func (p *noder) chanDir(dir syntax.ChanDir) types.ChanDir {
	switch dir {
	case 0:
		return types.Cboth
	case syntax.SendOnly:
		return types.Csend
	case syntax.RecvOnly:
		return types.Crecv
	}
	panic("unhandled ChanDir")
}

func (p *noder) structType(expr *syntax.StructType) ir.Node {
	l := make([]*ir.Field, 0, len(expr.FieldList))
	for i, field := range expr.FieldList {
		p.setlineno(field)
		var n *ir.Field
		if field.Name == nil {
			n = p.embedded(field.Type)
		} else {
			n = ir.NewField(p.pos(field), p.name(field.Name), p.typeExpr(field.Type), nil)
		}
		if i < len(expr.TagList) && expr.TagList[i] != nil {
			n.Note = constant.StringVal(p.basicLit(expr.TagList[i]))
		}
		l = append(l, n)
	}

	p.setlineno(expr)
	return ir.NewStructType(p.pos(expr), l)
}

func (p *noder) interfaceType(expr *syntax.InterfaceType) ir.Node {
	l := make([]*ir.Field, 0, len(expr.MethodList))
	for _, method := range expr.MethodList {
		p.setlineno(method)
		var n *ir.Field
		if method.Name == nil {
			n = ir.NewField(p.pos(method), nil, importName(p.packname(method.Type)).(ir.Ntype), nil)
		} else {
			mname := p.name(method.Name)
			if mname.IsBlank() {
				base.Errorf("methods must have a unique non-blank name")
				continue
			}
			sig := p.typeExpr(method.Type).(*ir.FuncType)
			sig.Recv = fakeRecv()
			n = ir.NewField(p.pos(method), mname, sig, nil)
		}
		l = append(l, n)
	}

	return ir.NewInterfaceType(p.pos(expr), l)
}

func (p *noder) packname(expr syntax.Expr) *types.Sym {
	switch expr := expr.(type) {
	case *syntax.Name:
		name := p.name(expr)
		if n := oldname(name); n.Name() != nil && n.Name().PkgName != nil {
			n.Name().PkgName.Used = true
		}
		return name
	case *syntax.SelectorExpr:
		name := p.name(expr.X.(*syntax.Name))
		def := ir.AsNode(name.Def)
		if def == nil {
			base.Errorf("undefined: %v", name)
			return name
		}
		var pkg *types.Pkg
		if def.Op() != ir.OPACK {
			base.Errorf("%v is not a package", name)
			pkg = ir.LocalPkg
		} else {
			def := def.(*ir.PkgName)
			def.Used = true
			pkg = def.Pkg
		}
		return pkg.Lookup(expr.Sel.Value)
	}
	panic(fmt.Sprintf("unexpected packname: %#v", expr))
}

func (p *noder) embedded(typ syntax.Expr) *ir.Field {
	op, isStar := typ.(*syntax.Operation)
	if isStar {
		if op.Op != syntax.Mul || op.Y != nil {
			panic("unexpected Operation")
		}
		typ = op.X
	}

	sym := p.packname(typ)
	n := ir.NewField(p.pos(typ), lookup(sym.Name), importName(sym).(ir.Ntype), nil)
	n.Embedded = true

	if isStar {
		n.Ntype = ir.NewStarExpr(p.pos(op), n.Ntype)
	}
	return n
}

func (p *noder) stmts(stmts []syntax.Stmt) []ir.Node {
	return p.stmtsFall(stmts, false)
}

func (p *noder) stmtsFall(stmts []syntax.Stmt, fallOK bool) []ir.Node {
	var nodes []ir.Node
	for i, stmt := range stmts {
		s := p.stmtFall(stmt, fallOK && i+1 == len(stmts))
		if s == nil {
		} else if s.Op() == ir.OBLOCK && s.List().Len() > 0 {
			// Inline non-empty block.
			// Empty blocks must be preserved for checkreturn.
			nodes = append(nodes, s.List().Slice()...)
		} else {
			nodes = append(nodes, s)
		}
	}
	return nodes
}

func (p *noder) stmt(stmt syntax.Stmt) ir.Node {
	return p.stmtFall(stmt, false)
}

func (p *noder) stmtFall(stmt syntax.Stmt, fallOK bool) ir.Node {
	p.setlineno(stmt)
	switch stmt := stmt.(type) {
	case *syntax.EmptyStmt:
		return nil
	case *syntax.LabeledStmt:
		return p.labeledStmt(stmt, fallOK)
	case *syntax.BlockStmt:
		l := p.blockStmt(stmt)
		if len(l) == 0 {
			// TODO(mdempsky): Line number?
			return ir.Nod(ir.OBLOCK, nil, nil)
		}
		return liststmt(l)
	case *syntax.ExprStmt:
		return p.wrapname(stmt, p.expr(stmt.X))
	case *syntax.SendStmt:
		return p.nod(stmt, ir.OSEND, p.expr(stmt.Chan), p.expr(stmt.Value))
	case *syntax.DeclStmt:
		return liststmt(p.decls(stmt.DeclList))
	case *syntax.AssignStmt:
		if stmt.Op != 0 && stmt.Op != syntax.Def {
			n := p.nod(stmt, ir.OASOP, p.expr(stmt.Lhs), p.expr(stmt.Rhs))
			n.SetImplicit(stmt.Rhs == syntax.ImplicitOne)
			n.SetSubOp(p.binOp(stmt.Op))
			return n
		}

		rhs := p.exprList(stmt.Rhs)
		if list, ok := stmt.Lhs.(*syntax.ListExpr); ok && len(list.ElemList) != 1 || len(rhs) != 1 {
			n := p.nod(stmt, ir.OAS2, nil, nil)
			n.PtrList().Set(p.assignList(stmt.Lhs, n, stmt.Op == syntax.Def))
			n.PtrRlist().Set(rhs)
			return n
		}

		n := p.nod(stmt, ir.OAS, nil, nil)
		n.SetLeft(p.assignList(stmt.Lhs, n, stmt.Op == syntax.Def)[0])
		n.SetRight(rhs[0])
		return n

	case *syntax.BranchStmt:
		var op ir.Op
		switch stmt.Tok {
		case syntax.Break:
			op = ir.OBREAK
		case syntax.Continue:
			op = ir.OCONTINUE
		case syntax.Fallthrough:
			if !fallOK {
				base.Errorf("fallthrough statement out of place")
			}
			op = ir.OFALL
		case syntax.Goto:
			op = ir.OGOTO
		default:
			panic("unhandled BranchStmt")
		}
		n := p.nod(stmt, op, nil, nil)
		if stmt.Label != nil {
			n.SetSym(p.name(stmt.Label))
		}
		return n
	case *syntax.CallStmt:
		var op ir.Op
		switch stmt.Tok {
		case syntax.Defer:
			op = ir.ODEFER
		case syntax.Go:
			op = ir.OGO
		default:
			panic("unhandled CallStmt")
		}
		return p.nod(stmt, op, p.expr(stmt.Call), nil)
	case *syntax.ReturnStmt:
		var results []ir.Node
		if stmt.Results != nil {
			results = p.exprList(stmt.Results)
		}
		n := p.nod(stmt, ir.ORETURN, nil, nil)
		n.PtrList().Set(results)
		if n.List().Len() == 0 && Curfn != nil {
			for _, ln := range Curfn.Dcl {
				if ln.Class() == ir.PPARAM {
					continue
				}
				if ln.Class() != ir.PPARAMOUT {
					break
				}
				if ln.Sym().Def != ln {
					base.Errorf("%s is shadowed during return", ln.Sym().Name)
				}
			}
		}
		return n
	case *syntax.IfStmt:
		return p.ifStmt(stmt)
	case *syntax.ForStmt:
		return p.forStmt(stmt)
	case *syntax.SwitchStmt:
		return p.switchStmt(stmt)
	case *syntax.SelectStmt:
		return p.selectStmt(stmt)
	}
	panic("unhandled Stmt")
}

func (p *noder) assignList(expr syntax.Expr, defn ir.Node, colas bool) []ir.Node {
	if !colas {
		return p.exprList(expr)
	}

	defn.SetColas(true)

	var exprs []syntax.Expr
	if list, ok := expr.(*syntax.ListExpr); ok {
		exprs = list.ElemList
	} else {
		exprs = []syntax.Expr{expr}
	}

	res := make([]ir.Node, len(exprs))
	seen := make(map[*types.Sym]bool, len(exprs))

	newOrErr := false
	for i, expr := range exprs {
		p.setlineno(expr)
		res[i] = ir.BlankNode

		name, ok := expr.(*syntax.Name)
		if !ok {
			p.errorAt(expr.Pos(), "non-name %v on left side of :=", p.expr(expr))
			newOrErr = true
			continue
		}

		sym := p.name(name)
		if sym.IsBlank() {
			continue
		}

		if seen[sym] {
			p.errorAt(expr.Pos(), "%v repeated on left side of :=", sym)
			newOrErr = true
			continue
		}
		seen[sym] = true

		if sym.Block == types.Block {
			res[i] = oldname(sym)
			continue
		}

		newOrErr = true
		n := NewName(sym)
		declare(n, dclcontext)
		n.Defn = defn
		defn.PtrInit().Append(ir.Nod(ir.ODCL, n, nil))
		res[i] = n
	}

	if !newOrErr {
		base.ErrorfAt(defn.Pos(), "no new variables on left side of :=")
	}
	return res
}

func (p *noder) blockStmt(stmt *syntax.BlockStmt) []ir.Node {
	p.openScope(stmt.Pos())
	nodes := p.stmts(stmt.List)
	p.closeScope(stmt.Rbrace)
	return nodes
}

func (p *noder) ifStmt(stmt *syntax.IfStmt) ir.Node {
	p.openScope(stmt.Pos())
	n := p.nod(stmt, ir.OIF, nil, nil)
	if stmt.Init != nil {
		n.PtrInit().Set1(p.stmt(stmt.Init))
	}
	if stmt.Cond != nil {
		n.SetLeft(p.expr(stmt.Cond))
	}
	n.PtrBody().Set(p.blockStmt(stmt.Then))
	if stmt.Else != nil {
		e := p.stmt(stmt.Else)
		if e.Op() == ir.OBLOCK {
			n.PtrRlist().Set(e.List().Slice())
		} else {
			n.PtrRlist().Set1(e)
		}
	}
	p.closeAnotherScope()
	return n
}

func (p *noder) forStmt(stmt *syntax.ForStmt) ir.Node {
	p.openScope(stmt.Pos())
	var n ir.Node
	if r, ok := stmt.Init.(*syntax.RangeClause); ok {
		if stmt.Cond != nil || stmt.Post != nil {
			panic("unexpected RangeClause")
		}

		n = p.nod(r, ir.ORANGE, nil, p.expr(r.X))
		if r.Lhs != nil {
			n.PtrList().Set(p.assignList(r.Lhs, n, r.Def))
		}
	} else {
		n = p.nod(stmt, ir.OFOR, nil, nil)
		if stmt.Init != nil {
			n.PtrInit().Set1(p.stmt(stmt.Init))
		}
		if stmt.Cond != nil {
			n.SetLeft(p.expr(stmt.Cond))
		}
		if stmt.Post != nil {
			n.SetRight(p.stmt(stmt.Post))
		}
	}
	n.PtrBody().Set(p.blockStmt(stmt.Body))
	p.closeAnotherScope()
	return n
}

func (p *noder) switchStmt(stmt *syntax.SwitchStmt) ir.Node {
	p.openScope(stmt.Pos())
	n := p.nod(stmt, ir.OSWITCH, nil, nil)
	if stmt.Init != nil {
		n.PtrInit().Set1(p.stmt(stmt.Init))
	}
	if stmt.Tag != nil {
		n.SetLeft(p.expr(stmt.Tag))
	}

	tswitch := n.Left()
	if tswitch != nil && tswitch.Op() != ir.OTYPESW {
		tswitch = nil
	}
	n.PtrList().Set(p.caseClauses(stmt.Body, tswitch, stmt.Rbrace))

	p.closeScope(stmt.Rbrace)
	return n
}

func (p *noder) caseClauses(clauses []*syntax.CaseClause, tswitch ir.Node, rbrace syntax.Pos) []ir.Node {
	nodes := make([]ir.Node, 0, len(clauses))
	for i, clause := range clauses {
		p.setlineno(clause)
		if i > 0 {
			p.closeScope(clause.Pos())
		}
		p.openScope(clause.Pos())

		n := p.nod(clause, ir.OCASE, nil, nil)
		if clause.Cases != nil {
			n.PtrList().Set(p.exprList(clause.Cases))
		}
		if tswitch != nil && tswitch.Left() != nil {
			nn := NewName(tswitch.Left().Sym())
			declare(nn, dclcontext)
			n.PtrRlist().Set1(nn)
			// keep track of the instances for reporting unused
			nn.Defn = tswitch
		}

		// Trim trailing empty statements. We omit them from
		// the Node AST anyway, and it's easier to identify
		// out-of-place fallthrough statements without them.
		body := clause.Body
		for len(body) > 0 {
			if _, ok := body[len(body)-1].(*syntax.EmptyStmt); !ok {
				break
			}
			body = body[:len(body)-1]
		}

		n.PtrBody().Set(p.stmtsFall(body, true))
		if l := n.Body().Len(); l > 0 && n.Body().Index(l-1).Op() == ir.OFALL {
			if tswitch != nil {
				base.Errorf("cannot fallthrough in type switch")
			}
			if i+1 == len(clauses) {
				base.Errorf("cannot fallthrough final case in switch")
			}
		}

		nodes = append(nodes, n)
	}
	if len(clauses) > 0 {
		p.closeScope(rbrace)
	}
	return nodes
}

func (p *noder) selectStmt(stmt *syntax.SelectStmt) ir.Node {
	n := p.nod(stmt, ir.OSELECT, nil, nil)
	n.PtrList().Set(p.commClauses(stmt.Body, stmt.Rbrace))
	return n
}

func (p *noder) commClauses(clauses []*syntax.CommClause, rbrace syntax.Pos) []ir.Node {
	nodes := make([]ir.Node, 0, len(clauses))
	for i, clause := range clauses {
		p.setlineno(clause)
		if i > 0 {
			p.closeScope(clause.Pos())
		}
		p.openScope(clause.Pos())

		n := p.nod(clause, ir.OCASE, nil, nil)
		if clause.Comm != nil {
			n.PtrList().Set1(p.stmt(clause.Comm))
		}
		n.PtrBody().Set(p.stmts(clause.Body))
		nodes = append(nodes, n)
	}
	if len(clauses) > 0 {
		p.closeScope(rbrace)
	}
	return nodes
}

func (p *noder) labeledStmt(label *syntax.LabeledStmt, fallOK bool) ir.Node {
	sym := p.name(label.Label)
	lhs := p.nodSym(label, ir.OLABEL, nil, sym)

	var ls ir.Node
	if label.Stmt != nil { // TODO(mdempsky): Should always be present.
		ls = p.stmtFall(label.Stmt, fallOK)
		switch label.Stmt.(type) {
		case *syntax.ForStmt, *syntax.SwitchStmt, *syntax.SelectStmt:
			// Attach label directly to control statement too.
			ls.SetSym(sym)
		}
	}

	l := []ir.Node{lhs}
	if ls != nil {
		if ls.Op() == ir.OBLOCK {
			l = append(l, ls.List().Slice()...)
		} else {
			l = append(l, ls)
		}
	}
	return liststmt(l)
}

var unOps = [...]ir.Op{
	syntax.Recv: ir.ORECV,
	syntax.Mul:  ir.ODEREF,
	syntax.And:  ir.OADDR,

	syntax.Not: ir.ONOT,
	syntax.Xor: ir.OBITNOT,
	syntax.Add: ir.OPLUS,
	syntax.Sub: ir.ONEG,
}

func (p *noder) unOp(op syntax.Operator) ir.Op {
	if uint64(op) >= uint64(len(unOps)) || unOps[op] == 0 {
		panic("invalid Operator")
	}
	return unOps[op]
}

var binOps = [...]ir.Op{
	syntax.OrOr:   ir.OOROR,
	syntax.AndAnd: ir.OANDAND,

	syntax.Eql: ir.OEQ,
	syntax.Neq: ir.ONE,
	syntax.Lss: ir.OLT,
	syntax.Leq: ir.OLE,
	syntax.Gtr: ir.OGT,
	syntax.Geq: ir.OGE,

	syntax.Add: ir.OADD,
	syntax.Sub: ir.OSUB,
	syntax.Or:  ir.OOR,
	syntax.Xor: ir.OXOR,

	syntax.Mul:    ir.OMUL,
	syntax.Div:    ir.ODIV,
	syntax.Rem:    ir.OMOD,
	syntax.And:    ir.OAND,
	syntax.AndNot: ir.OANDNOT,
	syntax.Shl:    ir.OLSH,
	syntax.Shr:    ir.ORSH,
}

func (p *noder) binOp(op syntax.Operator) ir.Op {
	if uint64(op) >= uint64(len(binOps)) || binOps[op] == 0 {
		panic("invalid Operator")
	}
	return binOps[op]
}

// checkLangCompat reports an error if the representation of a numeric
// literal is not compatible with the current language version.
func checkLangCompat(lit *syntax.BasicLit) {
	s := lit.Value
	if len(s) <= 2 || langSupported(1, 13, ir.LocalPkg) {
		return
	}
	// len(s) > 2
	if strings.Contains(s, "_") {
		base.ErrorfVers("go1.13", "underscores in numeric literals")
		return
	}
	if s[0] != '0' {
		return
	}
	radix := s[1]
	if radix == 'b' || radix == 'B' {
		base.ErrorfVers("go1.13", "binary literals")
		return
	}
	if radix == 'o' || radix == 'O' {
		base.ErrorfVers("go1.13", "0o/0O-style octal literals")
		return
	}
	if lit.Kind != syntax.IntLit && (radix == 'x' || radix == 'X') {
		base.ErrorfVers("go1.13", "hexadecimal floating-point literals")
	}
}

func (p *noder) basicLit(lit *syntax.BasicLit) constant.Value {
	// We don't use the errors of the conversion routines to determine
	// if a literal string is valid because the conversion routines may
	// accept a wider syntax than the language permits. Rely on lit.Bad
	// instead.
	if lit.Bad {
		return constant.MakeUnknown()
	}

	switch lit.Kind {
	case syntax.IntLit, syntax.FloatLit, syntax.ImagLit:
		checkLangCompat(lit)
	}

	v := constant.MakeFromLiteral(lit.Value, tokenForLitKind[lit.Kind], 0)
	if v.Kind() == constant.Unknown {
		// TODO(mdempsky): Better error message?
		p.errorAt(lit.Pos(), "malformed constant: %s", lit.Value)
	}

	// go/constant uses big.Rat by default, which is more precise, but
	// causes toolstash -cmp and some tests to fail. For now, convert
	// to big.Float to match cmd/compile's historical precision.
	// TODO(mdempsky): Remove.
	if v.Kind() == constant.Float {
		v = constant.Make(bigFloatVal(v))
	}

	return v
}

var tokenForLitKind = [...]token.Token{
	syntax.IntLit:    token.INT,
	syntax.RuneLit:   token.CHAR,
	syntax.FloatLit:  token.FLOAT,
	syntax.ImagLit:   token.IMAG,
	syntax.StringLit: token.STRING,
}

func (p *noder) name(name *syntax.Name) *types.Sym {
	return lookup(name.Value)
}

func (p *noder) mkname(name *syntax.Name) ir.Node {
	// TODO(mdempsky): Set line number?
	return mkname(p.name(name))
}

func (p *noder) wrapname(n syntax.Node, x ir.Node) ir.Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give them the correct line.
	switch x.Op() {
	case ir.OTYPE, ir.OLITERAL:
		if x.Sym() == nil {
			break
		}
		fallthrough
	case ir.ONAME, ir.ONONAME, ir.OPACK:
		x = p.nod(n, ir.OPAREN, x, nil)
		x.SetImplicit(true)
	}
	return x
}

func (p *noder) nod(orig syntax.Node, op ir.Op, left, right ir.Node) ir.Node {
	return ir.NodAt(p.pos(orig), op, left, right)
}

func (p *noder) nodSym(orig syntax.Node, op ir.Op, left ir.Node, sym *types.Sym) ir.Node {
	n := nodSym(op, left, sym)
	n.SetPos(p.pos(orig))
	return n
}

func (p *noder) pos(n syntax.Node) src.XPos {
	// TODO(gri): orig.Pos() should always be known - fix package syntax
	xpos := base.Pos
	if pos := n.Pos(); pos.IsKnown() {
		xpos = p.makeXPos(pos)
	}
	return xpos
}

func (p *noder) setlineno(n syntax.Node) {
	if n != nil {
		base.Pos = p.pos(n)
	}
}

// error is called concurrently if files are parsed concurrently.
func (p *noder) error(err error) {
	p.err <- err.(syntax.Error)
}

// pragmas that are allowed in the std lib, but don't have
// a syntax.Pragma value (see lex.go) associated with them.
var allowedStdPragmas = map[string]bool{
	"go:cgo_export_static":  true,
	"go:cgo_export_dynamic": true,
	"go:cgo_import_static":  true,
	"go:cgo_import_dynamic": true,
	"go:cgo_ldflag":         true,
	"go:cgo_dynamic_linker": true,
	"go:embed":              true,
	"go:generate":           true,
}

// *Pragma is the value stored in a syntax.Pragma during parsing.
type Pragma struct {
	Flag   ir.PragmaFlag // collected bits
	Pos    []PragmaPos   // position of each individual flag
	Embeds []PragmaEmbed
}

type PragmaPos struct {
	Flag ir.PragmaFlag
	Pos  syntax.Pos
}

type PragmaEmbed struct {
	Pos      syntax.Pos
	Patterns []string
}

func (p *noder) checkUnused(pragma *Pragma) {
	for _, pos := range pragma.Pos {
		if pos.Flag&pragma.Flag != 0 {
			p.errorAt(pos.Pos, "misplaced compiler directive")
		}
	}
	if len(pragma.Embeds) > 0 {
		for _, e := range pragma.Embeds {
			p.errorAt(e.Pos, "misplaced go:embed directive")
		}
	}
}

func (p *noder) checkUnusedDuringParse(pragma *Pragma) {
	for _, pos := range pragma.Pos {
		if pos.Flag&pragma.Flag != 0 {
			p.error(syntax.Error{Pos: pos.Pos, Msg: "misplaced compiler directive"})
		}
	}
	if len(pragma.Embeds) > 0 {
		for _, e := range pragma.Embeds {
			p.error(syntax.Error{Pos: e.Pos, Msg: "misplaced go:embed directive"})
		}
	}
}

// pragma is called concurrently if files are parsed concurrently.
func (p *noder) pragma(pos syntax.Pos, blankLine bool, text string, old syntax.Pragma) syntax.Pragma {
	pragma, _ := old.(*Pragma)
	if pragma == nil {
		pragma = new(Pragma)
	}

	if text == "" {
		// unused pragma; only called with old != nil.
		p.checkUnusedDuringParse(pragma)
		return nil
	}

	if strings.HasPrefix(text, "line ") {
		// line directives are handled by syntax package
		panic("unreachable")
	}

	if !blankLine {
		// directive must be on line by itself
		p.error(syntax.Error{Pos: pos, Msg: "misplaced compiler directive"})
		return pragma
	}

	switch {
	case strings.HasPrefix(text, "go:linkname "):
		f := strings.Fields(text)
		if !(2 <= len(f) && len(f) <= 3) {
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:linkname localname [linkname]"})
			break
		}
		// The second argument is optional. If omitted, we use
		// the default object symbol name for this and
		// linkname only serves to mark this symbol as
		// something that may be referenced via the object
		// symbol name from another package.
		var target string
		if len(f) == 3 {
			target = f[2]
		} else if base.Ctxt.Pkgpath != "" {
			// Use the default object symbol name if the
			// user didn't provide one.
			target = objabi.PathToPrefix(base.Ctxt.Pkgpath) + "." + f[1]
		} else {
			p.error(syntax.Error{Pos: pos, Msg: "//go:linkname requires linkname argument or -p compiler flag"})
			break
		}
		p.linknames = append(p.linknames, linkname{pos, f[1], target})

	case text == "go:embed", strings.HasPrefix(text, "go:embed "):
		args, err := parseGoEmbed(text[len("go:embed"):])
		if err != nil {
			p.error(syntax.Error{Pos: pos, Msg: err.Error()})
		}
		if len(args) == 0 {
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:embed pattern..."})
			break
		}
		pragma.Embeds = append(pragma.Embeds, PragmaEmbed{pos, args})

	case strings.HasPrefix(text, "go:cgo_import_dynamic "):
		// This is permitted for general use because Solaris
		// code relies on it in golang.org/x/sys/unix and others.
		fields := pragmaFields(text)
		if len(fields) >= 4 {
			lib := strings.Trim(fields[3], `"`)
			if lib != "" && !safeArg(lib) && !isCgoGeneratedFile(pos) {
				p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("invalid library name %q in cgo_import_dynamic directive", lib)})
			}
			p.pragcgo(pos, text)
			pragma.Flag |= pragmaFlag("go:cgo_import_dynamic")
			break
		}
		fallthrough
	case strings.HasPrefix(text, "go:cgo_"):
		// For security, we disallow //go:cgo_* directives other
		// than cgo_import_dynamic outside cgo-generated files.
		// Exception: they are allowed in the standard library, for runtime and syscall.
		if !isCgoGeneratedFile(pos) && !base.Flag.Std {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s only allowed in cgo-generated code", text)})
		}
		p.pragcgo(pos, text)
		fallthrough // because of //go:cgo_unsafe_args
	default:
		verb := text
		if i := strings.Index(text, " "); i >= 0 {
			verb = verb[:i]
		}
		flag := pragmaFlag(verb)
		const runtimePragmas = ir.Systemstack | ir.Nowritebarrier | ir.Nowritebarrierrec | ir.Yeswritebarrierrec
		if !base.Flag.CompilingRuntime && flag&runtimePragmas != 0 {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s only allowed in runtime", verb)})
		}
		if flag == 0 && !allowedStdPragmas[verb] && base.Flag.Std {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s is not allowed in the standard library", verb)})
		}
		pragma.Flag |= flag
		pragma.Pos = append(pragma.Pos, PragmaPos{flag, pos})
	}

	return pragma
}

// isCgoGeneratedFile reports whether pos is in a file
// generated by cgo, which is to say a file with name
// beginning with "_cgo_". Such files are allowed to
// contain cgo directives, and for security reasons
// (primarily misuse of linker flags), other files are not.
// See golang.org/issue/23672.
func isCgoGeneratedFile(pos syntax.Pos) bool {
	return strings.HasPrefix(filepath.Base(filepath.Clean(fileh(pos.Base().Filename()))), "_cgo_")
}

// safeArg reports whether arg is a "safe" command-line argument,
// meaning that when it appears in a command-line, it probably
// doesn't have some special meaning other than its own name.
// This is copied from SafeArg in cmd/go/internal/load/pkg.go.
func safeArg(name string) bool {
	if name == "" {
		return false
	}
	c := name[0]
	return '0' <= c && c <= '9' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || c == '.' || c == '_' || c == '/' || c >= utf8.RuneSelf
}

func mkname(sym *types.Sym) ir.Node {
	n := oldname(sym)
	if n.Name() != nil && n.Name().PkgName != nil {
		n.Name().PkgName.Used = true
	}
	return n
}

// parseGoEmbed parses the text following "//go:embed" to extract the glob patterns.
// It accepts unquoted space-separated patterns as well as double-quoted and back-quoted Go strings.
// go/build/read.go also processes these strings and contains similar logic.
func parseGoEmbed(args string) ([]string, error) {
	var list []string
	for args = strings.TrimSpace(args); args != ""; args = strings.TrimSpace(args) {
		var path string
	Switch:
		switch args[0] {
		default:
			i := len(args)
			for j, c := range args {
				if unicode.IsSpace(c) {
					i = j
					break
				}
			}
			path = args[:i]
			args = args[i:]

		case '`':
			i := strings.Index(args[1:], "`")
			if i < 0 {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
			path = args[1 : 1+i]
			args = args[1+i+1:]

		case '"':
			i := 1
			for ; i < len(args); i++ {
				if args[i] == '\\' {
					i++
					continue
				}
				if args[i] == '"' {
					q, err := strconv.Unquote(args[:i+1])
					if err != nil {
						return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args[:i+1])
					}
					path = q
					args = args[i+1:]
					break Switch
				}
			}
			if i >= len(args) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}

		if args != "" {
			r, _ := utf8.DecodeRuneInString(args)
			if !unicode.IsSpace(r) {
				return nil, fmt.Errorf("invalid quoted string in //go:embed: %s", args)
			}
		}
		list = append(list, path)
	}
	return list, nil
}
