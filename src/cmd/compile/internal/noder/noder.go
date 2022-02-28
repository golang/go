// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"errors"
	"fmt"
	"go/constant"
	"go/token"
	"internal/buildcfg"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/dwarfgen"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

func LoadPackage(filenames []string) {
	base.Timer.Start("fe", "parse")

	// -G=3 and unified expect generics syntax, but -G=0 does not.
	supportsGenerics := base.Flag.G != 0 || buildcfg.Experiment.Unified

	mode := syntax.CheckBranches
	if supportsGenerics {
		mode |= syntax.AllowGenerics
	}

	// Limit the number of simultaneously open files.
	sem := make(chan struct{}, runtime.GOMAXPROCS(0)+10)

	noders := make([]*noder, len(filenames))
	for i, filename := range filenames {
		p := noder{
			err:         make(chan syntax.Error),
			trackScopes: base.Flag.Dwarf,
		}
		noders[i] = &p

		filename := filename
		go func() {
			sem <- struct{}{}
			defer func() { <-sem }()
			defer close(p.err)
			fbase := syntax.NewFileBase(filename)

			f, err := os.Open(filename)
			if err != nil {
				p.error(syntax.Error{Msg: err.Error()})
				return
			}
			defer f.Close()

			p.file, _ = syntax.Parse(fbase, f, p.error, p.pragma, mode) // errors are tracked via p.error
		}()
	}

	var lines uint
	for _, p := range noders {
		for e := range p.err {
			p.errorAt(e.Pos, "%s", e.Msg)
		}
		if p.file == nil {
			base.ErrorExit()
		}
		lines += p.file.EOF.Line()
	}
	base.Timer.AddEvent(int64(lines), "lines")

	if base.Debug.Unified != 0 {
		unified(noders)
		return
	}

	if base.Flag.G != 0 {
		// Use types2 to type-check and possibly generate IR.
		check2(noders)
		return
	}

	for _, p := range noders {
		p.node()
		p.file = nil // release memory
	}

	if base.SyntaxErrors() != 0 {
		base.ErrorExit()
	}
	types.CheckDclstack()

	for _, p := range noders {
		p.processPragmas()
	}

	// Typecheck.
	types.LocalPkg.Height = myheight
	typecheck.DeclareUniverse()
	typecheck.TypecheckAllowed = true

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	//
	//   We also defer type alias declarations until phase 2
	//   to avoid cycles like #18640.
	//   TODO(gri) Remove this again once we have a fix for #25838.
	//
	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.

	// Don't use range--typecheck can add closures to Target.Decls.
	for phase, name := range []string{"top1", "top2"} {
		base.Timer.Start("fe", "typecheck", name)
		for i := 0; i < len(typecheck.Target.Decls); i++ {
			n := typecheck.Target.Decls[i]
			op := n.Op()

			// Closure function declarations are typechecked as part of the
			// closure expression.
			if fn, ok := n.(*ir.Func); ok && fn.OClosure != nil {
				continue
			}

			// We don't actually add ir.ODCL nodes to Target.Decls. Make sure of that.
			if op == ir.ODCL {
				base.FatalfAt(n.Pos(), "unexpected top declaration: %v", op)
			}

			// Identify declarations that should be deferred to the second
			// iteration.
			late := op == ir.OAS || op == ir.OAS2 || op == ir.ODCLTYPE && n.(*ir.Decl).X.Alias()

			if late == (phase == 1) {
				typecheck.Target.Decls[i] = typecheck.Stmt(n)
			}
		}
	}

	// Phase 3: Type check function bodies.
	// Don't use range--typecheck can add closures to Target.Decls.
	base.Timer.Start("fe", "typecheck", "func")
	for i := 0; i < len(typecheck.Target.Decls); i++ {
		if fn, ok := typecheck.Target.Decls[i].(*ir.Func); ok {
			if base.Flag.W > 1 {
				s := fmt.Sprintf("\nbefore typecheck %v", fn)
				ir.Dump(s, fn)
			}
			typecheck.FuncBody(fn)
			if base.Flag.W > 1 {
				s := fmt.Sprintf("\nafter typecheck %v", fn)
				ir.Dump(s, fn)
			}
		}
	}

	// Phase 4: Check external declarations.
	// TODO(mdempsky): This should be handled when type checking their
	// corresponding ODCL nodes.
	base.Timer.Start("fe", "typecheck", "externdcls")
	for i, n := range typecheck.Target.Externs {
		if n.Op() == ir.ONAME {
			typecheck.Target.Externs[i] = typecheck.Expr(typecheck.Target.Externs[i])
		}
	}

	// Phase 5: With all user code type-checked, it's now safe to verify map keys.
	// With all user code typechecked, it's now safe to verify unused dot imports.
	typecheck.CheckMapKeys()
	CheckDotImports()
	base.ExitIfErrors()
}

func (p *noder) errorAt(pos syntax.Pos, format string, args ...interface{}) {
	base.ErrorfAt(p.makeXPos(pos), format, args...)
}

// trimFilename returns the "trimmed" filename of b, which is the
// absolute filename after applying -trimpath processing. This
// filename form is suitable for use in object files and export data.
//
// If b's filename has already been trimmed (i.e., because it was read
// in from an imported package's export data), then the filename is
// returned unchanged.
func trimFilename(b *syntax.PosBase) string {
	filename := b.Filename()
	if !b.Trimmed() {
		dir := ""
		if b.IsFileBase() {
			dir = base.Ctxt.Pathname
		}
		filename = objabi.AbsFile(dir, filename, base.Flag.TrimPath)
	}
	return filename
}

// noder transforms package syntax's AST into a Node tree.
type noder struct {
	posMap

	file           *syntax.File
	linknames      []linkname
	pragcgobuf     [][]string
	err            chan syntax.Error
	importedUnsafe bool
	importedEmbed  bool
	trackScopes    bool

	funcState *funcState
}

// funcState tracks all per-function state to make handling nested
// functions easier.
type funcState struct {
	// scopeVars is a stack tracking the number of variables declared in
	// the current function at the moment each open scope was opened.
	scopeVars []int
	marker    dwarfgen.ScopeMarker

	lastCloseScopePos syntax.Pos
}

func (p *noder) funcBody(fn *ir.Func, block *syntax.BlockStmt) {
	outerFuncState := p.funcState
	p.funcState = new(funcState)
	typecheck.StartFuncBody(fn)

	if block != nil {
		body := p.stmts(block.List)
		if body == nil {
			body = []ir.Node{ir.NewBlockStmt(base.Pos, nil)}
		}
		fn.Body = body

		base.Pos = p.makeXPos(block.Rbrace)
		fn.Endlineno = base.Pos
	}

	typecheck.FinishFuncBody()
	p.funcState.marker.WriteTo(fn)
	p.funcState = outerFuncState
}

func (p *noder) openScope(pos syntax.Pos) {
	fs := p.funcState
	types.Markdcl()

	if p.trackScopes {
		fs.scopeVars = append(fs.scopeVars, len(ir.CurFunc.Dcl))
		fs.marker.Push(p.makeXPos(pos))
	}
}

func (p *noder) closeScope(pos syntax.Pos) {
	fs := p.funcState
	fs.lastCloseScopePos = pos
	types.Popdcl()

	if p.trackScopes {
		scopeVars := fs.scopeVars[len(fs.scopeVars)-1]
		fs.scopeVars = fs.scopeVars[:len(fs.scopeVars)-1]
		if scopeVars == len(ir.CurFunc.Dcl) {
			// no variables were declared in this scope, so we can retract it.
			fs.marker.Unpush()
		} else {
			fs.marker.Pop(p.makeXPos(pos))
		}
	}
}

// closeAnotherScope is like closeScope, but it reuses the same mark
// position as the last closeScope call. This is useful for "for" and
// "if" statements, as their implicit blocks always end at the same
// position as an explicit block.
func (p *noder) closeAnotherScope() {
	p.closeScope(p.funcState.lastCloseScopePos)
}

// linkname records a //go:linkname directive.
type linkname struct {
	pos    syntax.Pos
	local  string
	remote string
}

func (p *noder) node() {
	p.importedUnsafe = false
	p.importedEmbed = false

	p.setlineno(p.file.PkgName)
	mkpackage(p.file.PkgName.Value)

	if pragma, ok := p.file.Pragma.(*pragmas); ok {
		pragma.Flag &^= ir.GoBuildPragma
		p.checkUnused(pragma)
	}

	typecheck.Target.Decls = append(typecheck.Target.Decls, p.decls(p.file.DeclList)...)

	base.Pos = src.NoXPos
	clearImports()
}

func (p *noder) processPragmas() {
	for _, l := range p.linknames {
		if !p.importedUnsafe {
			p.errorAt(l.pos, "//go:linkname only allowed in Go files that import \"unsafe\"")
			continue
		}
		n := ir.AsNode(typecheck.Lookup(l.local).Def)
		if n == nil || n.Op() != ir.ONAME {
			p.errorAt(l.pos, "//go:linkname must refer to declared function or variable")
			continue
		}
		if n.Sym().Linkname != "" {
			p.errorAt(l.pos, "duplicate //go:linkname for %s", l.local)
			continue
		}
		n.Sym().Linkname = l.remote
	}
	typecheck.Target.CgoPragmas = append(typecheck.Target.CgoPragmas, p.pragcgobuf...)
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
	if imp.Path == nil || imp.Path.Bad {
		return // avoid follow-on errors if there was a syntax error
	}

	if pragma, ok := imp.Pragma.(*pragmas); ok {
		p.checkUnused(pragma)
	}

	ipkg := importfile(imp)
	if ipkg == nil {
		if base.Errors() == 0 {
			base.Fatalf("phase error in import")
		}
		return
	}

	if ipkg == types.UnsafePkg {
		p.importedUnsafe = true
	}
	if ipkg.Path == "embed" {
		p.importedEmbed = true
	}

	var my *types.Sym
	if imp.LocalPkgName != nil {
		my = p.name(imp.LocalPkgName)
	} else {
		my = typecheck.Lookup(ipkg.Name)
	}

	pack := ir.NewPkgName(p.pos(imp), my, ipkg)

	switch my.Name {
	case ".":
		importDot(pack)
		return
	case "init":
		base.ErrorfAt(pack.Pos(), "cannot import package as init - init must be a func")
		return
	case "_":
		return
	}
	if my.Def != nil {
		typecheck.Redeclared(pack.Pos(), my, "as imported package name")
	}
	my.Def = pack
	my.Lastlineno = pack.Pos()
	my.Block = 1 // at top level
}

func (p *noder) varDecl(decl *syntax.VarDecl) []ir.Node {
	names := p.declNames(ir.ONAME, decl.NameList)
	typ := p.typeExprOrNil(decl.Type)
	exprs := p.exprList(decl.Values)

	if pragma, ok := decl.Pragma.(*pragmas); ok {
		varEmbed(p.makeXPos, names[0], decl, pragma, p.importedEmbed)
		p.checkUnused(pragma)
	}

	var init []ir.Node
	p.setlineno(decl)

	if len(names) > 1 && len(exprs) == 1 {
		as2 := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, exprs)
		for _, v := range names {
			as2.Lhs.Append(v)
			typecheck.Declare(v, typecheck.DeclContext)
			v.Ntype = typ
			v.Defn = as2
			if ir.CurFunc != nil {
				init = append(init, ir.NewDecl(base.Pos, ir.ODCL, v))
			}
		}

		return append(init, as2)
	}

	for i, v := range names {
		var e ir.Node
		if i < len(exprs) {
			e = exprs[i]
		}

		typecheck.Declare(v, typecheck.DeclContext)
		v.Ntype = typ

		if ir.CurFunc != nil {
			init = append(init, ir.NewDecl(base.Pos, ir.ODCL, v))
		}
		as := ir.NewAssignStmt(base.Pos, v, e)
		init = append(init, as)
		if e != nil || ir.CurFunc == nil {
			v.Defn = as
		}
	}

	if len(exprs) != 0 && len(names) != len(exprs) {
		base.Errorf("assignment mismatch: %d variables but %d values", len(names), len(exprs))
	}

	return init
}

// constState tracks state between constant specifiers within a
// declaration group. This state is kept separate from noder so nested
// constant declarations are handled correctly (e.g., issue 15550).
type constState struct {
	group  *syntax.Group
	typ    ir.Ntype
	values syntax.Expr
	iota   int64
}

func (p *noder) constDecl(decl *syntax.ConstDecl, cs *constState) []ir.Node {
	if decl.Group == nil || decl.Group != cs.group {
		*cs = constState{
			group: decl.Group,
		}
	}

	if pragma, ok := decl.Pragma.(*pragmas); ok {
		p.checkUnused(pragma)
	}

	names := p.declNames(ir.OLITERAL, decl.NameList)
	typ := p.typeExprOrNil(decl.Type)

	if decl.Values != nil {
		cs.typ, cs.values = typ, decl.Values
	} else {
		if typ != nil {
			base.Errorf("const declaration cannot have type without expression")
		}
		typ = cs.typ
	}
	values := p.exprList(cs.values)

	nn := make([]ir.Node, 0, len(names))
	for i, n := range names {
		if i >= len(values) {
			base.Errorf("missing value in const declaration")
			break
		}

		v := values[i]
		if decl.Values == nil {
			ir.Visit(v, func(v ir.Node) {
				if ir.HasUniquePos(v) {
					v.SetPos(n.Pos())
				}
			})
		}

		typecheck.Declare(n, typecheck.DeclContext)

		n.Ntype = typ
		n.Defn = v
		n.SetIota(cs.iota)

		nn = append(nn, ir.NewDecl(p.pos(decl), ir.ODCLCONST, n))
	}

	if len(values) > len(names) {
		base.Errorf("extra expression in const declaration")
	}

	cs.iota++

	return nn
}

func (p *noder) typeDecl(decl *syntax.TypeDecl) ir.Node {
	n := p.declName(ir.OTYPE, decl.Name)
	typecheck.Declare(n, typecheck.DeclContext)

	// decl.Type may be nil but in that case we got a syntax error during parsing
	typ := p.typeExprOrNil(decl.Type)

	n.Ntype = typ
	n.SetAlias(decl.Alias)
	if pragma, ok := decl.Pragma.(*pragmas); ok {
		if !decl.Alias {
			n.SetPragma(pragma.Flag & typePragmas)
			pragma.Flag &^= typePragmas
		}
		p.checkUnused(pragma)
	}

	nod := ir.NewDecl(p.pos(decl), ir.ODCLTYPE, n)
	if n.Alias() && !types.AllowsGoVersion(types.LocalPkg, 1, 9) {
		base.ErrorfAt(nod.Pos(), "type aliases only supported as of -lang=go1.9")
	}
	return nod
}

func (p *noder) declNames(op ir.Op, names []*syntax.Name) []*ir.Name {
	nodes := make([]*ir.Name, 0, len(names))
	for _, name := range names {
		nodes = append(nodes, p.declName(op, name))
	}
	return nodes
}

func (p *noder) declName(op ir.Op, name *syntax.Name) *ir.Name {
	return ir.NewDeclNameAt(p.pos(name), op, p.name(name))
}

func (p *noder) funcDecl(fun *syntax.FuncDecl) ir.Node {
	name := p.name(fun.Name)
	t := p.signature(fun.Recv, fun.Type)
	f := ir.NewFunc(p.pos(fun))

	if fun.Recv == nil {
		if name.Name == "init" {
			name = renameinit()
			if len(t.Params) > 0 || len(t.Results) > 0 {
				base.ErrorfAt(f.Pos(), "func init must have no arguments and no return values")
			}
			typecheck.Target.Inits = append(typecheck.Target.Inits, f)
		}

		if types.LocalPkg.Name == "main" && name.Name == "main" {
			if len(t.Params) > 0 || len(t.Results) > 0 {
				base.ErrorfAt(f.Pos(), "func main must have no arguments and no return values")
			}
		}
	} else {
		f.Shortname = name
		name = ir.BlankNode.Sym() // filled in by tcFunc
	}

	f.Nname = ir.NewNameAt(p.pos(fun.Name), name)
	f.Nname.Func = f
	f.Nname.Defn = f
	f.Nname.Ntype = t

	if pragma, ok := fun.Pragma.(*pragmas); ok {
		f.Pragma = pragma.Flag & funcPragmas
		if pragma.Flag&ir.Systemstack != 0 && pragma.Flag&ir.Nosplit != 0 {
			base.ErrorfAt(f.Pos(), "go:nosplit and go:systemstack cannot be combined")
		}
		pragma.Flag &^= funcPragmas
		p.checkUnused(pragma)
	}

	if fun.Recv == nil {
		typecheck.Declare(f.Nname, ir.PFUNC)
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
		if i > 0 && params[i].Type == params[i-1].Type {
			nodes[i].Ntype = nodes[i-1].Ntype
		}
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
	switch expr := expr.(type) {
	case nil:
		return nil
	case *syntax.ListExpr:
		return p.exprs(expr.ElemList)
	default:
		return []ir.Node{p.expr(expr)}
	}
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
		n := ir.NewBasicLit(p.pos(expr), p.basicLit(expr))
		if expr.Kind == syntax.RuneLit {
			n.SetType(types.UntypedRune)
		}
		n.SetDiag(expr.Bad || n.Val().Kind() == constant.Unknown) // avoid follow-on errors if there was a syntax error
		return n
	case *syntax.CompositeLit:
		n := ir.NewCompLitExpr(p.pos(expr), ir.OCOMPLIT, p.typeExpr(expr.Type), nil)
		l := p.exprs(expr.ElemList)
		for i, e := range l {
			l[i] = p.wrapname(expr.ElemList[i], e)
		}
		n.List = l
		base.Pos = p.makeXPos(expr.Rbrace)
		return n
	case *syntax.KeyValueExpr:
		// use position of expr.Key rather than of expr (which has position of ':')
		return ir.NewKeyExpr(p.pos(expr.Key), p.expr(expr.Key), p.wrapname(expr.Value, p.expr(expr.Value)))
	case *syntax.FuncLit:
		return p.funcLit(expr)
	case *syntax.ParenExpr:
		return ir.NewParenExpr(p.pos(expr), p.expr(expr.X))
	case *syntax.SelectorExpr:
		// parser.new_dotname
		obj := p.expr(expr.X)
		if obj.Op() == ir.OPACK {
			pack := obj.(*ir.PkgName)
			pack.Used = true
			return importName(pack.Pkg.Lookup(expr.Sel.Value))
		}
		n := ir.NewSelectorExpr(base.Pos, ir.OXDOT, obj, p.name(expr.Sel))
		n.SetPos(p.pos(expr)) // lineno may have been changed by p.expr(expr.X)
		return n
	case *syntax.IndexExpr:
		return ir.NewIndexExpr(p.pos(expr), p.expr(expr.X), p.expr(expr.Index))
	case *syntax.SliceExpr:
		op := ir.OSLICE
		if expr.Full {
			op = ir.OSLICE3
		}
		x := p.expr(expr.X)
		var index [3]ir.Node
		for i, n := range &expr.Index {
			if n != nil {
				index[i] = p.expr(n)
			}
		}
		return ir.NewSliceExpr(p.pos(expr), op, x, index[0], index[1], index[2])
	case *syntax.AssertExpr:
		return ir.NewTypeAssertExpr(p.pos(expr), p.expr(expr.X), p.typeExpr(expr.Type))
	case *syntax.Operation:
		if expr.Op == syntax.Add && expr.Y != nil {
			return p.sum(expr)
		}
		x := p.expr(expr.X)
		if expr.Y == nil {
			pos, op := p.pos(expr), p.unOp(expr.Op)
			switch op {
			case ir.OADDR:
				return typecheck.NodAddrAt(pos, x)
			case ir.ODEREF:
				return ir.NewStarExpr(pos, x)
			}
			return ir.NewUnaryExpr(pos, op, x)
		}

		pos, op, y := p.pos(expr), p.binOp(expr.Op), p.expr(expr.Y)
		switch op {
		case ir.OANDAND, ir.OOROR:
			return ir.NewLogicalExpr(pos, op, x, y)
		}
		return ir.NewBinaryExpr(pos, op, x, y)
	case *syntax.CallExpr:
		n := ir.NewCallExpr(p.pos(expr), ir.OCALL, p.expr(expr.Fun), p.exprs(expr.ArgList))
		n.IsDDD = expr.HasDots
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
		var tag *ir.Ident
		if expr.Lhs != nil {
			tag = ir.NewIdent(p.pos(expr.Lhs), p.name(expr.Lhs))
			if ir.IsBlank(tag) {
				base.Errorf("invalid variable name %v in type switch", tag)
			}
		}
		return ir.NewTypeSwitchGuard(p.pos(expr), tag, p.expr(expr.X))
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
		n = ir.NewBinaryExpr(p.pos(add), ir.OADD, n, r)
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
		if i > 0 && expr.FieldList[i].Type == expr.FieldList[i-1].Type {
			n.Ntype = l[i-1].Ntype
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
			pkg = types.LocalPkg
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
	pos := p.pos(syntax.StartPos(typ))

	op, isStar := typ.(*syntax.Operation)
	if isStar {
		if op.Op != syntax.Mul || op.Y != nil {
			panic("unexpected Operation")
		}
		typ = op.X
	}

	sym := p.packname(typ)
	n := ir.NewField(pos, typecheck.Lookup(sym.Name), importName(sym).(ir.Ntype), nil)
	n.Embedded = true

	if isStar {
		n.Ntype = ir.NewStarExpr(pos, n.Ntype)
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
		} else if s.Op() == ir.OBLOCK && len(s.(*ir.BlockStmt).List) > 0 {
			// Inline non-empty block.
			// Empty blocks must be preserved for CheckReturn.
			nodes = append(nodes, s.(*ir.BlockStmt).List...)
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
	case nil, *syntax.EmptyStmt:
		return nil
	case *syntax.LabeledStmt:
		return p.labeledStmt(stmt, fallOK)
	case *syntax.BlockStmt:
		l := p.blockStmt(stmt)
		if len(l) == 0 {
			// TODO(mdempsky): Line number?
			return ir.NewBlockStmt(base.Pos, nil)
		}
		return ir.NewBlockStmt(src.NoXPos, l)
	case *syntax.ExprStmt:
		return p.wrapname(stmt, p.expr(stmt.X))
	case *syntax.SendStmt:
		return ir.NewSendStmt(p.pos(stmt), p.expr(stmt.Chan), p.expr(stmt.Value))
	case *syntax.DeclStmt:
		return ir.NewBlockStmt(src.NoXPos, p.decls(stmt.DeclList))
	case *syntax.AssignStmt:
		if stmt.Rhs == nil {
			pos := p.pos(stmt)
			n := ir.NewAssignOpStmt(pos, p.binOp(stmt.Op), p.expr(stmt.Lhs), ir.NewBasicLit(pos, one))
			n.IncDec = true
			return n
		}

		if stmt.Op != 0 && stmt.Op != syntax.Def {
			n := ir.NewAssignOpStmt(p.pos(stmt), p.binOp(stmt.Op), p.expr(stmt.Lhs), p.expr(stmt.Rhs))
			return n
		}

		rhs := p.exprList(stmt.Rhs)
		if list, ok := stmt.Lhs.(*syntax.ListExpr); ok && len(list.ElemList) != 1 || len(rhs) != 1 {
			n := ir.NewAssignListStmt(p.pos(stmt), ir.OAS2, nil, nil)
			n.Def = stmt.Op == syntax.Def
			n.Lhs = p.assignList(stmt.Lhs, n, n.Def)
			n.Rhs = rhs
			return n
		}

		n := ir.NewAssignStmt(p.pos(stmt), nil, nil)
		n.Def = stmt.Op == syntax.Def
		n.X = p.assignList(stmt.Lhs, n, n.Def)[0]
		n.Y = rhs[0]
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
		var sym *types.Sym
		if stmt.Label != nil {
			sym = p.name(stmt.Label)
		}
		return ir.NewBranchStmt(p.pos(stmt), op, sym)
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
		return ir.NewGoDeferStmt(p.pos(stmt), op, p.expr(stmt.Call))
	case *syntax.ReturnStmt:
		n := ir.NewReturnStmt(p.pos(stmt), p.exprList(stmt.Results))
		if len(n.Results) == 0 && ir.CurFunc != nil {
			for _, ln := range ir.CurFunc.Dcl {
				if ln.Class == ir.PPARAM {
					continue
				}
				if ln.Class != ir.PPARAMOUT {
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

func (p *noder) assignList(expr syntax.Expr, defn ir.InitNode, colas bool) []ir.Node {
	if !colas {
		return p.exprList(expr)
	}

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
		n := typecheck.NewName(sym)
		typecheck.Declare(n, typecheck.DeclContext)
		n.Defn = defn
		defn.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, n))
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
	init := p.stmt(stmt.Init)
	n := ir.NewIfStmt(p.pos(stmt), p.expr(stmt.Cond), p.blockStmt(stmt.Then), nil)
	if init != nil {
		n.SetInit([]ir.Node{init})
	}
	if stmt.Else != nil {
		e := p.stmt(stmt.Else)
		if e.Op() == ir.OBLOCK {
			e := e.(*ir.BlockStmt)
			n.Else = e.List
		} else {
			n.Else = []ir.Node{e}
		}
	}
	p.closeAnotherScope()
	return n
}

func (p *noder) forStmt(stmt *syntax.ForStmt) ir.Node {
	p.openScope(stmt.Pos())
	if r, ok := stmt.Init.(*syntax.RangeClause); ok {
		if stmt.Cond != nil || stmt.Post != nil {
			panic("unexpected RangeClause")
		}

		n := ir.NewRangeStmt(p.pos(r), nil, nil, p.expr(r.X), nil)
		if r.Lhs != nil {
			n.Def = r.Def
			lhs := p.assignList(r.Lhs, n, n.Def)
			n.Key = lhs[0]
			if len(lhs) > 1 {
				n.Value = lhs[1]
			}
		}
		n.Body = p.blockStmt(stmt.Body)
		p.closeAnotherScope()
		return n
	}

	n := ir.NewForStmt(p.pos(stmt), p.stmt(stmt.Init), p.expr(stmt.Cond), p.stmt(stmt.Post), p.blockStmt(stmt.Body))
	p.closeAnotherScope()
	return n
}

func (p *noder) switchStmt(stmt *syntax.SwitchStmt) ir.Node {
	p.openScope(stmt.Pos())

	init := p.stmt(stmt.Init)
	n := ir.NewSwitchStmt(p.pos(stmt), p.expr(stmt.Tag), nil)
	if init != nil {
		n.SetInit([]ir.Node{init})
	}

	var tswitch *ir.TypeSwitchGuard
	if l := n.Tag; l != nil && l.Op() == ir.OTYPESW {
		tswitch = l.(*ir.TypeSwitchGuard)
	}
	n.Cases = p.caseClauses(stmt.Body, tswitch, stmt.Rbrace)

	p.closeScope(stmt.Rbrace)
	return n
}

func (p *noder) caseClauses(clauses []*syntax.CaseClause, tswitch *ir.TypeSwitchGuard, rbrace syntax.Pos) []*ir.CaseClause {
	nodes := make([]*ir.CaseClause, 0, len(clauses))
	for i, clause := range clauses {
		p.setlineno(clause)
		if i > 0 {
			p.closeScope(clause.Pos())
		}
		p.openScope(clause.Pos())

		n := ir.NewCaseStmt(p.pos(clause), p.exprList(clause.Cases), nil)
		if tswitch != nil && tswitch.Tag != nil {
			nn := typecheck.NewName(tswitch.Tag.Sym())
			typecheck.Declare(nn, typecheck.DeclContext)
			n.Var = nn
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

		n.Body = p.stmtsFall(body, true)
		if l := len(n.Body); l > 0 && n.Body[l-1].Op() == ir.OFALL {
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
	return ir.NewSelectStmt(p.pos(stmt), p.commClauses(stmt.Body, stmt.Rbrace))
}

func (p *noder) commClauses(clauses []*syntax.CommClause, rbrace syntax.Pos) []*ir.CommClause {
	nodes := make([]*ir.CommClause, len(clauses))
	for i, clause := range clauses {
		p.setlineno(clause)
		if i > 0 {
			p.closeScope(clause.Pos())
		}
		p.openScope(clause.Pos())

		nodes[i] = ir.NewCommStmt(p.pos(clause), p.stmt(clause.Comm), p.stmts(clause.Body))
	}
	if len(clauses) > 0 {
		p.closeScope(rbrace)
	}
	return nodes
}

func (p *noder) labeledStmt(label *syntax.LabeledStmt, fallOK bool) ir.Node {
	sym := p.name(label.Label)
	lhs := ir.NewLabelStmt(p.pos(label), sym)

	var ls ir.Node
	if label.Stmt != nil { // TODO(mdempsky): Should always be present.
		ls = p.stmtFall(label.Stmt, fallOK)
		// Attach label directly to control statement too.
		if ls != nil {
			switch ls.Op() {
			case ir.OFOR:
				ls := ls.(*ir.ForStmt)
				ls.Label = sym
			case ir.ORANGE:
				ls := ls.(*ir.RangeStmt)
				ls.Label = sym
			case ir.OSWITCH:
				ls := ls.(*ir.SwitchStmt)
				ls.Label = sym
			case ir.OSELECT:
				ls := ls.(*ir.SelectStmt)
				ls.Label = sym
			}
		}
	}

	l := []ir.Node{lhs}
	if ls != nil {
		if ls.Op() == ir.OBLOCK {
			ls := ls.(*ir.BlockStmt)
			l = append(l, ls.List...)
		} else {
			l = append(l, ls)
		}
	}
	return ir.NewBlockStmt(src.NoXPos, l)
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
	if len(s) <= 2 || types.AllowsGoVersion(types.LocalPkg, 1, 13) {
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
		// The max. mantissa precision for untyped numeric values
		// is 512 bits, or 4048 bits for each of the two integer
		// parts of a fraction for floating-point numbers that are
		// represented accurately in the go/constant package.
		// Constant literals that are longer than this many bits
		// are not meaningful; and excessively long constants may
		// consume a lot of space and time for a useless conversion.
		// Cap constant length with a generous upper limit that also
		// allows for separators between all digits.
		const limit = 10000
		if len(lit.Value) > limit {
			p.errorAt(lit.Pos(), "excessively long constant: %s... (%d chars)", lit.Value[:10], len(lit.Value))
			return constant.MakeUnknown()
		}
	}

	v := constant.MakeFromLiteral(lit.Value, tokenForLitKind[lit.Kind], 0)
	if v.Kind() == constant.Unknown {
		// TODO(mdempsky): Better error message?
		p.errorAt(lit.Pos(), "malformed constant: %s", lit.Value)
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
	return typecheck.Lookup(name.Value)
}

func (p *noder) mkname(name *syntax.Name) ir.Node {
	// TODO(mdempsky): Set line number?
	return mkname(p.name(name))
}

func wrapname(pos src.XPos, x ir.Node) ir.Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give them the correct line.
	switch x.Op() {
	case ir.OTYPE, ir.OLITERAL:
		if x.Sym() == nil {
			break
		}
		fallthrough
	case ir.ONAME, ir.ONONAME, ir.OPACK:
		p := ir.NewParenExpr(pos, x)
		p.SetImplicit(true)
		return p
	}
	return x
}

func (p *noder) wrapname(n syntax.Node, x ir.Node) ir.Node {
	return wrapname(p.pos(n), x)
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

// *pragmas is the value stored in a syntax.pragmas during parsing.
type pragmas struct {
	Flag   ir.PragmaFlag // collected bits
	Pos    []pragmaPos   // position of each individual flag
	Embeds []pragmaEmbed
}

type pragmaPos struct {
	Flag ir.PragmaFlag
	Pos  syntax.Pos
}

type pragmaEmbed struct {
	Pos      syntax.Pos
	Patterns []string
}

func (p *noder) checkUnused(pragma *pragmas) {
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

func (p *noder) checkUnusedDuringParse(pragma *pragmas) {
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
	pragma, _ := old.(*pragmas)
	if pragma == nil {
		pragma = new(pragmas)
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
		pragma.Embeds = append(pragma.Embeds, pragmaEmbed{pos, args})

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
		pragma.Pos = append(pragma.Pos, pragmaPos{flag, pos})
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
	return strings.HasPrefix(filepath.Base(trimFilename(pos.Base())), "_cgo_")
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

func fakeRecv() *ir.Field {
	return ir.NewField(base.Pos, nil, nil, types.FakeRecvType())
}

func (p *noder) funcLit(expr *syntax.FuncLit) ir.Node {
	fn := ir.NewClosureFunc(p.pos(expr), ir.CurFunc != nil)
	fn.Nname.Ntype = p.typeExpr(expr.Type)

	p.funcBody(fn, expr.Body)

	ir.FinishCaptureNames(base.Pos, ir.CurFunc, fn)

	return fn.OClosure
}

// A function named init is a special case.
// It is called by the initialization before main is run.
// To make it unique within a package and also uncallable,
// the name, normally "pkg.init", is altered to "pkg.init.0".
var renameinitgen int

func renameinit() *types.Sym {
	s := typecheck.LookupNum("init.", renameinitgen)
	renameinitgen++
	return s
}

// oldname returns the Node that declares symbol s in the current scope.
// If no such Node currently exists, an ONONAME Node is returned instead.
// Automatically creates a new closure variable if the referenced symbol was
// declared in a different (containing) function.
func oldname(s *types.Sym) ir.Node {
	if s.Pkg != types.LocalPkg {
		return ir.NewIdent(base.Pos, s)
	}

	n := ir.AsNode(s.Def)
	if n == nil {
		// Maybe a top-level declaration will come along later to
		// define s. resolve will check s.Def again once all input
		// source has been processed.
		return ir.NewIdent(base.Pos, s)
	}

	if n, ok := n.(*ir.Name); ok {
		// TODO(rsc): If there is an outer variable x and we
		// are parsing x := 5 inside the closure, until we get to
		// the := it looks like a reference to the outer x so we'll
		// make x a closure variable unnecessarily.
		return ir.CaptureName(base.Pos, ir.CurFunc, n)
	}

	return n
}

func varEmbed(makeXPos func(syntax.Pos) src.XPos, name *ir.Name, decl *syntax.VarDecl, pragma *pragmas, haveEmbed bool) {
	pragmaEmbeds := pragma.Embeds
	pragma.Embeds = nil
	if len(pragmaEmbeds) == 0 {
		return
	}

	if err := checkEmbed(decl, haveEmbed, typecheck.DeclContext != ir.PEXTERN); err != nil {
		base.ErrorfAt(makeXPos(pragmaEmbeds[0].Pos), "%s", err)
		return
	}

	var embeds []ir.Embed
	for _, e := range pragmaEmbeds {
		embeds = append(embeds, ir.Embed{Pos: makeXPos(e.Pos), Patterns: e.Patterns})
	}
	typecheck.Target.Embeds = append(typecheck.Target.Embeds, name)
	name.Embed = &embeds
}

func checkEmbed(decl *syntax.VarDecl, haveEmbed, withinFunc bool) error {
	switch {
	case !haveEmbed:
		return errors.New("go:embed only allowed in Go files that import \"embed\"")
	case len(decl.NameList) > 1:
		return errors.New("go:embed cannot apply to multiple vars")
	case decl.Values != nil:
		return errors.New("go:embed cannot apply to var with initializer")
	case decl.Type == nil:
		// Should not happen, since Values == nil now.
		return errors.New("go:embed cannot apply to var without type")
	case withinFunc:
		return errors.New("go:embed cannot apply to var inside func")
	case !types.AllowsGoVersion(types.LocalPkg, 1, 16):
		return fmt.Errorf("go:embed requires go1.16 or later (-lang was set to %s; check go.mod)", base.Flag.Lang)

	default:
		return nil
	}
}
