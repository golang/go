// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

func parseFiles(filenames []string) uint {
	var lines uint
	var noders []*noder

	for _, filename := range filenames {
		p := &noder{err: make(chan syntax.Error)}
		noders = append(noders, p)

		go func(filename string) {
			defer close(p.err)
			base := src.NewFileBase(filename, absFilename(filename))

			f, err := os.Open(filename)
			if err != nil {
				p.error(syntax.Error{Pos: src.MakePos(base, 0, 0), Msg: err.Error()})
				return
			}
			defer f.Close()

			p.file, _ = syntax.Parse(base, f, p.error, p.pragma, syntax.CheckBranches) // errors are tracked via p.error
		}(filename)
	}

	for _, p := range noders {
		for e := range p.err {
			yyerrorpos(e.Pos, "%s", e.Msg)
		}

		p.node()
		lines += p.file.Lines
		p.file = nil // release memory

		if nsyntaxerrors != 0 {
			errorexit()
		}
		// Always run testdclstack here, even when debug_dclstack is not set, as a sanity measure.
		testdclstack()
	}

	return lines
}

func yyerrorpos(pos src.Pos, format string, args ...interface{}) {
	yyerrorl(Ctxt.PosTable.XPos(pos), format, args...)
}

var pathPrefix string

func absFilename(name string) string {
	return objabi.AbsFile(Ctxt.Pathname, name, pathPrefix)
}

// noder transforms package syntax's AST into a Node tree.
type noder struct {
	file       *syntax.File
	linknames  []linkname
	pragcgobuf string
	err        chan syntax.Error
	scope      ScopeID
}

func (p *noder) funchdr(n *Node, pos src.Pos) ScopeID {
	old := p.scope
	p.scope = 0
	funchdr(n)
	return old
}

func (p *noder) funcbody(n *Node, pos src.Pos, old ScopeID) {
	funcbody(n)
	p.scope = old
}

func (p *noder) openScope(pos src.Pos) {
	types.Markdcl()

	if trackScopes {
		Curfn.Func.Parents = append(Curfn.Func.Parents, p.scope)
		p.scope = ScopeID(len(Curfn.Func.Parents))

		p.markScope(pos)
	}
}

func (p *noder) closeScope(pos src.Pos) {
	types.Popdcl()

	if trackScopes {
		p.scope = Curfn.Func.Parents[p.scope-1]

		p.markScope(pos)
	}
}

func (p *noder) markScope(pos src.Pos) {
	xpos := Ctxt.PosTable.XPos(pos)
	if i := len(Curfn.Func.Marks); i > 0 && Curfn.Func.Marks[i-1].Pos == xpos {
		Curfn.Func.Marks[i-1].Scope = p.scope
	} else {
		Curfn.Func.Marks = append(Curfn.Func.Marks, Mark{xpos, p.scope})
	}
}

// closeAnotherScope is like closeScope, but it reuses the same mark
// position as the last closeScope call. This is useful for "for" and
// "if" statements, as their implicit blocks always end at the same
// position as an explicit block.
func (p *noder) closeAnotherScope() {
	types.Popdcl()

	if trackScopes {
		p.scope = Curfn.Func.Parents[p.scope-1]
		Curfn.Func.Marks[len(Curfn.Func.Marks)-1].Scope = p.scope
	}
}

// linkname records a //go:linkname directive.
type linkname struct {
	pos    src.Pos
	local  string
	remote string
}

func (p *noder) node() {
	types.Block = 1
	imported_unsafe = false

	p.lineno(p.file.PkgName)
	mkpackage(p.file.PkgName.Value)

	xtop = append(xtop, p.decls(p.file.DeclList)...)

	for _, n := range p.linknames {
		if imported_unsafe {
			lookup(n.local).Linkname = n.remote
		} else {
			yyerrorpos(n.pos, "//go:linkname only allowed in Go files that import \"unsafe\"")
		}
	}

	pragcgobuf += p.pragcgobuf
	lineno = src.NoXPos
	clearImports()
}

func (p *noder) decls(decls []syntax.Decl) (l []*Node) {
	var cs constState

	for _, decl := range decls {
		p.lineno(decl)
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
	val := p.basicLit(imp.Path)
	ipkg := importfile(&val)

	if ipkg == nil {
		if nerrors == 0 {
			Fatalf("phase error in import")
		}
		return
	}

	ipkg.Direct = true

	var my *types.Sym
	if imp.LocalPkgName != nil {
		my = p.name(imp.LocalPkgName)
	} else {
		my = lookup(ipkg.Name)
	}

	pack := p.nod(imp, OPACK, nil, nil)
	pack.Sym = my
	pack.Name.Pkg = ipkg

	if my.Name == "." {
		importdot(ipkg, pack)
		return
	}
	if my.Name == "init" {
		yyerrorl(pack.Pos, "cannot import package as init - init must be a func")
		return
	}
	if my.Name == "_" {
		return
	}
	if my.Def != nil {
		lineno = pack.Pos
		redeclare(my, "as imported package name")
	}
	my.Def = asTypesNode(pack)
	my.Lastlineno = pack.Pos
	my.Block = 1 // at top level
}

func (p *noder) varDecl(decl *syntax.VarDecl) []*Node {
	names := p.declNames(decl.NameList)
	typ := p.typeExprOrNil(decl.Type)

	var exprs []*Node
	if decl.Values != nil {
		exprs = p.exprList(decl.Values)
	}

	p.lineno(decl)
	return variter(names, typ, exprs)
}

// constState tracks state between constant specifiers within a
// declaration group. This state is kept separate from noder so nested
// constant declarations are handled correctly (e.g., issue 15550).
type constState struct {
	group  *syntax.Group
	typ    *Node
	values []*Node
	iota   int64
}

func (p *noder) constDecl(decl *syntax.ConstDecl, cs *constState) []*Node {
	if decl.Group == nil || decl.Group != cs.group {
		*cs = constState{
			group: decl.Group,
		}
	}

	names := p.declNames(decl.NameList)
	typ := p.typeExprOrNil(decl.Type)

	var values []*Node
	if decl.Values != nil {
		values = p.exprList(decl.Values)
		cs.typ, cs.values = typ, values
	} else {
		if typ != nil {
			yyerror("const declaration cannot have type without expression")
		}
		typ, values = cs.typ, cs.values
	}

	var nn []*Node
	for i, n := range names {
		if i >= len(values) {
			yyerror("missing value in const declaration")
			break
		}
		v := values[i]
		if decl.Values == nil {
			v = treecopy(v, n.Pos)
		}

		n.Op = OLITERAL
		declare(n, dclcontext)

		n.Name.Param.Ntype = typ
		n.Name.Defn = v
		n.SetIota(cs.iota)

		nn = append(nn, p.nod(decl, ODCLCONST, n, nil))
	}

	if len(values) > len(names) {
		yyerror("extra expression in const declaration")
	}

	cs.iota++

	return nn
}

func (p *noder) typeDecl(decl *syntax.TypeDecl) *Node {
	n := p.declName(decl.Name)
	n.Op = OTYPE
	declare(n, dclcontext)
	n.SetLocal(true)

	// decl.Type may be nil but in that case we got a syntax error during parsing
	typ := p.typeExprOrNil(decl.Type)

	param := n.Name.Param
	param.Ntype = typ
	param.Pragma = decl.Pragma
	param.Alias = decl.Alias
	if param.Alias && param.Pragma != 0 {
		yyerror("cannot specify directive with type alias")
		param.Pragma = 0
	}

	return p.nod(decl, ODCLTYPE, n, nil)

}

func (p *noder) declNames(names []*syntax.Name) []*Node {
	var nodes []*Node
	for _, name := range names {
		nodes = append(nodes, p.declName(name))
	}
	return nodes
}

func (p *noder) declName(name *syntax.Name) *Node {
	// TODO(mdempsky): Set lineno?
	return dclname(p.name(name))
}

func (p *noder) funcDecl(fun *syntax.FuncDecl) *Node {
	name := p.name(fun.Name)
	t := p.signature(fun.Recv, fun.Type)
	f := p.nod(fun, ODCLFUNC, nil, nil)

	if fun.Recv == nil {
		if name.Name == "init" {
			name = renameinit()
			if t.List.Len() > 0 || t.Rlist.Len() > 0 {
				yyerrorl(f.Pos, "func init must have no arguments and no return values")
			}
		}

		if localpkg.Name == "main" && name.Name == "main" {
			if t.List.Len() > 0 || t.Rlist.Len() > 0 {
				yyerrorl(f.Pos, "func main must have no arguments and no return values")
			}
		}
	} else {
		f.Func.Shortname = name
		name = nblank.Sym // filled in by typecheckfunc
	}

	f.Func.Nname = newfuncname(name)
	f.Func.Nname.Name.Defn = f
	f.Func.Nname.Name.Param.Ntype = t

	pragma := fun.Pragma
	f.Func.Pragma = fun.Pragma
	f.SetNoescape(pragma&Noescape != 0)
	if pragma&Systemstack != 0 && pragma&Nosplit != 0 {
		yyerrorl(f.Pos, "go:nosplit and go:systemstack cannot be combined")
	}

	if fun.Recv == nil {
		declare(f.Func.Nname, PFUNC)
	}

	oldScope := p.funchdr(f, fun.Pos())

	endPos := fun.Pos()
	if fun.Body != nil {
		if f.Noescape() {
			yyerrorl(f.Pos, "can only use //go:noescape with external func implementations")
		}

		body := p.stmts(fun.Body.List)
		if body == nil {
			body = []*Node{p.nod(fun, OEMPTY, nil, nil)}
		}
		f.Nbody.Set(body)

		endPos = fun.Body.Rbrace
		lineno = Ctxt.PosTable.XPos(fun.Body.Rbrace)
		f.Func.Endlineno = lineno
	} else {
		if pure_go || strings.HasPrefix(f.funcname(), "init.") {
			yyerrorl(f.Pos, "missing function body for %q", f.funcname())
		}
	}

	p.funcbody(f, endPos, oldScope)
	return f
}

func (p *noder) signature(recv *syntax.Field, typ *syntax.FuncType) *Node {
	n := p.nod(typ, OTFUNC, nil, nil)
	if recv != nil {
		n.Left = p.param(recv, false, false)
	}
	n.List.Set(p.params(typ.ParamList, true))
	n.Rlist.Set(p.params(typ.ResultList, false))
	return n
}

func (p *noder) params(params []*syntax.Field, dddOk bool) []*Node {
	var nodes []*Node
	for i, param := range params {
		p.lineno(param)
		nodes = append(nodes, p.param(param, dddOk, i+1 == len(params)))
	}
	return nodes
}

func (p *noder) param(param *syntax.Field, dddOk, final bool) *Node {
	var name *Node
	if param.Name != nil {
		name = p.newname(param.Name)
	}

	typ := p.typeExpr(param.Type)
	n := p.nod(param, ODCLFIELD, name, typ)

	// rewrite ...T parameter
	if typ.Op == ODDD {
		if !dddOk {
			yyerror("cannot use ... in receiver or result parameter list")
		} else if !final {
			yyerror("can only use ... with final parameter in list")
		}
		typ.Op = OTARRAY
		typ.Right = typ.Left
		typ.Left = nil
		n.SetIsddd(true)
		if n.Left != nil {
			n.Left.SetIsddd(true)
		}
	}

	return n
}

func (p *noder) exprList(expr syntax.Expr) []*Node {
	if list, ok := expr.(*syntax.ListExpr); ok {
		return p.exprs(list.ElemList)
	}
	return []*Node{p.expr(expr)}
}

func (p *noder) exprs(exprs []syntax.Expr) []*Node {
	var nodes []*Node
	for _, expr := range exprs {
		nodes = append(nodes, p.expr(expr))
	}
	return nodes
}

func (p *noder) expr(expr syntax.Expr) *Node {
	p.lineno(expr)
	switch expr := expr.(type) {
	case nil, *syntax.BadExpr:
		return nil
	case *syntax.Name:
		return p.mkname(expr)
	case *syntax.BasicLit:
		return p.setlineno(expr, nodlit(p.basicLit(expr)))

	case *syntax.CompositeLit:
		n := p.nod(expr, OCOMPLIT, nil, nil)
		if expr.Type != nil {
			n.Right = p.expr(expr.Type)
		}
		l := p.exprs(expr.ElemList)
		for i, e := range l {
			l[i] = p.wrapname(expr.ElemList[i], e)
		}
		n.List.Set(l)
		lineno = Ctxt.PosTable.XPos(expr.Rbrace)
		return n
	case *syntax.KeyValueExpr:
		return p.nod(expr, OKEY, p.expr(expr.Key), p.wrapname(expr.Value, p.expr(expr.Value)))
	case *syntax.FuncLit:
		return p.funcLit(expr)
	case *syntax.ParenExpr:
		return p.nod(expr, OPAREN, p.expr(expr.X), nil)
	case *syntax.SelectorExpr:
		// parser.new_dotname
		obj := p.expr(expr.X)
		if obj.Op == OPACK {
			obj.Name.SetUsed(true)
			return oldname(restrictlookup(expr.Sel.Value, obj.Name.Pkg))
		}
		return p.setlineno(expr, nodSym(OXDOT, obj, p.name(expr.Sel)))
	case *syntax.IndexExpr:
		return p.nod(expr, OINDEX, p.expr(expr.X), p.expr(expr.Index))
	case *syntax.SliceExpr:
		op := OSLICE
		if expr.Full {
			op = OSLICE3
		}
		n := p.nod(expr, op, p.expr(expr.X), nil)
		var index [3]*Node
		for i, x := range expr.Index {
			if x != nil {
				index[i] = p.expr(x)
			}
		}
		n.SetSliceBounds(index[0], index[1], index[2])
		return n
	case *syntax.AssertExpr:
		if expr.Type == nil {
			panic("unexpected AssertExpr")
		}
		// TODO(mdempsky): parser.pexpr uses p.expr(), but
		// seems like the type field should be parsed with
		// ntype? Shrug, doesn't matter here.
		return p.nod(expr, ODOTTYPE, p.expr(expr.X), p.expr(expr.Type))
	case *syntax.Operation:
		x := p.expr(expr.X)
		if expr.Y == nil {
			if expr.Op == syntax.And {
				x = unparen(x) // TODO(mdempsky): Needed?
				if x.Op == OCOMPLIT {
					// Special case for &T{...}: turn into (*T){...}.
					// TODO(mdempsky): Switch back to p.nod after we
					// get rid of gcCompat.
					x.Right = nod(OIND, x.Right, nil)
					x.Right.SetImplicit(true)
					return x
				}
			}
			return p.nod(expr, p.unOp(expr.Op), x, nil)
		}
		return p.nod(expr, p.binOp(expr.Op), x, p.expr(expr.Y))
	case *syntax.CallExpr:
		n := p.nod(expr, OCALL, p.expr(expr.Fun), nil)
		n.List.Set(p.exprs(expr.ArgList))
		n.SetIsddd(expr.HasDots)
		return n

	case *syntax.ArrayType:
		var len *Node
		if expr.Len != nil {
			len = p.expr(expr.Len)
		} else {
			len = p.nod(expr, ODDD, nil, nil)
		}
		return p.nod(expr, OTARRAY, len, p.typeExpr(expr.Elem))
	case *syntax.SliceType:
		return p.nod(expr, OTARRAY, nil, p.typeExpr(expr.Elem))
	case *syntax.DotsType:
		return p.nod(expr, ODDD, p.typeExpr(expr.Elem), nil)
	case *syntax.StructType:
		return p.structType(expr)
	case *syntax.InterfaceType:
		return p.interfaceType(expr)
	case *syntax.FuncType:
		return p.signature(nil, expr)
	case *syntax.MapType:
		return p.nod(expr, OTMAP, p.typeExpr(expr.Key), p.typeExpr(expr.Value))
	case *syntax.ChanType:
		n := p.nod(expr, OTCHAN, p.typeExpr(expr.Elem), nil)
		n.Etype = types.EType(p.chanDir(expr.Dir))
		return n

	case *syntax.TypeSwitchGuard:
		n := p.nod(expr, OTYPESW, nil, p.expr(expr.X))
		if expr.Lhs != nil {
			n.Left = p.declName(expr.Lhs)
			if isblank(n.Left) {
				yyerror("invalid variable name %v in type switch", n.Left)
			}
		}
		return n
	}
	panic("unhandled Expr")
}

func (p *noder) typeExpr(typ syntax.Expr) *Node {
	// TODO(mdempsky): Be stricter? typecheck should handle errors anyway.
	return p.expr(typ)
}

func (p *noder) typeExprOrNil(typ syntax.Expr) *Node {
	if typ != nil {
		return p.expr(typ)
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

func (p *noder) structType(expr *syntax.StructType) *Node {
	var l []*Node
	for i, field := range expr.FieldList {
		p.lineno(field)
		var n *Node
		if field.Name == nil {
			n = p.embedded(field.Type)
		} else {
			n = p.nod(field, ODCLFIELD, p.newname(field.Name), p.typeExpr(field.Type))
		}
		if i < len(expr.TagList) && expr.TagList[i] != nil {
			n.SetVal(p.basicLit(expr.TagList[i]))
		}
		l = append(l, n)
	}

	p.lineno(expr)
	n := p.nod(expr, OTSTRUCT, nil, nil)
	n.List.Set(l)
	return n
}

func (p *noder) interfaceType(expr *syntax.InterfaceType) *Node {
	var l []*Node
	for _, method := range expr.MethodList {
		p.lineno(method)
		var n *Node
		if method.Name == nil {
			n = p.nod(method, ODCLFIELD, nil, oldname(p.packname(method.Type)))
		} else {
			mname := p.newname(method.Name)
			sig := p.typeExpr(method.Type)
			sig.Left = fakeRecv()
			n = p.nod(method, ODCLFIELD, mname, sig)
			ifacedcl(n)
		}
		l = append(l, n)
	}

	n := p.nod(expr, OTINTER, nil, nil)
	n.List.Set(l)
	return n
}

func (p *noder) packname(expr syntax.Expr) *types.Sym {
	switch expr := expr.(type) {
	case *syntax.Name:
		name := p.name(expr)
		if n := oldname(name); n.Name != nil && n.Name.Pack != nil {
			n.Name.Pack.Name.SetUsed(true)
		}
		return name
	case *syntax.SelectorExpr:
		name := p.name(expr.X.(*syntax.Name))
		var pkg *types.Pkg
		if asNode(name.Def) == nil || asNode(name.Def).Op != OPACK {
			yyerror("%v is not a package", name)
			pkg = localpkg
		} else {
			asNode(name.Def).Name.SetUsed(true)
			pkg = asNode(name.Def).Name.Pkg
		}
		return restrictlookup(expr.Sel.Value, pkg)
	}
	panic(fmt.Sprintf("unexpected packname: %#v", expr))
}

func (p *noder) embedded(typ syntax.Expr) *Node {
	op, isStar := typ.(*syntax.Operation)
	if isStar {
		if op.Op != syntax.Mul || op.Y != nil {
			panic("unexpected Operation")
		}
		typ = op.X
	}
	n := embedded(p.packname(typ), localpkg)
	if isStar {
		n.Right = p.nod(op, OIND, n.Right, nil)
	}
	return n
}

func (p *noder) stmts(stmts []syntax.Stmt) []*Node {
	var nodes []*Node
	for _, stmt := range stmts {
		s := p.stmt(stmt)
		if s == nil {
		} else if s.Op == OBLOCK && s.Ninit.Len() == 0 {
			nodes = append(nodes, s.List.Slice()...)
		} else {
			nodes = append(nodes, s)
		}
	}
	return nodes
}

func (p *noder) stmt(stmt syntax.Stmt) *Node {
	p.lineno(stmt)
	switch stmt := stmt.(type) {
	case *syntax.EmptyStmt:
		return nil
	case *syntax.LabeledStmt:
		return p.labeledStmt(stmt)
	case *syntax.BlockStmt:
		l := p.blockStmt(stmt)
		if len(l) == 0 {
			// TODO(mdempsky): Line number?
			return nod(OEMPTY, nil, nil)
		}
		return liststmt(l)
	case *syntax.ExprStmt:
		return p.wrapname(stmt, p.expr(stmt.X))
	case *syntax.SendStmt:
		return p.nod(stmt, OSEND, p.expr(stmt.Chan), p.expr(stmt.Value))
	case *syntax.DeclStmt:
		return liststmt(p.decls(stmt.DeclList))
	case *syntax.AssignStmt:
		if stmt.Op != 0 && stmt.Op != syntax.Def {
			n := p.nod(stmt, OASOP, p.expr(stmt.Lhs), p.expr(stmt.Rhs))
			n.SetImplicit(stmt.Rhs == syntax.ImplicitOne)
			n.Etype = types.EType(p.binOp(stmt.Op))
			return n
		}

		lhs := p.exprList(stmt.Lhs)
		rhs := p.exprList(stmt.Rhs)

		n := p.nod(stmt, OAS, nil, nil) // assume common case

		if stmt.Op == syntax.Def {
			n.SetColas(true)
			colasdefn(lhs, n) // modifies lhs, call before using lhs[0] in common case
		}

		if len(lhs) == 1 && len(rhs) == 1 {
			// common case
			n.Left = lhs[0]
			n.Right = rhs[0]
		} else {
			n.Op = OAS2
			n.List.Set(lhs)
			n.Rlist.Set(rhs)
		}
		return n

	case *syntax.BranchStmt:
		var op Op
		switch stmt.Tok {
		case syntax.Break:
			op = OBREAK
		case syntax.Continue:
			op = OCONTINUE
		case syntax.Fallthrough:
			op = OXFALL
		case syntax.Goto:
			op = OGOTO
		default:
			panic("unhandled BranchStmt")
		}
		n := p.nod(stmt, op, nil, nil)
		if stmt.Label != nil {
			n.Left = p.newname(stmt.Label)
		}
		if op == OXFALL {
			n.Xoffset = int64(types.Block)
		}
		return n
	case *syntax.CallStmt:
		var op Op
		switch stmt.Tok {
		case syntax.Defer:
			op = ODEFER
		case syntax.Go:
			op = OPROC
		default:
			panic("unhandled CallStmt")
		}
		return p.nod(stmt, op, p.expr(stmt.Call), nil)
	case *syntax.ReturnStmt:
		var results []*Node
		if stmt.Results != nil {
			results = p.exprList(stmt.Results)
		}
		n := p.nod(stmt, ORETURN, nil, nil)
		n.List.Set(results)
		if n.List.Len() == 0 && Curfn != nil {
			for _, ln := range Curfn.Func.Dcl {
				if ln.Class() == PPARAM {
					continue
				}
				if ln.Class() != PPARAMOUT {
					break
				}
				if asNode(ln.Sym.Def) != ln {
					yyerror("%s is shadowed during return", ln.Sym.Name)
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

func (p *noder) blockStmt(stmt *syntax.BlockStmt) []*Node {
	p.openScope(stmt.Pos())
	nodes := p.stmts(stmt.List)
	p.closeScope(stmt.Rbrace)
	return nodes
}

func (p *noder) ifStmt(stmt *syntax.IfStmt) *Node {
	p.openScope(stmt.Pos())
	n := p.nod(stmt, OIF, nil, nil)
	if stmt.Init != nil {
		n.Ninit.Set1(p.stmt(stmt.Init))
	}
	if stmt.Cond != nil {
		n.Left = p.expr(stmt.Cond)
	}
	n.Nbody.Set(p.blockStmt(stmt.Then))
	if stmt.Else != nil {
		e := p.stmt(stmt.Else)
		if e.Op == OBLOCK && e.Ninit.Len() == 0 {
			n.Rlist.Set(e.List.Slice())
		} else {
			n.Rlist.Set1(e)
		}
	}
	p.closeAnotherScope()
	return n
}

func (p *noder) forStmt(stmt *syntax.ForStmt) *Node {
	p.openScope(stmt.Pos())
	var n *Node
	if r, ok := stmt.Init.(*syntax.RangeClause); ok {
		if stmt.Cond != nil || stmt.Post != nil {
			panic("unexpected RangeClause")
		}

		n = p.nod(r, ORANGE, nil, p.expr(r.X))
		if r.Lhs != nil {
			lhs := p.exprList(r.Lhs)
			n.List.Set(lhs)
			if r.Def {
				n.SetColas(true)
				colasdefn(lhs, n)
			}
		}
	} else {
		n = p.nod(stmt, OFOR, nil, nil)
		if stmt.Init != nil {
			n.Ninit.Set1(p.stmt(stmt.Init))
		}
		if stmt.Cond != nil {
			n.Left = p.expr(stmt.Cond)
		}
		if stmt.Post != nil {
			n.Right = p.stmt(stmt.Post)
		}
	}
	n.Nbody.Set(p.blockStmt(stmt.Body))
	p.closeAnotherScope()
	return n
}

func (p *noder) switchStmt(stmt *syntax.SwitchStmt) *Node {
	p.openScope(stmt.Pos())
	n := p.nod(stmt, OSWITCH, nil, nil)
	if stmt.Init != nil {
		n.Ninit.Set1(p.stmt(stmt.Init))
	}
	if stmt.Tag != nil {
		n.Left = p.expr(stmt.Tag)
	}

	tswitch := n.Left
	if tswitch != nil && (tswitch.Op != OTYPESW || tswitch.Left == nil) {
		tswitch = nil
	}
	n.List.Set(p.caseClauses(stmt.Body, tswitch, stmt.Rbrace))

	p.closeScope(stmt.Rbrace)
	return n
}

func (p *noder) caseClauses(clauses []*syntax.CaseClause, tswitch *Node, rbrace src.Pos) []*Node {
	var nodes []*Node
	for i, clause := range clauses {
		p.lineno(clause)
		if i > 0 {
			p.closeScope(clause.Pos())
		}
		p.openScope(clause.Pos())

		n := p.nod(clause, OXCASE, nil, nil)
		if clause.Cases != nil {
			n.List.Set(p.exprList(clause.Cases))
		}
		if tswitch != nil {
			nn := newname(tswitch.Left.Sym)
			declare(nn, dclcontext)
			n.Rlist.Set1(nn)
			// keep track of the instances for reporting unused
			nn.Name.Defn = tswitch
		}
		n.Xoffset = int64(types.Block)
		n.Nbody.Set(p.stmts(clause.Body))
		nodes = append(nodes, n)
	}
	if len(clauses) > 0 {
		p.closeScope(rbrace)
	}
	return nodes
}

func (p *noder) selectStmt(stmt *syntax.SelectStmt) *Node {
	n := p.nod(stmt, OSELECT, nil, nil)
	n.List.Set(p.commClauses(stmt.Body, stmt.Rbrace))
	return n
}

func (p *noder) commClauses(clauses []*syntax.CommClause, rbrace src.Pos) []*Node {
	var nodes []*Node
	for i, clause := range clauses {
		p.lineno(clause)
		if i > 0 {
			p.closeScope(clause.Pos())
		}
		p.openScope(clause.Pos())

		n := p.nod(clause, OXCASE, nil, nil)
		if clause.Comm != nil {
			n.List.Set1(p.stmt(clause.Comm))
		}
		n.Xoffset = int64(types.Block)
		n.Nbody.Set(p.stmts(clause.Body))
		nodes = append(nodes, n)
	}
	if len(clauses) > 0 {
		p.closeScope(rbrace)
	}
	return nodes
}

func (p *noder) labeledStmt(label *syntax.LabeledStmt) *Node {
	lhs := p.nod(label, OLABEL, p.newname(label.Label), nil)

	var ls *Node
	if label.Stmt != nil { // TODO(mdempsky): Should always be present.
		ls = p.stmt(label.Stmt)
	}

	lhs.Name.Defn = ls
	l := []*Node{lhs}
	if ls != nil {
		if ls.Op == OBLOCK && ls.Ninit.Len() == 0 {
			l = append(l, ls.List.Slice()...)
		} else {
			l = append(l, ls)
		}
	}
	return liststmt(l)
}

var unOps = [...]Op{
	syntax.Recv: ORECV,
	syntax.Mul:  OIND,
	syntax.And:  OADDR,

	syntax.Not: ONOT,
	syntax.Xor: OCOM,
	syntax.Add: OPLUS,
	syntax.Sub: OMINUS,
}

func (p *noder) unOp(op syntax.Operator) Op {
	if uint64(op) >= uint64(len(unOps)) || unOps[op] == 0 {
		panic("invalid Operator")
	}
	return unOps[op]
}

var binOps = [...]Op{
	syntax.OrOr:   OOROR,
	syntax.AndAnd: OANDAND,

	syntax.Eql: OEQ,
	syntax.Neq: ONE,
	syntax.Lss: OLT,
	syntax.Leq: OLE,
	syntax.Gtr: OGT,
	syntax.Geq: OGE,

	syntax.Add: OADD,
	syntax.Sub: OSUB,
	syntax.Or:  OOR,
	syntax.Xor: OXOR,

	syntax.Mul:    OMUL,
	syntax.Div:    ODIV,
	syntax.Rem:    OMOD,
	syntax.And:    OAND,
	syntax.AndNot: OANDNOT,
	syntax.Shl:    OLSH,
	syntax.Shr:    ORSH,
}

func (p *noder) binOp(op syntax.Operator) Op {
	if uint64(op) >= uint64(len(binOps)) || binOps[op] == 0 {
		panic("invalid Operator")
	}
	return binOps[op]
}

func (p *noder) basicLit(lit *syntax.BasicLit) Val {
	// TODO: Don't try to convert if we had syntax errors (conversions may fail).
	//       Use dummy values so we can continue to compile. Eventually, use a
	//       form of "unknown" literals that are ignored during type-checking so
	//       we can continue type-checking w/o spurious follow-up errors.
	switch s := lit.Value; lit.Kind {
	case syntax.IntLit:
		x := new(Mpint)
		x.SetString(s)
		return Val{U: x}

	case syntax.FloatLit:
		x := newMpflt()
		x.SetString(s)
		return Val{U: x}

	case syntax.ImagLit:
		x := new(Mpcplx)
		x.Imag.SetString(strings.TrimSuffix(s, "i"))
		return Val{U: x}

	case syntax.RuneLit:
		var r rune
		if u, err := strconv.Unquote(s); err == nil && len(u) > 0 {
			// Package syntax already reported any errors.
			// Check for them again though because 0 is a
			// better fallback value for invalid rune
			// literals than 0xFFFD.
			if len(u) == 1 {
				r = rune(u[0])
			} else {
				r, _ = utf8.DecodeRuneInString(u)
			}
		}
		x := new(Mpint)
		x.SetInt64(int64(r))
		x.Rune = true
		return Val{U: x}

	case syntax.StringLit:
		if len(s) > 0 && s[0] == '`' {
			// strip carriage returns from raw string
			s = strings.Replace(s, "\r", "", -1)
		}
		// Ignore errors because package syntax already reported them.
		u, _ := strconv.Unquote(s)
		return Val{U: u}

	default:
		panic("unhandled BasicLit kind")
	}
}

func (p *noder) name(name *syntax.Name) *types.Sym {
	return lookup(name.Value)
}

func (p *noder) mkname(name *syntax.Name) *Node {
	// TODO(mdempsky): Set line number?
	return mkname(p.name(name))
}

func (p *noder) newname(name *syntax.Name) *Node {
	// TODO(mdempsky): Set line number?
	return newname(p.name(name))
}

func (p *noder) wrapname(n syntax.Node, x *Node) *Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give them the correct line.
	switch x.Op {
	case OTYPE, OLITERAL:
		if x.Sym == nil {
			break
		}
		fallthrough
	case ONAME, ONONAME, OPACK:
		x = p.nod(n, OPAREN, x, nil)
		x.SetImplicit(true)
	}
	return x
}

func (p *noder) nod(orig syntax.Node, op Op, left, right *Node) *Node {
	return p.setlineno(orig, nod(op, left, right))
}

func (p *noder) setlineno(src_ syntax.Node, dst *Node) *Node {
	pos := src_.Pos()
	if !pos.IsKnown() {
		// TODO(mdempsky): Shouldn't happen. Fix package syntax.
		return dst
	}
	dst.Pos = Ctxt.PosTable.XPos(pos)
	return dst
}

func (p *noder) lineno(n syntax.Node) {
	if n == nil {
		return
	}
	pos := n.Pos()
	if !pos.IsKnown() {
		// TODO(mdempsky): Shouldn't happen. Fix package syntax.
		return
	}
	lineno = Ctxt.PosTable.XPos(pos)
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
	"go:generate":           true,
}

// pragma is called concurrently if files are parsed concurrently.
func (p *noder) pragma(pos src.Pos, text string) syntax.Pragma {
	switch {
	case strings.HasPrefix(text, "line "):
		// line directives are handled by syntax package
		panic("unreachable")

	case strings.HasPrefix(text, "go:linkname "):
		f := strings.Fields(text)
		if len(f) != 3 {
			p.error(syntax.Error{Pos: pos, Msg: "usage: //go:linkname localname linkname"})
			break
		}
		p.linknames = append(p.linknames, linkname{pos, f[1], f[2]})

	case strings.HasPrefix(text, "go:cgo_"):
		p.pragcgobuf += p.pragcgo(pos, text)
		fallthrough // because of //go:cgo_unsafe_args
	default:
		verb := text
		if i := strings.Index(text, " "); i >= 0 {
			verb = verb[:i]
		}
		prag := pragmaValue(verb)
		const runtimePragmas = Systemstack | Nowritebarrier | Nowritebarrierrec | Yeswritebarrierrec
		if !compiling_runtime && prag&runtimePragmas != 0 {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s only allowed in runtime", verb)})
		}
		if prag == 0 && !allowedStdPragmas[verb] && compiling_std {
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf("//%s is not allowed in the standard library", verb)})
		}
		return prag
	}

	return 0
}

func mkname(sym *types.Sym) *Node {
	n := oldname(sym)
	if n.Name != nil && n.Name.Pack != nil {
		n.Name.Pack.Name.SetUsed(true)
	}
	return n
}

func unparen(x *Node) *Node {
	for x.Op == OPAREN {
		x = x.Left
	}
	return x
}
