// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"cmd/compile/internal/syntax"
	"cmd/internal/obj"
)

func parseFile(filename string) {
	errh := func(_, line int, msg string) {
		yyerrorl(lexlineno+int32(line)-1, "%s", msg)
	}

	file, err := syntax.ReadFile(filename, errh, 0)
	if err != nil {
		Fatalf("syntax.ReadFile %s: %v", filename, err)
	}

	p := noder{pragmas: file.Pragmas}

	p.lineno(file.PkgName)
	mkpackage(file.PkgName.Value)

	xtop = append(xtop, p.decls(file.DeclList, true)...)
	p.globalPragmas()
	lexlineno += p.maxline

	if nsyntaxerrors == 0 {
		testdclstack()
	}
}

// noder transforms package syntax's AST into a Nod tree.
type noder struct {
	indent  []byte
	pragmas []syntax.Pragma
	pline   int32
	maxline int32
}

func (p *noder) decls(decls []syntax.Decl, top bool) (l []*Node) {
	var lastConstGroup *syntax.Group
	var lastConstRHS []*Node
	var iotaVal int32

	for _, decl := range decls {
		p.lineno(decl)
		switch decl := decl.(type) {
		case *syntax.ImportDecl:
			p.importDecl(decl)

		case *syntax.VarDecl:
			l = append(l, p.varDecl(decl)...)
		case *syntax.ConstDecl:
			// Tricky to handle golang.org/issue/15550 correctly.

			prevIota := iota_

			if decl.Group == nil || decl.Group != lastConstGroup {
				iotaVal = 0
				lastConstRHS = nil
			}

			iota_ = iotaVal
			lastconst = lastConstRHS

			l = append(l, p.constDecl(decl)...)

			lastConstRHS = lastconst
			lastconst = nil

			iota_ = prevIota
			iotaVal++

			lastConstGroup = decl.Group

		case *syntax.TypeDecl:
			l = append(l, p.typeDecl(decl))

		case *syntax.FuncDecl:
			l = append(l, p.funcDecl(decl))

		default:
			panic("unhandled Decl")
		}

		if top {
			p.pline = p.maxline
		}
	}

	return
}

func (p *noder) importDecl(imp *syntax.ImportDecl) {
	val := p.basicLit(imp.Path)
	importfile(&val, p.indent)
	ipkg := importpkg
	importpkg = nil

	if ipkg == nil {
		if nerrors == 0 {
			Fatalf("phase error in import")
		}
		return
	}

	ipkg.Direct = true

	var my *Sym
	if imp.LocalPkgName != nil {
		my = p.name(imp.LocalPkgName)
	} else {
		my = Lookup(ipkg.Name)
	}

	pack := p.nod(imp, OPACK, nil, nil)
	pack.Sym = my
	pack.Name.Pkg = ipkg

	if my.Name == "." {
		importdot(ipkg, pack)
		return
	}
	if my.Name == "init" {
		yyerrorl(pack.Lineno, "cannot import package as init - init must be a func")
		return
	}
	if my.Name == "_" {
		return
	}
	if my.Def != nil {
		lineno = pack.Lineno
		redeclare(my, "as imported package name")
	}
	my.Def = pack
	my.Lastlineno = pack.Lineno
	my.Block = 1 // at top level
}

func (p *noder) varDecl(decl *syntax.VarDecl) []*Node {
	names := p.declNames(decl.NameList)

	var typ *Node
	if decl.Type != nil {
		typ = p.typeExpr(decl.Type)
	}

	var exprs []*Node
	if decl.Values != nil {
		exprs = p.exprList(decl.Values)
	}

	return variter(names, typ, exprs)
}

func (p *noder) constDecl(decl *syntax.ConstDecl) []*Node {
	names := p.declNames(decl.NameList)

	var typ *Node
	if decl.Type != nil {
		typ = p.typeExpr(decl.Type)
	}

	var exprs []*Node
	if decl.Values != nil {
		exprs = p.exprList(decl.Values)
	}

	return constiter(names, typ, exprs)
}

func (p *noder) typeDecl(decl *syntax.TypeDecl) *Node {
	name := typedcl0(p.name(decl.Name))

	var typ *Node
	if decl.Type != nil {
		typ = p.typeExpr(decl.Type)
	}

	return typedcl1(name, typ, true)
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
	f := p.funcHeader(fun)
	if f == nil {
		return nil
	}

	var body []*Node
	if fun.Body != nil {
		body = p.stmts(fun.Body)
		if body == nil {
			body = []*Node{p.nod(fun, OEMPTY, nil, nil)}
		}
	}

	pragma := p.pragma()

	f.Nbody.Set(body)
	f.Noescape = pragma&Noescape != 0
	if f.Noescape && len(body) != 0 {
		Yyerror("can only use //go:noescape with external func implementations")
	}
	f.Func.Pragma = pragma
	lineno = lexlineno + int32(fun.EndLine) - 1
	f.Func.Endlineno = lineno

	funcbody(f)

	return f
}

func (p *noder) funcHeader(fun *syntax.FuncDecl) *Node {
	name := p.name(fun.Name)
	t := p.signature(fun.Recv, fun.Type)
	f := p.nod(fun, ODCLFUNC, nil, nil)

	if fun.Recv == nil {
		// FunctionName Signature
		if name.Name == "init" {
			name = renameinit()
			if t.List.Len() > 0 || t.Rlist.Len() > 0 {
				Yyerror("func init must have no arguments and no return values")
			}
		}

		if localpkg.Name == "main" && name.Name == "main" {
			if t.List.Len() > 0 || t.Rlist.Len() > 0 {
				Yyerror("func main must have no arguments and no return values")
			}
		}

		f.Func.Nname = newfuncname(name)
	} else {
		// Receiver MethodName Signature

		f.Func.Shortname = newfuncname(name)
		f.Func.Nname = methodname1(f.Func.Shortname, t.Left.Right)
	}

	f.Func.Nname.Name.Defn = f
	f.Func.Nname.Name.Param.Ntype = t // TODO: check if nname already has an ntype

	declare(f.Func.Nname, PFUNC)
	funchdr(f)
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
			Yyerror("cannot use ... in receiver or result parameter list")
		} else if !final {
			Yyerror("can only use ... with final parameter in list")
		}
		typ.Op = OTARRAY
		typ.Right = typ.Left
		typ.Left = nil
		n.Isddd = true
		if n.Left != nil {
			n.Left.Isddd = true
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
	case nil:
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
		// TODO(mdempsky): Should apply wrapname to n.List nodes.
		n.List.Set(p.exprs(expr.ElemList))
		return n
	case *syntax.KeyValueExpr:
		return p.nod(expr, OKEY, p.expr(expr.Key), p.wrapname(expr.Value, p.expr(expr.Value)))
	case *syntax.FuncLit:
		closurehdr(p.typeExpr(expr.Type))
		body := p.stmts(expr.Body)
		lineno = lexlineno + int32(expr.EndLine) - 1
		return p.setlineno(expr, closurebody(body))
	case *syntax.ParenExpr:
		return p.nod(expr, OPAREN, p.expr(expr.X), nil)
	case *syntax.SelectorExpr:
		// parser.new_dotname
		obj := p.expr(expr.X)
		sel := p.name(expr.Sel)
		if obj.Op == OPACK {
			s := restrictlookup(sel.Name, obj.Name.Pkg)
			obj.Used = true
			return oldname(s)
		}
		return p.setlineno(expr, NodSym(OXDOT, obj, sel))
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
					x.Right = p.nod(expr, OIND, x.Right, nil)
					x.Right.Implicit = true
					return x
				}
			}
			return p.nod(expr, p.unOp(expr.Op), x, nil)
		}
		return p.nod(expr, p.binOp(expr.Op), x, p.expr(expr.Y))
	case *syntax.CallExpr:
		n := p.nod(expr, OCALL, p.expr(expr.Fun), nil)
		n.List.Set(p.exprs(expr.ArgList))
		n.Isddd = expr.HasDots
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
		n.Etype = EType(p.chanDir(expr.Dir))
		return n

	case *syntax.TypeSwitchGuard:
		n := p.nod(expr, OTYPESW, nil, p.expr(expr.X))
		if expr.Lhs != nil {
			n.Left = p.declName(expr.Lhs)
			if isblank(n.Left) {
				Yyerror("invalid variable name %s in type switch", n.Left)
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

func (p *noder) chanDir(dir syntax.ChanDir) ChanDir {
	switch dir {
	case 0:
		return Cboth
	case syntax.SendOnly:
		return Csend
	case syntax.RecvOnly:
		return Crecv
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
			sig.Left = fakethis()
			n = p.nod(method, ODCLFIELD, mname, sig)
			ifacedcl(n)
		}
		l = append(l, n)
	}

	n := p.nod(expr, OTINTER, nil, nil)
	n.List.Set(l)
	return n
}

func (p *noder) packname(expr syntax.Expr) *Sym {
	switch expr := expr.(type) {
	case *syntax.Name:
		name := p.name(expr)
		if n := oldname(name); n.Name != nil && n.Name.Pack != nil {
			n.Name.Pack.Used = true
		}
		return name
	case *syntax.SelectorExpr:
		name := p.name(expr.X.(*syntax.Name))
		s := p.name(expr.Sel)
		var pkg *Pkg
		if name.Def == nil || name.Def.Op != OPACK {
			Yyerror("%v is not a package", name)
			pkg = localpkg
		} else {
			name.Def.Used = true
			pkg = name.Def.Name.Pkg
		}
		return restrictlookup(s.Name, pkg)
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
		return p.body(stmt.Body)
	case *syntax.ExprStmt:
		return p.wrapname(stmt, p.expr(stmt.X))
	case *syntax.SendStmt:
		return p.nod(stmt, OSEND, p.expr(stmt.Chan), p.expr(stmt.Value))
	case *syntax.DeclStmt:
		return liststmt(p.decls(stmt.DeclList, false))
	case *syntax.AssignStmt:
		if stmt.Op != 0 && stmt.Op != syntax.Def {
			n := p.nod(stmt, OASOP, p.expr(stmt.Lhs), p.expr(stmt.Rhs))
			n.Implicit = stmt.Rhs == syntax.ImplicitOne
			n.Etype = EType(p.binOp(stmt.Op))
			return n
		}

		lhs := p.exprList(stmt.Lhs)
		rhs := p.exprList(stmt.Rhs)

		n := p.nod(stmt, OAS, nil, nil) // assume common case

		if stmt.Op == syntax.Def {
			n.Colas = true
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
		if op == OGOTO {
			n.Sym = dclstack // context, for goto restriction
		}
		if op == OXFALL {
			n.Xoffset = int64(block)
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
				if ln.Class == PPARAM {
					continue
				}
				if ln.Class != PPARAMOUT {
					break
				}
				if ln.Sym.Def != ln {
					Yyerror("%s is shadowed during return", ln.Sym.Name)
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

func (p *noder) body(body []syntax.Stmt) *Node {
	l := p.bodyList(body)
	if len(l) == 0 {
		// TODO(mdempsky): Line number?
		return Nod(OEMPTY, nil, nil)
	}
	return liststmt(l)
}

func (p *noder) bodyList(body []syntax.Stmt) []*Node {
	markdcl()
	nodes := p.stmts(body)
	popdcl()
	return nodes
}

func (p *noder) ifStmt(stmt *syntax.IfStmt) *Node {
	markdcl()
	n := p.nod(stmt, OIF, nil, nil)
	if stmt.Init != nil {
		n.Ninit.Set1(p.stmt(stmt.Init))
	}
	if stmt.Cond != nil {
		n.Left = p.expr(stmt.Cond)
	}
	n.Nbody.Set(p.bodyList(stmt.Then))
	if stmt.Else != nil {
		e := p.stmt(stmt.Else)
		if e.Op == OBLOCK && e.Ninit.Len() == 0 {
			n.Rlist.Set(e.List.Slice())
		} else {
			n.Rlist.Set1(e)
		}
	}
	popdcl()
	return n
}

func (p *noder) forStmt(stmt *syntax.ForStmt) *Node {
	markdcl()
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
				n.Colas = true
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
	n.Nbody.Set(p.bodyList(stmt.Body))
	popdcl()
	return n
}

func (p *noder) switchStmt(stmt *syntax.SwitchStmt) *Node {
	markdcl()
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

	n.List.Set(p.caseClauses(stmt.Body, tswitch))

	popdcl()
	return n
}

func (p *noder) caseClauses(clauses []*syntax.CaseClause, tswitch *Node) []*Node {
	var nodes []*Node
	for _, clause := range clauses {
		p.lineno(clause)
		markdcl()
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
		n.Xoffset = int64(block)
		n.Nbody.Set(p.stmts(clause.Body))
		popdcl()
		nodes = append(nodes, n)
	}
	return nodes
}

func (p *noder) selectStmt(stmt *syntax.SelectStmt) *Node {
	n := p.nod(stmt, OSELECT, nil, nil)
	n.List.Set(p.commClauses(stmt.Body))
	return n
}

func (p *noder) commClauses(clauses []*syntax.CommClause) []*Node {
	var nodes []*Node
	for _, clause := range clauses {
		p.lineno(clause)
		markdcl()
		n := p.nod(clause, OXCASE, nil, nil)
		if clause.Comm != nil {
			n.List.Set1(p.stmt(clause.Comm))
		}
		n.Xoffset = int64(block)
		n.Nbody.Set(p.stmts(clause.Body))
		popdcl()
		nodes = append(nodes, n)
	}
	return nodes
}

func (p *noder) labeledStmt(label *syntax.LabeledStmt) *Node {
	lhs := p.nod(label, OLABEL, p.newname(label.Label), nil)
	lhs.Sym = dclstack

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

func (p *noder) name(name *syntax.Name) *Sym {
	return Lookup(name.Value)
}

func (p *noder) mkname(name *syntax.Name) *Node {
	// TODO(mdempsky): Set line number?
	return mkname(p.name(name))
}

func (p *noder) newname(name *syntax.Name) *Node {
	// TODO(mdempsky): Set line number?
	return newname(p.name(name))
}

func (p *noder) oldname(name *syntax.Name) *Node {
	return oldname(p.name(name))
}

func (p *noder) wrapname(n syntax.Node, x *Node) *Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give them the correct line.
	switch x.Op {
	case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
		x = p.nod(n, OPAREN, x, nil)
		x.Implicit = true
	}
	return x
}

func (p *noder) nod(orig syntax.Node, op Op, left, right *Node) *Node {
	return p.setlineno(orig, Nod(op, left, right))
}

func (p *noder) setlineno(src syntax.Node, dst *Node) *Node {
	l := int32(src.Line())
	if l == 0 {
		// TODO(mdempsky): Shouldn't happen. Fix package syntax.
		return dst
	}
	if l > p.maxline {
		p.maxline = l
	}
	dst.Lineno = lexlineno + l - 1
	return dst
}

func (p *noder) lineno(n syntax.Node) {
	if n == nil {
		return
	}
	l := int32(n.Line())
	if l == 0 {
		// TODO(mdempsky): Shouldn't happen. Fix package syntax.
		return
	}
	if l > p.maxline {
		p.maxline = l
	}
	lineno = lexlineno + l - 1
}

func (p *noder) pragma() Pragma {
	lo := sort.Search(len(p.pragmas), func(i int) bool { return int32(p.pragmas[i].Line) >= p.pline })
	hi := sort.Search(len(p.pragmas), func(i int) bool { return int32(p.pragmas[i].Line) > p.maxline })

	var res Pragma
	for _, prag := range p.pragmas[lo:hi] {
		text := prag.Text
		if !strings.HasPrefix(text, "go:") {
			continue
		}

		verb := text
		if i := strings.Index(text, " "); i >= 0 {
			verb = verb[:i]
		}

		switch verb {
		case "go:nointerface":
			if obj.Fieldtrack_enabled != 0 {
				res |= Nointerface
			}
		case "go:noescape":
			res |= Noescape
		case "go:norace":
			res |= Norace
		case "go:nosplit":
			res |= Nosplit
		case "go:noinline":
			res |= Noinline
		case "go:systemstack":
			if !compiling_runtime {
				Yyerror("//go:systemstack only allowed in runtime")
			}
			res |= Systemstack
		case "go:nowritebarrier":
			if !compiling_runtime {
				Yyerror("//go:nowritebarrier only allowed in runtime")
			}
			res |= Nowritebarrier
		case "go:nowritebarrierrec":
			if !compiling_runtime {
				Yyerror("//go:nowritebarrierrec only allowed in runtime")
			}
			res |= Nowritebarrierrec | Nowritebarrier // implies Nowritebarrier
		case "go:cgo_unsafe_args":
			res |= CgoUnsafeArgs
		case "go:uintptrescapes":
			// For the next function declared in the file
			// any uintptr arguments may be pointer values
			// converted to uintptr. This directive
			// ensures that the referenced allocated
			// object, if any, is retained and not moved
			// until the call completes, even though from
			// the types alone it would appear that the
			// object is no longer needed during the
			// call. The conversion to uintptr must appear
			// in the argument list.
			// Used in syscall/dll_windows.go.
			res |= UintptrEscapes
		}
	}
	return res
}

func (p *noder) globalPragmas() {
	origlexlineno := lexlineno
	defer func() {
		lexlineno = origlexlineno
	}()

	for _, prag := range p.pragmas {
		text := prag.Text

		if strings.HasPrefix(text, "go:cgo_") {
			pragcgobuf += pragcgo(text)
		}

		if strings.HasPrefix(text, "go:linkname ") {
			if !imported_unsafe {
				Yyerror("//go:linkname only allowed in Go files that import \"unsafe\"")
			}
			f := strings.Fields(text)
			if len(f) != 3 {
				Yyerror("usage: //go:linkname localname linkname")
				break
			}
			Lookup(f[1]).Linkname = f[2]
		}

		// TODO(mdempsky): Move into package syntax.
		if strings.HasPrefix(text, "line ") {
			i := strings.IndexByte(text, ':')
			if i < 0 {
				continue
			}
			n, err := strconv.Atoi(text[i+1:])
			if err != nil {
				// todo: make this an error instead? it is almost certainly a bug.
				continue
			}
			if n > 1e8 {
				Yyerror("line number out of range")
				errorexit()
			}
			if n <= 0 {
				continue
			}
			lexlineno = origlexlineno + int32(prag.Line)
			linehistupdate(text[5:i], n)
		}
	}
}
