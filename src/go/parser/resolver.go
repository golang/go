// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

const debugResolve = false

// resolveFile walks the given file to resolve identifiers within the file
// scope, updating ast.Ident.Obj fields with declaration information.
//
// If declErr is non-nil, it is used to report declaration errors during
// resolution. tok is used to format position in error messages.
func resolveFile(file *ast.File, handle *token.File, declErr func(token.Pos, string)) {
	pkgScope := ast.NewScope(nil)
	r := &resolver{
		handle:   handle,
		declErr:  declErr,
		topScope: pkgScope,
		pkgScope: pkgScope,
		depth:    1,
	}

	for _, decl := range file.Decls {
		ast.Walk(r, decl)
	}

	r.closeScope()
	assert(r.topScope == nil, "unbalanced scopes")
	assert(r.labelScope == nil, "unbalanced label scopes")

	// resolve global identifiers within the same file
	i := 0
	for _, ident := range r.unresolved {
		// i <= index for current ident
		assert(ident.Obj == unresolved, "object already resolved")
		ident.Obj = r.pkgScope.Lookup(ident.Name) // also removes unresolved sentinel
		if ident.Obj == nil {
			r.unresolved[i] = ident
			i++
		} else if debugResolve {
			pos := ident.Obj.Decl.(interface{ Pos() token.Pos }).Pos()
			r.trace("resolved %s@%v to package object %v", ident.Name, ident.Pos(), pos)
		}
	}
	file.Scope = r.pkgScope
	file.Unresolved = r.unresolved[0:i]
}

const maxScopeDepth int = 1e3

type resolver struct {
	handle  *token.File
	declErr func(token.Pos, string)

	// Ordinary identifier scopes
	pkgScope   *ast.Scope   // pkgScope.Outer == nil
	topScope   *ast.Scope   // top-most scope; may be pkgScope
	unresolved []*ast.Ident // unresolved identifiers
	depth      int          // scope depth

	// Label scopes
	// (maintained by open/close LabelScope)
	labelScope  *ast.Scope     // label scope for current function
	targetStack [][]*ast.Ident // stack of unresolved labels
}

func (r *resolver) trace(format string, args ...any) {
	fmt.Println(strings.Repeat(". ", r.depth) + r.sprintf(format, args...))
}

func (r *resolver) sprintf(format string, args ...any) string {
	for i, arg := range args {
		switch arg := arg.(type) {
		case token.Pos:
			args[i] = r.handle.Position(arg)
		}
	}
	return fmt.Sprintf(format, args...)
}

func (r *resolver) openScope(pos token.Pos) {
	r.depth++
	if r.depth > maxScopeDepth {
		panic(bailout{pos: pos, msg: "exceeded max scope depth during object resolution"})
	}
	if debugResolve {
		r.trace("opening scope @%v", pos)
	}
	r.topScope = ast.NewScope(r.topScope)
}

func (r *resolver) closeScope() {
	r.depth--
	if debugResolve {
		r.trace("closing scope")
	}
	r.topScope = r.topScope.Outer
}

func (r *resolver) openLabelScope() {
	r.labelScope = ast.NewScope(r.labelScope)
	r.targetStack = append(r.targetStack, nil)
}

func (r *resolver) closeLabelScope() {
	// resolve labels
	n := len(r.targetStack) - 1
	scope := r.labelScope
	for _, ident := range r.targetStack[n] {
		ident.Obj = scope.Lookup(ident.Name)
		if ident.Obj == nil && r.declErr != nil {
			r.declErr(ident.Pos(), fmt.Sprintf("label %s undefined", ident.Name))
		}
	}
	// pop label scope
	r.targetStack = r.targetStack[0:n]
	r.labelScope = r.labelScope.Outer
}

func (r *resolver) declare(decl, data any, scope *ast.Scope, kind ast.ObjKind, idents ...*ast.Ident) {
	for _, ident := range idents {
		if ident.Obj != nil {
			panic(fmt.Sprintf("%v: identifier %s already declared or resolved", ident.Pos(), ident.Name))
		}
		obj := ast.NewObj(kind, ident.Name)
		// remember the corresponding declaration for redeclaration
		// errors and global variable resolution/typechecking phase
		obj.Decl = decl
		obj.Data = data
		// Identifiers (for receiver type parameters) are written to the scope, but
		// never set as the resolved object. See go.dev/issue/50956.
		if _, ok := decl.(*ast.Ident); !ok {
			ident.Obj = obj
		}
		if ident.Name != "_" {
			if debugResolve {
				r.trace("declaring %s@%v", ident.Name, ident.Pos())
			}
			if alt := scope.Insert(obj); alt != nil && r.declErr != nil {
				prevDecl := ""
				if pos := alt.Pos(); pos.IsValid() {
					prevDecl = r.sprintf("\n\tprevious declaration at %v", pos)
				}
				r.declErr(ident.Pos(), fmt.Sprintf("%s redeclared in this block%s", ident.Name, prevDecl))
			}
		}
	}
}

func (r *resolver) shortVarDecl(decl *ast.AssignStmt) {
	// Go spec: A short variable declaration may redeclare variables
	// provided they were originally declared in the same block with
	// the same type, and at least one of the non-blank variables is new.
	n := 0 // number of new variables
	for _, x := range decl.Lhs {
		if ident, isIdent := x.(*ast.Ident); isIdent {
			assert(ident.Obj == nil, "identifier already declared or resolved")
			obj := ast.NewObj(ast.Var, ident.Name)
			// remember corresponding assignment for other tools
			obj.Decl = decl
			ident.Obj = obj
			if ident.Name != "_" {
				if debugResolve {
					r.trace("declaring %s@%v", ident.Name, ident.Pos())
				}
				if alt := r.topScope.Insert(obj); alt != nil {
					ident.Obj = alt // redeclaration
				} else {
					n++ // new declaration
				}
			}
		}
	}
	if n == 0 && r.declErr != nil {
		r.declErr(decl.Lhs[0].Pos(), "no new variables on left side of :=")
	}
}

// The unresolved object is a sentinel to mark identifiers that have been added
// to the list of unresolved identifiers. The sentinel is only used for verifying
// internal consistency.
var unresolved = new(ast.Object)

// If x is an identifier, resolve attempts to resolve x by looking up
// the object it denotes. If no object is found and collectUnresolved is
// set, x is marked as unresolved and collected in the list of unresolved
// identifiers.
func (r *resolver) resolve(ident *ast.Ident, collectUnresolved bool) {
	if ident.Obj != nil {
		panic(r.sprintf("%v: identifier %s already declared or resolved", ident.Pos(), ident.Name))
	}
	// '_' should never refer to existing declarations, because it has special
	// handling in the spec.
	if ident.Name == "_" {
		return
	}
	for s := r.topScope; s != nil; s = s.Outer {
		if obj := s.Lookup(ident.Name); obj != nil {
			if debugResolve {
				r.trace("resolved %v:%s to %v", ident.Pos(), ident.Name, obj)
			}
			assert(obj.Name != "", "obj with no name")
			// Identifiers (for receiver type parameters) are written to the scope,
			// but never set as the resolved object. See go.dev/issue/50956.
			if _, ok := obj.Decl.(*ast.Ident); !ok {
				ident.Obj = obj
			}
			return
		}
	}
	// all local scopes are known, so any unresolved identifier
	// must be found either in the file scope, package scope
	// (perhaps in another file), or universe scope --- collect
	// them so that they can be resolved later
	if collectUnresolved {
		ident.Obj = unresolved
		r.unresolved = append(r.unresolved, ident)
	}
}

func (r *resolver) walkExprs(list []ast.Expr) {
	for _, node := range list {
		ast.Walk(r, node)
	}
}

func (r *resolver) walkLHS(list []ast.Expr) {
	for _, expr := range list {
		expr := ast.Unparen(expr)
		if _, ok := expr.(*ast.Ident); !ok && expr != nil {
			ast.Walk(r, expr)
		}
	}
}

func (r *resolver) walkStmts(list []ast.Stmt) {
	for _, stmt := range list {
		ast.Walk(r, stmt)
	}
}

func (r *resolver) Visit(node ast.Node) ast.Visitor {
	if debugResolve && node != nil {
		r.trace("node %T@%v", node, node.Pos())
	}

	switch n := node.(type) {

	// Expressions.
	case *ast.Ident:
		r.resolve(n, true)

	case *ast.FuncLit:
		r.openScope(n.Pos())
		defer r.closeScope()
		r.walkFuncType(n.Type)
		r.walkBody(n.Body)

	case *ast.SelectorExpr:
		ast.Walk(r, n.X)
		// Note: don't try to resolve n.Sel, as we don't support qualified
		// resolution.

	case *ast.StructType:
		r.openScope(n.Pos())
		defer r.closeScope()
		r.walkFieldList(n.Fields, ast.Var)

	case *ast.FuncType:
		r.openScope(n.Pos())
		defer r.closeScope()
		r.walkFuncType(n)

	case *ast.CompositeLit:
		if n.Type != nil {
			ast.Walk(r, n.Type)
		}
		for _, e := range n.Elts {
			if kv, _ := e.(*ast.KeyValueExpr); kv != nil {
				// See go.dev/issue/45160: try to resolve composite lit keys, but don't
				// collect them as unresolved if resolution failed. This replicates
				// existing behavior when resolving during parsing.
				if ident, _ := kv.Key.(*ast.Ident); ident != nil {
					r.resolve(ident, false)
				} else {
					ast.Walk(r, kv.Key)
				}
				ast.Walk(r, kv.Value)
			} else {
				ast.Walk(r, e)
			}
		}

	case *ast.InterfaceType:
		r.openScope(n.Pos())
		defer r.closeScope()
		r.walkFieldList(n.Methods, ast.Fun)

	// Statements
	case *ast.LabeledStmt:
		r.declare(n, nil, r.labelScope, ast.Lbl, n.Label)
		ast.Walk(r, n.Stmt)

	case *ast.AssignStmt:
		r.walkExprs(n.Rhs)
		if n.Tok == token.DEFINE {
			r.shortVarDecl(n)
		} else {
			r.walkExprs(n.Lhs)
		}

	case *ast.BranchStmt:
		// add to list of unresolved targets
		if n.Tok != token.FALLTHROUGH && n.Label != nil {
			depth := len(r.targetStack) - 1
			r.targetStack[depth] = append(r.targetStack[depth], n.Label)
		}

	case *ast.BlockStmt:
		r.openScope(n.Pos())
		defer r.closeScope()
		r.walkStmts(n.List)

	case *ast.IfStmt:
		r.openScope(n.Pos())
		defer r.closeScope()
		if n.Init != nil {
			ast.Walk(r, n.Init)
		}
		ast.Walk(r, n.Cond)
		ast.Walk(r, n.Body)
		if n.Else != nil {
			ast.Walk(r, n.Else)
		}

	case *ast.CaseClause:
		r.walkExprs(n.List)
		r.openScope(n.Pos())
		defer r.closeScope()
		r.walkStmts(n.Body)

	case *ast.SwitchStmt:
		r.openScope(n.Pos())
		defer r.closeScope()
		if n.Init != nil {
			ast.Walk(r, n.Init)
		}
		if n.Tag != nil {
			// The scope below reproduces some unnecessary behavior of the parser,
			// opening an extra scope in case this is a type switch. It's not needed
			// for expression switches.
			// TODO: remove this once we've matched the parser resolution exactly.
			if n.Init != nil {
				r.openScope(n.Tag.Pos())
				defer r.closeScope()
			}
			ast.Walk(r, n.Tag)
		}
		if n.Body != nil {
			r.walkStmts(n.Body.List)
		}

	case *ast.TypeSwitchStmt:
		if n.Init != nil {
			r.openScope(n.Pos())
			defer r.closeScope()
			ast.Walk(r, n.Init)
		}
		r.openScope(n.Assign.Pos())
		defer r.closeScope()
		ast.Walk(r, n.Assign)
		// s.Body consists only of case clauses, so does not get its own
		// scope.
		if n.Body != nil {
			r.walkStmts(n.Body.List)
		}

	case *ast.CommClause:
		r.openScope(n.Pos())
		defer r.closeScope()
		if n.Comm != nil {
			ast.Walk(r, n.Comm)
		}
		r.walkStmts(n.Body)

	case *ast.SelectStmt:
		// as for switch statements, select statement bodies don't get their own
		// scope.
		if n.Body != nil {
			r.walkStmts(n.Body.List)
		}

	case *ast.ForStmt:
		r.openScope(n.Pos())
		defer r.closeScope()
		if n.Init != nil {
			ast.Walk(r, n.Init)
		}
		if n.Cond != nil {
			ast.Walk(r, n.Cond)
		}
		if n.Post != nil {
			ast.Walk(r, n.Post)
		}
		ast.Walk(r, n.Body)

	case *ast.RangeStmt:
		r.openScope(n.Pos())
		defer r.closeScope()
		ast.Walk(r, n.X)
		var lhs []ast.Expr
		if n.Key != nil {
			lhs = append(lhs, n.Key)
		}
		if n.Value != nil {
			lhs = append(lhs, n.Value)
		}
		if len(lhs) > 0 {
			if n.Tok == token.DEFINE {
				// Note: we can't exactly match the behavior of object resolution
				// during the parsing pass here, as it uses the position of the RANGE
				// token for the RHS OpPos. That information is not contained within
				// the AST.
				as := &ast.AssignStmt{
					Lhs:    lhs,
					Tok:    token.DEFINE,
					TokPos: n.TokPos,
					Rhs:    []ast.Expr{&ast.UnaryExpr{Op: token.RANGE, X: n.X}},
				}
				// TODO(rFindley): this walkLHS reproduced the parser resolution, but
				// is it necessary? By comparison, for a normal AssignStmt we don't
				// walk the LHS in case there is an invalid identifier list.
				r.walkLHS(lhs)
				r.shortVarDecl(as)
			} else {
				r.walkExprs(lhs)
			}
		}
		ast.Walk(r, n.Body)

	// Declarations
	case *ast.GenDecl:
		switch n.Tok {
		case token.CONST, token.VAR:
			for i, spec := range n.Specs {
				spec := spec.(*ast.ValueSpec)
				kind := ast.Con
				if n.Tok == token.VAR {
					kind = ast.Var
				}
				r.walkExprs(spec.Values)
				if spec.Type != nil {
					ast.Walk(r, spec.Type)
				}
				r.declare(spec, i, r.topScope, kind, spec.Names...)
			}
		case token.TYPE:
			for _, spec := range n.Specs {
				spec := spec.(*ast.TypeSpec)
				// Go spec: The scope of a type identifier declared inside a function begins
				// at the identifier in the TypeSpec and ends at the end of the innermost
				// containing block.
				r.declare(spec, nil, r.topScope, ast.Typ, spec.Name)
				if spec.TypeParams != nil {
					r.openScope(spec.Pos())
					defer r.closeScope()
					r.walkTParams(spec.TypeParams)
				}
				ast.Walk(r, spec.Type)
			}
		}

	case *ast.FuncDecl:
		// Open the function scope.
		r.openScope(n.Pos())
		defer r.closeScope()

		r.walkRecv(n.Recv)

		// Type parameters are walked normally: they can reference each other, and
		// can be referenced by normal parameters.
		if n.Type.TypeParams != nil {
			r.walkTParams(n.Type.TypeParams)
			// TODO(rFindley): need to address receiver type parameters.
		}

		// Resolve and declare parameters in a specific order to get duplicate
		// declaration errors in the correct location.
		r.resolveList(n.Type.Params)
		r.resolveList(n.Type.Results)
		r.declareList(n.Recv, ast.Var)
		r.declareList(n.Type.Params, ast.Var)
		r.declareList(n.Type.Results, ast.Var)

		r.walkBody(n.Body)
		if n.Recv == nil && n.Name.Name != "init" {
			r.declare(n, nil, r.pkgScope, ast.Fun, n.Name)
		}

	default:
		return r
	}

	return nil
}

func (r *resolver) walkFuncType(typ *ast.FuncType) {
	// typ.TypeParams must be walked separately for FuncDecls.
	r.resolveList(typ.Params)
	r.resolveList(typ.Results)
	r.declareList(typ.Params, ast.Var)
	r.declareList(typ.Results, ast.Var)
}

func (r *resolver) resolveList(list *ast.FieldList) {
	if list == nil {
		return
	}
	for _, f := range list.List {
		if f.Type != nil {
			ast.Walk(r, f.Type)
		}
	}
}

func (r *resolver) declareList(list *ast.FieldList, kind ast.ObjKind) {
	if list == nil {
		return
	}
	for _, f := range list.List {
		r.declare(f, nil, r.topScope, kind, f.Names...)
	}
}

func (r *resolver) walkRecv(recv *ast.FieldList) {
	// If our receiver has receiver type parameters, we must declare them before
	// trying to resolve the rest of the receiver, and avoid re-resolving the
	// type parameter identifiers.
	if recv == nil || len(recv.List) == 0 {
		return // nothing to do
	}
	typ := recv.List[0].Type
	if ptr, ok := typ.(*ast.StarExpr); ok {
		typ = ptr.X
	}

	var declareExprs []ast.Expr // exprs to declare
	var resolveExprs []ast.Expr // exprs to resolve
	switch typ := typ.(type) {
	case *ast.IndexExpr:
		declareExprs = []ast.Expr{typ.Index}
		resolveExprs = append(resolveExprs, typ.X)
	case *ast.IndexListExpr:
		declareExprs = typ.Indices
		resolveExprs = append(resolveExprs, typ.X)
	default:
		resolveExprs = append(resolveExprs, typ)
	}
	for _, expr := range declareExprs {
		if id, _ := expr.(*ast.Ident); id != nil {
			r.declare(expr, nil, r.topScope, ast.Typ, id)
		} else {
			// The receiver type parameter expression is invalid, but try to resolve
			// it anyway for consistency.
			resolveExprs = append(resolveExprs, expr)
		}
	}
	for _, expr := range resolveExprs {
		if expr != nil {
			ast.Walk(r, expr)
		}
	}
	// The receiver is invalid, but try to resolve it anyway for consistency.
	for _, f := range recv.List[1:] {
		if f.Type != nil {
			ast.Walk(r, f.Type)
		}
	}
}

func (r *resolver) walkFieldList(list *ast.FieldList, kind ast.ObjKind) {
	if list == nil {
		return
	}
	r.resolveList(list)
	r.declareList(list, kind)
}

// walkTParams is like walkFieldList, but declares type parameters eagerly so
// that they may be resolved in the constraint expressions held in the field
// Type.
func (r *resolver) walkTParams(list *ast.FieldList) {
	r.declareList(list, ast.Typ)
	r.resolveList(list)
}

func (r *resolver) walkBody(body *ast.BlockStmt) {
	if body == nil {
		return
	}
	r.openLabelScope()
	defer r.closeLabelScope()
	r.walkStmts(body.List)
}
