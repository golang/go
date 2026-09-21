// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package free defines utilities for computing the free variables of
// a syntax tree without type information. This is inherently
// heuristic because of the T{f: x} ambiguity, in which f may or may
// not be a lexical reference depending on whether T is a struct type.
package free

import (
	"go/ast"
	"go/token"
)

// Copied, with considerable changes, from go/parser/resolver.go
// at af53bd2c03.

// Names computes an approximation to the set of free names of the AST
// at node n based solely on syntax.
//
// In the absence of composite literals, the set of free names is exact. Composite
// literals introduce an ambiguity that can only be resolved with type information:
// whether F is a field name or a value in `T{F: ...}`.
// If includeComplitIdents is true, this function conservatively assumes
// T is not a struct type, so freeishNames overapproximates: the resulting
// set may contain spurious entries that are not free lexical references
// but are references to struct fields.
// If includeComplitIdents is false, this function assumes that T *is*
// a struct type, so freeishNames underapproximates: the resulting set
// may omit names that are free lexical references.
//
// TODO(adonovan): includeComplitIdents is a crude hammer: the caller
// may have partial or heuristic information about whether a given T
// is struct type. Replace includeComplitIdents with a hook to query
// the caller.
//
// The code is based on go/parser.resolveFile, but heavily simplified. Crucial
// differences are:
//   - Instead of resolving names to their objects, this function merely records
//     whether they are free.
//   - Labels are ignored: they do not refer to values.
//   - This is never called on ImportSpecs, so the function panics if it sees one.
func Names(n ast.Node, includeComplitIdents bool) map[string]bool {
	v := &freeVisitor{
		free:                 make(map[string]bool),
		includeComplitIdents: includeComplitIdents,
	}
	// Begin with a scope, even though n might not be a form that establishes a scope.
	// For example, n might be:
	//    x := ...
	// Then we need to add the first x to some scope.
	v.openScope()
	ast.Walk(v, n)
	v.closeScope()
	if v.scope != nil {
		panic("unbalanced scopes")
	}
	return v.free
}

// A freeVisitor holds state for a free-name analysis.
type freeVisitor struct {
	scope                *scope          // the current innermost scope
	free                 map[string]bool // free names seen so far
	includeComplitIdents bool            // include identifier key in composite literals
}

// scope contains all the names defined in a lexical scope.
// It is like ast.Scope, but without deprecation warnings.
type scope struct {
	names map[string]bool
	outer *scope
}

func (s *scope) defined(name string) bool {
	for ; s != nil; s = s.outer {
		if s.names[name] {
			return true
		}
	}
	return false
}

func (v *freeVisitor) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {

	// Expressions.
	case *ast.Ident:
		v.use(n)

	case *ast.FuncLit:
		v.openScope()
		defer v.closeScope()
		v.walkFuncType(nil, n.Type)
		v.walkBody(n.Body)

	case *ast.SelectorExpr:
		v.walk(n.X)
		// Skip n.Sel: it cannot be free.

	case *ast.StructType:
		v.openScope()
		defer v.closeScope()
		v.walkFieldList(n.Fields)

	case *ast.FuncType:
		v.openScope()
		defer v.closeScope()
		v.walkFuncType(nil, n)

	case *ast.CompositeLit:
		v.walk(n.Type)
		for _, e := range n.Elts {
			if kv, _ := e.(*ast.KeyValueExpr); kv != nil {
				if ident, _ := kv.Key.(*ast.Ident); ident != nil {
					// It is not possible from syntax alone to know whether
					// an identifier used as a composite literal key is
					// a struct field (if n.Type is a struct) or a value
					// (if n.Type is a map, slice or array).
					if v.includeComplitIdents {
						// Over-approximate by treating both cases as potentially
						// free names.
						v.use(ident)
					} else {
						// Under-approximate by ignoring potentially free names.
					}
				} else {
					v.walk(kv.Key)
				}
				v.walk(kv.Value)
			} else {
				v.walk(e)
			}
		}

	case *ast.InterfaceType:
		v.openScope()
		defer v.closeScope()
		v.walkFieldList(n.Methods)

	// Statements
	case *ast.AssignStmt:
		walkSlice(v, n.Rhs)
		if n.Tok == token.DEFINE {
			v.shortVarDecl(n.Lhs)
		} else {
			walkSlice(v, n.Lhs)
		}

	case *ast.LabeledStmt:
		// Ignore labels.
		v.walk(n.Stmt)

	case *ast.BranchStmt:
		// Ignore labels.

	case *ast.BlockStmt:
		v.openScope()
		defer v.closeScope()
		walkSlice(v, n.List)

	case *ast.IfStmt:
		v.openScope()
		defer v.closeScope()
		v.walk(n.Init)
		v.walk(n.Cond)
		v.walk(n.Body)
		v.walk(n.Else)

	case *ast.CaseClause:
		walkSlice(v, n.List)
		v.openScope()
		defer v.closeScope()
		walkSlice(v, n.Body)

	case *ast.SwitchStmt:
		v.openScope()
		defer v.closeScope()
		v.walk(n.Init)
		v.walk(n.Tag)
		v.walkBody(n.Body)

	case *ast.TypeSwitchStmt:
		v.openScope()
		defer v.closeScope()
		if n.Init != nil {
			v.walk(n.Init)
		}
		v.walk(n.Assign)
		// We can use walkBody here because we don't track label scopes.
		v.walkBody(n.Body)

	case *ast.CommClause:
		v.openScope()
		defer v.closeScope()
		v.walk(n.Comm)
		walkSlice(v, n.Body)

	case *ast.SelectStmt:
		v.walkBody(n.Body)

	case *ast.ForStmt:
		v.openScope()
		defer v.closeScope()
		v.walk(n.Init)
		v.walk(n.Cond)
		v.walk(n.Post)
		v.walk(n.Body)

	case *ast.RangeStmt:
		v.openScope()
		defer v.closeScope()
		v.walk(n.X)
		var lhs []ast.Expr
		if n.Key != nil {
			lhs = append(lhs, n.Key)
		}
		if n.Value != nil {
			lhs = append(lhs, n.Value)
		}
		if len(lhs) > 0 {
			if n.Tok == token.DEFINE {
				v.shortVarDecl(lhs)
			} else {
				walkSlice(v, lhs)
			}
		}
		v.walk(n.Body)

	// Declarations
	case *ast.GenDecl:
		switch n.Tok {
		case token.CONST, token.VAR:
			for _, spec := range n.Specs {
				spec := spec.(*ast.ValueSpec)
				walkSlice(v, spec.Values)
				v.walk(spec.Type)
				v.declare(spec.Names...)
			}

		case token.TYPE:
			for _, spec := range n.Specs {
				spec := spec.(*ast.TypeSpec)
				// Go spec: The scope of a type identifier declared inside a
				// function begins at the identifier in the TypeSpec and ends
				// at the end of the innermost containing block.
				v.declare(spec.Name)
				if spec.TypeParams != nil {
					v.openScope()
					defer v.closeScope()
					v.walkTypeParams(spec.TypeParams)
				}
				v.walk(spec.Type)
			}

		case token.IMPORT:
			panic("encountered import declaration in free analysis")
		}

	case *ast.FuncDecl:
		if n.Recv == nil && n.Name.Name != "init" { // package-level function
			v.declare(n.Name)
		}
		v.openScope()
		defer v.closeScope()
		v.walkTypeParams(n.Type.TypeParams)
		v.walkFuncType(n.Recv, n.Type)
		v.walkBody(n.Body)

	default:
		return v
	}

	return nil
}

func (v *freeVisitor) openScope() {
	v.scope = &scope{map[string]bool{}, v.scope}
}

func (v *freeVisitor) closeScope() {
	v.scope = v.scope.outer
}

func (v *freeVisitor) walk(n ast.Node) {
	if n != nil {
		ast.Walk(v, n)
	}
}

func (v *freeVisitor) walkFuncType(recv *ast.FieldList, typ *ast.FuncType) {
	// First use field types...
	v.walkRecvFieldType(recv)
	v.walkFieldTypes(typ.Params)
	v.walkFieldTypes(typ.Results)

	// ...then declare field names.
	v.declareFieldNames(recv)
	v.declareFieldNames(typ.Params)
	v.declareFieldNames(typ.Results)
}

// A receiver field is not like a param or result field because
// "func (recv R[T]) method()" uses R but declares T.
func (v *freeVisitor) walkRecvFieldType(list *ast.FieldList) {
	if list == nil {
		return
	}
	for _, f := range list.List { // valid => len=1
		typ := f.Type
		if ptr, ok := typ.(*ast.StarExpr); ok {
			typ = ptr.X
		}

		// Analyze receiver type as Base[Index, ...]
		var (
			base    ast.Expr
			indices []ast.Expr
		)
		switch typ := typ.(type) {
		case *ast.IndexExpr: // B[T]
			base, indices = typ.X, []ast.Expr{typ.Index}
		case *ast.IndexListExpr: // B[K, V]
			base, indices = typ.X, typ.Indices
		default: // B
			base = typ
		}
		for _, expr := range indices {
			if id, ok := expr.(*ast.Ident); ok {
				v.declare(id)
			}
		}
		v.walk(base)
	}
}

// walkTypeParams is like walkFieldList, but declares type parameters eagerly so
// that they may be resolved in the constraint expressions held in the field
// Type.
func (v *freeVisitor) walkTypeParams(list *ast.FieldList) {
	v.declareFieldNames(list)
	v.walkFieldTypes(list) // constraints
}

func (v *freeVisitor) walkBody(body *ast.BlockStmt) {
	if body == nil {
		return
	}
	walkSlice(v, body.List)
}

func (v *freeVisitor) walkFieldList(list *ast.FieldList) {
	if list == nil {
		return
	}
	v.walkFieldTypes(list)    // .Type may contain references
	v.declareFieldNames(list) // .Names declares names
}

func (v *freeVisitor) shortVarDecl(lhs []ast.Expr) {
	// Go spec: A short variable declaration may redeclare variables provided
	// they were originally declared in the same block with the same type, and
	// at least one of the non-blank variables is new.
	//
	// However, it doesn't matter to free analysis whether a variable is declared
	// fresh or redeclared.
	for _, x := range lhs {
		// In a well-formed program each expr must be an identifier,
		// but be forgiving.
		if id, ok := x.(*ast.Ident); ok {
			v.declare(id)
		}
	}
}

func walkSlice[S ~[]E, E ast.Node](r *freeVisitor, list S) {
	for _, e := range list {
		r.walk(e)
	}
}

// walkFieldTypes resolves the types of the walkFieldTypes in list.
// The companion method declareFieldList declares the names of the walkFieldTypes.
func (v *freeVisitor) walkFieldTypes(list *ast.FieldList) {
	if list != nil {
		for _, f := range list.List {
			v.walk(f.Type)
		}
	}
}

// declareFieldNames declares the names of the fields in list.
// (Names in a FieldList always establish new bindings.)
// The companion method resolveFieldList resolves the types of the fields.
func (v *freeVisitor) declareFieldNames(list *ast.FieldList) {
	if list != nil {
		for _, f := range list.List {
			v.declare(f.Names...)
		}
	}
}

// use marks ident as free if it is not in scope.
func (v *freeVisitor) use(ident *ast.Ident) {
	if s := ident.Name; s != "_" && !v.scope.defined(s) {
		v.free[s] = true
	}
}

// declare adds each non-blank ident to the current scope.
func (v *freeVisitor) declare(idents ...*ast.Ident) {
	for _, id := range idents {
		if id.Name != "_" {
			v.scope.names[id.Name] = true
		}
	}
}
