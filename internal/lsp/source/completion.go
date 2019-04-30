// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/snippet"
)

type CompletionItem struct {
	// Label is the primary text the user sees for this completion item.
	Label string

	// Detail is supplemental information to present to the user.
	// This often contains the type or return type of the completion item.
	Detail string

	// InsertText is the text to insert if this item is selected.
	// Any of the prefix that has already been typed is not trimmed.
	// The insert text does not contain snippets.
	InsertText string

	Kind CompletionItemKind

	// Score is the internal relevance score.
	// A higher score indicates that this completion item is more relevant.
	Score float64

	// Snippet is the LSP snippet for the completion item, without placeholders.
	// The LSP specification contains details about LSP snippets.
	// For example, a snippet for a function with the following signature:
	//
	//     func foo(a, b, c int)
	//
	// would be:
	//
	//     foo(${1:})
	//
	Snippet *snippet.Builder

	// PlaceholderSnippet is the LSP snippet for the completion ite, containing
	// placeholders. The LSP specification contains details about LSP snippets.
	// For example, a placeholder snippet for a function with the following signature:
	//
	//     func foo(a, b, c int)
	//
	// would be:
	//
	//     foo(${1:a int}, ${2: b int}, ${3: c int})
	//
	PlaceholderSnippet *snippet.Builder
}

type CompletionItemKind int

const (
	Unknown CompletionItemKind = iota
	InterfaceCompletionItem
	StructCompletionItem
	TypeCompletionItem
	ConstantCompletionItem
	FieldCompletionItem
	ParameterCompletionItem
	VariableCompletionItem
	FunctionCompletionItem
	MethodCompletionItem
	PackageCompletionItem
)

// Scoring constants are used for weighting the relevance of different candidates.
const (
	// stdScore is the base score for all completion items.
	stdScore float64 = 1.0

	// highScore indicates a very relevant completion item.
	highScore float64 = 10.0

	// lowScore indicates an irrelevant or not useful completion item.
	lowScore float64 = 0.01
)

// completer contains the necessary information for a single completion request.
type completer struct {
	// Package-specific fields.
	types *types.Package
	info  *types.Info
	qf    types.Qualifier

	// view is the View associated with this completion request.
	view View

	// ctx is the context associated with this completion request.
	ctx context.Context

	// pos is the position at which the request was triggered.
	pos token.Pos

	// path is the path of AST nodes enclosing the position.
	path []ast.Node

	// seen is the map that ensures we do not return duplicate results.
	seen map[types.Object]bool

	// items is the list of completion items returned.
	items []CompletionItem

	// prefix is the already-typed portion of the completion candidates.
	prefix string

	// expectedType is the type we expect the completion candidate to be.
	// It may not be set.
	expectedType types.Type

	// enclosingFunction is the function declaration enclosing the position.
	enclosingFunction *types.Signature

	// preferTypeNames is true if we are completing at a position that expects a type,
	// not a value.
	preferTypeNames bool

	// enclosingCompositeLiteral is the composite literal enclosing the position.
	enclosingCompositeLiteral *ast.CompositeLit

	// enclosingKeyValue is the key value expression enclosing the position.
	enclosingKeyValue *ast.KeyValueExpr

	// inCompositeLiteralField is true if we are completing a composite literal field.
	inCompositeLiteralField bool
}

// found adds a candidate completion.
//
// Only the first candidate of a given name is considered.
func (c *completer) found(obj types.Object, weight float64) {
	if obj.Pkg() != nil && obj.Pkg() != c.types && !obj.Exported() {
		return // inaccessible
	}
	if c.seen[obj] {
		return
	}
	c.seen[obj] = true
	if c.matchingType(obj.Type()) {
		weight *= highScore
	}
	if _, ok := obj.(*types.TypeName); !ok && c.preferTypeNames {
		weight *= lowScore
	}
	c.items = append(c.items, c.item(obj, weight))
}

// Completion returns a list of possible candidates for completion, given a
// a file and a position.
//
// The prefix is computed based on the preceding identifier and can be used by
// the client to score the quality of the completion. For instance, some clients
// may tolerate imperfect matches as valid completion results, since users may make typos.
func Completion(ctx context.Context, f File, pos token.Pos) ([]CompletionItem, string, error) {
	file := f.GetAST(ctx)
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, "", fmt.Errorf("package for %s is ill typed", f.URI())
	}

	// Completion is based on what precedes the cursor.
	// Find the path to the position before pos.
	path, _ := astutil.PathEnclosingInterval(file, pos-1, pos-1)
	if path == nil {
		return nil, "", fmt.Errorf("cannot find node enclosing position")
	}
	// Skip completion inside comments.
	for _, g := range file.Comments {
		if g.Pos() <= pos && pos <= g.End() {
			return nil, "", nil
		}
	}
	// Skip completion inside any kind of literal.
	if _, ok := path[0].(*ast.BasicLit); ok {
		return nil, "", nil
	}

	lit, kv, inCompositeLiteralField := enclosingCompositeLiteral(path, pos)
	c := &completer{
		types:                     pkg.GetTypes(),
		info:                      pkg.GetTypesInfo(),
		qf:                        qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo()),
		view:                      f.View(),
		ctx:                       ctx,
		path:                      path,
		pos:                       pos,
		seen:                      make(map[types.Object]bool),
		expectedType:              expectedType(path, pos, pkg.GetTypesInfo()),
		enclosingFunction:         enclosingFunction(path, pos, pkg.GetTypesInfo()),
		preferTypeNames:           preferTypeNames(path, pos),
		enclosingCompositeLiteral: lit,
		enclosingKeyValue:         kv,
		inCompositeLiteralField:   inCompositeLiteralField,
	}

	// Composite literals are handled entirely separately.
	if c.enclosingCompositeLiteral != nil {
		c.expectedType = c.expectedCompositeLiteralType(c.enclosingCompositeLiteral, c.enclosingKeyValue)

		if c.inCompositeLiteralField {
			if err := c.compositeLiteral(c.enclosingCompositeLiteral, c.enclosingKeyValue); err != nil {
				return nil, "", err
			}
			return c.items, c.prefix, nil
		}
	}

	switch n := path[0].(type) {
	case *ast.Ident:
		// Set the filter prefix.
		c.prefix = n.Name[:pos-n.Pos()]

		// Is this the Sel part of a selector?
		if sel, ok := path[1].(*ast.SelectorExpr); ok && sel.Sel == n {
			if err := c.selector(sel); err != nil {
				return nil, "", err
			}
			return c.items, c.prefix, nil
		}
		// reject defining identifiers
		if obj, ok := pkg.GetTypesInfo().Defs[n]; ok {
			if v, ok := obj.(*types.Var); ok && v.IsField() {
				// An anonymous field is also a reference to a type.
			} else {
				of := ""
				if obj != nil {
					qual := types.RelativeTo(pkg.GetTypes())
					of += ", of " + types.ObjectString(obj, qual)
				}
				return nil, "", fmt.Errorf("this is a definition%s", of)
			}
		}
		if err := c.lexical(); err != nil {
			return nil, "", err
		}

	// The function name hasn't been typed yet, but the parens are there:
	//   recv.‸(arg)
	case *ast.TypeAssertExpr:
		// Create a fake selector expression.
		if err := c.selector(&ast.SelectorExpr{X: n.X}); err != nil {
			return nil, "", err
		}

	case *ast.SelectorExpr:
		if err := c.selector(n); err != nil {
			return nil, "", err
		}

	default:
		// fallback to lexical completions
		if err := c.lexical(); err != nil {
			return nil, "", err
		}
	}
	return c.items, c.prefix, nil
}

// selector finds completions for the specified selector expression.
func (c *completer) selector(sel *ast.SelectorExpr) error {
	// Is sel a qualified identifier?
	if id, ok := sel.X.(*ast.Ident); ok {
		if pkgname, ok := c.info.Uses[id].(*types.PkgName); ok {
			// Enumerate package members.
			scope := pkgname.Imported().Scope()
			for _, name := range scope.Names() {
				c.found(scope.Lookup(name), stdScore)
			}
			return nil
		}
	}

	// Invariant: sel is a true selector.
	tv, ok := c.info.Types[sel.X]
	if !ok {
		return fmt.Errorf("cannot resolve %s", sel.X)
	}

	// Add methods of T.
	mset := types.NewMethodSet(tv.Type)
	for i := 0; i < mset.Len(); i++ {
		c.found(mset.At(i).Obj(), stdScore)
	}

	// Add methods of *T.
	if tv.Addressable() && !types.IsInterface(tv.Type) && !isPointer(tv.Type) {
		mset := types.NewMethodSet(types.NewPointer(tv.Type))
		for i := 0; i < mset.Len(); i++ {
			c.found(mset.At(i).Obj(), stdScore)
		}
	}

	// Add fields of T.
	for _, f := range fieldSelections(tv.Type) {
		c.found(f, stdScore)
	}
	return nil
}

// lexical finds completions in the lexical environment.
func (c *completer) lexical() error {
	var scopes []*types.Scope // scopes[i], where i<len(path), is the possibly nil Scope of path[i].
	for _, n := range c.path {
		switch node := n.(type) {
		case *ast.FuncDecl:
			n = node.Type
		case *ast.FuncLit:
			n = node.Type
		}
		scopes = append(scopes, c.info.Scopes[n])
	}
	scopes = append(scopes, c.types.Scope(), types.Universe)

	// Track seen variables to avoid showing completions for shadowed variables.
	// This works since we look at scopes from innermost to outermost.
	seen := make(map[string]struct{})

	// Process scopes innermost first.
	for i, scope := range scopes {
		if scope == nil {
			continue
		}
		for _, name := range scope.Names() {
			declScope, obj := scope.LookupParent(name, c.pos)
			if declScope != scope {
				continue // Name was declared in some enclosing scope, or not at all.
			}
			// If obj's type is invalid, find the AST node that defines the lexical block
			// containing the declaration of obj. Don't resolve types for packages.
			if _, ok := obj.(*types.PkgName); !ok && obj.Type() == types.Typ[types.Invalid] {
				// Match the scope to its ast.Node. If the scope is the package scope,
				// use the *ast.File as the starting node.
				var node ast.Node
				if i < len(c.path) {
					node = c.path[i]
				} else if i == len(c.path) { // use the *ast.File for package scope
					node = c.path[i-1]
				}
				if node != nil {
					if resolved := resolveInvalid(obj, node, c.info); resolved != nil {
						obj = resolved
					}
				}
			}

			score := stdScore
			// Rank builtins significantly lower than other results.
			if scope == types.Universe {
				score *= 0.1
			}
			// If we haven't already added a candidate for an object with this name.
			if _, ok := seen[obj.Name()]; !ok {
				seen[obj.Name()] = struct{}{}
				c.found(obj, score)
			}
		}
	}
	return nil
}

// compositeLiteral finds completions for field names inside a composite literal.
func (c *completer) compositeLiteral(lit *ast.CompositeLit, kv *ast.KeyValueExpr) error {
	switch n := c.path[0].(type) {
	case *ast.Ident:
		c.prefix = n.Name[:c.pos-n.Pos()]
	}
	// Mark fields of the composite literal that have already been set,
	// except for the current field.
	hasKeys := kv != nil // true if the composite literal already has key-value pairs
	addedFields := make(map[*types.Var]bool)
	for _, el := range lit.Elts {
		if kvExpr, ok := el.(*ast.KeyValueExpr); ok {
			if kv == kvExpr {
				continue
			}

			hasKeys = true
			if key, ok := kvExpr.Key.(*ast.Ident); ok {
				if used, ok := c.info.Uses[key]; ok {
					if usedVar, ok := used.(*types.Var); ok {
						addedFields[usedVar] = true
					}
				}
			}
		}
	}
	// If the underlying type of the composite literal is a struct,
	// collect completions for the fields of this struct.
	if tv, ok := c.info.Types[lit]; ok {
		switch t := tv.Type.Underlying().(type) {
		case *types.Struct:
			var structPkg *types.Package // package that struct is declared in
			for i := 0; i < t.NumFields(); i++ {
				field := t.Field(i)
				if i == 0 {
					structPkg = field.Pkg()
				}
				if !addedFields[field] {
					c.found(field, highScore)
				}
			}
			// Add lexical completions if the user hasn't typed a key value expression
			// and if the struct fields are defined in the same package as the user is in.
			if !hasKeys && structPkg == c.types {
				return c.lexical()
			}
		default:
			return c.lexical()
		}
	}
	return nil
}

func enclosingCompositeLiteral(path []ast.Node, pos token.Pos) (lit *ast.CompositeLit, kv *ast.KeyValueExpr, ok bool) {
	for _, n := range path {
		switch n := n.(type) {
		case *ast.CompositeLit:
			// The enclosing node will be a composite literal if the user has just
			// opened the curly brace (e.g. &x{<>) or the completion request is triggered
			// from an already completed composite literal expression (e.g. &x{foo: 1, <>})
			//
			// The position is not part of the composite literal unless it falls within the
			// curly braces (e.g. "foo.Foo<>Struct{}").
			if n.Lbrace <= pos && pos <= n.Rbrace {
				lit = n

				// If the cursor position is within a key-value expression inside the composite
				// literal, we try to determine if it is before or after the colon. If it is before
				// the colon, we return field completions. If the cursor does not belong to any
				// expression within the composite literal, we show composite literal completions.
				if expr, isKeyValue := exprAtPos(pos, n.Elts).(*ast.KeyValueExpr); kv == nil && isKeyValue {
					kv = expr

					// If the position belongs to a key-value expression and is after the colon,
					// don't show composite literal completions.
					ok = pos <= kv.Colon
				} else if kv == nil {
					ok = true
				}
			}
			return lit, kv, ok
		case *ast.KeyValueExpr:
			if kv == nil {
				kv = n

				// If the position belongs to a key-value expression and is after the colon,
				// don't show composite literal completions.
				ok = pos <= kv.Colon
			}
		case *ast.FuncType, *ast.CallExpr, *ast.TypeAssertExpr:
			// These node types break the type link between the leaf node and
			// the composite literal. The type of the leaf node becomes unrelated
			// to the type of the composite literal, so we return nil to avoid
			// inappropriate completions. For example, "Foo{Bar: x.Baz(<>)}"
			// should complete as a function argument to Baz, not part of the Foo
			// composite literal.
			return nil, nil, false
		}
	}
	return lit, kv, ok
}

// enclosingFunction returns the signature of the function enclosing the given position.
func enclosingFunction(path []ast.Node, pos token.Pos, info *types.Info) *types.Signature {
	for _, node := range path {
		switch t := node.(type) {
		case *ast.FuncDecl:
			if obj, ok := info.Defs[t.Name]; ok {
				return obj.Type().(*types.Signature)
			}
		case *ast.FuncLit:
			if typ, ok := info.Types[t]; ok {
				return typ.Type.(*types.Signature)
			}
		}
	}
	return nil
}

func (c *completer) expectedCompositeLiteralType(lit *ast.CompositeLit, kv *ast.KeyValueExpr) types.Type {
	litType, ok := c.info.Types[lit]
	if !ok {
		return nil
	}
	switch t := litType.Type.Underlying().(type) {
	case *types.Slice:
		return t.Elem()
	case *types.Array:
		return t.Elem()
	case *types.Map:
		if kv == nil || c.pos <= kv.Colon {
			return t.Key()
		}
		return t.Elem()
	case *types.Struct:
		//  If we are in a key-value expression.
		if kv != nil {
			// There is no expected type for a struct field name.
			if c.pos <= kv.Colon {
				return nil
			}
			// Find the type of the struct field whose name matches the key.
			if key, ok := kv.Key.(*ast.Ident); ok {
				for i := 0; i < t.NumFields(); i++ {
					if field := t.Field(i); field.Name() == key.Name {
						return field.Type()
					}
				}
			}
			return nil
		}
		// We are in a struct literal, but not a specific key-value pair.
		// If the struct literal doesn't have explicit field names,
		// we may still be able to suggest an expected type.
		for _, el := range lit.Elts {
			if _, ok := el.(*ast.KeyValueExpr); ok {
				return nil
			}
		}
		// The order of the literal fields must match the order in the struct definition.
		// Find the element that the position belongs to and suggest that field's type.
		if i := indexExprAtPos(c.pos, lit.Elts); i < t.NumFields() {
			return t.Field(i).Type()
		}
	}
	return nil
}

// expectedType returns the expected type for an expression at the query position.
func expectedType(path []ast.Node, pos token.Pos, info *types.Info) types.Type {
	for i, node := range path {
		if i == 2 {
			break
		}
		switch expr := node.(type) {
		case *ast.BinaryExpr:
			// Determine if query position comes from left or right of op.
			e := expr.X
			if pos < expr.OpPos {
				e = expr.Y
			}
			if tv, ok := info.Types[e]; ok {
				return tv.Type
			}
		case *ast.AssignStmt:
			// Only rank completions if you are on the right side of the token.
			if pos <= expr.TokPos {
				break
			}
			i := indexExprAtPos(pos, expr.Rhs)
			if i >= len(expr.Lhs) {
				i = len(expr.Lhs) - 1
			}
			if tv, ok := info.Types[expr.Lhs[i]]; ok {
				return tv.Type
			}
		case *ast.CallExpr:
			if tv, ok := info.Types[expr.Fun]; ok {
				if sig, ok := tv.Type.(*types.Signature); ok {
					if sig.Params().Len() == 0 {
						return nil
					}
					i := indexExprAtPos(pos, expr.Args)
					// Make sure not to run past the end of expected parameters.
					if i >= sig.Params().Len() {
						i = sig.Params().Len() - 1
					}
					return sig.Params().At(i).Type()
				}
			}
		}
	}
	return nil
}

// preferTypeNames checks if given token position is inside func receiver,
// type params, or type results. For example:
//
// func (<>) foo(<>) (<>) {}
//
func preferTypeNames(path []ast.Node, pos token.Pos) bool {
	for _, p := range path {
		switch n := p.(type) {
		case *ast.FuncDecl:
			if r := n.Recv; r != nil && r.Pos() <= pos && pos <= r.End() {
				return true
			}
			if t := n.Type; t != nil {
				if p := t.Params; p != nil && p.Pos() <= pos && pos <= p.End() {
					return true
				}
				if r := t.Results; r != nil && r.Pos() <= pos && pos <= r.End() {
					return true
				}
			}
			return false
		}
	}
	return false
}

// matchingTypes reports whether actual is a good candidate type
// for a completion in a context of the expected type.
func (c *completer) matchingType(actual types.Type) bool {
	if c.expectedType == nil {
		return false
	}
	// Use a function's return type as its type.
	if sig, ok := actual.(*types.Signature); ok {
		if sig.Results().Len() == 1 {
			actual = sig.Results().At(0).Type()
		}
	}
	return types.Identical(types.Default(c.expectedType), types.Default(actual))
}
