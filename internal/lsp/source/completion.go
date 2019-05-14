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
	plainSnippet *snippet.Builder

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
	placeholderSnippet *snippet.Builder
}

// Snippet is a convenience function that determines the snippet that should be
// used for an item, depending on if the callee wants placeholders or not.
func (i *CompletionItem) Snippet(usePlaceholders bool) string {
	if usePlaceholders {
		if i.placeholderSnippet != nil {
			return i.placeholderSnippet.String()
		}
	}
	if i.plainSnippet != nil {
		return i.plainSnippet.String()
	}
	return i.InsertText
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
	prefix Prefix

	// expectedType is the type we expect the completion candidate to be.
	// It may not be set.
	expectedType types.Type

	// enclosingFunction is the function declaration enclosing the position.
	enclosingFunction *types.Signature

	// preferTypeNames is true if we are completing at a position that expects a type,
	// not a value.
	preferTypeNames bool

	// enclosingCompositeLiteral contains information about the composite literal
	// enclosing the position.
	enclosingCompositeLiteral *compLitInfo
}

type compLitInfo struct {
	// cl is the *ast.CompositeLit enclosing the position.
	cl *ast.CompositeLit

	// clType is the type of cl.
	clType types.Type

	// kv is the *ast.KeyValueExpr enclosing the position, if any.
	kv *ast.KeyValueExpr

	// inKey is true if we are certain the position is in the key side
	// of a key-value pair.
	inKey bool

	// maybeInFieldName is true if inKey is false and it is possible
	// we are completing a struct field name. For example,
	// "SomeStruct{<>}" will be inKey=false, but maybeInFieldName=true
	// because we _could_ be completing a field name.
	maybeInFieldName bool
}

type Prefix struct {
	content string
	pos     token.Pos
}

func (p Prefix) Content() string { return p.content }
func (p Prefix) Pos() token.Pos  { return p.pos }

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
func Completion(ctx context.Context, f File, pos token.Pos) ([]CompletionItem, Prefix, error) {
	file := f.GetAST(ctx)
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, Prefix{}, fmt.Errorf("package for %s is ill typed", f.URI())
	}

	// Completion is based on what precedes the cursor.
	// Find the path to the position before pos.
	path, _ := astutil.PathEnclosingInterval(file, pos-1, pos-1)
	if path == nil {
		return nil, Prefix{}, fmt.Errorf("cannot find node enclosing position")
	}
	// Skip completion inside comments.
	for _, g := range file.Comments {
		if g.Pos() <= pos && pos <= g.End() {
			return nil, Prefix{}, nil
		}
	}
	// Skip completion inside any kind of literal.
	if _, ok := path[0].(*ast.BasicLit); ok {
		return nil, Prefix{}, nil
	}

	clInfo := enclosingCompositeLiteral(path, pos, pkg.GetTypesInfo())
	c := &completer{
		types:                     pkg.GetTypes(),
		info:                      pkg.GetTypesInfo(),
		qf:                        qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo()),
		view:                      f.View(),
		ctx:                       ctx,
		path:                      path,
		pos:                       pos,
		seen:                      make(map[types.Object]bool),
		enclosingFunction:         enclosingFunction(path, pos, pkg.GetTypesInfo()),
		preferTypeNames:           preferTypeNames(path, pos),
		enclosingCompositeLiteral: clInfo,
	}

	// Set the filter prefix.
	if ident, ok := path[0].(*ast.Ident); ok {
		c.prefix = Prefix{
			content: ident.Name[:pos-ident.Pos()],
			pos:     ident.Pos(),
		}
	}

	c.expectedType = expectedType(c)

	// Struct literals are handled entirely separately.
	if c.wantStructFieldCompletions() {
		if err := c.structLiteralFieldName(); err != nil {
			return nil, Prefix{}, err
		}
		return c.items, c.prefix, nil
	}

	switch n := path[0].(type) {
	case *ast.Ident:
		// Is this the Sel part of a selector?
		if sel, ok := path[1].(*ast.SelectorExpr); ok && sel.Sel == n {
			if err := c.selector(sel); err != nil {
				return nil, Prefix{}, err
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
				return nil, Prefix{}, fmt.Errorf("this is a definition%s", of)
			}
		}
		if err := c.lexical(); err != nil {
			return nil, Prefix{}, err
		}

	// The function name hasn't been typed yet, but the parens are there:
	//   recv.‸(arg)
	case *ast.TypeAssertExpr:
		// Create a fake selector expression.
		if err := c.selector(&ast.SelectorExpr{X: n.X}); err != nil {
			return nil, Prefix{}, err
		}

	case *ast.SelectorExpr:
		if err := c.selector(n); err != nil {
			return nil, Prefix{}, err
		}

	default:
		// fallback to lexical completions
		if err := c.lexical(); err != nil {
			return nil, Prefix{}, err
		}
	}

	return c.items, c.prefix, nil
}

func (c *completer) wantStructFieldCompletions() bool {
	clInfo := c.enclosingCompositeLiteral
	if clInfo == nil {
		return false
	}

	return clInfo.isStruct() && (clInfo.inKey || clInfo.maybeInFieldName)
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

// structLiteralFieldName finds completions for struct field names inside a struct literal.
func (c *completer) structLiteralFieldName() error {
	clInfo := c.enclosingCompositeLiteral

	// Mark fields of the composite literal that have already been set,
	// except for the current field.
	addedFields := make(map[*types.Var]bool)
	for _, el := range clInfo.cl.Elts {
		if kvExpr, ok := el.(*ast.KeyValueExpr); ok {
			if clInfo.kv == kvExpr {
				continue
			}

			if key, ok := kvExpr.Key.(*ast.Ident); ok {
				if used, ok := c.info.Uses[key]; ok {
					if usedVar, ok := used.(*types.Var); ok {
						addedFields[usedVar] = true
					}
				}
			}
		}
	}

	switch t := clInfo.clType.(type) {
	case *types.Struct:
		for i := 0; i < t.NumFields(); i++ {
			field := t.Field(i)
			if !addedFields[field] {
				c.found(field, highScore)
			}
		}

		// Add lexical completions if we aren't certain we are in the key part of a
		// key-value pair.
		if clInfo.maybeInFieldName {
			return c.lexical()
		}
	default:
		return c.lexical()
	}

	return nil
}

func (cl *compLitInfo) isStruct() bool {
	_, ok := cl.clType.(*types.Struct)
	return ok
}

// enclosingCompositeLiteral returns information about the composite literal enclosing the
// position.
func enclosingCompositeLiteral(path []ast.Node, pos token.Pos, info *types.Info) *compLitInfo {
	for _, n := range path {
		switch n := n.(type) {
		case *ast.CompositeLit:
			// The enclosing node will be a composite literal if the user has just
			// opened the curly brace (e.g. &x{<>) or the completion request is triggered
			// from an already completed composite literal expression (e.g. &x{foo: 1, <>})
			//
			// The position is not part of the composite literal unless it falls within the
			// curly braces (e.g. "foo.Foo<>Struct{}").
			if !(n.Lbrace <= pos && pos <= n.Rbrace) {
				return nil
			}

			tv, ok := info.Types[n]
			if !ok {
				return nil
			}

			clInfo := compLitInfo{
				cl:     n,
				clType: tv.Type.Underlying(),
			}

			var (
				expr    ast.Expr
				hasKeys bool
			)
			for _, el := range n.Elts {
				// Remember the expression that the position falls in, if any.
				if el.Pos() <= pos && pos <= el.End() {
					expr = el
				}

				if kv, ok := el.(*ast.KeyValueExpr); ok {
					hasKeys = true
					// If expr == el then we know the position falls in this expression,
					// so also record kv as the enclosing *ast.KeyValueExpr.
					if expr == el {
						clInfo.kv = kv
						break
					}
				}
			}

			if clInfo.kv != nil {
				// If in a *ast.KeyValueExpr, we know we are in the key if the position
				// is to the left of the colon (e.g. "Foo{F<>: V}".
				clInfo.inKey = pos <= clInfo.kv.Colon
			} else if hasKeys {
				// If we aren't in a *ast.KeyValueExpr but the composite literal has
				// other *ast.KeyValueExprs, we must be on the key side of a new
				// *ast.KeyValueExpr (e.g. "Foo{F: V, <>}").
				clInfo.inKey = true
			} else {
				switch clInfo.clType.(type) {
				case *types.Struct:
					if len(n.Elts) == 0 {
						// If the struct literal is empty, next could be a struct field
						// name or an expression (e.g. "Foo{<>}" could become "Foo{F:}"
						// or "Foo{someVar}").
						clInfo.maybeInFieldName = true
					} else if len(n.Elts) == 1 {
						// If there is one expression and the position is in that expression
						// and the expression is an identifier, we may be writing a field
						// name or an expression (e.g. "Foo{F<>}").
						_, clInfo.maybeInFieldName = expr.(*ast.Ident)
					}
				case *types.Map:
					// If we aren't in a *ast.KeyValueExpr we must be adding a new key
					// to the map.
					clInfo.inKey = true
				}
			}

			return &clInfo
		default:
			if breaksExpectedTypeInference(n) {
				return nil
			}
		}
	}

	return nil
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

func (c *completer) expectedCompositeLiteralType() types.Type {
	clInfo := c.enclosingCompositeLiteral
	switch t := clInfo.clType.(type) {
	case *types.Slice:
		if clInfo.inKey {
			return types.Typ[types.Int]
		}
		return t.Elem()
	case *types.Array:
		if clInfo.inKey {
			return types.Typ[types.Int]
		}
		return t.Elem()
	case *types.Map:
		if clInfo.inKey {
			return t.Key()
		}
		return t.Elem()
	case *types.Struct:
		// If we are completing a key (i.e. field name), there is no expected type.
		if clInfo.inKey {
			return nil
		}

		// If we are in a key-value pair, but not in the key, then we must be on the
		// value side. The expected type of the value will be determined from the key.
		if clInfo.kv != nil {
			if key, ok := clInfo.kv.Key.(*ast.Ident); ok {
				for i := 0; i < t.NumFields(); i++ {
					if field := t.Field(i); field.Name() == key.Name {
						return field.Type()
					}
				}
			}
		} else {
			// If we aren't in a key-value pair and aren't in the key, we must be using
			// implicit field names.

			// The order of the literal fields must match the order in the struct definition.
			// Find the element that the position belongs to and suggest that field's type.
			if i := indexExprAtPos(c.pos, clInfo.cl.Elts); i < t.NumFields() {
				return t.Field(i).Type()
			}
		}
	}
	return nil
}

// expectedType returns the expected type for an expression at the query position.
func expectedType(c *completer) types.Type {
	if c.enclosingCompositeLiteral != nil {
		return c.expectedCompositeLiteralType()
	}

	var (
		derefCount int // count of deref "*" operators
		refCount   int // count of reference "&" operators
		typ        types.Type
	)

Nodes:
	for _, node := range c.path {
		switch expr := node.(type) {
		case *ast.BinaryExpr:
			// Determine if query position comes from left or right of op.
			e := expr.X
			if c.pos < expr.OpPos {
				e = expr.Y
			}
			if tv, ok := c.info.Types[e]; ok {
				typ = tv.Type
				break Nodes
			}
		case *ast.AssignStmt:
			// Only rank completions if you are on the right side of the token.
			if c.pos > expr.TokPos {
				i := indexExprAtPos(c.pos, expr.Rhs)
				if i >= len(expr.Lhs) {
					i = len(expr.Lhs) - 1
				}
				if tv, ok := c.info.Types[expr.Lhs[i]]; ok {
					typ = tv.Type
					break Nodes
				}
			}
			return nil
		case *ast.CallExpr:
			// Only consider CallExpr args if position falls between parens.
			if expr.Lparen <= c.pos && c.pos <= expr.Rparen {
				if tv, ok := c.info.Types[expr.Fun]; ok {
					if sig, ok := tv.Type.(*types.Signature); ok {
						if sig.Params().Len() == 0 {
							return nil
						}
						i := indexExprAtPos(c.pos, expr.Args)
						// Make sure not to run past the end of expected parameters.
						if i >= sig.Params().Len() {
							i = sig.Params().Len() - 1
						}
						typ = sig.Params().At(i).Type()
						break Nodes
					}
				}
			}
			return nil
		case *ast.ReturnStmt:
			if sig := c.enclosingFunction; sig != nil {
				// Find signature result that corresponds to our return expression.
				if resultIdx := indexExprAtPos(c.pos, expr.Results); resultIdx < len(expr.Results) {
					if resultIdx < sig.Results().Len() {
						typ = sig.Results().At(resultIdx).Type()
						break Nodes
					}
				}
			}

			return nil
		case *ast.StarExpr:
			derefCount++
		case *ast.UnaryExpr:
			if expr.Op == token.AND {
				refCount++
			}
		default:
			if breaksExpectedTypeInference(node) {
				return nil
			}
		}
	}

	if typ != nil {
		// For every "*" deref operator, add another pointer layer to expected type.
		for i := 0; i < derefCount; i++ {
			typ = types.NewPointer(typ)
		}
		// For every "&" ref operator, remove a pointer layer from expected type.
		for i := 0; i < refCount; i++ {
			if ptr, ok := typ.(*types.Pointer); ok {
				typ = ptr.Elem()
			} else {
				break
			}
		}
	}

	return typ
}

// breaksExpectedTypeInference reports if an expression node's type is unrelated
// to its child expression node types. For example, "Foo{Bar: x.Baz(<>)}" should
// expect a function argument, not a composite literal value.
func breaksExpectedTypeInference(n ast.Node) bool {
	switch n.(type) {
	case *ast.FuncLit, *ast.CallExpr, *ast.TypeAssertExpr, *ast.IndexExpr, *ast.SliceExpr, *ast.CompositeLit:
		return true
	default:
		return false
	}
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
