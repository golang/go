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
	"golang.org/x/tools/internal/span"
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

	// surrounding describes the identifier surrounding the position.
	surrounding *Selection

	// expectedType conains information about the type we expect the completion
	// candidate to be. It will be the zero value if no information is available.
	expectedType typeInference

	// enclosingFunction is the function declaration enclosing the position.
	enclosingFunction *types.Signature

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

// A Selection represents the cursor position and surrounding identifier.
type Selection struct {
	Content string
	Range   span.Range
	Cursor  token.Pos
}

func (p Selection) Prefix() string {
	return p.Content[:p.Cursor-p.Range.Start]
}

func (c *completer) setSurrounding(ident *ast.Ident) {
	if c.surrounding != nil {
		return
	}

	if !(ident.Pos() <= c.pos && c.pos <= ident.End()) {
		return
	}

	c.surrounding = &Selection{
		Content: ident.Name,
		Range:   span.NewRange(c.view.Session().Cache().FileSet(), ident.Pos(), ident.End()),
		Cursor:  c.pos,
	}
}

// found adds a candidate completion.
//
// Only the first candidate of a given name is considered.
func (c *completer) found(obj types.Object, score float64) {
	if obj.Pkg() != nil && obj.Pkg() != c.types && !obj.Exported() {
		return // inaccessible
	}
	if c.seen[obj] {
		return
	}
	c.seen[obj] = true

	cand := candidate{
		obj:   obj,
		score: score,
	}

	if c.matchingType(&cand) {
		cand.score *= highScore
	}

	if c.wantTypeName() && !isTypeName(obj) {
		cand.score *= lowScore
	}

	c.items = append(c.items, c.item(cand))
}

// candidate represents a completion candidate.
type candidate struct {
	// obj is the types.Object to complete to.
	obj types.Object

	// score is used to rank candidates.
	score float64

	// expandFuncCall is true if obj should be invoked in the completion.
	// For example, expandFuncCall=true yields "foo()", expandFuncCall=false yields "foo".
	expandFuncCall bool
}

// Completion returns a list of possible candidates for completion, given a
// a file and a position.
//
// The selection is computed based on the preceding identifier and can be used by
// the client to score the quality of the completion. For instance, some clients
// may tolerate imperfect matches as valid completion results, since users may make typos.
func Completion(ctx context.Context, f GoFile, pos token.Pos) ([]CompletionItem, *Selection, error) {
	file := f.GetAST(ctx)
	if file == nil {
		return nil, nil, fmt.Errorf("no AST for %s", f.URI())
	}
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, nil, fmt.Errorf("package for %s is ill typed", f.URI())
	}

	// Completion is based on what precedes the cursor.
	// Find the path to the position before pos.
	path, _ := astutil.PathEnclosingInterval(file, pos-1, pos-1)
	if path == nil {
		return nil, nil, fmt.Errorf("cannot find node enclosing position")
	}
	// Skip completion inside comments.
	for _, g := range file.Comments {
		if g.Pos() <= pos && pos <= g.End() {
			return nil, nil, nil
		}
	}
	// Skip completion inside any kind of literal.
	if _, ok := path[0].(*ast.BasicLit); ok {
		return nil, nil, nil
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
		enclosingCompositeLiteral: clInfo,
	}

	// Set the filter surrounding.
	if ident, ok := path[0].(*ast.Ident); ok {
		c.setSurrounding(ident)
	}

	c.expectedType = expectedType(c)

	// Struct literals are handled entirely separately.
	if c.wantStructFieldCompletions() {
		if err := c.structLiteralFieldName(); err != nil {
			return nil, nil, err
		}
		return c.items, c.surrounding, nil
	}

	switch n := path[0].(type) {
	case *ast.Ident:
		// Is this the Sel part of a selector?
		if sel, ok := path[1].(*ast.SelectorExpr); ok && sel.Sel == n {
			if err := c.selector(sel); err != nil {
				return nil, nil, err
			}
			return c.items, c.surrounding, nil
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
				return nil, nil, fmt.Errorf("this is a definition%s", of)
			}
		}
		if err := c.lexical(); err != nil {
			return nil, nil, err
		}

	// The function name hasn't been typed yet, but the parens are there:
	//   recv.‸(arg)
	case *ast.TypeAssertExpr:
		// Create a fake selector expression.
		if err := c.selector(&ast.SelectorExpr{X: n.X}); err != nil {
			return nil, nil, err
		}

	case *ast.SelectorExpr:
		// The go parser inserts a phantom "_" Sel node when the selector is
		// not followed by an identifier or a "(". The "_" isn't actually in
		// the text, so don't think it is our surrounding.
		// TODO: Find a way to differentiate between phantom "_" and real "_",
		//       perhaps by checking if "_" is present in file contents.
		if n.Sel.Name != "_" || c.pos != n.Sel.Pos() {
			c.setSurrounding(n.Sel)
		}

		if err := c.selector(n); err != nil {
			return nil, nil, err
		}

	default:
		// fallback to lexical completions
		if err := c.lexical(); err != nil {
			return nil, nil, err
		}
	}

	return c.items, c.surrounding, nil
}

func (c *completer) wantStructFieldCompletions() bool {
	clInfo := c.enclosingCompositeLiteral
	if clInfo == nil {
		return false
	}

	return clInfo.isStruct() && (clInfo.inKey || clInfo.maybeInFieldName)
}

func (c *completer) wantTypeName() bool {
	return c.expectedType.wantTypeName
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

// typeModifier represents an operator that changes the expected type.
type typeModifier int

const (
	dereference typeModifier = iota // dereference ("*") operator
	reference                       // reference ("&") operator
	chanRead                        // channel read ("<-") operator
)

// typeInference holds information we have inferred about a type that can be
// used at the current position.
type typeInference struct {
	// objType is the desired type of an object used at the query position.
	objType types.Type

	// wantTypeName is true if we expect the name of a type.
	wantTypeName bool

	// modifiers are prefixes such as "*", "&" or "<-" that influence how
	// a candidate type relates to the expected type.
	modifiers []typeModifier
}

// expectedType returns information about the expected type for an expression at
// the query position.
func expectedType(c *completer) typeInference {
	if ti := expectTypeName(c); ti.wantTypeName {
		return ti
	}

	if c.enclosingCompositeLiteral != nil {
		return typeInference{objType: c.expectedCompositeLiteralType()}
	}

	var (
		modifiers []typeModifier
		typ       types.Type
	)

Nodes:
	for i, node := range c.path {
		switch node := node.(type) {
		case *ast.BinaryExpr:
			// Determine if query position comes from left or right of op.
			e := node.X
			if c.pos < node.OpPos {
				e = node.Y
			}
			if tv, ok := c.info.Types[e]; ok {
				typ = tv.Type
				break Nodes
			}
		case *ast.AssignStmt:
			// Only rank completions if you are on the right side of the token.
			if c.pos > node.TokPos {
				i := indexExprAtPos(c.pos, node.Rhs)
				if i >= len(node.Lhs) {
					i = len(node.Lhs) - 1
				}
				if tv, ok := c.info.Types[node.Lhs[i]]; ok {
					typ = tv.Type
					break Nodes
				}
			}
			return typeInference{}
		case *ast.CallExpr:
			// Only consider CallExpr args if position falls between parens.
			if node.Lparen <= c.pos && c.pos <= node.Rparen {
				if tv, ok := c.info.Types[node.Fun]; ok {
					if sig, ok := tv.Type.(*types.Signature); ok {
						if sig.Params().Len() == 0 {
							return typeInference{}
						}
						i := indexExprAtPos(c.pos, node.Args)
						// Make sure not to run past the end of expected parameters.
						if i >= sig.Params().Len() {
							i = sig.Params().Len() - 1
						}
						typ = sig.Params().At(i).Type()
						break Nodes
					}
				}
			}
			return typeInference{}
		case *ast.ReturnStmt:
			if sig := c.enclosingFunction; sig != nil {
				// Find signature result that corresponds to our return statement.
				if resultIdx := indexExprAtPos(c.pos, node.Results); resultIdx < len(node.Results) {
					if resultIdx < sig.Results().Len() {
						typ = sig.Results().At(resultIdx).Type()
						break Nodes
					}
				}
			}
			return typeInference{}
		case *ast.CaseClause:
			if swtch, ok := findSwitchStmt(c.path[i+1:], c.pos, node).(*ast.SwitchStmt); ok {
				if tv, ok := c.info.Types[swtch.Tag]; ok {
					typ = tv.Type
					break Nodes
				}
			}
			return typeInference{}
		case *ast.SliceExpr:
			// Make sure position falls within the brackets (e.g. "foo[a:<>]").
			if node.Lbrack < c.pos && c.pos <= node.Rbrack {
				typ = types.Typ[types.Int]
				break Nodes
			}
			return typeInference{}
		case *ast.IndexExpr:
			// Make sure position falls within the brackets (e.g. "foo[<>]").
			if node.Lbrack < c.pos && c.pos <= node.Rbrack {
				if tv, ok := c.info.Types[node.X]; ok {
					switch t := tv.Type.Underlying().(type) {
					case *types.Map:
						typ = t.Key()
					case *types.Slice, *types.Array:
						typ = types.Typ[types.Int]
					default:
						return typeInference{}
					}
					break Nodes
				}
			}
			return typeInference{}
		case *ast.SendStmt:
			// Make sure we are on right side of arrow (e.g. "foo <- <>").
			if c.pos > node.Arrow+1 {
				if tv, ok := c.info.Types[node.Chan]; ok {
					if ch, ok := tv.Type.Underlying().(*types.Chan); ok {
						typ = ch.Elem()
						break Nodes
					}
				}
			}
			return typeInference{}
		case *ast.StarExpr:
			modifiers = append(modifiers, dereference)
		case *ast.UnaryExpr:
			switch node.Op {
			case token.AND:
				modifiers = append(modifiers, reference)
			case token.ARROW:
				modifiers = append(modifiers, chanRead)
			}
		default:
			if breaksExpectedTypeInference(node) {
				return typeInference{}
			}
		}
	}

	return typeInference{
		objType:   typ,
		modifiers: modifiers,
	}
}

// applyTypeModifiers applies the list of type modifiers to a type.
func (ti typeInference) applyTypeModifiers(typ types.Type) types.Type {
	for _, mod := range ti.modifiers {
		switch mod {
		case dereference:
			// For every "*" deref operator, remove a pointer layer from candidate type.
			typ = deref(typ)
		case reference:
			// For every "&" ref operator, add another pointer layer to candidate type.
			typ = types.NewPointer(typ)
		case chanRead:
			// For every "<-" operator, remove a layer of channelness.
			if ch, ok := typ.(*types.Chan); ok {
				typ = ch.Elem()
			}
		}
	}
	return typ
}

// findSwitchStmt returns an *ast.CaseClause's corresponding *ast.SwitchStmt or
// *ast.TypeSwitchStmt. path should start from the case clause's first ancestor.
func findSwitchStmt(path []ast.Node, pos token.Pos, c *ast.CaseClause) ast.Stmt {
	// Make sure position falls within a "case <>:" clause.
	if exprAtPos(pos, c.List) == nil {
		return nil
	}
	// A case clause is always nested within a block statement in a switch statement.
	if len(path) < 2 {
		return nil
	}
	if _, ok := path[0].(*ast.BlockStmt); !ok {
		return nil
	}
	switch s := path[1].(type) {
	case *ast.SwitchStmt:
		return s
	case *ast.TypeSwitchStmt:
		return s
	default:
		return nil
	}
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

// expectTypeName returns information about the expected type name at position.
func expectTypeName(c *completer) typeInference {
	var wantTypeName bool

Nodes:
	for i, p := range c.path {
		switch n := p.(type) {
		case *ast.FuncDecl:
			// Expect type names in a function declaration receiver, params and results.

			if r := n.Recv; r != nil && r.Pos() <= c.pos && c.pos <= r.End() {
				wantTypeName = true
				break Nodes
			}
			if t := n.Type; t != nil {
				if p := t.Params; p != nil && p.Pos() <= c.pos && c.pos <= p.End() {
					wantTypeName = true
					break Nodes
				}
				if r := t.Results; r != nil && r.Pos() <= c.pos && c.pos <= r.End() {
					wantTypeName = true
					break Nodes
				}
			}
			return typeInference{}
		case *ast.CaseClause:
			// Expect type names in type switch case clauses.
			if _, ok := findSwitchStmt(c.path[i+1:], c.pos, n).(*ast.TypeSwitchStmt); ok {
				wantTypeName = true
				break Nodes
			}
			return typeInference{}
		case *ast.TypeAssertExpr:
			// Expect type names in type assert expressions.
			if n.Lparen < c.pos && c.pos <= n.Rparen {
				wantTypeName = true
				break Nodes
			}
			return typeInference{}
		default:
			if breaksExpectedTypeInference(p) {
				return typeInference{}
			}
		}
	}

	return typeInference{
		wantTypeName: wantTypeName,
	}
}

// matchingType reports whether an object is a good completion candidate
// in the context of the expected type.
func (c *completer) matchingType(cand *candidate) bool {
	objType := cand.obj.Type()

	// Default to invoking *types.Func candidates. This is so function
	// completions in an empty statement (or other cases with no expected type)
	// are invoked by default.
	cand.expandFuncCall = isFunc(cand.obj)

	typeMatches := func(actual types.Type) bool {
		// Take into account any type modifiers on the expected type.
		actual = c.expectedType.applyTypeModifiers(actual)

		if c.expectedType.objType != nil {
			// AssignableTo covers the case where the types are equal, but also handles
			// cases like assigning a concrete type to an interface type.
			return types.AssignableTo(types.Default(actual), types.Default(c.expectedType.objType))
		}

		return false
	}

	if typeMatches(objType) {
		// If obj's type matches, we don't want to expand to an invocation of obj.
		cand.expandFuncCall = false
		return true
	}

	// Try using a function's return type as its type.
	if sig, ok := objType.Underlying().(*types.Signature); ok && sig.Results().Len() == 1 {
		if typeMatches(sig.Results().At(0).Type()) {
			// If obj's return value matches the expected type, we need to invoke obj
			// in the completion.
			cand.expandFuncCall = true
			return true
		}
	}

	return false
}
