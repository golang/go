// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"strings"
	"time"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
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

	// An optional array of additional TextEdits that are applied when
	// selecting this completion.
	//
	// Additional text edits should be used to change text unrelated to the current cursor position
	// (for example adding an import statement at the top of the file if the completion item will
	// insert an unqualified type).
	AdditionalTextEdits []protocol.TextEdit

	// Depth is how many levels were searched to find this completion.
	// For example when completing "foo<>", "fooBar" is depth 0, and
	// "fooBar.Baz" is depth 1.
	Depth int

	// Score is the internal relevance score.
	// A higher score indicates that this completion item is more relevant.
	Score float64

	// snippet is the LSP snippet for the completion item. The LSP
	// specification contains details about LSP snippets. For example, a
	// snippet for a function with the following signature:
	//
	//     func foo(a, b, c int)
	//
	// would be:
	//
	//     foo(${1:a int}, ${2: b int}, ${3: c int})
	//
	// If Placeholders is false in the CompletionOptions, the above
	// snippet would instead be:
	//
	//     foo(${1:})
	snippet *snippet.Builder

	// Documentation is the documentation for the completion item.
	Documentation string
}

// Snippet is a convenience returns the snippet if available, otherwise
// the InsertText.
// used for an item, depending on if the callee wants placeholders or not.
func (i *CompletionItem) Snippet() string {
	if i.snippet != nil {
		return i.snippet.String()
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

// matcher matches a candidate's label against the user input. The
// returned score reflects the quality of the match. A score of zero
// indicates no match, and a score of one means a perfect match.
type matcher interface {
	Score(candidateLabel string) (score float32)
}

// prefixMatcher implements case insensitive prefix matching.
type prefixMatcher string

func (pm prefixMatcher) Score(candidateLabel string) float32 {
	if strings.HasPrefix(strings.ToLower(candidateLabel), string(pm)) {
		return 1
	}
	return -1
}

// completer contains the necessary information for a single completion request.
type completer struct {
	pkg Package

	qf   types.Qualifier
	opts CompletionOptions

	// view is the View associated with this completion request.
	view View

	// ctx is the context associated with this completion request.
	ctx context.Context

	// filename is the name of the file associated with this completion request.
	filename string

	// file is the AST of the file associated with this completion request.
	file *ast.File

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

	// expectedType contains information about the type we expect the completion
	// candidate to be. It will be the zero value if no information is available.
	expectedType typeInference

	// enclosingFunction is the function declaration enclosing the position.
	enclosingFunction *types.Signature

	// enclosingCompositeLiteral contains information about the composite literal
	// enclosing the position.
	enclosingCompositeLiteral *compLitInfo

	// deepState contains the current state of our deep completion search.
	deepState deepCompletionState

	// matcher matches the candidates against the surrounding prefix.
	matcher matcher

	// methodSetCache caches the types.NewMethodSet call, which is relatively
	// expensive and can be called many times for the same type while searching
	// for deep completions.
	methodSetCache map[methodSetKey]*types.MethodSet

	// mapper converts the positions in the file from which the completion originated.
	mapper *protocol.ColumnMapper

	// startTime is when we started processing this completion request. It does
	// not include any time the request spent in the queue.
	startTime time.Time
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

type methodSetKey struct {
	typ         types.Type
	addressable bool
}

// A Selection represents the cursor position and surrounding identifier.
type Selection struct {
	content string
	cursor  token.Pos
	mappedRange
}

func (p Selection) Prefix() string {
	return p.content[:p.cursor-p.spanRange.Start]
}

func (c *completer) setSurrounding(ident *ast.Ident) {
	if c.surrounding != nil {
		return
	}
	if !(ident.Pos() <= c.pos && c.pos <= ident.End()) {
		return
	}
	c.surrounding = &Selection{
		content: ident.Name,
		cursor:  c.pos,
		mappedRange: mappedRange{
			spanRange: span.NewRange(c.view.Session().Cache().FileSet(), ident.Pos(), ident.End()),
			m:         c.mapper,
		},
	}

	if c.opts.FuzzyMatching {
		c.matcher = fuzzy.NewMatcher(c.surrounding.Prefix(), fuzzy.Symbol)
	} else {
		c.matcher = prefixMatcher(strings.ToLower(c.surrounding.Prefix()))
	}
}

func (c *completer) getSurrounding() *Selection {
	if c.surrounding == nil {
		c.surrounding = &Selection{
			content: "",
			cursor:  c.pos,
			mappedRange: mappedRange{
				spanRange: span.NewRange(c.view.Session().Cache().FileSet(), c.pos, c.pos),
				m:         c.mapper,
			},
		}
	}
	return c.surrounding
}

// found adds a candidate completion. We will also search through the object's
// members for more candidates.
func (c *completer) found(obj types.Object, score float64, imp *imports.ImportInfo) {
	if obj.Pkg() != nil && obj.Pkg() != c.pkg.GetTypes() && !obj.Exported() {
		// obj is not accessible because it lives in another package and is not
		// exported. Don't treat it as a completion candidate.
		return
	}

	if c.inDeepCompletion() {
		// When searching deep, just make sure we don't have a cycle in our chain.
		// We don't dedupe by object because we want to allow both "foo.Baz" and
		// "bar.Baz" even though "Baz" is represented the same types.Object in both.
		for _, seenObj := range c.deepState.chain {
			if seenObj == obj {
				return
			}
		}
	} else {
		// At the top level, dedupe by object.
		if c.seen[obj] {
			return
		}
		c.seen[obj] = true
	}

	// If we are running out of budgeted time we must limit our search for deep
	// completion candidates.
	if c.shouldPrune() {
		return
	}

	cand := candidate{
		obj:   obj,
		score: score,
		imp:   imp,
	}

	if c.matchingCandidate(&cand) {
		cand.score *= highScore
	} else if isTypeName(obj) {
		// If obj is a *types.TypeName that didn't otherwise match, check
		// if a literal object of this type makes a good candidate.
		c.literal(obj.Type())
	}

	// Favor shallow matches by lowering weight according to depth.
	cand.score -= cand.score * float64(len(c.deepState.chain)) / 10
	if cand.score < 0 {
		cand.score = 0
	}

	cand.name = c.deepState.chainString(obj.Name())
	matchScore := c.matcher.Score(cand.name)
	if matchScore > 0 {
		cand.score *= float64(matchScore)

		// Avoid calling c.item() for deep candidates that wouldn't be in the top
		// MaxDeepCompletions anyway.
		if !c.inDeepCompletion() || c.deepState.isHighScore(cand.score) {
			if item, err := c.item(cand); err == nil {
				c.items = append(c.items, item)
			} else {
				log.Error(c.ctx, "error generating completion item", err)
			}
		}
	}

	c.deepSearch(obj)
}

// candidate represents a completion candidate.
type candidate struct {
	// obj is the types.Object to complete to.
	obj types.Object

	// score is used to rank candidates.
	score float64

	// name is the deep object name path, e.g. "foo.bar"
	name string

	// expandFuncCall is true if obj should be invoked in the completion.
	// For example, expandFuncCall=true yields "foo()", expandFuncCall=false yields "foo".
	expandFuncCall bool

	// imp is the import that needs to be added to this package in order
	// for this candidate to be valid. nil if no import needed.
	imp *imports.ImportInfo
}

// Completion returns a list of possible candidates for completion, given a
// a file and a position.
//
// The selection is computed based on the preceding identifier and can be used by
// the client to score the quality of the completion. For instance, some clients
// may tolerate imperfect matches as valid completion results, since users may make typos.
func Completion(ctx context.Context, view View, f GoFile, pos protocol.Position, opts CompletionOptions) ([]CompletionItem, *Selection, error) {
	ctx, done := trace.StartSpan(ctx, "source.Completion")
	defer done()

	startTime := time.Now()

	cphs, err := f.CheckPackageHandles(ctx)
	if err != nil {
		return nil, nil, err
	}
	cph := NarrowestCheckPackageHandle(cphs)
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, nil, err
	}
	ph, err := pkg.File(f.URI())
	if err != nil {
		return nil, nil, err
	}
	file, m, _, err := ph.Cached(ctx)
	if err != nil {
		return nil, nil, err
	}
	spn, err := m.PointSpan(pos)
	if err != nil {
		return nil, nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, nil, err
	}
	// Completion is based on what precedes the cursor.
	// Find the path to the position before pos.
	path, _ := astutil.PathEnclosingInterval(file, rng.Start-1, rng.Start-1)
	if path == nil {
		return nil, nil, errors.Errorf("cannot find node enclosing position")
	}
	// Skip completion inside comments.
	for _, g := range file.Comments {
		if g.Pos() <= rng.Start && rng.Start <= g.End() {
			return nil, nil, nil
		}
	}
	// Skip completion inside any kind of literal.
	if _, ok := path[0].(*ast.BasicLit); ok {
		return nil, nil, nil
	}

	clInfo := enclosingCompositeLiteral(path, rng.Start, pkg.GetTypesInfo())
	c := &completer{
		pkg:                       pkg,
		qf:                        qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo()),
		view:                      view,
		ctx:                       ctx,
		filename:                  f.URI().Filename(),
		file:                      file,
		path:                      path,
		pos:                       rng.Start,
		seen:                      make(map[types.Object]bool),
		enclosingFunction:         enclosingFunction(path, rng.Start, pkg.GetTypesInfo()),
		enclosingCompositeLiteral: clInfo,
		opts:                      opts,
		// default to a matcher that always matches
		matcher:        prefixMatcher(""),
		methodSetCache: make(map[methodSetKey]*types.MethodSet),
		mapper:         m,
		startTime:      startTime,
	}

	if opts.Deep {
		// Initialize max search depth to unlimited.
		c.deepState.maxDepth = -1
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
		return c.items, c.getSurrounding(), nil
	}

	switch n := path[0].(type) {
	case *ast.Ident:
		// Is this the Sel part of a selector?
		if sel, ok := path[1].(*ast.SelectorExpr); ok && sel.Sel == n {
			if err := c.selector(sel); err != nil {
				return nil, nil, err
			}
			return c.items, c.getSurrounding(), nil
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
				return nil, nil, errors.Errorf("this is a definition%s", of)
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

	return c.items, c.getSurrounding(), nil
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
		if pkgname, ok := c.pkg.GetTypesInfo().Uses[id].(*types.PkgName); ok {
			c.packageMembers(pkgname)
			return nil
		}
	}

	// Invariant: sel is a true selector.
	tv, ok := c.pkg.GetTypesInfo().Types[sel.X]
	if !ok {
		return errors.Errorf("cannot resolve %s", sel.X)
	}

	return c.methodsAndFields(tv.Type, tv.Addressable())
}

func (c *completer) packageMembers(pkg *types.PkgName) {
	scope := pkg.Imported().Scope()
	for _, name := range scope.Names() {
		c.found(scope.Lookup(name), stdScore, nil)
	}
}

func (c *completer) methodsAndFields(typ types.Type, addressable bool) error {
	mset := c.methodSetCache[methodSetKey{typ, addressable}]
	if mset == nil {
		if addressable && !types.IsInterface(typ) && !isPointer(typ) {
			// Add methods of *T, which includes methods with receiver T.
			mset = types.NewMethodSet(types.NewPointer(typ))
		} else {
			// Add methods of T.
			mset = types.NewMethodSet(typ)
		}
		c.methodSetCache[methodSetKey{typ, addressable}] = mset
	}

	for i := 0; i < mset.Len(); i++ {
		c.found(mset.At(i).Obj(), stdScore, nil)
	}

	// Add fields of T.
	for _, f := range fieldSelections(typ) {
		c.found(f, stdScore, nil)
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
		scopes = append(scopes, c.pkg.GetTypesInfo().Scopes[n])
	}
	scopes = append(scopes, c.pkg.GetTypes().Scope(), types.Universe)

	builtinIota := types.Universe.Lookup("iota")

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
					if resolved := resolveInvalid(obj, node, c.pkg.GetTypesInfo()); resolved != nil {
						obj = resolved
					}
				}
			}

			// Don't suggest "iota" outside of const decls.
			if obj == builtinIota && !c.inConstDecl() {
				continue
			}

			// If we haven't already added a candidate for an object with this name.
			if _, ok := seen[obj.Name()]; !ok {
				seen[obj.Name()] = struct{}{}
				c.found(obj, stdScore, nil)
			}
		}
	}

	if c.opts.Unimported {
		// Suggest packages that have not been imported yet.
		pkgs, err := CandidateImports(c.ctx, c.view, c.filename)
		if err != nil {
			return err
		}
		score := stdScore
		// Rank unimported packages significantly lower than other results.
		score *= 0.07
		for _, pkg := range pkgs {
			if _, ok := seen[pkg.IdentName]; !ok {
				// Do not add the unimported packages to seen, since we can have
				// multiple packages of the same name as completion suggestions, since
				// only one will be chosen.
				obj := types.NewPkgName(0, nil, pkg.IdentName, types.NewPackage(pkg.StmtInfo.ImportPath, pkg.IdentName))
				c.found(obj, score, &pkg.StmtInfo)
			}
		}
	}

	if c.expectedType.objType != nil {
		// If we have an expected type and it is _not_ a named type, see
		// if an object literal makes a good candidate. Named types are
		// handled during the normal lexical object search. For example,
		// if our expected type is "[]int", this will add a candidate of
		// "[]int{}".
		if _, named := deref(c.expectedType.objType).(*types.Named); !named {
			c.literal(c.expectedType.objType)
		}
	}

	return nil
}

func (c *completer) inConstDecl() bool {
	for _, n := range c.path {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.CONST {
			return true
		}
	}
	return false
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
				if used, ok := c.pkg.GetTypesInfo().Uses[key]; ok {
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
				c.found(field, highScore, nil)
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
				clType: deref(tv.Type).Underlying(),
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
	star      typeModifier = iota // dereference operator for expressions, pointer indicator for types
	reference                     // reference ("&") operator
	chanRead                      // channel read ("<-") operator
)

// typeInference holds information we have inferred about a type that can be
// used at the current position.
type typeInference struct {
	// objType is the desired type of an object used at the query position.
	objType types.Type

	// variadic is true if objType is a slice type from an initial
	// variadic param.
	variadic bool

	// wantTypeName is true if we expect the name of a type.
	wantTypeName bool

	// modifiers are prefixes such as "*", "&" or "<-" that influence how
	// a candidate type relates to the expected type.
	modifiers []typeModifier

	// assertableFrom is a type that must be assertable to our candidate type.
	assertableFrom types.Type

	// convertibleTo is a type our candidate type must be convertible to.
	convertibleTo types.Type
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
		modifiers     []typeModifier
		variadic      bool
		typ           types.Type
		convertibleTo types.Type
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
			if tv, ok := c.pkg.GetTypesInfo().Types[e]; ok {
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
				if tv, ok := c.pkg.GetTypesInfo().Types[node.Lhs[i]]; ok {
					typ = tv.Type
					break Nodes
				}
			}
			return typeInference{}
		case *ast.CallExpr:
			// Only consider CallExpr args if position falls between parens.
			if node.Lparen <= c.pos && c.pos <= node.Rparen {
				// For type conversions like "int64(foo)" we can only infer our
				// desired type is convertible to int64.
				if typ := typeConversion(node, c.pkg.GetTypesInfo()); typ != nil {
					convertibleTo = typ
					break Nodes
				}

				if tv, ok := c.pkg.GetTypesInfo().Types[node.Fun]; ok {
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
						variadic = sig.Variadic() && i == sig.Params().Len()-1
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
				if tv, ok := c.pkg.GetTypesInfo().Types[swtch.Tag]; ok {
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
				if tv, ok := c.pkg.GetTypesInfo().Types[node.X]; ok {
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
				if tv, ok := c.pkg.GetTypesInfo().Types[node.Chan]; ok {
					if ch, ok := tv.Type.Underlying().(*types.Chan); ok {
						typ = ch.Elem()
						break Nodes
					}
				}
			}
			return typeInference{}
		case *ast.StarExpr:
			modifiers = append(modifiers, star)
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
		variadic:      variadic,
		objType:       typ,
		modifiers:     modifiers,
		convertibleTo: convertibleTo,
	}
}

// applyTypeModifiers applies the list of type modifiers to a type.
func (ti typeInference) applyTypeModifiers(typ types.Type) types.Type {
	for _, mod := range ti.modifiers {
		switch mod {
		case star:
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

// applyTypeNameModifiers applies the list of type modifiers to a type name.
func (ti typeInference) applyTypeNameModifiers(typ types.Type) types.Type {
	for _, mod := range ti.modifiers {
		switch mod {
		case star:
			// For every "*" indicator, add a pointer layer to type name.
			typ = types.NewPointer(typ)
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
	var (
		wantTypeName   bool
		modifiers      []typeModifier
		assertableFrom types.Type
	)

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
			if swtch, ok := findSwitchStmt(c.path[i+1:], c.pos, n).(*ast.TypeSwitchStmt); ok {
				// The case clause types must be assertable from the type switch parameter.
				ast.Inspect(swtch.Assign, func(n ast.Node) bool {
					if ta, ok := n.(*ast.TypeAssertExpr); ok {
						assertableFrom = c.pkg.GetTypesInfo().TypeOf(ta.X)
						return false
					}
					return true
				})
				wantTypeName = true
				break Nodes
			}
			return typeInference{}
		case *ast.TypeAssertExpr:
			// Expect type names in type assert expressions.
			if n.Lparen < c.pos && c.pos <= n.Rparen {
				// The type in parens must be assertable from the expression type.
				assertableFrom = c.pkg.GetTypesInfo().TypeOf(n.X)
				wantTypeName = true
				break Nodes
			}
			return typeInference{}
		case *ast.StarExpr:
			modifiers = append(modifiers, star)
		default:
			if breaksExpectedTypeInference(p) {
				return typeInference{}
			}
		}
	}

	return typeInference{
		wantTypeName:   wantTypeName,
		modifiers:      modifiers,
		assertableFrom: assertableFrom,
	}
}

// matchingType reports whether a type matches the expected type.
func (c *completer) matchingType(T types.Type) bool {
	fakeObj := types.NewVar(token.NoPos, c.pkg.GetTypes(), "", T)
	return c.matchingCandidate(&candidate{obj: fakeObj})
}

// matchingCandidate reports whether a candidate matches our type
// inferences.
func (c *completer) matchingCandidate(cand *candidate) bool {
	if isTypeName(cand.obj) {
		return c.matchingTypeName(cand)
	}

	objType := cand.obj.Type()

	// Default to invoking *types.Func candidates. This is so function
	// completions in an empty statement (or other cases with no expected type)
	// are invoked by default.
	cand.expandFuncCall = isFunc(cand.obj)

	typeMatches := func(candType types.Type) bool {
		// Take into account any type modifiers on the expected type.
		candType = c.expectedType.applyTypeModifiers(candType)

		if c.expectedType.objType != nil {
			wantType := types.Default(c.expectedType.objType)

			// Handle untyped values specially since AssignableTo gives false negatives
			// for them (see https://golang.org/issue/32146).
			if candBasic, ok := candType.(*types.Basic); ok && candBasic.Info()&types.IsUntyped > 0 {
				if wantBasic, ok := wantType.Underlying().(*types.Basic); ok {
					// Check that their constant kind (bool|int|float|complex|string) matches.
					// This doesn't take into account the constant value, so there will be some
					// false positives due to integer sign and overflow.
					return candBasic.Info()&types.IsConstType == wantBasic.Info()&types.IsConstType
				}
				return false
			}

			// AssignableTo covers the case where the types are equal, but also handles
			// cases like assigning a concrete type to an interface type.
			return types.AssignableTo(candType, wantType)
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

	if c.expectedType.convertibleTo != nil {
		return types.ConvertibleTo(objType, c.expectedType.convertibleTo)
	}

	return false
}

func (c *completer) matchingTypeName(cand *candidate) bool {
	if !c.wantTypeName() {
		return false
	}

	// Take into account any type name modifier prefixes.
	actual := c.expectedType.applyTypeNameModifiers(cand.obj.Type())

	if c.expectedType.assertableFrom != nil {
		// Don't suggest the starting type in type assertions. For example,
		// if "foo" is an io.Writer, don't suggest "foo.(io.Writer)".
		if types.Identical(c.expectedType.assertableFrom, actual) {
			return false
		}

		if intf, ok := c.expectedType.assertableFrom.Underlying().(*types.Interface); ok {
			if !types.AssertableTo(intf, actual) {
				return false
			}
		}
	}

	// Default to saying any type name is a match.
	return true
}
