// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/constant"
	"go/scanner"
	"go/token"
	"go/types"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
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

	Kind protocol.CompletionItemKind

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

// prefixMatcher implements case sensitive prefix matching.
type prefixMatcher string

func (pm prefixMatcher) Score(candidateLabel string) float32 {
	if strings.HasPrefix(candidateLabel, string(pm)) {
		return 1
	}
	return -1
}

// insensitivePrefixMatcher implements case insensitive prefix matching.
type insensitivePrefixMatcher string

func (ipm insensitivePrefixMatcher) Score(candidateLabel string) float32 {
	if strings.HasPrefix(strings.ToLower(candidateLabel), string(ipm)) {
		return 1
	}
	return -1
}

// completer contains the necessary information for a single completion request.
type completer struct {
	snapshot Snapshot
	pkg      Package
	qf       types.Qualifier
	opts     *completionOptions

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

	// inference contains information we've inferred about ideal
	// candidates such as the candidate's type.
	inference candidateInference

	// enclosingFunc contains information about the function enclosing
	// the position.
	enclosingFunc *funcInfo

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

// funcInfo holds info about a function object.
type funcInfo struct {
	// sig is the function declaration enclosing the position.
	sig *types.Signature

	// body is the function's body.
	body *ast.BlockStmt
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

type importInfo struct {
	importPath string
	name       string
	pkg        Package
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

func (p Selection) Suffix() string {
	return p.content[p.cursor-p.spanRange.Start:]
}

func (c *completer) deepCompletionContext() (context.Context, context.CancelFunc) {
	if c.opts.budget == 0 {
		return context.WithCancel(c.ctx)
	}
	return context.WithDeadline(c.ctx, c.startTime.Add(c.opts.budget))
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
		// Overwrite the prefix only.
		mappedRange: newMappedRange(c.snapshot.View().Session().Cache().FileSet(), c.mapper, ident.Pos(), ident.End()),
	}

	switch c.opts.matcher {
	case Fuzzy:
		c.matcher = fuzzy.NewMatcher(c.surrounding.Prefix())
	case CaseSensitive:
		c.matcher = prefixMatcher(c.surrounding.Prefix())
	default:
		c.matcher = insensitivePrefixMatcher(strings.ToLower(c.surrounding.Prefix()))
	}
}

func (c *completer) getSurrounding() *Selection {
	if c.surrounding == nil {
		c.surrounding = &Selection{
			content:     "",
			cursor:      c.pos,
			mappedRange: newMappedRange(c.snapshot.View().Session().Cache().FileSet(), c.mapper, c.pos, c.pos),
		}
	}
	return c.surrounding
}

// found adds a candidate completion. We will also search through the object's
// members for more candidates.
func (c *completer) found(cand candidate) {
	obj := cand.obj

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

	// If we know we want a type name, don't offer non-type name
	// candidates. However, do offer package names since they can
	// contain type names, and do offer any candidate without a type
	// since we aren't sure if it is a type name or not (i.e. unimported
	// candidate).
	if c.wantTypeName() && obj.Type() != nil && !isTypeName(obj) && !isPkgName(obj) {
		return
	}

	if c.matchingCandidate(&cand) {
		cand.score *= highScore
	} else if isTypeName(obj) {
		// If obj is a *types.TypeName that didn't otherwise match, check
		// if a literal object of this type makes a good candidate.

		// We only care about named types (i.e. don't want builtin types).
		if _, isNamed := obj.Type().(*types.Named); isNamed {
			c.literal(obj.Type(), cand.imp)
		}
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
			}
		}
	}

	c.deepSearch(cand)
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

	// takeAddress is true if the completion should take a pointer to obj.
	// For example, takeAddress=true yields "&foo", takeAddress=false yields "foo".
	takeAddress bool

	// addressable is true if a pointer can be taken to the candidate.
	addressable bool

	// makePointer is true if the candidate type name T should be made into *T.
	makePointer bool

	// dereference is a count of how many times to dereference the candidate obj.
	// For example, dereference=2 turns "foo" into "**foo" when formatting.
	dereference int

	// imp is the import that needs to be added to this package in order
	// for this candidate to be valid. nil if no import needed.
	imp *importInfo
}

// ErrIsDefinition is an error that informs the user they got no
// completions because they tried to complete the name of a new object
// being defined.
type ErrIsDefinition struct {
	objStr string
}

func (e ErrIsDefinition) Error() string {
	msg := "this is a definition"
	if e.objStr != "" {
		msg += " of " + e.objStr
	}
	return msg
}

// Completion returns a list of possible candidates for completion, given a
// a file and a position.
//
// The selection is computed based on the preceding identifier and can be used by
// the client to score the quality of the completion. For instance, some clients
// may tolerate imperfect matches as valid completion results, since users may make typos.
func Completion(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]CompletionItem, *Selection, error) {
	ctx, done := trace.StartSpan(ctx, "source.Completion")
	defer done()

	startTime := time.Now()

	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, nil, fmt.Errorf("getting file for Completion: %v", err)
	}
	file, src, m, _, err := pgh.Cached()
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

	// Skip completion inside any kind of literal.
	if _, ok := path[0].(*ast.BasicLit); ok {
		return nil, nil, nil
	}

	opts := snapshot.View().Options()
	c := &completer{
		pkg:                       pkg,
		snapshot:                  snapshot,
		qf:                        qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo()),
		ctx:                       ctx,
		filename:                  fh.Identity().URI.Filename(),
		file:                      file,
		path:                      path,
		pos:                       rng.Start,
		seen:                      make(map[types.Object]bool),
		enclosingFunc:             enclosingFunction(path, rng.Start, pkg.GetTypesInfo()),
		enclosingCompositeLiteral: enclosingCompositeLiteral(path, rng.Start, pkg.GetTypesInfo()),
		opts: &completionOptions{
			matcher:           opts.Matcher,
			deepCompletion:    opts.DeepCompletion,
			unimported:        opts.UnimportedCompletion,
			documentation:     opts.CompletionDocumentation,
			fullDocumentation: opts.HoverKind == FullDocumentation,
			placeholders:      opts.Placeholders,
			literal:           opts.InsertTextFormat == protocol.SnippetTextFormat,
			budget:            opts.CompletionBudget,
		},
		// default to a matcher that always matches
		matcher:        prefixMatcher(""),
		methodSetCache: make(map[methodSetKey]*types.MethodSet),
		mapper:         m,
		startTime:      startTime,
	}

	if c.opts.deepCompletion {
		// Initialize max search depth to unlimited.
		c.deepState.maxDepth = -1
	}

	// Detect our surrounding identifier.
	switch leaf := path[0].(type) {
	case *ast.Ident:
		// In the normal case, our leaf AST node is the identifier being completed.
		c.setSurrounding(leaf)
	case *ast.BadDecl:
		// You don't get *ast.Idents at the file level, so look for bad
		// decls and manually extract the surrounding token.
		pos, _, lit := c.scanToken(ctx, src)
		if pos.IsValid() {
			c.setSurrounding(&ast.Ident{Name: lit, NamePos: pos})
		}
	default:
		// Otherwise, manually extract the prefix if our containing token
		// is a keyword. This improves completion after an "accidental
		// keyword", e.g. completing to "variance" in "someFunc(var<>)".
		pos, tkn, lit := c.scanToken(ctx, src)
		if pos.IsValid() && tkn.IsKeyword() {
			c.setSurrounding(&ast.Ident{Name: lit, NamePos: pos})
		}
	}

	c.inference = expectedCandidate(c)

	defer c.sortItems()

	// If we're inside a comment return comment completions
	for _, comment := range file.Comments {
		if comment.Pos() < rng.Start && rng.Start <= comment.End() {
			c.populateCommentCompletions(comment)
			return c.items, c.getSurrounding(), nil
		}
	}

	// Struct literals are handled entirely separately.
	if c.wantStructFieldCompletions() {
		if err := c.structLiteralFieldName(); err != nil {
			return nil, nil, err
		}
		return c.items, c.getSurrounding(), nil
	}

	if lt := c.wantLabelCompletion(); lt != labelNone {
		c.labels(lt)
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
			if v, ok := obj.(*types.Var); ok && v.IsField() && v.Embedded() {
				// An anonymous field is also a reference to a type.
			} else {
				objStr := ""
				if obj != nil {
					qual := types.RelativeTo(pkg.GetTypes())
					objStr = types.ObjectString(obj, qual)
				}
				return nil, nil, ErrIsDefinition{objStr: objStr}
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
		if err := c.selector(n); err != nil {
			return nil, nil, err
		}

	// At the file scope, only keywords are allowed.
	case *ast.BadDecl, *ast.File:
		c.addKeywordCompletions()

	default:
		// fallback to lexical completions
		if err := c.lexical(); err != nil {
			return nil, nil, err
		}
	}

	return c.items, c.getSurrounding(), nil
}

// scanToken scans pgh's contents for the token containing pos.
func (c *completer) scanToken(ctx context.Context, contents []byte) (token.Pos, token.Token, string) {
	tok := c.snapshot.View().Session().Cache().FileSet().File(c.pos)

	var s scanner.Scanner
	s.Init(tok, contents, nil, 0)
	for {
		tknPos, tkn, lit := s.Scan()
		if tkn == token.EOF || tknPos >= c.pos {
			return token.NoPos, token.ILLEGAL, ""
		}

		if len(lit) > 0 && tknPos <= c.pos && c.pos <= tknPos+token.Pos(len(lit)) {
			return tknPos, tkn, lit
		}
	}
}

func (c *completer) sortItems() {
	sort.SliceStable(c.items, func(i, j int) bool {
		// Sort by score first.
		if c.items[i].Score != c.items[j].Score {
			return c.items[i].Score > c.items[j].Score
		}

		// Then sort by label so order stays consistent. This also has the
		// effect of prefering shorter candidates.
		return c.items[i].Label < c.items[j].Label
	})
}

// populateCommentCompletions yields completions for an exported
// variable immediately preceding comment.
func (c *completer) populateCommentCompletions(comment *ast.CommentGroup) {

	// Using the comment position find the line after
	fset := c.snapshot.View().Session().Cache().FileSet()
	file := fset.File(comment.Pos())
	if file == nil {
		return
	}

	line := file.Line(comment.Pos())
	nextLinePos := file.LineStart(line + 1)
	if !nextLinePos.IsValid() {
		return
	}

	// Using the next line pos, grab and parse the exported variable on that line
	for _, n := range c.file.Decls {
		if n.Pos() != nextLinePos {
			continue
		}
		switch node := n.(type) {
		case *ast.GenDecl:
			if node.Tok != token.VAR {
				return
			}
			for _, spec := range node.Specs {
				if value, ok := spec.(*ast.ValueSpec); ok {
					for _, name := range value.Names {
						if name.Name == "_" || !name.IsExported() {
							continue
						}

						exportedVar := c.pkg.GetTypesInfo().ObjectOf(name)
						c.found(candidate{obj: exportedVar, score: stdScore})
					}
				}
			}
		}
	}
}

func (c *completer) wantStructFieldCompletions() bool {
	clInfo := c.enclosingCompositeLiteral
	if clInfo == nil {
		return false
	}

	return clInfo.isStruct() && (clInfo.inKey || clInfo.maybeInFieldName)
}

func (c *completer) wantTypeName() bool {
	return c.inference.typeName.wantTypeName
}

// See https://golang.org/issue/36001. Unimported completions are expensive.
const (
	maxUnimportedPackageNames = 5
	unimportedMemberTarget    = 100
)

// selector finds completions for the specified selector expression.
func (c *completer) selector(sel *ast.SelectorExpr) error {
	// Is sel a qualified identifier?
	if id, ok := sel.X.(*ast.Ident); ok {
		if pkgName, ok := c.pkg.GetTypesInfo().Uses[id].(*types.PkgName); ok {
			c.packageMembers(pkgName.Imported(), stdScore, nil)
			return nil
		}
	}

	// Invariant: sel is a true selector.
	tv, ok := c.pkg.GetTypesInfo().Types[sel.X]
	if ok {
		return c.methodsAndFields(tv.Type, tv.Addressable(), nil)
	}

	// Try unimported packages.
	if id, ok := sel.X.(*ast.Ident); ok && c.opts.unimported {
		if err := c.unimportedMembers(id); err != nil {
			return err
		}
	}
	return nil
}

func (c *completer) unimportedMembers(id *ast.Ident) error {
	// Try loaded packages first. They're relevant, fast, and fully typed.
	known, err := c.snapshot.CachedImportPaths(c.ctx)
	if err != nil {
		return err
	}

	var paths []string
	for path, pkg := range known {
		if pkg.GetTypes().Name() != id.Name {
			continue
		}
		paths = append(paths, path)
	}

	var relevances map[string]int
	if len(paths) != 0 {
		c.snapshot.View().RunProcessEnvFunc(c.ctx, func(opts *imports.Options) error {
			relevances = imports.ScoreImportPaths(c.ctx, opts.Env, paths)
			return nil
		})
	}

	for path, pkg := range known {
		if pkg.GetTypes().Name() != id.Name {
			continue
		}
		imp := &importInfo{
			importPath: path,
			pkg:        pkg,
		}
		if imports.ImportPathToAssumedName(path) != pkg.GetTypes().Name() {
			imp.name = pkg.GetTypes().Name()
		}
		c.packageMembers(pkg.GetTypes(), stdScore+.01*float64(relevances[path]), imp)
		if len(c.items) >= unimportedMemberTarget {
			return nil
		}
	}

	ctx, cancel := c.deepCompletionContext()
	defer cancel()
	var mu sync.Mutex
	add := func(pkgExport imports.PackageExport) {
		mu.Lock()
		defer mu.Unlock()
		if _, ok := known[pkgExport.Fix.StmtInfo.ImportPath]; ok {
			return // We got this one above.
		}

		// Continue with untyped proposals.
		pkg := types.NewPackage(pkgExport.Fix.StmtInfo.ImportPath, pkgExport.Fix.IdentName)
		for _, export := range pkgExport.Exports {
			score := stdScore + 0.01*float64(pkgExport.Fix.Relevance)
			c.found(candidate{
				obj:   types.NewVar(0, pkg, export, nil),
				score: score,
				imp: &importInfo{
					importPath: pkgExport.Fix.StmtInfo.ImportPath,
					name:       pkgExport.Fix.StmtInfo.Name,
				},
			})
		}
		if len(c.items) >= unimportedMemberTarget {
			cancel()
		}
	}
	return c.snapshot.View().RunProcessEnvFunc(ctx, func(opts *imports.Options) error {
		return imports.GetPackageExports(ctx, add, id.Name, c.filename, c.pkg.GetTypes().Name(), opts)
	})
}

func (c *completer) packageMembers(pkg *types.Package, score float64, imp *importInfo) {
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		c.found(candidate{
			obj:         obj,
			score:       score,
			imp:         imp,
			addressable: isVar(obj),
		})
	}
}

func (c *completer) methodsAndFields(typ types.Type, addressable bool, imp *importInfo) error {
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
		c.found(candidate{
			obj:         mset.At(i).Obj(),
			score:       stdScore,
			imp:         imp,
			addressable: addressable || isPointer(typ),
		})
	}

	// Add fields of T.
	for _, f := range fieldSelections(typ) {
		c.found(candidate{
			obj:         f,
			score:       stdScore - 0.01,
			imp:         imp,
			addressable: addressable || isPointer(typ),
		})
	}
	return nil
}

// lexical finds completions in the lexical environment.
func (c *completer) lexical() error {
	var scopes []*types.Scope // scopes[i], where i<len(path), is the possibly nil Scope of path[i].
	for _, n := range c.path {
		// Include *FuncType scope if pos is inside the function body.
		switch node := n.(type) {
		case *ast.FuncDecl:
			if node.Body != nil && nodeContains(node.Body, c.pos) {
				n = node.Type
			}
		case *ast.FuncLit:
			if node.Body != nil && nodeContains(node.Body, c.pos) {
				n = node.Type
			}
		}
		scopes = append(scopes, c.pkg.GetTypesInfo().Scopes[n])
	}
	scopes = append(scopes, c.pkg.GetTypes().Scope(), types.Universe)

	var (
		builtinIota = types.Universe.Lookup("iota")
		builtinNil  = types.Universe.Lookup("nil")
	)

	// Track seen variables to avoid showing completions for shadowed variables.
	// This works since we look at scopes from innermost to outermost.
	seen := make(map[string]struct{})

	// Process scopes innermost first.
	for i, scope := range scopes {
		if scope == nil {
			continue
		}

	Names:
		for _, name := range scope.Names() {
			declScope, obj := scope.LookupParent(name, c.pos)
			if declScope != scope {
				continue // Name was declared in some enclosing scope, or not at all.
			}

			// If obj's type is invalid, find the AST node that defines the lexical block
			// containing the declaration of obj. Don't resolve types for packages.
			if !isPkgName(obj) && !typeIsValid(obj.Type()) {
				// Match the scope to its ast.Node. If the scope is the package scope,
				// use the *ast.File as the starting node.
				var node ast.Node
				if i < len(c.path) {
					node = c.path[i]
				} else if i == len(c.path) { // use the *ast.File for package scope
					node = c.path[i-1]
				}
				if node != nil {
					fset := c.snapshot.View().Session().Cache().FileSet()
					if resolved := resolveInvalid(fset, obj, node, c.pkg.GetTypesInfo()); resolved != nil {
						obj = resolved
					}
				}
			}

			// Don't use LHS of value spec in RHS.
			if vs := enclosingValueSpec(c.path, c.pos); vs != nil {
				for _, ident := range vs.Names {
					if obj.Pos() == ident.Pos() {
						continue Names
					}
				}
			}

			// Don't suggest "iota" outside of const decls.
			if obj == builtinIota && !c.inConstDecl() {
				continue
			}

			// Rank outer scopes lower than inner.
			score := stdScore * math.Pow(.99, float64(i))

			// Dowrank "nil" a bit so it is ranked below more interesting candidates.
			if obj == builtinNil {
				score /= 2
			}

			// If we haven't already added a candidate for an object with this name.
			if _, ok := seen[obj.Name()]; !ok {
				seen[obj.Name()] = struct{}{}
				c.found(candidate{
					obj:         obj,
					score:       score,
					addressable: isVar(obj),
				})
			}
		}
	}

	if c.inference.objType != nil {
		if named, _ := deref(c.inference.objType).(*types.Named); named != nil {
			// If we expected a named type, check the type's package for
			// completion items. This is useful when the current file hasn't
			// imported the type's package yet.

			if named.Obj() != nil && named.Obj().Pkg() != nil {
				pkg := named.Obj().Pkg()

				// Make sure the package name isn't already in use by another
				// object, and that this file doesn't import the package yet.
				if _, ok := seen[pkg.Name()]; !ok && pkg != c.pkg.GetTypes() && !alreadyImports(c.file, pkg.Path()) {
					seen[pkg.Name()] = struct{}{}
					obj := types.NewPkgName(0, nil, pkg.Name(), pkg)
					imp := &importInfo{
						importPath: pkg.Path(),
					}
					if imports.ImportPathToAssumedName(pkg.Path()) != pkg.Name() {
						imp.name = pkg.Name()
					}
					c.found(candidate{
						obj:   obj,
						score: stdScore,
						imp:   imp,
					})
				}
			}
		}
	}

	if c.opts.unimported {
		ctx, cancel := c.deepCompletionContext()
		defer cancel()
		// Suggest packages that have not been imported yet.
		prefix := ""
		if c.surrounding != nil {
			prefix = c.surrounding.Prefix()
		}
		var (
			mu               sync.Mutex
			initialItemCount = len(c.items)
		)
		add := func(pkg imports.ImportFix) {
			mu.Lock()
			defer mu.Unlock()
			if _, ok := seen[pkg.IdentName]; ok {
				return
			}

			if len(c.items)-initialItemCount >= maxUnimportedPackageNames {
				cancel()
				return
			}

			// Rank unimported packages significantly lower than other results.
			score := 0.01 * float64(pkg.Relevance)

			// Do not add the unimported packages to seen, since we can have
			// multiple packages of the same name as completion suggestions, since
			// only one will be chosen.
			obj := types.NewPkgName(0, nil, pkg.IdentName, types.NewPackage(pkg.StmtInfo.ImportPath, pkg.IdentName))
			c.found(candidate{
				obj:   obj,
				score: score,
				imp: &importInfo{
					importPath: pkg.StmtInfo.ImportPath,
					name:       pkg.StmtInfo.Name,
				},
			})
		}
		if err := c.snapshot.View().RunProcessEnvFunc(ctx, func(opts *imports.Options) error {
			return imports.GetAllCandidates(ctx, add, prefix, c.filename, c.pkg.GetTypes().Name(), opts)
		}); err != nil {
			return err
		}
	}

	if c.inference.objType != nil {
		// If we have an expected type and it is _not_ a named type, see
		// if an object literal makes a good candidate. For example, if
		// our expected type is "[]int", this will add a candidate of
		// "[]int{}".
		t := deref(c.inference.objType)
		if _, named := t.(*types.Named); !named {
			c.literal(t, nil)
		}
	}

	// Add keyword completion items appropriate in the current context.
	c.addKeywordCompletions()

	return nil
}

// alreadyImports reports whether f has an import with the specified path.
func alreadyImports(f *ast.File, path string) bool {
	for _, s := range f.Imports {
		if importPath(s) == path {
			return true
		}
	}
	return false
}

// importPath returns the unquoted import path of s,
// or "" if the path is not properly quoted.
func importPath(s *ast.ImportSpec) string {
	t, err := strconv.Unquote(s.Path.Value)
	if err != nil {
		return ""
	}
	return t
}

func nodeContains(n ast.Node, pos token.Pos) bool {
	return n != nil && n.Pos() <= pos && pos <= n.End()
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
				c.found(candidate{
					obj:   field,
					score: highScore,
				})
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
			if !(n.Lbrace < pos && pos <= n.Rbrace) {
				// Keep searching since we may yet be inside a composite literal.
				// For example "Foo{B: Ba<>{}}".
				break
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

// enclosingFunction returns the signature and body of the function
// enclosing the given position.
func enclosingFunction(path []ast.Node, pos token.Pos, info *types.Info) *funcInfo {
	for _, node := range path {
		switch t := node.(type) {
		case *ast.FuncDecl:
			if obj, ok := info.Defs[t.Name]; ok {
				return &funcInfo{
					sig:  obj.Type().(*types.Signature),
					body: t.Body,
				}
			}
		case *ast.FuncLit:
			if typ, ok := info.Types[t]; ok {
				return &funcInfo{
					sig:  typ.Type.(*types.Signature),
					body: t.Body,
				}
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
			if i := exprAtPos(c.pos, clInfo.cl.Elts); i < t.NumFields() {
				return t.Field(i).Type()
			}
		}
	}
	return nil
}

// typeModifier represents an operator that changes the expected type.
type typeModifier struct {
	mod      typeMod
	arrayLen int64
}

type typeMod int

const (
	star     typeMod = iota // pointer indirection for expressions, pointer indicator for types
	address                 // address operator ("&")
	chanRead                // channel read operator ("<-")
	slice                   // make a slice type ("[]" in "[]int")
	array                   // make an array type ("[2]" in "[2]int")
)

type objKind int

const (
	kindArray objKind = 1 << iota
	kindSlice
	kindChan
	kindMap
	kindStruct
	kindString
)

// candidateInference holds information we have inferred about a type that can be
// used at the current position.
type candidateInference struct {
	// objType is the desired type of an object used at the query position.
	objType types.Type

	// objKind is a mask of expected kinds of types such as "map", "slice", etc.
	objKind objKind

	// variadicType is the scalar variadic element type. For example,
	// when completing "append([]T{}, <>)" objType is []T and
	// variadicType is T.
	variadicType types.Type

	// modifiers are prefixes such as "*", "&" or "<-" that influence how
	// a candidate type relates to the expected type.
	modifiers []typeModifier

	// convertibleTo is a type our candidate type must be convertible to.
	convertibleTo types.Type

	// typeName holds information about the expected type name at
	// position, if any.
	typeName typeNameInference

	// assignees are the types that would receive a function call's
	// results at the position. For example:
	//
	// foo := 123
	// foo, bar := <>
	//
	// at "<>", the assignees are [int, <invalid>].
	assignees []types.Type

	// variadicAssignees is true if we could be completing an inner
	// function call that fills out an outer function call's variadic
	// params. For example:
	//
	// func foo(int, ...string) {}
	//
	// foo(<>)         // variadicAssignees=true
	// foo(bar<>)      // variadicAssignees=true
	// foo(bar, baz<>) // variadicAssignees=false
	variadicAssignees bool
}

// typeNameInference holds information about the expected type name at
// position.
type typeNameInference struct {
	// wantTypeName is true if we expect the name of a type.
	wantTypeName bool

	// modifiers are prefixes such as "*", "&" or "<-" that influence how
	// a candidate type relates to the expected type.
	modifiers []typeModifier

	// assertableFrom is a type that must be assertable to our candidate type.
	assertableFrom types.Type

	// wantComparable is true if we want a comparable type.
	wantComparable bool
}

// expectedCandidate returns information about the expected candidate
// for an expression at the query position.
func expectedCandidate(c *completer) (inf candidateInference) {
	inf.typeName = expectTypeName(c)

	if c.enclosingCompositeLiteral != nil {
		inf.objType = c.expectedCompositeLiteralType()
	}

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
				inf.objType = tv.Type
				break Nodes
			}
		case *ast.AssignStmt:
			// Only rank completions if you are on the right side of the token.
			if c.pos > node.TokPos {
				i := exprAtPos(c.pos, node.Rhs)
				if i >= len(node.Lhs) {
					i = len(node.Lhs) - 1
				}
				if tv, ok := c.pkg.GetTypesInfo().Types[node.Lhs[i]]; ok {
					inf.objType = tv.Type
				}

				// If we have a single expression on the RHS, record the LHS
				// assignees so we can favor multi-return function calls with
				// matching result values.
				if len(node.Rhs) <= 1 {
					for _, lhs := range node.Lhs {
						inf.assignees = append(inf.assignees, c.pkg.GetTypesInfo().TypeOf(lhs))
					}
				} else {
					// Otherwse, record our single assignee, even if its type is
					// not available. We use this info to downrank functions
					// with the wrong number of result values.
					inf.assignees = append(inf.assignees, c.pkg.GetTypesInfo().TypeOf(node.Lhs[i]))
				}
			}
			return inf
		case *ast.ValueSpec:
			if node.Type != nil && c.pos > node.Type.End() {
				inf.objType = c.pkg.GetTypesInfo().TypeOf(node.Type)
			}
			return inf
		case *ast.CallExpr:
			// Only consider CallExpr args if position falls between parens.
			if node.Lparen <= c.pos && c.pos <= node.Rparen {
				// For type conversions like "int64(foo)" we can only infer our
				// desired type is convertible to int64.
				if typ := typeConversion(node, c.pkg.GetTypesInfo()); typ != nil {
					inf.convertibleTo = typ
					break Nodes
				}

				if tv, ok := c.pkg.GetTypesInfo().Types[node.Fun]; ok {
					if sig, ok := tv.Type.(*types.Signature); ok {
						numParams := sig.Params().Len()
						if numParams == 0 {
							return inf
						}

						var (
							exprIdx         = exprAtPos(c.pos, node.Args)
							isLastParam     = exprIdx == numParams-1
							beyondLastParam = exprIdx >= numParams
						)

						// If we have one or zero arg expressions, we may be
						// completing to a function call that returns multiple
						// values, in turn getting passed in to the surrounding
						// call. Record the assignees so we can favor function
						// calls that return matching values.
						if len(node.Args) <= 1 {
							for i := 0; i < sig.Params().Len(); i++ {
								inf.assignees = append(inf.assignees, sig.Params().At(i).Type())
							}

							// Record that we may be completing into variadic parameters.
							inf.variadicAssignees = sig.Variadic()
						}

						if sig.Variadic() {
							variadicType := deslice(sig.Params().At(numParams - 1).Type())

							// If we are beyond the last param or we are the last
							// param w/ further expressions, we expect a single
							// variadic item.
							if beyondLastParam || isLastParam && len(node.Args) > numParams {
								inf.objType = variadicType
								break Nodes
							}

							// Otherwise if we are at the last param then we are
							// completing the variadic positition (i.e. we expect a
							// slice type []T or an individual item T).
							if isLastParam {
								inf.variadicType = variadicType
							}
						}

						// Make sure not to run past the end of expected parameters.
						if beyondLastParam {
							inf.objType = sig.Params().At(numParams - 1).Type()
						} else {
							inf.objType = sig.Params().At(exprIdx).Type()
						}
					}
				}

				if funIdent, ok := node.Fun.(*ast.Ident); ok {
					obj := c.pkg.GetTypesInfo().ObjectOf(funIdent)

					if obj != nil && obj.Parent() == types.Universe {
						// Defer call to builtinArgType so we can provide it the
						// inferred type from its parent node.
						defer func() {
							inf = c.builtinArgType(obj, node, inf)
							inf.objKind = c.builtinArgKind(obj, node)
						}()

						// The expected type of builtin arguments like append() is
						// the expected type of the builtin call itself. For
						// example:
						//
						// var foo []int = append(<>)
						//
						// To find the expected type at <> we "skip" the append()
						// node and get the expected type one level up, which is
						// []int.
						continue Nodes
					}
				}
			}
			return inf
		case *ast.ReturnStmt:
			if c.enclosingFunc != nil {
				sig := c.enclosingFunc.sig
				// Find signature result that corresponds to our return statement.
				if resultIdx := exprAtPos(c.pos, node.Results); resultIdx < len(node.Results) {
					if resultIdx < sig.Results().Len() {
						inf.objType = sig.Results().At(resultIdx).Type()
					}
				}
			}
			return inf
		case *ast.CaseClause:
			if swtch, ok := findSwitchStmt(c.path[i+1:], c.pos, node).(*ast.SwitchStmt); ok {
				if tv, ok := c.pkg.GetTypesInfo().Types[swtch.Tag]; ok {
					inf.objType = tv.Type
				}
			}
			return inf
		case *ast.SliceExpr:
			// Make sure position falls within the brackets (e.g. "foo[a:<>]").
			if node.Lbrack < c.pos && c.pos <= node.Rbrack {
				inf.objType = types.Typ[types.Int]
			}
			return inf
		case *ast.IndexExpr:
			// Make sure position falls within the brackets (e.g. "foo[<>]").
			if node.Lbrack < c.pos && c.pos <= node.Rbrack {
				if tv, ok := c.pkg.GetTypesInfo().Types[node.X]; ok {
					switch t := tv.Type.Underlying().(type) {
					case *types.Map:
						inf.objType = t.Key()
					case *types.Slice, *types.Array:
						inf.objType = types.Typ[types.Int]
					}
				}
			}
			return inf
		case *ast.SendStmt:
			// Make sure we are on right side of arrow (e.g. "foo <- <>").
			if c.pos > node.Arrow+1 {
				if tv, ok := c.pkg.GetTypesInfo().Types[node.Chan]; ok {
					if ch, ok := tv.Type.Underlying().(*types.Chan); ok {
						inf.objType = ch.Elem()
					}
				}
			}
			return inf
		case *ast.RangeStmt:
			if nodeContains(node.X, c.pos) {
				inf.objKind |= kindSlice | kindArray | kindMap | kindString
				if node.Value == nil {
					inf.objKind |= kindChan
				}
			}
			return inf
		case *ast.StarExpr:
			inf.modifiers = append(inf.modifiers, typeModifier{mod: star})
		case *ast.UnaryExpr:
			switch node.Op {
			case token.AND:
				inf.modifiers = append(inf.modifiers, typeModifier{mod: address})
			case token.ARROW:
				inf.modifiers = append(inf.modifiers, typeModifier{mod: chanRead})
			}
		default:
			if breaksExpectedTypeInference(node) {
				return inf
			}
		}
	}

	return inf
}

// applyTypeModifiers applies the list of type modifiers to a type.
// It returns nil if the modifiers could not be applied.
func (ci candidateInference) applyTypeModifiers(typ types.Type, addressable bool) types.Type {
	for _, mod := range ci.modifiers {
		switch mod.mod {
		case star:
			// For every "*" indirection operator, remove a pointer layer
			// from candidate type.
			if ptr, ok := typ.Underlying().(*types.Pointer); ok {
				typ = ptr.Elem()
			} else {
				return nil
			}
		case address:
			// For every "&" address operator, add another pointer layer to
			// candidate type, if the candidate is addressable.
			if addressable {
				typ = types.NewPointer(typ)
			} else {
				return nil
			}
		case chanRead:
			// For every "<-" operator, remove a layer of channelness.
			if ch, ok := typ.(*types.Chan); ok {
				typ = ch.Elem()
			} else {
				return nil
			}
		}
	}

	return typ
}

// applyTypeNameModifiers applies the list of type modifiers to a type name.
func (ci candidateInference) applyTypeNameModifiers(typ types.Type) types.Type {
	for _, mod := range ci.typeName.modifiers {
		switch mod.mod {
		case star:
			// For every "*" indicator, add a pointer layer to type name.
			typ = types.NewPointer(typ)
		case array:
			typ = types.NewArray(typ, mod.arrayLen)
		case slice:
			typ = types.NewSlice(typ)
		}
	}
	return typ
}

// matchesVariadic returns true if we are completing a variadic
// parameter and candType is a compatible slice type.
func (ci candidateInference) matchesVariadic(candType types.Type) bool {
	return ci.variadicType != nil && types.AssignableTo(ci.objType, candType)

}

// findSwitchStmt returns an *ast.CaseClause's corresponding *ast.SwitchStmt or
// *ast.TypeSwitchStmt. path should start from the case clause's first ancestor.
func findSwitchStmt(path []ast.Node, pos token.Pos, c *ast.CaseClause) ast.Stmt {
	// Make sure position falls within a "case <>:" clause.
	if exprAtPos(pos, c.List) >= len(c.List) {
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
	case *ast.FuncLit, *ast.CallExpr, *ast.IndexExpr, *ast.SliceExpr, *ast.CompositeLit:
		return true
	default:
		return false
	}
}

// expectTypeName returns information about the expected type name at position.
func expectTypeName(c *completer) typeNameInference {
	var (
		wantTypeName   bool
		wantComparable bool
		modifiers      []typeModifier
		assertableFrom types.Type
	)

Nodes:
	for i, p := range c.path {
		switch n := p.(type) {
		case *ast.FieldList:
			// Expect a type name if pos is in a FieldList. This applies to
			// FuncType params/results, FuncDecl receiver, StructType, and
			// InterfaceType. We don't need to worry about the field name
			// because completion bails out early if pos is in an *ast.Ident
			// that defines an object.
			wantTypeName = true
			break Nodes
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
			return typeNameInference{}
		case *ast.TypeAssertExpr:
			// Expect type names in type assert expressions.
			if n.Lparen < c.pos && c.pos <= n.Rparen {
				// The type in parens must be assertable from the expression type.
				assertableFrom = c.pkg.GetTypesInfo().TypeOf(n.X)
				wantTypeName = true
				break Nodes
			}
			return typeNameInference{}
		case *ast.StarExpr:
			modifiers = append(modifiers, typeModifier{mod: star})
		case *ast.CompositeLit:
			// We want a type name if position is in the "Type" part of a
			// composite literal (e.g. "Foo<>{}").
			if n.Type != nil && n.Type.Pos() <= c.pos && c.pos <= n.Type.End() {
				wantTypeName = true
			}
			break Nodes
		case *ast.ArrayType:
			// If we are inside the "Elt" part of an array type, we want a type name.
			if n.Elt.Pos() <= c.pos && c.pos <= n.Elt.End() {
				wantTypeName = true
				if n.Len == nil {
					// No "Len" expression means a slice type.
					modifiers = append(modifiers, typeModifier{mod: slice})
				} else {
					// Try to get the array type using the constant value of "Len".
					tv, ok := c.pkg.GetTypesInfo().Types[n.Len]
					if ok && tv.Value != nil && tv.Value.Kind() == constant.Int {
						if arrayLen, ok := constant.Int64Val(tv.Value); ok {
							modifiers = append(modifiers, typeModifier{mod: array, arrayLen: arrayLen})
						}
					}
				}

				// ArrayTypes can be nested, so keep going if our parent is an
				// ArrayType.
				if i < len(c.path)-1 {
					if _, ok := c.path[i+1].(*ast.ArrayType); ok {
						continue Nodes
					}
				}

				break Nodes
			}
		case *ast.MapType:
			wantTypeName = true
			if n.Key != nil {
				wantComparable = n.Key.Pos() <= c.pos && c.pos <= n.Key.End()
			} else {
				// If the key is empty, assume we are completing the key if
				// pos is directly after the "map[".
				wantComparable = c.pos == n.Pos()+token.Pos(len("map["))
			}
			break Nodes
		case *ast.ValueSpec:
			if n.Type != nil && n.Type.Pos() <= c.pos && c.pos <= n.Type.End() {
				wantTypeName = true
			}
			break Nodes
		default:
			if breaksExpectedTypeInference(p) {
				return typeNameInference{}
			}
		}
	}

	return typeNameInference{
		wantTypeName:   wantTypeName,
		wantComparable: wantComparable,
		modifiers:      modifiers,
		assertableFrom: assertableFrom,
	}
}

func (c *completer) fakeObj(T types.Type) *types.Var {
	return types.NewVar(token.NoPos, c.pkg.GetTypes(), "", T)
}

// anyCandType reports whether f returns true for any candidate type
// derivable from c. For example, from "foo" we might derive "&foo",
// and "foo()".
func (c *candidate) anyCandType(f func(t types.Type, addressable bool) bool) bool {
	if c.obj == nil || c.obj.Type() == nil {
		return false
	}

	objType := c.obj.Type()

	if f(objType, c.addressable) {
		return true
	}

	// If c is a func type with a single result, offer the result type.
	if sig, ok := objType.Underlying().(*types.Signature); ok {
		if sig.Results().Len() == 1 && f(sig.Results().At(0).Type(), false) {
			// Mark the candidate so we know to append "()" when formatting.
			c.expandFuncCall = true
			return true
		}
	}

	var (
		seenPtrTypes map[types.Type]bool
		ptrType      = objType
		ptrDepth     int
	)

	// Check if dereferencing c would match our type inference. We loop
	// since c could have arbitrary levels of pointerness.
	for {
		ptr, ok := ptrType.Underlying().(*types.Pointer)
		if !ok {
			break
		}

		ptrDepth++

		// Avoid pointer type cycles.
		if seenPtrTypes[ptrType] {
			break
		}

		if _, named := ptrType.(*types.Named); named {
			// Lazily allocate "seen" since it isn't used normally.
			if seenPtrTypes == nil {
				seenPtrTypes = make(map[types.Type]bool)
			}

			// Track named pointer types we have seen to detect cycles.
			seenPtrTypes[ptrType] = true
		}

		if f(ptr.Elem(), false) {
			// Mark the candidate so we know to prepend "*" when formatting.
			c.dereference = ptrDepth
			return true
		}

		ptrType = ptr.Elem()
	}

	// Check if c is addressable and a pointer to c matches our type inference.
	if c.addressable && f(types.NewPointer(objType), false) {
		// Mark the candidate so we know to prepend "&" when formatting.
		c.takeAddress = true
		return true
	}

	return false
}

// matchingCandidate reports whether cand matches our type inferences.
func (c *completer) matchingCandidate(cand *candidate) bool {
	if isTypeName(cand.obj) {
		return c.matchingTypeName(cand)
	} else if c.wantTypeName() {
		// If we want a type, a non-type object never matches.
		return false
	}

	if c.inference.candTypeMatches(cand) {
		return true
	}

	candType := cand.obj.Type()
	if candType == nil {
		return false
	}

	if sig, ok := candType.Underlying().(*types.Signature); ok {
		if c.inference.assigneesMatch(cand, sig) {
			// Invoke the candidate if its results are multi-assignable.
			cand.expandFuncCall = true
			return true
		}
	}

	// Default to invoking *types.Func candidates. This is so function
	// completions in an empty statement (or other cases with no expected type)
	// are invoked by default.
	cand.expandFuncCall = isFunc(cand.obj)

	return false
}

// candTypeMatches reports whether cand makes a good completion
// candidate given the candidate inference. cand's score may be
// mutated to downrank the candidate in certain situations.
func (ci *candidateInference) candTypeMatches(cand *candidate) bool {
	expTypes := make([]types.Type, 0, 2)
	if ci.objType != nil {
		expTypes = append(expTypes, ci.objType)
	}
	if ci.variadicType != nil {
		expTypes = append(expTypes, ci.variadicType)
	}

	return cand.anyCandType(func(candType types.Type, addressable bool) bool {
		// Take into account any type modifiers on the expected type.
		candType = ci.applyTypeModifiers(candType, addressable)
		if candType == nil {
			return false
		}

		if ci.convertibleTo != nil && types.ConvertibleTo(candType, ci.convertibleTo) {
			return true
		}

		if len(expTypes) == 0 {
			// If we have no expected type but were able to apply type
			// modifiers to our candidate type, count that as a match. This
			// handles cases like:
			//
			//   var foo chan int
			//   <-fo<>
			//
			// There is no exected type at "<>", but we were able to apply
			// the "<-" type modifier to "foo", so it matches.
			if len(ci.modifiers) > 0 {
				return true
			}

			// If we have no expected type, fall back to checking the
			// expected "kind" of object, if available.
			return ci.kindMatches(candType)
		}

		for _, expType := range expTypes {
			matches, untyped := ci.typeMatches(expType, candType)
			if !matches {
				continue
			}

			// Lower candidate score for untyped conversions. This avoids
			// ranking untyped constants above candidates with an exact type
			// match. Don't lower score of builtin constants, e.g. "true".
			if untyped && !types.Identical(candType, expType) && cand.obj.Parent() != types.Universe {
				cand.score /= 2
			}

			return true
		}

		return false
	})
}

// typeMatches reports whether an object of candType makes a good
// completion candidate given the expected type expType. It also
// returns a second bool which is true if both types are basic types
// of the same kind, and at least one is untyped.
func (ci *candidateInference) typeMatches(expType, candType types.Type) (bool, bool) {
	// Handle untyped values specially since AssignableTo gives false negatives
	// for them (see https://golang.org/issue/32146).
	if candBasic, ok := candType.Underlying().(*types.Basic); ok {
		if wantBasic, ok := expType.Underlying().(*types.Basic); ok {
			// Make sure at least one of them is untyped.
			if isUntyped(candType) || isUntyped(expType) {
				// Check that their constant kind (bool|int|float|complex|string) matches.
				// This doesn't take into account the constant value, so there will be some
				// false positives due to integer sign and overflow.
				if candBasic.Info()&types.IsConstType == wantBasic.Info()&types.IsConstType {
					return true, true
				}
			}
		}
	}

	// AssignableTo covers the case where the types are equal, but also handles
	// cases like assigning a concrete type to an interface type.
	return types.AssignableTo(candType, expType), false
}

// kindMatches reports whether candType's kind matches our expected
// kind (e.g. slice, map, etc.).
func (ci *candidateInference) kindMatches(candType types.Type) bool {
	return ci.objKind&candKind(candType) > 0
}

// assigneesMatch reports whether an invocation of sig matches the
// number and type of any assignees.
func (ci *candidateInference) assigneesMatch(cand *candidate, sig *types.Signature) bool {
	if len(ci.assignees) == 0 {
		return false
	}

	// Uniresult functions are always usable and are handled by the
	// normal, non-assignees type matching logic.
	if sig.Results().Len() == 1 {
		return false
	}

	var numberOfResultsCouldMatch bool
	if ci.variadicAssignees {
		numberOfResultsCouldMatch = sig.Results().Len() >= len(ci.assignees)-1
	} else {
		numberOfResultsCouldMatch = sig.Results().Len() == len(ci.assignees)
	}

	// If our signature doesn't return the right number of values, it's
	// not a match, so downrank it. For example:
	//
	//  var foo func() (int, int)
	//  a, b, c := <> // downrank "foo()" since it only returns two values
	if !numberOfResultsCouldMatch {
		cand.score /= 2
		return false
	}

	// If at least one assignee has a valid type, and all valid
	// assignees match the corresponding sig result value, the signature
	// is a match.
	allMatch := false
	for i := 0; i < sig.Results().Len(); i++ {
		var assignee types.Type

		// If we are completing into variadic parameters, deslice the
		// expected variadic type.
		if ci.variadicAssignees && i >= len(ci.assignees)-1 {
			assignee = ci.assignees[len(ci.assignees)-1]
			if elem := deslice(assignee); elem != nil {
				assignee = elem
			}
		} else {
			assignee = ci.assignees[i]
		}

		if assignee == nil {
			continue
		}

		allMatch, _ = ci.typeMatches(assignee, sig.Results().At(i).Type())
		if !allMatch {
			break
		}
	}
	return allMatch
}

func (c *completer) matchingTypeName(cand *candidate) bool {
	if !c.wantTypeName() {
		return false
	}

	typeMatches := func(candType types.Type) bool {
		// Take into account any type name modifier prefixes.
		candType = c.inference.applyTypeNameModifiers(candType)

		if from := c.inference.typeName.assertableFrom; from != nil {
			// Don't suggest the starting type in type assertions. For example,
			// if "foo" is an io.Writer, don't suggest "foo.(io.Writer)".
			if types.Identical(from, candType) {
				return false
			}

			if intf, ok := from.Underlying().(*types.Interface); ok {
				if !types.AssertableTo(intf, candType) {
					return false
				}
			}
		}

		if c.inference.typeName.wantComparable && !types.Comparable(candType) {
			return false
		}

		// We can expect a type name and have an expected type in cases like:
		//
		//   var foo []int
		//   foo = []i<>
		//
		// Where our expected type is "[]int", and we expect a type name.
		if c.inference.objType != nil {
			return types.AssignableTo(candType, c.inference.objType)
		}

		// Default to saying any type name is a match.
		return true
	}

	if typeMatches(cand.obj.Type()) {
		return true
	}

	if typeMatches(types.NewPointer(cand.obj.Type())) {
		cand.makePointer = true
		return true
	}

	return false
}

// candKind returns the objKind of candType, if any.
func candKind(candType types.Type) objKind {
	switch t := candType.Underlying().(type) {
	case *types.Array:
		return kindArray
	case *types.Slice:
		return kindSlice
	case *types.Chan:
		return kindChan
	case *types.Map:
		return kindMap
	case *types.Pointer:
		// Some builtins handle array pointers as arrays, so just report a pointer
		// to an array as an array.
		if _, isArray := t.Elem().Underlying().(*types.Array); isArray {
			return kindArray
		}
	case *types.Basic:
		if t.Info()&types.IsString > 0 {
			return kindString
		}
	}

	return 0
}
