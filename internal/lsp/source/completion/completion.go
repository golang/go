// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package completion provides core functionality for code completion in Go
// editors and tools.
package completion

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
	"unicode"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/lsp/source"
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

	// obj is the object from which this candidate was derived, if any.
	// obj is for internal use only.
	obj types.Object
}

// completionOptions holds completion specific configuration.
type completionOptions struct {
	unimported        bool
	documentation     bool
	fullDocumentation bool
	placeholders      bool
	literal           bool
	matcher           source.Matcher
	budget            time.Duration
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
	snapshot source.Snapshot
	pkg      source.Package
	qf       types.Qualifier
	opts     *completionOptions

	// completionContext contains information about the trigger for this
	// completion request.
	completionContext completionContext

	// fh is a handle to the file associated with this completion request.
	fh source.FileHandle

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

	// completionCallbacks is a list of callbacks to collect completions that
	// require expensive operations. This includes operations where we search
	// through the entire module cache.
	completionCallbacks []func(opts *imports.Options) error

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
	pkg        source.Package
}

type methodSetKey struct {
	typ         types.Type
	addressable bool
}

type completionContext struct {
	// triggerCharacter is the character used to trigger completion at current
	// position, if any.
	triggerCharacter string

	// triggerKind is information about how a completion was triggered.
	triggerKind protocol.CompletionTriggerKind

	// commentCompletion is true if we are completing a comment.
	commentCompletion bool

	// packageCompletion is true if we are completing a package name.
	packageCompletion bool
}

// A Selection represents the cursor position and surrounding identifier.
type Selection struct {
	content string
	cursor  token.Pos
	source.MappedRange
}

func (p Selection) Content() string {
	return p.content
}

func (p Selection) Start() token.Pos {
	return p.MappedRange.SpanRange().Start
}

func (p Selection) End() token.Pos {
	return p.MappedRange.SpanRange().End
}

func (p Selection) Prefix() string {
	return p.content[:p.cursor-p.SpanRange().Start]
}

func (p Selection) Suffix() string {
	return p.content[p.cursor-p.SpanRange().Start:]
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
		MappedRange: source.NewMappedRange(c.snapshot.FileSet(), c.mapper, ident.Pos(), ident.End()),
	}

	c.setMatcherFromPrefix(c.surrounding.Prefix())
}

func (c *completer) setMatcherFromPrefix(prefix string) {
	switch c.opts.matcher {
	case source.Fuzzy:
		c.matcher = fuzzy.NewMatcher(prefix)
	case source.CaseSensitive:
		c.matcher = prefixMatcher(prefix)
	default:
		c.matcher = insensitivePrefixMatcher(strings.ToLower(prefix))
	}
}

func (c *completer) getSurrounding() *Selection {
	if c.surrounding == nil {
		c.surrounding = &Selection{
			content:     "",
			cursor:      c.pos,
			MappedRange: source.NewMappedRange(c.snapshot.FileSet(), c.mapper, c.pos, c.pos),
		}
	}
	return c.surrounding
}

// candidate represents a completion candidate.
type candidate struct {
	// obj is the types.Object to complete to.
	obj types.Object

	// score is used to rank candidates.
	score float64

	// name is the deep object name path, e.g. "foo.bar"
	name string

	// detail is additional information about this item. If not specified,
	// defaults to type string for the object.
	detail string

	// path holds the path from the search root (excluding the candidate
	// itself) for a deep candidate.
	path []types.Object

	// names tracks the names of objects from search root (excluding the
	// candidate itself) for a deep candidate. This also includes
	// expanded calls for function invocations.
	names []string

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

	// variadic is true if this candidate fills a variadic param and
	// needs "..." appended.
	variadic bool

	// convertTo is a type that this candidate should be cast to. For
	// example, if convertTo is float64, "foo" should be formatted as
	// "float64(foo)".
	convertTo types.Type

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
func Completion(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, protoPos protocol.Position, protoContext protocol.CompletionContext) ([]CompletionItem, *Selection, error) {
	ctx, done := event.Start(ctx, "completion.Completion")
	defer done()

	startTime := time.Now()

	pkg, pgf, err := source.GetParsedFile(ctx, snapshot, fh, source.NarrowestPackage)
	if err != nil || pgf.File.Package == token.NoPos {
		// If we can't parse this file or find position for the package
		// keyword, it may be missing a package declaration. Try offering
		// suggestions for the package declaration.
		// Note that this would be the case even if the keyword 'package' is
		// present but no package name exists.
		items, surrounding, innerErr := packageClauseCompletions(ctx, snapshot, fh, protoPos)
		if innerErr != nil {
			// return the error for GetParsedFile since it's more relevant in this situation.
			return nil, nil, errors.Errorf("getting file for Completion: %w (package completions: %v)", err, innerErr)
		}
		return items, surrounding, nil
	}
	spn, err := pgf.Mapper.PointSpan(protoPos)
	if err != nil {
		return nil, nil, err
	}
	rng, err := spn.Range(pgf.Mapper.Converter)
	if err != nil {
		return nil, nil, err
	}
	// Completion is based on what precedes the cursor.
	// Find the path to the position before pos.
	path, _ := astutil.PathEnclosingInterval(pgf.File, rng.Start-1, rng.Start-1)
	if path == nil {
		return nil, nil, errors.Errorf("cannot find node enclosing position")
	}

	pos := rng.Start

	// Check if completion at this position is valid. If not, return early.
	switch n := path[0].(type) {
	case *ast.BasicLit:
		// Skip completion inside literals except for ImportSpec
		if len(path) > 1 {
			if _, ok := path[1].(*ast.ImportSpec); ok {
				break
			}
		}
		return nil, nil, nil
	case *ast.CallExpr:
		if n.Ellipsis.IsValid() && pos > n.Ellipsis && pos <= n.Ellipsis+token.Pos(len("...")) {
			// Don't offer completions inside or directly after "...". For
			// example, don't offer completions at "<>" in "foo(bar...<>").
			return nil, nil, nil
		}
	case *ast.Ident:
		// reject defining identifiers
		if obj, ok := pkg.GetTypesInfo().Defs[n]; ok {
			if v, ok := obj.(*types.Var); ok && v.IsField() && v.Embedded() {
				// An anonymous field is also a reference to a type.
			} else if pgf.File.Name == n {
				// Don't skip completions if Ident is for package name.
				break
			} else {
				objStr := ""
				if obj != nil {
					qual := types.RelativeTo(pkg.GetTypes())
					objStr = types.ObjectString(obj, qual)
				}
				return nil, nil, ErrIsDefinition{objStr: objStr}
			}
		}
	}

	opts := snapshot.View().Options()
	c := &completer{
		pkg:      pkg,
		snapshot: snapshot,
		qf:       source.Qualifier(pgf.File, pkg.GetTypes(), pkg.GetTypesInfo()),
		completionContext: completionContext{
			triggerCharacter: protoContext.TriggerCharacter,
			triggerKind:      protoContext.TriggerKind,
		},
		fh:                        fh,
		filename:                  fh.URI().Filename(),
		file:                      pgf.File,
		path:                      path,
		pos:                       pos,
		seen:                      make(map[types.Object]bool),
		enclosingFunc:             enclosingFunction(path, pkg.GetTypesInfo()),
		enclosingCompositeLiteral: enclosingCompositeLiteral(path, rng.Start, pkg.GetTypesInfo()),
		deepState: deepCompletionState{
			enabled: opts.DeepCompletion,
		},
		opts: &completionOptions{
			matcher:           opts.Matcher,
			unimported:        opts.CompleteUnimported,
			documentation:     opts.CompletionDocumentation && opts.HoverKind != source.NoDocumentation,
			fullDocumentation: opts.HoverKind == source.FullDocumentation,
			placeholders:      opts.UsePlaceholders,
			literal:           opts.LiteralCompletions && opts.InsertTextFormat == protocol.SnippetTextFormat,
			budget:            opts.CompletionBudget,
		},
		// default to a matcher that always matches
		matcher:        prefixMatcher(""),
		methodSetCache: make(map[methodSetKey]*types.MethodSet),
		mapper:         pgf.Mapper,
		startTime:      startTime,
	}

	var cancel context.CancelFunc
	if c.opts.budget == 0 {
		ctx, cancel = context.WithCancel(ctx)
	} else {
		// timeoutDuration is the completion budget remaining. If less than
		// 10ms, set to 10ms
		timeoutDuration := time.Until(c.startTime.Add(c.opts.budget))
		if timeoutDuration < 10*time.Millisecond {
			timeoutDuration = 10 * time.Millisecond
		}
		ctx, cancel = context.WithTimeout(ctx, timeoutDuration)
	}
	defer cancel()

	if surrounding := c.containingIdent(pgf.Src); surrounding != nil {
		c.setSurrounding(surrounding)
	}

	c.inference = expectedCandidate(ctx, c)

	err = c.collectCompletions(ctx)
	if err != nil {
		return nil, nil, err
	}

	// Deep search collected candidates and their members for more candidates.
	c.deepSearch(ctx)
	c.deepState.searchQueue = nil

	for _, callback := range c.completionCallbacks {
		if err := c.snapshot.RunProcessEnvFunc(ctx, callback); err != nil {
			return nil, nil, err
		}
	}

	// Search candidates populated by expensive operations like
	// unimportedMembers etc. for more completion items.
	c.deepSearch(ctx)

	// Statement candidates offer an entire statement in certain contexts, as
	// opposed to a single object. Add statement candidates last because they
	// depend on other candidates having already been collected.
	c.addStatementCandidates()

	c.sortItems()
	return c.items, c.getSurrounding(), nil
}

// collectCompletions adds possible completion candidates to either the deep
// search queue or completion items directly for different completion contexts.
func (c *completer) collectCompletions(ctx context.Context) error {
	// Inside import blocks, return completions for unimported packages.
	for _, importSpec := range c.file.Imports {
		if !(importSpec.Path.Pos() <= c.pos && c.pos <= importSpec.Path.End()) {
			continue
		}
		return c.populateImportCompletions(ctx, importSpec)
	}

	// Inside comments, offer completions for the name of the relevant symbol.
	for _, comment := range c.file.Comments {
		if comment.Pos() < c.pos && c.pos <= comment.End() {
			c.populateCommentCompletions(ctx, comment)
			return nil
		}
	}

	// Struct literals are handled entirely separately.
	if c.wantStructFieldCompletions() {
		// If we are definitely completing a struct field name, deep completions
		// don't make sense.
		if c.enclosingCompositeLiteral.inKey {
			c.deepState.enabled = false
		}
		return c.structLiteralFieldName(ctx)
	}

	if lt := c.wantLabelCompletion(); lt != labelNone {
		c.labels(lt)
		return nil
	}

	if c.emptySwitchStmt() {
		// Empty switch statements only admit "default" and "case" keywords.
		c.addKeywordItems(map[string]bool{}, highScore, CASE, DEFAULT)
		return nil
	}

	switch n := c.path[0].(type) {
	case *ast.Ident:
		if c.file.Name == n {
			return c.packageNameCompletions(ctx, c.fh.URI(), n)
		} else if sel, ok := c.path[1].(*ast.SelectorExpr); ok && sel.Sel == n {
			// Is this the Sel part of a selector?
			return c.selector(ctx, sel)
		}
		return c.lexical(ctx)
	// The function name hasn't been typed yet, but the parens are there:
	//   recv.‸(arg)
	case *ast.TypeAssertExpr:
		// Create a fake selector expression.
		return c.selector(ctx, &ast.SelectorExpr{X: n.X})
	case *ast.SelectorExpr:
		return c.selector(ctx, n)
	// At the file scope, only keywords are allowed.
	case *ast.BadDecl, *ast.File:
		c.addKeywordCompletions()
	default:
		// fallback to lexical completions
		return c.lexical(ctx)
	}

	return nil
}

// containingIdent returns the *ast.Ident containing pos, if any. It
// synthesizes an *ast.Ident to allow completion in the face of
// certain syntax errors.
func (c *completer) containingIdent(src []byte) *ast.Ident {
	// In the normal case, our leaf AST node is the identifer being completed.
	if ident, ok := c.path[0].(*ast.Ident); ok {
		return ident
	}

	pos, tkn, lit := c.scanToken(src)
	if !pos.IsValid() {
		return nil
	}

	fakeIdent := &ast.Ident{Name: lit, NamePos: pos}

	if _, isBadDecl := c.path[0].(*ast.BadDecl); isBadDecl {
		// You don't get *ast.Idents at the file level, so look for bad
		// decls and use the manually extracted token.
		return fakeIdent
	} else if c.emptySwitchStmt() {
		// Only keywords are allowed in empty switch statements.
		// *ast.Idents are not parsed, so we must use the manually
		// extracted token.
		return fakeIdent
	} else if tkn.IsKeyword() {
		// Otherwise, manually extract the prefix if our containing token
		// is a keyword. This improves completion after an "accidental
		// keyword", e.g. completing to "variance" in "someFunc(var<>)".
		return fakeIdent
	}

	return nil
}

// scanToken scans pgh's contents for the token containing pos.
func (c *completer) scanToken(contents []byte) (token.Pos, token.Token, string) {
	tok := c.snapshot.FileSet().File(c.pos)

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
		// effect of preferring shorter candidates.
		return c.items[i].Label < c.items[j].Label
	})
}

// emptySwitchStmt reports whether pos is in an empty switch or select
// statement.
func (c *completer) emptySwitchStmt() bool {
	block, ok := c.path[0].(*ast.BlockStmt)
	if !ok || len(block.List) > 0 || len(c.path) == 1 {
		return false
	}

	switch c.path[1].(type) {
	case *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
		return true
	default:
		return false
	}
}

// populateImportCompletions yields completions for an import path around the cursor.
//
// Completions are suggested at the directory depth of the given import path so
// that we don't overwhelm the user with a large list of possibilities. As an
// example, a completion for the prefix "golang" results in "golang.org/".
// Completions for "golang.org/" yield its subdirectories
// (i.e. "golang.org/x/"). The user is meant to accept completion suggestions
// until they reach a complete import path.
func (c *completer) populateImportCompletions(ctx context.Context, searchImport *ast.ImportSpec) error {
	// deepSearch is not valuable for import completions.
	c.deepState.enabled = false

	importPath := searchImport.Path.Value

	// Extract the text between the quotes (if any) in an import spec.
	// prefix is the part of import path before the cursor.
	prefixEnd := c.pos - searchImport.Path.Pos()
	prefix := strings.Trim(importPath[:prefixEnd], `"`)

	// The number of directories in the import path gives us the depth at
	// which to search.
	depth := len(strings.Split(prefix, "/")) - 1

	content := importPath
	start, end := searchImport.Path.Pos(), searchImport.Path.End()
	namePrefix, nameSuffix := `"`, `"`
	// If a starting quote is present, adjust surrounding to either after the
	// cursor or after the first slash (/), except if cursor is at the starting
	// quote. Otherwise we provide a completion including the starting quote.
	if strings.HasPrefix(importPath, `"`) && c.pos > searchImport.Path.Pos() {
		content = content[1:]
		start++
		if depth > 0 {
			// Adjust textEdit start to replacement range. For ex: if current
			// path was "golang.or/x/to<>ols/internal/", where <> is the cursor
			// position, start of the replacement range would be after
			// "golang.org/x/".
			path := strings.SplitAfter(prefix, "/")
			numChars := len(strings.Join(path[:len(path)-1], ""))
			content = content[numChars:]
			start += token.Pos(numChars)
		}
		namePrefix = ""
	}

	// We won't provide an ending quote if one is already present, except if
	// cursor is after the ending quote but still in import spec. This is
	// because cursor has to be in our textEdit range.
	if strings.HasSuffix(importPath, `"`) && c.pos < searchImport.Path.End() {
		end--
		content = content[:len(content)-1]
		nameSuffix = ""
	}

	c.surrounding = &Selection{
		content:     content,
		cursor:      c.pos,
		MappedRange: source.NewMappedRange(c.snapshot.FileSet(), c.mapper, start, end),
	}

	seenImports := make(map[string]struct{})
	for _, importSpec := range c.file.Imports {
		if importSpec.Path.Value == importPath {
			continue
		}
		seenImportPath, err := strconv.Unquote(importSpec.Path.Value)
		if err != nil {
			return err
		}
		seenImports[seenImportPath] = struct{}{}
	}

	var mu sync.Mutex // guard c.items locally, since searchImports is called in parallel
	seen := make(map[string]struct{})
	searchImports := func(pkg imports.ImportFix) {
		path := pkg.StmtInfo.ImportPath
		if _, ok := seenImports[path]; ok {
			return
		}

		// Any package path containing fewer directories than the search
		// prefix is not a match.
		pkgDirList := strings.Split(path, "/")
		if len(pkgDirList) < depth+1 {
			return
		}
		pkgToConsider := strings.Join(pkgDirList[:depth+1], "/")

		name := pkgDirList[depth]
		// if we're adding an opening quote to completion too, set name to full
		// package path since we'll need to overwrite that range.
		if namePrefix == `"` {
			name = pkgToConsider
		}

		score := pkg.Relevance
		if len(pkgDirList)-1 == depth {
			score *= highScore
		} else {
			// For incomplete package paths, add a terminal slash to indicate that the
			// user should keep triggering completions.
			name += "/"
			pkgToConsider += "/"
		}

		if _, ok := seen[pkgToConsider]; ok {
			return
		}
		seen[pkgToConsider] = struct{}{}

		mu.Lock()
		defer mu.Unlock()

		name = namePrefix + name + nameSuffix
		obj := types.NewPkgName(0, nil, name, types.NewPackage(pkgToConsider, name))
		c.deepState.enqueue(candidate{
			obj:    obj,
			detail: fmt.Sprintf("%q", pkgToConsider),
			score:  score,
		})
	}

	c.completionCallbacks = append(c.completionCallbacks, func(opts *imports.Options) error {
		return imports.GetImportPaths(ctx, searchImports, prefix, c.filename, c.pkg.GetTypes().Name(), opts.Env)
	})
	return nil
}

// populateCommentCompletions yields completions for comments preceding or in declarations.
func (c *completer) populateCommentCompletions(ctx context.Context, comment *ast.CommentGroup) {
	// If the completion was triggered by a period, ignore it. These types of
	// completions will not be useful in comments.
	if c.completionContext.triggerCharacter == "." {
		return
	}

	// Using the comment position find the line after
	file := c.snapshot.FileSet().File(comment.End())
	if file == nil {
		return
	}

	// Deep completion doesn't work properly in comments since we don't
	// have a type object to complete further.
	c.deepState.enabled = false
	c.completionContext.commentCompletion = true

	// Documentation isn't useful in comments, since it might end up being the
	// comment itself.
	c.opts.documentation = false

	commentLine := file.Line(comment.End())

	// comment is valid, set surrounding as word boundaries around cursor
	c.setSurroundingForComment(comment)

	// Using the next line pos, grab and parse the exported symbol on that line
	for _, n := range c.file.Decls {
		declLine := file.Line(n.Pos())
		// if the comment is not in, directly above or on the same line as a declaration
		if declLine != commentLine && declLine != commentLine+1 &&
			!(n.Pos() <= comment.Pos() && comment.End() <= n.End()) {
			continue
		}
		switch node := n.(type) {
		// handle const, vars, and types
		case *ast.GenDecl:
			for _, spec := range node.Specs {
				switch spec := spec.(type) {
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if name.String() == "_" {
							continue
						}
						obj := c.pkg.GetTypesInfo().ObjectOf(name)
						c.deepState.enqueue(candidate{obj: obj, score: stdScore})
					}
				case *ast.TypeSpec:
					// add TypeSpec fields to completion
					switch typeNode := spec.Type.(type) {
					case *ast.StructType:
						c.addFieldItems(ctx, typeNode.Fields)
					case *ast.FuncType:
						c.addFieldItems(ctx, typeNode.Params)
						c.addFieldItems(ctx, typeNode.Results)
					case *ast.InterfaceType:
						c.addFieldItems(ctx, typeNode.Methods)
					}

					if spec.Name.String() == "_" {
						continue
					}

					obj := c.pkg.GetTypesInfo().ObjectOf(spec.Name)
					// Type name should get a higher score than fields but not highScore by default
					// since field near a comment cursor gets a highScore
					score := stdScore * 1.1
					// If type declaration is on the line after comment, give it a highScore.
					if declLine == commentLine+1 {
						score = highScore
					}

					c.deepState.enqueue(candidate{obj: obj, score: score})
				}
			}
		// handle functions
		case *ast.FuncDecl:
			c.addFieldItems(ctx, node.Recv)
			c.addFieldItems(ctx, node.Type.Params)
			c.addFieldItems(ctx, node.Type.Results)

			// collect receiver struct fields
			if node.Recv != nil {
				for _, fields := range node.Recv.List {
					for _, name := range fields.Names {
						obj := c.pkg.GetTypesInfo().ObjectOf(name)
						if obj == nil {
							continue
						}

						recvType := obj.Type().Underlying()
						if ptr, ok := recvType.(*types.Pointer); ok {
							recvType = ptr.Elem()
						}
						recvStruct, ok := recvType.Underlying().(*types.Struct)
						if !ok {
							continue
						}
						for i := 0; i < recvStruct.NumFields(); i++ {
							field := recvStruct.Field(i)
							c.deepState.enqueue(candidate{obj: field, score: lowScore})
						}
					}
				}
			}

			if node.Name.String() == "_" {
				continue
			}

			obj := c.pkg.GetTypesInfo().ObjectOf(node.Name)
			if obj == nil || obj.Pkg() != nil && obj.Pkg() != c.pkg.GetTypes() {
				continue
			}

			c.deepState.enqueue(candidate{obj: obj, score: highScore})
		}
	}
}

// sets word boundaries surrounding a cursor for a comment
func (c *completer) setSurroundingForComment(comments *ast.CommentGroup) {
	var cursorComment *ast.Comment
	for _, comment := range comments.List {
		if c.pos >= comment.Pos() && c.pos <= comment.End() {
			cursorComment = comment
			break
		}
	}
	// if cursor isn't in the comment
	if cursorComment == nil {
		return
	}

	// index of cursor in comment text
	cursorOffset := int(c.pos - cursorComment.Pos())
	start, end := cursorOffset, cursorOffset
	for start > 0 && isValidIdentifierChar(cursorComment.Text[start-1]) {
		start--
	}
	for end < len(cursorComment.Text) && isValidIdentifierChar(cursorComment.Text[end]) {
		end++
	}

	c.surrounding = &Selection{
		content: cursorComment.Text[start:end],
		cursor:  c.pos,
		MappedRange: source.NewMappedRange(c.snapshot.FileSet(), c.mapper,
			token.Pos(int(cursorComment.Slash)+start), token.Pos(int(cursorComment.Slash)+end)),
	}
	c.setMatcherFromPrefix(c.surrounding.Prefix())
}

// isValidIdentifierChar returns true if a byte is a valid go identifier
// character, i.e. unicode letter or digit or underscore.
func isValidIdentifierChar(char byte) bool {
	charRune := rune(char)
	return unicode.In(charRune, unicode.Letter, unicode.Digit) || char == '_'
}

// adds struct fields, interface methods, function declaration fields to completion
func (c *completer) addFieldItems(ctx context.Context, fields *ast.FieldList) {
	if fields == nil {
		return
	}

	cursor := c.surrounding.cursor
	for _, field := range fields.List {
		for _, name := range field.Names {
			if name.String() == "_" {
				continue
			}
			obj := c.pkg.GetTypesInfo().ObjectOf(name)
			if obj == nil {
				continue
			}

			// if we're in a field comment/doc, score that field as more relevant
			score := stdScore
			if field.Comment != nil && field.Comment.Pos() <= cursor && cursor <= field.Comment.End() {
				score = highScore
			} else if field.Doc != nil && field.Doc.Pos() <= cursor && cursor <= field.Doc.End() {
				score = highScore
			}

			c.deepState.enqueue(candidate{obj: obj, score: score})
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
	return !c.completionContext.commentCompletion && c.inference.typeName.wantTypeName
}

// See https://golang.org/issue/36001. Unimported completions are expensive.
const (
	maxUnimportedPackageNames = 5
	unimportedMemberTarget    = 100
)

// selector finds completions for the specified selector expression.
func (c *completer) selector(ctx context.Context, sel *ast.SelectorExpr) error {
	c.inference.objChain = objChain(c.pkg.GetTypesInfo(), sel.X)

	// Is sel a qualified identifier?
	if id, ok := sel.X.(*ast.Ident); ok {
		if pkgName, ok := c.pkg.GetTypesInfo().Uses[id].(*types.PkgName); ok {
			candidates := c.packageMembers(pkgName.Imported(), stdScore, nil)
			for _, cand := range candidates {
				c.deepState.enqueue(cand)
			}
			return nil
		}
	}

	// Invariant: sel is a true selector.
	tv, ok := c.pkg.GetTypesInfo().Types[sel.X]
	if ok {
		candidates := c.methodsAndFields(tv.Type, tv.Addressable(), nil)
		for _, cand := range candidates {
			c.deepState.enqueue(cand)
		}
		return nil
	}

	// Try unimported packages.
	if id, ok := sel.X.(*ast.Ident); ok && c.opts.unimported {
		if err := c.unimportedMembers(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

func (c *completer) unimportedMembers(ctx context.Context, id *ast.Ident) error {
	// Try loaded packages first. They're relevant, fast, and fully typed.
	known, err := c.snapshot.CachedImportPaths(ctx)
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

	var relevances map[string]float64
	if len(paths) != 0 {
		if err := c.snapshot.RunProcessEnvFunc(ctx, func(opts *imports.Options) error {
			var err error
			relevances, err = imports.ScoreImportPaths(ctx, opts.Env, paths)
			return err
		}); err != nil {
			return err
		}
	}
	sort.Slice(paths, func(i, j int) bool {
		return relevances[paths[i]] > relevances[paths[j]]
	})

	for _, path := range paths {
		pkg := known[path]
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
		candidates := c.packageMembers(pkg.GetTypes(), unimportedScore(relevances[path]), imp)
		for _, cand := range candidates {
			c.deepState.enqueue(cand)
		}
		if len(c.items) >= unimportedMemberTarget {
			return nil
		}
	}

	ctx, cancel := context.WithCancel(ctx)

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
			score := unimportedScore(pkgExport.Fix.Relevance)
			c.deepState.enqueue(candidate{
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

	c.completionCallbacks = append(c.completionCallbacks, func(opts *imports.Options) error {
		defer cancel()
		return imports.GetPackageExports(ctx, add, id.Name, c.filename, c.pkg.GetTypes().Name(), opts.Env)
	})
	return nil
}

// unimportedScore returns a score for an unimported package that is generally
// lower than other candidates.
func unimportedScore(relevance float64) float64 {
	return (stdScore + .1*relevance) / 2
}

func (c *completer) packageMembers(pkg *types.Package, score float64, imp *importInfo) []candidate {
	var candidates []candidate
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		candidates = append(candidates, candidate{
			obj:         obj,
			score:       score,
			imp:         imp,
			addressable: isVar(obj),
		})
	}
	return candidates
}

func (c *completer) methodsAndFields(typ types.Type, addressable bool, imp *importInfo) []candidate {
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

	var candidates []candidate
	for i := 0; i < mset.Len(); i++ {
		candidates = append(candidates, candidate{
			obj:         mset.At(i).Obj(),
			score:       stdScore,
			imp:         imp,
			addressable: addressable || isPointer(typ),
		})
	}

	// Add fields of T.
	eachField(typ, func(v *types.Var) {
		candidates = append(candidates, candidate{
			obj:         v,
			score:       stdScore - 0.01,
			imp:         imp,
			addressable: addressable || isPointer(typ),
		})
	})

	return candidates
}

// lexical finds completions in the lexical environment.
func (c *completer) lexical(ctx context.Context) error {
	scopes := source.CollectScopes(c.pkg.GetTypesInfo(), c.path, c.pos)
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
					if resolved := resolveInvalid(c.snapshot.FileSet(), obj, node, c.pkg.GetTypesInfo()); resolved != nil {
						obj = resolved
					}
				}
			}

			// Don't use LHS of value spec in RHS.
			if vs := enclosingValueSpec(c.path); vs != nil {
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
				c.deepState.enqueue(candidate{
					obj:         obj,
					score:       score,
					addressable: isVar(obj),
				})
			}
		}
	}

	if c.inference.objType != nil {
		if named, _ := source.Deref(c.inference.objType).(*types.Named); named != nil {
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
					c.deepState.enqueue(candidate{
						obj:   obj,
						score: stdScore,
						imp:   imp,
					})
				}
			}
		}
	}

	if c.opts.unimported {
		if err := c.unimportedPackages(ctx, seen); err != nil {
			return err
		}
	}

	if t := c.inference.objType; t != nil {
		t = source.Deref(t)

		// If we have an expected type and it is _not_ a named type,
		// handle it specially. Non-named types like "[]int" will never be
		// considered via a lexical search, so we need to directly inject
		// them.
		if _, named := t.(*types.Named); !named {
			// If our expected type is "[]int", this will add a literal
			// candidate of "[]int{}".
			c.literal(ctx, t, nil)

			if _, isBasic := t.(*types.Basic); !isBasic {
				// If we expect a non-basic type name (e.g. "[]int"), hack up
				// a named type whose name is literally "[]int". This allows
				// us to reuse our object based completion machinery.
				fakeNamedType := candidate{
					obj:   types.NewTypeName(token.NoPos, nil, types.TypeString(t, c.qf), t),
					score: stdScore,
				}
				// Make sure the type name matches before considering
				// candidate. This cuts down on useless candidates.
				if c.matchingTypeName(&fakeNamedType) {
					c.deepState.enqueue(fakeNamedType)
				}
			}
		}
	}

	// Add keyword completion items appropriate in the current context.
	c.addKeywordCompletions()

	return nil
}

func (c *completer) unimportedPackages(ctx context.Context, seen map[string]struct{}) error {
	var prefix string
	if c.surrounding != nil {
		prefix = c.surrounding.Prefix()
	}
	count := 0

	known, err := c.snapshot.CachedImportPaths(ctx)
	if err != nil {
		return err
	}
	var paths []string
	for path, pkg := range known {
		if !strings.HasPrefix(pkg.GetTypes().Name(), prefix) {
			continue
		}
		paths = append(paths, path)
	}

	var relevances map[string]float64
	if len(paths) != 0 {
		if err := c.snapshot.RunProcessEnvFunc(ctx, func(opts *imports.Options) error {
			var err error
			relevances, err = imports.ScoreImportPaths(ctx, opts.Env, paths)
			return err
		}); err != nil {
			return err
		}
	}
	sort.Slice(paths, func(i, j int) bool {
		return relevances[paths[i]] > relevances[paths[j]]
	})

	for _, path := range paths {
		pkg := known[path]
		if _, ok := seen[pkg.GetTypes().Name()]; ok {
			continue
		}
		imp := &importInfo{
			importPath: path,
			pkg:        pkg,
		}
		if imports.ImportPathToAssumedName(path) != pkg.GetTypes().Name() {
			imp.name = pkg.GetTypes().Name()
		}
		if count >= maxUnimportedPackageNames {
			return nil
		}
		c.deepState.enqueue(candidate{
			obj:   types.NewPkgName(0, nil, pkg.GetTypes().Name(), pkg.GetTypes()),
			score: unimportedScore(relevances[path]),
			imp:   imp,
		})
		count++
	}

	ctx, cancel := context.WithCancel(ctx)

	var mu sync.Mutex
	add := func(pkg imports.ImportFix) {
		mu.Lock()
		defer mu.Unlock()
		if _, ok := seen[pkg.IdentName]; ok {
			return
		}
		if _, ok := relevances[pkg.StmtInfo.ImportPath]; ok {
			return
		}

		if count >= maxUnimportedPackageNames {
			cancel()
			return
		}

		// Do not add the unimported packages to seen, since we can have
		// multiple packages of the same name as completion suggestions, since
		// only one will be chosen.
		obj := types.NewPkgName(0, nil, pkg.IdentName, types.NewPackage(pkg.StmtInfo.ImportPath, pkg.IdentName))
		c.deepState.enqueue(candidate{
			obj:   obj,
			score: unimportedScore(pkg.Relevance),
			imp: &importInfo{
				importPath: pkg.StmtInfo.ImportPath,
				name:       pkg.StmtInfo.Name,
			},
		})
		count++
	}
	c.completionCallbacks = append(c.completionCallbacks, func(opts *imports.Options) error {
		defer cancel()
		return imports.GetAllCandidates(ctx, add, prefix, c.filename, c.pkg.GetTypes().Name(), opts.Env)
	})
	return nil
}

// alreadyImports reports whether f has an import with the specified path.
func alreadyImports(f *ast.File, path string) bool {
	for _, s := range f.Imports {
		if source.ImportPath(s) == path {
			return true
		}
	}
	return false
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
func (c *completer) structLiteralFieldName(ctx context.Context) error {
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

	deltaScore := 0.0001
	switch t := clInfo.clType.(type) {
	case *types.Struct:
		for i := 0; i < t.NumFields(); i++ {
			field := t.Field(i)
			if !addedFields[field] {
				c.deepState.enqueue(candidate{
					obj:   field,
					score: highScore - float64(i)*deltaScore,
				})
			}
		}

		// Add lexical completions if we aren't certain we are in the key part of a
		// key-value pair.
		if clInfo.maybeInFieldName {
			return c.lexical(ctx)
		}
	default:
		return c.lexical(ctx)
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
				clType: source.Deref(tv.Type).Underlying(),
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
			if breaksExpectedTypeInference(n, pos) {
				return nil
			}
		}
	}

	return nil
}

// enclosingFunction returns the signature and body of the function
// enclosing the given position.
func enclosingFunction(path []ast.Node, info *types.Info) *funcInfo {
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
	dereference typeMod = iota // pointer indirection: "*"
	reference                  // adds level of pointer: "&" for values, "*" for type names
	chanRead                   // channel read operator ("<-")
	slice                      // make a slice type ("[]" in "[]int")
	array                      // make an array type ("[2]" in "[2]int")
)

type objKind int

const (
	kindAny   objKind = 0
	kindArray objKind = 1 << iota
	kindSlice
	kindChan
	kindMap
	kindStruct
	kindString
	kindInt
	kindBool
	kindBytes
	kindPtr
	kindFloat
	kindComplex
	kindError
	kindStringer
	kindFunc
)

// penalizedObj represents an object that should be disfavored as a
// completion candidate.
type penalizedObj struct {
	// objChain is the full "chain", e.g. "foo.bar().baz" becomes
	// []types.Object{foo, bar, baz}.
	objChain []types.Object
	// penalty is score penalty in the range (0, 1).
	penalty float64
}

// candidateInference holds information we have inferred about a type that can be
// used at the current position.
type candidateInference struct {
	// objType is the desired type of an object used at the query position.
	objType types.Type

	// objKind is a mask of expected kinds of types such as "map", "slice", etc.
	objKind objKind

	// variadic is true if we are completing the initial variadic
	// parameter. For example:
	//   append([]T{}, <>)      // objType=T variadic=true
	//   append([]T{}, T{}, <>) // objType=T variadic=false
	variadic bool

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

	// penalized holds expressions that should be disfavored as
	// candidates. For example, it tracks expressions already used in a
	// switch statement's other cases. Each expression is tracked using
	// its entire object "chain" allowing differentiation between
	// "a.foo" and "b.foo" when "a" and "b" are the same type.
	penalized []penalizedObj

	// objChain contains the chain of objects representing the
	// surrounding *ast.SelectorExpr. For example, if we are completing
	// "foo.bar.ba<>", objChain will contain []types.Object{foo, bar}.
	objChain []types.Object
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

	// seenTypeSwitchCases tracks types that have already been used by
	// the containing type switch.
	seenTypeSwitchCases []types.Type

	// compLitType is true if we are completing a composite literal type
	// name, e.g "foo<>{}".
	compLitType bool
}

// expectedCandidate returns information about the expected candidate
// for an expression at the query position.
func expectedCandidate(ctx context.Context, c *completer) (inf candidateInference) {
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
				switch node.Op {
				case token.LAND, token.LOR:
					// Don't infer "bool" type for "&&" or "||". Often you want
					// to compose a boolean expression from non-boolean
					// candidates.
				default:
					inf.objType = tv.Type
				}
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
			if node.Lparen < c.pos && c.pos <= node.Rparen {
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

						exprIdx := exprAtPos(c.pos, node.Args)

						// If we have one or zero arg expressions, we may be
						// completing to a function call that returns multiple
						// values, in turn getting passed in to the surrounding
						// call. Record the assignees so we can favor function
						// calls that return matching values.
						if len(node.Args) <= 1 && exprIdx == 0 {
							for i := 0; i < sig.Params().Len(); i++ {
								inf.assignees = append(inf.assignees, sig.Params().At(i).Type())
							}

							// Record that we may be completing into variadic parameters.
							inf.variadicAssignees = sig.Variadic()
						}

						// Make sure not to run past the end of expected parameters.
						if exprIdx >= numParams {
							inf.objType = sig.Params().At(numParams - 1).Type()
						} else {
							inf.objType = sig.Params().At(exprIdx).Type()
						}

						if sig.Variadic() && exprIdx >= (numParams-1) {
							// If we are completing a variadic param, deslice the variadic type.
							inf.objType = deslice(inf.objType)
							// Record whether we are completing the initial variadic param.
							inf.variadic = exprIdx == numParams-1 && len(node.Args) <= numParams

							// Check if we can infer object kind from printf verb.
							inf.objKind |= printfArgKind(c.pkg.GetTypesInfo(), node, exprIdx)
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
							inf.objKind = c.builtinArgKind(ctx, obj, node)
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

				return inf
			}
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

					// Record which objects have already been used in the case
					// statements so we don't suggest them again.
					for _, cc := range swtch.Body.List {
						for _, caseExpr := range cc.(*ast.CaseClause).List {
							// Don't record the expression we are currently completing.
							if caseExpr.Pos() < c.pos && c.pos <= caseExpr.End() {
								continue
							}

							if objs := objChain(c.pkg.GetTypesInfo(), caseExpr); len(objs) > 0 {
								inf.penalized = append(inf.penalized, penalizedObj{objChain: objs, penalty: 0.1})
							}
						}
					}
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
			if source.NodeContains(node.X, c.pos) {
				inf.objKind |= kindSlice | kindArray | kindMap | kindString
				if node.Value == nil {
					inf.objKind |= kindChan
				}
			}
			return inf
		case *ast.StarExpr:
			inf.modifiers = append(inf.modifiers, typeModifier{mod: dereference})
		case *ast.UnaryExpr:
			switch node.Op {
			case token.AND:
				inf.modifiers = append(inf.modifiers, typeModifier{mod: reference})
			case token.ARROW:
				inf.modifiers = append(inf.modifiers, typeModifier{mod: chanRead})
			}
		case *ast.DeferStmt, *ast.GoStmt:
			inf.objKind |= kindFunc
			return inf
		default:
			if breaksExpectedTypeInference(node, c.pos) {
				return inf
			}
		}
	}

	return inf
}

// objChain decomposes e into a chain of objects if possible. For
// example, "foo.bar().baz" will yield []types.Object{foo, bar, baz}.
// If any part can't be turned into an object, return nil.
func objChain(info *types.Info, e ast.Expr) []types.Object {
	var objs []types.Object

	for e != nil {
		switch n := e.(type) {
		case *ast.Ident:
			obj := info.ObjectOf(n)
			if obj == nil {
				return nil
			}
			objs = append(objs, obj)
			e = nil
		case *ast.SelectorExpr:
			obj := info.ObjectOf(n.Sel)
			if obj == nil {
				return nil
			}
			objs = append(objs, obj)
			e = n.X
		case *ast.CallExpr:
			if len(n.Args) > 0 {
				return nil
			}
			e = n.Fun
		default:
			return nil
		}
	}

	// Reverse order so the layout matches the syntactic order.
	for i := 0; i < len(objs)/2; i++ {
		objs[i], objs[len(objs)-1-i] = objs[len(objs)-1-i], objs[i]
	}

	return objs
}

// applyTypeModifiers applies the list of type modifiers to a type.
// It returns nil if the modifiers could not be applied.
func (ci candidateInference) applyTypeModifiers(typ types.Type, addressable bool) types.Type {
	for _, mod := range ci.modifiers {
		switch mod.mod {
		case dereference:
			// For every "*" indirection operator, remove a pointer layer
			// from candidate type.
			if ptr, ok := typ.Underlying().(*types.Pointer); ok {
				typ = ptr.Elem()
			} else {
				return nil
			}
		case reference:
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
		case reference:
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
	return ci.variadic && ci.objType != nil && types.AssignableTo(candType, types.NewSlice(ci.objType))
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
func breaksExpectedTypeInference(n ast.Node, pos token.Pos) bool {
	switch n := n.(type) {
	case *ast.CompositeLit:
		// Doesn't break inference if pos is in type name.
		// For example: "Foo<>{Bar: 123}"
		return !source.NodeContains(n.Type, pos)
	case *ast.CallExpr:
		// Doesn't break inference if pos is in func name.
		// For example: "Foo<>(123)"
		return !source.NodeContains(n.Fun, pos)
	case *ast.FuncLit, *ast.IndexExpr, *ast.SliceExpr:
		return true
	default:
		return false
	}
}

// expectTypeName returns information about the expected type name at position.
func expectTypeName(c *completer) typeNameInference {
	var inf typeNameInference

Nodes:
	for i, p := range c.path {
		switch n := p.(type) {
		case *ast.FieldList:
			// Expect a type name if pos is in a FieldList. This applies to
			// FuncType params/results, FuncDecl receiver, StructType, and
			// InterfaceType. We don't need to worry about the field name
			// because completion bails out early if pos is in an *ast.Ident
			// that defines an object.
			inf.wantTypeName = true
			break Nodes
		case *ast.CaseClause:
			// Expect type names in type switch case clauses.
			if swtch, ok := findSwitchStmt(c.path[i+1:], c.pos, n).(*ast.TypeSwitchStmt); ok {
				// The case clause types must be assertable from the type switch parameter.
				ast.Inspect(swtch.Assign, func(n ast.Node) bool {
					if ta, ok := n.(*ast.TypeAssertExpr); ok {
						inf.assertableFrom = c.pkg.GetTypesInfo().TypeOf(ta.X)
						return false
					}
					return true
				})
				inf.wantTypeName = true

				// Track the types that have already been used in this
				// switch's case statements so we don't recommend them.
				for _, e := range swtch.Body.List {
					for _, typeExpr := range e.(*ast.CaseClause).List {
						// Skip if type expression contains pos. We don't want to
						// count it as already used if the user is completing it.
						if typeExpr.Pos() < c.pos && c.pos <= typeExpr.End() {
							continue
						}

						if t := c.pkg.GetTypesInfo().TypeOf(typeExpr); t != nil {
							inf.seenTypeSwitchCases = append(inf.seenTypeSwitchCases, t)
						}
					}
				}

				break Nodes
			}
			return typeNameInference{}
		case *ast.TypeAssertExpr:
			// Expect type names in type assert expressions.
			if n.Lparen < c.pos && c.pos <= n.Rparen {
				// The type in parens must be assertable from the expression type.
				inf.assertableFrom = c.pkg.GetTypesInfo().TypeOf(n.X)
				inf.wantTypeName = true
				break Nodes
			}
			return typeNameInference{}
		case *ast.StarExpr:
			inf.modifiers = append(inf.modifiers, typeModifier{mod: reference})
		case *ast.CompositeLit:
			// We want a type name if position is in the "Type" part of a
			// composite literal (e.g. "Foo<>{}").
			if n.Type != nil && n.Type.Pos() <= c.pos && c.pos <= n.Type.End() {
				inf.wantTypeName = true
				inf.compLitType = true

				if i < len(c.path)-1 {
					// Track preceding "&" operator. Technically it applies to
					// the composite literal and not the type name, but if
					// affects our type completion nonetheless.
					if u, ok := c.path[i+1].(*ast.UnaryExpr); ok && u.Op == token.AND {
						inf.modifiers = append(inf.modifiers, typeModifier{mod: reference})
					}
				}
			}
			break Nodes
		case *ast.ArrayType:
			// If we are inside the "Elt" part of an array type, we want a type name.
			if n.Elt.Pos() <= c.pos && c.pos <= n.Elt.End() {
				inf.wantTypeName = true
				if n.Len == nil {
					// No "Len" expression means a slice type.
					inf.modifiers = append(inf.modifiers, typeModifier{mod: slice})
				} else {
					// Try to get the array type using the constant value of "Len".
					tv, ok := c.pkg.GetTypesInfo().Types[n.Len]
					if ok && tv.Value != nil && tv.Value.Kind() == constant.Int {
						if arrayLen, ok := constant.Int64Val(tv.Value); ok {
							inf.modifiers = append(inf.modifiers, typeModifier{mod: array, arrayLen: arrayLen})
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
			inf.wantTypeName = true
			if n.Key != nil {
				inf.wantComparable = source.NodeContains(n.Key, c.pos)
			} else {
				// If the key is empty, assume we are completing the key if
				// pos is directly after the "map[".
				inf.wantComparable = c.pos == n.Pos()+token.Pos(len("map["))
			}
			break Nodes
		case *ast.ValueSpec:
			inf.wantTypeName = source.NodeContains(n.Type, c.pos)
			break Nodes
		case *ast.TypeSpec:
			inf.wantTypeName = source.NodeContains(n.Type, c.pos)
		default:
			if breaksExpectedTypeInference(p, c.pos) {
				return typeNameInference{}
			}
		}
	}

	return inf
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
// It mutates cand's score in certain cases.
func (c *completer) matchingCandidate(cand *candidate) bool {
	if c.completionContext.commentCompletion {
		return false
	}

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
	var (
		expTypes     = make([]types.Type, 0, 2)
		variadicType types.Type
	)
	if ci.objType != nil {
		expTypes = append(expTypes, ci.objType)

		if ci.variadic {
			variadicType = types.NewSlice(ci.objType)
			expTypes = append(expTypes, variadicType)
		}
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

		for _, expType := range expTypes {
			if isEmptyInterface(expType) {
				continue
			}

			matches, untyped := ci.typeMatches(expType, candType)
			if !matches {
				// If candType doesn't otherwise match, consider if we can
				// convert candType directly to expType.
				if considerTypeConversion(candType, expType, cand.path) {
					cand.convertTo = expType
					// Give a major score penalty so we always prefer directly
					// assignable candidates, all else equal.
					cand.score *= 0.5
					return true
				}

				continue
			}

			if expType == variadicType {
				cand.variadic = true
			}

			// Lower candidate score for untyped conversions. This avoids
			// ranking untyped constants above candidates with an exact type
			// match. Don't lower score of builtin constants, e.g. "true".
			if untyped && !types.Identical(candType, expType) && cand.obj.Parent() != types.Universe {
				// Bigger penalty for deep completions into other packages to
				// avoid random constants from other packages popping up all
				// the time.
				if len(cand.path) > 0 && isPkgName(cand.path[0]) {
					cand.score *= 0.5
				} else {
					cand.score *= 0.75
				}
			}

			return true
		}

		// If we don't have a specific expected type, fall back to coarser
		// object kind checks.
		if ci.objType == nil || isEmptyInterface(ci.objType) {
			// If we were able to apply type modifiers to our candidate type,
			// count that as a match. For example:
			//
			//   var foo chan int
			//   <-fo<>
			//
			// We were able to apply the "<-" type modifier to "foo", so "foo"
			// matches.
			if len(ci.modifiers) > 0 {
				return true
			}

			// If we didn't have an exact type match, check if our object kind
			// matches.
			if ci.kindMatches(candType) {
				if ci.objKind == kindFunc {
					cand.expandFuncCall = true
				}
				return true
			}
		}

		return false
	})
}

// considerTypeConversion returns true if we should offer a completion
// automatically converting "from" to "to".
func considerTypeConversion(from, to types.Type, path []types.Object) bool {
	// Don't offer to convert deep completions from other packages.
	// Otherwise there are many random package level consts/vars that
	// pop up as candidates all the time.
	if len(path) > 0 && isPkgName(path[0]) {
		return false
	}

	if !types.ConvertibleTo(from, to) {
		return false
	}

	// Don't offer to convert ints to strings since that probably
	// doesn't do what the user wants.
	if isBasicKind(from, types.IsInteger) && isBasicKind(to, types.IsString) {
		return false
	}

	return true
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
	return ci.objKind > 0 && ci.objKind&candKind(candType) > 0
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

		// Skip this type if it has already been used in another type
		// switch case.
		for _, seen := range c.inference.typeName.seenTypeSwitchCases {
			if types.Identical(candType, seen) {
				return false
			}
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

	t := cand.obj.Type()

	if typeMatches(t) {
		return true
	}

	if !source.IsInterface(t) && typeMatches(types.NewPointer(t)) {
		if c.inference.typeName.compLitType {
			// If we are completing a composite literal type as in
			// "foo<>{}", to make a pointer we must prepend "&".
			cand.takeAddress = true
		} else {
			// If we are completing a normal type name such as "foo<>", to
			// make a pointer we must prepend "*".
			cand.makePointer = true
		}
		return true
	}

	return false
}

var (
	// "interface { Error() string }" (i.e. error)
	errorIntf = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

	// "interface { String() string }" (i.e. fmt.Stringer)
	stringerIntf = types.NewInterfaceType([]*types.Func{
		types.NewFunc(token.NoPos, nil, "String", types.NewSignature(
			nil,
			nil,
			types.NewTuple(types.NewParam(token.NoPos, nil, "", types.Typ[types.String])),
			false,
		)),
	}, nil).Complete()

	byteType = types.Universe.Lookup("byte").Type()
)

// candKind returns the objKind of candType, if any.
func candKind(candType types.Type) objKind {
	var kind objKind

	switch t := candType.Underlying().(type) {
	case *types.Array:
		kind |= kindArray
		if t.Elem() == byteType {
			kind |= kindBytes
		}
	case *types.Slice:
		kind |= kindSlice
		if t.Elem() == byteType {
			kind |= kindBytes
		}
	case *types.Chan:
		kind |= kindChan
	case *types.Map:
		kind |= kindMap
	case *types.Pointer:
		kind |= kindPtr

		// Some builtins handle array pointers as arrays, so just report a pointer
		// to an array as an array.
		if _, isArray := t.Elem().Underlying().(*types.Array); isArray {
			kind |= kindArray
		}
	case *types.Basic:
		switch info := t.Info(); {
		case info&types.IsString > 0:
			kind |= kindString
		case info&types.IsInteger > 0:
			kind |= kindInt
		case info&types.IsFloat > 0:
			kind |= kindFloat
		case info&types.IsComplex > 0:
			kind |= kindComplex
		case info&types.IsBoolean > 0:
			kind |= kindBool
		}
	case *types.Signature:
		return kindFunc
	}

	if types.Implements(candType, errorIntf) {
		kind |= kindError
	}

	if types.Implements(candType, stringerIntf) {
		kind |= kindStringer
	}

	return kind
}
