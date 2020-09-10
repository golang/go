// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

// maxSymbols defines the maximum number of symbol results that should ever be
// sent in response to a client.
const maxSymbols = 100

// WorkspaceSymbols matches symbols across views using the given query,
// according to the SymbolMatcher matcher.
//
// The workspace symbol method is defined in the spec as follows:
//
//  > The workspace symbol request is sent from the client to the server to
//  > list project-wide symbols matching the query string.
//
// It is unclear what "project-wide" means here, but given the parameters of
// workspace/symbol do not include any workspace identifier, then it has to be
// assumed that "project-wide" means "across all workspaces".  Hence why
// WorkspaceSymbols receives the views []View.
//
// However, it then becomes unclear what it would mean to call WorkspaceSymbols
// with a different configured SymbolMatcher per View. Therefore we assume that
// Session level configuration will define the SymbolMatcher to be used for the
// WorkspaceSymbols method.
func WorkspaceSymbols(ctx context.Context, matcherType SymbolMatcher, style SymbolStyle, views []View, query string) ([]protocol.SymbolInformation, error) {
	ctx, done := event.Start(ctx, "source.WorkspaceSymbols")
	defer done()
	if query == "" {
		return nil, nil
	}
	sc := newSymbolCollector(matcherType, style, query)
	return sc.walk(ctx, views)
}

// A matcherFunc determines the matching score of a symbol.
//
// See the comment for symbolCollector for more information.
type matcherFunc func(name string) float64

// A symbolizer returns the best symbol match for name with pkg, according to
// some heuristic.
//
// See the comment for symbolCollector for more information.
type symbolizer func(name string, pkg Package, m matcherFunc) (string, float64)

func fullyQualifiedSymbolMatch(name string, pkg Package, matcher matcherFunc) (string, float64) {
	fullyQualified := pkg.PkgPath() + "." + name
	if matcher(fullyQualified) > 0 {
		return fullyQualified, 1
	}
	return "", 0
}

func dynamicSymbolMatch(name string, pkg Package, matcher matcherFunc) (string, float64) {
	// Prefer any package-qualified match.
	pkgQualified := pkg.Name() + "." + name
	if match, score := bestMatch(pkgQualified, matcher); match != "" {
		return match, score
	}
	fullyQualified := pkg.PkgPath() + "." + name
	if match, score := bestMatch(fullyQualified, matcher); match != "" {
		return match, score
	}
	return "", 0
}

func packageSymbolMatch(name string, pkg Package, matcher matcherFunc) (string, float64) {
	qualified := pkg.Name() + "." + name
	if matcher(qualified) > 0 {
		return qualified, 1
	}
	return "", 0
}

// bestMatch returns the highest scoring symbol suffix of fullPath, starting
// from the right and splitting on selectors and path components.
//
// e.g. given a symbol path of the form 'host.com/dir/pkg.type.field', we
// check the match quality of the following:
//  - field
//  - type.field
//  - pkg.type.field
//  - dir/pkg.type.field
//  - host.com/dir/pkg.type.field
//
// and return the best match, along with its score.
//
// This is used to implement the 'dynamic' symbol style.
func bestMatch(fullPath string, matcher matcherFunc) (string, float64) {
	pathParts := strings.Split(fullPath, "/")
	dottedParts := strings.Split(pathParts[len(pathParts)-1], ".")

	var best string
	var score float64

	for i := 0; i < len(dottedParts); i++ {
		path := strings.Join(dottedParts[len(dottedParts)-1-i:], ".")
		if match := matcher(path); match > score {
			best = path
			score = match
		}
	}
	for i := 0; i < len(pathParts); i++ {
		path := strings.Join(pathParts[len(pathParts)-1-i:], "/")
		if match := matcher(path); match > score {
			best = path
			score = match
		}
	}
	return best, score
}

// symbolCollector holds context as we walk Packages, gathering symbols that
// match a given query.
//
// How we match symbols is parameterized by two interfaces:
//  * A matcherFunc determines how well a string symbol matches a query. It
//    returns a non-negative score indicating the quality of the match. A score
//    of zero indicates no match.
//  * A symbolizer determines how we extract the symbol for an object. This
//    enables the 'symbolStyle' configuration option.
type symbolCollector struct {
	// query is the user-supplied query passed to the Symbol method.
	query string

	// These types parameterize the symbol-matching pass.
	matcher    matcherFunc
	symbolizer symbolizer

	// current holds metadata for the package we are currently walking.
	current *pkgView
	curFile *ParsedGoFile

	res [maxSymbols]symbolInformation
}

func newSymbolCollector(matcher SymbolMatcher, style SymbolStyle, query string) *symbolCollector {
	var m matcherFunc
	switch matcher {
	case SymbolFuzzy:
		m = parseQuery(query)
	case SymbolCaseSensitive:
		m = func(s string) float64 {
			if strings.Contains(s, query) {
				return 1
			}
			return 0
		}
	case SymbolCaseInsensitive:
		q := strings.ToLower(query)
		m = func(s string) float64 {
			if strings.Contains(strings.ToLower(s), q) {
				return 1
			}
			return 0
		}
	default:
		panic(fmt.Errorf("unknown symbol matcher: %v", matcher))
	}
	var s symbolizer
	switch style {
	case DynamicSymbols:
		s = dynamicSymbolMatch
	case FullyQualifiedSymbols:
		s = fullyQualifiedSymbolMatch
	case PackageQualifiedSymbols:
		s = packageSymbolMatch
	default:
		panic(fmt.Errorf("unknown symbol style: %v", style))
	}
	return &symbolCollector{
		matcher:    m,
		symbolizer: s,
	}
}

// parseQuery parses a field-separated symbol query, extracting the special
// characters listed below, and returns a matcherFunc corresponding to the AND
// of all field queries.
//
// Special characters:
//   ^  match exact prefix
//   $  match exact suffix
//   '  match exact
//
// In all three of these special queries, matches are 'smart-cased', meaning
// they are case sensitive if the symbol query contains any upper-case
// characters, and case insensitive otherwise.
func parseQuery(q string) matcherFunc {
	fields := strings.Fields(q)
	if len(fields) == 0 {
		return func(string) float64 { return 0 }
	}
	var funcs []matcherFunc
	for _, field := range fields {
		var f matcherFunc
		switch {
		case strings.HasPrefix(field, "^"):
			prefix := field[1:]
			f = smartCase(prefix, func(s string) float64 {
				if strings.HasPrefix(s, prefix) {
					return 1
				}
				return 0
			})
		case strings.HasPrefix(field, "'"):
			exact := field[1:]
			f = smartCase(exact, func(s string) float64 {
				if strings.Contains(s, exact) {
					return 1
				}
				return 0
			})
		case strings.HasSuffix(field, "$"):
			suffix := field[0 : len(field)-1]
			f = smartCase(suffix, func(s string) float64 {
				if strings.HasSuffix(s, suffix) {
					return 1
				}
				return 0
			})
		default:
			fm := fuzzy.NewMatcher(field)
			f = func(s string) float64 {
				return float64(fm.Score(s))
			}
		}
		funcs = append(funcs, f)
	}
	return comboMatcher(funcs).match
}

// smartCase returns a matcherFunc that is case-sensitive if q contains any
// upper-case characters, and case-insensitive otherwise.
func smartCase(q string, m matcherFunc) matcherFunc {
	insensitive := strings.ToLower(q) == q
	return func(s string) float64 {
		if insensitive {
			s = strings.ToLower(s)
		}
		return m(s)
	}
}

type comboMatcher []matcherFunc

func (c comboMatcher) match(s string) float64 {
	score := 1.0
	for _, f := range c {
		score *= f(s)
	}
	return score
}

// walk walks views, gathers symbols, and returns the results.
func (sc *symbolCollector) walk(ctx context.Context, views []View) (_ []protocol.SymbolInformation, err error) {
	toWalk, release, err := sc.collectPackages(ctx, views)
	defer release()
	if err != nil {
		return nil, err
	}
	// Make sure we only walk files once (we might see them more than once due to
	// build constraints).
	seen := make(map[span.URI]bool)
	for _, pv := range toWalk {
		sc.current = pv
		for _, pgf := range pv.pkg.CompiledGoFiles() {
			if seen[pgf.URI] {
				continue
			}
			seen[pgf.URI] = true
			sc.curFile = pgf
			sc.walkFilesDecls(pgf.File.Decls)
		}
	}
	return sc.results(), nil
}

func (sc *symbolCollector) results() []protocol.SymbolInformation {
	var res []protocol.SymbolInformation
	for _, si := range sc.res {
		if si.score <= 0 {
			return res
		}
		res = append(res, si.asProtocolSymbolInformation())
	}
	return res
}

// collectPackages gathers the packages we are going to inspect for symbols.
// This pre-step is required in order to filter out any "duplicate"
// *types.Package. The duplicates arise for packages that have test variants.
// For example, if package mod.com/p has test files, then we will visit two
// packages that have the PkgPath() mod.com/p: the first is the actual package
// mod.com/p, the second is a special version that includes the non-XTest
// _test.go files. If we were to walk both of of these packages, then we would
// get duplicate matching symbols and we would waste effort. Therefore where
// test variants exist we walk those (because they include any symbols defined
// in non-XTest _test.go files).
//
// One further complication is that even after this filtering, packages between
// views might not be "identical" because they can be built using different
// build constraints (via the "env" config option).
//
// Therefore on a per view basis we first build up a map of package path ->
// *types.Package preferring the test variants if they exist. Then we merge the
// results between views, de-duping by *types.Package.
func (sc *symbolCollector) collectPackages(ctx context.Context, views []View) ([]*pkgView, func(), error) {
	gathered := make(map[string]map[*types.Package]*pkgView)
	var releaseFuncs []func()
	release := func() {
		for _, releaseFunc := range releaseFuncs {
			releaseFunc()
		}
	}
	var toWalk []*pkgView
	for _, v := range views {
		seen := make(map[string]*pkgView)
		snapshot, release := v.Snapshot(ctx)
		releaseFuncs = append(releaseFuncs, release)
		knownPkgs, err := snapshot.KnownPackages(ctx)
		if err != nil {
			return nil, release, err
		}
		workspacePackages, err := snapshot.WorkspacePackages(ctx)
		if err != nil {
			return nil, release, err
		}
		isWorkspacePkg := make(map[Package]bool)
		for _, wp := range workspacePackages {
			isWorkspacePkg[wp] = true
		}
		var forTests []*pkgView
		for _, pkg := range knownPkgs {
			toAdd := &pkgView{
				pkg:         pkg,
				snapshot:    snapshot,
				isWorkspace: isWorkspacePkg[pkg],
			}
			// Defer test packages, so that they overwrite seen for this package
			// path.
			if pkg.ForTest() != "" {
				forTests = append(forTests, toAdd)
			} else {
				seen[pkg.PkgPath()] = toAdd
			}
		}
		for _, pkg := range forTests {
			seen[pkg.pkg.PkgPath()] = pkg
		}
		for _, pkg := range seen {
			pm, ok := gathered[pkg.pkg.PkgPath()]
			if !ok {
				pm = make(map[*types.Package]*pkgView)
				gathered[pkg.pkg.PkgPath()] = pm
			}
			pm[pkg.pkg.GetTypes()] = pkg
		}
	}
	for _, pm := range gathered {
		for _, pkg := range pm {
			toWalk = append(toWalk, pkg)
		}
	}
	// Now sort for stability of results. We order by
	// (pkgView.isWorkspace, pkgView.p.ID())
	sort.Slice(toWalk, func(i, j int) bool {
		lhs := toWalk[i]
		rhs := toWalk[j]
		switch {
		case lhs.isWorkspace == rhs.isWorkspace:
			return lhs.pkg.ID() < rhs.pkg.ID()
		case lhs.isWorkspace:
			return true
		default:
			return false
		}
	})
	return toWalk, release, nil
}

func (sc *symbolCollector) walkFilesDecls(decls []ast.Decl) {
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			fn := decl.Name.Name
			kind := protocol.Function
			if decl.Recv != nil {
				kind = protocol.Method
				switch typ := decl.Recv.List[0].Type.(type) {
				case *ast.StarExpr:
					fn = typ.X.(*ast.Ident).Name + "." + fn
				case *ast.Ident:
					fn = typ.Name + "." + fn
				}
			}
			sc.match(fn, kind, decl.Name)
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					target := spec.Name.Name
					sc.match(target, typeToKind(sc.current.pkg.GetTypesInfo().TypeOf(spec.Type)), spec.Name)
					switch st := spec.Type.(type) {
					case *ast.StructType:
						for _, field := range st.Fields.List {
							sc.walkField(field, protocol.Field, target)
						}
					case *ast.InterfaceType:
						for _, field := range st.Methods.List {
							kind := protocol.Method
							if len(field.Names) == 0 {
								kind = protocol.Interface
							}
							sc.walkField(field, kind, target)
						}
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						target := name.Name
						kind := protocol.Variable
						if decl.Tok == token.CONST {
							kind = protocol.Constant
						}
						sc.match(target, kind, name)
					}
				}
			}
		}
	}
}

func (sc *symbolCollector) walkField(field *ast.Field, kind protocol.SymbolKind, prefix string) {
	if len(field.Names) == 0 {
		name := types.ExprString(field.Type)
		target := prefix + "." + name
		sc.match(target, kind, field)
		return
	}
	for _, name := range field.Names {
		target := prefix + "." + name.Name
		sc.match(target, kind, name)
	}
}

func typeToKind(typ types.Type) protocol.SymbolKind {
	switch typ := typ.Underlying().(type) {
	case *types.Interface:
		return protocol.Interface
	case *types.Struct:
		return protocol.Struct
	case *types.Signature:
		if typ.Recv() != nil {
			return protocol.Method
		}
		return protocol.Function
	case *types.Named:
		return typeToKind(typ.Underlying())
	case *types.Basic:
		i := typ.Info()
		switch {
		case i&types.IsNumeric != 0:
			return protocol.Number
		case i&types.IsBoolean != 0:
			return protocol.Boolean
		case i&types.IsString != 0:
			return protocol.String
		}
	}
	return protocol.Variable
}

// match finds matches and gathers the symbol identified by name, kind and node
// via the symbolCollector's matcher after first de-duping against previously
// seen symbols.
func (sc *symbolCollector) match(name string, kind protocol.SymbolKind, node ast.Node) {
	if !node.Pos().IsValid() || !node.End().IsValid() {
		return
	}

	// Arbitrary factors to apply to the match score for the purpose of
	// downranking results.
	//
	// There is no science behind this, other than the principle that symbols
	// outside of a workspace should be downranked. Adjust as necessary.
	const (
		nonWorkspaceFactor = 0.5
	)
	factor := 1.0
	if !sc.current.isWorkspace {
		factor *= nonWorkspaceFactor
	}
	symbol, score := sc.symbolizer(name, sc.current.pkg, sc.matcher)
	score *= factor
	if score <= sc.res[len(sc.res)-1].score {
		return
	}

	mrng := NewMappedRange(sc.current.snapshot.FileSet(), sc.curFile.Mapper, node.Pos(), node.End())
	rng, err := mrng.Range()
	if err != nil {
		return
	}
	si := symbolInformation{
		score:     score,
		name:      name,
		symbol:    symbol,
		container: sc.current.pkg.PkgPath(),
		kind:      kind,
		location: protocol.Location{
			URI:   protocol.URIFromSpanURI(mrng.URI()),
			Range: rng,
		},
	}
	insertAt := sort.Search(len(sc.res), func(i int) bool {
		return sc.res[i].score < score
	})
	if insertAt < len(sc.res)-1 {
		copy(sc.res[insertAt+1:], sc.res[insertAt:len(sc.res)-1])
	}
	sc.res[insertAt] = si
}

// pkgView holds information related to a package that we are going to walk.
type pkgView struct {
	pkg         Package
	snapshot    Snapshot
	isWorkspace bool
}

// symbolInformation is a cut-down version of protocol.SymbolInformation that
// allows struct values of this type to be used as map keys.
type symbolInformation struct {
	score     float64
	name      string
	symbol    string
	container string
	kind      protocol.SymbolKind
	location  protocol.Location
}

// asProtocolSymbolInformation converts s to a protocol.SymbolInformation value.
//
// TODO: work out how to handle tags if/when they are needed.
func (s symbolInformation) asProtocolSymbolInformation() protocol.SymbolInformation {
	return protocol.SymbolInformation{
		Name:          s.symbol,
		Kind:          s.kind,
		Location:      s.location,
		ContainerName: s.container,
	}
}
