// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/types"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/fuzzy"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

// Symbol holds a precomputed symbol value. Note: we avoid using the
// protocol.SymbolInformation struct here in order to reduce the size of each
// symbol.
type Symbol struct {
	Name  string
	Kind  protocol.SymbolKind
	Range protocol.Range
}

// maxSymbols defines the maximum number of symbol results that should ever be
// sent in response to a client.
const maxSymbols = 100

// WorkspaceSymbols matches symbols across all views using the given query,
// according to the match semantics parameterized by matcherType and style.
//
// The workspace symbol method is defined in the spec as follows:
//
//   The workspace symbol request is sent from the client to the server to
//   list project-wide symbols matching the query string.
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

// A matcherFunc returns the index and score of a symbol match.
//
// See the comment for symbolCollector for more information.
type matcherFunc func(chunks []string) (int, float64)

// A symbolizer returns the best symbol match for a name with pkg, according to
// some heuristic. The symbol name is passed as the slice nameParts of logical
// name pieces. For example, for myType.field the caller can pass either
// []string{"myType.field"} or []string{"myType.", "field"}.
//
// See the comment for symbolCollector for more information.
type symbolizer func(name string, pkg Metadata, m matcherFunc) ([]string, float64)

func fullyQualifiedSymbolMatch(name string, pkg Metadata, matcher matcherFunc) ([]string, float64) {
	_, score := dynamicSymbolMatch(name, pkg, matcher)
	if score > 0 {
		return []string{pkg.PackagePath(), ".", name}, score
	}
	return nil, 0
}

func dynamicSymbolMatch(name string, pkg Metadata, matcher matcherFunc) ([]string, float64) {
	var score float64

	endsInPkgName := strings.HasSuffix(pkg.PackagePath(), pkg.PackageName())

	// If the package path does not end in the package name, we need to check the
	// package-qualified symbol as an extra pass first.
	if !endsInPkgName {
		pkgQualified := []string{pkg.PackageName(), ".", name}
		idx, score := matcher(pkgQualified)
		nameStart := len(pkg.PackageName()) + 1
		if score > 0 {
			// If our match is contained entirely within the unqualified portion,
			// just return that.
			if idx >= nameStart {
				return []string{name}, score
			}
			// Lower the score for matches that include the package name.
			return pkgQualified, score * 0.8
		}
	}

	// Now try matching the fully qualified symbol.
	fullyQualified := []string{pkg.PackagePath(), ".", name}
	idx, score := matcher(fullyQualified)

	// As above, check if we matched just the unqualified symbol name.
	nameStart := len(pkg.PackagePath()) + 1
	if idx >= nameStart {
		return []string{name}, score
	}

	// If our package path ends in the package name, we'll have skipped the
	// initial pass above, so check if we matched just the package-qualified
	// name.
	if endsInPkgName && idx >= 0 {
		pkgStart := len(pkg.PackagePath()) - len(pkg.PackageName())
		if idx >= pkgStart {
			return []string{pkg.PackageName(), ".", name}, score
		}
	}

	// Our match was not contained within the unqualified or package qualified
	// symbol. Return the fully qualified symbol but discount the score.
	return fullyQualified, score * 0.6
}

func packageSymbolMatch(name string, pkg Metadata, matcher matcherFunc) ([]string, float64) {
	qualified := []string{pkg.PackageName(), ".", name}
	if _, s := matcher(qualified); s > 0 {
		return qualified, s
	}
	return nil, 0
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
	// These types parameterize the symbol-matching pass.
	matchers   []matcherFunc
	symbolizer symbolizer

	seen map[span.URI]bool
	symbolStore
}

func newSymbolCollector(matcher SymbolMatcher, style SymbolStyle, query string) *symbolCollector {
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
	sc := &symbolCollector{symbolizer: s}
	sc.matchers = make([]matcherFunc, runtime.GOMAXPROCS(-1))
	for i := range sc.matchers {
		sc.matchers[i] = buildMatcher(matcher, query)
	}
	return sc
}

func buildMatcher(matcher SymbolMatcher, query string) matcherFunc {
	switch matcher {
	case SymbolFuzzy:
		return parseQuery(query, newFuzzyMatcher)
	case SymbolFastFuzzy:
		return parseQuery(query, func(query string) matcherFunc {
			return fuzzy.NewSymbolMatcher(query).Match
		})
	case SymbolCaseSensitive:
		return matchExact(query)
	case SymbolCaseInsensitive:
		q := strings.ToLower(query)
		exact := matchExact(q)
		wrapper := []string{""}
		return func(chunks []string) (int, float64) {
			s := strings.Join(chunks, "")
			wrapper[0] = strings.ToLower(s)
			return exact(wrapper)
		}
	}
	panic(fmt.Errorf("unknown symbol matcher: %v", matcher))
}

func newFuzzyMatcher(query string) matcherFunc {
	fm := fuzzy.NewMatcher(query)
	return func(chunks []string) (int, float64) {
		score := float64(fm.ScoreChunks(chunks))
		ranges := fm.MatchedRanges()
		if len(ranges) > 0 {
			return ranges[0], score
		}
		return -1, score
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
func parseQuery(q string, newMatcher func(string) matcherFunc) matcherFunc {
	fields := strings.Fields(q)
	if len(fields) == 0 {
		return func([]string) (int, float64) { return -1, 0 }
	}
	var funcs []matcherFunc
	for _, field := range fields {
		var f matcherFunc
		switch {
		case strings.HasPrefix(field, "^"):
			prefix := field[1:]
			f = smartCase(prefix, func(chunks []string) (int, float64) {
				s := strings.Join(chunks, "")
				if strings.HasPrefix(s, prefix) {
					return 0, 1
				}
				return -1, 0
			})
		case strings.HasPrefix(field, "'"):
			exact := field[1:]
			f = smartCase(exact, matchExact(exact))
		case strings.HasSuffix(field, "$"):
			suffix := field[0 : len(field)-1]
			f = smartCase(suffix, func(chunks []string) (int, float64) {
				s := strings.Join(chunks, "")
				if strings.HasSuffix(s, suffix) {
					return len(s) - len(suffix), 1
				}
				return -1, 0
			})
		default:
			f = newMatcher(field)
		}
		funcs = append(funcs, f)
	}
	if len(funcs) == 1 {
		return funcs[0]
	}
	return comboMatcher(funcs).match
}

func matchExact(exact string) matcherFunc {
	return func(chunks []string) (int, float64) {
		s := strings.Join(chunks, "")
		if idx := strings.LastIndex(s, exact); idx >= 0 {
			return idx, 1
		}
		return -1, 0
	}
}

// smartCase returns a matcherFunc that is case-sensitive if q contains any
// upper-case characters, and case-insensitive otherwise.
func smartCase(q string, m matcherFunc) matcherFunc {
	insensitive := strings.ToLower(q) == q
	wrapper := []string{""}
	return func(chunks []string) (int, float64) {
		s := strings.Join(chunks, "")
		if insensitive {
			s = strings.ToLower(s)
		}
		wrapper[0] = s
		return m(wrapper)
	}
}

type comboMatcher []matcherFunc

func (c comboMatcher) match(chunks []string) (int, float64) {
	score := 1.0
	first := 0
	for _, f := range c {
		idx, s := f(chunks)
		if idx < first {
			first = idx
		}
		score *= s
	}
	return first, score
}

func (sc *symbolCollector) walk(ctx context.Context, views []View) ([]protocol.SymbolInformation, error) {
	// Use the root view URIs for determining (lexically) whether a uri is in any
	// open workspace.
	var roots []string
	for _, v := range views {
		roots = append(roots, strings.TrimRight(string(v.Folder()), "/"))
	}

	results := make(chan *symbolStore)
	matcherlen := len(sc.matchers)
	files := make(map[span.URI]symbolFile)

	for _, v := range views {
		snapshot, release := v.Snapshot(ctx)
		defer release()
		psyms, err := snapshot.Symbols(ctx)
		if err != nil {
			return nil, err
		}

		filters := v.Options().DirectoryFilters
		folder := filepath.ToSlash(v.Folder().Filename())
		for uri, syms := range psyms {
			norm := filepath.ToSlash(uri.Filename())
			nm := strings.TrimPrefix(norm, folder)
			if FiltersDisallow(nm, filters) {
				continue
			}
			// Only scan each file once.
			if _, ok := files[uri]; ok {
				continue
			}
			mds, err := snapshot.MetadataForFile(ctx, uri)
			if err != nil {
				event.Error(ctx, fmt.Sprintf("missing metadata for %q", uri), err)
				continue
			}
			if len(mds) == 0 {
				// TODO: should use the bug reporting API
				continue
			}
			files[uri] = symbolFile{uri, mds[0], syms}
		}
	}

	var work []symbolFile
	for _, f := range files {
		work = append(work, f)
	}

	// Compute matches concurrently. Each symbolWorker has its own symbolStore,
	// which we merge at the end.
	for i, matcher := range sc.matchers {
		go func(i int, matcher matcherFunc) {
			w := &symbolWorker{
				symbolizer: sc.symbolizer,
				matcher:    matcher,
				ss:         &symbolStore{},
				roots:      roots,
			}
			for j := i; j < len(work); j += matcherlen {
				w.matchFile(work[j])
			}
			results <- w.ss
		}(i, matcher)
	}

	for i := 0; i < matcherlen; i++ {
		ss := <-results
		for _, si := range ss.res {
			sc.store(si)
		}
	}
	return sc.results(), nil
}

// FilterDisallow is code from the body of cache.pathExcludedByFilter in cache/view.go
// Exporting and using that function would cause an import cycle.
// Moving it here and exporting it would leave behind view_test.go.
// (This code is exported and used in the body of cache.pathExcludedByFilter)
func FiltersDisallow(path string, filters []string) bool {
	path = strings.TrimPrefix(path, "/")
	var excluded bool
	for _, filter := range filters {
		op, prefix := filter[0], filter[1:]
		// Non-empty prefixes have to be precise directory matches.
		if prefix != "" {
			prefix = prefix + "/"
			path = path + "/"
		}
		if !strings.HasPrefix(path, prefix) {
			continue
		}
		excluded = op == '-'
	}
	return excluded
}

// symbolFile holds symbol information for a single file.
type symbolFile struct {
	uri  span.URI
	md   Metadata
	syms []Symbol
}

// symbolWorker matches symbols and captures the highest scoring results.
type symbolWorker struct {
	symbolizer symbolizer
	matcher    matcherFunc
	ss         *symbolStore
	roots      []string
}

func (w *symbolWorker) matchFile(i symbolFile) {
	for _, sym := range i.syms {
		symbolParts, score := w.symbolizer(sym.Name, i.md, w.matcher)

		// Check if the score is too low before applying any downranking.
		if w.ss.tooLow(score) {
			continue
		}

		// Factors to apply to the match score for the purpose of downranking
		// results.
		//
		// These numbers were crudely calibrated based on trial-and-error using a
		// small number of sample queries. Adjust as necessary.
		//
		// All factors are multiplicative, meaning if more than one applies they are
		// multiplied together.
		const (
			// nonWorkspaceFactor is applied to symbols outside of any active
			// workspace. Developers are less likely to want to jump to code that they
			// are not actively working on.
			nonWorkspaceFactor = 0.5
			// nonWorkspaceUnexportedFactor is applied to unexported symbols outside of
			// any active workspace. Since one wouldn't usually jump to unexported
			// symbols to understand a package API, they are particularly irrelevant.
			nonWorkspaceUnexportedFactor = 0.5
			// every field or method nesting level to access the field decreases
			// the score by a factor of 1.0 - depth*depthFactor, up to a depth of
			// 3.
			depthFactor = 0.2
		)

		startWord := true
		exported := true
		depth := 0.0
		for _, r := range sym.Name {
			if startWord && !unicode.IsUpper(r) {
				exported = false
			}
			if r == '.' {
				startWord = true
				depth++
			} else {
				startWord = false
			}
		}

		inWorkspace := false
		for _, root := range w.roots {
			if strings.HasPrefix(string(i.uri), root) {
				inWorkspace = true
				break
			}
		}

		// Apply downranking based on workspace position.
		if !inWorkspace {
			score *= nonWorkspaceFactor
			if !exported {
				score *= nonWorkspaceUnexportedFactor
			}
		}

		// Apply downranking based on symbol depth.
		if depth > 3 {
			depth = 3
		}
		score *= 1.0 - depth*depthFactor

		if w.ss.tooLow(score) {
			continue
		}

		si := symbolInformation{
			score:     score,
			symbol:    strings.Join(symbolParts, ""),
			kind:      sym.Kind,
			uri:       i.uri,
			rng:       sym.Range,
			container: i.md.PackagePath(),
		}
		w.ss.store(si)
	}
}

type symbolStore struct {
	res [maxSymbols]symbolInformation
}

// store inserts si into the sorted results, if si has a high enough score.
func (sc *symbolStore) store(si symbolInformation) {
	if sc.tooLow(si.score) {
		return
	}
	insertAt := sort.Search(len(sc.res), func(i int) bool {
		// Sort by score, then symbol length, and finally lexically.
		if sc.res[i].score != si.score {
			return sc.res[i].score < si.score
		}
		if len(sc.res[i].symbol) != len(si.symbol) {
			return len(sc.res[i].symbol) > len(si.symbol)
		}
		return sc.res[i].symbol > si.symbol
	})
	if insertAt < len(sc.res)-1 {
		copy(sc.res[insertAt+1:], sc.res[insertAt:len(sc.res)-1])
	}
	sc.res[insertAt] = si
}

func (sc *symbolStore) tooLow(score float64) bool {
	return score <= sc.res[len(sc.res)-1].score
}

func (sc *symbolStore) results() []protocol.SymbolInformation {
	var res []protocol.SymbolInformation
	for _, si := range sc.res {
		if si.score <= 0 {
			return res
		}
		res = append(res, si.asProtocolSymbolInformation())
	}
	return res
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

// symbolInformation is a cut-down version of protocol.SymbolInformation that
// allows struct values of this type to be used as map keys.
type symbolInformation struct {
	score     float64
	symbol    string
	container string
	kind      protocol.SymbolKind
	uri       span.URI
	rng       protocol.Range
}

// asProtocolSymbolInformation converts s to a protocol.SymbolInformation value.
//
// TODO: work out how to handle tags if/when they are needed.
func (s symbolInformation) asProtocolSymbolInformation() protocol.SymbolInformation {
	return protocol.SymbolInformation{
		Name: s.symbol,
		Kind: s.kind,
		Location: protocol.Location{
			URI:   protocol.URIFromSpanURI(s.uri),
			Range: s.rng,
		},
		ContainerName: s.container,
	}
}
