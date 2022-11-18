// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/types"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/fuzzy"
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
//	The workspace symbol request is sent from the client to the server to
//	list project-wide symbols matching the query string.
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
func WorkspaceSymbols(ctx context.Context, matcher SymbolMatcher, style SymbolStyle, views []View, query string) ([]protocol.SymbolInformation, error) {
	ctx, done := event.Start(ctx, "source.WorkspaceSymbols")
	defer done()
	if query == "" {
		return nil, nil
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

	return collectSymbols(ctx, views, matcher, s, query)
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
//
// The space argument is an empty slice with spare capacity that may be used
// to allocate the result.
type symbolizer func(space []string, name string, pkg *Metadata, m matcherFunc) ([]string, float64)

func fullyQualifiedSymbolMatch(space []string, name string, pkg *Metadata, matcher matcherFunc) ([]string, float64) {
	if _, score := dynamicSymbolMatch(space, name, pkg, matcher); score > 0 {
		return append(space, string(pkg.PkgPath), ".", name), score
	}
	return nil, 0
}

func dynamicSymbolMatch(space []string, name string, pkg *Metadata, matcher matcherFunc) ([]string, float64) {
	if IsCommandLineArguments(pkg.ID) {
		// command-line-arguments packages have a non-sensical package path, so
		// just use their package name.
		return packageSymbolMatch(space, name, pkg, matcher)
	}

	var score float64

	endsInPkgName := strings.HasSuffix(string(pkg.PkgPath), string(pkg.Name))

	// If the package path does not end in the package name, we need to check the
	// package-qualified symbol as an extra pass first.
	if !endsInPkgName {
		pkgQualified := append(space, string(pkg.Name), ".", name)
		idx, score := matcher(pkgQualified)
		nameStart := len(pkg.Name) + 1
		if score > 0 {
			// If our match is contained entirely within the unqualified portion,
			// just return that.
			if idx >= nameStart {
				return append(space, name), score
			}
			// Lower the score for matches that include the package name.
			return pkgQualified, score * 0.8
		}
	}

	// Now try matching the fully qualified symbol.
	fullyQualified := append(space, string(pkg.PkgPath), ".", name)
	idx, score := matcher(fullyQualified)

	// As above, check if we matched just the unqualified symbol name.
	nameStart := len(pkg.PkgPath) + 1
	if idx >= nameStart {
		return append(space, name), score
	}

	// If our package path ends in the package name, we'll have skipped the
	// initial pass above, so check if we matched just the package-qualified
	// name.
	if endsInPkgName && idx >= 0 {
		pkgStart := len(pkg.PkgPath) - len(pkg.Name)
		if idx >= pkgStart {
			return append(space, string(pkg.Name), ".", name), score
		}
	}

	// Our match was not contained within the unqualified or package qualified
	// symbol. Return the fully qualified symbol but discount the score.
	return fullyQualified, score * 0.6
}

func packageSymbolMatch(space []string, name string, pkg *Metadata, matcher matcherFunc) ([]string, float64) {
	qualified := append(space, string(pkg.Name), ".", name)
	if _, s := matcher(qualified); s > 0 {
		return qualified, s
	}
	return nil, 0
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
//
//	^  match exact prefix
//	$  match exact suffix
//	'  match exact
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

// collectSymbols calls snapshot.Symbols to walk the syntax trees of
// all files in the views' current snapshots, and returns a sorted,
// scored list of symbols that best match the parameters.
//
// How it matches symbols is parameterized by two interfaces:
//   - A matcherFunc determines how well a string symbol matches a query. It
//     returns a non-negative score indicating the quality of the match. A score
//     of zero indicates no match.
//   - A symbolizer determines how we extract the symbol for an object. This
//     enables the 'symbolStyle' configuration option.
func collectSymbols(ctx context.Context, views []View, matcherType SymbolMatcher, symbolizer symbolizer, query string) ([]protocol.SymbolInformation, error) {

	// Extract symbols from all files.
	var work []symbolFile
	var roots []string
	seen := make(map[span.URI]bool)
	// TODO(adonovan): opt: parallelize this loop? How often is len > 1?
	for _, v := range views {
		snapshot, release := v.Snapshot(ctx)
		defer release()

		// Use the root view URIs for determining (lexically)
		// whether a URI is in any open workspace.
		roots = append(roots, strings.TrimRight(string(v.Folder()), "/"))

		filters := v.Options().DirectoryFilters
		filterer := NewFilterer(filters)
		folder := filepath.ToSlash(v.Folder().Filename())
		for uri, syms := range snapshot.Symbols(ctx) {
			norm := filepath.ToSlash(uri.Filename())
			nm := strings.TrimPrefix(norm, folder)
			if filterer.Disallow(nm) {
				continue
			}
			// Only scan each file once.
			if seen[uri] {
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
			seen[uri] = true
			work = append(work, symbolFile{uri, mds[0], syms})
		}
	}

	// Match symbols in parallel.
	// Each worker has its own symbolStore,
	// which we merge at the end.
	nmatchers := runtime.GOMAXPROCS(-1) // matching is CPU bound
	results := make(chan *symbolStore)
	for i := 0; i < nmatchers; i++ {
		go func(i int) {
			matcher := buildMatcher(matcherType, query)
			store := new(symbolStore)
			// Assign files to workers in round-robin fashion.
			for j := i; j < len(work); j += nmatchers {
				matchFile(store, symbolizer, matcher, roots, work[j])
			}
			results <- store
		}(i)
	}

	// Gather and merge results as they arrive.
	var unified symbolStore
	for i := 0; i < nmatchers; i++ {
		store := <-results
		for _, syms := range store.res {
			unified.store(syms)
		}
	}
	return unified.results(), nil
}

type Filterer struct {
	// Whether a filter is excluded depends on the operator (first char of the raw filter).
	// Slices filters and excluded then should have the same length.
	filters  []*regexp.Regexp
	excluded []bool
}

// NewFilterer computes regular expression form of all raw filters
func NewFilterer(rawFilters []string) *Filterer {
	var f Filterer
	for _, filter := range rawFilters {
		filter = path.Clean(filepath.ToSlash(filter))
		// TODO(dungtuanle): fix: validate [+-] prefix.
		op, prefix := filter[0], filter[1:]
		// convertFilterToRegexp adds "/" at the end of prefix to handle cases where a filter is a prefix of another filter.
		// For example, it prevents [+foobar, -foo] from excluding "foobar".
		f.filters = append(f.filters, convertFilterToRegexp(filepath.ToSlash(prefix)))
		f.excluded = append(f.excluded, op == '-')
	}

	return &f
}

// Disallow return true if the path is excluded from the filterer's filters.
func (f *Filterer) Disallow(path string) bool {
	// Ensure trailing but not leading slash.
	path = strings.TrimPrefix(path, "/")
	if !strings.HasSuffix(path, "/") {
		path += "/"
	}

	// TODO(adonovan): opt: iterate in reverse and break at first match.
	excluded := false
	for i, filter := range f.filters {
		if filter.MatchString(path) {
			excluded = f.excluded[i] // last match wins
		}
	}
	return excluded
}

// convertFilterToRegexp replaces glob-like operator substrings in a string file path to their equivalent regex forms.
// Supporting glob-like operators:
//   - **: match zero or more complete path segments
func convertFilterToRegexp(filter string) *regexp.Regexp {
	if filter == "" {
		return regexp.MustCompile(".*")
	}
	var ret strings.Builder
	ret.WriteString("^")
	segs := strings.Split(filter, "/")
	for _, seg := range segs {
		// Inv: seg != "" since path is clean.
		if seg == "**" {
			ret.WriteString(".*")
		} else {
			ret.WriteString(regexp.QuoteMeta(seg))
		}
		ret.WriteString("/")
	}
	pattern := ret.String()

	// Remove unnecessary "^.*" prefix, which increased
	// BenchmarkWorkspaceSymbols time by ~20% (even though
	// filter CPU time increased by only by ~2.5%) when the
	// default filter was changed to "**/node_modules".
	pattern = strings.TrimPrefix(pattern, "^.*")

	return regexp.MustCompile(pattern)
}

// symbolFile holds symbol information for a single file.
type symbolFile struct {
	uri  span.URI
	md   *Metadata
	syms []Symbol
}

// matchFile scans a symbol file and adds matching symbols to the store.
func matchFile(store *symbolStore, symbolizer symbolizer, matcher matcherFunc, roots []string, i symbolFile) {
	space := make([]string, 0, 3)
	for _, sym := range i.syms {
		symbolParts, score := symbolizer(space, sym.Name, i.md, matcher)

		// Check if the score is too low before applying any downranking.
		if store.tooLow(score) {
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
		for _, root := range roots {
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

		if store.tooLow(score) {
			continue
		}

		si := symbolInformation{
			score:     score,
			symbol:    strings.Join(symbolParts, ""),
			kind:      sym.Kind,
			uri:       i.uri,
			rng:       sym.Range,
			container: string(i.md.PkgPath),
		}
		store.store(si)
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
