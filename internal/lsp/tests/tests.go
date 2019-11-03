// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tests exports functionality to be used across a variety of gopls tests.
package tests

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"io/ioutil"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/txtar"
)

const (
	overlayFileSuffix = ".overlay"
	goldenFileSuffix  = ".golden"
	inFileSuffix      = ".in"
	testModule        = "golang.org/x/tools/internal/lsp"
)

var UpdateGolden = flag.Bool("golden", false, "Update golden files")

type Diagnostics map[span.URI][]source.Diagnostic
type CompletionItems map[token.Pos]*source.CompletionItem
type Completions map[span.Span][]Completion
type CompletionSnippets map[span.Span][]CompletionSnippet
type UnimportedCompletions map[span.Span][]Completion
type DeepCompletions map[span.Span][]Completion
type FuzzyCompletions map[span.Span][]Completion
type CaseSensitiveCompletions map[span.Span][]Completion
type RankCompletions map[span.Span][]Completion
type FoldingRanges []span.Span
type Formats []span.Span
type Imports []span.Span
type SuggestedFixes []span.Span
type Definitions map[span.Span]Definition
type Implementationses map[span.Span]Implementations
type Highlights map[string][]span.Span
type References map[span.Span][]span.Span
type Renames map[span.Span]string
type PrepareRenames map[span.Span]*source.PrepareItem
type Symbols map[span.URI][]protocol.DocumentSymbol
type SymbolsChildren map[string][]protocol.DocumentSymbol
type Signatures map[span.Span]*source.SignatureInformation
type Links map[span.URI][]Link

type Data struct {
	Config                   packages.Config
	Exported                 *packagestest.Exported
	Diagnostics              Diagnostics
	CompletionItems          CompletionItems
	Completions              Completions
	CompletionSnippets       CompletionSnippets
	UnimportedCompletions    UnimportedCompletions
	DeepCompletions          DeepCompletions
	FuzzyCompletions         FuzzyCompletions
	CaseSensitiveCompletions CaseSensitiveCompletions
	RankCompletions          RankCompletions
	FoldingRanges            FoldingRanges
	Formats                  Formats
	Imports                  Imports
	SuggestedFixes           SuggestedFixes
	Definitions              Definitions
	Implementationses        Implementationses
	Highlights               Highlights
	References               References
	Renames                  Renames
	PrepareRenames           PrepareRenames
	Symbols                  Symbols
	symbolsChildren          SymbolsChildren
	Signatures               Signatures
	Links                    Links

	t         testing.TB
	fragments map[string]string
	dir       string
	golden    map[string]*Golden

	mappersMu sync.Mutex
	mappers   map[span.URI]*protocol.ColumnMapper
}

type Tests interface {
	Diagnostics(*testing.T, span.URI, []source.Diagnostic)
	Completion(*testing.T, span.Span, Completion, CompletionItems)
	CompletionSnippet(*testing.T, span.Span, CompletionSnippet, bool, CompletionItems)
	UnimportedCompletion(*testing.T, span.Span, Completion, CompletionItems)
	DeepCompletion(*testing.T, span.Span, Completion, CompletionItems)
	FuzzyCompletion(*testing.T, span.Span, Completion, CompletionItems)
	CaseSensitiveCompletion(*testing.T, span.Span, Completion, CompletionItems)
	RankCompletion(*testing.T, span.Span, Completion, CompletionItems)
	FoldingRange(*testing.T, span.Span)
	Format(*testing.T, span.Span)
	Import(*testing.T, span.Span)
	SuggestedFix(*testing.T, span.Span)
	Definition(*testing.T, span.Span, Definition)
	Implementation(*testing.T, span.Span, Implementations)
	Highlight(*testing.T, string, []span.Span)
	References(*testing.T, span.Span, []span.Span)
	Rename(*testing.T, span.Span, string)
	PrepareRename(*testing.T, span.Span, *source.PrepareItem)
	Symbols(*testing.T, span.URI, []protocol.DocumentSymbol)
	SignatureHelp(*testing.T, span.Span, *source.SignatureInformation)
	Link(*testing.T, span.URI, []Link)
}

type Definition struct {
	Name      string
	IsType    bool
	OnlyHover bool
	Src, Def  span.Span
}

type Implementations struct {
	Src             span.Span
	Implementations []span.Span
}

type CompletionTestType int

const (
	// Default runs the standard completion tests.
	CompletionDefault = CompletionTestType(iota)

	// Unimported tests the autocompletion of unimported packages.
	CompletionUnimported

	// Deep tests deep completion.
	CompletionDeep

	// Fuzzy tests deep completion and fuzzy matching.
	CompletionFuzzy

	// CaseSensitive tests case sensitive completion
	CompletionCaseSensitve

	// CompletionRank candidates in test must be valid and in the right relative order.
	CompletionRank
)

type Completion struct {
	CompletionItems []token.Pos
}

type CompletionSnippet struct {
	CompletionItem     token.Pos
	PlainSnippet       string
	PlaceholderSnippet string
}

type Link struct {
	Src          span.Span
	Target       string
	NotePosition token.Position
}

type Golden struct {
	Filename string
	Archive  *txtar.Archive
	Modified bool
}

func Context(t testing.TB) context.Context {
	return context.Background()
}

func DefaultOptions() source.Options {
	o := source.DefaultOptions
	o.SupportedCodeActions = map[source.FileKind]map[protocol.CodeActionKind]bool{
		source.Go: {
			protocol.SourceOrganizeImports: true,
			protocol.QuickFix:              true,
		},
		source.Mod: {},
		source.Sum: {},
	}
	o.HoverKind = source.SynopsisDocumentation
	o.InsertTextFormat = protocol.SnippetTextFormat
	return o
}

func Load(t testing.TB, exporter packagestest.Exporter, dir string) *Data {
	t.Helper()

	data := &Data{
		Diagnostics:              make(Diagnostics),
		CompletionItems:          make(CompletionItems),
		Completions:              make(Completions),
		CompletionSnippets:       make(CompletionSnippets),
		UnimportedCompletions:    make(UnimportedCompletions),
		DeepCompletions:          make(DeepCompletions),
		FuzzyCompletions:         make(FuzzyCompletions),
		RankCompletions:          make(RankCompletions),
		CaseSensitiveCompletions: make(CaseSensitiveCompletions),
		Definitions:              make(Definitions),
		Implementationses:        make(Implementationses),
		Highlights:               make(Highlights),
		References:               make(References),
		Renames:                  make(Renames),
		PrepareRenames:           make(PrepareRenames),
		Symbols:                  make(Symbols),
		symbolsChildren:          make(SymbolsChildren),
		Signatures:               make(Signatures),
		Links:                    make(Links),

		t:         t,
		dir:       dir,
		fragments: map[string]string{},
		golden:    map[string]*Golden{},
		mappers:   map[span.URI]*protocol.ColumnMapper{},
	}

	files := packagestest.MustCopyFileTree(dir)
	overlays := map[string][]byte{}
	for fragment, operation := range files {
		if trimmed := strings.TrimSuffix(fragment, goldenFileSuffix); trimmed != fragment {
			delete(files, fragment)
			goldFile := filepath.Join(dir, fragment)
			archive, err := txtar.ParseFile(goldFile)
			if err != nil {
				t.Fatalf("could not read golden file %v: %v", fragment, err)
			}
			data.golden[trimmed] = &Golden{
				Filename: goldFile,
				Archive:  archive,
			}
		} else if trimmed := strings.TrimSuffix(fragment, inFileSuffix); trimmed != fragment {
			delete(files, fragment)
			files[trimmed] = operation
		} else if index := strings.Index(fragment, overlayFileSuffix); index >= 0 {
			delete(files, fragment)
			partial := fragment[:index] + fragment[index+len(overlayFileSuffix):]
			contents, err := ioutil.ReadFile(filepath.Join(dir, fragment))
			if err != nil {
				t.Fatal(err)
			}
			overlays[partial] = contents
		}
	}
	modules := []packagestest.Module{
		{
			Name:    testModule,
			Files:   files,
			Overlay: overlays,
		},
		{
			Name: "example.com/extramodule",
			Files: map[string]interface{}{
				"pkg/x.go": "package pkg\n",
			},
		},
	}
	data.Exported = packagestest.Export(t, exporter, modules)
	for fragment := range files {
		filename := data.Exported.File(testModule, fragment)
		data.fragments[filename] = fragment
	}

	// Turn off go/packages debug logging.
	data.Exported.Config.Logf = nil
	data.Config.Logf = nil

	// Merge the exported.Config with the view.Config.
	data.Config = *data.Exported.Config
	data.Config.Fset = token.NewFileSet()
	data.Config.Context = Context(nil)
	data.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		panic("ParseFile should not be called")
	}

	// Do a first pass to collect special markers for completion.
	if err := data.Exported.Expect(map[string]interface{}{
		"item": func(name string, r packagestest.Range, _ []string) {
			data.Exported.Mark(name, r)
		},
	}); err != nil {
		t.Fatal(err)
	}

	// Collect any data that needs to be used by subsequent tests.
	if err := data.Exported.Expect(map[string]interface{}{
		"diag":            data.collectDiagnostics,
		"item":            data.collectCompletionItems,
		"complete":        data.collectCompletions(CompletionDefault),
		"unimported":      data.collectCompletions(CompletionUnimported),
		"deep":            data.collectCompletions(CompletionDeep),
		"fuzzy":           data.collectCompletions(CompletionFuzzy),
		"casesensitive":   data.collectCompletions(CompletionCaseSensitve),
		"rank":            data.collectCompletions(CompletionRank),
		"snippet":         data.collectCompletionSnippets,
		"fold":            data.collectFoldingRanges,
		"format":          data.collectFormats,
		"import":          data.collectImports,
		"godef":           data.collectDefinitions,
		"implementations": data.collectImplementations,
		"typdef":          data.collectTypeDefinitions,
		"hover":           data.collectHoverDefinitions,
		"highlight":       data.collectHighlights,
		"refs":            data.collectReferences,
		"rename":          data.collectRenames,
		"prepare":         data.collectPrepareRenames,
		"symbol":          data.collectSymbols,
		"signature":       data.collectSignatures,
		"link":            data.collectLinks,
		"suggestedfix":    data.collectSuggestedFixes,
	}); err != nil {
		t.Fatal(err)
	}
	for _, symbols := range data.Symbols {
		for i := range symbols {
			children := data.symbolsChildren[symbols[i].Name]
			symbols[i].Children = children
		}
	}
	// Collect names for the entries that require golden files.
	if err := data.Exported.Expect(map[string]interface{}{
		"godef": data.collectDefinitionNames,
		"hover": data.collectDefinitionNames,
	}); err != nil {
		t.Fatal(err)
	}
	return data
}

func Run(t *testing.T, tests Tests, data *Data) {
	t.Helper()
	checkData(t, data)

	eachCompletion := func(t *testing.T, cases map[span.Span][]Completion, test func(*testing.T, span.Span, Completion, CompletionItems)) {
		t.Helper()

		for src, exp := range cases {
			for i, e := range exp {
				t.Run(spanName(src)+"_"+strconv.Itoa(i), func(t *testing.T) {
					t.Helper()
					test(t, src, e, data.CompletionItems)
				})
			}

		}
	}

	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.Completions, tests.Completion)
	})

	t.Run("CompletionSnippets", func(t *testing.T) {
		t.Helper()
		for _, placeholders := range []bool{true, false} {
			for src, expecteds := range data.CompletionSnippets {
				for i, expected := range expecteds {
					name := spanName(src) + "_" + strconv.Itoa(i+1)
					if placeholders {
						name += "_placeholders"
					}

					t.Run(name, func(t *testing.T) {
						t.Helper()
						tests.CompletionSnippet(t, src, expected, placeholders, data.CompletionItems)
					})
				}
			}
		}
	})

	t.Run("UnimportedCompletion", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.UnimportedCompletions, tests.UnimportedCompletion)
	})

	t.Run("DeepCompletion", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.DeepCompletions, tests.DeepCompletion)
	})

	t.Run("FuzzyCompletion", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.FuzzyCompletions, tests.FuzzyCompletion)
	})

	t.Run("CaseSensitiveCompletion", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.CaseSensitiveCompletions, tests.CaseSensitiveCompletion)
	})

	t.Run("RankCompletions", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.RankCompletions, tests.RankCompletion)
	})

	t.Run("Diagnostics", func(t *testing.T) {
		t.Helper()
		for uri, want := range data.Diagnostics {
			t.Run(uriName(uri), func(t *testing.T) {
				t.Helper()
				tests.Diagnostics(t, uri, want)
			})
		}
	})

	t.Run("FoldingRange", func(t *testing.T) {
		t.Helper()
		for _, spn := range data.FoldingRanges {
			t.Run(uriName(spn.URI()), func(t *testing.T) {
				t.Helper()
				tests.FoldingRange(t, spn)
			})
		}
	})

	t.Run("Format", func(t *testing.T) {
		t.Helper()
		for _, spn := range data.Formats {
			t.Run(uriName(spn.URI()), func(t *testing.T) {
				t.Helper()
				tests.Format(t, spn)
			})
		}
	})

	t.Run("Import", func(t *testing.T) {
		t.Helper()
		for _, spn := range data.Imports {
			t.Run(uriName(spn.URI()), func(t *testing.T) {
				t.Helper()
				tests.Import(t, spn)
			})
		}
	})

	t.Run("SuggestedFix", func(t *testing.T) {
		t.Helper()
		for _, spn := range data.SuggestedFixes {
			t.Run(spanName(spn), func(t *testing.T) {
				t.Helper()
				tests.SuggestedFix(t, spn)
			})
		}
	})

	t.Run("Definition", func(t *testing.T) {
		t.Helper()
		for spn, d := range data.Definitions {
			t.Run(spanName(spn), func(t *testing.T) {
				t.Helper()
				tests.Definition(t, spn, d)
			})
		}
	})

	t.Run("Implementation", func(t *testing.T) {
		t.Helper()
		for spn, m := range data.Implementationses {
			t.Run(spanName(spn), func(t *testing.T) {
				t.Helper()
				tests.Implementation(t, spn, m)
			})
		}
	})

	t.Run("Highlight", func(t *testing.T) {
		t.Helper()
		for name, locations := range data.Highlights {
			t.Run(name, func(t *testing.T) {
				t.Helper()
				tests.Highlight(t, name, locations)
			})
		}
	})

	t.Run("References", func(t *testing.T) {
		t.Helper()
		for src, itemList := range data.References {
			t.Run(spanName(src), func(t *testing.T) {
				t.Helper()
				tests.References(t, src, itemList)
			})
		}
	})

	t.Run("Renames", func(t *testing.T) {
		t.Helper()
		for spn, newText := range data.Renames {
			t.Run(uriName(spn.URI())+"_"+newText, func(t *testing.T) {
				t.Helper()
				tests.Rename(t, spn, newText)
			})
		}
	})

	t.Run("PrepareRenames", func(t *testing.T) {
		t.Helper()
		for src, want := range data.PrepareRenames {
			t.Run(spanName(src), func(t *testing.T) {
				t.Helper()
				tests.PrepareRename(t, src, want)
			})
		}
	})

	t.Run("Symbols", func(t *testing.T) {
		t.Helper()
		for uri, expectedSymbols := range data.Symbols {
			t.Run(uriName(uri), func(t *testing.T) {
				t.Helper()
				tests.Symbols(t, uri, expectedSymbols)
			})
		}
	})

	t.Run("SignatureHelp", func(t *testing.T) {
		t.Helper()
		for spn, expectedSignature := range data.Signatures {
			t.Run(spanName(spn), func(t *testing.T) {
				t.Helper()
				tests.SignatureHelp(t, spn, expectedSignature)
			})
		}
	})

	t.Run("Link", func(t *testing.T) {
		t.Helper()
		for uri, wantLinks := range data.Links {
			t.Run(uriName(uri), func(t *testing.T) {
				t.Helper()
				tests.Link(t, uri, wantLinks)
			})
		}
	})

	if *UpdateGolden {
		for _, golden := range data.golden {
			if !golden.Modified {
				continue
			}
			sort.Slice(golden.Archive.Files, func(i, j int) bool {
				return golden.Archive.Files[i].Name < golden.Archive.Files[j].Name
			})
			if err := ioutil.WriteFile(golden.Filename, txtar.Format(golden.Archive), 0666); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func checkData(t *testing.T, data *Data) {
	buf := &bytes.Buffer{}
	diagnosticsCount := 0
	for _, want := range data.Diagnostics {
		diagnosticsCount += len(want)
	}
	linksCount := 0
	for _, want := range data.Links {
		linksCount += len(want)
	}
	definitionCount := 0
	typeDefinitionCount := 0
	for _, d := range data.Definitions {
		if d.IsType {
			typeDefinitionCount++
		} else {
			definitionCount++
		}
	}

	snippetCount := 0
	for _, want := range data.CompletionSnippets {
		snippetCount += len(want)
	}

	countCompletions := func(c map[span.Span][]Completion) (count int) {
		for _, want := range c {
			count += len(want)
		}
		return count
	}

	fmt.Fprintf(buf, "CompletionsCount = %v\n", countCompletions(data.Completions))
	fmt.Fprintf(buf, "CompletionSnippetCount = %v\n", snippetCount)
	fmt.Fprintf(buf, "UnimportedCompletionsCount = %v\n", countCompletions(data.UnimportedCompletions))
	fmt.Fprintf(buf, "DeepCompletionsCount = %v\n", countCompletions(data.DeepCompletions))
	fmt.Fprintf(buf, "FuzzyCompletionsCount = %v\n", countCompletions(data.FuzzyCompletions))
	fmt.Fprintf(buf, "RankedCompletionsCount = %v\n", countCompletions(data.RankCompletions))
	fmt.Fprintf(buf, "CaseSensitiveCompletionsCount = %v\n", countCompletions(data.CaseSensitiveCompletions))
	fmt.Fprintf(buf, "DiagnosticsCount = %v\n", diagnosticsCount)
	fmt.Fprintf(buf, "FoldingRangesCount = %v\n", len(data.FoldingRanges))
	fmt.Fprintf(buf, "FormatCount = %v\n", len(data.Formats))
	fmt.Fprintf(buf, "ImportCount = %v\n", len(data.Imports))
	fmt.Fprintf(buf, "SuggestedFixCount = %v\n", len(data.SuggestedFixes))
	fmt.Fprintf(buf, "DefinitionsCount = %v\n", definitionCount)
	fmt.Fprintf(buf, "TypeDefinitionsCount = %v\n", typeDefinitionCount)
	fmt.Fprintf(buf, "HighlightsCount = %v\n", len(data.Highlights))
	fmt.Fprintf(buf, "ReferencesCount = %v\n", len(data.References))
	fmt.Fprintf(buf, "RenamesCount = %v\n", len(data.Renames))
	fmt.Fprintf(buf, "PrepareRenamesCount = %v\n", len(data.PrepareRenames))
	fmt.Fprintf(buf, "SymbolsCount = %v\n", len(data.Symbols))
	fmt.Fprintf(buf, "SignaturesCount = %v\n", len(data.Signatures))
	fmt.Fprintf(buf, "LinksCount = %v\n", linksCount)

	want := string(data.Golden("summary", "summary.txt", func() ([]byte, error) {
		return buf.Bytes(), nil
	}))
	got := buf.String()
	if want != got {
		t.Errorf("test summary does not match, want\n%s\ngot:\n%s", want, got)
	}
}

func (data *Data) Mapper(uri span.URI) (*protocol.ColumnMapper, error) {
	data.mappersMu.Lock()
	defer data.mappersMu.Unlock()

	if _, ok := data.mappers[uri]; !ok {
		content, err := data.Exported.FileContents(uri.Filename())
		if err != nil {
			return nil, err
		}
		converter := span.NewContentConverter(uri.Filename(), content)
		data.mappers[uri] = &protocol.ColumnMapper{
			URI:       uri,
			Converter: converter,
			Content:   content,
		}
	}
	return data.mappers[uri], nil
}

func (data *Data) Golden(tag string, target string, update func() ([]byte, error)) []byte {
	data.t.Helper()
	fragment, found := data.fragments[target]
	if !found {
		if filepath.IsAbs(target) {
			data.t.Fatalf("invalid golden file fragment %v", target)
		}
		fragment = target
	}
	golden := data.golden[fragment]
	if golden == nil {
		if !*UpdateGolden {
			data.t.Fatalf("could not find golden file %v: %v", fragment, tag)
		}
		golden = &Golden{
			Filename: filepath.Join(data.dir, fragment+goldenFileSuffix),
			Archive:  &txtar.Archive{},
			Modified: true,
		}
		data.golden[fragment] = golden
	}
	var file *txtar.File
	for i := range golden.Archive.Files {
		f := &golden.Archive.Files[i]
		if f.Name == tag {
			file = f
			break
		}
	}
	if *UpdateGolden {
		if file == nil {
			golden.Archive.Files = append(golden.Archive.Files, txtar.File{
				Name: tag,
			})
			file = &golden.Archive.Files[len(golden.Archive.Files)-1]
		}
		contents, err := update()
		if err != nil {
			data.t.Fatalf("could not update golden file %v: %v", fragment, err)
		}
		file.Data = append(contents, '\n') // add trailing \n for txtar
		golden.Modified = true
	}
	if file == nil {
		data.t.Fatalf("could not find golden contents %v: %v", fragment, tag)
	}
	return file.Data[:len(file.Data)-1] // drop the trailing \n
}

func (data *Data) collectDiagnostics(spn span.Span, msgSource, msg string) {
	if _, ok := data.Diagnostics[spn.URI()]; !ok {
		data.Diagnostics[spn.URI()] = []source.Diagnostic{}
	}
	severity := protocol.SeverityError
	if strings.Contains(string(spn.URI()), "analyzer") {
		severity = protocol.SeverityWarning
	}
	// This is not the correct way to do this,
	// but it seems excessive to do the full conversion here.
	want := source.Diagnostic{
		URI: spn.URI(),
		Range: protocol.Range{
			Start: protocol.Position{
				Line:      float64(spn.Start().Line()) - 1,
				Character: float64(spn.Start().Column()) - 1,
			},
			End: protocol.Position{
				Line:      float64(spn.End().Line()) - 1,
				Character: float64(spn.End().Column()) - 1,
			},
		},
		Severity: severity,
		Source:   msgSource,
		Message:  msg,
	}
	data.Diagnostics[spn.URI()] = append(data.Diagnostics[spn.URI()], want)
}

func (data *Data) collectCompletions(typ CompletionTestType) func(span.Span, []token.Pos) {
	result := func(m map[span.Span][]Completion, src span.Span, expected []token.Pos) {
		m[src] = append(m[src], Completion{
			CompletionItems: expected,
		})
	}
	switch typ {
	case CompletionDeep:
		return func(src span.Span, expected []token.Pos) {
			result(data.DeepCompletions, src, expected)
		}
	case CompletionUnimported:
		return func(src span.Span, expected []token.Pos) {
			result(data.UnimportedCompletions, src, expected)
		}
	case CompletionFuzzy:
		return func(src span.Span, expected []token.Pos) {
			result(data.FuzzyCompletions, src, expected)
		}
	case CompletionRank:
		return func(src span.Span, expected []token.Pos) {
			result(data.RankCompletions, src, expected)
		}
	case CompletionCaseSensitve:
		return func(src span.Span, expected []token.Pos) {
			result(data.CaseSensitiveCompletions, src, expected)
		}
	default:
		return func(src span.Span, expected []token.Pos) {
			result(data.Completions, src, expected)
		}
	}
}

func (data *Data) collectCompletionItems(pos token.Pos, args []string) {
	if len(args) < 3 {
		return
	}
	label, detail, kind := args[0], args[1], args[2]
	var documentation string
	if len(args) == 4 {
		documentation = args[3]
	}
	data.CompletionItems[pos] = &source.CompletionItem{
		Label:         label,
		Detail:        detail,
		Kind:          protocol.ParseCompletionItemKind(kind),
		Documentation: documentation,
	}
}

func (data *Data) collectFoldingRanges(spn span.Span) {
	data.FoldingRanges = append(data.FoldingRanges, spn)
}

func (data *Data) collectFormats(spn span.Span) {
	data.Formats = append(data.Formats, spn)
}

func (data *Data) collectImports(spn span.Span) {
	data.Imports = append(data.Imports, spn)
}

func (data *Data) collectSuggestedFixes(spn span.Span) {
	data.SuggestedFixes = append(data.SuggestedFixes, spn)
}

func (data *Data) collectDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src: src,
		Def: target,
	}
}

func (data *Data) collectImplementations(src, target span.Span) {
	// Add target to the list of expected implementations for src
	imps := data.Implementationses[src]
	imps.Src = src // Src is already set if imps already exists, but then we're setting it to the same thing.
	imps.Implementations = append(imps.Implementations, target)
	data.Implementationses[src] = imps
}

func (data *Data) collectHoverDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src:       src,
		Def:       target,
		OnlyHover: true,
	}
}

func (data *Data) collectTypeDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src:    src,
		Def:    target,
		IsType: true,
	}
}

func (data *Data) collectDefinitionNames(src span.Span, name string) {
	d := data.Definitions[src]
	d.Name = name
	data.Definitions[src] = d
}

func (data *Data) collectHighlights(name string, rng span.Span) {
	data.Highlights[name] = append(data.Highlights[name], rng)
}

func (data *Data) collectReferences(src span.Span, expected []span.Span) {
	data.References[src] = expected
}

func (data *Data) collectRenames(src span.Span, newText string) {
	data.Renames[src] = newText
}

func (data *Data) collectPrepareRenames(src span.Span, rng span.Range, placeholder string) {
	if int(rng.End-rng.Start) != len(placeholder) {
		// If the length of the placeholder and the length of the range do not match,
		// make the range just be the start.
		rng = span.NewRange(rng.FileSet, rng.Start, rng.Start)
	}
	m, err := data.Mapper(src.URI())
	if err != nil {
		data.t.Fatal(err)
	}
	// Convert range to span and then to protocol.Range.
	spn, err := rng.Span()
	if err != nil {
		data.t.Fatal(err)
	}
	prng, err := m.Range(spn)
	if err != nil {
		data.t.Fatal(err)
	}
	data.PrepareRenames[src] = &source.PrepareItem{
		Range: prng,
		Text:  placeholder,
	}
}

func (data *Data) collectSymbols(name string, spn span.Span, kind string, parentName string) {
	m, err := data.Mapper(spn.URI())
	if err != nil {
		data.t.Fatal(err)
	}
	rng, err := m.Range(spn)
	if err != nil {
		data.t.Fatal(err)
	}
	sym := protocol.DocumentSymbol{
		Name:           name,
		Kind:           protocol.ParseSymbolKind(kind),
		SelectionRange: rng,
	}
	if parentName == "" {
		data.Symbols[spn.URI()] = append(data.Symbols[spn.URI()], sym)
	} else {
		data.symbolsChildren[parentName] = append(data.symbolsChildren[parentName], sym)
	}
}

func (data *Data) collectSignatures(spn span.Span, signature string, activeParam int64) {
	data.Signatures[spn] = &source.SignatureInformation{
		Label:           signature,
		ActiveParameter: int(activeParam),
	}
	// Hardcode special case to test the lack of a signature.
	if signature == "" && activeParam == 0 {
		data.Signatures[spn] = nil
	}
}

func (data *Data) collectCompletionSnippets(spn span.Span, item token.Pos, plain, placeholder string) {
	data.CompletionSnippets[spn] = append(data.CompletionSnippets[spn], CompletionSnippet{
		CompletionItem:     item,
		PlainSnippet:       plain,
		PlaceholderSnippet: placeholder,
	})
}

func (data *Data) collectLinks(spn span.Span, link string, note *expect.Note, fset *token.FileSet) {
	position := fset.Position(note.Pos)
	uri := spn.URI()
	data.Links[uri] = append(data.Links[uri], Link{
		Src:          spn,
		Target:       link,
		NotePosition: position,
	})
}

func uriName(uri span.URI) string {
	return filepath.Base(strings.TrimSuffix(uri.Filename(), ".go"))
}

func spanName(spn span.Span) string {
	return fmt.Sprintf("%v_%v_%v", uriName(spn.URI()), spn.Start().Line(), spn.Start().Column())
}
