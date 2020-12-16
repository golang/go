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
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/source/completion"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

const (
	overlayFileSuffix = ".overlay"
	goldenFileSuffix  = ".golden"
	inFileSuffix      = ".in"
	summaryFile       = "summary.txt"
	testModule        = "golang.org/x/tools/internal/lsp"
)

var UpdateGolden = flag.Bool("golden", false, "Update golden files")

type CallHierarchy map[span.Span]*CallHierarchyResult
type CodeLens map[span.URI][]protocol.CodeLens
type Diagnostics map[span.URI][]*source.Diagnostic
type CompletionItems map[token.Pos]*completion.CompletionItem
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
type SemanticTokens []span.Span
type SuggestedFixes map[span.Span][]string
type FunctionExtractions map[span.Span]span.Span
type Definitions map[span.Span]Definition
type Implementations map[span.Span][]span.Span
type Highlights map[span.Span][]span.Span
type References map[span.Span][]span.Span
type Renames map[span.Span]string
type PrepareRenames map[span.Span]*source.PrepareItem
type Symbols map[span.URI][]protocol.DocumentSymbol
type SymbolsChildren map[string][]protocol.DocumentSymbol
type SymbolInformation map[span.Span]protocol.SymbolInformation
type WorkspaceSymbols map[WorkspaceSymbolsTestType]map[span.URI][]string
type Signatures map[span.Span]*protocol.SignatureHelp
type Links map[span.URI][]Link

type Data struct {
	Config                   packages.Config
	Exported                 *packagestest.Exported
	CallHierarchy            CallHierarchy
	CodeLens                 CodeLens
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
	SemanticTokens           SemanticTokens
	SuggestedFixes           SuggestedFixes
	FunctionExtractions      FunctionExtractions
	Definitions              Definitions
	Implementations          Implementations
	Highlights               Highlights
	References               References
	Renames                  Renames
	PrepareRenames           PrepareRenames
	Symbols                  Symbols
	symbolsChildren          SymbolsChildren
	symbolInformation        SymbolInformation
	WorkspaceSymbols         WorkspaceSymbols
	Signatures               Signatures
	Links                    Links

	t         testing.TB
	fragments map[string]string
	dir       string
	golden    map[string]*Golden
	mode      string

	ModfileFlagAvailable bool

	mappersMu sync.Mutex
	mappers   map[span.URI]*protocol.ColumnMapper
}

type Tests interface {
	CallHierarchy(*testing.T, span.Span, *CallHierarchyResult)
	CodeLens(*testing.T, span.URI, []protocol.CodeLens)
	Diagnostics(*testing.T, span.URI, []*source.Diagnostic)
	Completion(*testing.T, span.Span, Completion, CompletionItems)
	CompletionSnippet(*testing.T, span.Span, CompletionSnippet, bool, CompletionItems)
	UnimportedCompletion(*testing.T, span.Span, Completion, CompletionItems)
	DeepCompletion(*testing.T, span.Span, Completion, CompletionItems)
	FuzzyCompletion(*testing.T, span.Span, Completion, CompletionItems)
	CaseSensitiveCompletion(*testing.T, span.Span, Completion, CompletionItems)
	RankCompletion(*testing.T, span.Span, Completion, CompletionItems)
	FoldingRanges(*testing.T, span.Span)
	Format(*testing.T, span.Span)
	Import(*testing.T, span.Span)
	SemanticTokens(*testing.T, span.Span)
	SuggestedFix(*testing.T, span.Span, []string, int)
	FunctionExtraction(*testing.T, span.Span, span.Span)
	Definition(*testing.T, span.Span, Definition)
	Implementation(*testing.T, span.Span, []span.Span)
	Highlight(*testing.T, span.Span, []span.Span)
	References(*testing.T, span.Span, []span.Span)
	Rename(*testing.T, span.Span, string)
	PrepareRename(*testing.T, span.Span, *source.PrepareItem)
	Symbols(*testing.T, span.URI, []protocol.DocumentSymbol)
	WorkspaceSymbols(*testing.T, span.URI, string, WorkspaceSymbolsTestType)
	SignatureHelp(*testing.T, span.Span, *protocol.SignatureHelp)
	Link(*testing.T, span.URI, []Link)
}

type Definition struct {
	Name      string
	IsType    bool
	OnlyHover bool
	Src, Def  span.Span
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

	// CaseSensitive tests case sensitive completion.
	CompletionCaseSensitive

	// CompletionRank candidates in test must be valid and in the right relative order.
	CompletionRank
)

type WorkspaceSymbolsTestType int

const (
	// Default runs the standard workspace symbols tests.
	WorkspaceSymbolsDefault = WorkspaceSymbolsTestType(iota)

	// Fuzzy tests workspace symbols with fuzzy matching.
	WorkspaceSymbolsFuzzy

	// CaseSensitive tests workspace symbols with case sensitive.
	WorkspaceSymbolsCaseSensitive
)

type Completion struct {
	CompletionItems []token.Pos
}

type CompletionSnippet struct {
	CompletionItem     token.Pos
	PlainSnippet       string
	PlaceholderSnippet string
}

type CallHierarchyResult struct {
	IncomingCalls, OutgoingCalls []protocol.CallHierarchyItem
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

func DefaultOptions(o *source.Options) {
	o.SupportedCodeActions = map[source.FileKind]map[protocol.CodeActionKind]bool{
		source.Go: {
			protocol.SourceOrganizeImports: true,
			protocol.QuickFix:              true,
			protocol.RefactorRewrite:       true,
			protocol.RefactorExtract:       true,
			protocol.SourceFixAll:          true,
		},
		source.Mod: {
			protocol.SourceOrganizeImports: true,
		},
		source.Sum: {},
	}
	o.UserOptions.Codelenses[source.CommandTest.Name] = true
	o.HoverKind = source.SynopsisDocumentation
	o.InsertTextFormat = protocol.SnippetTextFormat
	o.CompletionBudget = time.Minute
	o.HierarchicalDocumentSymbolSupport = true
	o.ExperimentalWorkspaceModule = true
	o.SemanticTokens = true
}

func RunTests(t *testing.T, dataDir string, includeMultiModule bool, f func(*testing.T, *Data)) {
	t.Helper()
	modes := []string{"Modules", "GOPATH"}
	if includeMultiModule {
		modes = append(modes, "MultiModule")
	}
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			t.Helper()
			if mode == "MultiModule" {
				// Some bug in 1.12 breaks reading markers, and it's not worth figuring out.
				testenv.NeedsGo1Point(t, 13)
			}
			datum := load(t, mode, dataDir)
			f(t, datum)
		})
	}
}

func load(t testing.TB, mode string, dir string) *Data {
	t.Helper()

	datum := &Data{
		CallHierarchy:            make(CallHierarchy),
		CodeLens:                 make(CodeLens),
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
		Implementations:          make(Implementations),
		Highlights:               make(Highlights),
		References:               make(References),
		Renames:                  make(Renames),
		PrepareRenames:           make(PrepareRenames),
		SuggestedFixes:           make(SuggestedFixes),
		FunctionExtractions:      make(FunctionExtractions),
		Symbols:                  make(Symbols),
		symbolsChildren:          make(SymbolsChildren),
		symbolInformation:        make(SymbolInformation),
		WorkspaceSymbols:         make(WorkspaceSymbols),
		Signatures:               make(Signatures),
		Links:                    make(Links),

		t:         t,
		dir:       dir,
		fragments: map[string]string{},
		golden:    map[string]*Golden{},
		mode:      mode,
		mappers:   map[span.URI]*protocol.ColumnMapper{},
	}

	if !*UpdateGolden {
		summary := filepath.Join(filepath.FromSlash(dir), summaryFile+goldenFileSuffix)
		if _, err := os.Stat(summary); os.IsNotExist(err) {
			t.Fatalf("could not find golden file summary.txt in %#v", dir)
		}
		archive, err := txtar.ParseFile(summary)
		if err != nil {
			t.Fatalf("could not read golden file %v/%v: %v", dir, summary, err)
		}
		datum.golden[summaryFile] = &Golden{
			Filename: summary,
			Archive:  archive,
		}
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
			datum.golden[trimmed] = &Golden{
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
	}
	switch mode {
	case "Modules":
		datum.Exported = packagestest.Export(t, packagestest.Modules, modules)
	case "GOPATH":
		datum.Exported = packagestest.Export(t, packagestest.GOPATH, modules)
	case "MultiModule":
		files := map[string]interface{}{}
		for k, v := range modules[0].Files {
			files[filepath.Join("testmodule", k)] = v
		}
		modules[0].Files = files

		overlays := map[string][]byte{}
		for k, v := range modules[0].Overlay {
			overlays[filepath.Join("testmodule", k)] = v
		}
		modules[0].Overlay = overlays

		golden := map[string]*Golden{}
		for k, v := range datum.golden {
			if k == summaryFile {
				golden[k] = v
			} else {
				golden[filepath.Join("testmodule", k)] = v
			}
		}
		datum.golden = golden

		datum.Exported = packagestest.Export(t, packagestest.Modules, modules)
	default:
		panic("unknown mode " + mode)
	}

	for _, m := range modules {
		for fragment := range m.Files {
			filename := datum.Exported.File(m.Name, fragment)
			datum.fragments[filename] = fragment
		}
	}

	// Turn off go/packages debug logging.
	datum.Exported.Config.Logf = nil
	datum.Config.Logf = nil

	// Merge the exported.Config with the view.Config.
	datum.Config = *datum.Exported.Config
	datum.Config.Fset = token.NewFileSet()
	datum.Config.Context = Context(nil)
	datum.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		panic("ParseFile should not be called")
	}

	// Do a first pass to collect special markers for completion and workspace symbols.
	if err := datum.Exported.Expect(map[string]interface{}{
		"item": func(name string, r packagestest.Range, _ []string) {
			datum.Exported.Mark(name, r)
		},
		"symbol": func(name string, r packagestest.Range, _ []string) {
			datum.Exported.Mark(name, r)
		},
	}); err != nil {
		t.Fatal(err)
	}

	// Collect any data that needs to be used by subsequent tests.
	if err := datum.Exported.Expect(map[string]interface{}{
		"codelens":        datum.collectCodeLens,
		"diag":            datum.collectDiagnostics,
		"item":            datum.collectCompletionItems,
		"complete":        datum.collectCompletions(CompletionDefault),
		"unimported":      datum.collectCompletions(CompletionUnimported),
		"deep":            datum.collectCompletions(CompletionDeep),
		"fuzzy":           datum.collectCompletions(CompletionFuzzy),
		"casesensitive":   datum.collectCompletions(CompletionCaseSensitive),
		"rank":            datum.collectCompletions(CompletionRank),
		"snippet":         datum.collectCompletionSnippets,
		"fold":            datum.collectFoldingRanges,
		"format":          datum.collectFormats,
		"import":          datum.collectImports,
		"semantic":        datum.collectSemanticTokens,
		"godef":           datum.collectDefinitions,
		"implementations": datum.collectImplementations,
		"typdef":          datum.collectTypeDefinitions,
		"hover":           datum.collectHoverDefinitions,
		"highlight":       datum.collectHighlights,
		"refs":            datum.collectReferences,
		"rename":          datum.collectRenames,
		"prepare":         datum.collectPrepareRenames,
		"symbol":          datum.collectSymbols,
		"signature":       datum.collectSignatures,
		"link":            datum.collectLinks,
		"suggestedfix":    datum.collectSuggestedFixes,
		"extractfunc":     datum.collectFunctionExtractions,
		"incomingcalls":   datum.collectIncomingCalls,
		"outgoingcalls":   datum.collectOutgoingCalls,
	}); err != nil {
		t.Fatal(err)
	}
	for _, symbols := range datum.Symbols {
		for i := range symbols {
			children := datum.symbolsChildren[symbols[i].Name]
			symbols[i].Children = children
		}
	}
	// Collect names for the entries that require golden files.
	if err := datum.Exported.Expect(map[string]interface{}{
		"godef":                        datum.collectDefinitionNames,
		"hover":                        datum.collectDefinitionNames,
		"workspacesymbol":              datum.collectWorkspaceSymbols(WorkspaceSymbolsDefault),
		"workspacesymbolfuzzy":         datum.collectWorkspaceSymbols(WorkspaceSymbolsFuzzy),
		"workspacesymbolcasesensitive": datum.collectWorkspaceSymbols(WorkspaceSymbolsCaseSensitive),
	}); err != nil {
		t.Fatal(err)
	}
	if mode == "MultiModule" {
		if err := os.Rename(filepath.Join(datum.Config.Dir, "go.mod"), filepath.Join(datum.Config.Dir, "testmodule/go.mod")); err != nil {
			t.Fatal(err)
		}
	}

	return datum
}

func Run(t *testing.T, tests Tests, data *Data) {
	t.Helper()
	checkData(t, data)

	eachCompletion := func(t *testing.T, cases map[span.Span][]Completion, test func(*testing.T, span.Span, Completion, CompletionItems)) {
		t.Helper()

		for src, exp := range cases {
			for i, e := range exp {
				t.Run(SpanName(src)+"_"+strconv.Itoa(i), func(t *testing.T) {
					t.Helper()
					if strings.Contains(t.Name(), "cgo") {
						testenv.NeedsTool(t, "cgo")
					}
					if strings.Contains(t.Name(), "declarecgo") {
						testenv.NeedsGo1Point(t, 15)
					}
					test(t, src, e, data.CompletionItems)
				})
			}

		}
	}

	t.Run("CallHierarchy", func(t *testing.T) {
		t.Helper()
		for spn, callHierarchyResult := range data.CallHierarchy {
			t.Run(SpanName(spn), func(t *testing.T) {
				t.Helper()
				tests.CallHierarchy(t, spn, callHierarchyResult)
			})
		}
	})

	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		eachCompletion(t, data.Completions, tests.Completion)
	})

	t.Run("CompletionSnippets", func(t *testing.T) {
		t.Helper()
		for _, placeholders := range []bool{true, false} {
			for src, expecteds := range data.CompletionSnippets {
				for i, expected := range expecteds {
					name := SpanName(src) + "_" + strconv.Itoa(i+1)
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

	t.Run("CodeLens", func(t *testing.T) {
		t.Helper()
		for uri, want := range data.CodeLens {
			// Check if we should skip this URI if the -modfile flag is not available.
			if shouldSkip(data, uri) {
				continue
			}
			t.Run(uriName(uri), func(t *testing.T) {
				t.Helper()
				tests.CodeLens(t, uri, want)
			})
		}
	})

	t.Run("Diagnostics", func(t *testing.T) {
		t.Helper()
		for uri, want := range data.Diagnostics {
			// Check if we should skip this URI if the -modfile flag is not available.
			if shouldSkip(data, uri) {
				continue
			}
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
				tests.FoldingRanges(t, spn)
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

	t.Run("SemanticTokens", func(t *testing.T) {
		t.Helper()
		for _, spn := range data.SemanticTokens {
			t.Run(uriName(spn.URI()), func(t *testing.T) {
				t.Helper()
				tests.SemanticTokens(t, spn)
			})
		}
	})

	t.Run("SuggestedFix", func(t *testing.T) {
		t.Helper()
		for spn, actionKinds := range data.SuggestedFixes {
			// Check if we should skip this spn if the -modfile flag is not available.
			if shouldSkip(data, spn.URI()) {
				continue
			}
			t.Run(SpanName(spn), func(t *testing.T) {
				t.Helper()
				tests.SuggestedFix(t, spn, actionKinds, 1)
			})
		}
	})

	t.Run("FunctionExtraction", func(t *testing.T) {
		t.Helper()
		for start, end := range data.FunctionExtractions {
			// Check if we should skip this spn if the -modfile flag is not available.
			if shouldSkip(data, start.URI()) {
				continue
			}
			t.Run(SpanName(start), func(t *testing.T) {
				t.Helper()
				tests.FunctionExtraction(t, start, end)
			})
		}
	})

	t.Run("Definition", func(t *testing.T) {
		t.Helper()
		for spn, d := range data.Definitions {
			t.Run(SpanName(spn), func(t *testing.T) {
				t.Helper()
				if strings.Contains(t.Name(), "cgo") {
					testenv.NeedsTool(t, "cgo")
				}
				if strings.Contains(t.Name(), "declarecgo") {
					testenv.NeedsGo1Point(t, 15)
				}
				tests.Definition(t, spn, d)
			})
		}
	})

	t.Run("Implementation", func(t *testing.T) {
		t.Helper()
		for spn, m := range data.Implementations {
			t.Run(SpanName(spn), func(t *testing.T) {
				t.Helper()
				tests.Implementation(t, spn, m)
			})
		}
	})

	t.Run("Highlight", func(t *testing.T) {
		t.Helper()
		for pos, locations := range data.Highlights {
			t.Run(SpanName(pos), func(t *testing.T) {
				t.Helper()
				tests.Highlight(t, pos, locations)
			})
		}
	})

	t.Run("References", func(t *testing.T) {
		t.Helper()
		for src, itemList := range data.References {
			t.Run(SpanName(src), func(t *testing.T) {
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
			t.Run(SpanName(src), func(t *testing.T) {
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

	t.Run("WorkspaceSymbols", func(t *testing.T) {
		t.Helper()

		for _, typ := range []WorkspaceSymbolsTestType{
			WorkspaceSymbolsDefault,
			WorkspaceSymbolsCaseSensitive,
			WorkspaceSymbolsFuzzy,
		} {
			for uri, cases := range data.WorkspaceSymbols[typ] {
				for _, query := range cases {
					name := query
					if name == "" {
						name = "EmptyQuery"
					}
					t.Run(name, func(t *testing.T) {
						t.Helper()
						tests.WorkspaceSymbols(t, uri, query, typ)
					})
				}
			}
		}

	})

	t.Run("SignatureHelp", func(t *testing.T) {
		t.Helper()
		for spn, expectedSignature := range data.Signatures {
			t.Run(SpanName(spn), func(t *testing.T) {
				t.Helper()
				tests.SignatureHelp(t, spn, expectedSignature)
			})
		}
	})

	t.Run("Link", func(t *testing.T) {
		t.Helper()
		for uri, wantLinks := range data.Links {
			// If we are testing GOPATH, then we do not want links with the versions
			// attached (pkg.go.dev/repoa/moda@v1.1.0/pkg), unless the file is a
			// go.mod, then we can skip it altogether.
			if data.Exported.Exporter == packagestest.GOPATH {
				if strings.HasSuffix(uri.Filename(), ".mod") {
					continue
				}
				re := regexp.MustCompile(`@v\d+\.\d+\.[\w-]+`)
				for i, link := range wantLinks {
					wantLinks[i].Target = re.ReplaceAllString(link.Target, "")
				}
			}
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

	countCodeLens := func(c map[span.URI][]protocol.CodeLens) (count int) {
		for _, want := range c {
			count += len(want)
		}
		return count
	}

	countWorkspaceSymbols := func(c map[WorkspaceSymbolsTestType]map[span.URI][]string) (count int) {
		for _, typs := range c {
			for _, queries := range typs {
				count += len(queries)
			}
		}
		return count
	}

	fmt.Fprintf(buf, "CallHierarchyCount = %v\n", len(data.CallHierarchy))
	fmt.Fprintf(buf, "CodeLensCount = %v\n", countCodeLens(data.CodeLens))
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
	fmt.Fprintf(buf, "SemanticTokenCount = %v\n", len(data.SemanticTokens))
	fmt.Fprintf(buf, "SuggestedFixCount = %v\n", len(data.SuggestedFixes))
	fmt.Fprintf(buf, "FunctionExtractionCount = %v\n", len(data.FunctionExtractions))
	fmt.Fprintf(buf, "DefinitionsCount = %v\n", definitionCount)
	fmt.Fprintf(buf, "TypeDefinitionsCount = %v\n", typeDefinitionCount)
	fmt.Fprintf(buf, "HighlightsCount = %v\n", len(data.Highlights))
	fmt.Fprintf(buf, "ReferencesCount = %v\n", len(data.References))
	fmt.Fprintf(buf, "RenamesCount = %v\n", len(data.Renames))
	fmt.Fprintf(buf, "PrepareRenamesCount = %v\n", len(data.PrepareRenames))
	fmt.Fprintf(buf, "SymbolsCount = %v\n", len(data.Symbols))
	fmt.Fprintf(buf, "WorkspaceSymbolsCount = %v\n", countWorkspaceSymbols(data.WorkspaceSymbols))
	fmt.Fprintf(buf, "SignaturesCount = %v\n", len(data.Signatures))
	fmt.Fprintf(buf, "LinksCount = %v\n", linksCount)
	fmt.Fprintf(buf, "ImplementationsCount = %v\n", len(data.Implementations))

	want := string(data.Golden("summary", summaryFile, func() ([]byte, error) {
		return buf.Bytes(), nil
	}))
	got := buf.String()
	if want != got {
		t.Errorf("test summary does not match:\n%s", Diff(t, want, got))
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
	if len(file.Data) == 0 {
		return file.Data
	}
	return file.Data[:len(file.Data)-1] // drop the trailing \n
}

func (data *Data) collectCodeLens(spn span.Span, title, cmd string) {
	if _, ok := data.CodeLens[spn.URI()]; !ok {
		data.CodeLens[spn.URI()] = []protocol.CodeLens{}
	}
	m, err := data.Mapper(spn.URI())
	if err != nil {
		return
	}
	rng, err := m.Range(spn)
	if err != nil {
		return
	}
	data.CodeLens[spn.URI()] = append(data.CodeLens[spn.URI()], protocol.CodeLens{
		Range: rng,
		Command: protocol.Command{
			Title:   title,
			Command: cmd,
		},
	})
}

func (data *Data) collectDiagnostics(spn span.Span, msgSource, msg, msgSeverity string) {
	if _, ok := data.Diagnostics[spn.URI()]; !ok {
		data.Diagnostics[spn.URI()] = []*source.Diagnostic{}
	}
	m, err := data.Mapper(spn.URI())
	if err != nil {
		return
	}
	rng, err := m.Range(spn)
	if err != nil {
		return
	}
	severity := protocol.SeverityError
	switch msgSeverity {
	case "error":
		severity = protocol.SeverityError
	case "warning":
		severity = protocol.SeverityWarning
	case "hint":
		severity = protocol.SeverityHint
	case "information":
		severity = protocol.SeverityInformation
	}
	// This is not the correct way to do this, but it seems excessive to do the full conversion here.
	want := &source.Diagnostic{
		Range:    rng,
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
	case CompletionCaseSensitive:
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
		loc := data.Exported.ExpectFileSet.Position(pos)
		data.t.Fatalf("%s:%d: @item expects at least 3 args, got %d",
			loc.Filename, loc.Line, len(args))
	}
	label, detail, kind := args[0], args[1], args[2]
	var documentation string
	if len(args) == 4 {
		documentation = args[3]
	}
	data.CompletionItems[pos] = &completion.CompletionItem{
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

func (data *Data) collectSemanticTokens(spn span.Span) {
	data.SemanticTokens = append(data.SemanticTokens, spn)
}

func (data *Data) collectSuggestedFixes(spn span.Span, actionKind string) {
	if _, ok := data.SuggestedFixes[spn]; !ok {
		data.SuggestedFixes[spn] = []string{}
	}
	data.SuggestedFixes[spn] = append(data.SuggestedFixes[spn], actionKind)
}

func (data *Data) collectFunctionExtractions(start span.Span, end span.Span) {
	if _, ok := data.FunctionExtractions[start]; !ok {
		data.FunctionExtractions[start] = end
	}
}

func (data *Data) collectDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src: src,
		Def: target,
	}
}

func (data *Data) collectImplementations(src span.Span, targets []span.Span) {
	data.Implementations[src] = targets
}

func (data *Data) collectIncomingCalls(src span.Span, calls []span.Span) {
	for _, call := range calls {
		m, err := data.Mapper(call.URI())
		if err != nil {
			data.t.Fatal(err)
		}
		rng, err := m.Range(call)
		if err != nil {
			data.t.Fatal(err)
		}
		// we're only comparing protocol.range
		if data.CallHierarchy[src] != nil {
			data.CallHierarchy[src].IncomingCalls = append(data.CallHierarchy[src].IncomingCalls,
				protocol.CallHierarchyItem{
					URI:   protocol.DocumentURI(call.URI()),
					Range: rng,
				})
		} else {
			data.CallHierarchy[src] = &CallHierarchyResult{
				IncomingCalls: []protocol.CallHierarchyItem{
					{URI: protocol.DocumentURI(call.URI()), Range: rng},
				},
			}
		}
	}
}

func (data *Data) collectOutgoingCalls(src span.Span, calls []span.Span) {
	for _, call := range calls {
		m, err := data.Mapper(call.URI())
		if err != nil {
			data.t.Fatal(err)
		}
		rng, err := m.Range(call)
		if err != nil {
			data.t.Fatal(err)
		}
		// we're only comparing protocol.range
		if data.CallHierarchy[src] != nil {
			data.CallHierarchy[src].OutgoingCalls = append(data.CallHierarchy[src].OutgoingCalls,
				protocol.CallHierarchyItem{
					URI:   protocol.DocumentURI(call.URI()),
					Range: rng,
				})
		} else {
			data.CallHierarchy[src] = &CallHierarchyResult{
				OutgoingCalls: []protocol.CallHierarchyItem{
					{URI: protocol.DocumentURI(call.URI()), Range: rng},
				},
			}
		}
	}
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

func (data *Data) collectHighlights(src span.Span, expected []span.Span) {
	// Declaring a highlight in a test file: @highlight(src, expected1, expected2)
	data.Highlights[src] = append(data.Highlights[src], expected...)
}

func (data *Data) collectReferences(src span.Span, expected []span.Span) {
	data.References[src] = expected
}

func (data *Data) collectRenames(src span.Span, newText string) {
	data.Renames[src] = newText
}

func (data *Data) collectPrepareRenames(src span.Span, rng span.Range, placeholder string) {
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

// collectSymbols is responsible for collecting @symbol annotations.
func (data *Data) collectSymbols(name string, spn span.Span, kind string, parentName string, siName string) {
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

	// Reuse @symbol in the workspace symbols tests.
	si := protocol.SymbolInformation{
		Name: siName,
		Kind: sym.Kind,
		Location: protocol.Location{
			URI:   protocol.URIFromSpanURI(spn.URI()),
			Range: sym.SelectionRange,
		},
	}
	data.symbolInformation[spn] = si
}

func (data *Data) collectWorkspaceSymbols(typ WorkspaceSymbolsTestType) func(*expect.Note, string) {
	return func(note *expect.Note, query string) {
		if data.WorkspaceSymbols[typ] == nil {
			data.WorkspaceSymbols[typ] = make(map[span.URI][]string)
		}
		pos := data.Exported.ExpectFileSet.Position(note.Pos)
		uri := span.URIFromPath(pos.Filename)
		data.WorkspaceSymbols[typ][uri] = append(data.WorkspaceSymbols[typ][uri], query)
	}
}

func (data *Data) collectSignatures(spn span.Span, signature string, activeParam int64) {
	data.Signatures[spn] = &protocol.SignatureHelp{
		Signatures: []protocol.SignatureInformation{
			{
				Label: signature,
			},
		},
		ActiveParameter: float64(activeParam),
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

func SpanName(spn span.Span) string {
	return fmt.Sprintf("%v_%v_%v", uriName(spn.URI()), spn.Start().Line(), spn.Start().Column())
}

func CopyFolderToTempDir(folder string) (string, error) {
	if _, err := os.Stat(folder); err != nil {
		return "", err
	}
	dst, err := ioutil.TempDir("", "modfile_test")
	if err != nil {
		return "", err
	}
	fds, err := ioutil.ReadDir(folder)
	if err != nil {
		return "", err
	}
	for _, fd := range fds {
		srcfp := filepath.Join(folder, fd.Name())
		stat, err := os.Stat(srcfp)
		if err != nil {
			return "", err
		}
		if !stat.Mode().IsRegular() {
			return "", fmt.Errorf("cannot copy non regular file %s", srcfp)
		}
		contents, err := ioutil.ReadFile(srcfp)
		if err != nil {
			return "", err
		}
		if err := ioutil.WriteFile(filepath.Join(dst, fd.Name()), contents, stat.Mode()); err != nil {
			return "", err
		}
	}
	return dst, nil
}

func shouldSkip(data *Data, uri span.URI) bool {
	if data.ModfileFlagAvailable {
		return false
	}
	// If the -modfile flag is not available, then we do not want to run
	// any tests on the go.mod file.
	if strings.HasSuffix(uri.Filename(), ".mod") {
		return true
	}
	// If the -modfile flag is not available, then we do not want to test any
	// uri that contains "go mod tidy".
	m, err := data.Mapper(uri)
	return err == nil && strings.Contains(string(m.Content), ", \"go mod tidy\",")
}
