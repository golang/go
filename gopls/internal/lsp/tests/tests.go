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
	"io"
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
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/completion"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/txtar"
)

const (
	overlayFileSuffix = ".overlay"
	goldenFileSuffix  = ".golden"
	inFileSuffix      = ".in"

	// The module path containing the testdata packages.
	//
	// Warning: the length of this module path matters, as we have bumped up
	// against command-line limitations on windows (golang/go#54800).
	testModule = "golang.org/lsptests"
)

var summaryFile = "summary.txt"

func init() {
	if testenv.Go1Point() >= 21 {
		summaryFile = "summary_go1.21.txt"
	} else if testenv.Go1Point() >= 18 {
		summaryFile = "summary_go1.18.txt"
	}
}

var UpdateGolden = flag.Bool("golden", false, "Update golden files")

// These type names apparently avoid the need to repeat the
// type in the field name and the make() expression.
type CallHierarchy = map[span.Span]*CallHierarchyResult
type CodeLens = map[span.URI][]protocol.CodeLens
type Diagnostics = map[span.URI][]*source.Diagnostic
type CompletionItems = map[token.Pos]*completion.CompletionItem
type Completions = map[span.Span][]Completion
type CompletionSnippets = map[span.Span][]CompletionSnippet
type UnimportedCompletions = map[span.Span][]Completion
type DeepCompletions = map[span.Span][]Completion
type FuzzyCompletions = map[span.Span][]Completion
type CaseSensitiveCompletions = map[span.Span][]Completion
type RankCompletions = map[span.Span][]Completion
type FoldingRanges = []span.Span
type SemanticTokens = []span.Span
type SuggestedFixes = map[span.Span][]SuggestedFix
type MethodExtractions = map[span.Span]span.Span
type Definitions = map[span.Span]Definition
type Highlights = map[span.Span][]span.Span
type Renames = map[span.Span]string
type PrepareRenames = map[span.Span]*source.PrepareItem
type InlayHints = []span.Span
type Signatures = map[span.Span]*protocol.SignatureHelp
type Links = map[span.URI][]Link
type AddImport = map[span.URI]string
type SelectionRanges = []span.Span

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
	SemanticTokens           SemanticTokens
	SuggestedFixes           SuggestedFixes
	MethodExtractions        MethodExtractions
	Definitions              Definitions
	Highlights               Highlights
	Renames                  Renames
	InlayHints               InlayHints
	PrepareRenames           PrepareRenames
	Signatures               Signatures
	Links                    Links
	AddImport                AddImport
	SelectionRanges          SelectionRanges

	fragments map[string]string
	dir       string
	golden    map[string]*Golden
	mode      string

	ModfileFlagAvailable bool

	mappersMu sync.Mutex
	mappers   map[span.URI]*protocol.Mapper
}

// The Tests interface abstracts the LSP-based implementation of the marker
// test operators (such as @codelens) appearing in files beneath ../testdata/.
//
// TODO(adonovan): reduce duplication; see https://github.com/golang/go/issues/54845.
// There is only one implementation (*runner in ../lsp_test.go), so
// we can abolish the interface now.
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
	SemanticTokens(*testing.T, span.Span)
	SuggestedFix(*testing.T, span.Span, []SuggestedFix, int)
	MethodExtraction(*testing.T, span.Span, span.Span)
	Definition(*testing.T, span.Span, Definition)
	Highlight(*testing.T, span.Span, []span.Span)
	InlayHints(*testing.T, span.Span)
	Rename(*testing.T, span.Span, string)
	PrepareRename(*testing.T, span.Span, *source.PrepareItem)
	SignatureHelp(*testing.T, span.Span, *protocol.SignatureHelp)
	Link(*testing.T, span.URI, []Link)
	AddImport(*testing.T, span.URI, string)
	SelectionRanges(*testing.T, span.Span)
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

type SuggestedFix struct {
	ActionKind, Title string
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
		source.Sum:  {},
		source.Work: {},
		source.Tmpl: {},
	}
	o.UserOptions.Codelenses[string(command.Test)] = true
	o.HoverKind = source.SynopsisDocumentation
	o.InsertTextFormat = protocol.SnippetTextFormat
	o.CompletionBudget = time.Minute
	o.HierarchicalDocumentSymbolSupport = true
	o.SemanticTokens = true
	o.InternalOptions.NewDiff = "both"
}

func RunTests(t *testing.T, dataDir string, includeMultiModule bool, f func(*testing.T, *Data)) {
	t.Helper()
	modes := []string{"Modules", "GOPATH"}
	if includeMultiModule {
		modes = append(modes, "MultiModule")
	}
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			datum := load(t, mode, dataDir)
			t.Helper()
			f(t, datum)
		})
	}
}

func load(t testing.TB, mode string, dir string) *Data {
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
		Highlights:               make(Highlights),
		Renames:                  make(Renames),
		PrepareRenames:           make(PrepareRenames),
		SuggestedFixes:           make(SuggestedFixes),
		MethodExtractions:        make(MethodExtractions),
		Signatures:               make(Signatures),
		Links:                    make(Links),
		AddImport:                make(AddImport),

		dir:       dir,
		fragments: map[string]string{},
		golden:    map[string]*Golden{},
		mode:      mode,
		mappers:   map[span.URI]*protocol.Mapper{},
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
	// Prune test cases that exercise generics.
	if !typeparams.Enabled {
		for name := range files {
			if strings.Contains(name, "_generics") {
				delete(files, name)
			}
		}
	}
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
		"codelens":       datum.collectCodeLens,
		"diag":           datum.collectDiagnostics,
		"item":           datum.collectCompletionItems,
		"complete":       datum.collectCompletions(CompletionDefault),
		"unimported":     datum.collectCompletions(CompletionUnimported),
		"deep":           datum.collectCompletions(CompletionDeep),
		"fuzzy":          datum.collectCompletions(CompletionFuzzy),
		"casesensitive":  datum.collectCompletions(CompletionCaseSensitive),
		"rank":           datum.collectCompletions(CompletionRank),
		"snippet":        datum.collectCompletionSnippets,
		"fold":           datum.collectFoldingRanges,
		"semantic":       datum.collectSemanticTokens,
		"godef":          datum.collectDefinitions,
		"typdef":         datum.collectTypeDefinitions,
		"hoverdef":       datum.collectHoverDefinitions,
		"highlight":      datum.collectHighlights,
		"inlayHint":      datum.collectInlayHints,
		"rename":         datum.collectRenames,
		"prepare":        datum.collectPrepareRenames,
		"signature":      datum.collectSignatures,
		"link":           datum.collectLinks,
		"suggestedfix":   datum.collectSuggestedFixes,
		"extractmethod":  datum.collectMethodExtractions,
		"incomingcalls":  datum.collectIncomingCalls,
		"outgoingcalls":  datum.collectOutgoingCalls,
		"addimport":      datum.collectAddImports,
		"selectionrange": datum.collectSelectionRanges,
	}); err != nil {
		t.Fatal(err)
	}

	// Collect names for the entries that require golden files.
	if err := datum.Exported.Expect(map[string]interface{}{
		"godef":    datum.collectDefinitionNames,
		"hoverdef": datum.collectDefinitionNames,
	}); err != nil {
		t.Fatal(err)
	}
	if mode == "MultiModule" {
		if err := moveFile(filepath.Join(datum.Config.Dir, "go.mod"), filepath.Join(datum.Config.Dir, "testmodule/go.mod")); err != nil {
			t.Fatal(err)
		}
	}

	return datum
}

// moveFile moves the file at oldpath to newpath, by renaming if possible
// or copying otherwise.
func moveFile(oldpath, newpath string) (err error) {
	renameErr := os.Rename(oldpath, newpath)
	if renameErr == nil {
		return nil
	}

	src, err := os.Open(oldpath)
	if err != nil {
		return err
	}
	defer func() {
		src.Close()
		if err == nil {
			err = os.Remove(oldpath)
		}
	}()

	perm := os.ModePerm
	fi, err := src.Stat()
	if err == nil {
		perm = fi.Mode().Perm()
	}

	dst, err := os.OpenFile(newpath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return err
	}

	_, err = io.Copy(dst, src)
	if closeErr := dst.Close(); err == nil {
		err = closeErr
	}
	return err
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

	t.Run("MethodExtraction", func(t *testing.T) {
		t.Helper()
		for start, end := range data.MethodExtractions {
			// Check if we should skip this spn if the -modfile flag is not available.
			if shouldSkip(data, start.URI()) {
				continue
			}
			t.Run(SpanName(start), func(t *testing.T) {
				t.Helper()
				tests.MethodExtraction(t, start, end)
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
				tests.Definition(t, spn, d)
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

	t.Run("InlayHints", func(t *testing.T) {
		t.Helper()
		for _, src := range data.InlayHints {
			t.Run(SpanName(src), func(t *testing.T) {
				t.Helper()
				tests.InlayHints(t, src)
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

	t.Run("AddImport", func(t *testing.T) {
		t.Helper()
		for uri, exp := range data.AddImport {
			t.Run(uriName(uri), func(t *testing.T) {
				tests.AddImport(t, uri, exp)
			})
		}
	})

	t.Run("SelectionRanges", func(t *testing.T) {
		t.Helper()
		for _, span := range data.SelectionRanges {
			t.Run(SpanName(span), func(t *testing.T) {
				tests.SelectionRanges(t, span)
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
	fmt.Fprintf(buf, "SemanticTokenCount = %v\n", len(data.SemanticTokens))
	fmt.Fprintf(buf, "SuggestedFixCount = %v\n", len(data.SuggestedFixes))
	fmt.Fprintf(buf, "MethodExtractionCount = %v\n", len(data.MethodExtractions))
	fmt.Fprintf(buf, "DefinitionsCount = %v\n", definitionCount)
	fmt.Fprintf(buf, "TypeDefinitionsCount = %v\n", typeDefinitionCount)
	fmt.Fprintf(buf, "HighlightsCount = %v\n", len(data.Highlights))
	fmt.Fprintf(buf, "InlayHintsCount = %v\n", len(data.InlayHints))
	fmt.Fprintf(buf, "RenamesCount = %v\n", len(data.Renames))
	fmt.Fprintf(buf, "PrepareRenamesCount = %v\n", len(data.PrepareRenames))
	fmt.Fprintf(buf, "SignaturesCount = %v\n", len(data.Signatures))
	fmt.Fprintf(buf, "LinksCount = %v\n", linksCount)
	fmt.Fprintf(buf, "SelectionRangesCount = %v\n", len(data.SelectionRanges))

	want := string(data.Golden(t, "summary", summaryFile, func() ([]byte, error) {
		return buf.Bytes(), nil
	}))
	got := buf.String()
	if want != got {
		// These counters change when assertions are added or removed.
		// They act as an independent safety net to ensure that the
		// tests didn't spuriously pass because they did no work.
		t.Errorf("test summary does not match:\n%s\n(Run with -golden to update golden file; also, there may be one per Go version.)", compare.Text(want, got))
	}
}

func (data *Data) Mapper(uri span.URI) (*protocol.Mapper, error) {
	data.mappersMu.Lock()
	defer data.mappersMu.Unlock()

	if _, ok := data.mappers[uri]; !ok {
		content, err := data.Exported.FileContents(uri.Filename())
		if err != nil {
			return nil, err
		}
		data.mappers[uri] = protocol.NewMapper(uri, content)
	}
	return data.mappers[uri], nil
}

func (data *Data) Golden(t *testing.T, tag, target string, update func() ([]byte, error)) []byte {
	t.Helper()
	fragment, found := data.fragments[target]
	if !found {
		if filepath.IsAbs(target) {
			t.Fatalf("invalid golden file fragment %v", target)
		}
		fragment = target
	}
	golden := data.golden[fragment]
	if golden == nil {
		if !*UpdateGolden {
			t.Fatalf("could not find golden file %v: %v", fragment, tag)
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
			t.Fatalf("could not update golden file %v: %v", fragment, err)
		}
		file.Data = append(contents, '\n') // add trailing \n for txtar
		golden.Modified = true

	}
	if file == nil {
		t.Fatalf("could not find golden contents %v: %v", fragment, tag)
	}
	if len(file.Data) == 0 {
		return file.Data
	}
	return file.Data[:len(file.Data)-1] // drop the trailing \n
}

func (data *Data) collectCodeLens(spn span.Span, title, cmd string) {
	data.CodeLens[spn.URI()] = append(data.CodeLens[spn.URI()], protocol.CodeLens{
		Range: data.mustRange(spn),
		Command: &protocol.Command{
			Title:   title,
			Command: cmd,
		},
	})
}

func (data *Data) collectDiagnostics(spn span.Span, msgSource, msgPattern, msgSeverity string) {
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

	data.Diagnostics[spn.URI()] = append(data.Diagnostics[spn.URI()], &source.Diagnostic{
		Range:    data.mustRange(spn),
		Severity: severity,
		Source:   source.DiagnosticSource(msgSource),
		Message:  msgPattern,
	})
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

func (data *Data) collectCompletionItems(pos token.Pos, label, detail, kind string, args []string) {
	var documentation string
	if len(args) > 3 {
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

func (data *Data) collectAddImports(spn span.Span, imp string) {
	data.AddImport[spn.URI()] = imp
}

func (data *Data) collectSemanticTokens(spn span.Span) {
	data.SemanticTokens = append(data.SemanticTokens, spn)
}

func (data *Data) collectSuggestedFixes(spn span.Span, actionKind, fix string) {
	data.SuggestedFixes[spn] = append(data.SuggestedFixes[spn], SuggestedFix{actionKind, fix})
}

func (data *Data) collectMethodExtractions(start span.Span, end span.Span) {
	if _, ok := data.MethodExtractions[start]; !ok {
		data.MethodExtractions[start] = end
	}
}

func (data *Data) collectDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src: src,
		Def: target,
	}
}

func (data *Data) collectSelectionRanges(spn span.Span) {
	data.SelectionRanges = append(data.SelectionRanges, spn)
}

func (data *Data) collectIncomingCalls(src span.Span, calls []span.Span) {
	for _, call := range calls {
		rng := data.mustRange(call)
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
	if data.CallHierarchy[src] == nil {
		data.CallHierarchy[src] = &CallHierarchyResult{}
	}
	for _, call := range calls {
		// we're only comparing protocol.range
		data.CallHierarchy[src].OutgoingCalls = append(data.CallHierarchy[src].OutgoingCalls,
			protocol.CallHierarchyItem{
				URI:   protocol.DocumentURI(call.URI()),
				Range: data.mustRange(call),
			})
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

func (data *Data) collectInlayHints(src span.Span) {
	data.InlayHints = append(data.InlayHints, src)
}

func (data *Data) collectRenames(src span.Span, newText string) {
	data.Renames[src] = newText
}

func (data *Data) collectPrepareRenames(src, spn span.Span, placeholder string) {
	data.PrepareRenames[src] = &source.PrepareItem{
		Range: data.mustRange(spn),
		Text:  placeholder,
	}
}

// mustRange converts spn into a protocol.Range, panicking on any error.
func (data *Data) mustRange(spn span.Span) protocol.Range {
	m, err := data.Mapper(spn.URI())
	rng, err := m.SpanRange(spn)
	if err != nil {
		panic(fmt.Sprintf("converting span %s to range: %v", spn, err))
	}
	return rng
}

func (data *Data) collectSignatures(spn span.Span, signature string, activeParam int64) {
	data.Signatures[spn] = &protocol.SignatureHelp{
		Signatures: []protocol.SignatureInformation{
			{
				Label: signature,
			},
		},
		ActiveParameter: uint32(activeParam),
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
	position := safetoken.StartPosition(fset, note.Pos)
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

// TODO(golang/go#54845): improve the formatting here to match standard
// line:column position formatting.
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
