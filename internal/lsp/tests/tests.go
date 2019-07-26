// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"context"
	"flag"
	"go/ast"
	"go/token"
	"io/ioutil"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/txtar"
)

// We hardcode the expected number of test cases to ensure that all tests
// are being executed. If a test is added, this number must be changed.
const (
	ExpectedCompletionsCount       = 144
	ExpectedCompletionSnippetCount = 15
	ExpectedDiagnosticsCount       = 17
	ExpectedFormatCount            = 6
	ExpectedImportCount            = 2
	ExpectedDefinitionsCount       = 38
	ExpectedTypeDefinitionsCount   = 2
	ExpectedHighlightsCount        = 2
	ExpectedReferencesCount        = 5
	ExpectedRenamesCount           = 16
	ExpectedSymbolsCount           = 1
	ExpectedSignaturesCount        = 21
	ExpectedLinksCount             = 4
)

const (
	overlayFileSuffix = ".overlay"
	goldenFileSuffix  = ".golden"
	inFileSuffix      = ".in"
	testModule        = "golang.org/x/tools/internal/lsp"
)

var updateGolden = flag.Bool("golden", false, "Update golden files")

type Diagnostics map[span.URI][]source.Diagnostic
type CompletionItems map[token.Pos]*source.CompletionItem
type Completions map[span.Span][]token.Pos
type CompletionSnippets map[span.Span]CompletionSnippet
type Formats []span.Span
type Imports []span.Span
type Definitions map[span.Span]Definition
type Highlights map[string][]span.Span
type References map[span.Span][]span.Span
type Renames map[span.Span]string
type Symbols map[span.URI][]source.Symbol
type SymbolsChildren map[string][]source.Symbol
type Signatures map[span.Span]*source.SignatureInformation
type Links map[span.URI][]Link

type Data struct {
	Config             packages.Config
	Exported           *packagestest.Exported
	Diagnostics        Diagnostics
	CompletionItems    CompletionItems
	Completions        Completions
	CompletionSnippets CompletionSnippets
	Formats            Formats
	Imports            Imports
	Definitions        Definitions
	Highlights         Highlights
	References         References
	Renames            Renames
	Symbols            Symbols
	symbolsChildren    SymbolsChildren
	Signatures         Signatures
	Links              Links

	t         testing.TB
	fragments map[string]string
	dir       string
	golden    map[string]*Golden
}

type Tests interface {
	Diagnostics(*testing.T, Diagnostics)
	Completion(*testing.T, Completions, CompletionSnippets, CompletionItems)
	Format(*testing.T, Formats)
	Import(*testing.T, Imports)
	Definition(*testing.T, Definitions)
	Highlight(*testing.T, Highlights)
	Reference(*testing.T, References)
	Rename(*testing.T, Renames)
	Symbol(*testing.T, Symbols)
	SignatureHelp(*testing.T, Signatures)
	Link(*testing.T, Links)
}

type Definition struct {
	Name      string
	Src       span.Span
	IsType    bool
	OnlyHover bool
	Def       span.Span
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

func Load(t testing.TB, exporter packagestest.Exporter, dir string) *Data {
	t.Helper()

	data := &Data{
		Diagnostics:        make(Diagnostics),
		CompletionItems:    make(CompletionItems),
		Completions:        make(Completions),
		CompletionSnippets: make(CompletionSnippets),
		Definitions:        make(Definitions),
		Highlights:         make(Highlights),
		References:         make(References),
		Renames:            make(Renames),
		Symbols:            make(Symbols),
		symbolsChildren:    make(SymbolsChildren),
		Signatures:         make(Signatures),
		Links:              make(Links),

		t:         t,
		dir:       dir,
		fragments: map[string]string{},
		golden:    map[string]*Golden{},
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
	}
	data.Exported = packagestest.Export(t, exporter, modules)
	for fragment, _ := range files {
		filename := data.Exported.File(testModule, fragment)
		data.fragments[filename] = fragment
	}
	data.Exported.Config.Logf = t.Logf

	// Merge the exported.Config with the view.Config.
	data.Config = *data.Exported.Config
	data.Config.Fset = token.NewFileSet()
	data.Config.Logf = t.Logf
	data.Config.Context = Context(nil)
	data.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		panic("ParseFile should not be called")
	}

	// Do a first pass to collect special markers for completion.
	if err := data.Exported.Expect(map[string]interface{}{
		"item": func(name string, r packagestest.Range, _, _ string) {
			data.Exported.Mark(name, r)
		},
	}); err != nil {
		t.Fatal(err)
	}

	// Collect any data that needs to be used by subsequent tests.
	if err := data.Exported.Expect(map[string]interface{}{
		"diag":      data.collectDiagnostics,
		"item":      data.collectCompletionItems,
		"complete":  data.collectCompletions,
		"format":    data.collectFormats,
		"import":    data.collectImports,
		"godef":     data.collectDefinitions,
		"typdef":    data.collectTypeDefinitions,
		"hover":     data.collectHoverDefinitions,
		"highlight": data.collectHighlights,
		"refs":      data.collectReferences,
		"rename":    data.collectRenames,
		"symbol":    data.collectSymbols,
		"signature": data.collectSignatures,
		"snippet":   data.collectCompletionSnippets,
		"link":      data.collectLinks,
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
	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		if len(data.Completions) != ExpectedCompletionsCount {
			t.Errorf("got %v completions expected %v", len(data.Completions), ExpectedCompletionsCount)
		}
		if len(data.CompletionSnippets) != ExpectedCompletionSnippetCount {
			t.Errorf("got %v snippets expected %v", len(data.CompletionSnippets), ExpectedCompletionSnippetCount)
		}
		tests.Completion(t, data.Completions, data.CompletionSnippets, data.CompletionItems)
	})

	t.Run("Diagnostics", func(t *testing.T) {
		t.Helper()
		diagnosticsCount := 0
		for _, want := range data.Diagnostics {
			diagnosticsCount += len(want)
		}
		if diagnosticsCount != ExpectedDiagnosticsCount {
			t.Errorf("got %v diagnostics expected %v", diagnosticsCount, ExpectedDiagnosticsCount)
		}
		tests.Diagnostics(t, data.Diagnostics)
	})

	t.Run("Format", func(t *testing.T) {
		t.Helper()
		if len(data.Formats) != ExpectedFormatCount {
			t.Errorf("got %v formats expected %v", len(data.Formats), ExpectedFormatCount)
		}
		tests.Format(t, data.Formats)
	})

	t.Run("Import", func(t *testing.T) {
		t.Helper()
		if len(data.Imports) != ExpectedImportCount {
			t.Errorf("got %v imports expected %v", len(data.Imports), ExpectedImportCount)
		}
		tests.Import(t, data.Imports)
	})

	t.Run("Definition", func(t *testing.T) {
		t.Helper()
		if len(data.Definitions) != ExpectedDefinitionsCount {
			t.Errorf("got %v definitions expected %v", len(data.Definitions), ExpectedDefinitionsCount)
		}
		tests.Definition(t, data.Definitions)
	})

	t.Run("Highlight", func(t *testing.T) {
		t.Helper()
		if len(data.Highlights) != ExpectedHighlightsCount {
			t.Errorf("got %v highlights expected %v", len(data.Highlights), ExpectedHighlightsCount)
		}
		tests.Highlight(t, data.Highlights)
	})

	t.Run("References", func(t *testing.T) {
		t.Helper()
		if len(data.References) != ExpectedReferencesCount {
			t.Errorf("got %v references expected %v", len(data.References), ExpectedReferencesCount)
		}
		tests.Reference(t, data.References)
	})

	t.Run("Renames", func(t *testing.T) {
		t.Helper()
		if len(data.Renames) != ExpectedRenamesCount {
			t.Errorf("got %v renames expected %v", len(data.Renames), ExpectedRenamesCount)
		}
		tests.Rename(t, data.Renames)
	})

	t.Run("Symbols", func(t *testing.T) {
		t.Helper()
		if len(data.Symbols) != ExpectedSymbolsCount {
			t.Errorf("got %v symbols expected %v", len(data.Symbols), ExpectedSymbolsCount)
		}
		tests.Symbol(t, data.Symbols)
	})

	t.Run("SignatureHelp", func(t *testing.T) {
		t.Helper()
		if len(data.Signatures) != ExpectedSignaturesCount {
			t.Errorf("got %v signatures expected %v", len(data.Signatures), ExpectedSignaturesCount)
		}
		tests.SignatureHelp(t, data.Signatures)
	})

	t.Run("Link", func(t *testing.T) {
		t.Helper()
		linksCount := 0
		for _, want := range data.Links {
			linksCount += len(want)
		}
		if linksCount != ExpectedLinksCount {
			t.Errorf("got %v links expected %v", linksCount, ExpectedLinksCount)
		}
		tests.Link(t, data.Links)
	})

	if *updateGolden {
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
		if !*updateGolden {
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
	if *updateGolden {
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
	// If a file has an empty diagnostic message, return. This allows us to
	// avoid testing diagnostics in files that may have a lot of them.
	if msg == "" {
		return
	}
	severity := source.SeverityError
	if strings.Contains(string(spn.URI()), "analyzer") {
		severity = source.SeverityWarning
	}
	want := source.Diagnostic{
		Span:     spn,
		Severity: severity,
		Source:   msgSource,
		Message:  msg,
	}
	data.Diagnostics[spn.URI()] = append(data.Diagnostics[spn.URI()], want)
}

func (data *Data) collectCompletions(src span.Span, expected []token.Pos) {
	data.Completions[src] = expected
}

func (data *Data) collectCompletionItems(pos token.Pos, label, detail, kind string) {
	data.CompletionItems[pos] = &source.CompletionItem{
		Label:  label,
		Detail: detail,
		Kind:   source.ParseCompletionItemKind(kind),
	}
}

func (data *Data) collectFormats(spn span.Span) {
	data.Formats = append(data.Formats, spn)
}

func (data *Data) collectImports(spn span.Span) {
	data.Imports = append(data.Imports, spn)
}

func (data *Data) collectDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src: src,
		Def: target,
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

func (data *Data) collectHighlights(name string, rng span.Span) {
	data.Highlights[name] = append(data.Highlights[name], rng)
}

func (data *Data) collectReferences(src span.Span, expected []span.Span) {
	data.References[src] = expected
}

func (data *Data) collectRenames(src span.Span, newText string) {
	data.Renames[src] = newText
}

func (data *Data) collectSymbols(name string, spn span.Span, kind string, parentName string) {
	sym := source.Symbol{
		Name:          name,
		Kind:          source.ParseSymbolKind(kind),
		SelectionSpan: spn,
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
	data.CompletionSnippets[spn] = CompletionSnippet{
		CompletionItem:     item,
		PlainSnippet:       plain,
		PlaceholderSnippet: placeholder,
	}
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
