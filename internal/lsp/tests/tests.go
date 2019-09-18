// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

// We hardcode the expected number of test cases to ensure that all tests
// are being executed. If a test is added, this number must be changed.
const (
	ExpectedCompletionsCount       = 165
	ExpectedCompletionSnippetCount = 35
	ExpectedDiagnosticsCount       = 21
	ExpectedFormatCount            = 6
	ExpectedImportCount            = 2
	ExpectedSuggestedFixCount      = 1
	ExpectedDefinitionsCount       = 39
	ExpectedTypeDefinitionsCount   = 2
	ExpectedFoldingRangesCount     = 2
	ExpectedHighlightsCount        = 2
	ExpectedReferencesCount        = 6
	ExpectedRenamesCount           = 20
	ExpectedPrepareRenamesCount    = 8
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
type Completions map[span.Span]Completion
type CompletionSnippets map[span.Span]CompletionSnippet
type FoldingRanges []span.Span
type Formats []span.Span
type Imports []span.Span
type SuggestedFixes []span.Span
type Definitions map[span.Span]Definition
type Highlights map[string][]span.Span
type References map[span.Span][]span.Span
type Renames map[span.Span]string
type PrepareRenames map[span.Span]*source.PrepareItem
type Symbols map[span.URI][]protocol.DocumentSymbol
type SymbolsChildren map[string][]protocol.DocumentSymbol
type Signatures map[span.Span]*source.SignatureInformation
type Links map[span.URI][]Link

type Data struct {
	Config             packages.Config
	Exported           *packagestest.Exported
	Diagnostics        Diagnostics
	CompletionItems    CompletionItems
	Completions        Completions
	CompletionSnippets CompletionSnippets
	FoldingRanges      FoldingRanges
	Formats            Formats
	Imports            Imports
	SuggestedFixes     SuggestedFixes
	Definitions        Definitions
	Highlights         Highlights
	References         References
	Renames            Renames
	PrepareRenames     PrepareRenames
	Symbols            Symbols
	symbolsChildren    SymbolsChildren
	Signatures         Signatures
	Links              Links

	t         testing.TB
	fragments map[string]string
	dir       string
	golden    map[string]*Golden

	mappersMu sync.Mutex
	mappers   map[span.URI]*protocol.ColumnMapper
}

type Tests interface {
	Diagnostics(*testing.T, Diagnostics)
	Completion(*testing.T, Completions, CompletionSnippets, CompletionItems)
	FoldingRange(*testing.T, FoldingRanges)
	Format(*testing.T, Formats)
	Import(*testing.T, Imports)
	SuggestedFix(*testing.T, SuggestedFixes)
	Definition(*testing.T, Definitions)
	Highlight(*testing.T, Highlights)
	Reference(*testing.T, References)
	Rename(*testing.T, Renames)
	PrepareRename(*testing.T, PrepareRenames)
	Symbol(*testing.T, Symbols)
	SignatureHelp(*testing.T, Signatures)
	Link(*testing.T, Links)
}

type Definition struct {
	Name      string
	IsType    bool
	OnlyHover bool
	Src, Def  span.Span
}

type CompletionTestType int

const (
	// Full means candidates in test must match full list of candidates.
	CompletionFull CompletionTestType = iota

	// Partial means candidates in test must be valid and in the right relative order.
	CompletionPartial
)

type Completion struct {
	CompletionItems []token.Pos
	Type            CompletionTestType
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
		PrepareRenames:     make(PrepareRenames),
		Symbols:            make(Symbols),
		symbolsChildren:    make(SymbolsChildren),
		Signatures:         make(Signatures),
		Links:              make(Links),

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
	}
	data.Exported = packagestest.Export(t, exporter, modules)
	for fragment, _ := range files {
		filename := data.Exported.File(testModule, fragment)
		data.fragments[filename] = fragment
	}
	data.Exported.Config.Logf = nil

	// Merge the exported.Config with the view.Config.
	data.Config = *data.Exported.Config
	data.Config.Fset = token.NewFileSet()
	data.Config.Logf = nil
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
		"complete":        data.collectCompletions(CompletionFull),
		"completePartial": data.collectCompletions(CompletionPartial),
		"fold":            data.collectFoldingRanges,
		"format":          data.collectFormats,
		"import":          data.collectImports,
		"godef":           data.collectDefinitions,
		"typdef":          data.collectTypeDefinitions,
		"hover":           data.collectHoverDefinitions,
		"highlight":       data.collectHighlights,
		"refs":            data.collectReferences,
		"rename":          data.collectRenames,
		"prepare":         data.collectPrepareRenames,
		"symbol":          data.collectSymbols,
		"signature":       data.collectSignatures,
		"snippet":         data.collectCompletionSnippets,
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

	t.Run("FoldingRange", func(t *testing.T) {
		t.Helper()
		if len(data.FoldingRanges) != ExpectedFoldingRangesCount {
			t.Errorf("got %v folding ranges expected %v", len(data.FoldingRanges), ExpectedFoldingRangesCount)
		}
		tests.FoldingRange(t, data.FoldingRanges)
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

	t.Run("SuggestedFix", func(t *testing.T) {
		t.Helper()
		if len(data.SuggestedFixes) != ExpectedSuggestedFixCount {
			t.Errorf("got %v suggested fixes expected %v", len(data.SuggestedFixes), ExpectedSuggestedFixCount)
		}
		tests.SuggestedFix(t, data.SuggestedFixes)
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

	t.Run("PrepareRenames", func(t *testing.T) {
		t.Helper()
		if len(data.PrepareRenames) != ExpectedPrepareRenamesCount {
			t.Errorf("got %v prepare renames expected %v", len(data.PrepareRenames), ExpectedPrepareRenamesCount)
		}

		tests.PrepareRename(t, data.PrepareRenames)
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
	severity := source.SeverityError
	if strings.Contains(string(spn.URI()), "analyzer") {
		severity = source.SeverityWarning
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

// diffDiagnostics prints the diff between expected and actual diagnostics test
// results.
func DiffDiagnostics(uri span.URI, want, got []source.Diagnostic) string {
	sortDiagnostics(want)
	sortDiagnostics(got)

	if len(got) != len(want) {
		return summarizeDiagnostics(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Message != g.Message {
			return summarizeDiagnostics(i, want, got, "incorrect Message got %v want %v", g.Message, w.Message)
		}
		if protocol.ComparePosition(w.Range.Start, g.Range.Start) != 0 {
			return summarizeDiagnostics(i, want, got, "incorrect Start got %v want %v", g.Range.Start, w.Range.Start)
		}
		// Special case for diagnostics on parse errors.
		if strings.Contains(string(uri), "noparse") {
			if protocol.ComparePosition(g.Range.Start, g.Range.End) != 0 || protocol.ComparePosition(w.Range.Start, g.Range.End) != 0 {
				return summarizeDiagnostics(i, want, got, "incorrect End got %v want %v", g.Range.End, w.Range.Start)
			}
		} else if !protocol.IsPoint(g.Range) { // Accept any 'want' range if the diagnostic returns a zero-length range.
			if protocol.ComparePosition(w.Range.End, g.Range.End) != 0 {
				return summarizeDiagnostics(i, want, got, "incorrect End got %v want %v", g.Range.End, w.Range.End)
			}
		}
		if w.Severity != g.Severity {
			return summarizeDiagnostics(i, want, got, "incorrect Severity got %v want %v", g.Severity, w.Severity)
		}
		if w.Source != g.Source {
			return summarizeDiagnostics(i, want, got, "incorrect Source got %v want %v", g.Source, w.Source)
		}
	}
	return ""
}

func sortDiagnostics(d []source.Diagnostic) {
	sort.Slice(d, func(i int, j int) bool {
		return compareDiagnostic(d[i], d[j]) < 0
	})
}

func compareDiagnostic(a, b source.Diagnostic) int {
	if r := span.CompareURI(a.URI, b.URI); r != 0 {
		return r
	}
	if r := protocol.CompareRange(a.Range, b.Range); r != 0 {
		return r
	}
	if a.Message < b.Message {
		return -1
	}
	if a.Message == b.Message {
		return 0
	} else {
		return 1
	}
}

func summarizeDiagnostics(i int, want []source.Diagnostic, got []source.Diagnostic, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "diagnostics failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, d := range want {
		fmt.Fprintf(msg, "  %s:%v: %s\n", d.URI, d.Range, d.Message)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %s:%v: %s\n", d.URI, d.Range, d.Message)
	}
	return msg.String()
}

func (data *Data) collectCompletions(typ CompletionTestType) func(span.Span, []token.Pos) {
	return func(src span.Span, expected []token.Pos) {
		data.Completions[src] = Completion{
			CompletionItems: expected,
			Type:            typ,
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
		Kind:          source.ParseCompletionItemKind(kind),
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
