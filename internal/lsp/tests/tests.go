// Copyright 2019q The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"context"
	"flag"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// We hardcode the expected number of test cases to ensure that all tests
// are being executed. If a test is added, this number must be changed.
const (
	ExpectedCompletionsCount     = 85
	ExpectedDiagnosticsCount     = 17
	ExpectedFormatCount          = 4
	ExpectedDefinitionsCount     = 21
	ExpectedTypeDefinitionsCount = 2
	ExpectedHighlightsCount      = 2
	ExpectedSymbolsCount         = 1
	ExpectedSignaturesCount      = 19
)

const (
	overlayFile = ".overlay"
	goldenFile  = ".golden"
	inFile      = ".in"
	testModule  = "golang.org/x/tools/internal/lsp"
)

var updateGolden = flag.Bool("golden", false, "Update golden files")

type Diagnostics map[span.URI][]source.Diagnostic
type CompletionItems map[token.Pos]*source.CompletionItem
type Completions map[span.Span][]token.Pos
type Formats []span.Span
type Definitions map[span.Span]Definition
type Highlights map[string][]span.Span
type Symbols map[span.URI][]source.Symbol
type SymbolsChildren map[string][]source.Symbol
type Signatures map[span.Span]source.SignatureInformation

type Data struct {
	Config          packages.Config
	Exported        *packagestest.Exported
	Diagnostics     Diagnostics
	CompletionItems CompletionItems
	Completions     Completions
	Formats         Formats
	Definitions     Definitions
	Highlights      Highlights
	Symbols         Symbols
	symbolsChildren SymbolsChildren
	Signatures      Signatures

	t         testing.TB
	fragments map[string]string
	dir       string
}

type Tests interface {
	Diagnostics(*testing.T, Diagnostics)
	Completion(*testing.T, Completions, CompletionItems)
	Format(*testing.T, Formats)
	Definition(*testing.T, Definitions)
	Highlight(*testing.T, Highlights)
	Symbol(*testing.T, Symbols)
	Signature(*testing.T, Signatures)
}

type Definition struct {
	Src    span.Span
	IsType bool
	Flags  string
	Def    span.Span
	Match  string
}

func Load(t testing.TB, exporter packagestest.Exporter, dir string) *Data {
	t.Helper()

	data := &Data{
		Diagnostics:     make(Diagnostics),
		CompletionItems: make(CompletionItems),
		Completions:     make(Completions),
		Definitions:     make(Definitions),
		Highlights:      make(Highlights),
		Symbols:         make(Symbols),
		symbolsChildren: make(SymbolsChildren),
		Signatures:      make(Signatures),

		t:         t,
		dir:       dir,
		fragments: map[string]string{},
	}

	files := packagestest.MustCopyFileTree(dir)
	overlays := map[string][]byte{}
	for fragment, operation := range files {
		if strings.Contains(fragment, goldenFile) {
			delete(files, fragment)
		} else if trimmed := strings.TrimSuffix(fragment, inFile); trimmed != fragment {
			delete(files, fragment)
			files[trimmed] = operation
		} else if index := strings.Index(fragment, overlayFile); index >= 0 {
			delete(files, fragment)
			partial := fragment[:index] + fragment[index+len(overlayFile):]
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

	// Merge the exported.Config with the view.Config.
	data.Config = *data.Exported.Config
	data.Config.Fset = token.NewFileSet()
	data.Config.Context = context.Background()
	data.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
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
		"godef":     data.collectDefinitions,
		"typdef":    data.collectTypeDefinitions,
		"highlight": data.collectHighlights,
		"symbol":    data.collectSymbols,
		"signature": data.collectSignatures,
	}); err != nil {
		t.Fatal(err)
	}
	for _, symbols := range data.Symbols {
		for i := range symbols {
			children := data.symbolsChildren[symbols[i].Name]
			symbols[i].Children = children
		}
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
		tests.Completion(t, data.Completions, data.CompletionItems)
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
		if _, err := exec.LookPath("gofmt"); err != nil {
			switch runtime.GOOS {
			case "android":
				t.Skip("gofmt is not installed")
			default:
				t.Fatal(err)
			}
		}
		if len(data.Formats) != ExpectedFormatCount {
			t.Errorf("got %v formats expected %v", len(data.Formats), ExpectedFormatCount)
		}
		tests.Format(t, data.Formats)
	})

	t.Run("Definitions", func(t *testing.T) {
		t.Helper()
		if len(data.Definitions) != ExpectedDefinitionsCount {
			t.Errorf("got %v definitions expected %v", len(data.Definitions), ExpectedDefinitionsCount)
		}
		tests.Definition(t, data.Definitions)
	})

	t.Run("Highlights", func(t *testing.T) {
		t.Helper()
		if len(data.Highlights) != ExpectedHighlightsCount {
			t.Errorf("got %v highlights expected %v", len(data.Highlights), ExpectedHighlightsCount)
		}
		tests.Highlight(t, data.Highlights)
	})

	t.Run("Symbols", func(t *testing.T) {
		t.Helper()
		if len(data.Symbols) != ExpectedSymbolsCount {
			t.Errorf("got %v symbols expected %v", len(data.Symbols), ExpectedSymbolsCount)
		}
		tests.Symbol(t, data.Symbols)
	})

	t.Run("Signatures", func(t *testing.T) {
		t.Helper()
		if len(data.Signatures) != ExpectedSignaturesCount {
			t.Errorf("got %v signatures expected %v", len(data.Signatures), ExpectedSignaturesCount)
		}
		tests.Signature(t, data.Signatures)
	})
}

func (data *Data) Golden(tag string, target string, update func(golden string) error) []byte {
	data.t.Helper()
	fragment, found := data.fragments[target]
	if !found {
		if filepath.IsAbs(target) {
			data.t.Fatalf("invalid golden file fragment %v", target)
		}
		fragment = target
	}
	dir, file := path.Split(fragment)
	prefix, suffix := file, ""
	// we deliberately use the first . not the last
	if dot := strings.IndexRune(file, '.'); dot >= 0 {
		prefix = file[:dot]
		suffix = file[dot:]
	}
	golden := path.Join(data.dir, dir, prefix) + "." + tag + goldenFile + suffix
	if *updateGolden {
		if err := update(golden); err != nil {
			data.t.Fatalf("could not update golden file %v: %v", golden, err)
		}
	}
	contents, err := ioutil.ReadFile(golden)
	if err != nil {
		data.t.Fatalf("could not read golden file %v: %v", golden, err)
	}
	return contents
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

func (data *Data) collectDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src: src,
		Def: target,
	}
}

func (data *Data) collectTypeDefinitions(src, target span.Span) {
	data.Definitions[src] = Definition{
		Src:    src,
		Def:    target,
		IsType: true,
	}
}

func (data *Data) collectHighlights(name string, rng span.Span) {
	data.Highlights[name] = append(data.Highlights[name], rng)
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
	data.Signatures[spn] = source.SignatureInformation{
		Label:           signature,
		ActiveParameter: int(activeParam),
	}
}
