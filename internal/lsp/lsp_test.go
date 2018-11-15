// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func TestLSP(t *testing.T) {
	packagestest.TestAll(t, testLSP)
}

func testLSP(t *testing.T, exporter packagestest.Exporter) {
	const dir = "testdata"
	const expectedCompletionsCount = 4
	const expectedDiagnosticsCount = 9
	const expectedFormatCount = 3
	const expectedDefinitionsCount = 16

	files := packagestest.MustCopyFileTree(dir)
	for fragment, operation := range files {
		if trimmed := strings.TrimSuffix(fragment, ".in"); trimmed != fragment {
			delete(files, fragment)
			files[trimmed] = operation
		}
	}
	modules := []packagestest.Module{
		{
			Name:  "golang.org/x/tools/internal/lsp",
			Files: files,
		},
	}
	exported := packagestest.Export(t, exporter, modules)
	defer exported.Cleanup()

	dirs := make(map[string]bool)

	// collect results for certain tests
	expectedDiagnostics := make(diagnostics)
	completionItems := make(completionItems)
	expectedCompletions := make(completions)
	expectedFormat := make(formats)
	expectedDefinitions := make(definitions)

	s := &server{
		view: source.NewView(),
	}
	// merge the config objects
	cfg := *exported.Config
	cfg.Fset = s.view.Config.Fset
	cfg.Mode = packages.LoadSyntax
	s.view.Config = &cfg

	for _, module := range modules {
		for fragment := range module.Files {
			if !strings.HasSuffix(fragment, ".go") {
				continue
			}
			filename := exporter.Filename(exported, module.Name, fragment)
			expectedDiagnostics[filename] = []protocol.Diagnostic{}
			dirs[filepath.Dir(filename)] = true
		}
	}
	// Do a first pass to collect special markers
	if err := exported.Expect(map[string]interface{}{
		"item": func(name string, r packagestest.Range, _, _ string) {
			exported.Mark(name, r)
		},
	}); err != nil {
		t.Fatal(err)
	}
	// Collect any data that needs to be used by subsequent tests.
	if err := exported.Expect(map[string]interface{}{
		"diag":     expectedDiagnostics.collect,
		"item":     completionItems.collect,
		"complete": expectedCompletions.collect,
		"format":   expectedFormat.collect,
		"godef":    expectedDefinitions.collect,
	}); err != nil {
		t.Fatal(err)
	}

	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		if len(expectedCompletions) != expectedCompletionsCount {
			t.Errorf("got %v completions expected %v", len(expectedCompletions), expectedCompletionsCount)
		}
		expectedCompletions.test(t, exported, s, completionItems)
	})

	t.Run("Diagnostics", func(t *testing.T) {
		t.Helper()
		diagnosticsCount := expectedDiagnostics.test(t, exported, s.view, dirs)
		if diagnosticsCount != expectedDiagnosticsCount {
			t.Errorf("got %v diagnostics expected %v", diagnosticsCount, expectedDiagnosticsCount)
		}
	})

	t.Run("Format", func(t *testing.T) {
		t.Helper()
		if len(expectedFormat) != expectedFormatCount {
			t.Errorf("got %v formats expected %v", len(expectedFormat), expectedFormatCount)
		}
		expectedFormat.test(t, s)
	})

	t.Run("Definitions", func(t *testing.T) {
		t.Helper()
		if len(expectedDefinitions) != expectedDefinitionsCount {
			t.Errorf("got %v definitions expected %v", len(expectedDefinitions), expectedDefinitionsCount)
		}
		expectedDefinitions.test(t, s)
	})
}

type diagnostics map[string][]protocol.Diagnostic
type completionItems map[token.Pos]*protocol.CompletionItem
type completions map[token.Position][]token.Pos
type formats map[string]string
type definitions map[protocol.Location]protocol.Location

func (c completions) test(t *testing.T, exported *packagestest.Exported, s *server, items completionItems) {
	for src, itemList := range c {
		var want []protocol.CompletionItem
		for _, pos := range itemList {
			want = append(want, *items[pos])
		}
		list, err := s.Completion(context.Background(), &protocol.CompletionParams{
			TextDocumentPositionParams: protocol.TextDocumentPositionParams{
				TextDocument: protocol.TextDocumentIdentifier{
					URI: protocol.DocumentURI(source.ToURI(src.Filename)),
				},
				Position: protocol.Position{
					Line:      float64(src.Line - 1),
					Character: float64(src.Column - 1),
				},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		got := list.Items
		if equal := reflect.DeepEqual(want, got); !equal {
			t.Errorf("completion failed for %s:%v:%v: (expected: %v), (got: %v)", filepath.Base(src.Filename), src.Line, src.Column, want, got)
		}
	}
}

func (c completions) collect(src token.Position, expected []token.Pos) {
	c[src] = expected
}

func (i completionItems) collect(pos token.Pos, label, detail, kind string) {
	var k protocol.CompletionItemKind
	switch kind {
	case "struct":
		k = protocol.StructCompletion
	case "func":
		k = protocol.FunctionCompletion
	case "var":
		k = protocol.VariableCompletion
	case "type":
		k = protocol.TypeParameterCompletion
	case "field":
		k = protocol.FieldCompletion
	case "interface":
		k = protocol.InterfaceCompletion
	case "const":
		k = protocol.ConstantCompletion
	case "method":
		k = protocol.MethodCompletion
	}
	i[pos] = &protocol.CompletionItem{
		Label:  label,
		Detail: detail,
		Kind:   float64(k),
	}
}

func (d diagnostics) test(t *testing.T, exported *packagestest.Exported, v *source.View, dirs map[string]bool) int {
	// first trigger a load to get the diagnostics
	var dirList []string
	for dir := range dirs {
		dirList = append(dirList, dir)
	}
	exported.Config.Mode = packages.LoadFiles
	pkgs, err := packages.Load(exported.Config, dirList...)
	if err != nil {
		t.Fatal(err)
	}
	// and now see if they match the expected ones
	count := 0
	for _, pkg := range pkgs {
		for _, filename := range pkg.GoFiles {
			f := v.GetFile(source.ToURI(filename))
			diagnostics, err := source.Diagnostics(context.Background(), v, f)
			if err != nil {
				t.Fatal(err)
			}
			got := toProtocolDiagnostics(v, diagnostics[filename])
			sort.Slice(got, func(i int, j int) bool {
				return got[i].Range.Start.Line < got[j].Range.Start.Line
			})
			want := d[filename]
			if equal := reflect.DeepEqual(want, got); !equal {
				msg := &bytes.Buffer{}
				fmt.Fprintf(msg, "diagnostics failed for %s: expected:\n", filepath.Base(filename))
				for _, d := range want {
					fmt.Fprintf(msg, "  %v\n", d)
				}
				fmt.Fprintf(msg, "got:\n")
				for _, d := range got {
					fmt.Fprintf(msg, "  %v\n", d)
				}
				t.Error(msg.String())
			}
			count += len(want)
		}
	}
	return count
}

func (d diagnostics) collect(pos token.Position, msg string) {
	line := float64(pos.Line - 1)
	col := float64(pos.Column - 1)
	want := protocol.Diagnostic{
		Range: protocol.Range{
			Start: protocol.Position{
				Line:      line,
				Character: col,
			},
			End: protocol.Position{
				Line:      line,
				Character: col,
			},
		},
		Severity: protocol.SeverityError,
		Source:   "LSP",
		Message:  msg,
	}
	d[pos.Filename] = append(d[pos.Filename], want)
}

func (f formats) test(t *testing.T, s *server) {
	for filename, gofmted := range f {
		edits, err := s.Formatting(context.Background(), &protocol.DocumentFormattingParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.DocumentURI(source.ToURI(filename)),
			},
		})
		if err != nil || len(edits) == 0 {
			if gofmted != "" {
				t.Error(err)
			}
			return
		}
		edit := edits[0]
		if edit.NewText != gofmted {
			t.Errorf("formatting failed: (got: %s), (expected: %s)", edit.NewText, gofmted)
		}
	}
}

func (f formats) collect(pos token.Position) {
	cmd := exec.Command("gofmt", pos.Filename)
	stdout := bytes.NewBuffer(nil)
	cmd.Stdout = stdout
	cmd.Run() // ignore error, sometimes we have intentionally ungofmt-able files
	f[pos.Filename] = stdout.String()
}

func (d definitions) test(t *testing.T, s *server) {
	for src, target := range d {
		locs, err := s.Definition(context.Background(), &protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: src.URI,
			},
			Position: src.Range.Start,
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(locs) != 1 {
			t.Errorf("got %d locations for definition, expected 1", len(locs))
		}
		if locs[0] != target {
			t.Errorf("for %v got %v want %v", src, locs[0], target)
		}
	}
}

func (d definitions) collect(fset *token.FileSet, src, target packagestest.Range) {
	sRange := source.Range{Start: src.Start, End: src.End}
	sLoc := toProtocolLocation(fset, sRange)
	tRange := source.Range{Start: target.Start, End: target.End}
	tLoc := toProtocolLocation(fset, tRange)
	d[sLoc] = tLoc
}
