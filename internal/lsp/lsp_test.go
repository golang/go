// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
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
	expectedDiagnostics := make(map[string][]protocol.Diagnostic)
	expectedCompletions := make(map[token.Position]*protocol.CompletionItem)
	expectedFormat := make(map[string]string)

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
		"diag": func(pos token.Position, msg string) {
			collectDiagnostics(t, expectedDiagnostics, pos, msg)
		},
		"item": func(pos token.Position, label, detail, kind string) {
			collectCompletionItems(expectedCompletions, pos, label, detail, kind)
		},
		"format": func(pos token.Position) {
			collectFormat(expectedFormat, pos)
		},
	}); err != nil {
		t.Fatal(err)
	}

	// test completion
	testCompletion(t, exported, s, expectedCompletions)

	// test diagnostics
	var dirList []string
	for dir := range dirs {
		dirList = append(dirList, dir)
	}
	exported.Config.Mode = packages.LoadFiles
	pkgs, err := packages.Load(exported.Config, dirList...)
	if err != nil {
		t.Fatal(err)
	}
	testDiagnostics(t, s.view, pkgs, expectedDiagnostics)

	// test format
	testFormat(t, s, expectedFormat)
}

func testCompletion(t *testing.T, exported *packagestest.Exported, s *server, wants map[token.Position]*protocol.CompletionItem) {
	if err := exported.Expect(map[string]interface{}{
		"complete": func(src token.Position, expected []token.Position) {
			var want []protocol.CompletionItem
			for _, pos := range expected {
				want = append(want, *wants[pos])
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
		},
	}); err != nil {
		t.Fatal(err)
	}
}

func collectCompletionItems(expectedCompletions map[token.Position]*protocol.CompletionItem, pos token.Position, label, detail, kind string) {
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
	expectedCompletions[pos] = &protocol.CompletionItem{
		Label:  label,
		Detail: detail,
		Kind:   float64(k),
	}
}

func testDiagnostics(t *testing.T, v *source.View, pkgs []*packages.Package, wants map[string][]protocol.Diagnostic) {
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
			want := wants[filename]
			if equal := reflect.DeepEqual(want, got); !equal {
				t.Errorf("diagnostics failed for %s: (expected: %v), (got: %v)", filepath.Base(filename), want, got)
			}
		}
	}
}

func collectDiagnostics(t *testing.T, expectedDiagnostics map[string][]protocol.Diagnostic, pos token.Position, msg string) {
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
	if _, ok := expectedDiagnostics[pos.Filename]; ok {
		expectedDiagnostics[pos.Filename] = append(expectedDiagnostics[pos.Filename], want)
	} else {
		t.Errorf("unexpected filename: %v", pos.Filename)
	}
}

func testFormat(t *testing.T, s *server, expectedFormat map[string]string) {
	for filename, gofmted := range expectedFormat {
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

func collectFormat(expectedFormat map[string]string, pos token.Position) {
	cmd := exec.Command("gofmt", pos.Filename)
	stdout := bytes.NewBuffer(nil)
	cmd.Stdout = stdout
	cmd.Run() // ignore error, sometimes we have intentionally ungofmt-able files
	expectedFormat[pos.Filename] = stdout.String()
}
