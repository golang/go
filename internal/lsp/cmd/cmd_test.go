// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// We hardcode the expected number of test cases to ensure that all tests
// are being executed. If a test is added, this number must be changed.
const (
	expectedCompletionsCount = 64
	expectedDiagnosticsCount = 16
	expectedFormatCount      = 4
)

func TestCommandLine(t *testing.T) {
	packagestest.TestAll(t, testCommandLine)
}

func testCommandLine(t *testing.T, exporter packagestest.Exporter) {
	const dir = "../testdata"

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

	// Do a first pass to collect special markers for completion.
	if err := exported.Expect(map[string]interface{}{
		"item": func(name string, r packagestest.Range, _, _ string) {
			exported.Mark(name, r)
		},
	}); err != nil {
		t.Fatal(err)
	}

	expectedDiagnostics := make(diagnostics)
	completionItems := make(completionItems)
	expectedCompletions := make(completions)
	expectedFormat := make(formats)
	expectedDefinitions := make(definitions)
	expectedTypeDefinitions := make(definitions)

	// Collect any data that needs to be used by subsequent tests.
	if err := exported.Expect(map[string]interface{}{
		"diag":       expectedDiagnostics.collect,
		"item":       completionItems.collect,
		"complete":   expectedCompletions.collect,
		"format":     expectedFormat.collect,
		"godef":      expectedDefinitions.godef,
		"definition": expectedDefinitions.definition,
		"typdef":     expectedTypeDefinitions.typdef,
	}); err != nil {
		t.Fatal(err)
	}

	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		expectedCompletions.test(t, exported, completionItems)
	})

	t.Run("Diagnostics", func(t *testing.T) {
		t.Helper()
		expectedDiagnostics.test(t, exported)
	})

	t.Run("Format", func(t *testing.T) {
		t.Helper()
		expectedFormat.test(t, exported)
	})

	t.Run("Definitions", func(t *testing.T) {
		t.Helper()
		expectedDefinitions.testDefinitions(t, exported)
	})

	t.Run("TypeDefinitions", func(t *testing.T) {
		t.Helper()
		expectedTypeDefinitions.testTypeDefinitions(t, exported)
	})
}

type completionItems map[span.Range]*source.CompletionItem
type completions map[span.Span][]span.Span
type formats map[span.URI]span.Span

func (l completionItems) collect(spn span.Range, label, detail, kind string) {
	var k source.CompletionItemKind
	switch kind {
	case "struct":
		k = source.StructCompletionItem
	case "func":
		k = source.FunctionCompletionItem
	case "var":
		k = source.VariableCompletionItem
	case "type":
		k = source.TypeCompletionItem
	case "field":
		k = source.FieldCompletionItem
	case "interface":
		k = source.InterfaceCompletionItem
	case "const":
		k = source.ConstantCompletionItem
	case "method":
		k = source.MethodCompletionItem
	case "package":
		k = source.PackageCompletionItem
	}
	l[spn] = &source.CompletionItem{
		Label:  label,
		Detail: detail,
		Kind:   k,
	}
}

func (l completions) collect(src span.Span, expected []span.Span) {
	l[src] = expected
}

func (l completions) test(t *testing.T, e *packagestest.Exported, items completionItems) {
	if len(l) != expectedCompletionsCount {
		t.Errorf("got %v completions expected %v", len(l), expectedCompletionsCount)
	}
	//TODO: add command line completions tests when it works
}

func (l formats) collect(src span.Span) {
	l[src.URI()] = src
}

func (l formats) test(t *testing.T, e *packagestest.Exported) {
	if len(l) != expectedFormatCount {
		t.Errorf("got %v formats expected %v", len(l), expectedFormatCount)
	}
	//TODO: add command line formatting tests when it works
}

func captureStdOut(t testing.TB, f func()) string {
	r, out, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stdout
	defer func() {
		os.Stdout = old
		out.Close()
		r.Close()
	}()
	os.Stdout = out
	f()
	out.Close()
	data, err := ioutil.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(data))
}
