// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
)

func (r *runner) WorkspaceSymbols(t *testing.T, uri span.URI, query string, typ tests.WorkspaceSymbolsTestType) {
	var matcher string
	switch typ {
	case tests.WorkspaceSymbolsFuzzy:
		matcher = "fuzzy"
	case tests.WorkspaceSymbolsCaseSensitive:
		matcher = "caseSensitive"
	case tests.WorkspaceSymbolsDefault:
		matcher = "caseInsensitive"
	}
	r.runWorkspaceSymbols(t, uri, matcher, query)
}

func (r *runner) runWorkspaceSymbols(t *testing.T, uri span.URI, matcher, query string) {
	t.Helper()

	out, _ := r.runGoplsCmd(t, "workspace_symbol", "-matcher", matcher, query)
	var filtered []string
	dir := filepath.Dir(uri.Filename())
	for _, line := range strings.Split(out, "\n") {
		if source.InDir(dir, line) {
			filtered = append(filtered, filepath.ToSlash(line))
		}
	}
	sort.Strings(filtered)
	got := r.Normalize(strings.Join(filtered, "\n") + "\n")

	expect := string(r.data.Golden(t, fmt.Sprintf("workspace_symbol-%s-%s", strings.ToLower(string(matcher)), query), uri.Filename(), func() ([]byte, error) {
		return []byte(got), nil
	}))

	if expect != got {
		t.Errorf("workspace_symbol failed for %s:\n%s", query, compare.Text(expect, got))
	}
}
