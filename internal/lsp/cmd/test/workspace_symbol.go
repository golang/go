// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"path"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
)

func (r *runner) WorkspaceSymbols(t *testing.T, query string, expectedSymbols []protocol.SymbolInformation, dirs map[string]struct{}) {
	r.runWorkspaceSymbols(t, "default", query, dirs)
}

func (r *runner) FuzzyWorkspaceSymbols(t *testing.T, query string, expectedSymbols []protocol.SymbolInformation, dirs map[string]struct{}) {
	r.runWorkspaceSymbols(t, "fuzzy", query, dirs)
}

func (r *runner) CaseSensitiveWorkspaceSymbols(t *testing.T, query string, expectedSymbols []protocol.SymbolInformation, dirs map[string]struct{}) {
	r.runWorkspaceSymbols(t, "caseSensitive", query, dirs)
}

func (r *runner) runWorkspaceSymbols(t *testing.T, matcher, query string, dirs map[string]struct{}) {
	t.Helper()

	out, _ := r.runGoplsCmd(t, "workspace_symbol", "-matcher", matcher, query)
	var filtered []string
	for _, line := range strings.Split(out, "\n") {
		for dir := range dirs {
			if strings.HasPrefix(line, dir) {
				filtered = append(filtered, line)
				break
			}
		}
	}
	sort.Strings(filtered)
	got := r.Normalize(strings.Join(filtered, "\n"))

	expect := string(r.data.Golden("workspace_symbol", workspaceSymbolsGolden(matcher, query), func() ([]byte, error) {
		return []byte(got), nil
	}))

	if expect != got {
		t.Errorf("workspace_symbol failed for %s expected:\n%s\ngot:\n%s", query, expect, got)
	}
}

var workspaceSymbolsDir = map[string]string{
	"default":       "",
	"fuzzy":         "fuzzy",
	"caseSensitive": "casesensitive",
}

func workspaceSymbolsGolden(matcher, query string) string {
	dir := []string{"workspacesymbol", workspaceSymbolsDir[matcher]}
	if query == "" {
		return path.Join(append(dir, "EmptyQuery")...)
	}

	var name []rune
	for _, r := range query {
		if 'A' <= r && r <= 'Z' {
			// Escape uppercase to '!' + lowercase for case insensitive file systems.
			name = append(name, '!', r+'a'-'A')
		} else {
			name = append(name, r)
		}
	}
	return path.Join(append(dir, string(name))...)
}
