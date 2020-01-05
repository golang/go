// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"bytes"
	"fmt"
	"path/filepath"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
)

// FilterWorkspaceSymbols filters to got contained in the given dirs.
func FilterWorkspaceSymbols(got []protocol.SymbolInformation, dirs map[string]struct{}) []protocol.SymbolInformation {
	var result []protocol.SymbolInformation
	for _, si := range got {
		if _, ok := dirs[filepath.Dir(si.Location.URI)]; ok {
			result = append(result, si)
		}
	}
	return result
}

// DiffWorkspaceSymbols prints the diff between expected and actual workspace
// symbols test results.
func DiffWorkspaceSymbols(want, got []protocol.SymbolInformation) string {
	sort.Slice(want, func(i, j int) bool { return fmt.Sprintf("%v", want[i]) < fmt.Sprintf("%v", want[j]) })
	sort.Slice(got, func(i, j int) bool { return fmt.Sprintf("%v", got[i]) < fmt.Sprintf("%v", got[j]) })
	if len(got) != len(want) {
		return summarizeWorkspaceSymbols(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Name != g.Name {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect name got %v want %v", g.Name, w.Name)
		}
		if w.Kind != g.Kind {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect kind got %v want %v", g.Kind, w.Kind)
		}
		if w.Location.URI != g.Location.URI {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect uri got %v want %v", g.Location.URI, w.Location.URI)
		}
		if protocol.CompareRange(w.Location.Range, g.Location.Range) != 0 {
			return summarizeWorkspaceSymbols(i, want, got, "incorrect range got %v want %v", g.Location.Range, w.Location.Range)
		}
	}
	return ""
}

func summarizeWorkspaceSymbols(i int, want, got []protocol.SymbolInformation, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "workspace symbols failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, s := range want {
		fmt.Fprintf(msg, "  %v %v %v:%v\n", s.Name, s.Kind, s.Location.URI, s.Location.Range)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, s := range got {
		fmt.Fprintf(msg, "  %v %v %v:%v\n", s.Name, s.Kind, s.Location.URI, s.Location.Range)
	}
	return msg.String()
}
