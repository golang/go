// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmdtest contains the test suite for the command line behavior of gopls.
package cmdtest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

type runner struct {
	exporter    packagestest.Exporter
	data        *tests.Data
	ctx         context.Context
	options     func(*source.Options)
	normalizers []normalizer
	remote      string
}

type normalizer struct {
	path     string
	slashed  string
	escaped  string
	fragment string
}

func NewRunner(exporter packagestest.Exporter, data *tests.Data, ctx context.Context, remote string, options func(*source.Options)) *runner {
	r := &runner{
		exporter:    exporter,
		data:        data,
		ctx:         ctx,
		options:     options,
		normalizers: make([]normalizer, 0, len(data.Exported.Modules)),
		remote:      remote,
	}
	// build the path normalizing patterns
	for _, m := range data.Exported.Modules {
		for fragment := range m.Files {
			n := normalizer{
				path:     data.Exported.File(m.Name, fragment),
				fragment: fragment,
			}
			if n.slashed = filepath.ToSlash(n.path); n.slashed == n.path {
				n.slashed = ""
			}
			quoted := strconv.Quote(n.path)
			if n.escaped = quoted[1 : len(quoted)-1]; n.escaped == n.path {
				n.escaped = ""
			}
			r.normalizers = append(r.normalizers, n)
		}
	}
	return r
}

func (r *runner) CodeLens(t *testing.T, spn span.Span, want []protocol.CodeLens) {
	//TODO: add command line completions tests when it works
}

func (r *runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) CompletionSnippet(t *testing.T, src span.Span, expected tests.CompletionSnippet, placeholders bool, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) UnimportedCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) FuzzyCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) CaseSensitiveCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) RankCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) WorkspaceSymbols(*testing.T, string, []protocol.SymbolInformation, map[string]struct{}) {
	//TODO: add command line workspace symbol tests when it works
}

func (r *runner) FuzzyWorkspaceSymbols(*testing.T, string, []protocol.SymbolInformation, map[string]struct{}) {
	//TODO: add command line workspace symbol tests when it works
}

func (r *runner) CaseSensitiveWorkspaceSymbols(*testing.T, string, []protocol.SymbolInformation, map[string]struct{}) {
	//TODO: add command line workspace symbol tests when it works
}

func (r *runner) runGoplsCmd(t testing.TB, args ...string) (string, string) {
	rStdout, wStdout, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStdout := os.Stdout
	rStderr, wStderr, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStderr := os.Stderr
	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		io.Copy(stdout, rStdout)
		wg.Done()
	}()
	go func() {
		io.Copy(stderr, rStderr)
		wg.Done()
	}()
	os.Stdout, os.Stderr = wStdout, wStderr
	app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Exported.Config.Env, r.options)
	remote := r.remote
	err = tool.Run(tests.Context(t),
		app,
		append([]string{fmt.Sprintf("-remote=internal@%s", remote)}, args...))
	if err != nil {
		fmt.Fprint(os.Stderr, err)
	}
	wStdout.Close()
	wStderr.Close()
	wg.Wait()
	os.Stdout, os.Stderr = oldStdout, oldStderr
	rStdout.Close()
	rStderr.Close()
	return stdout.String(), stderr.String()
}

// NormalizeGoplsCmd runs the gopls command and normalizes its output.
func (r *runner) NormalizeGoplsCmd(t testing.TB, args ...string) (string, string) {
	stdout, stderr := r.runGoplsCmd(t, args...)
	return r.Normalize(stdout), r.Normalize(stderr)
}

// NormalizePrefix normalizes a single path at the front of the input string.
func (r *runner) NormalizePrefix(s string) string {
	for _, n := range r.normalizers {
		if t := strings.TrimPrefix(s, n.path); t != s {
			return n.fragment + t
		}
		if t := strings.TrimPrefix(s, n.slashed); t != s {
			return n.fragment + t
		}
		if t := strings.TrimPrefix(s, n.escaped); t != s {
			return n.fragment + t
		}
	}
	return s
}

// Normalize replaces all paths present in s with just the fragment portion
// this is used to make golden files not depend on the temporary paths of the files
func (r *runner) Normalize(s string) string {
	type entry struct {
		path     string
		index    int
		fragment string
	}
	match := make([]entry, 0, len(r.normalizers))
	// collect the initial state of all the matchers
	for _, n := range r.normalizers {
		index := strings.Index(s, n.path)
		if index >= 0 {
			match = append(match, entry{n.path, index, n.fragment})
		}
		if n.slashed != "" {
			index := strings.Index(s, n.slashed)
			if index >= 0 {
				match = append(match, entry{n.slashed, index, n.fragment})
			}
		}
		if n.escaped != "" {
			index := strings.Index(s, n.escaped)
			if index >= 0 {
				match = append(match, entry{n.escaped, index, n.fragment})
			}
		}
	}
	// result should be the same or shorter than the input
	buf := bytes.NewBuffer(make([]byte, 0, len(s)))
	last := 0
	for {
		// find the nearest path match to the start of the buffer
		next := -1
		nearest := len(s)
		for i, c := range match {
			if c.index >= 0 && nearest > c.index {
				nearest = c.index
				next = i
			}
		}
		// if there are no matches, we copy the rest of the string and are done
		if next < 0 {
			buf.WriteString(s[last:])
			return buf.String()
		}
		// we have a match
		n := &match[next]
		// copy up to the start of the match
		buf.WriteString(s[last:n.index])
		// skip over the filename
		last = n.index + len(n.path)
		// add in the fragment instead
		buf.WriteString(n.fragment)
		// see what the next match for this path is
		n.index = strings.Index(s[last:], n.path)
		if n.index >= 0 {
			n.index += last
		}
	}
}
