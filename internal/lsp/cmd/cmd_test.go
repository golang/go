// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/tests"
)

var isRace = false

type runner struct {
	exporter packagestest.Exporter
	data     *tests.Data
	app      *cmd.Application
}

func TestCommandLine(t *testing.T) {
	packagestest.TestAll(t, testCommandLine)
}

func testCommandLine(t *testing.T, exporter packagestest.Exporter) {
	data := tests.Load(t, exporter, "../testdata")
	defer data.Exported.Cleanup()

	r := &runner{
		exporter: exporter,
		data:     data,
		app:      cmd.New(data.Config.Dir, data.Exported.Config.Env),
	}
	tests.Run(t, r, data)
}

func (r *runner) Completion(t *testing.T, data tests.Completions, snippets tests.CompletionSnippets, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) Highlight(t *testing.T, data tests.Highlights) {
	//TODO: add command line highlight tests when it works
}

func (r *runner) Reference(t *testing.T, data tests.References) {
	//TODO: add command line references tests when it works
}

func (r *runner) Rename(t *testing.T, data tests.Renames) {
	//TODO: add command line rename tests when it works
}

func (r *runner) Symbol(t *testing.T, data tests.Symbols) {
	//TODO: add command line symbol tests when it works
}

func (r *runner) SignatureHelp(t *testing.T, data tests.Signatures) {
	//TODO: add command line signature tests when it works
}

func (r *runner) Link(t *testing.T, data tests.Links) {
	//TODO: add command line link tests when it works
}

func (r *runner) Import(t *testing.T, data tests.Imports) {
	//TODO: add command line imports tests when it works
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
	return string(data)
}

// normalizePaths replaces all paths present in s with just the fragment portion
// this is used to make golden files not depend on the temporary paths of the files
func normalizePaths(data *tests.Data, s string) string {
	type entry struct {
		path     string
		index    int
		fragment string
	}
	match := make([]entry, 0, len(data.Exported.Modules))
	// collect the initial state of all the matchers
	for _, m := range data.Exported.Modules {
		for fragment := range m.Files {
			filename := data.Exported.File(m.Name, fragment)
			index := strings.Index(s, filename)
			if index >= 0 {
				match = append(match, entry{filename, index, fragment})
			}
			if slash := filepath.ToSlash(filename); slash != filename {
				index := strings.Index(s, slash)
				if index >= 0 {
					match = append(match, entry{slash, index, fragment})
				}
			}
			quoted := strconv.Quote(filename)
			if escaped := quoted[1 : len(quoted)-1]; escaped != filename {
				index := strings.Index(s, escaped)
				if index >= 0 {
					match = append(match, entry{escaped, index, fragment})
				}
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
