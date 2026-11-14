// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"

	"cmd/internal/logos/editmap"
)

const aLgo = `package a

import "os"

func Read(path string) (string, error) {
	data := os.ReadFile(path)?
	return string(data), nil
}
`

var questionLine = regexp.MustCompile(`^(\s*)(\w+) := (.+)\?$`)

const siteTrailer = "\nvar _logosSite0 = 0\n"

func lowerQuestion(src []byte) ([]byte, editmap.EditMap, bool) {
	if bytes.Contains(src, []byte("SYNTAX ERROR")) {
		return nil, editmap.EditMap{}, false
	}
	type copyRun struct {
		start, end int
		orig       int
	}
	type step struct {
		span   editmap.EditSpan
		text   string
		copies []copyRun
	}
	var steps []step
	off := 0
	for _, line := range strings.SplitAfter(string(src), "\n") {
		text := strings.TrimSuffix(line, "\n")
		if m := questionLine.FindStringSubmatch(text); m != nil {
			indent := m[1]
			repl := fmt.Sprintf("%s, _e := %s\n%s_logosTrace(_e)\n%sif _e != nil { return \"\", _e }", m[2], m[3], indent, indent)
			name, operand := m[2], m[3]
			operandStart := len(name) + len(", _e := ")
			steps = append(steps, step{
				span: editmap.EditSpan{Start: off + len(indent), End: off + len(text), Len: len(repl)},
				text: repl,
				copies: []copyRun{
					{start: 0, end: len(name), orig: off + len(indent)},
					{start: operandStart, end: operandStart + len(operand), orig: off + len(indent) + len(name) + len(" := ")},
				},
			})
		}
		off += len(line)
	}
	steps = append(steps, step{editmap.EditSpan{Start: len(src), End: len(src), Len: len(siteTrailer)}, siteTrailer, nil})
	var out []byte
	var spans []editmap.EditSpan
	var segs segBuilder
	prev := 0
	for _, s := range steps {
		out = append(out, src[prev:s.span.Start]...)
		segs.copy(prev, s.span.Start-prev)
		out = append(out, s.text...)
		rp := 0
		for _, c := range s.copies {
			segs.gen(c.start - rp)
			segs.copy(c.orig, c.end-c.start)
			rp = c.end
		}
		segs.gen(len(s.text) - rp)
		prev = s.span.End
		spans = append(spans, s.span)
	}
	out = append(out, src[prev:]...)
	segs.copy(prev, len(src)-prev)
	return out, editmap.EditMap{Version: editmap.Version, Size: len(src), Out: len(out), Edits: spans, Segments: segs.done()}, true
}

type fakeCompiler struct {
	mu      sync.Mutex
	closure map[string][]string
	diags   map[string][]Diagnostic
	errs    map[string]error
	missing map[string]bool
	waits   map[string]chan struct{}
	lowered [][]string
	buffers []map[string][]byte
	closed  bool
	changes int
}

func newFakeCompiler() *fakeCompiler {
	return &fakeCompiler{closure: map[string][]string{}, diags: map[string][]Diagnostic{}, errs: map[string]error{}, missing: map[string]bool{}, waits: map[string]chan struct{}{}}
}

func (f *fakeCompiler) DiskChanged() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.changes++
}

func (f *fakeCompiler) setMissing(paths ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, path := range paths {
		f.missing[path] = true
	}
}

func (f *fakeCompiler) setLowerWait(dir string) chan struct{} {
	f.mu.Lock()
	defer f.mu.Unlock()
	gate := make(chan struct{})
	f.waits[dir] = gate
	return gate
}

func (f *fakeCompiler) setLowerError(dir string, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.errs[dir] = err
}

func (f *fakeCompiler) setClosure(dirs ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closure[dirs[0]] = dirs
}

func (f *fakeCompiler) setDiagnostics(dir string, diags ...Diagnostic) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.diags[dir] = diags
}

func (f *fakeCompiler) lowerCalls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var dirs []string
	for _, l := range f.lowered {
		dirs = append(dirs, l[0])
	}
	return dirs
}

func (f *fakeCompiler) bufferCalls() []map[string][]byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.buffers)
}

func (f *fakeCompiler) Lower(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, error) {
	f.mu.Lock()
	dirs := f.closure[dir]
	if dirs == nil {
		dirs = []string{dir}
	}
	lerr := f.errs[dir]
	gate, gated := f.waits[dir]
	f.lowered = append(f.lowered, dirs)
	f.buffers = append(f.buffers, buffers)
	f.mu.Unlock()
	if gated {
		select {
		case <-gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if lerr != nil {
		return nil, lerr
	}
	low := &Lowering{Dirs: dirs}
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".lgo") {
				f.lowerFile(low, filepath.Join(d, e.Name()), buffers)
			}
		}
	}
	return low, nil
}

func (f *fakeCompiler) lowerFile(low *Lowering, path string, buffers map[string][]byte) {
	f.mu.Lock()
	missing := f.missing[path]
	f.mu.Unlock()
	if missing {
		low.Missing = append(low.Missing, path)
		return
	}
	src, ok := buffers[path]
	if !ok {
		src, _ = os.ReadFile(path)
	}
	out, m, ok := lowerQuestion(src)
	if !ok {
		low.Missing = append(low.Missing, path)
		return
	}
	m.Source = filepath.Base(path)
	low.Files = append(low.Files, LoweredFile{Path: path, Src: src, Out: out, Map: m})
}

func (f *fakeCompiler) Diagnose(ctx context.Context, dir string, buffers map[string][]byte) ([]Diagnostic, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.diags[dir]), nil
}

func (f *fakeCompiler) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
	return nil
}

func (f *fakeCompiler) isClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed
}

type diagSpec struct {
	Path, Severity, Rule, Message string
	Start, End                    [2]int
}

func compilerDiag(t *testing.T, ds diagSpec) Diagnostic {
	t.Helper()
	line, err := json.Marshal(map[string]any{
		"schema_version": "1.0",
		"severity":       ds.Severity,
		"rule_id":        ds.Rule,
		"file":           ds.Path,
		"span": map[string]int{
			"start_line": ds.Start[0], "start_col": ds.Start[1], "end_line": ds.End[0], "end_col": ds.End[1],
		},
		"message":            ds.Message,
		"autofix_candidates": []any{},
	})
	if err != nil {
		t.Fatal(err)
	}
	d, ok := parseDiagnostic(line)
	if !ok {
		t.Fatalf("parseDiagnostic rejected %s", line)
	}
	return d
}

func findPos(t *testing.T, text, needle string, nth int) Position {
	t.Helper()
	off := -1
	for i := 0; i <= nth; i++ {
		next := strings.Index(text[off+1:], needle)
		if next < 0 {
			t.Fatalf("%q occurs fewer than %d times in %q", needle, nth+1, text)
		}
		off += 1 + next
	}
	return newText([]byte(text)).position(off)
}

func shift(p Position, n int) Position { return Position{Line: p.Line, Character: p.Character + n} }

func span(p Position, n int) Range { return Range{Start: p, End: shift(p, n)} }

func lineOf(text string, n int) string { return strings.Split(text, "\n")[n] }
