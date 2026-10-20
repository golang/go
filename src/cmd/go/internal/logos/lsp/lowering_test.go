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
)

// aLgo is a loGo/s source with a ? statement, the lowering of which the
// fake compiler imitates.
const aLgo = `package a

import "os"

func Read(path string) (string, error) {
	data := os.ReadFile(path)?
	return string(data), nil
}
`

// questionLine matches a statement the fake compiler lowers: an assignment
// whose call ends in ?.
var questionLine = regexp.MustCompile(`^(\s*)(\w+) := (.+)\?$`)

// siteTrailer is the declaration the fake lowering appends, as the real
// one appends its error sites.
const siteTrailer = "\nvar _logosSite0 = 0\n"

// lowerQuestion imitates the lowering of ? statements. Each statement
// `x := f()?` becomes three lines (the call with its error, a trace, and
// the error return), and a declaration is appended at the end of the file.
// A source holding "SYNTAX ERROR" does not lower.
func lowerQuestion(src []byte) ([]byte, EditMap, bool) {
	if bytes.Contains(src, []byte("SYNTAX ERROR")) {
		return nil, EditMap{}, false
	}
	type step struct {
		span EditSpan
		text string
	}
	var steps []step
	off := 0
	for _, line := range strings.SplitAfter(string(src), "\n") {
		text := strings.TrimSuffix(line, "\n")
		if m := questionLine.FindStringSubmatch(text); m != nil {
			indent := m[1]
			repl := fmt.Sprintf("%s, _e := %s\n%s_logosTrace(_e)\n%sif _e != nil { return \"\", _e }", m[2], m[3], indent, indent)
			steps = append(steps, step{EditSpan{Start: off + len(indent), End: off + len(text), Len: len(repl)}, repl})
		}
		off += len(line)
	}
	steps = append(steps, step{EditSpan{Start: len(src), End: len(src), Len: len(siteTrailer)}, siteTrailer})
	var out []byte
	var spans []EditSpan
	prev := 0
	for _, s := range steps {
		out = append(out, src[prev:s.span.Start]...)
		out = append(out, s.text...)
		prev = s.span.End
		spans = append(spans, s.span)
	}
	out = append(out, src[prev:]...)
	return out, EditMap{Version: EditMapVersion, Size: len(src), Out: len(out), Edits: spans}, true
}

// fakeCompiler is a Compiler that lowers with lowerQuestion. Every .lgo
// file of a package directory lowers; a source holding "SYNTAX ERROR" is
// missing from the result, as a file of a package that fails to compile
// is. It never writes a file.
type fakeCompiler struct {
	mu      sync.Mutex
	closure map[string][]string
	diags   map[string][]Diagnostic
	lowered [][]string
	buffers []map[string][]byte
	closed  bool
}

// newFakeCompiler returns a fake compiler with no closures and no
// diagnostics.
func newFakeCompiler() *fakeCompiler {
	return &fakeCompiler{closure: map[string][]string{}, diags: map[string][]Diagnostic{}}
}

// setClosure makes the packages of dirs, the first being the package
// itself, the dependency closure of the package in dirs[0].
func (f *fakeCompiler) setClosure(dirs ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closure[dirs[0]] = dirs
}

// setDiagnostics sets the diagnostics Diagnose returns for dir.
func (f *fakeCompiler) setDiagnostics(dir string, diags ...Diagnostic) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.diags[dir] = diags
}

// lowerCalls returns the directories Lower was called with, in order.
func (f *fakeCompiler) lowerCalls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var dirs []string
	for _, l := range f.lowered {
		dirs = append(dirs, l[0])
	}
	return dirs
}

// bufferCalls returns the buffers of each Lower call.
func (f *fakeCompiler) bufferCalls() []map[string][]byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.buffers)
}

// Lower implements Compiler.
func (f *fakeCompiler) Lower(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, error) {
	f.mu.Lock()
	dirs := f.closure[dir]
	if dirs == nil {
		dirs = []string{dir}
	}
	f.lowered = append(f.lowered, dirs)
	f.buffers = append(f.buffers, buffers)
	f.mu.Unlock()
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

// lowerFile adds the lowering of the source at path to low.
func (f *fakeCompiler) lowerFile(low *Lowering, path string, buffers map[string][]byte) {
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

// Diagnose implements Compiler.
func (f *fakeCompiler) Diagnose(ctx context.Context, dir string, buffers map[string][]byte) ([]Diagnostic, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.diags[dir]), nil
}

// Close implements io.Closer, so a test sees the proxy release its compiler.
func (f *fakeCompiler) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
	return nil
}

// isClosed reports whether the proxy closed the compiler.
func (f *fakeCompiler) isClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed
}

// diagSpec describes a compiler diagnostic for compilerDiag. Start and End
// are 1-based line and byte column pairs.
type diagSpec struct {
	Path, Severity, Rule, Message string
	Start, End                    [2]int
}

// compilerDiag builds the compiler diagnostic ds describes, and keeps its
// JSON line as Raw, as the proxy does for a line of the compiler.
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

// findPos returns the position of the nth (from 0) occurrence of needle in
// text, as the editor would report it.
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

// shift returns p moved n characters to the right on its line.
func shift(p Position, n int) Position { return Position{Line: p.Line, Character: p.Character + n} }

// span returns the range of n characters from p.
func span(p Position, n int) Range { return Range{Start: p, End: shift(p, n)} }

// lineOf returns line n (from 0) of text, without its newline.
func lineOf(text string, n int) string { return strings.Split(text, "\n")[n] }
