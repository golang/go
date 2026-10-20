// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"bytes"
	"encoding/json"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf16"
)

// fixture names a source file of the fixtures: the compiler's hints
// fixtures serve the render tests, so each test reads what the compiler
// really wrote for the files the compiler tests pin.
type fixture struct {
	path string
	src  []byte
}

// compilerFixture loads a fixture of cmd/compile/internal/logos/testdata/hints.
func compilerFixture(t *testing.T, name string) fixture {
	t.Helper()
	path := filepath.Join(testenv.GOROOT(t), "src", "cmd", "compile", "internal", "logos", "testdata", "hints", name)
	return loadFixture(t, path)
}

// localFixture loads a fixture of this package's testdata.
func localFixture(t *testing.T, name string) fixture {
	t.Helper()
	path, err := filepath.Abs(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return loadFixture(t, path)
}

func loadFixture(t *testing.T, path string) fixture {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return fixture{path: path, src: src}
}

// importConfig writes the import config of the packages loGo/s lowering
// imports and returns its path.
func importConfig(t *testing.T, dir string) string {
	t.Helper()
	cmd := exec.Command(testenv.GoToolPath(t), "list", "-export", "-deps",
		"-f", "{{if .Export}}packagefile {{.ImportPath}}={{.Export}}{{end}}", "logos/rt", "strconv")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go list -export: %v", err)
	}
	path := filepath.Join(dir, "importcfg")
	if err := os.WriteFile(path, out, 0o666); err != nil {
		t.Fatal(err)
	}
	return path
}

// compile runs go tool compile over f with extra flags and returns its
// stderr and stdout.
func compile(t *testing.T, f fixture, flags ...string) []byte {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	args := []string{"tool", "compile", "-p", "p", "-importcfg", importConfig(t, dir), "-o", filepath.Join(dir, "o.a")}
	args = append(append(args, flags...), f.path)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			t.Fatal(err)
		}
	}
	return out.Bytes()
}

// compileHints returns the report the compiler wrote for f in hints mode.
func compileHints(t *testing.T, f fixture) (*Report, json.RawMessage) {
	t.Helper()
	out := filepath.Join(t.TempDir(), "hints.json")
	if msg := compile(t, f, "-logos.mode=hints", "-logos.hintsout="+out); len(msg) != 0 {
		t.Fatalf("hints compile printed %q", msg)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("hints compile wrote no report: %v", err)
	}
	r, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	return r, data
}

// positionAt converts a byte offset of src to an LSP position, independently
// of the code under test.
func positionAt(src []byte, off int) Position {
	before := string(src[:off])
	text := before[strings.LastIndex(before, "\n")+1:]
	return Position{Line: strings.Count(before, "\n"), Character: len(utf16.Encode([]rune(text)))}
}

func startOf(t *testing.T, src []byte, needle string) Position {
	t.Helper()
	i := bytes.Index(src, []byte(needle))
	if i < 0 {
		t.Fatalf("fixture lacks %q", needle)
	}
	return positionAt(src, i)
}

func endOf(t *testing.T, src []byte, needle string) Position {
	t.Helper()
	i := bytes.Index(src, []byte(needle))
	if i < 0 {
		t.Fatalf("fixture lacks %q", needle)
	}
	return positionAt(src, i+len(needle))
}

// wholeFile is a range that covers every position of any fixture.
var wholeFile = Range{End: Position{Line: 1 << 20}}

// labelsAt returns the labels of the hints at pos, in order.
func labelsAt(hs []InlayHint, pos Position) []string {
	var out []string
	for _, h := range hs {
		if h.Position == pos {
			out = append(out, h.Label)
		}
	}
	return out
}

// TestInlayNarrowedAndUnion checks the narrowed-type and union-member hints
// that the compiler wrote for a loGo/s file (SPEC Sec 21.2).
func TestInlayNarrowedAndUnion(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	r, _ := compileHints(t, f)
	hs := InlayHints(r, f.path, wholeFile)
	for _, c := range []struct {
		pos   Position
		label string
	}{
		{endOf(t, f.src, "\t\treturn u"), "/* narrowed: *User */"},
		{endOf(t, f.src, "\t\treturn \"é😀\" + u"), "/* narrowed: *User */"},
		{endOf(t, f.src, "switch s"), "/* union: Circle | Square */"},
	} {
		if got := labelsAt(hs, c.pos); !slices.Contains(got, c.label) {
			t.Errorf("hints at %+v: %q, want %q", c.pos, got, c.label)
		}
	}
	for _, h := range hs {
		if strings.HasPrefix(h.Label, "/* narrowed") && h.Position.Line == startOf(t, f.src, "if u != nil").Line {
			t.Errorf("the nil test itself carries %q", h.Label)
		}
	}
}

// TestInlayAllocationBadges checks the four tier badges and that a plain Go
// file gets them too (SPEC Sec 21.2).
func TestInlayAllocationBadges(t *testing.T) {
	lgo := compilerFixture(t, "sites.lgo")
	r, _ := compileHints(t, lgo)
	hs := InlayHints(r, lgo.path, wholeFile)
	for _, c := range []struct{ needle, badge string }{
		{`&User{Name: "kept"}`, "[heap]"},
		{`&User{Name: "tmp"}`, "[stack]"},
		{"primes", "[rodata]"},
	} {
		if got := labelsAt(hs, startOf(t, lgo.src, c.needle)); !slices.Contains(got, c.badge) {
			t.Errorf("%s: hints %q, want %s", c.needle, got, c.badge)
		}
	}
	goFile := compilerFixture(t, "alloc.go")
	gr, _ := compileHints(t, goFile)
	ghs := InlayHints(gr, goFile.path, wholeFile)
	for _, c := range []struct{ needle, badge string }{
		{"&T{1, 2}", "[heap]"},
		{"&T{3, 4}", "[stack]"},
	} {
		if got := labelsAt(ghs, startOf(t, goFile.src, c.needle)); !slices.Contains(got, c.badge) {
			t.Errorf("Go file %s: hints %q, want %s", c.needle, got, c.badge)
		}
	}
}

// TestInlayArenaBadge checks that the renderer shows the arena tier when a
// report carries it.
func TestInlayArenaBadge(t *testing.T) {
	r := &Report{Files: []File{{File: "a.lgo", Allocs: []Alloc{
		{Position: Position{Line: 3, Character: 4}, Tier: TierArena, Message: "make([]byte, n) does not escape"},
	}}}}
	hs := InlayHints(r, "/w/a.lgo", wholeFile)
	if len(hs) != 1 || hs[0].Label != "[arena]" || hs[0].Tooltip != "make([]byte, n) does not escape" {
		t.Errorf("arena hint: %+v", hs)
	}
}

// TestInlayHonorsRange checks that only the hints inside the requested range
// come back.
func TestInlayHonorsRange(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	r, _ := compileHints(t, f)
	from := startOf(t, f.src, "func narrowed(")
	to := startOf(t, f.src, "func narrowedUnicode(")
	var narrowed []InlayHint
	for _, h := range InlayHints(r, f.path, Range{Start: from, End: to}) {
		if strings.HasPrefix(h.Label, "/* narrowed") {
			narrowed = append(narrowed, h)
		}
	}
	if len(narrowed) != 1 || narrowed[0].Position.Line >= to.Line || narrowed[0].Position.Line < from.Line {
		t.Errorf("narrowed hints in the narrowed function: %+v", narrowed)
	}
}

// TestInlayUnknownFile checks that a file the report does not cover yields no
// hints.
func TestInlayUnknownFile(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	r, _ := compileHints(t, f)
	if hs := InlayHints(r, "/w/other.lgo", wholeFile); len(hs) != 0 {
		t.Errorf("hints for an unknown file: %+v", hs)
	}
}

// TestHoverLoweredSites checks the hover of `?`, `const switch T`, and
// `$"..."`: the lowered code of the site (SPEC Sec 21.3).
func TestHoverLoweredSites(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	r, _ := compileHints(t, f)
	for _, c := range []struct {
		name   string
		pos    Position
		title  string
		inCode []string
	}{
		{"try", startOf(t, f.src, "?\n\t_ = n"), "`?`", []string{"get()", "!= nil", "return"}},
		{"const switch", startOf(t, f.src, "const switch"), "`const switch`", []string{"switch"}},
		{"interpolation", startOf(t, f.src, `$"user {name}!"`), "`$\"...\"`", []string{"make([]byte, 0, 6+len("}},
	} {
		h := HoverAt(r, f.path, c.pos)
		if h == nil {
			t.Errorf("%s: no hover at %+v", c.name, c.pos)
			continue
		}
		if h.Contents.Kind != "markdown" || !strings.Contains(h.Contents.Value, c.title) {
			t.Errorf("%s: hover %+v lacks title %s", c.name, h.Contents, c.title)
		}
		for _, frag := range c.inCode {
			if !strings.Contains(h.Contents.Value, frag) {
				t.Errorf("%s: hover lacks %q:\n%s", c.name, frag, h.Contents.Value)
			}
		}
		if h.Range == nil || h.Range.Start.Line != c.pos.Line {
			t.Errorf("%s: hover range %+v", c.name, h.Range)
		}
	}
	if h := HoverAt(r, f.path, startOf(t, f.src, "func area")); h != nil {
		t.Errorf("hover outside any site: %+v", h)
	}
}

// TestHoverFixedBytes checks the `Fixed: <n>B + dynamic` line of an
// interpolation hover with the precomputed fixed byte count (SPEC Sec 21.3).
func TestHoverFixedBytes(t *testing.T) {
	f := compilerFixture(t, "interp.lgo")
	r, _ := compileHints(t, f)
	for _, c := range []struct{ needle, fixed string }{
		{`$"user {name}: {n} items, {ratio}, {ok}, {raw}"`, "Fixed: 68B + dynamic"},
		{`$"{{id}}={n}"`, "Fixed: 25B + dynamic"},
		{`$"plain"`, "Fixed: 5B + dynamic"},
	} {
		h := HoverAt(r, f.path, startOf(t, f.src, c.needle))
		if h == nil || !strings.Contains(h.Contents.Value, c.fixed) {
			t.Errorf("%s: hover %+v, want %q", c.needle, h, c.fixed)
		}
	}
}

// TestHoverFixedBytesOnlyForInterpolation checks that no other site shows a
// byte count.
func TestHoverFixedBytesOnlyForInterpolation(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	r, _ := compileHints(t, f)
	for _, needle := range []string{"?\n\t_ = n", "const switch"} {
		h := HoverAt(r, f.path, startOf(t, f.src, needle))
		if h == nil || strings.Contains(h.Contents.Value, "Fixed:") {
			t.Errorf("hover at %q: %+v", needle, h)
		}
	}
}

// TestServe checks the JSON-RPC entry point the proxy calls: inlayHint
// returns an array, hover returns a Hover or null, and anything else is an
// error.
func TestServe(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	_, data := compileHints(t, f)
	uri := "file://" + filepath.ToSlash(f.path)
	inlay := mustJSON(t, map[string]any{
		"textDocument": map[string]any{"uri": uri},
		"range":        wholeFile,
	})
	out, err := Serve("textDocument/inlayHint", inlay, data)
	if err != nil {
		t.Fatal(err)
	}
	var hs []InlayHint
	if err := json.Unmarshal(out, &hs); err != nil || len(hs) == 0 {
		t.Fatalf("inlayHint result %s: %v", out, err)
	}
	hover := mustJSON(t, map[string]any{
		"textDocument": map[string]any{"uri": uri},
		"position":     startOf(t, f.src, "const switch"),
	})
	out, err = Serve("textDocument/hover", hover, data)
	var h Hover
	if err != nil || json.Unmarshal(out, &h) != nil || !strings.Contains(h.Contents.Value, "switch") {
		t.Fatalf("hover result %s: %v", out, err)
	}
	miss := mustJSON(t, map[string]any{
		"textDocument": map[string]any{"uri": uri},
		"position":     startOf(t, f.src, "func area"),
	})
	if out, err := Serve("textDocument/hover", miss, data); err != nil || out != nil {
		t.Errorf("hover outside a site: %s, %v", out, err)
	}
	if _, err := Serve("textDocument/definition", hover, data); err == nil {
		t.Error("Serve accepted a method it does not own")
	}
	if _, err := Serve("textDocument/hover", hover, json.RawMessage(`{"schemaVersion":"9.9"}`)); err == nil {
		t.Error("Serve accepted a report of another schema version")
	}
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// diagnostics returns the M9 JSON diagnostics the compiler prints for f
// under the default lint policy.
func diagnostics(t *testing.T, f fixture) []Diagnostic {
	t.Helper()
	policy := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(policy, []byte(`{"version":1}`), 0o666); err != nil {
		t.Fatal(err)
	}
	var out []Diagnostic
	for line := range bytes.SplitSeq(compile(t, f, "-logos.json", "-logos.cfg="+policy), []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		d, err := DecodeDiagnostic(line)
		if err != nil {
			t.Fatalf("diagnostic line %q: %v", line, err)
		}
		out = append(out, d)
	}
	return out
}

// applyEdits applies LSP text edits to src; the fixtures are ASCII, so a
// character is a byte.
func applyEdits(t *testing.T, src []byte, edits []TextEdit) string {
	t.Helper()
	offset := func(p Position) int {
		off := 0
		for range p.Line {
			off += bytes.IndexByte(src[off:], '\n') + 1
		}
		return off + p.Character
	}
	slices.SortFunc(edits, func(a, b TextEdit) int { return offset(b.Range.Start) - offset(a.Range.Start) })
	text := string(src)
	for _, e := range edits {
		text = text[:offset(e.Range.Start)] + e.NewText + text[offset(e.Range.End):]
	}
	return text
}

// TestCodeActionsApplyAutofix checks that each M9 autofix candidate becomes a
// quick fix whose edit rewrites the source the way the candidate says: a
// replacement over the span, and an insertion before a line (SPEC Sec 21.4).
func TestCodeActionsApplyAutofix(t *testing.T) {
	const uri = "file:///w/fixes.lgo"
	f := localFixture(t, "fixes.lgo")
	diags := diagnostics(t, f)
	if len(diags) != 1 || len(diags[0].AutofixCandidates) != 2 {
		t.Fatalf("diagnostics %+v", diags)
	}
	line := Range{Start: startOf(t, f.src, "\treturn u.Name"), End: endOf(t, f.src, "\treturn u.Name")}
	actions := CodeActions(diags, uri, f.src, line)
	want := map[string]string{
		"use optional chaining with a ?? default": "package p\n\ntype User struct{ Name string }\n\nfunc name(u *User?) string {\n\treturn (u?.Name ?? \"\")\n}\n",
		"return early when u is nil":              "package p\n\ntype User struct{ Name string }\n\nfunc name(u *User?) string {\n\tif u == nil {\n\t\treturn \"\"\n\t}\n\treturn u.Name\n}\n",
	}
	if len(actions) != len(want) {
		t.Fatalf("%d actions, want %d: %+v", len(actions), len(want), actions)
	}
	for _, a := range actions {
		wantText, ok := want[a.Title]
		if !ok || a.Kind != "quickfix" || len(a.Diagnostics) != 1 || a.Diagnostics[0].Code != "correctness::nil_deref_unproven" {
			t.Errorf("action %+v", a)
			continue
		}
		if got := applyEdits(t, f.src, a.Edit.Changes[uri]); got != wantText {
			t.Errorf("%s:\n got %q\nwant %q", a.Title, got, wantText)
		}
	}
}

// TestCodeActionsAddTry checks the `?` fix of a discarded error.
func TestCodeActionsAddTry(t *testing.T) {
	const uri = "file:///w/discard.lgo"
	f := localFixture(t, "discard.lgo")
	diags := diagnostics(t, f)
	cursor := startOf(t, f.src, "f()\n\treturn nil")
	actions := CodeActions(diags, uri, f.src, Range{Start: cursor, End: cursor})
	if len(actions) != 1 || actions[0].Title != "add ? to propagate the error" {
		t.Fatalf("actions %+v", actions)
	}
	want := "package p\n\nfunc f() error { return nil }\n\nfunc g() error {\n\tf()?\n\treturn nil\n}\n"
	if got := applyEdits(t, f.src, actions[0].Edit.Changes[uri]); got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

// TestCodeActionsOutsideRange checks that a diagnostic away from the request
// range offers no action.
func TestCodeActionsOutsideRange(t *testing.T) {
	f := localFixture(t, "discard.lgo")
	diags := diagnostics(t, f)
	top := Range{Start: Position{}, End: Position{Line: 1}}
	if actions := CodeActions(diags, "file:///w/discard.lgo", f.src, top); len(actions) != 0 {
		t.Errorf("actions away from the diagnostic: %+v", actions)
	}
}

// TestCodeActionsUTF16 checks that a fix on a line with non-ASCII text lands
// on UTF-16 columns, not byte columns.
func TestCodeActionsUTF16(t *testing.T) {
	src := []byte("x := \"é😀\" + f()\n")
	at := bytes.Index(src, []byte("f()"))
	d := Diagnostic{
		SchemaVersion: "1.0", Severity: "deny", RuleID: "slop::discarded_error",
		Span:              Span{StartLine: 1, StartCol: at + 1, EndLine: 1, EndCol: at + 4},
		AutofixCandidates: []Fix{{Description: "add ? to propagate the error", Replacement: "f()?"}},
	}
	const uri = "file:///w/u.lgo"
	actions := CodeActions([]Diagnostic{d}, uri, src, wholeFile)
	if len(actions) != 1 {
		t.Fatalf("actions %+v", actions)
	}
	edit := actions[0].Edit.Changes[uri][0]
	want := Range{Start: startOf(t, src, "f()"), End: endOf(t, src, "f()")}
	if edit.Range != want || want.Start.Character != 13 {
		t.Errorf("edit range %+v, want %+v (character 13)", edit.Range, want)
	}
}
