// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"cmd/go/internal/logos/hints"
)

type hintsRecorder struct {
	mu       sync.Mutex
	runs     int
	buffers  []map[string][]byte
	calls    []hintCall
	failWith error
}

type hintCall struct {
	method string
	params json.RawMessage
	hints  json.RawMessage
}

func (h *hintsRecorder) run(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.runs++
	h.buffers = append(h.buffers, buffers)
	return json.RawMessage(`{"run":` + strconv.Itoa(h.runs) + `}`), nil
}

func (h *hintsRecorder) serve(method string, params, hints json.RawMessage) (json.RawMessage, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.calls = append(h.calls, hintCall{method, params, hints})
	if h.failWith != nil {
		return nil, h.failWith
	}
	if method == "textDocument/inlayHint" {
		return json.RawMessage(`[{"position":{"line":5,"character":5},"label":"/* narrowed: *User */"}]`), nil
	}
	return json.RawMessage(`{"contents":{"kind":"markdown","value":"Fixed: 12B + dynamic"}}`), nil
}

func (h *hintsRecorder) snapshot() (int, []hintCall) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.runs, slices.Clone(h.calls)
}

func hintsFixture(t *testing.T, rec *hintsRecorder) *fixture {
	t.Helper()
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowCall := span(findPos(t, string(low), "os.ReadFile", 0), len("os.ReadFile"))
	lowData := shift(findPos(t, string(low), "data", 0), len("data"))
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{
					"contents": map[string]any{"kind": "markdown", "value": "gopls hover"},
					"range":    lowCall,
				}),
				"textDocument/inlayHint": toJSON(t, []any{map[string]any{"position": lowData, "label": ": []byte"}}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			o.Hints = rec.serve
			o.HintsJSON = rec.run
		},
	})
	f.open("a/a.lgo", 1)
	return f
}

func TestProxyHintsHook(t *testing.T) {
	rec := new(hintsRecorder)
	f := hintsFixture(t, rec)
	aURI := f.uri("a/a.lgo")
	origData := shift(findPos(t, aLgo, "data", 0), len("data"))
	origCall := span(findPos(t, aLgo, "os.ReadFile", 0), len("os.ReadFile"))

	inlayParams := map[string]any{
		"textDocument": map[string]any{"uri": aURI},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	}
	res := f.client.call("textDocument/inlayHint", inlayParams)
	sameJSON(t, "inlay hints", res, []any{
		map[string]any{"position": origData, "label": ": []byte"},
		map[string]any{"position": Position{Line: 5, Character: 5}, "label": "/* narrowed: *User */"},
	})

	hoverParams := positionParams(aURI, shift(findPos(t, aLgo, "os.ReadFile", 0), 1))
	res = f.client.call("textDocument/hover", hoverParams)
	sameJSON(t, "hover", res, map[string]any{
		"contents": map[string]any{"kind": "markdown", "value": "gopls hover\n\n---\n\nFixed: 12B + dynamic"},
		"range":    origCall,
	})

	runs, calls := rec.snapshot()
	if runs != 1 {
		t.Errorf("the hints runner ran %d times for two requests, want once", runs)
	}
	if len(calls) != 2 || calls[0].method != "textDocument/inlayHint" || calls[1].method != "textDocument/hover" {
		t.Fatalf("hook calls = %+v, want inlayHint then hover", calls)
	}
	sameJSON(t, "params given to the hook", calls[0].params, inlayParams)
	sameJSON(t, "hover params given to the hook", calls[1].params, hoverParams)
	sameJSON(t, "hints given to the hook", calls[0].hints, map[string]any{"run": 1})

	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "\n"}},
	})
	f.client.call("textDocument/inlayHint", inlayParams)
	runs, calls = rec.snapshot()
	if runs != 2 {
		t.Errorf("after an edit the hints runner ran %d times, want 2", runs)
	}
	sameJSON(t, "hints after the edit", calls[2].hints, map[string]any{"run": 2})
	rec.mu.Lock()
	last := rec.buffers[len(rec.buffers)-1]
	rec.mu.Unlock()
	if !bytes.Equal(last[f.path("a/a.lgo")], []byte(aLgo+"\n")) {
		t.Errorf("the hints runner got buffers %v, want the unsaved text", last)
	}

	rec.mu.Lock()
	rec.failWith = errors.New("hook failed")
	rec.mu.Unlock()
	res = f.client.call("textDocument/inlayHint", inlayParams)
	sameJSON(t, "inlay hints when the hook fails", res, []any{map[string]any{"position": origData, "label": ": []byte"}})
}

func TestProxyHintsOnGoDocuments(t *testing.T) {
	rec := new(hintsRecorder)
	f := newFixture(t, map[string]string{"p/p.go": "package p\n\nvar x = 1\n"}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/inlayHint": toJSON(t, []any{map[string]any{"position": Position{Line: 2, Character: 5}, "label": ": int"}}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			o.Hints = rec.serve
			o.HintsJSON = rec.run
		},
	})
	f.open("p/p.go", 1)
	res := f.client.call("textDocument/inlayHint", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("p/p.go")},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 3}},
	})
	sameJSON(t, "inlay hints for a Go file", res, []any{
		map[string]any{"position": Position{Line: 2, Character: 5}, "label": ": int"},
		map[string]any{"position": Position{Line: 5, Character: 5}, "label": "/* narrowed: *User */"},
	})
}

func TestProxyWithoutHintsHookServesGopls(t *testing.T) {
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowData := shift(findPos(t, string(low), "data", 0), len("data"))
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/inlayHint": toJSON(t, []any{map[string]any{"position": lowData, "label": ": []byte"}}),
			}}
		},
	})
	f.open("a/a.lgo", 1)
	res := f.client.call("textDocument/inlayHint", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	})
	sameJSON(t, "inlay hints", res, []any{
		map[string]any{"position": shift(findPos(t, aLgo, "data", 0), len("data")), "label": ": []byte"},
	})
}

func TestProxyCodeActionsHook(t *testing.T) {
	type call struct {
		diags [][]byte
		uri   string
		src   []byte
		rng   json.RawMessage
	}
	var mu sync.Mutex
	var got []call
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/codeAction": toJSON(t, []any{map[string]any{"title": "from gopls"}}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			o.CodeActions = func(diags [][]byte, uri string, src []byte, rng json.RawMessage) (json.RawMessage, error) {
				mu.Lock()
				defer mu.Unlock()
				got = append(got, call{diags, uri, src, rng})
				return json.RawMessage(`[{"title":"autofix","kind":"quickfix"}]`), nil
			}
		},
	})
	aPath := f.path("a/a.lgo")
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Rule: "style::x", Message: "fixable", Start: [2]int{6, 2}, End: [2]int{6, 6}}),
		compilerDiag(t, diagSpec{Path: filepath.Join(filepath.Dir(aPath), "other.lgo"), Severity: "deny", Message: "elsewhere", Start: [2]int{1, 1}, End: [2]int{1, 2}}),
	)
	f.open("a/a.lgo", 1)
	rng := span(Position{Line: 5, Character: 1}, 4)
	res := f.client.call("textDocument/codeAction", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        rng,
		"context":      map[string]any{"diagnostics": []any{}},
	})
	sameJSON(t, "code actions", res, []any{
		map[string]any{"title": "from gopls"},
		map[string]any{"title": "autofix", "kind": "quickfix"},
	})
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 {
		t.Fatalf("the code action hook ran %d times, want once", len(got))
	}
	c := got[0]
	if len(c.diags) != 1 {
		t.Fatalf("hook got %d diagnostic lines, want the one of this file", len(c.diags))
	}
	if d, ok := parseDiagnostic(c.diags[0]); !ok || d.Message != "fixable" {
		t.Errorf("hook got the line %s, want the compiler's JSON for the diagnostic", c.diags[0])
	}
	if c.uri != f.uri("a/a.lgo") || string(c.src) != aLgo {
		t.Errorf("hook got uri %s and text %q, want the document's", c.uri, c.src)
	}
	sameJSON(t, "range given to the hook", c.rng, rng)
}

func TestProxyHintsCacheKeepsEmptyAnswer(t *testing.T) {
	var mu sync.Mutex
	runs := 0
	rec := new(hintsRecorder)
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Tune: func(f *fixture, o *Options) {
			o.Hints = rec.serve
			o.HintsJSON = func(context.Context, string, map[string][]byte) (json.RawMessage, error) {
				mu.Lock()
				defer mu.Unlock()
				runs++
				return json.RawMessage("null"), nil
			}
		},
	})
	f.open("a/a.lgo", 1)
	params := map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	}
	f.client.call("textDocument/inlayHint", params)
	f.client.call("textDocument/inlayHint", params)
	mu.Lock()
	defer mu.Unlock()
	if runs != 1 {
		t.Errorf("a runner that answers null ran %d times for two requests, want once", runs)
	}
}

func TestProxyDoesNotCacheHintsOfAnOlderCompile(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var mu sync.Mutex
	runs := 0
	rec := new(hintsRecorder)
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Tune: func(f *fixture, o *Options) {
			o.Hints = rec.serve
			o.HintsJSON = func(context.Context, string, map[string][]byte) (json.RawMessage, error) {
				mu.Lock()
				runs++
				first := runs == 1
				mu.Unlock()
				if first {
					close(started)
					<-release
				}
				return json.RawMessage(`{}`), nil
			}
		},
	})
	f.open("a/a.lgo", 1)
	params := map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	}
	id := f.client.request("textDocument/inlayHint", params)
	select {
	case <-started:
	case <-time.After(waitLimit):
		t.Fatal("the hints runner never started")
	}
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "\n"}},
	})
	f.client.call("textDocument/definition", positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	close(release)
	f.client.await("textDocument/inlayHint", id)
	f.client.call("textDocument/inlayHint", params)
	mu.Lock()
	defer mu.Unlock()
	if runs != 2 {
		t.Errorf("the runner ran %d times, want 2: the run of the old text must not be cached", runs)
	}
}

func TestProxyHintsSupersededBufferAreDropped(t *testing.T) {
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowData := shift(findPos(t, string(low), "data", 0), len("data"))
	started, release := make(chan struct{}), make(chan struct{})
	var mu sync.Mutex
	runs := 0
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/inlayHint": toJSON(t, []any{map[string]any{"position": lowData, "label": ": []byte"}}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			o.Hints = func(string, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`[{"position":{"line":5,"character":5},"label":"/* stale */"}]`), nil
			}
			o.HintsJSON = func(context.Context, string, map[string][]byte) (json.RawMessage, error) {
				mu.Lock()
				runs++
				first := runs == 1
				mu.Unlock()
				if first {
					close(started)
					<-release
				}
				return json.RawMessage(`{"run":1}`), nil
			}
		},
	})
	f.open("a/a.lgo", 1)
	params := map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	}
	id := f.client.request("textDocument/inlayHint", params)
	select {
	case <-started:
	case <-time.After(waitLimit):
		t.Fatal("the hints runner never started")
	}
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "\n"}},
	})
	f.sync()
	close(release)
	m := f.client.await("textDocument/inlayHint", id)
	sameJSON(t, "hints from a superseded buffer", m.Result, []any{
		map[string]any{"position": lowData, "label": ": []byte"},
	})
}

func TestProxyHintsEditBeforeResponseRecomputes(t *testing.T) {
	rec := new(hintsRecorder)
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{
				DelayMillis: 200,
				Responses: map[string]json.RawMessage{
					"textDocument/inlayHint": toJSON(t, []any{}),
				},
			}
		},
		Tune: func(f *fixture, o *Options) {
			o.Hints = rec.serve
			o.HintsJSON = rec.run
		},
	})
	f.open("a/a.lgo", 1)
	params := map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
	}
	f.client.call("textDocument/inlayHint", params)
	if runs, _ := rec.snapshot(); runs != 1 {
		t.Fatalf("the hints runner ran %d times for one request, want once", runs)
	}
	id := f.client.request("textDocument/inlayHint", params)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "\n"}},
	})
	f.client.await("textDocument/inlayHint", id)
	if runs, _ := rec.snapshot(); runs != 2 {
		t.Errorf("the hints runner ran %d times across an edit, want 2: cached hints of the older buffer must not be reused", runs)
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	last := rec.buffers[len(rec.buffers)-1]
	if !bytes.Equal(last[f.path("a/a.lgo")], []byte(aLgo+"\n")) {
		t.Errorf("the recompute used %q, want the edited text", last[f.path("a/a.lgo")])
	}
}

func TestProxyHintsGoPackageAlloc(t *testing.T) {
	root, tc := realWorkspace(t)
	dir := filepath.Join(root, "gomod", "alloc")
	if err := os.MkdirAll(dir, 0o777); err != nil {
		t.Fatal(err)
	}
	src := "package alloc\n\nfunc F(v int) []int { return []int{v} }\n"
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	scriptData, err := json.Marshal(fakeScript{})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scriptPath, scriptData, 0o666); err != nil {
		t.Fatal(err)
	}
	client := startProxy(t, Options{
		Gopls:     []string{os.Args[0], "-mode=stdio"},
		Env:       append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
		Compiler:  tc,
		HintsJSON: tc.Hints,
		Hints:     hints.Serve,
	})
	client.call("initialize", map[string]any{
		"processId": nil,
		"rootUri":   pathURI(root),
		"capabilities": map[string]any{
			"general": map[string]any{"positionEncodings": []string{"utf-8", "utf-16"}},
		},
	})
	client.notify("initialized", map[string]any{})
	uri := pathURI(filepath.Join(dir, "a.go"))
	client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": uri, "languageId": "go", "version": 1, "text": src},
	})
	res := client.call("textDocument/inlayHint", map[string]any{
		"textDocument": map[string]any{"uri": uri},
		"range":        Range{Start: Position{Line: 0}, End: Position{Line: 10}},
	})
	var items []map[string]any
	if err := json.Unmarshal(res, &items); err != nil {
		t.Fatalf("inlayHint result is not a list: %v (%s)", err, res)
	}
	for _, it := range items {
		if label, _ := it["label"].(string); label == "[heap]" || label == "[stack]" || label == "[arena]" || label == "[rodata]" {
			return
		}
	}
	t.Errorf("no allocation badge for the Go package in %s", res)
}
