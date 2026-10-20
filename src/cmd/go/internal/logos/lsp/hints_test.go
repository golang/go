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
)

// hintsRecorder is a HintsFunc and HintsJSONFunc pair that records its
// calls, standing in for M21-S2's package.
type hintsRecorder struct {
	mu       sync.Mutex
	runs     int
	buffers  []map[string][]byte
	calls    []hintCall
	failWith error
}

// hintCall is one call of the hints hook.
type hintCall struct {
	method string
	params json.RawMessage
	hints  json.RawMessage
}

// run is the HintsJSON hook: it returns a document that numbers the runs.
func (h *hintsRecorder) run(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.runs++
	h.buffers = append(h.buffers, buffers)
	return json.RawMessage(`{"run":` + strconv.Itoa(h.runs) + `}`), nil
}

// serve is the Hints hook.
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

// snapshot returns the recorded runs and calls.
func (h *hintsRecorder) snapshot() (int, []hintCall) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.runs, slices.Clone(h.calls)
}

// hintsFixture serves aLgo with a fake gopls that answers hover and
// inlayHint, and the recorder wired as the hints hooks.
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

// TestProxyHintsHook checks the M21-S2 hook: for inlayHint and hover the
// proxy calls the hints function with the method, the client's own params,
// and the package's hints JSON, appends its hints to gopls's, joins its
// hover text under gopls's, runs the hints runner once per compile, and
// keeps the answer of gopls when the hook fails.
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

// TestProxyHintsOnGoDocuments checks that hints reach Go documents too:
// the request goes to gopls unchanged, and the hook's hints join its
// answer.
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

// TestProxyWithoutHintsHookServesGopls checks that the hook is optional:
// with none set, inlay hints are what gopls answered.
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

// TestProxyCodeActionsHook checks the autofix hook: it gets the compiler
// diagnostic lines of the document, its URI and text, and the request
// range as the client sent it, and its actions follow gopls's.
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

// TestProxyRecompilesAfterDebounce checks the background path: with a
// debounce delay set, an edit reaches gopls without any request, and one
// recompile covers a burst of edits.
func TestProxyRecompilesAfterDebounce(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Tune: func(f *fixture, o *Options) { o.Debounce = 80 * time.Millisecond },
	})
	f.open("a/a.lgo", 1)
	f.sync()
	before := len(f.comp.lowerCalls())
	for v := 2; v <= 4; v++ {
		f.client.notify("textDocument/didChange", map[string]any{
			"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": v},
			"contentChanges": []any{map[string]any{"text": aLgo + "// v\n"}},
		})
	}
	eventually(t, "the background recompile", func() bool { return len(f.comp.lowerCalls()) > before })
	f.sync()
	if got := len(f.receivedMethod("textDocument/didChange")); got != 1 {
		t.Errorf("gopls got %d lowered changes, want the one of the background recompile", got)
	}
	if got := len(f.comp.lowerCalls()) - before; got != 1 {
		t.Errorf("a burst of 3 edits caused %d recompiles, want 1", got)
	}
	data, _ := os.ReadFile(f.path("a/a.lgo"))
	if string(data) != aLgo {
		t.Errorf("the file on disk changed to %q", data)
	}
}

// TestProxyHintsCacheKeepsEmptyAnswer checks the cache keys on a hit flag,
// not on the content: a runner whose answer is JSON null is not run again
// for the next request.
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

// TestProxyDoesNotCacheHintsOfAnOlderCompile checks the generation guard: a
// hints run that finishes after the package was edited and compiled again
// is not kept, so the next request runs the runner on the new text.
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

// eventually polls cond until it holds, failing after waitLimit.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(waitLimit)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
