// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAdvStaleVersionIgnored(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 5)
	aURI := f.uri("a/a.lgo")
	changed := strings.Replace(aLgo, "import \"os\"\n", "// v6\nimport \"os\"\n", 1)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 6},
		"contentChanges": []any{map[string]any{"text": changed}},
	})
	f.client.call("textDocument/hover", positionParams(aURI, Position{}))
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 3},
		"contentChanges": []any{map[string]any{"text": aLgo}},
	})
	f.client.call("textDocument/hover", positionParams(aURI, Position{}))
	calls := f.comp.bufferCalls()
	last := calls[len(calls)-1]
	for p, v := range last {
		if strings.HasSuffix(p, "a.lgo") && string(v) != changed {
			t.Errorf("stale version 3 overwrote version 6: got %q, want the version 6 text", v)
		}
	}
}

func TestAdvChangeBeforeOpenAnswersNull(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aURI := f.uri("a/a.lgo")
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 1},
		"contentChanges": []any{map[string]any{"text": aLgo}},
	})
	res := f.client.call("textDocument/hover", positionParams(aURI, Position{Line: 5, Character: 3}))
	if string(res) != "null" {
		t.Errorf("hover after change-before-open = %s, want null (no document state)", res)
	}
	if got := len(f.comp.lowerCalls()); got != 0 {
		t.Errorf("change-before-open compiled %d times, want none (no document state)", got)
	}
}

func TestAdvStaleGoChangeNotForwarded(t *testing.T) {
	f := newFixture(t, map[string]string{"p/p.go": "package p\n"}, fixtureOptions{})
	pURI := f.uri("p/p.go")
	f.open("p/p.go", 1)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": pURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": "package p\n// v2\n"}},
	})
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": pURI, "version": 1},
		"contentChanges": []any{map[string]any{"text": "package p\n// stale\n"}},
	})
	f.client.call("textDocument/hover", positionParams(pURI, Position{Line: 0, Character: 0}))
	var versions []int
	for _, m := range f.receivedMethod("textDocument/didChange") {
		td, _ := m["textDocument"].(map[string]any)
		uri, _ := td["uri"].(string)
		if uri != pURI {
			continue
		}
		v, _ := number(td["version"])
		versions = append(versions, v)
	}
	if !slices.Equal(versions, []int{2}) {
		t.Errorf("gopls saw didChange versions %v, want only the newest [2]", versions)
	}
}

func TestAdvCloseUnknownPathIsQuiet(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{"contents": "mapped"}),
			}}
		},
	})
	unsaved := aLgo + "// unsaved\n"
	f.openText("a/a.lgo", 1, unsaved)
	f.client.notify("textDocument/didClose", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("missing/m.lgo")},
	})
	res := f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	if !strings.Contains(string(res), "mapped") {
		t.Errorf("hover after closing an unknown path = %s, want the mapped answer", res)
	}
	last := f.comp.bufferCalls()[len(f.comp.bufferCalls())-1]
	v, ok := last[f.path("a/a.lgo")]
	if !ok {
		t.Fatalf("closing an unknown path dropped the open document from the compile buffers: %v", last)
	}
	if string(v) != unsaved {
		t.Errorf("closing an unknown path discarded the open document: got %q, want the unsaved text", v)
	}
}

func TestAdvDuplicateOpenResetsText(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	other := strings.Replace(aLgo, "import \"os\"\n", "// other\nimport \"os\"\n", 1)
	f.openText("a/a.lgo", 1, other)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": other + "// more\n"}},
	})
	f.openText("a/a.lgo", 1, aLgo)
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))
	calls := f.comp.bufferCalls()
	last := calls[len(calls)-1]
	for p, v := range last {
		if strings.HasSuffix(p, "a.lgo") && string(v) != aLgo {
			t.Errorf("re-open did not reset the document: got %q, want the reopened text", v)
		}
	}
}

func TestAdvUntrackedURINotLowered(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": "untracked-scheme:a/a.lgo", "languageId": "logos", "version": 1, "text": aLgo},
	})
	f.client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/note.txt"), "languageId": "plaintext", "version": 1, "text": "hi"},
	})
	f.open("a/a.lgo", 1)
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))
	low, _, ok := lowerQuestion([]byte(aLgo))
	if !ok {
		t.Fatal("aLgo does not lower")
	}
	var lowered []string
	var forwarded []string
	for _, m := range f.receivedMethod("textDocument/didOpen") {
		td, _ := m["textDocument"].(map[string]any)
		uri, _ := td["uri"].(string)
		text, _ := td["text"].(string)
		if text == string(low) {
			lowered = append(lowered, uri)
			continue
		}
		forwarded = append(forwarded, uri)
	}
	if want := []string{f.virtURI("a/a.lgo")}; !slices.Equal(lowered, want) {
		t.Errorf("lowered documents opened in gopls = %v, want %v", lowered, want)
	}
	slices.Sort(forwarded)
	if want := []string{f.uri("a/note.txt"), "untracked-scheme:a/a.lgo"}; !slices.Equal(forwarded, want) {
		t.Errorf("untracked documents forwarded to gopls = %v, want %v", forwarded, want)
	}
}

func TestAdvConcurrentChangesShareStore(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{"contents": "mapped"}),
			}}
		},
	})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			f.client.notify("textDocument/didChange", map[string]any{
				"textDocument":   map[string]any{"uri": aURI, "version": 10 + i},
				"contentChanges": []any{map[string]any{"text": aLgo + fmt.Sprintf("// %d\n", i)}},
			})
		})
	}
	wg.Wait()
	final := aLgo + "// final\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 100},
		"contentChanges": []any{map[string]any{"text": final}},
	})
	res := f.client.call("textDocument/hover", positionParams(aURI, Position{Line: 5, Character: 3}))
	if !strings.Contains(string(res), "mapped") {
		t.Errorf("hover after concurrent edits = %s, want the mapped answer", res)
	}
	last := f.comp.bufferCalls()[len(f.comp.bufferCalls())-1]
	if got := string(last[f.path("a/a.lgo")]); got != final {
		t.Errorf("final text after concurrent edits = %q, want the version 100 text", got)
	}
}

func TestAdvDidSaveUnopenedSkipsCompile(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	f.client.notify("textDocument/didSave", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("missing/m.lgo")},
	})
	if got := len(f.comp.lowerCalls()); got != 0 {
		t.Errorf("didSave on an unopened document compiled %d times, want none", got)
	}
}

func TestAdvEnsureCancelledSkipsCompile(t *testing.T) {
	p := newProxy(io.Discard, Options{Compiler: newFakeCompiler()})
	p.server = &writer{w: io.Discard}
	p.st.setDoc("file:///w/a.lgo", "/w/a.lgo", 1, newText([]byte("package a\n")))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		p.ensure(ctx)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("ensure with a canceled context did not return")
	}
	if got := len(p.opts.Compiler.(*fakeCompiler).lowerCalls()); got != 0 {
		t.Errorf("canceled ensure compiled %d times, want none", got)
	}
}

func TestAdvUTF16AstralBoundary(t *testing.T) {
	const src = "package a\n\nvar s = \"\U0001F600\"\n"
	tx := newText([]byte(src))
	emojiOff := strings.Index(src, "\U0001F600")
	prefix := tx.position(emojiOff)
	for _, ch := range []int{1, 2} {
		if got := tx.offset(Position{Line: prefix.Line, Character: prefix.Character + ch}); got != emojiOff+4 {
			t.Errorf("offset past astral char %+1d = %d, want the rune end %d", ch, got, emojiOff+4)
		}
	}
	if got := tx.position(emojiOff + 4); got.Character != prefix.Character+2 {
		t.Errorf("position after astral char = %v, want character %d", got, prefix.Character+2)
	}
}

func TestAdvOverlayReleasedAfterRun(t *testing.T) {
	ov, err := newOverlay(map[string][]byte{"/w/a.lgo": []byte("package a\n")})
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Dir(ov.json)
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("overlay dir missing before release: %v", err)
	}
	if err := ov.release(); err != nil {
		t.Fatalf("release = %v", err)
	}
	if err := ov.release(); err != nil {
		t.Errorf("second release = %v, want nil", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("overlay dir still present after release: %v", err)
	}
}

func TestAdvStaleLgoChangeNotForwarded(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aURI := f.uri("a/a.lgo")
	v2 := aLgo + "// v2\n"
	stale := aLgo + "// stale\n"
	f.open("a/a.lgo", 1)
	f.client.call("textDocument/hover", positionParams(aURI, Position{Line: 5, Character: 3}))
	lowV2, _, ok := lowerQuestion([]byte(v2))
	if !ok {
		t.Fatal("v2 does not lower")
	}
	lowStale, _, ok := lowerQuestion([]byte(stale))
	if !ok {
		t.Fatal("stale does not lower")
	}
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": v2}},
	})
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 1},
		"contentChanges": []any{map[string]any{"text": stale}},
	})
	f.client.call("textDocument/hover", positionParams(aURI, Position{Line: 5, Character: 3}))
	var changed []string
	for _, m := range f.receivedMethod("textDocument/didChange") {
		td, _ := m["textDocument"].(map[string]any)
		uri, _ := td["uri"].(string)
		if uri != f.virtURI("a/a.lgo") {
			continue
		}
		changes, _ := m["contentChanges"].([]any)
		for _, c := range changes {
			text, _ := c.(map[string]any)["text"].(string)
			changed = append(changed, text)
		}
	}
	if len(changed) != 1 || changed[0] != string(lowV2) {
		t.Errorf("gopls saw %d virtual changes, want the one lowering of version 2", len(changed))
	}
	for _, text := range changed {
		if text == string(lowStale) {
			t.Errorf("the stale version 1 text reached gopls after version 2")
		}
	}
}
