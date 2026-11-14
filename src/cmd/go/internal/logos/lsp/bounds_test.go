// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"errors"
	"io"
	"strconv"
	"strings"
	"testing"
	"time"
)

var errLowerBoom = errors.New("boom")

func TestCancelRequestFreesPendingSlot(t *testing.T) {
	p := newProxy(io.Discard, Options{Stderr: new(syncBuffer)})
	p.server = &writer{w: io.Discard}
	p.initialize(&message{ID: []byte("1"), Method: "initialize", Params: []byte(`{}`)})
	p.clientNotification(t.Context(), &message{Method: "$/cancelRequest", Params: []byte(`{"id":1}`)})
	p.mu.Lock()
	n := len(p.pending)
	p.mu.Unlock()
	if n != 0 {
		t.Fatalf("a cancelled request kept %d pending slots, want none", n)
	}
}

func TestDiagnosticCachesStayBounded(t *testing.T) {
	p := newProxy(io.Discard, Options{Stderr: new(syncBuffer)})
	p.server = &writer{w: io.Discard}
	keepURI := "file:///w/keep.go"
	p.st.setDoc(keepURI, "/w/keep.go", 1, newText([]byte("package keep\n")))
	p.mu.Lock()
	for i := range maxDiagURIs + 2 {
		uri := "file:///w/f" + strconv.Itoa(i) + ".go"
		if i == 0 {
			uri = keepURI
		}
		raw := []byte(`[{"message":"x"}]`)
		p.srvDiags[uri] = serverDiags{raw: raw}
		p.sent[uri] = string(raw)
	}
	p.mu.Unlock()
	extra, err := encode(map[string]any{
		"uri":         "file:///w/extra.go",
		"version":     1,
		"diagnostics": []any{map[string]any{"message": "x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	p.serverDiagnostics(&message{Method: "textDocument/publishDiagnostics", Params: extra})
	p.mu.Lock()
	n := len(p.srvDiags)
	_, kept := p.srvDiags[keepURI]
	p.mu.Unlock()
	if n > maxDiagURIs {
		t.Fatalf("diagnostic cache holds %d uris, want at most %d", n, maxDiagURIs)
	}
	if !kept {
		t.Fatal("eviction dropped the cache of an open document")
	}
}

func TestOversizedDocumentNotStored(t *testing.T) {
	p := newProxy(io.Discard, Options{Stderr: new(syncBuffer)})
	p.server = &writer{w: io.Discard}
	raw, err := encode(map[string]any{"textDocument": map[string]any{
		"uri":        "file:///w/big.lgo",
		"languageId": "logos",
		"version":    1,
		"text":       strings.Repeat("x", (8<<20)+1),
	}})
	if err != nil {
		t.Fatal(err)
	}
	p.didOpen(t.Context(), &message{Method: "textDocument/didOpen", Params: raw})
	if p.st.open("/w/big.lgo") {
		t.Fatal("didOpen stored a document past the size cap")
	}
}

func TestLoweringErrorAnswersRequestFailed(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	f.comp.setLowerError(f.path("a"), errLowerBoom)
	m := f.client.callMessage("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	v, err := decode(m.Error)
	if err != nil {
		t.Fatalf("error payload = %s, want a JSON-RPC error: %v", m.Error, err)
	}
	code, _ := number(v.(map[string]any)["code"])
	if code != codeRequestFailed {
		t.Fatalf("hover over a failed lowering answered code %v, want %d", code, codeRequestFailed)
	}
	if string(m.ID) != "2" {
		t.Fatalf("error reply id = %s, want the request id 2", m.ID)
	}
}

func TestPendingCapRefusesOverflow(t *testing.T) {
	p := newProxy(io.Discard, Options{Stderr: new(syncBuffer), Compiler: newFakeCompiler()})
	p.server = &writer{w: io.Discard}
	for i := range maxPending {
		p.addPending([]byte(strconv.Itoa(i)), pending{method: "textDocument/hover"})
	}
	raw, err := encode(map[string]any{"textDocument": map[string]any{"uri": "file:///w/p.go"}})
	if err != nil {
		t.Fatal(err)
	}
	p.request(t.Context(), &message{ID: []byte("overflow"), Method: "textDocument/hover", Params: raw})
	p.mu.Lock()
	n := len(p.pending)
	p.mu.Unlock()
	if n != maxPending {
		t.Fatalf("pending holds %d requests, want the cap %d", n, maxPending)
	}
	select {
	case m := <-p.toClient:
		v, err := decode(m.Error)
		if err != nil {
			t.Fatalf("error payload = %s, want a JSON-RPC error: %v", m.Error, err)
		}
		code, _ := number(v.(map[string]any)["code"])
		if code != codeRequestFailed || string(m.ID) != "overflow" {
			t.Fatalf("refused request answered id %s code %v, want overflow and %d", m.ID, code, codeRequestFailed)
		}
	default:
		t.Fatal("refused request got no error reply")
	}
}

func TestLateResultOfCanceledRequestIsNotForwarded(t *testing.T) {
	p := newProxy(io.Discard, Options{Stderr: new(syncBuffer), Compiler: newFakeCompiler()})
	p.server = &writer{w: io.Discard}
	p.addPending([]byte("7"), pending{method: "textDocument/definition"})
	p.clientNotification(t.Context(), &message{Method: "$/cancelRequest", Params: []byte(`{"id":7}`)})
	for len(p.toServer) > 0 {
		<-p.toServer
	}
	const lowered = `[{"uri":"file:///w/.logos/a.lgo.go","range":{"start":{"line":9,"character":1},"end":{"line":9,"character":7}}}]`
	p.serverResponse(t.Context(), &message{ID: []byte("7"), Result: []byte(lowered)})
	select {
	case m := <-p.toClient:
		if len(m.Result) != 0 {
			t.Fatalf("client got the untranslated result %s, want a canceled error", m.Result)
		}
		v, err := decode(m.Error)
		if err != nil {
			t.Fatalf("error payload = %s, want a JSON-RPC error: %v", m.Error, err)
		}
		code, _ := number(v.(map[string]any)["code"])
		if code != codeRequestCancelled || string(m.ID) != "7" {
			t.Fatalf("late result answered id %s code %v, want 7 and %d", m.ID, code, codeRequestCancelled)
		}
	default:
		t.Fatal("late result of a canceled request got no reply")
	}
}

func TestSlowFailingLoweringAnswersRequestFailed(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	dir := f.path("a")
	gate := f.comp.setLowerWait(dir)
	f.comp.setLowerError(dir, errLowerBoom)
	ch := make(chan *message, 1)
	go func() {
		ch <- f.client.callMessage("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	}()
	entered := time.After(waitLimit)
	for {
		inside := false
		for _, c := range f.comp.lowerCalls() {
			if c == dir {
				inside = true
			}
		}
		if inside {
			break
		}
		select {
		case <-entered:
			t.Fatal("the hover never reached the lowering")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	close(gate)
	select {
	case m := <-ch:
		v, err := decode(m.Error)
		if err != nil {
			t.Fatalf("error payload = %s, want a JSON-RPC error: %v", m.Error, err)
		}
		code, _ := number(v.(map[string]any)["code"])
		if code != codeRequestFailed || string(m.ID) != "2" {
			t.Fatalf("failed lowering answered id %s code %v, want 2 and %d", m.ID, code, codeRequestFailed)
		}
	case <-time.After(waitLimit):
		t.Fatal("a killed lowering never answered the request")
	}
}
