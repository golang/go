// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The tests of the proxy run it against a fake gopls: this test binary,
// started again with fakeEnv set, answers LSP requests from a script of
// canned results and records every message it receives. The compiler is a
// fakeCompiler (lowering_test.go), so the tests pin the proxy's mapping and
// ordering without a toolchain; toolchain_test.go covers the real one.

const (
	// fakeEnv names the script file of the fake gopls; its presence in the
	// environment turns the test binary into the fake.
	fakeEnv = "LOGOS_LSP_FAKE_GOPLS"

	// fakeLogEnv names the file the fake gopls appends every message to.
	fakeLogEnv = "LOGOS_LSP_FAKE_LOG"

	// waitLimit bounds every wait of a test on the proxy.
	waitLimit = 30 * time.Second
)

// fakeScript is what the fake gopls answers.
type fakeScript struct {
	// Responses maps a request method to its result.
	Responses map[string]json.RawMessage `json:"responses"`

	// Diagnostics maps a document URI to the diagnostics array the fake
	// publishes on every didOpen and didChange of that document.
	Diagnostics map[string]json.RawMessage `json:"diagnostics"`

	// DelayMillis delays each response to a request.
	DelayMillis int `json:"delayMillis"`

	// CrashOn names a request method that makes the fake exit instead of
	// answering.
	CrashOn string `json:"crashOn"`
}

// TestMain turns the test binary into the fake gopls when asked to.
func TestMain(m *testing.M) {
	if os.Getenv(fakeEnv) != "" {
		os.Exit(runFakeGopls())
	}
	os.Exit(m.Run())
}

// runFakeGopls is the fake gopls: it answers requests from its script and
// logs what it receives. It returns the exit status.
func runFakeGopls() int {
	var script fakeScript
	data, err := os.ReadFile(os.Getenv(fakeEnv))
	if err == nil {
		err = json.Unmarshal(data, &script)
	}
	logFile, err2 := os.OpenFile(os.Getenv(fakeLogEnv), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o666)
	if err != nil || err2 != nil {
		fmt.Fprintln(os.Stderr, "fake gopls:", err, err2)
		return 2
	}
	defer logFile.Close()
	rd, wr := newReader(os.Stdin), &writer{w: os.Stdout}
	for {
		m, err := rd.read()
		if err != nil || m.Method == "exit" {
			return 0
		}
		line, err := json.Marshal(m)
		if err == nil {
			_, err = logFile.Write(append(line, '\n'))
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "fake gopls: logging a message:", err)
			return 2
		}
		if m.Method == script.CrashOn && m.isRequest() {
			return 3
		}
		if err := fakeHandle(wr, &script, m); err != nil {
			fmt.Fprintln(os.Stderr, "fake gopls: answering a message:", err)
			return 2
		}
	}
}

// fakeHandle answers one message of the proxy. It returns the error of
// writing the answer.
func fakeHandle(wr *writer, script *fakeScript, m *message) error {
	switch {
	case m.isRequest():
		time.Sleep(time.Duration(script.DelayMillis) * time.Millisecond)
		return wr.respond(m.ID, fakeResult(script, m.Method))
	case m.Method == "textDocument/didOpen" || m.Method == "textDocument/didChange":
		var p struct {
			TextDocument struct {
				URI     string `json:"uri"`
				Version int    `json:"version"`
			} `json:"textDocument"`
		}
		if json.Unmarshal(m.Params, &p) != nil {
			return nil
		}
		if diags, ok := script.Diagnostics[p.TextDocument.URI]; ok {
			return wr.notify("textDocument/publishDiagnostics", map[string]any{
				"uri": p.TextDocument.URI, "version": p.TextDocument.Version, "diagnostics": diags,
			})
		}
	}
	return nil
}

// fakeResult returns the scripted result of method.
func fakeResult(script *fakeScript, method string) json.RawMessage {
	switch method {
	case "initialize":
		return json.RawMessage(`{"capabilities":{"hoverProvider":true,"diagnosticProvider":{"interFileDependencies":true,"workspaceDiagnostics":false}}}`)
	case "shutdown":
		return nullResult
	}
	if r, ok := script.Responses[method]; ok {
		return r
	}
	return nullResult
}

// syncBuffer is a bytes.Buffer safe for concurrent writes.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

// Write implements io.Writer.
func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

// String returns what was written.
func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// testClient is an LSP client of a proxy running in the test.
type testClient struct {
	t      *testing.T
	w      *writer
	closeW func()

	// done closes when Serve has returned, and err is then its result.
	done chan struct{}
	err  error

	mu      sync.Mutex
	cond    *sync.Cond
	msgs    []*message
	nextID  int
	expired bool
}

// startProxy serves a new client with opts and returns it. The test
// cleanup ends the session.
func startProxy(t *testing.T, opts Options) *testClient {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	c := &testClient{t: t, w: &writer{w: inW}, closeW: func() { inW.Close() }, done: make(chan struct{})}
	c.cond = sync.NewCond(&c.mu)
	stderr := new(syncBuffer)
	opts.Stderr = stderr
	ctx := t.Context()
	go func() {
		c.err = Serve(ctx, inR, outW, opts)
		outW.Close()
		close(c.done)
	}()
	go c.readLoop(newReader(outR))
	t.Cleanup(func() {
		inW.Close()
		outR.Close()
		select {
		case <-c.done:
		case <-time.After(waitLimit):
			t.Error("Serve did not return after the test")
		}
		if t.Failed() {
			t.Logf("proxy stderr:\n%s", stderr.String())
		}
	})
	return c
}

// wait returns what Serve returned, waiting for it to return.
func (c *testClient) wait() error {
	c.t.Helper()
	select {
	case <-c.done:
		return c.err
	case <-time.After(waitLimit):
		c.t.Fatal("Serve did not return")
		return nil
	}
}

// readLoop collects the proxy's output.
func (c *testClient) readLoop(rd *reader) {
	for {
		m, err := rd.read()
		c.mu.Lock()
		if err != nil {
			c.expired = true
		} else {
			c.msgs = append(c.msgs, m)
		}
		c.cond.Broadcast()
		c.mu.Unlock()
		if err != nil {
			return
		}
	}
}

// waitFor returns the first message of the proxy's output that match
// accepts, waiting for it up to waitLimit.
func (c *testClient) waitFor(what string, match func(*message) bool) *message {
	c.t.Helper()
	timer := time.AfterFunc(waitLimit, func() {
		c.mu.Lock()
		c.expired = true
		c.cond.Broadcast()
		c.mu.Unlock()
	})
	defer timer.Stop()
	c.mu.Lock()
	defer c.mu.Unlock()
	for {
		for _, m := range c.msgs {
			if match(m) {
				return m
			}
		}
		if c.expired {
			c.t.Fatalf("no %s from the proxy", what)
		}
		c.cond.Wait()
	}
}

// call sends a request and returns its result, failing on an error
// response.
func (c *testClient) call(method string, params any) json.RawMessage {
	c.t.Helper()
	m := c.callMessage(method, params)
	if len(m.Error) > 0 {
		c.t.Fatalf("%s failed: %s", method, m.Error)
	}
	return m.Result
}

// callMessage sends a request and returns its whole response.
func (c *testClient) callMessage(method string, params any) *message {
	c.t.Helper()
	return c.await(method, c.request(method, params))
}

// request sends a request and returns its id, for a test that does other
// work before it waits for the response.
func (c *testClient) request(method string, params any) json.RawMessage {
	c.t.Helper()
	c.mu.Lock()
	c.nextID++
	id := json.RawMessage(fmt.Sprint(c.nextID))
	c.mu.Unlock()
	raw, err := json.Marshal(params)
	if err != nil {
		c.t.Fatal(err)
	}
	if err := c.w.write(&message{ID: id, Method: method, Params: raw}); err != nil {
		c.t.Fatalf("sending %s: %v", method, err)
	}
	return id
}

// await returns the response to the request id.
func (c *testClient) await(method string, id json.RawMessage) *message {
	c.t.Helper()
	return c.waitFor("response to "+method, func(m *message) bool {
		return m.isResponse() && bytes.Equal(m.ID, id)
	})
}

// notify sends a notification.
func (c *testClient) notify(method string, params any) {
	c.t.Helper()
	if err := c.w.notify(method, params); err != nil {
		c.t.Fatalf("sending %s: %v", method, err)
	}
}

// notification waits for a notification of method whose decoded params
// satisfy accept, and returns them.
func (c *testClient) notification(method string, accept func(map[string]any) bool) map[string]any {
	c.t.Helper()
	var found map[string]any
	c.waitFor(method, func(m *message) bool {
		if m.Method != method {
			return false
		}
		v, err := decode(m.Params)
		root, _ := v.(map[string]any)
		if err != nil || root == nil || !accept(root) {
			return false
		}
		found = root
		return true
	})
	return found
}

// shutdown ends the session cleanly and returns what Serve returned.
func (c *testClient) shutdown() error {
	c.t.Helper()
	c.call("shutdown", nil)
	c.notify("exit", nil)
	return c.wait()
}

// fixture is a workspace on disk served by a proxy with a fake gopls and a
// fake compiler.
type fixture struct {
	t      *testing.T
	root   string
	comp   *fakeCompiler
	log    string
	client *testClient
}

// fixtureOptions tunes newFixture.
type fixtureOptions struct {
	// Script builds the script of the fake gopls for the workspace root.
	Script func(root string) fakeScript

	// Tune changes the proxy options before the proxy starts.
	Tune func(f *fixture, o *Options)
}

// newFixture writes files (a map from relative path to content) under a
// fresh workspace, starts a proxy over it, and runs the initialize
// handshake.
func newFixture(t *testing.T, files map[string]string, fo fixtureOptions) *fixture {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	var script fakeScript
	if fo.Script != nil {
		script = fo.Script(root)
	}
	data, err := json.Marshal(script)
	if err != nil {
		t.Fatal(err)
	}
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	if err := os.WriteFile(scriptPath, data, 0o666); err != nil {
		t.Fatal(err)
	}
	f := &fixture{t: t, root: root, comp: newFakeCompiler(), log: filepath.Join(scratch, "gopls.log")}
	opts := Options{
		Gopls:    []string{os.Args[0], "-mode=stdio"},
		Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+f.log),
		Compiler: f.comp,
		Debounce: -1,
	}
	if fo.Tune != nil {
		fo.Tune(f, &opts)
	}
	f.client = startProxy(t, opts)
	f.initialize()
	return f
}

// initialize runs the handshake. The client advertises position encodings,
// which the proxy must not pass on.
func (f *fixture) initialize() {
	f.t.Helper()
	f.client.call("initialize", map[string]any{
		"processId": nil,
		"rootUri":   pathURI(f.root),
		"capabilities": map[string]any{
			"general": map[string]any{"positionEncodings": []string{"utf-8", "utf-16"}},
		},
	})
	f.client.notify("initialized", map[string]any{})
}

// path returns the absolute path of a workspace file.
func (f *fixture) path(rel string) string { return filepath.Join(f.root, filepath.FromSlash(rel)) }

// uri returns the URI of a workspace file.
func (f *fixture) uri(rel string) string { return pathURI(f.path(rel)) }

// virtURI returns the URI of the lowered document of a loGo/s source.
func (f *fixture) virtURI(rel string) string { return pathURI(virtualPath(f.path(rel))) }

// open opens a workspace file in the client with its on-disk text.
func (f *fixture) open(rel string, version int) {
	f.t.Helper()
	data, err := os.ReadFile(f.path(rel))
	if err != nil {
		f.t.Fatal(err)
	}
	f.openText(rel, version, string(data))
}

// openText opens a workspace file in the client with the given text.
func (f *fixture) openText(rel string, version int, text string) {
	f.t.Helper()
	f.client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": f.uri(rel), "languageId": "logos", "version": version, "text": text},
	})
}

// sync waits until gopls has handled everything the proxy sent it so far:
// a request on no document goes to gopls after the earlier notifications,
// and its answer comes back after gopls logged them.
func (f *fixture) sync() {
	f.t.Helper()
	f.client.call("test/sync", map[string]any{})
}

// received returns the messages the fake gopls got, in order.
func (f *fixture) received() []*message {
	f.t.Helper()
	data, err := os.ReadFile(f.log)
	if err != nil {
		f.t.Fatal(err)
	}
	var msgs []*message
	for line := range bytes.SplitSeq(data, []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		m := new(message)
		if err := json.Unmarshal(line, m); err != nil {
			f.t.Fatalf("bad log line %q: %v", line, err)
		}
		msgs = append(msgs, m)
	}
	return msgs
}

// receivedMethod returns the params of the messages of one method the fake
// gopls got, in order.
func (f *fixture) receivedMethod(method string) []map[string]any {
	f.t.Helper()
	var out []map[string]any
	for _, m := range f.received() {
		if m.Method != method {
			continue
		}
		v, err := decode(m.Params)
		root, _ := v.(map[string]any)
		if err != nil || root == nil {
			f.t.Fatalf("params of %s: %v", method, err)
		}
		out = append(out, root)
	}
	return out
}

// workspaceFiles lists every file under the workspace root, relative and
// sorted.
func (f *fixture) workspaceFiles() []string {
	f.t.Helper()
	var files []string
	err := filepath.WalkDir(f.root, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			rel, _ := filepath.Rel(f.root, path)
			files = append(files, filepath.ToSlash(rel))
		}
		return err
	})
	if err != nil {
		f.t.Fatal(err)
	}
	slices.Sort(files)
	return files
}

// toJSON encodes v as raw JSON.
func toJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// sameJSON fails the test unless got and want are the same JSON value.
func sameJSON(t *testing.T, what string, got json.RawMessage, want any) {
	t.Helper()
	g, err := decode(got)
	if err != nil {
		t.Fatalf("%s: %v (%s)", what, err, got)
	}
	w, err := decode(toJSON(t, want))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Errorf("%s:\n got %s\nwant %s", what, got, toJSON(t, want))
	}
}

// positionParams builds the params of a position request on a document.
func positionParams(uri string, p Position) map[string]any {
	return map[string]any{"textDocument": map[string]any{"uri": uri}, "position": p}
}

// forwardedPosition returns the position of the nth request of method the
// fake gopls got, and the URI of its document.
func forwardedPosition(t *testing.T, f *fixture, method string, nth int) (string, Position) {
	t.Helper()
	reqs := f.receivedMethod(method)
	if len(reqs) <= nth {
		t.Fatalf("gopls got %d %s requests, want more than %d", len(reqs), method, nth)
	}
	td, _ := reqs[nth]["textDocument"].(map[string]any)
	uri, _ := td["uri"].(string)
	pos, ok := asPosition(reqs[nth]["position"].(map[string]any))
	if !ok {
		t.Fatalf("%s forwarded without a position: %v", method, reqs[nth])
	}
	return uri, pos
}

// TestProxyMapsRequestsAndResponses is the main attack test of M21-S1:
// hover, definition, references, completion, codeAction, inlayHint, and
// documentHighlight requests on a .lgo document reach gopls as requests on
// the lowered document with their positions mapped, including positions
// inside the three lines a ? statement expands to, and the answers come
// back with URIs and ranges mapped to the .lgo source.
func TestProxyMapsRequestsAndResponses(t *testing.T) {
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowText := string(low)

	origOperand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	lowOperand := shift(findPos(t, lowText, "ReadFile(path)", 0), len("ReadFile("))
	origQ := findPos(t, aLgo, "?", 0)
	lowAfterCall := shift(findPos(t, lowText, "(path)", 0), len("(path)"))
	origCall := span(findPos(t, aLgo, "os.ReadFile", 0), len("os.ReadFile"))
	lowCall := span(findPos(t, lowText, "os.ReadFile", 0), len("os.ReadFile"))
	origRead := span(findPos(t, aLgo, "Read(", 0), len("Read"))
	lowRead := span(findPos(t, lowText, "Read(", 0), len("Read"))
	origData := findPos(t, aLgo, "data", 0)
	lowDataStart := findPos(t, lowText, "data", 0)
	lowData := shift(lowDataStart, len("data"))
	genRange := Range{Start: Position{Line: 6, Character: 0}, End: Position{Line: 7, Character: len(lineOf(lowText, 7))}}
	goFile := Range{Start: Position{Line: 3, Character: 5}, End: Position{Line: 3, Character: 9}}

	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(root string) fakeScript {
			virt := pathURI(filepath.Join(root, "a", "a.go"))
			other := pathURI(filepath.Join(root, "b", "b.go"))
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{
					"contents": map[string]any{"kind": "markdown", "value": "func ReadFile"},
					"range":    lowCall,
				}),
				"textDocument/definition": toJSON(t, []any{
					map[string]any{"uri": virt, "range": lowRead},
					map[string]any{"uri": other, "range": goFile},
				}),
				"textDocument/references": toJSON(t, []any{
					map[string]any{"uri": virt, "range": lowRead},
					map[string]any{"uri": virt, "range": span(lowDataStart, len("data"))},
					map[string]any{"uri": other, "range": goFile},
				}),
				"textDocument/completion": toJSON(t, map[string]any{
					"isIncomplete": false,
					"items": []any{map[string]any{"label": "ReadFile", "textEdit": map[string]any{
						"range": span(lowOperand, len("path")), "newText": "ReadFile",
					}}},
				}),
				"textDocument/codeAction": toJSON(t, []any{map[string]any{
					"title": "fix", "kind": "quickfix",
					"edit": map[string]any{
						"changes": map[string]any{virt: []any{map[string]any{"range": span(lowOperand, len("path")), "newText": "p"}}},
						"documentChanges": []any{map[string]any{
							"textDocument": map[string]any{"uri": virt, "version": 4},
							"edits":        []any{map[string]any{"range": lowCall, "newText": "io.ReadFile"}},
						}},
					},
				}}),
				"textDocument/inlayHint": toJSON(t, []any{
					map[string]any{"position": lowData, "label": ": []byte"},
				}),
				"textDocument/documentHighlight": toJSON(t, []any{
					map[string]any{"range": genRange, "kind": 1},
				}),
			}}
		},
	})
	f.open("a/a.lgo", 3)
	f.sync()
	aURI := f.uri("a/a.lgo")

	if opened := f.receivedMethod("textDocument/didOpen"); len(opened) != 1 {
		t.Fatalf("gopls got %d didOpen messages, want one for the lowered document: %v", len(opened), opened)
	} else {
		item, _ := opened[0]["textDocument"].(map[string]any)
		if item["uri"] != f.virtURI("a/a.lgo") || item["text"] != lowText || item["languageId"] != "go" {
			t.Errorf("lowered document opened as %v, want %s with the lowered text as go", item, f.virtURI("a/a.lgo"))
		}
	}

	t.Run("hover on an operand inside the ? statement", func(t *testing.T) {
		res := f.client.call("textDocument/hover", positionParams(aURI, origOperand))
		uri, pos := forwardedPosition(t, f, "textDocument/hover", 0)
		if uri != f.virtURI("a/a.lgo") || pos != lowOperand {
			t.Errorf("forwarded hover = %s %v, want %s %v", uri, pos, f.virtURI("a/a.lgo"), lowOperand)
		}
		sameJSON(t, "hover result", res, map[string]any{
			"contents": map[string]any{"kind": "markdown", "value": "func ReadFile"},
			"range":    origCall,
		})
	})
	t.Run("hover on the ?", func(t *testing.T) {
		f.client.call("textDocument/hover", positionParams(aURI, origQ))
		_, pos := forwardedPosition(t, f, "textDocument/hover", 1)
		if pos != lowAfterCall {
			t.Errorf("hover on the ? forwarded at %v, want the start of the expansion %v", pos, lowAfterCall)
		}
	})
	t.Run("definition crosses files", func(t *testing.T) {
		res := f.client.call("textDocument/definition", positionParams(aURI, origOperand))
		sameJSON(t, "definition result", res, []any{
			map[string]any{"uri": aURI, "range": origRead},
			map[string]any{"uri": f.uri("b/b.go"), "range": goFile},
		})
	})
	t.Run("references", func(t *testing.T) {
		res := f.client.call("textDocument/references", map[string]any{
			"textDocument": map[string]any{"uri": aURI}, "position": origOperand,
			"context": map[string]any{"includeDeclaration": true},
		})
		sameJSON(t, "references result", res, []any{
			map[string]any{"uri": aURI, "range": origRead},
			map[string]any{"uri": aURI, "range": span(origData, len("data"))},
			map[string]any{"uri": f.uri("b/b.go"), "range": goFile},
		})
		reqs := f.receivedMethod("textDocument/references")
		if ctx, _ := reqs[0]["context"].(map[string]any); ctx["includeDeclaration"] != true {
			t.Errorf("references context was not passed on: %v", reqs[0])
		}
	})
	t.Run("completion", func(t *testing.T) {
		res := f.client.call("textDocument/completion", positionParams(aURI, origOperand))
		sameJSON(t, "completion result", res, map[string]any{
			"isIncomplete": false,
			"items": []any{map[string]any{"label": "ReadFile", "textEdit": map[string]any{
				"range": span(origOperand, len("path")), "newText": "ReadFile",
			}}},
		})
	})
	t.Run("codeAction maps the range, the diagnostics, and the edit", func(t *testing.T) {
		res := f.client.call("textDocument/codeAction", map[string]any{
			"textDocument": map[string]any{"uri": aURI},
			"range":        span(origOperand, len("path")),
			"context": map[string]any{"diagnostics": []any{map[string]any{
				"range": span(origOperand, len("path")), "message": "bad",
			}}},
		})
		req := f.receivedMethod("textDocument/codeAction")[0]
		if r, _ := asRange(req["range"].(map[string]any)); r != span(lowOperand, len("path")) {
			t.Errorf("code action range forwarded as %v, want %v", r, span(lowOperand, len("path")))
		}
		ctx, _ := req["context"].(map[string]any)
		diags, _ := ctx["diagnostics"].([]any)
		if r, _ := asRange(diags[0].(map[string]any)["range"].(map[string]any)); r != span(lowOperand, len("path")) {
			t.Errorf("diagnostic range forwarded as %v, want %v", r, span(lowOperand, len("path")))
		}
		sameJSON(t, "codeAction result", res, []any{map[string]any{
			"title": "fix", "kind": "quickfix",
			"edit": map[string]any{
				"changes": map[string]any{aURI: []any{map[string]any{"range": span(origOperand, len("path")), "newText": "p"}}},
				"documentChanges": []any{map[string]any{
					"textDocument": map[string]any{"uri": aURI, "version": nil},
					"edits":        []any{map[string]any{"range": origCall, "newText": "io.ReadFile"}},
				}},
			},
		}})
	})
	t.Run("inlayHint", func(t *testing.T) {
		res := f.client.call("textDocument/inlayHint", map[string]any{
			"textDocument": map[string]any{"uri": aURI},
			"range":        Range{Start: Position{Line: 0}, End: Position{Line: 8}},
		})
		sameJSON(t, "inlayHint result", res, []any{
			map[string]any{"position": shift(origData, len("data")), "label": ": []byte"},
		})
	})
	t.Run("a range over the generated lines maps to the ? token", func(t *testing.T) {
		res := f.client.call("textDocument/documentHighlight", positionParams(aURI, origQ))
		sameJSON(t, "documentHighlight result", res, []any{
			map[string]any{"range": span(origQ, 1), "kind": 1},
		})
	})
	t.Run("nothing is written to the workspace", func(t *testing.T) {
		if got := f.workspaceFiles(); !slices.Equal(got, []string{"a/a.lgo"}) {
			t.Errorf("workspace files = %v, want only a/a.lgo", got)
		}
	})
	if err := f.client.shutdown(); err != nil {
		t.Errorf("Serve = %v, want nil", err)
	}
}

// TestProxyHandshakeShapesCapabilities checks the two rewrites of the
// handshake: position encodings are not offered to gopls, so it answers in
// UTF-16, and diagnosticProvider is dropped from the result, so clients
// take diagnostics through publishDiagnostics, where the merge happens.
func TestProxyHandshakeShapesCapabilities(t *testing.T) {
	f := newFixture(t, nil, fixtureOptions{})
	inits := f.receivedMethod("initialize")
	if len(inits) != 1 {
		t.Fatalf("gopls got %d initialize requests, want 1", len(inits))
	}
	caps, _ := inits[0]["capabilities"].(map[string]any)
	general, _ := caps["general"].(map[string]any)
	if _, ok := general["positionEncodings"]; ok {
		t.Errorf("positionEncodings reached gopls: %v", general)
	}
	res := f.client.call("initialize", map[string]any{"capabilities": map[string]any{}})
	var got struct {
		Capabilities map[string]json.RawMessage `json:"capabilities"`
	}
	if err := json.Unmarshal(res, &got); err != nil {
		t.Fatal(err)
	}
	if _, ok := got.Capabilities["diagnosticProvider"]; ok {
		t.Errorf("diagnosticProvider reached the client: %v", got.Capabilities)
	}
	if _, ok := got.Capabilities["hoverProvider"]; !ok {
		t.Errorf("other capabilities were dropped: %v", got.Capabilities)
	}
}

// TestProxyEditsUpdateOverlay checks that an edit of an open .lgo document
// reaches the overlay before the next request: the proxy recompiles with
// the unsaved text, sends the new lowered text to gopls ahead of the
// request, and maps the request through the new lowering. Incremental
// changes splice in UTF-16 positions.
func TestProxyEditsUpdateOverlay(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	aPath, aURI := f.path("a/a.lgo"), f.uri("a/a.lgo")

	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument": map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{
			"range": Range{Start: Position{Line: 2, Character: 0}, End: Position{Line: 2, Character: 0}},
			"text":  "// edited é\n",
		}},
	})
	edited := strings.Replace(aLgo, "import \"os\"\n", "// edited é\nimport \"os\"\n", 1)
	lowEdited, _, _ := lowerQuestion([]byte(edited))

	origOperand := shift(findPos(t, edited, "ReadFile(path)?", 0), len("ReadFile("))
	lowOperand := shift(findPos(t, string(lowEdited), "ReadFile(path)", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, origOperand))

	var order []string
	for _, m := range f.received() {
		switch m.Method {
		case "textDocument/didChange", "textDocument/hover":
			order = append(order, m.Method)
		}
	}
	if want := []string{"textDocument/didChange", "textDocument/hover"}; !slices.Equal(order, want) {
		t.Fatalf("gopls got %v, want the lowered change before the request: %v", order, want)
	}
	changes := f.receivedMethod("textDocument/didChange")
	edits, _ := changes[0]["contentChanges"].([]any)
	if text, _ := edits[0].(map[string]any)["text"].(string); text != string(lowEdited) {
		t.Errorf("gopls got the lowered text %q, want %q", text, lowEdited)
	}
	if td, _ := changes[0]["textDocument"].(map[string]any); td["uri"] != f.virtURI("a/a.lgo") {
		t.Errorf("change sent for %v, want the lowered document", td["uri"])
	}
	if _, pos := forwardedPosition(t, f, "textDocument/hover", 0); pos != lowOperand {
		t.Errorf("hover forwarded at %v, want %v through the new lowering", pos, lowOperand)
	}
	calls := f.comp.bufferCalls()
	if last := calls[len(calls)-1]; string(last[aPath]) != edited {
		t.Errorf("the compiler read %q for the document, want the unsaved text", last[aPath])
	}
	onDisk, _ := os.ReadFile(aPath)
	if string(onDisk) != aLgo {
		t.Errorf("the file on disk changed to %q", onDisk)
	}
}

// TestProxyStaleLoweringAnswersNull checks the stale-lowering rule: while
// the document does not lower at the client's text, position requests get
// null, never a position computed from an older text, and the proxy
// recovers when the text compiles again.
func TestProxyStaleLoweringAnswersNull(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(root string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{"contents": map[string]any{"kind": "plaintext", "value": "x"}}),
			}}
		},
	})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	p := Position{Line: 5, Character: 3}

	if res := f.client.call("textDocument/hover", positionParams(aURI, p)); string(res) == "null" {
		t.Fatal("hover on a lowered document answered null")
	}
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "// SYNTAX ERROR\n"}},
	})
	if res := f.client.call("textDocument/hover", positionParams(aURI, p)); string(res) != "null" {
		t.Errorf("hover on a stale lowering answered %s, want null", res)
	}
	if got := len(f.receivedMethod("textDocument/hover")); got != 1 {
		t.Errorf("gopls got %d hover requests, want the stale one withheld", got)
	}
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 3},
		"contentChanges": []any{map[string]any{"text": aLgo}},
	})
	if res := f.client.call("textDocument/hover", positionParams(aURI, p)); string(res) == "null" {
		t.Error("hover stayed null after the text compiled again")
	}
}

// TestProxyMergesDiagnostics checks publishDiagnostics: what gopls
// reports on the lowered document, mapped to the source, plus the compiler
// diagnostics of the package, with a duplicate reported once, under the
// version of the client's document.
func TestProxyMergesDiagnostics(t *testing.T) {
	var lowCall Range
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(root string) fakeScript {
			low, _, _ := lowerQuestion([]byte(aLgo))
			lowCall = span(findPos(t, string(low), "os.ReadFile", 0), len("os.ReadFile"))
			virt := pathURI(filepath.Join(root, "a", "a.go"))
			return fakeScript{Diagnostics: map[string]json.RawMessage{virt: toJSON(t, []any{
				map[string]any{"range": lowCall, "severity": 1, "message": "gopls: bad call"},
				map[string]any{
					"range":    Range{Start: Position{Line: 6, Character: 1}, End: Position{Line: 6, Character: 5}},
					"severity": 2, "message": "gopls: inside the expansion",
				},
			})}}
		},
	})
	aPath := f.path("a/a.lgo")
	callLine, callCol := 6, strings.Index(lineOf(aLgo, 5), "os.ReadFile")+1
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Rule: "correctness::example", Message: "compiler: unproven",
			Start: [2]int{callLine, callCol}, End: [2]int{callLine, callCol + len("os.ReadFile")}}),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Message: "gopls: bad call",
			Start: [2]int{callLine, callCol}, End: [2]int{callLine, callCol + len("os.ReadFile")}}),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "warn", Rule: "style::note", Message: "compiler: warning",
			Start: [2]int{3, 1}, End: [2]int{3, 7}}),
	)
	f.open("a/a.lgo", 7)

	params := f.client.notification("textDocument/publishDiagnostics", func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return m["uri"] == f.uri("a/a.lgo") && len(items) == 4
	})
	if v, _ := number(params["version"]); v != 7 {
		t.Errorf("diagnostics published for version %v, want the client's version 7", params["version"])
	}
	type diag struct {
		severity int
		start    Position
		end      Position
		code     string
	}
	got := map[string]diag{}
	for _, item := range params["diagnostics"].([]any) {
		m := item.(map[string]any)
		r, _ := asRange(m["range"].(map[string]any))
		sev, _ := number(m["severity"])
		code, _ := m["code"].(string)
		got[m["message"].(string)] = diag{sev, r.Start, r.End, code}
	}
	origCall := span(findPos(t, aLgo, "os.ReadFile", 0), len("os.ReadFile"))
	origQ := findPos(t, aLgo, "?", 0)
	want := map[string]diag{
		"gopls: bad call":             {1, origCall.Start, origCall.End, ""},
		"gopls: inside the expansion": {2, origQ, shift(origQ, 1), ""},
		"compiler: unproven":          {1, origCall.Start, origCall.End, "correctness::example"},
		"compiler: warning":           {2, Position{Line: 2, Character: 0}, Position{Line: 2, Character: 6}, "style::note"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("diagnostics:\n got %v\nwant %v", got, want)
	}
}

// TestProxyDiagnosticsFollowEdits checks that compiler diagnostics are
// never shown against a text they were not computed for: after an edit
// they disappear until the recompile replaces them.
func TestProxyDiagnosticsFollowEdits(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aPath := f.path("a/a.lgo")
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Message: "first", Start: [2]int{1, 1}, End: [2]int{1, 8}}))
	f.open("a/a.lgo", 1)
	f.client.notification("textDocument/publishDiagnostics", func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return len(items) == 1
	})
	f.comp.setDiagnostics(filepath.Dir(aPath))
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "\n"}},
	})
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))
	f.client.notification("textDocument/publishDiagnostics", func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		v, _ := number(m["version"])
		return len(items) == 0 && v == 2
	})
}

// TestProxyMixedWorkspaceDefinition checks the cross-kind rule of R-U2 for
// a Go package that imports a loGo/s package: definition on the loGo/s
// symbol imported by the .go file answers with its position in the .lgo
// file. The lowered document of the imported package is open in gopls,
// the Go document is forwarded unchanged, and the workspace gains no file.
func TestProxyMixedWorkspaceDefinition(t *testing.T) {
	const libLgo = "package lib\n\nfunc Hello() string {\n\treturn \"hi\"\n}\n"
	const appGo = "package app\n\nimport \"example.com/m/lib\"\n\nvar Greeting = lib.Hello()\n"
	files := map[string]string{
		"go.mod":      "module example.com/m\n\ngo 1.27\n",
		"lib/lib.lgo": libLgo,
		"app/app.go":  appGo,
	}
	libLow, _, _ := lowerQuestion([]byte(libLgo))
	helloLow := span(findPos(t, string(libLow), "Hello", 0), len("Hello"))
	helloOrig := span(findPos(t, libLgo, "Hello", 0), len("Hello"))

	f := newFixture(t, files, fixtureOptions{
		Script: func(root string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/definition": toJSON(t, []any{
					map[string]any{"uri": pathURI(filepath.Join(root, "lib", "lib.go")), "range": helloLow},
				}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			f.comp.setClosure(f.path("app"), f.path("lib"))
		},
	})
	before := f.workspaceFiles()

	f.open("app/app.go", 1)
	f.sync()
	opened := f.receivedMethod("textDocument/didOpen")
	var uris []string
	for _, o := range opened {
		td, _ := o["textDocument"].(map[string]any)
		uris = append(uris, td["uri"].(string))
	}
	slices.Sort(uris)
	if want := []string{f.uri("app/app.go"), f.virtURI("lib/lib.lgo")}; !slices.Equal(uris, want) {
		t.Fatalf("gopls opened %v, want the Go document and the lowered loGo/s dependency %v", uris, want)
	}

	appCall := shift(findPos(t, appGo, "lib.Hello", 0), len("lib."))
	res := f.client.call("textDocument/definition", positionParams(f.uri("app/app.go"), appCall))
	if uri, pos := forwardedPosition(t, f, "textDocument/definition", 0); uri != f.uri("app/app.go") || pos != appCall {
		t.Errorf("a Go document request was forwarded as %s %v, want it unchanged", uri, pos)
	}
	sameJSON(t, "definition of a loGo/s symbol from a Go file", res, []any{
		map[string]any{"uri": f.uri("lib/lib.lgo"), "range": helloOrig},
	})

	if after := f.workspaceFiles(); !slices.Equal(before, after) {
		t.Errorf("workspace files changed from %v to %v", before, after)
	}
}

// TestProxyGoDefinitionFromLogosFile checks the other cross-kind
// direction: a definition that gopls resolves in a real .go file reaches
// the client with the position of that file, untouched by any mapping.
func TestProxyGoDefinitionFromLogosFile(t *testing.T) {
	const app2Lgo = "package app2\n\nimport \"example.com/m/golib\"\n\nvar N = golib.Thing()\n"
	const goLib = "package golib\n\nfunc Thing() int { return 1 }\n"
	thing := Range{Start: Position{Line: 2, Character: 5}, End: Position{Line: 2, Character: 10}}
	f := newFixture(t, map[string]string{
		"go.mod":        "module example.com/m\n\ngo 1.27\n",
		"golib/g.go":    goLib,
		"app2/app2.lgo": app2Lgo,
	}, fixtureOptions{
		Script: func(root string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/definition": toJSON(t, []any{
					map[string]any{"uri": pathURI(filepath.Join(root, "golib", "g.go")), "range": thing},
				}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			f.comp.setClosure(f.path("app2"), f.path("golib"))
		},
	})
	before := f.workspaceFiles()
	f.open("app2/app2.lgo", 1)
	call := shift(findPos(t, app2Lgo, "golib.Thing", 0), len("golib."))
	res := f.client.call("textDocument/definition", positionParams(f.uri("app2/app2.lgo"), call))
	sameJSON(t, "definition of a Go symbol from a loGo/s file", res, []any{
		map[string]any{"uri": f.uri("golib/g.go"), "range": thing},
	})
	if uri, _ := forwardedPosition(t, f, "textDocument/definition", 0); uri != f.virtURI("app2/app2.lgo") {
		t.Errorf("the loGo/s request went to %s, want the lowered document", uri)
	}
	if after := f.workspaceFiles(); !slices.Equal(before, after) {
		t.Errorf("workspace files changed from %v to %v", before, after)
	}
}

// TestProxyClosesLoweredDocuments checks that closing the last document of
// a package takes its lowered documents out of gopls, and clears the
// diagnostics published for it.
func TestProxyClosesLoweredDocuments(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aPath := f.path("a/a.lgo")
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Message: "m", Start: [2]int{1, 1}, End: [2]int{1, 2}}))
	f.open("a/a.lgo", 1)
	f.client.notification("textDocument/publishDiagnostics", func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return len(items) == 1
	})
	f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": f.uri("a/a.lgo")}})
	f.client.call("textDocument/hover", positionParams(f.uri("b/b.go"), Position{}))
	closed := f.receivedMethod("textDocument/didClose")
	if len(closed) != 1 {
		t.Fatalf("gopls got %d didClose messages, want 1", len(closed))
	}
	if td, _ := closed[0]["textDocument"].(map[string]any); td["uri"] != f.virtURI("a/a.lgo") {
		t.Errorf("gopls closed %v, want the lowered document", td["uri"])
	}
	f.client.notification("textDocument/publishDiagnostics", func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return m["uri"] == f.uri("a/a.lgo") && len(items) == 0
	})
}

// TestProxyAnswersRequestsInFlightWhenClientCloses checks the drain: a
// client that sends a request and closes its output still gets the answer
// before the proxy stops, and Serve returns nil.
func TestProxyAnswersRequestsInFlightWhenClientCloses(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{
				DelayMillis: 300,
				Responses:   map[string]json.RawMessage{"textDocument/hover": toJSON(t, map[string]any{"contents": "late"})},
			}
		},
	})
	f.open("a/a.lgo", 1)
	raw := toJSON(t, positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	if err := f.client.w.write(&message{ID: json.RawMessage("99"), Method: "textDocument/hover", Params: raw}); err != nil {
		t.Fatal(err)
	}
	f.client.closeW()
	m := f.client.waitFor("the late hover response", func(m *message) bool { return m.isResponse() && string(m.ID) == "99" })
	if !strings.Contains(string(m.Result), "late") {
		t.Errorf("hover result = %s, want the scripted answer", m.Result)
	}
	if err := f.client.wait(); err != nil {
		t.Errorf("Serve = %v, want nil after the client closed its output", err)
	}
	if !f.comp.isClosed() {
		t.Error("the compiler was not released")
	}
}

// TestServeExitWithoutShutdown checks the exit code rule of the protocol:
// exit before shutdown is an error.
func TestServeExitWithoutShutdown(t *testing.T) {
	f := newFixture(t, nil, fixtureOptions{})
	f.client.notify("exit", nil)
	if err := f.client.wait(); err == nil || !strings.Contains(err.Error(), "exit without shutdown") {
		t.Errorf("Serve = %v, want the exit without shutdown error", err)
	}
}

// TestServeReportsGoplsDeath checks that a dying gopls ends the session
// with an error instead of hanging the client.
func TestServeReportsGoplsDeath(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript { return fakeScript{CrashOn: "textDocument/hover"} },
	})
	f.open("a/a.lgo", 1)
	raw := toJSON(t, positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	if err := f.client.w.write(&message{ID: json.RawMessage("5"), Method: "textDocument/hover", Params: raw}); err != nil {
		t.Fatal(err)
	}
	if err := f.client.wait(); err == nil || !strings.Contains(err.Error(), "gopls exited") {
		t.Errorf("Serve = %v, want an error naming the gopls exit", err)
	}
}

// TestServeRejectsMissingParts checks the guard of Serve itself.
func TestServeRejectsMissingParts(t *testing.T) {
	err := Serve(context.Background(), strings.NewReader(""), io.Discard, Options{})
	if err == nil {
		t.Error("Serve accepted options with no gopls and no compiler")
	}
	err = Serve(context.Background(), strings.NewReader(""), io.Discard, Options{
		Gopls: []string{filepath.Join(t.TempDir(), "no-such-gopls")}, Compiler: newFakeCompiler(),
	})
	if err == nil || !strings.Contains(err.Error(), "starting gopls") {
		t.Errorf("Serve with an unstartable gopls = %v, want a starting gopls error", err)
	}
}

// TestGoplsMissing checks the message of SPEC Sec 21.1 when gopls is not
// on PATH, and that a gopls on PATH is the one that runs.
func TestGoplsMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	_, err := goplsCommand()
	if err == nil || err.Error() != "logos lsp: gopls not found on PATH" {
		t.Errorf("goplsCommand() error = %v, want %q", err, "logos lsp: gopls not found on PATH")
	}
	dir := t.TempDir()
	name := "gopls"
	if err := os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\n"), 0o777); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	cmd, err := goplsCommand()
	if err != nil {
		t.Skipf("a script named gopls is not executable here: %v", err)
	}
	if cmd[0] != filepath.Join(dir, name) || !slices.Contains(cmd, "-mode=stdio") {
		t.Errorf("goplsCommand() = %v, want the gopls of PATH with -mode=stdio", cmd)
	}
}

// TestGoplsEnvPutsToolchainFirst checks that the go command gopls runs is
// the one of the toolchain that lowers the sources.
func TestGoplsEnvPutsToolchainFirst(t *testing.T) {
	sep := string(os.PathListSeparator)
	got := goplsEnv([]string{"HOME=/h", "PATH=/usr/bin" + sep + "/bin", "X=1"}, "/goroot/bin")
	want := []string{"HOME=/h", "X=1", "PATH=/goroot/bin" + sep + "/usr/bin" + sep + "/bin"}
	if !slices.Equal(got, want) {
		t.Errorf("goplsEnv = %v, want %v", got, want)
	}
	if got := goplsEnv(nil, "/goroot/bin"); !slices.Equal(got, []string{"PATH=/goroot/bin"}) {
		t.Errorf("goplsEnv with no PATH = %v, want a PATH of the toolchain alone", got)
	}
}

// TestFraming checks the JSON-RPC framing: a round trip, header case and
// extra headers, a missing length, a corrupt length, and a truncated body.
func TestFraming(t *testing.T) {
	var buf bytes.Buffer
	w := &writer{w: &buf}
	if err := w.notify("a/b", map[string]int{"n": 1}); err != nil {
		t.Fatal(err)
	}
	if err := w.fail(json.RawMessage("7"), -32601, "nope"); err != nil {
		t.Fatal(err)
	}
	rd := newReader(&buf)
	m, err := rd.read()
	if err != nil || m.Method != "a/b" || m.isRequest() || m.isResponse() {
		t.Fatalf("first message = %+v, %v", m, err)
	}
	m, err = rd.read()
	if err != nil || !m.isResponse() || string(m.ID) != "7" || !strings.Contains(string(m.Error), "nope") {
		t.Fatalf("second message = %+v, %v", m, err)
	}
	if _, err := rd.read(); !errors.Is(err, io.EOF) {
		t.Errorf("read at the end = %v, want io.EOF", err)
	}

	body := `{"jsonrpc":"2.0","method":"x"}`
	ok := fmt.Sprintf("content-type: application/vscode-jsonrpc\r\nCONTENT-LENGTH: %d\r\n\r\n%s", len(body), body)
	if m, err := newReader(strings.NewReader(ok)).read(); err != nil || m.Method != "x" {
		t.Errorf("framed with odd headers = %+v, %v", m, err)
	}
	for name, in := range map[string]string{
		"missing length": "X-Other: 1\r\n\r\n{}",
		"corrupt length": "Content-Length: abc\r\n\r\n{}",
		"negative":       "Content-Length: -4\r\n\r\n{}",
		"truncated":      "Content-Length: 99\r\n\r\n{}",
		"not json":       "Content-Length: 3\r\n\r\nabc",
		"cut header":     "Content-Length: 3",
	} {
		if _, err := newReader(strings.NewReader(in)).read(); err == nil || errors.Is(err, io.EOF) {
			t.Errorf("%s: read = %v, want a framing error", name, err)
		}
	}
}

// TestRewriterLeavesForeignJSONAlone checks the rewriter on values it must
// not touch: ranges outside any file, other files' ranges, and objects
// that only look like positions.
func TestRewriterLeavesForeignJSONAlone(t *testing.T) {
	_, _, pm := questionMap(t)
	lw := &lowered{src: "/w/a.lgo", virt: "/w/a.go", pm: pm, fresh: true}
	tb := &tables{bySrc: map[string]*lowered{lw.src: lw}, byVirt: map[string]*lowered{lw.virt: lw}}
	in := toJSON(t, map[string]any{
		"other":  map[string]any{"uri": "file:///w/b.go", "range": span(Position{Line: 5, Character: 5}, 3)},
		"loose":  map[string]any{"line": 5, "character": 1, "extra": true},
		"scalar": "file:///w/a.go",
	})
	v, err := decode(in)
	if err != nil {
		t.Fatal(err)
	}
	out, err := encode(tb.toSource().rewrite(v, nil))
	if err != nil {
		t.Fatal(err)
	}
	sameJSON(t, "rewritten", out, map[string]any{
		"other":  map[string]any{"uri": "file:///w/b.go", "range": span(Position{Line: 5, Character: 5}, 3)},
		"loose":  map[string]any{"line": 5, "character": 1, "extra": true},
		"scalar": "file:///w/a.go",
	})
}

// TestProxyMergesDiagnosticsOfGoDocuments checks that the merge covers Go
// documents too: gopls's diagnostics pass through as they are, the
// compiler's are added, and a compiler diagnostic that gopls reported at
// the same place with the same text is shown once.
func TestProxyMergesDiagnosticsOfGoDocuments(t *testing.T) {
	const src = "package p\n\nvar x int = \"s\"\n"
	f := newFixture(t, map[string]string{"p/p.go": src}, fixtureOptions{
		Script: func(root string) fakeScript {
			return fakeScript{Diagnostics: map[string]json.RawMessage{
				pathURI(filepath.Join(root, "p", "p.go")): toJSON(t, []any{map[string]any{
					"range":    span(Position{Line: 2, Character: 12}, 3),
					"severity": 1, "message": "cannot use \"s\" as int value",
				}}),
			}}
		},
	})
	pPath := f.path("p/p.go")
	f.comp.setDiagnostics(filepath.Dir(pPath),
		compilerDiag(t, diagSpec{Path: pPath, Severity: "deny", Message: "cannot use \"s\" as int value",
			Start: [2]int{3, 13}, End: [2]int{3, 16}}),
		compilerDiag(t, diagSpec{Path: pPath, Severity: "warn", Rule: "style::x", Message: "compiler: style",
			Start: [2]int{3, 5}, End: [2]int{3, 6}}),
	)
	f.open("p/p.go", 4)
	params := f.client.notification("textDocument/publishDiagnostics", func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return m["uri"] == f.uri("p/p.go") && len(items) == 2
	})
	if v, _ := number(params["version"]); v != 4 {
		t.Errorf("diagnostics published for version %v, want 4", params["version"])
	}
	var msgs []string
	for _, item := range params["diagnostics"].([]any) {
		msgs = append(msgs, item.(map[string]any)["message"].(string))
	}
	slices.Sort(msgs)
	if want := []string{"cannot use \"s\" as int value", "compiler: style"}; !slices.Equal(msgs, want) {
		t.Errorf("messages = %q, want %q", msgs, want)
	}
}

// TestWriterRespondsWithNullForNoResult checks the bytes of a response with
// no result: JSON-RPC wants a result member, so it is null, never absent.
func TestWriterRespondsWithNullForNoResult(t *testing.T) {
	var buf bytes.Buffer
	if err := (&writer{w: &buf}).respond(json.RawMessage("3"), nil); err != nil {
		t.Fatal(err)
	}
	body := `{"jsonrpc":"2.0","id":3,"result":null}`
	if want := fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(body), body); buf.String() != want {
		t.Errorf("response = %q, want %q", buf.String(), want)
	}
}

// TestContentLengthLimit tests the cap on a message body at its edges: one
// below, at, and one above the limit, and the lengths that are no length.
func TestContentLengthLimit(t *testing.T) {
	for _, c := range []struct {
		in string
		ok bool
	}{
		{strconv.Itoa(maxMessage - 1), true},
		{strconv.Itoa(maxMessage), true},
		{strconv.Itoa(maxMessage + 1), false},
		{"0", true},
		{"-1", false},
		{"", false},
		{"12x", false},
	} {
		n, err := parseLength(c.in)
		if (err == nil) != c.ok {
			t.Errorf("parseLength(%q) = %d, %v, want ok = %v", c.in, n, err, c.ok)
		}
	}
	header := "Content-Length: " + strconv.Itoa(maxMessage+1) + "\r\n\r\n"
	if _, err := newReader(strings.NewReader(header)).read(); err == nil || !strings.Contains(err.Error(), "invalid Content-Length") {
		t.Errorf("a body over the cap = %v, want the invalid Content-Length error before any allocation", err)
	}
}

// TestProxyCodeActionWithoutLoweringUsesCompiler is the regression for the
// live M21 path: a source the compiler denies (slop::discarded_error on a
// bare call) has no lowering, yet textDocument/codeAction still answers
// the compiler autofix "add ? to propagate the error" from the real
// toolchain, without asking gopls. After the edit that removes the denied
// call, the old fix must not cross the edit.
func TestProxyCodeActionWithoutLoweringUsesCompiler(t *testing.T) {
	root, tc := realWorkspace(t)
	aPath := filepath.Join(root, filepath.FromSlash("lmod/a/a.lgo"))
	const denied = "package a\n\nfunc get() (int, error) { return 1, nil }\n\nfunc run() error {\n\tget()\n\treturn nil\n}\n"
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	scriptData, err := json.Marshal(fakeScript{})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scriptPath, scriptData, 0o666); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(scratch, "gopls.log")
	client := startProxy(t, Options{
		Gopls:       []string{os.Args[0], "-mode=stdio"},
		Env:         append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+logPath),
		Compiler:    tc,
		Debounce:    -1,
		CodeActions: codeActionsAdapter,
	})
	client.call("initialize", map[string]any{
		"processId": nil,
		"rootUri":   pathURI(root),
		"capabilities": map[string]any{
			"general": map[string]any{"positionEncodings": []string{"utf-8", "utf-16"}},
		},
	})
	client.notify("initialized", map[string]any{})
	client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": pathURI(aPath), "languageId": "logos", "version": 1, "text": denied},
	})
	call := findPos(t, denied, "get()", 1)
	rng := span(call, len("get()"))
	res := client.call("textDocument/codeAction", map[string]any{
		"textDocument": map[string]any{"uri": pathURI(aPath)},
		"range":        rng,
		"context":      map[string]any{"diagnostics": []any{}},
	})
	if string(res) == "null" {
		t.Fatal("codeAction on a denied source answered null, want the compiler autofix")
	}
	if !strings.Contains(string(res), "add ? to propagate the error") {
		t.Errorf("codeAction lacks the compiler autofix: %s", res)
	}
	if !strings.Contains(string(res), `get()?`) {
		t.Errorf("codeAction lacks the ? replacement: %s", res)
	}
	if data, err := os.ReadFile(logPath); err != nil {
		t.Fatal(err)
	} else if strings.Contains(string(data), `"method":"textDocument/codeAction"`) {
		t.Errorf("gopls got a codeAction request for a document with no lowering; the proxy must answer compiler actions directly")
	}
	fixed := strings.Replace(denied, "\tget()\n", "", 1)
	client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": pathURI(aPath), "version": 2},
		"contentChanges": []any{map[string]any{"text": fixed}},
	})
	res2 := client.call("textDocument/codeAction", map[string]any{
		"textDocument": map[string]any{"uri": pathURI(aPath)},
		"range":        rng,
		"context":      map[string]any{"diagnostics": []any{}},
	})
	if strings.Contains(string(res2), "add ? to propagate the error") {
		t.Errorf("a stale autofix crossed the edit: %s", res2)
	}
}
