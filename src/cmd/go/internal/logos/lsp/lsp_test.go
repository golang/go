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
	"internal/testenv"
	"io"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	fakeEnv = "LOGOS_LSP_FAKE_GOPLS"

	fakeLogEnv = "LOGOS_LSP_FAKE_LOG"

	waitLimit = 30 * time.Second
)

func replyWait(t *testing.T) time.Duration {
	t.Helper()
	deadline, ok := t.Deadline()
	if !ok {
		return math.MaxInt64
	}
	left := time.Until(deadline)
	return left - left/10
}

type fakeScript struct {
	Responses map[string]json.RawMessage `json:"responses"`

	Diagnostics map[string]json.RawMessage `json:"diagnostics"`

	Echo bool `json:"echo"`

	DelayMillis int `json:"delayMillis"`

	CrashOn string `json:"crashOn"`

	HoldMethod string `json:"holdMethod"`

	PublishEmptyOnClose bool `json:"publishEmptyOnClose"`

	HoldAddr string `json:"holdAddr"`

	held bool

	docs map[string]string
}

func TestMain(m *testing.M) {
	if os.Getenv(fakeEnv) != "" {
		os.Exit(runFakeGopls())
	}
	os.Exit(m.Run())
}

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
	defer func() {
		if err := logFile.Close(); err != nil {
			fmt.Fprintln(os.Stderr, "fake gopls closing log file:", err)
		}
	}()
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
func echoRespond(wr *writer, id json.RawMessage, v any) error {
	raw, err := encode(v)
	if err != nil {
		return err
	}
	return wr.respond(id, raw)
}

func echoWordByte(c byte) bool {
	return c == '_' || '0' <= c && c <= '9' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

func echoWord(doc string, p Position) Range {
	t := newText([]byte(doc))
	off := t.offset(p)
	start, end := off, off
	for start > 0 && echoWordByte(doc[start-1]) {
		start--
	}
	for end < len(doc) && echoWordByte(doc[end]) {
		end++
	}
	return Range{Start: t.position(start), End: t.position(end)}
}

func echoDiags(doc string) []any {
	i := strings.Index(doc, "ReadFile")
	if i < 0 {
		return []any{}
	}
	t := newText([]byte(doc))
	return []any{map[string]any{
		"range":    Range{Start: t.position(i), End: t.position(i + len("ReadFile"))},
		"severity": 1,
		"message":  stockDiagMessage,
		"source":   "stock",
	}}
}

func echoText(script *fakeScript, m *message) (string, int, string) {
	v, err := decode(m.Params)
	root, _ := v.(map[string]any)
	if err != nil || root == nil {
		return "", 0, ""
	}
	td, _ := root["textDocument"].(map[string]any)
	uri, _ := td["uri"].(string)
	version, _ := number(td["version"])
	if m.Method == "textDocument/didOpen" {
		text, _ := td["text"].(string)
		script.docs[uri] = text
		return uri, version, text
	}
	text := script.docs[uri]
	changes, _ := root["contentChanges"].([]any)
	for _, c := range changes {
		if full, ok := c.(map[string]any)["text"].(string); ok {
			text = full
		}
	}
	script.docs[uri] = text
	return uri, version, text
}

func echoAnswer(wr *writer, script *fakeScript, m *message) error {
	v, err := decode(m.Params)
	root, _ := v.(map[string]any)
	if err != nil || root == nil {
		return wr.respond(m.ID, nullResult)
	}
	td, _ := root["textDocument"].(map[string]any)
	uri, _ := td["uri"].(string)
	where, _ := root["position"].(map[string]any)
	pos, ok := asPosition(where)
	if !ok {
		return wr.respond(m.ID, nullResult)
	}
	span := echoWord(script.docs[uri], pos)
	switch m.Method {
	case "textDocument/hover":
		return echoRespond(wr, m.ID, map[string]any{"contents": map[string]any{"kind": "plaintext", "value": "stock hover"}, "range": span})
	case "textDocument/definition":
		return echoRespond(wr, m.ID, []any{map[string]any{"uri": uri, "range": span}})
	default:
		return echoRespond(wr, m.ID, map[string]any{"isIncomplete": false, "items": []any{map[string]any{"label": "ReadFile", "textEdit": map[string]any{"range": span, "newText": "ReadFile"}}}})
	}
}

func echoHandle(wr *writer, script *fakeScript, m *message) error {
	if script.docs == nil {
		script.docs = map[string]string{}
	}
	if m.isRequest() {
		switch m.Method {
		case "initialize":
			return echoRespond(wr, m.ID, map[string]any{"capabilities": map[string]any{"hoverProvider": true, "definitionProvider": true, "completionProvider": map[string]any{}}})
		case "shutdown":
			return wr.respond(m.ID, nullResult)
		case "textDocument/hover", "textDocument/definition", "textDocument/completion":
			return echoAnswer(wr, script, m)
		default:
			return wr.respond(m.ID, nullResult)
		}
	}
	switch m.Method {
	case "textDocument/didOpen", "textDocument/didChange":
		uri, version, text := echoText(script, m)
		return wr.notify("textDocument/publishDiagnostics", map[string]any{"uri": uri, "version": version, "diagnostics": echoDiags(text)})
	}
	return nil

}
func fakeHandle(wr *writer, script *fakeScript, m *message) error {
	if script.Echo {
		return echoHandle(wr, script, m)
	}
	switch {
	case m.isRequest():
		if m.Method == script.HoldMethod && script.HoldAddr != "" && !script.held {
			script.held = true
			if err := holdForRelease(script.HoldAddr); err != nil {
				return err
			}
		}
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
	case m.Method == "textDocument/didClose" && script.PublishEmptyOnClose:
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if json.Unmarshal(m.Params, &p) != nil {
			return nil
		}
		return wr.notify("textDocument/publishDiagnostics", map[string]any{
			"uri": p.TextDocument.URI, "diagnostics": []any{},
		})
	}
	return nil
}

func holdForRelease(addr string) (err error) {
	d := &net.Dialer{Timeout: waitLimit}
	c, dialErr := d.DialContext(context.Background(), "tcp", addr)
	if dialErr != nil {
		return dialErr
	}
	defer func() {
		if closeErr := c.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}()
	if err = c.SetDeadline(time.Now().Add(waitLimit)); err != nil {
		return err
	}
	var b [1]byte
	_, err = io.ReadFull(c, b[:])
	return err
}

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

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

type testClient struct {
	t      *testing.T
	w      *writer
	closeW func()

	done chan struct{}
	err  error

	mu      sync.Mutex
	cond    *sync.Cond
	msgs    []*message
	nextID  int
	expired bool
}

func startProxy(t *testing.T, opts Options) *testClient {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	c := &testClient{t: t, w: &writer{w: inW}, closeW: func() {
		if err := inW.Close(); err != nil {
			t.Errorf("closing client input pipe: %v", err)
		}
	}, done: make(chan struct{})}
	c.cond = sync.NewCond(&c.mu)
	stderr := new(syncBuffer)
	opts.Stderr = stderr
	ctx := t.Context()
	go func() {
		c.err = Serve(ctx, inR, outW, opts)
		if err := outW.Close(); err != nil {
			t.Errorf("closing output pipe: %v", err)
		}
		close(c.done)
	}()
	go c.readLoop(newReader(outR))
	t.Cleanup(func() {
		if err := inW.Close(); err != nil && !errors.Is(err, io.ErrClosedPipe) {
			t.Errorf("closing client input pipe: %v", err)
		}
		if err := outR.Close(); err != nil && !errors.Is(err, io.ErrClosedPipe) {
			t.Errorf("closing proxy output pipe: %v", err)
		}
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

func TestProxyShutdownDuringHintsStress(t *testing.T) {
	rec := new(hintsRecorder)
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowCall := span(findPos(t, string(low), "os.ReadFile", 0), len("os.ReadFile"))
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{
				DelayMillis: 200,
				Responses: map[string]json.RawMessage{
					"textDocument/hover": toJSON(t, map[string]any{
						"contents": map[string]any{"kind": "markdown", "value": "gopls hover"},
						"range":    lowCall,
					}),
				},
			}
		},
		Tune: func(f *fixture, o *Options) {
			o.Hints = rec.serve
			o.HintsJSON = rec.run
		},
	})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	operand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.request("textDocument/hover", positionParams(aURI, operand))
	f.client.notify("exit", nil)
	err := f.client.wait()
	if err == nil || !strings.Contains(err.Error(), "exit without shutdown") {
		t.Errorf("Serve = %v, want the exit without shutdown error", err)
	}
}

type hintLoad struct {
	mu       sync.Mutex
	cur, max int
	calls    int
}

func (h *hintLoad) run(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error) {
	h.mu.Lock()
	h.cur++
	if h.cur > h.max {
		h.max = h.cur
	}
	h.mu.Unlock()
	select {
	case <-time.After(200 * time.Millisecond):
	case <-ctx.Done():
	}
	h.mu.Lock()
	h.cur--
	h.calls++
	h.mu.Unlock()
	return json.RawMessage(`{}`), nil
}

func (h *hintLoad) snapshot() (int, int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.max, h.calls
}

func TestProxyHintComputationsStayBounded(t *testing.T) {
	var load hintLoad
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowCall := span(findPos(t, string(low), "os.ReadFile", 0), len("os.ReadFile"))
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{
					"contents": map[string]any{"kind": "markdown", "value": "gopls hover"},
					"range":    lowCall,
				}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			o.Hints = func(string, json.RawMessage, json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage("null"), nil
			}
			o.HintsJSON = load.run
		},
	})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	operand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	const rapid = 12
	ids := make([]json.RawMessage, 0, rapid)
	for range rapid {
		ids = append(ids, f.client.request("textDocument/hover", positionParams(aURI, operand)))
	}
	for _, id := range ids {
		f.client.await("textDocument/hover", id)
	}
	if max, _ := load.snapshot(); max > maxHintTasks {
		t.Errorf("hint computations peaked at %d concurrent subprocesses, want at most %d", max, maxHintTasks)
	}
}

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

func (c *testClient) waitFor(what string, match func(*message) bool) *message {
	c.t.Helper()
	timer := time.AfterFunc(replyWait(c.t), func() {
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

func (c *testClient) call(method string, params any) json.RawMessage {
	c.t.Helper()
	m := c.callMessage(method, params)
	if len(m.Error) > 0 {
		c.t.Fatalf("%s failed: %s", method, m.Error)
	}
	return m.Result
}

func (c *testClient) callMessage(method string, params any) *message {
	c.t.Helper()
	return c.await(method, c.request(method, params))
}

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

func (c *testClient) await(method string, id json.RawMessage) *message {
	c.t.Helper()
	return c.waitFor("response to "+method, func(m *message) bool {
		return m.isResponse() && bytes.Equal(m.ID, id)
	})
}

func (c *testClient) notify(method string, params any) {
	c.t.Helper()
	if err := c.w.notify(method, params); err != nil {
		c.t.Fatalf("sending %s: %v", method, err)
	}
}

func (c *testClient) notification(accept func(map[string]any) bool) map[string]any {
	c.t.Helper()
	var found map[string]any
	c.waitFor("textDocument/publishDiagnostics", func(m *message) bool {
		if m.Method != "textDocument/publishDiagnostics" {
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

func (c *testClient) shutdown() error {
	c.t.Helper()
	if res := c.call("shutdown", nil); string(res) != "null" {
		c.t.Errorf("shutdown = %s, want null", res)
	}
	c.notify("exit", nil)
	return c.wait()
}

type fixture struct {
	t      *testing.T
	root   string
	comp   *fakeCompiler
	log    string
	client *testClient
}

type fixtureOptions struct {
	Script func(root string) fakeScript

	Tune func(f *fixture, o *Options)
}

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
	}
	if fo.Tune != nil {
		fo.Tune(f, &opts)
	}
	f.client = startProxy(t, opts)
	f.initialize()
	return f
}

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

func (f *fixture) path(rel string) string { return filepath.Join(f.root, filepath.FromSlash(rel)) }

func (f *fixture) uri(rel string) string { return pathURI(f.path(rel)) }

func (f *fixture) virtURI(rel string) string { return pathURI(virtualPath(f.path(rel))) }

func (f *fixture) open(rel string, version int) {
	f.t.Helper()
	data, err := os.ReadFile(f.path(rel))
	if err != nil {
		f.t.Fatal(err)
	}
	f.openText(rel, version, string(data))
}

func (f *fixture) openText(rel string, version int, text string) {
	f.t.Helper()
	f.client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": f.uri(rel), "languageId": "logos", "version": version, "text": text},
	})
}

func (f *fixture) sync() {
	f.t.Helper()
	f.client.call("test/sync", map[string]any{})
}

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

func toJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

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

func positionParams(uri string, p Position) map[string]any {
	return map[string]any{"textDocument": map[string]any{"uri": uri}, "position": p}
}

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
	opened := f.receivedMethod("textDocument/didOpen")
	if len(opened) != 1 {
		t.Fatalf("gopls got %d didOpen messages, want one for the lowered document: %v", len(opened), opened)
	}
	doc, _ := opened[0]["textDocument"].(map[string]any)
	if doc["uri"] != f.virtURI("a/a.lgo") || doc["languageId"] != "go" {
		t.Errorf("lowered document opened as %v, want %s with lowered Go text", doc, f.virtURI("a/a.lgo"))
	}
	fwdText, _ := doc["text"].(string)
	if !strings.Contains(fwdText, "os.ReadFile(path)") || strings.Contains(fwdText, "ReadFile(path)?") {
		t.Errorf("gopls got the lowered text %q, want the lowered call without the ? marker", fwdText)
	}

	t.Run("hover on an operand inside the ? statement", func(t *testing.T) {
		res := f.client.call("textDocument/hover", positionParams(aURI, origOperand))
		uri, pos := forwardedPosition(t, f, "textDocument/hover", 0)
		if uri != f.virtURI("a/a.lgo") {
			t.Errorf("forwarded hover = %s, want %s", uri, f.virtURI("a/a.lgo"))
		}
		if got := lineOf(fwdText, pos.Line)[pos.Character:]; !strings.HasPrefix(got, "path)") {
			t.Errorf("forwarded hover at %v reaches %q, want the path operand", pos, got)
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
		r, _ := asRange(req["range"].(map[string]any))
		if got := lineOf(fwdText, r.Start.Line)[r.Start.Character:]; !strings.HasPrefix(got, "path)") {
			t.Errorf("code action range starts at %q, want the path operand", got)
		}
		ctx, _ := req["context"].(map[string]any)
		diags, _ := ctx["diagnostics"].([]any)
		dr, _ := asRange(diags[0].(map[string]any)["range"].(map[string]any))
		if got := lineOf(fwdText, dr.Start.Line)[dr.Start.Character:]; !strings.HasPrefix(got, "path)") {
			t.Errorf("diagnostic range starts at %q, want the path operand", got)
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

	operand := shift(findPos(t, edited, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))

	opened := f.receivedMethod("textDocument/didOpen")
	if len(opened) != 1 {
		t.Fatalf("gopls got %d lowered didOpen messages, want the one of the edited text", len(opened))
	}
	doc, _ := opened[0]["textDocument"].(map[string]any)
	if doc["uri"] != f.virtURI("a/a.lgo") {
		t.Errorf("didOpen sent for %v, want the lowered document", doc["uri"])
	}
	fwdText, _ := doc["text"].(string)
	if !strings.Contains(fwdText, "// edited é") {
		t.Errorf("gopls got the lowered text %q, want the edit in it", fwdText)
	}
	var order []string
	for _, m := range f.received() {
		switch m.Method {
		case "textDocument/didOpen", "textDocument/hover":
			order = append(order, m.Method)
		}
	}
	if want := []string{"textDocument/didOpen", "textDocument/hover"}; !slices.Equal(order, want) {
		t.Fatalf("gopls got %v, want the lowered document before the request: %v", order, want)
	}
	_, pos := forwardedPosition(t, f, "textDocument/hover", 0)
	lines := strings.Split(fwdText, "\n")
	if pos.Line < 0 || pos.Line >= len(lines) || pos.Character < 0 || pos.Character > len(lines[pos.Line]) {
		t.Fatalf("hover forwarded at %v, outside the lowered text", pos)
	}
	if got := lines[pos.Line][pos.Character:]; !strings.HasPrefix(got, "path)") {
		t.Errorf("hover forwarded at %v reaches %q, want the path operand of the new lowering", pos, got)
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

func TestProxySnapshotCountsStayAtOne(t *testing.T) {
	const otherLgo = "package a\n\nfunc Other() int { return 1 }\n"
	const secondLgo = "package b\n\nfunc Second() int { return 2 }\n"
	f := newFixture(t, map[string]string{
		"a/a.lgo": aLgo,
		"b/c.lgo": secondLgo,
	}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	f.open("b/c.lgo", 1)
	want := map[string]int{f.path("a"): 1, f.path("b"): 1}
	counts := func() snapshotCountsResult {
		t.Helper()
		var got snapshotCountsResult
		if err := json.Unmarshal(f.client.call("logos/snapshotCounts", map[string]any{}), &got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	check := func(what string, got snapshotCountsResult) {
		t.Helper()
		if !reflect.DeepEqual(got.Dirs, want) || got.DiagCache != 0 {
			t.Fatalf("snapshot counts %s = %+v, want dirs %v with an empty diagnostic cache", what, got, want)
		}
	}
	check("at open", counts())
	change := func(rel, text string, version int) {
		t.Helper()
		f.client.notify("textDocument/didChange", map[string]any{
			"textDocument":   map[string]any{"uri": f.uri(rel), "version": version},
			"contentChanges": []any{map[string]any{"text": text}},
		})
		check(fmt.Sprintf("after %s version %d", rel, version), counts())
	}
	change("a/a.lgo", aLgo+"// edit one\n", 2)
	change("a/a.lgo", aLgo+"// edit two\n", 3)
	if err := os.WriteFile(f.path("a/b.lgo"), []byte(otherLgo), 0o666); err != nil {
		t.Fatal(err)
	}
	f.open("a/b.lgo", 1)
	check("after adding a/b.lgo", counts())
	change("a/b.lgo", otherLgo+"// edit three\n", 2)
	change("b/c.lgo", secondLgo+"// edit four\n", 2)
}

func TestProxyCloseFreesDiagnosticCaches(t *testing.T) {
	const goSrc = "package p\n\nvar _ = 1\n"
	names := []string{"j0.go", "j1.go", "j2.go", "j3.go", "j4.go", "j5.go", "j6.go", "j7.go"}
	files := make(map[string]string, len(names))
	for _, n := range names {
		files[n] = goSrc
	}
	f := newFixture(t, files, fixtureOptions{
		Script: func(root string) fakeScript {
			diags := make(map[string]json.RawMessage, len(names))
			for _, n := range names {
				diags[pathURI(filepath.Join(root, n))] = toJSON(t, []any{
					map[string]any{"range": span(Position{Line: 1, Character: 0}, 1), "severity": 1, "message": "gopls: bad"},
				})
			}
			return fakeScript{Diagnostics: diags, PublishEmptyOnClose: true}
		},
	})
	counts := func() snapshotCountsResult {
		t.Helper()
		var got snapshotCountsResult
		if err := json.Unmarshal(f.client.call("logos/snapshotCounts", map[string]any{}), &got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	for _, n := range names {
		uri := f.uri(n)
		f.client.notify("textDocument/didOpen", map[string]any{
			"textDocument": map[string]any{"uri": uri, "languageId": "go", "version": 1, "text": goSrc},
		})
		f.client.notification(func(m map[string]any) bool {
			items, _ := m["diagnostics"].([]any)
			return m["uri"] == uri && len(items) == 1
		})
		f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": uri}})
	}
	if got := counts(); got.DiagCache != 0 {
		t.Errorf("diagnostic cache holds %d blobs after closing every document, want none retained", got.DiagCache)
	}
}

func TestProxySnapshotCountsCarriesDiagCache(t *testing.T) {
	const goSrc = "package p\n\nvar _ = 1\n"
	names := []string{"k0.go", "k1.go", "k2.go"}
	files := make(map[string]string, len(names))
	for _, n := range names {
		files[n] = goSrc
	}
	f := newFixture(t, files, fixtureOptions{
		Script: func(root string) fakeScript {
			diags := make(map[string]json.RawMessage, len(names))
			for _, n := range names {
				diags[pathURI(filepath.Join(root, n))] = toJSON(t, []any{
					map[string]any{"range": span(Position{Line: 1, Character: 0}, 1), "severity": 1, "message": "gopls: bad"},
				})
			}
			return fakeScript{Diagnostics: diags, PublishEmptyOnClose: true}
		},
	})
	counts := func() (snapshotCountsResult, json.RawMessage) {
		t.Helper()
		raw := f.client.call("logos/snapshotCounts", map[string]any{})
		var got snapshotCountsResult
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		return got, raw
	}
	for _, n := range names {
		uri := f.uri(n)
		f.client.notify("textDocument/didOpen", map[string]any{
			"textDocument": map[string]any{"uri": uri, "languageId": "go", "version": 1, "text": goSrc},
		})
		f.client.notification(func(m map[string]any) bool {
			items, _ := m["diagnostics"].([]any)
			return m["uri"] == uri && len(items) == 1
		})
	}
	got, raw := counts()
	if got.DiagCache != len(names) {
		t.Fatalf("diagCache = %d, want %d with open diagnostics", got.DiagCache, len(names))
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["diagCache"]; !ok {
		t.Fatalf("snapshotCounts reply omits diagCache key: %s", string(raw))
	}
	f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": f.uri(names[0])}})
	f.client.notification(func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return m["uri"] == f.uri(names[0]) && len(items) == 0
	})
	got, _ = counts()
	if got.DiagCache != len(names)-1 {
		t.Fatalf("diagCache after one close = %d, want %d", got.DiagCache, len(names)-1)
	}
	for _, n := range names[1:] {
		f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": f.uri(n)}})
	}
	got, raw = counts()
	if got.DiagCache != 0 {
		t.Fatalf("diagCache after every close = %d, want 0", got.DiagCache)
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["diagCache"]; !ok {
		t.Fatalf("snapshotCounts reply omits diagCache key at zero: %s", string(raw))
	}
}

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
				map[string]any{
					"range":    Range{Start: Position{Line: 6, Character: 0}, End: Position{Line: 7, Character: len(lineOf(string(low), 7))}},
					"severity": 2, "message": "gopls: across the expansion",
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
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))

	params := f.client.notification(func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return m["uri"] == f.uri("a/a.lgo") && len(items) == 5
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
		"gopls: across the expansion": {2, origQ, shift(origQ, 1), ""},
		"compiler: unproven":          {1, origCall.Start, origCall.End, "correctness::example"},
		"compiler: warning":           {2, Position{Line: 2, Character: 0}, Position{Line: 2, Character: 6}, "style::note"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("diagnostics:\n got %v\nwant %v", got, want)
	}
}

func TestProxyDiagnosticsFollowEdits(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aPath := f.path("a/a.lgo")
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Message: "first", Start: [2]int{1, 1}, End: [2]int{1, 8}}))
	f.open("a/a.lgo", 1)
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))
	f.client.notification(func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return len(items) == 1
	})
	f.comp.setDiagnostics(filepath.Dir(aPath))
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "\n"}},
	})
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))
	f.client.notification(func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		v, _ := number(m["version"])
		return len(items) == 0 && v == 2
	})
}

func TestProxyMixedWorkspaceDefinition(t *testing.T) {
	const libLgo = "package lib\n\nimport \"os\"\n\nfunc load(path string) string {\n\tdata := os.ReadFile(path)?\n\treturn string(data)\n}\n\nfunc Hello() string {\n\treturn \"hi\"\n}\n"
	const appGo = "package app\n\nimport \"example.com/m/lib\"\n\nvar Greeting = lib.Hello()\n"
	files := map[string]string{
		"go.mod":      "module example.com/m\n\ngo 1.27\n",
		"lib/lib.lgo": libLgo,
		"app/app.go":  appGo,
	}
	helloOrig := span(findPos(t, libLgo, "Hello", 0), len("Hello"))
	libLow, _, ok := lowerQuestion([]byte(libLgo))
	if !ok {
		t.Fatal("lib fixture does not lower")
	}
	helloLow := span(findPos(t, string(libLow), "Hello", 0), len("Hello"))

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
	for _, o := range opened {
		td, _ := o["textDocument"].(map[string]any)
		switch td["uri"] {
		case f.uri("app/app.go"):
			if td["text"] != appGo {
				t.Errorf("gopls opened the Go document with %q, want the disk bytes", td["text"])
			}
		case f.virtURI("lib/lib.lgo"):
			if text, _ := td["text"].(string); !strings.Contains(text, "func Hello() string") {
				t.Errorf("gopls opened the lowered dependency with %q, want the lowered lib", text)
			}
		default:
			t.Errorf("gopls opened unexpected %v: no virtual document may exist for a Go file", td["uri"])
		}
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

func TestProxyClosesLoweredDocuments(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aPath := f.path("a/a.lgo")
	f.comp.setDiagnostics(filepath.Dir(aPath),
		compilerDiag(t, diagSpec{Path: aPath, Severity: "deny", Message: "m", Start: [2]int{1, 1}, End: [2]int{1, 2}}))
	f.open("a/a.lgo", 1)
	f.client.call("textDocument/hover", positionParams(f.uri("a/a.lgo"), Position{}))
	f.client.notification(func(m map[string]any) bool {
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
	f.client.notification(func(m map[string]any) bool {
		items, _ := m["diagnostics"].([]any)
		return m["uri"] == f.uri("a/a.lgo") && len(items) == 0
	})
}

func TestProxyAnswersRequestsInFlightWhenClientCloses(t *testing.T) {
	ln, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.ParseIP("127.0.0.1")})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ln.Close(); err != nil {
			t.Errorf("closing gate listener: %v", err)
		}
	}()
	if err := ln.SetDeadline(time.Now().Add(waitLimit)); err != nil {
		t.Fatalf("arming the gate accept: %v", err)
	}
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{
				HoldMethod: "textDocument/hover",
				HoldAddr:   ln.Addr().String(),
				Responses:  map[string]json.RawMessage{"textDocument/hover": toJSON(t, map[string]any{"contents": "late"})},
			}
		},
	})
	f.open("a/a.lgo", 1)
	raw := toJSON(t, positionParams(f.uri("a/a.lgo"), Position{Line: 5, Character: 3}))
	if err := f.client.w.write(&message{ID: json.RawMessage("99"), Method: "textDocument/hover", Params: raw}); err != nil {
		t.Fatal(err)
	}
	gate, err := ln.Accept()
	if err != nil {
		t.Fatalf("the held backend never took the hover: %v", err)
	}
	defer func() {
		if err := gate.Close(); err != nil {
			t.Errorf("closing gate conn: %v", err)
		}
	}()
	if err := gate.SetWriteDeadline(time.Now().Add(waitLimit)); err != nil {
		t.Fatalf("arming the gate write: %v", err)
	}
	f.client.closeW()
	if _, err := gate.Write([]byte{1}); err != nil {
		t.Fatalf("releasing the held backend: %v", err)
	}
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

func TestServeExitWithoutShutdown(t *testing.T) {
	f := newFixture(t, nil, fixtureOptions{})
	f.client.notify("exit", nil)
	if err := f.client.wait(); err == nil || !strings.Contains(err.Error(), "exit without shutdown") {
		t.Errorf("Serve = %v, want the exit without shutdown error", err)
	}
}

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
func TestServePrefersClientCloseOverGoplsExit(t *testing.T) {
	t.Run("close", func(t *testing.T) {
		for i := range 50 {
			f := newFixture(t, nil, fixtureOptions{})
			if err := f.client.shutdown(); err != nil {
				t.Fatalf("iteration %d: Serve = %v, want nil", i, err)
			}
		}
	})
	t.Run("death", func(t *testing.T) {
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
	})
}

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

func lspHandshake(t *testing.T, stdin io.WriteCloser) {
	t.Helper()
	for _, body := range []string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}`,
		`{"jsonrpc":"2.0","method":"initialized","params":{}}`,
		`{"jsonrpc":"2.0","id":2,"method":"shutdown","params":null}`,
		`{"jsonrpc":"2.0","method":"exit","params":{}}`,
	} {
		if _, err := fmt.Fprintf(stdin, "Content-Length: %d\r\n\r\n%s", len(body), body); err != nil {
			t.Fatal(err)
		}
	}
	if err := stdin.Close(); err != nil {
		t.Fatal(err)
	}
}

func probeGoplsThroughLogos(t *testing.T, logosBin, fakeDir string) ([]string, string) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(fakeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	pathLog := filepath.Join(t.TempDir(), "path.log")
	argsLog := filepath.Join(t.TempDir(), "args.log")
	wrapper := "#!/bin/sh\necho \"$PATH\" >> \"" + pathLog + "\"\necho \"$@\" >> \"" + argsLog + "\"\nexec \"" + exe + "\" \"$@\"\n"
	if err := os.WriteFile(filepath.Join(fakeDir, "gopls"), []byte(wrapper), 0o755); err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(t.TempDir(), "script.json")
	if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
		t.Fatal(err)
	}
	sep := string(os.PathListSeparator)
	cmd := exec.CommandContext(t.Context(), logosBin, "lsp")
	cmd.Env = append(os.Environ(),
		"PATH="+fakeDir+sep+"/usr/bin"+sep+"/bin",
		fakeEnv+"="+scriptPath,
		fakeLogEnv+"="+filepath.Join(t.TempDir(), "gopls.log"))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stdout = io.Discard
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	lspHandshake(t, stdin)
	if err := cmd.Wait(); err != nil {
		t.Fatalf("logos lsp = %v, stderr %q", err, stderr.String())
	}
	pathData, err := os.ReadFile(pathLog)
	if err != nil {
		t.Fatal(err)
	}
	argsData, err := os.ReadFile(argsLog)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(pathData)), sep), string(argsData)
}

func TestLogosLspGoplsDiscovery(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Setenv("GOFLAGS", "")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOPROXY", "off")
	logosBin := filepath.Join(filepath.Dir(testenv.GoToolPath(t)), "logos")
	if _, err := os.Stat(logosBin); err != nil {
		t.Fatalf("logos binary not present: build the tree first: %v", err)
	}
	binDir := filepath.Dir(logosBin)
	t.Run("missing", func(t *testing.T) {
		cmd := exec.CommandContext(t.Context(), logosBin, "lsp")
		cmd.Env = append(os.Environ(), "PATH="+t.TempDir())
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		if err := cmd.Run(); err == nil || !strings.Contains(stderr.String(), "gopls not found on PATH") {
			t.Errorf("logos lsp without gopls = %v, stderr %q, want the not-found error", err, stderr.String())
		}
	})
	t.Run("found", func(t *testing.T) {
		if _, err := exec.LookPath("gopls"); err != nil {
			t.Skipf("real gopls not on PATH: %v", err)
		}
		cmd := exec.CommandContext(t.Context(), logosBin, "lsp")
		cmd.Env = os.Environ()
		stdin, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		cmd.Stdout = io.Discard
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		lspHandshake(t, stdin)
		waitErr := make(chan error, 1)
		go func() { waitErr <- cmd.Wait() }()
		select {
		case err := <-waitErr:
			if err != nil {
				t.Fatalf("logos lsp with real gopls = %v, stderr %q", err, stderr.String())
			}
		case <-time.After(90 * time.Second):
			_ = cmd.Process.Kill()
			t.Fatal("logos lsp with real gopls did not exit within 90s")
		}
		if strings.Contains(stderr.String(), "gopls not found on PATH") {
			t.Errorf("logos lsp with real gopls on PATH reported %q", stderr.String())
		}
	})
	t.Run("order", func(t *testing.T) {
		fakeDir := t.TempDir()
		entries, argv := probeGoplsThroughLogos(t, logosBin, fakeDir)
		if !slices.Contains(entries, binDir) || !slices.Contains(entries, fakeDir) {
			t.Errorf("gopls PATH = %q, want the toolchain bin and the fake gopls on it", entries)
		}
		if !strings.Contains(argv, "-mode=stdio") {
			t.Errorf("gopls argv = %q, want -mode=stdio", argv)
		}
	})
	t.Run("spaces", func(t *testing.T) {
		fakeDir := filepath.Join(t.TempDir(), "dir with spaces")
		entries, argv := probeGoplsThroughLogos(t, logosBin, fakeDir)
		if !slices.Contains(entries, binDir) || !slices.Contains(entries, fakeDir) {
			t.Errorf("gopls PATH = %q, want the toolchain bin and the spaced fake gopls dir on it", entries)
		}
		if !strings.Contains(argv, "-mode=stdio") {
			t.Errorf("gopls argv = %q, want -mode=stdio", argv)
		}
	})
}

func TestLogosLspStdioWiring(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Setenv("GOFLAGS", "")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOPROXY", "off")
	logosBin := filepath.Join(filepath.Dir(testenv.GoToolPath(t)), "logos")
	if _, err := os.Stat(logosBin); err != nil {
		t.Fatalf("logos binary not present: build the tree first: %v", err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	fakeDir := t.TempDir()
	scriptPath := filepath.Join(t.TempDir(), "script.json")
	if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
		t.Fatal(err)
	}
	wrapper := "#!/bin/sh\nexec \"" + exe + "\" \"$@\"\n"
	if err := os.WriteFile(filepath.Join(fakeDir, "gopls"), []byte(wrapper), 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(t.Context(), logosBin, "lsp")
	cmd.Env = append(os.Environ(),
		"PATH="+fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"),
		fakeEnv+"="+scriptPath,
		fakeLogEnv+"="+filepath.Join(t.TempDir(), "gopls.log"))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	replies := make(chan *message, 16)
	go func() {
		rd := newReader(stdout)
		defer close(replies)
		for {
			m, err := rd.read()
			if err != nil {
				return
			}
			select {
			case replies <- m:
			default:
			}
		}
	}()
	w := &writer{w: stdin}
	if err := w.write(&message{ID: json.RawMessage("1"), Method: "initialize", Params: json.RawMessage(`{"capabilities":{}}`)}); err != nil {
		t.Fatal(err)
	}
	select {
	case m, ok := <-replies:
		if !ok {
			t.Fatal("logos lsp closed stdout without an initialize reply")
		}
		if !m.isResponse() || string(m.ID) != "1" {
			t.Fatalf("initialize reply = %+v, want the response with id 1", m)
		}
		v, err := decode(m.Result)
		if err != nil {
			t.Fatal(err)
		}
		caps, _ := v.(map[string]any)["capabilities"].(map[string]any)
		if _, ok := caps["hoverProvider"]; !ok {
			t.Errorf("initialize reply capabilities = %v, want hoverProvider", caps)
		}
	case <-time.After(waitLimit):
		t.Fatal("logos lsp sent no initialize reply on stdout")
	}
	if err := w.write(&message{ID: json.RawMessage("2"), Method: "shutdown", Params: json.RawMessage("null")}); err != nil {
		t.Fatal(err)
	}
	if err := w.notify("exit", nil); err != nil {
		t.Fatal(err)
	}
	if err := stdin.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("logos lsp = %v, stderr %q", err, stderr.String())
	}
}

func TestServeRejectsBadFrames(t *testing.T) {
	frame := func(body string) string {
		return fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(body), body)
	}
	good := frame(`{"jsonrpc":"2.0","id":1,"method":"shutdown","params":null}`) +
		frame(`{"jsonrpc":"2.0","method":"exit","params":null}`)
	for _, c := range []struct {
		name string
		in   string
	}{
		{"missing length", "X-Other: 1\r\n\r\n{}"},
		{"corrupt length", "Content-Length: abc\r\n\r\n{}"},
		{"empty value", "Content-Length:\r\n\r\n{}"},
		{"junk suffix", "Content-Length: 12x\r\n\r\n{}"},
		{"negative", "Content-Length: -4\r\n\r\n{}"},
		{"truncated", "Content-Length: 99\r\n\r\n{}"},
		{"cut header", "Content-Length: 3"},
		{"over the cap", "Content-Length: " + strconv.Itoa(maxMessage+1) + "\r\n\r\n"},
	} {
		t.Run(c.name, func(t *testing.T) {
			scratch := t.TempDir()
			scriptPath := filepath.Join(scratch, "script.json")
			if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
				t.Fatal(err)
			}
			opts := Options{
				Gopls:    []string{os.Args[0], "-mode=stdio"},
				Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
				Compiler: newFakeCompiler(),
				Stderr:   new(syncBuffer),
			}
			inR, inW, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			outR, outW, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			ctx := t.Context()
			go func() { done <- Serve(ctx, inR, outW, opts) }()
			if _, err := io.WriteString(inW, c.in); err != nil {
				t.Fatal(err)
			}
			if err := inW.Close(); err != nil {
				t.Errorf("closing stdin writer: %v", err)
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatalf("Serve accepted a %s frame, want an error", c.name)
				}
			case <-time.After(waitLimit):
				t.Fatal("Serve did not return after the bad frame")
			}
			if err := outW.Close(); err != nil {
				t.Errorf("closing stdout writer: %v", err)
			}
			if err := outR.Close(); err != nil {
				t.Errorf("closing stdout reader: %v", err)
			}
		})
	}
	t.Run("good frames", func(t *testing.T) {
		scratch := t.TempDir()
		scriptPath := filepath.Join(scratch, "script.json")
		if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
			t.Fatal(err)
		}
		opts := Options{
			Gopls:    []string{os.Args[0], "-mode=stdio"},
			Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
			Compiler: newFakeCompiler(),
			Stderr:   new(syncBuffer),
		}
		inR, inW, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		outR, outW, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		ctx := t.Context()
		go func() { done <- Serve(ctx, inR, outW, opts) }()
		if _, err := io.WriteString(inW, good); err != nil {
			t.Fatal(err)
		}
		if err := inW.Close(); err != nil {
			t.Errorf("closing stdin writer: %v", err)
		}
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("Serve over good frames = %v, want nil", err)
			}
		case <-time.After(waitLimit):
			t.Fatal("Serve did not return after the good frames")
		}
		if err := outW.Close(); err != nil {
			t.Errorf("closing stdout writer: %v", err)
		}
		if err := outR.Close(); err != nil {
			t.Errorf("closing stdout reader: %v", err)
		}
	})
}

func TestServeAcceptsNonCanonicalHeaders(t *testing.T) {
	shutdown := `{"jsonrpc":"2.0","id":1,"method":"shutdown","params":null}`
	bye := `{"jsonrpc":"2.0","method":"exit","params":{}}`
	for _, c := range []struct {
		name string
		head func(n int) string
	}{
		{"content type first", func(n int) string {
			return fmt.Sprintf("Content-Type: application/vscode-jsonrpc; charset=utf-8\r\nContent-Length: %d\r\n\r\n", n)
		}},
		{"lowercase length", func(n int) string {
			return fmt.Sprintf("content-length: %d\r\n\r\n", n)
		}},
		{"uppercase length", func(n int) string {
			return fmt.Sprintf("CONTENT-LENGTH: %d\r\n\r\n", n)
		}},
		{"length after a custom header", func(n int) string {
			return fmt.Sprintf("X-Custom: 1\r\nContent-Length: %d\r\nContent-Type: application/vscode-jsonrpc\r\n\r\n", n)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			scratch := t.TempDir()
			scriptPath := filepath.Join(scratch, "script.json")
			if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
				t.Fatal(err)
			}
			opts := Options{
				Gopls:    []string{os.Args[0], "-mode=stdio"},
				Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
				Compiler: newFakeCompiler(),
				Stderr:   new(syncBuffer),
			}
			inR, inW, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			outR, outW, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			ctx := t.Context()
			go func() { done <- Serve(ctx, inR, outW, opts) }()
			in := c.head(len(shutdown)) + shutdown + c.head(len(bye)) + bye
			if _, err := io.WriteString(inW, in); err != nil {
				t.Fatal(err)
			}
			if err := inW.Close(); err != nil {
				t.Errorf("closing stdin writer: %v", err)
			}
			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("Serve over %s headers = %v, want nil", c.name, err)
				}
			case <-time.After(waitLimit):
				t.Fatal("Serve did not return after the non-canonical headers")
			}
			if err := outW.Close(); err != nil {
				t.Errorf("closing stdout writer: %v", err)
			}
			if err := outR.Close(); err != nil {
				t.Errorf("closing stdout reader: %v", err)
			}
		})
	}
}

func TestServeAnswersParseErrorAndContinues(t *testing.T) {
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
		t.Fatal(err)
	}
	opts := Options{
		Gopls:    []string{os.Args[0], "-mode=stdio"},
		Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
		Compiler: newFakeCompiler(),
		Stderr:   new(syncBuffer),
	}
	inR, inW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	outR, outW := io.Pipe()
	defer func() {
		if err := outW.Close(); err != nil {
			t.Errorf("closing stdout writer: %v", err)
		}
		if err := outR.Close(); err != nil {
			t.Errorf("closing stdout reader: %v", err)
		}
	}()
	done := make(chan error, 1)
	ctx := t.Context()
	go func() { done <- Serve(ctx, inR, outW, opts) }()
	frame := func(body string) string {
		return fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(body), body)
	}
	zero := frame("")
	bad := frame(`{"m":}`)
	shutdown := frame(`{"jsonrpc":"2.0","id":2,"method":"shutdown","params":null}`)
	bye := frame(`{"jsonrpc":"2.0","method":"exit","params":{}}`)
	if _, err := io.WriteString(inW, zero+bad+shutdown+bye); err != nil {
		t.Fatal(err)
	}
	if err := inW.Close(); err != nil {
		t.Errorf("closing stdin writer: %v", err)
	}
	deadline := time.After(waitLimit)
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		log, err := os.ReadFile(filepath.Join(scratch, "gopls.log"))
		if err == nil && strings.Contains(string(log), `"method":"shutdown"`) {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("Serve returned before the client read its replies: %v", err)
		case <-deadline:
			t.Fatal("gopls did not receive shutdown")
		case <-tick.C:
		}
	}
	select {
	case err := <-done:
		t.Fatalf("Serve returned with undelivered client replies: %v", err)
	case <-time.After(250 * time.Millisecond):
	}
	rd := newReader(outR)
	readFrame := func() *message {
		t.Helper()
		type result struct {
			m   *message
			err error
		}
		ch := make(chan result, 1)
		go func() {
			m, err := rd.read()
			ch <- result{m, err}
		}()
		select {
		case r := <-ch:
			if r.err != nil {
				t.Fatalf("reading the proxy reply: %v", r.err)
			}
			return r.m
		case <-time.After(waitLimit):
			t.Fatal("the proxy sent no reply frame")
			return nil
		}
	}
	for _, what := range []string{"zero-length", "malformed"} {
		m := readFrame()
		if !m.isResponse() || string(m.ID) != "null" {
			t.Fatalf("%s reply = %+v, want a response with a null id", what, m)
		}
		v, err := decode(m.Error)
		if err != nil {
			t.Fatal(err)
		}
		code, _ := number(v.(map[string]any)["code"])
		if code != codeParseError {
			t.Errorf("%s reply code = %v, want %d", what, code, codeParseError)
		}
	}
	second := readFrame()
	if !second.isResponse() || string(second.ID) != "2" {
		t.Fatalf("shutdown reply = %+v, want the response with id 2", second)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Serve after a parse error = %v, want nil", err)
		}
	case <-time.After(waitLimit):
		t.Fatal("Serve did not return after the parse error and shutdown")
	}
}

func TestServeLargeFrames(t *testing.T) {
	serve := func(t *testing.T, in string) (chan error, *reader, func()) {
		t.Helper()
		scratch := t.TempDir()
		scriptPath := filepath.Join(scratch, "script.json")
		if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
			t.Fatal(err)
		}
		opts := Options{
			Gopls:    []string{os.Args[0], "-mode=stdio"},
			Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
			Compiler: newFakeCompiler(),
			Stderr:   new(syncBuffer),
		}
		inR, inW, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		outR, outW, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		ctx := t.Context()
		go func() { done <- Serve(ctx, inR, outW, opts) }()
		if _, err := io.WriteString(inW, in); err != nil {
			t.Fatal(err)
		}
		if err := inW.Close(); err != nil {
			t.Errorf("closing stdin writer: %v", err)
		}
		return done, newReader(outR), func() {
			if err := outW.Close(); err != nil {
				t.Errorf("closing stdout writer: %v", err)
			}
			if err := outR.Close(); err != nil {
				t.Errorf("closing stdout reader: %v", err)
			}
		}
	}
	readFrame := func(t *testing.T, rd *reader) *message {
		t.Helper()
		type result struct {
			m   *message
			err error
		}
		ch := make(chan result, 1)
		go func() {
			m, err := rd.read()
			ch <- result{m, err}
		}()
		select {
		case r := <-ch:
			if r.err != nil {
				t.Fatalf("reading the proxy reply: %v", r.err)
			}
			return r.m
		case <-time.After(waitLimit):
			t.Fatal("the proxy sent no reply frame")
			return nil
		}
	}
	frame := func(body string) string {
		return fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(body), body)
	}
	t.Run("valid", func(t *testing.T) {
		pad := strings.Repeat("x", 200000)
		hover := fmt.Sprintf(`{"jsonrpc":"2.0","id":1,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///w/a.lgo"},"position":{"line":0,"character":0},"pad":%q}}`, pad)
		bye := frame(`{"jsonrpc":"2.0","method":"exit","params":{}}`)
		shutdown := frame(`{"jsonrpc":"2.0","id":2,"method":"shutdown","params":null}`)
		done, rd, cleanup := serve(t, frame(hover)+shutdown+bye)
		defer cleanup()
		first := readFrame(t, rd)
		if !first.isResponse() || string(first.ID) != "1" {
			t.Fatalf("hover reply = %+v, want the response with id 1", first)
		}
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("Serve over a large valid frame = %v, want nil", err)
			}
		case <-time.After(waitLimit):
			t.Fatal("Serve did not return after the large valid frame")
		}
	})
	t.Run("truncated", func(t *testing.T) {
		done, _, cleanup := serve(t, fmt.Sprintf("Content-Length: %d\r\n\r\n%s", 64<<20, "xx"))
		defer cleanup()
		select {
		case err := <-done:
			if err == nil {
				t.Fatal("Serve accepted a truncated 64 MiB frame, want an error")
			}
		case <-time.After(waitLimit):
			t.Fatal("Serve did not return after the truncated frame")
		}
	})
}

func TestProxyLeavesForeignJSONAlone(t *testing.T) {
	low, _, _ := lowerQuestion([]byte(aLgo))
	lowRead := span(findPos(t, string(low), "Read(", 0), len("Read"))
	foreign := map[string]any{"uri": "file:///foreign/x.go", "range": span(Position{Line: 5, Character: 5}, 3), "note": "keep"}
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(root string) fakeScript {
			virt := pathURI(filepath.Join(root, "a", "a.go"))
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/definition": toJSON(t, []any{
					map[string]any{"uri": virt, "range": lowRead},
					foreign,
					virt,
					map[string]any{"line": 5, "character": 1, "extra": true},
				}),
				"textDocument/hover": toJSON(t, map[string]any{
					"contents": map[string]any{"kind": "plaintext", "value": "known"},
					"range":    lowRead,
					"extra":    map[string]any{"note": "keep"},
				}),
			}}
		},
	})
	f.open("a/a.lgo", 1)
	f.sync()
	aURI := f.uri("a/a.lgo")
	origOperand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	origRead := span(findPos(t, aLgo, "Read(", 0), len("Read"))
	virt := f.virtURI("a/a.lgo")
	res := f.client.call("textDocument/definition", positionParams(aURI, origOperand))
	sameJSON(t, "definition keeps the foreign location", res, []any{
		map[string]any{"uri": aURI, "range": origRead},
		foreign,
		virt,
		map[string]any{"line": 5, "character": 1, "extra": true},
	})
	hres := f.client.call("textDocument/hover", positionParams(aURI, origOperand))
	sameJSON(t, "hover keeps the foreign field", hres, map[string]any{
		"contents": map[string]any{"kind": "plaintext", "value": "known"},
		"range":    origRead,
		"extra":    map[string]any{"note": "keep"},
	})
	if err := f.client.shutdown(); err != nil {
		t.Errorf("Serve = %v, want nil", err)
	}
}

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
	f.client.call("textDocument/hover", positionParams(f.uri("p/p.go"), Position{}))
	params := f.client.notification(func(m map[string]any) bool {
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

func TestProxyCodeActionWithoutLoweringUsesCompiler(t *testing.T) {
	const denied = "package a\n\nimport \"errors\"\n\nvar errGet = errors.New(\"get\")\n\nvar failGet = false\n\nfunc get() (int, error) {\n\tif failGet {\n\t\treturn 0, errGet\n\t}\n\treturn 1, nil\n}\n\nfunc run() error {\n\tget()\n\treturn nil\n}\n"
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Tune: func(f *fixture, o *Options) {
			o.CodeActions = codeActionsAdapter
		},
	})
	aPath := f.path("a/a.lgo")
	f.comp.setMissing(aPath)
	call := findPos(t, denied, "get()", 1)
	line := autofixLine(t, fixDiag{Schema: "1.0", File: aPath, Start: call, Len: len("get()"), Description: "add ? to propagate the error", Replacement: "get()?"})
	d, ok := parseDiagnostic(line)
	if !ok {
		t.Fatalf("parseDiagnostic rejected %s", line)
	}
	f.comp.setDiagnostics(filepath.Dir(aPath), d)
	f.openText("a/a.lgo", 1, denied)
	rng := span(call, len("get()"))
	res := f.client.call("textDocument/codeAction", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
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
	for _, m := range f.received() {
		if m.Method == "textDocument/codeAction" {
			t.Errorf("gopls got a codeAction request for a document with no lowering; the proxy must answer compiler actions directly")
		}
	}
	fixed := strings.Replace(denied, "\tget()\n", "", 1)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": fixed}},
	})
	f.comp.setDiagnostics(filepath.Dir(aPath))
	res2 := f.client.call("textDocument/codeAction", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        rng,
		"context":      map[string]any{"diagnostics": []any{}},
	})
	if strings.Contains(string(res2), "add ? to propagate the error") {
		t.Errorf("a stale autofix crossed the edit: %s", res2)
	}
}

func TestProxyCodeActionInflightEditDropsWholeAnswer(t *testing.T) {
	ln, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.ParseIP("127.0.0.1")})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ln.Close(); err != nil {
			t.Errorf("closing gate listener: %v", err)
		}
	}()
	if err := ln.SetDeadline(time.Now().Add(waitLimit)); err != nil {
		t.Fatalf("arming the gate accept: %v", err)
	}
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo, "b/b.lgo": "package b\n\nSYNTAX ERROR\n"}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{
				HoldMethod: "textDocument/codeAction",
				HoldAddr:   ln.Addr().String(),
				Responses: map[string]json.RawMessage{
					"textDocument/codeAction": toJSON(t, []any{map[string]any{"title": "from gopls"}}),
				},
			}
		},
		Tune: func(f *fixture, o *Options) {
			o.CodeActions = codeActionsAdapter
		},
	})
	aPath := f.path("a/a.lgo")
	line := autofixLine(t, fixDiag{Schema: "1.0", File: aPath, Start: Position{Line: 5, Character: 1}, Len: 4, Description: "stale-proof fix", Replacement: "p"})
	d, ok := parseDiagnostic(line)
	if !ok {
		t.Fatalf("parseDiagnostic rejected %s", line)
	}
	f.comp.setDiagnostics(filepath.Dir(aPath), d)
	f.open("a/a.lgo", 1)
	f.open("b/b.lgo", 1)
	params := map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
		"range":        span(Position{Line: 5, Character: 1}, 4),
		"context":      map[string]any{"diagnostics": []any{}},
	}
	id := f.client.request("textDocument/codeAction", params)
	gate, err := ln.Accept()
	if err != nil {
		t.Fatalf("the held backend never dialed the gate: %v", err)
	}
	defer func() {
		if err := gate.Close(); err != nil {
			t.Errorf("closing gate conn: %v", err)
		}
	}()
	if err := gate.SetWriteDeadline(time.Now().Add(waitLimit)); err != nil {
		t.Fatalf("arming the gate write: %v", err)
	}
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/a.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": aLgo + "// crossed\n"}},
	})
	if res := f.client.call("textDocument/hover", positionParams(f.uri("b/b.lgo"), Position{Line: 0})); string(res) != "null" {
		t.Fatalf("hover on a source with no lowering = %s, want null", res)
	}
	if _, err := gate.Write([]byte{1}); err != nil {
		t.Fatalf("releasing the held backend: %v", err)
	}
	stale := f.client.await("textDocument/codeAction", id)
	sameJSON(t, "stale codeAction crossed by an edit", stale.Result, []any{})
	fresh := f.client.call("textDocument/codeAction", params)
	if !strings.Contains(string(fresh), "from gopls") {
		t.Errorf("fresh codeAction lost the backend action: %s", fresh)
	}
	if !strings.Contains(string(fresh), "stale-proof fix") {
		t.Errorf("fresh codeAction lost the current autofix: %s", fresh)
	}
}

func TestProxyOneRunForTwoEditsBeforeHover(t *testing.T) {
	edited1 := aLgo + "// v2\n"
	edited2 := aLgo + "// v2\n// v3\n"
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	aURI, aPath := f.uri("a/a.lgo"), f.path("a/a.lgo")
	for i, text := range []string{edited1, edited2} {
		f.client.notify("textDocument/didChange", map[string]any{
			"textDocument":   map[string]any{"uri": aURI, "version": i + 2},
			"contentChanges": []any{map[string]any{"text": text}},
		})
	}
	operand := shift(findPos(t, edited2, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))

	calls := f.comp.lowerCalls()
	if len(calls) != 1 {
		t.Fatalf("the proxy compiled %d times for one open, two edits, and one hover, want the one run the hover asked for: %v", len(calls), calls)
	}
	buffers := f.comp.bufferCalls()
	if last := buffers[len(buffers)-1]; string(last[aPath]) != edited2 {
		t.Errorf("the one run lowered %q, want the text of version 3", last[aPath])
	}
	opened := f.receivedMethod("textDocument/didOpen")
	if len(opened) != 1 {
		t.Fatalf("gopls got %d lowered didOpen messages, want the one of the version 3 text", len(opened))
	}
	fwdText, _ := opened[0]["textDocument"].(map[string]any)["text"].(string)
	_, pos := forwardedPosition(t, f, "textDocument/hover", 0)
	lines := strings.Split(fwdText, "\n")
	if pos.Line < 0 || pos.Line >= len(lines) || pos.Character < 0 || pos.Character > len(lines[pos.Line]) {
		t.Fatalf("hover forwarded at %v, outside the lowered text", pos)
	} else if got := lines[pos.Line][pos.Character:]; !strings.HasPrefix(got, "path)") {
		t.Errorf("the hover was mapped at %v reaching %q, want the path operand", pos, got)
	}
}

func TestProxyDidChangeAloneDoesNotLower(t *testing.T) {
	edited := aLgo + "// v2\n"
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": edited}},
	})
	f.client.call("initialize", map[string]any{"capabilities": map[string]any{}})
	if calls := f.comp.lowerCalls(); len(calls) != 0 {
		t.Fatalf("the proxy compiled %d times for an edit with no request, want none: %v", len(calls), calls)
	}
	operand := shift(findPos(t, edited, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	if calls := f.comp.lowerCalls(); len(calls) != 1 {
		t.Fatalf("the proxy compiled %d times for one edit and one hover, want the single hover run: %v", len(calls), calls)
	}
}

func TestProxyWatchedAloneDoesNotLower(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	f.client.notify("workspace/didChangeWatchedFiles", map[string]any{
		"changes": []any{map[string]any{"uri": aURI, "type": 2}},
	})
	f.client.call("initialize", map[string]any{"capabilities": map[string]any{}})
	if calls := f.comp.lowerCalls(); len(calls) != 0 {
		t.Fatalf("the proxy compiled %d times for a watched-file notification with no request, want none: %v", len(calls), calls)
	}
	operand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	if calls := f.comp.lowerCalls(); len(calls) != 1 {
		t.Fatalf("the proxy compiled %d times for one watched notification and one hover, want the single hover run: %v", len(calls), calls)
	}
}

func TestProxyDidCloseOfSavedDocDoesNotLower(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	f.client.notify("textDocument/didClose", map[string]any{
		"textDocument": map[string]any{"uri": aURI},
	})
	f.client.call("initialize", map[string]any{"capabilities": map[string]any{}})
	if calls := f.comp.lowerCalls(); len(calls) != 0 {
		t.Fatalf("the proxy compiled %d times for closing a saved document with no request, want none: %v", len(calls), calls)
	}
	operand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	f.open("a/a.lgo", 2)
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	if calls := f.comp.lowerCalls(); len(calls) != 1 {
		t.Fatalf("the proxy compiled %d times for one reopen and one hover after the saved close, want the single hover run: %v", len(calls), calls)
	}
}

func TestProxyDidCloseAfterSaveDoesNotLowerAgain(t *testing.T) {
	saved := aLgo + "// v2\n"
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aPath, aURI := f.path("a/a.lgo"), f.uri("a/a.lgo")
	f.open("a/a.lgo", 1)
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": saved}},
	})
	if err := os.WriteFile(aPath, []byte(saved), 0o644); err != nil {
		t.Fatal(err)
	}
	f.client.notify("textDocument/didSave", map[string]any{"textDocument": map[string]any{"uri": aURI}})
	f.client.call("initialize", map[string]any{"capabilities": map[string]any{}})
	if calls := f.comp.lowerCalls(); len(calls) != 1 {
		t.Fatalf("the proxy compiled %d times for one didSave, want the save's single run: %v", len(calls), calls)
	}
	f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": aURI}})
	f.client.call("initialize", map[string]any{"capabilities": map[string]any{}})
	if calls := f.comp.lowerCalls(); len(calls) != 1 {
		t.Fatalf("the proxy compiled %d times after closing the saved document, want still the save's single run: %v", len(calls), calls)
	}
}

func TestProxyReversedChangeLeavesBuffer(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	aURI, aPath := f.uri("a/a.lgo"), f.path("a/a.lgo")
	operand := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument": map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{
			"range": map[string]any{
				"start": Position{Line: 5, Character: 10},
				"end":   Position{Line: 5, Character: 3},
			},
			"text": "XXX",
		}},
	})
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	buffers := f.comp.bufferCalls()
	if len(buffers) == 0 {
		t.Fatal("hover lowered nothing after the reversed change")
	}
	if last := buffers[len(buffers)-1]; string(last[aPath]) != aLgo {
		t.Errorf("a reversed change corrupted the overlay: %q", last[aPath])
	}
	edited := aLgo + "// v3\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 3},
		"contentChanges": []any{map[string]any{"text": edited}},
	})
	f.client.call("textDocument/hover", positionParams(aURI, shift(findPos(t, edited, "ReadFile(path)?", 0), len("ReadFile("))))
	buffers = f.comp.bufferCalls()
	if last := buffers[len(buffers)-1]; string(last[aPath]) != edited {
		t.Errorf("the valid change after a reversed one lowered %q, want the version 3 text", last[aPath])
	}
}

func TestProxyConcurrentHoversShareOneRun(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{DelayMillis: 50}
		},
	})
	f.open("a/a.lgo", 1)
	aURI := f.uri("a/a.lgo")
	p := Position{Line: 5, Character: 3}
	ids := []json.RawMessage{f.client.request("textDocument/hover", positionParams(aURI, p)), f.client.request("textDocument/hover", positionParams(aURI, p))}
	for _, id := range ids {
		f.client.await("textDocument/hover", id)
	}
	if got := len(f.comp.lowerCalls()); got != 1 {
		t.Errorf("two concurrent hovers caused %d lowerings, want the shared one", got)
	}
}

func TestProxyCloseLastOpenDocumentRetiresSnapshot(t *testing.T) {
	const goSrc = "package g\n\nfunc G() int { return 1 }\n"
	unsaved := "// unsaved\n" + aLgo
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo, "g/g.go": goSrc}, fixtureOptions{
		Script: func(string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/hover": toJSON(t, map[string]any{"contents": map[string]any{"kind": "plaintext", "value": "unsaved"}}),
			}}
		},
	})
	aPath, aURI := f.path("a/a.lgo"), f.uri("a/a.lgo")
	f.openText("a/a.lgo", 1, unsaved)
	operand := shift(findPos(t, unsaved, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	compiled := len(f.comp.bufferCalls())
	f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": aURI}})

	res := f.client.call("textDocument/hover", positionParams(aURI, operand))
	if string(res) != "null" {
		t.Fatalf("hover on the closed document = %s, want null", res)
	}
	f.sync()
	for _, b := range f.comp.bufferCalls()[compiled:] {
		if string(b[aPath]) == unsaved {
			t.Error("a lowering after the close used the discarded unsaved text")
		}
	}
	virt := f.virtURI("a/a.lgo")
	var opens, closes int
	for _, p := range f.receivedMethod("textDocument/didOpen") {
		if td, _ := p["textDocument"].(map[string]any); td["uri"] == virt {
			opens++
		}
	}
	for _, p := range f.receivedMethod("textDocument/didClose") {
		if td, _ := p["textDocument"].(map[string]any); td["uri"] == virt {
			closes++
		}
	}
	if opens != closes {
		t.Errorf("gopls saw %d didOpen and %d didClose for the lowered document, want every open matched by a close", opens, closes)
	}
	var counts snapshotCountsResult
	if err := json.Unmarshal(f.client.call("logos/snapshotCounts", map[string]any{}), &counts); err != nil {
		t.Fatal(err)
	}
	if _, ok := counts.Dirs[filepath.Dir(aPath)]; ok {
		t.Errorf("snapshotCounts reports %s after the close: %v", filepath.Dir(aPath), counts.Dirs)
	}
	if calls := f.comp.lowerCalls(); len(calls) != 1 {
		t.Errorf("the proxy lowered %d times, want only the pre-close hover: %v", len(calls), calls)
	}
}

func TestProxyWatchedFileAfterCloseDoesNotRecompile(t *testing.T) {
	const goSrc = "package g\n\nfunc G() int { return 1 }\n"
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo, "g/g.go": goSrc}, fixtureOptions{})
	aDir, aURI := f.path("a"), f.uri("a/a.lgo")
	f.open("a/a.lgo", 1)
	edited := aLgo + "// v2\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": edited}},
	})
	operand := shift(findPos(t, edited, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	f.client.notify("textDocument/didClose", map[string]any{"textDocument": map[string]any{"uri": aURI}})
	f.sync()
	if calls := f.comp.lowerCalls(); !slices.Equal(calls, []string{aDir}) {
		t.Fatalf("lowerings after the close = %v, want only the pre-close hover of %s", calls, aDir)
	}
	f.client.notify("workspace/didChangeWatchedFiles", map[string]any{
		"changes": []any{map[string]any{"uri": aURI}},
	})
	f.client.call("textDocument/hover", positionParams(f.uri("g/g.go"), Position{}))
	lowered := 0
	for _, d := range f.comp.lowerCalls() {
		if d == aDir {
			lowered++
		}
	}
	if lowered != 1 {
		t.Errorf("the compiler lowered %s %d times, want only the pre-close hover", aDir, lowered)
	}
}

func TestServeDuplicateContentLengthFails(t *testing.T) {
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
		t.Fatal(err)
	}
	opts := Options{
		Gopls:    []string{os.Args[0], "-mode=stdio"},
		Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
		Compiler: newFakeCompiler(),
		Stderr:   new(syncBuffer),
	}
	inR, inW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	ctx := t.Context()
	go func() { done <- Serve(ctx, inR, outW, opts) }()
	body := `{"jsonrpc":"2.0","method":"initialized","params":{}}`
	bye := `{"jsonrpc":"2.0","method":"exit","params":{}}`
	if _, err := fmt.Fprintf(inW, "Content-Length: %d\r\nContent-Length: %d\r\n\r\n%s", len(body), len(body), body); err != nil {
		t.Fatal(err)
	}
	if _, err := fmt.Fprintf(inW, "Content-Length: %d\r\n\r\n%s", len(bye), bye); err != nil {
		t.Fatal(err)
	}
	if err := inW.Close(); err != nil {
		t.Errorf("closing stdin writer: %v", err)
	}
	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "duplicate Content-Length") {
			t.Fatalf("Serve over a duplicate Content-Length frame = %v, want a duplicate Content-Length error", err)
		}
	case <-time.After(waitLimit):
		t.Fatal("Serve did not return after the duplicate Content-Length frame")
	}
	if err := outW.Close(); err != nil {
		t.Errorf("closing stdout writer: %v", err)
	}
	if err := outR.Close(); err != nil {
		t.Errorf("closing stdout reader: %v", err)
	}
}

func TestProxyPackageSnapshotHoldsEveryVersion(t *testing.T) {
	const bLgo = "package a\n\nfunc B() int { return 2 }\n"
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo, "a/b.lgo": bLgo}, fixtureOptions{})
	aURI, aPath := f.uri("a/a.lgo"), f.path("a/a.lgo")
	bURI, bPath := f.uri("a/b.lgo"), f.path("a/b.lgo")
	f.open("a/a.lgo", 1)
	f.open("a/b.lgo", 1)
	editedA2 := aLgo + "// v2\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": editedA2}},
	})
	operand := shift(findPos(t, editedA2, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	if got := len(f.comp.lowerCalls()); got != 1 {
		t.Fatalf("one edit and one hover caused %d lowerings, want the single run for the version set", got)
	}
	if last := f.comp.bufferCalls()[len(f.comp.bufferCalls())-1]; string(last[aPath]) != editedA2 || string(last[bPath]) != bLgo {
		t.Errorf("the run lowered a=%q b=%q, want the version 2 text of a and the version 1 text of b", last[aPath], last[bPath])
	}
	var fwdText string
	for _, o := range f.receivedMethod("textDocument/didOpen") {
		td, _ := o["textDocument"].(map[string]any)
		if td["uri"] == f.virtURI("a/a.lgo") {
			fwdText, _ = td["text"].(string)
		}
	}
	if fwdText == "" {
		t.Fatal("gopls got no lowered didOpen for a/a.lgo")
	}
	_, pos := forwardedPosition(t, f, "textDocument/hover", 0)
	lines := strings.Split(fwdText, "\n")
	if pos.Line < 0 || pos.Line >= len(lines) || pos.Character < 0 || pos.Character > len(lines[pos.Line]) {
		t.Fatalf("hover forwarded at %v, outside the lowered text", pos)
	} else if got := lines[pos.Line][pos.Character:]; !strings.HasPrefix(got, "path)") {
		t.Errorf("hover forwarded at %v reaching %q, want the path operand", pos, got)
	}
	f.client.call("textDocument/hover", positionParams(aURI, operand))
	if got := len(f.comp.lowerCalls()); got != 1 {
		t.Fatalf("a repeat hover with no edits caused %d lowerings, want still the one run", got)
	}
	editedB2 := bLgo + "// v2\n"
	editedA3 := editedA2 + "// v3\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": bURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": editedB2}},
	})
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": aURI, "version": 3},
		"contentChanges": []any{map[string]any{"text": editedA3}},
	})
	f.client.call("textDocument/hover", positionParams(bURI, Position{Line: 2}))
	if got := len(f.comp.lowerCalls()); got != 2 {
		t.Fatalf("edits to both documents caused %d lowerings, want only the second run", got)
	}
	if last := f.comp.bufferCalls()[len(f.comp.bufferCalls())-1]; string(last[aPath]) != editedA3 || string(last[bPath]) != editedB2 {
		t.Errorf("the rerun lowered a=%q b=%q, want the version 3 text of a and the version 2 text of b", last[aPath], last[bPath])
	}
}

func TestProxyLargeFileMapsExactly(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("package big\n\n")
	for i := range 300 {
		fmt.Fprintf(&sb, "var x%d = %d\n", i, i)
	}
	src := sb.String()
	if len(src) <= 2048 {
		t.Fatalf("fixture is %d bytes, want more than 2048 so a size cap would bite", len(src))
	}
	f := newFixture(t, map[string]string{"b/b.lgo": src}, fixtureOptions{})
	f.open("b/b.lgo", 1)
	at := Position{Line: 250, Character: 4}
	f.client.call("textDocument/hover", positionParams(f.uri("b/b.lgo"), at))
	if uri, pos := forwardedPosition(t, f, "textDocument/hover", 0); uri != f.virtURI("b/b.lgo") || pos != at {
		t.Errorf("hover at %v reached gopls as %s %v, want the exact position with no size cap", at, uri, pos)
	}
}

func TestProxyDuplicateTokenMapsToOwnSegment(t *testing.T) {
	const dupLgo = "package a\n\nimport \"os\"\n\nfunc first(path string) string {\n\tdata := os.ReadFile(path)?\n\treturn string(data)\n}\n\nfunc second(path string) string {\n\tdata := os.ReadFile(path)?\n\treturn string(data)\n}\n"
	low, _, ok := lowerQuestion([]byte(dupLgo))
	if !ok {
		t.Fatal("duplicate fixture does not lower")
	}
	lowText := string(low)
	f := newFixture(t, map[string]string{"a/a.lgo": dupLgo}, fixtureOptions{})
	f.open("a/a.lgo", 1)
	f.sync()
	aURI := f.uri("a/a.lgo")
	origFirst := shift(findPos(t, dupLgo, "ReadFile(path)?", 0), len("ReadFile("))
	origSecond := shift(findPos(t, dupLgo, "ReadFile(path)?", 1), len("ReadFile("))
	lowFirst := shift(findPos(t, lowText, "ReadFile(path)", 0), len("ReadFile("))
	lowSecond := shift(findPos(t, lowText, "ReadFile(path)", 1), len("ReadFile("))
	if lowFirst == lowSecond {
		t.Fatal("fixture copies lower to the same position; the test cannot discriminate them")
	}
	f.client.call("textDocument/hover", positionParams(aURI, origFirst))
	f.client.call("textDocument/hover", positionParams(aURI, origSecond))
	opened := f.receivedMethod("textDocument/didOpen")
	if len(opened) != 1 {
		t.Fatalf("gopls got %d lowered didOpen messages, want one", len(opened))
	}
	uri0, pos0 := forwardedPosition(t, f, "textDocument/hover", 0)
	uri1, pos1 := forwardedPosition(t, f, "textDocument/hover", 1)
	if uri0 != f.virtURI("a/a.lgo") || pos0 != lowFirst {
		t.Errorf("hover on the first copy reached gopls as %s %v, want %s %v", uri0, pos0, f.virtURI("a/a.lgo"), lowFirst)
	}
	if uri1 != f.virtURI("a/a.lgo") || pos1 != lowSecond {
		t.Errorf("hover on the second copy reached gopls as %s %v, want its own segment %s %v", uri1, pos1, f.virtURI("a/a.lgo"), lowSecond)
	}
}

func TestProxyDefinitionWritesNoBytes(t *testing.T) {
	const libLgo = "package lib\n\nimport \"os\"\n\nfunc load(path string) string {\n\tdata := os.ReadFile(path)?\n\treturn string(data)\n}\n\nfunc Hello() string {\n\treturn \"hi\"\n}\n"
	const appGo = "package app\n\nimport \"example.com/m/lib\"\n\nvar Greeting = lib.Hello()\n"
	const app2Lgo = "package app2\n\nimport \"example.com/m/golib\"\n\nvar N = golib.Thing()\n"
	const goLib = "package golib\n\nfunc Thing() int { return 1 }\n"
	thing := Range{Start: Position{Line: 2, Character: 5}, End: Position{Line: 2, Character: 10}}
	hello := span(findPos(t, libLgo, "Hello", 0), len("Hello"))
	libLow, _, ok := lowerQuestion([]byte(libLgo))
	if !ok {
		t.Fatal("lib fixture does not lower")
	}
	helloLow := span(findPos(t, string(libLow), "Hello", 0), len("Hello"))
	f := newFixture(t, map[string]string{
		"go.mod":        "module example.com/m\n\ngo 1.27\n",
		"lib/lib.lgo":   libLgo,
		"app/app.go":    appGo,
		"golib/g.go":    goLib,
		"app2/app2.lgo": app2Lgo,
	}, fixtureOptions{
		Script: func(root string) fakeScript {
			return fakeScript{Responses: map[string]json.RawMessage{
				"textDocument/definition": toJSON(t, []any{
					map[string]any{"uri": pathURI(filepath.Join(root, "lib", "lib.go")), "range": helloLow},
					map[string]any{"uri": pathURI(filepath.Join(root, "golib", "g.go")), "range": thing},
				}),
			}}
		},
		Tune: func(f *fixture, o *Options) {
			f.comp.setClosure(f.path("app"), f.path("lib"))
			f.comp.setClosure(f.path("app2"), f.path("golib"))
		},
	})
	f.open("app/app.go", 1)
	f.open("app2/app2.lgo", 1)
	f.sync()
	var libFwd string
	for _, o := range f.receivedMethod("textDocument/didOpen") {
		td, _ := o["textDocument"].(map[string]any)
		if td["uri"] == f.virtURI("lib/lib.lgo") {
			libFwd, _ = td["text"].(string)
		}
	}
	if libFwd == "" {
		t.Fatal("gopls got no lowered didOpen for lib/lib.lgo")
	}
	if !strings.Contains(libFwd, "os.ReadFile(path)") || strings.Contains(libFwd, "ReadFile(path)?") {
		t.Errorf("gopls got the lowered dependency text %q, want the lowered call without the ? marker", libFwd)
	}
	snap := func() (map[string][]byte, map[string]int64) {
		data := map[string][]byte{}
		mtime := map[string]int64{}
		for _, rel := range f.workspaceFiles() {
			path := filepath.Join(f.root, filepath.FromSlash(rel))
			content, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			st, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			data[rel] = content
			mtime[rel] = st.ModTime().UnixNano()
		}
		return data, mtime
	}
	before, beforeTime := snap()
	appCall := shift(findPos(t, appGo, "lib.Hello", 0), len("lib."))
	resLib := f.client.call("textDocument/definition", positionParams(f.uri("app/app.go"), appCall))
	sameJSON(t, "definition of a lowered dependency symbol", resLib, []any{
		map[string]any{"uri": f.uri("lib/lib.lgo"), "range": hello},
		map[string]any{"uri": f.uri("golib/g.go"), "range": thing},
	})
	thingCall := shift(findPos(t, app2Lgo, "golib.Thing", 0), len("golib."))
	resGo := f.client.call("textDocument/definition", positionParams(f.uri("app2/app2.lgo"), thingCall))
	sameJSON(t, "definition of a plain Go symbol", resGo, []any{
		map[string]any{"uri": f.uri("lib/lib.lgo"), "range": hello},
		map[string]any{"uri": f.uri("golib/g.go"), "range": thing},
	})
	after, afterTime := snap()
	if !reflect.DeepEqual(after, before) {
		t.Errorf("definition changed workspace bytes")
	}
	if !reflect.DeepEqual(afterTime, beforeTime) {
		t.Errorf("definition touched workspace mtimes")
	}
}
