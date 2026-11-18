// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	jsonv2 "encoding/json/v2"
)

const stockLgo = "package a\n\nimport \"os\"\n\nfunc Read(path string) (string, error) {\n\tdata := os.ReadFile(path)?\n\treturn string(data), nil\n}\n"

const stockDiagMessage = "stock type error: overloaded call"

type stockClient struct {
	t    *testing.T
	w    *writer
	mu   sync.Mutex
	cond *sync.Cond
	msgs []*message
	next int
	done chan struct{}
}

func (c *stockClient) readLoop(rd *reader) {
	defer close(c.done)
	for {
		m, err := rd.read()
		c.mu.Lock()
		if err != nil {
			c.mu.Unlock()
			return
		}
		c.msgs = append(c.msgs, m)
		c.cond.Broadcast()
		c.mu.Unlock()
	}
}

func (c *stockClient) waitFor(what string, match func(*message) bool) *message {
	c.t.Helper()
	expired := false
	timer := time.AfterFunc(replyWait(c.t), func() {
		c.mu.Lock()
		expired = true
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
		if expired {
			c.t.Fatalf("no %s from the proxy", what)
		}
		c.cond.Wait()
	}
}

func (c *stockClient) call(method string, params any) []byte {
	c.t.Helper()
	m, ok := c.awaitCall(c.sendCall(method, params))
	if !ok {
		c.t.Fatalf("no response to %s from the proxy", method)
	}
	if len(m.Error) > 0 {
		c.t.Fatalf("%s failed: %s", method, m.Error)
	}
	return []byte(m.Result)
}

func (c *stockClient) sendCall(method string, params any) int {
	c.t.Helper()
	c.mu.Lock()
	c.next++
	id := c.next
	c.mu.Unlock()
	raw, err := jsonv2.Marshal(params)
	if err != nil {
		c.t.Fatal(err)
	}
	if err := c.w.write(&message{ID: json.RawMessage(fmt.Sprint(id)), Method: method, Params: raw}); err != nil {
		c.t.Fatalf("sending %s: %v", method, err)
	}
	return id
}

func (c *stockClient) awaitCall(id int) (*message, bool) {
	c.t.Helper()
	expired := false
	timer := time.AfterFunc(replyWait(c.t), func() {
		c.mu.Lock()
		expired = true
		c.cond.Broadcast()
		c.mu.Unlock()
	})
	defer timer.Stop()
	c.mu.Lock()
	defer c.mu.Unlock()
	for {
		for _, m := range c.msgs {
			if m.isResponse() && string(m.ID) == fmt.Sprint(id) {
				return m, true
			}
		}
		if expired {
			return nil, false
		}
		c.cond.Wait()
	}
}

func (c *stockClient) notify(method string, params any) {
	c.t.Helper()
	if err := c.w.notify(method, params); err != nil {
		c.t.Fatalf("sending %s: %v", method, err)
	}
}

func (c *stockClient) notifyFor(what string, accept func(map[string]any) bool) map[string]any {
	c.t.Helper()
	var found map[string]any
	c.waitFor(what, func(m *message) bool {
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

func stockPos(t *testing.T, text string, off int) Position {
	t.Helper()
	if off < 0 || off > len(text) {
		t.Fatalf("offset %d outside %d bytes", off, len(text))
	}
	return Position{Line: strings.Count(text[:off], "\n"), Character: off - strings.LastIndex(text[:off], "\n") - 1}
}

func stockSpan(t *testing.T, text, needle string, nth int) Range {
	t.Helper()
	off := -1
	for i := 0; i <= nth; i++ {
		j := strings.Index(text[off+1:], needle)
		if j < 0 {
			t.Fatalf("needle %q occurrence %d missing", needle, i)
		}
		off += 1 + j
	}
	return Range{Start: stockPos(t, text, off), End: stockPos(t, text, off+len(needle))}
}

func stockRange(t *testing.T, what string, v any) Range {
	t.Helper()
	m, _ := v.(map[string]any)
	r, ok := asRange(m)
	if !ok {
		t.Fatalf("%s carries no range: %v", what, v)
	}
	return r
}

func stockDiagRange(t *testing.T, params map[string]any) Range {
	t.Helper()
	items, _ := params["diagnostics"].([]any)
	for _, item := range items {
		d, _ := item.(map[string]any)
		if d["message"] == stockDiagMessage {
			return stockRange(t, "diagnostic", d["range"])
		}
	}
	t.Fatalf("diagnostic %q missing from %v", stockDiagMessage, params)
	return Range{}
}

func TestStockClientThroughRealProxy(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Setenv("GOFLAGS", "")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOPROXY", "off")
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "a"), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/stock\n\ngo 1.27\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	lgoPath := filepath.Join(root, "a", "a.lgo")
	if err := os.WriteFile(lgoPath, []byte(stockLgo), 0o666); err != nil {
		t.Fatal(err)
	}
	fakeDir := t.TempDir()
	writeFakeGopls(t, fakeDir)
	script, err := jsonv2.Marshal(fakeScript{Echo: true})
	if err != nil {
		t.Fatal(err)
	}
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	if err := os.WriteFile(scriptPath, script, 0o666); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(scratch, "stock.log")
	logosBin := filepath.Join(filepath.Dir(testenv.GoToolPath(t)), "logos")
	if runtime.GOOS == "windows" {
		logosBin += ".exe"
	}
	if _, err := os.Stat(logosBin); err != nil {
		t.Skipf("logos binary not present: %v", err)
	}
	cmd := exec.CommandContext(t.Context(), logosBin, "lsp")
	cmd.Env = append(os.Environ(), "PATH="+fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"), fakeEnv+"="+scriptPath, fakeLogEnv+"="+logPath)
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
	c := &stockClient{t: t, w: &writer{w: stdin}, done: make(chan struct{})}
	c.cond = sync.NewCond(&c.mu)
	go c.readLoop(newReader(stdout))
	t.Cleanup(func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		if t.Failed() {
			t.Logf("proxy stderr:\n%s", stderr.String())
		}
	})
	res := c.call("initialize", map[string]any{
		"processId": os.Getpid(),
		"rootUri":   pathURI(root),
		"capabilities": map[string]any{
			"textDocument": map[string]any{
				"hover":           map[string]any{"contentFormat": []string{"plaintext"}},
				"synchronization": map[string]any{"didSave": true},
			},
		},
	})
	var caps struct {
		Capabilities map[string]any `json:"capabilities"`
	}
	if err := jsonv2.Unmarshal(res, &caps); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"hoverProvider", "definitionProvider", "completionProvider"} {
		if _, ok := caps.Capabilities[name]; !ok {
			t.Errorf("capability %s missing from the proxy answer: %s", name, res)
		}
	}
	c.notify("initialized", map[string]any{})
	lgoURI := pathURI(lgoPath)
	c.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": lgoURI, "languageId": "logos", "version": 1, "text": stockLgo},
	})
	operand := stockSpan(t, stockLgo, "path", 1)
	hover, err := decode(c.call("textDocument/hover", positionParams(lgoURI, operand.Start)))
	if err != nil {
		t.Fatal(err)
	}
	hm, _ := hover.(map[string]any)
	contents, _ := hm["contents"].(map[string]any)
	value, _ := contents["value"].(string)
	if !strings.Contains(value, "stock hover") {
		t.Errorf("hover contents = %q, want it to hold stock hover", value)
	}
	if r := stockRange(t, "hover", hm["range"]); r != operand {
		t.Errorf("hover range = %v, want %v", r, operand)
	}
	def, err := decode(c.call("textDocument/definition", positionParams(lgoURI, operand.Start)))
	if err != nil {
		t.Fatal(err)
	}
	locs, _ := def.([]any)
	if len(locs) != 1 {
		t.Fatalf("definition = %v, want one location", def)
	}
	loc, _ := locs[0].(map[string]any)
	if loc["uri"] != lgoURI {
		t.Errorf("definition uri = %v, want %s", loc["uri"], lgoURI)
	}
	if r := stockRange(t, "definition", loc["range"]); r != operand {
		t.Errorf("definition range = %v, want %v", r, operand)
	}
	comp, err := decode(c.call("textDocument/completion", positionParams(lgoURI, operand.Start)))
	if err != nil {
		t.Fatal(err)
	}
	cm, _ := comp.(map[string]any)
	items, _ := cm["items"].([]any)
	if len(items) != 1 {
		t.Fatalf("completion = %v, want one item", comp)
	}
	item, _ := items[0].(map[string]any)
	if item["label"] != "ReadFile" {
		t.Errorf("completion label = %v, want ReadFile", item["label"])
	}
	edit, _ := item["textEdit"].(map[string]any)
	if r := stockRange(t, "completion edit", edit["range"]); r != operand {
		t.Errorf("completion edit range = %v, want %v", r, operand)
	}
	got := c.notifyFor("diagnostics for the opener", func(m map[string]any) bool {
		if m["uri"] != lgoURI {
			return false
		}
		items, _ := m["diagnostics"].([]any)
		for _, entry := range items {
			d, _ := entry.(map[string]any)
			if d["message"] == stockDiagMessage {
				return true
			}
		}
		return false
	})
	if v, _ := number(got["version"]); v != 1 {
		t.Errorf("diagnostics version = %v, want 1", got["version"])
	}
	if r := stockDiagRange(t, got); r != stockSpan(t, stockLgo, "ReadFile", 0) {
		t.Errorf("diagnostic range = %v, want the ReadFile call", r)
	}
	changed := "// stock edit\n" + stockLgo
	c.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": lgoURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": changed}},
	})
	operand2 := stockSpan(t, changed, "path", 1)
	hover2, err := decode(c.call("textDocument/hover", positionParams(lgoURI, operand2.Start)))
	if err != nil {
		t.Fatal(err)
	}
	hm2, _ := hover2.(map[string]any)
	if r := stockRange(t, "hover after the edit", hm2["range"]); r != operand2 {
		t.Errorf("hover range after the edit = %v, want %v", r, operand2)
	}
	got2 := c.notifyFor("diagnostics for the edit", func(m map[string]any) bool {
		if m["uri"] != lgoURI {
			return false
		}
		if v, _ := number(m["version"]); v != 2 {
			return false
		}
		items, _ := m["diagnostics"].([]any)
		for _, entry := range items {
			d, _ := entry.(map[string]any)
			if d["message"] == stockDiagMessage {
				return true
			}
		}
		return false
	})
	if r := stockDiagRange(t, got2); r != stockSpan(t, changed, "ReadFile", 0) {
		t.Errorf("diagnostic range after the edit = %v, want the moved ReadFile call", r)
	}
	if res := c.call("shutdown", nil); string(res) != "null" {
		t.Errorf("shutdown = %s, want null", res)
	}
	c.notify("exit", nil)
	if err := stdin.Close(); err != nil {
		t.Errorf("closing the proxy input: %v", err)
	}
	select {
	case <-c.done:
	case <-time.After(replyWait(t)):
		t.Fatal("the proxy did not close the stock exchange")
	}
	if err := cmd.Wait(); err != nil {
		t.Errorf("proxy exit = %v, want nil", err)
	}
	logData, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(logData), "textDocument/hover") {
		t.Errorf("the downstream server saw no hover")
	}
}
