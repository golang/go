// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runWithClientInfo(t *testing.T, root string, clientInfo any) (string, string) {
	t.Helper()
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	if err := os.WriteFile(scriptPath, []byte("{}"), 0o666); err != nil {
		t.Fatal(err)
	}
	comp := newFakeCompiler()
	opts := Options{
		Gopls:    []string{os.Args[0], "-mode=stdio"},
		Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
		Compiler: comp,
	}
	client := startProxy(t, opts)
	params := map[string]any{
		"processId":    nil,
		"rootUri":      pathURI(root),
		"capabilities": map[string]any{},
	}
	if clientInfo != nil {
		params["clientInfo"] = clientInfo
	}
	caps := string(client.call("initialize", params))
	client.notify("initialized", map[string]any{})
	uri := pathURI(filepath.Join(root, "a", "a.lgo"))
	client.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{"uri": uri, "languageId": "logos", "version": 1, "text": aLgo},
	})
	hover := string(client.call("textDocument/hover", positionParams(uri, Position{Line: 5, Character: 3})))
	return caps, hover
}

func TestProxyInitializeIgnoresClientInfo(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "a"), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "a", "a.lgo"), []byte(aLgo), 0o666); err != nil {
		t.Fatal(err)
	}
	capsPlain, hoverPlain := runWithClientInfo(t, root, nil)
	capsCode, hoverCode := runWithClientInfo(t, root, map[string]any{"name": "Visual Studio Code", "version": "1.90.0"})
	capsUnknown, hoverUnknown := runWithClientInfo(t, root, map[string]any{"name": "unknown-editor-xyz", "version": "0.0"})
	if capsPlain != capsCode || capsPlain != capsUnknown {
		t.Errorf("initialize capabilities differ by clientInfo")
	}
	if hoverPlain != hoverCode || hoverPlain != hoverUnknown {
		t.Errorf("hover answers differ by clientInfo")
	}
	if hoverPlain != "null" {
		t.Errorf("hover without a lowering = %s, want null", hoverPlain)
	}
	if got := strings.Count(capsPlain, "hover"); got == 0 {
		t.Errorf("initialize answer lacks hover capability: %s", capsPlain)
	}
}
