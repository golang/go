// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestEqualIDsBothDirectionsStayRouted(t *testing.T) {
	var toClient, toServer bytes.Buffer
	p := newProxy(&toClient, Options{})
	p.server = &writer{w: &toServer}
	p.initialize(&message{ID: json.RawMessage("1"), Method: "initialize", Params: json.RawMessage(`{}`)})
	if _, ok := p.pending["1"]; !ok {
		t.Fatal("initialize did not track the client request id")
	}
	p.fromServer(t.Context(), &message{ID: json.RawMessage("1"), Method: "workspace/configuration", Params: json.RawMessage(`{}`)})
	p.fromClient(t.Context(), &message{ID: json.RawMessage("1"), Result: json.RawMessage("null")})
	p.flush(t)
	if _, ok := p.pending["1"]; !ok {
		t.Fatal("the client answer to the server request cleared the client request slot")
	}
	if !bytes.Contains(toServer.Bytes(), []byte(`"id":1`)) && !bytes.Contains(toServer.Bytes(), []byte(`"id": 1`)) {
		t.Fatalf("the client answer never reached the server: %q", toServer.Bytes())
	}
	p.serverResponse(t.Context(), &message{ID: json.RawMessage("1"), Result: json.RawMessage("null")})
	p.flush(t)
	if _, ok := p.pending["1"]; ok {
		t.Fatal("the server answer to the client request left a pending slot")
	}
	if !bytes.Contains(toClient.Bytes(), []byte(`"id":1`)) && !bytes.Contains(toClient.Bytes(), []byte(`"id": 1`)) {
		t.Fatalf("the server answer never reached the client: %q", toClient.Bytes())
	}
}

func TestStringAndNumberIDsStayDistinct(t *testing.T) {
	var toClient bytes.Buffer
	p := newProxy(&toClient, Options{})
	p.server = &writer{w: &toClient}
	p.addPending(json.RawMessage("1"), pending{method: "textDocument/hover"})
	p.addPending(json.RawMessage(`"1"`), pending{method: "textDocument/hover"})
	if len(p.pending) != 2 {
		t.Fatalf("string and number ids share one slot: %v", p.pending)
	}
}
