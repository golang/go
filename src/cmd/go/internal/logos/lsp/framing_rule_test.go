// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"io"
	"strconv"
	"testing"
)

func TestDuplicateLengthFailsTheSessionClosed(t *testing.T) {
	in := "Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}"
	if err := serveInputError(t, in); err == nil {
		t.Fatal("Serve accepted a duplicate Content-Length, want a fail-closed error")
	}
}

func TestServerLegParseErrorContinues(t *testing.T) {
	var toClient bytes.Buffer
	p := newProxy(&toClient, Options{Stderr: new(syncBuffer)})
	inR, inW := io.Pipe()
	done := make(chan error, 1)
	go func() { done <- p.serverLoop(t.Context(), newReader(inR)) }()
	frame := func(body string) string {
		return "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n" + body
	}
	diag := `{"jsonrpc":"2.0","method":"window/showMessage",` +
		`"params":{"type":1,"message":"hi"}}`
	if _, err := io.WriteString(inW, frame(`{"m":}`)+frame(diag)); err != nil {
		t.Fatal(err)
	}
	_ = inW.Close()
	if err := <-done; err != nil {
		t.Fatalf("serverLoop stopped on a bad body: %v", err)
	}
	p.flush(t)
	if !bytes.Contains(toClient.Bytes(), []byte("showMessage")) {
		t.Fatalf("the notification past the bad body never reached the client: %q", toClient.Bytes())
	}
}
