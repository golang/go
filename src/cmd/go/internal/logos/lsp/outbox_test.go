// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestUnreadingPeerFailsTheSessionBounded(t *testing.T) {
	inR, inW := io.Pipe()
	defer func() {
		_ = inR.Close()
		_ = inW.Close()
	}()
	p := newProxy(inW, Options{Stderr: new(syncBuffer)})
	pad := `"` + strings.Repeat("x", 1024) + `"`
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 200 {
			p.send(p.client, &message{Method: "m", Params: []byte(pad)})
		}
	}()
	select {
	case <-done:
	case <-time.After(waitLimit):
		t.Fatal("sends to an unreading peer never returned")
	}
	if p.sendError() == nil {
		t.Fatal("an unreading peer left the session open with no error")
	}
}

func TestServeFailsClosedWhenClientStopsReading(t *testing.T) {
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
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	defer func() {
		_ = inR.Close()
		_ = inW.Close()
		_ = outR.Close()
		_ = outW.Close()
	}()
	done := make(chan error, 1)
	go func() { done <- Serve(t.Context(), inR, outW, opts) }()
	var frames strings.Builder
	for range 200 {
		body := `{"m":}`
		fmt.Fprintf(&frames, "Content-Length: %d\r\n\r\n%s", len(body), body)
	}
	go func() {
		_, _ = io.WriteString(inW, frames.String())
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Serve kept a session with an unreading client open")
		}
	case <-time.After(waitLimit):
		t.Fatal("Serve never failed a session whose client stopped reading")
	}
}

func (p *Proxy) flush(t *testing.T) {
	t.Helper()
	for {
		select {
		case m := <-p.toClient:
			if err := p.client.write(m); err != nil {
				t.Fatal(err)
			}
		case m := <-p.toServer:
			if err := p.server.write(m); err != nil {
				t.Fatal(err)
			}
		default:
			return
		}
	}
}
