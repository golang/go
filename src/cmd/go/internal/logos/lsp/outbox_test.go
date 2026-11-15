// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
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

type burstWriter struct {
	seen  atomic.Int64
	total int64
	full  chan struct{}
}

func (b *burstWriter) Write(p []byte) (int, error) {
	if b.seen.Add(1) == b.total {
		close(b.full)
	}
	return len(p), nil
}

func TestSendBurstToLivePeerKeepsSessionOpen(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	total := int64(4 * outboxCap)
	bw := &burstWriter{total: total, full: make(chan struct{})}
	p := newProxy(bw, Options{Stderr: new(syncBuffer)})
	t.Cleanup(func() { close(p.done) })
	go p.pump(p.client, p.toClient)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range int(total) {
			p.send(p.client, &message{Method: "m"})
		}
	}()
	select {
	case <-done:
	case <-time.After(waitLimit):
		t.Fatal("send never returned while the pump made progress")
	}
	select {
	case <-bw.full:
	case <-time.After(waitLimit):
		t.Fatalf("delivered %d of %d frames", bw.seen.Load(), total)
	}
	if err := p.sendError(); err != nil {
		t.Fatalf("live peer failed the session: %v", err)
	}
}

type gatedWriter struct {
	mu      sync.Mutex
	written int
	gate    chan struct{}
	quit    chan struct{}
	total   int
	all     chan struct{}
}

func (g *gatedWriter) Write(b []byte) (int, error) {
	select {
	case <-g.gate:
	case <-g.quit:
		return 0, io.ErrClosedPipe
	}
	g.mu.Lock()
	g.written++
	n := g.written
	g.mu.Unlock()
	if n == g.total {
		close(g.all)
	}
	return len(b), nil
}

func (g *gatedWriter) count() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.written
}

func TestSlowPeerBurstRestartsStallTimer(t *testing.T) {
	total := 4 * outboxCap
	g := &gatedWriter{gate: make(chan struct{}, 1), quit: make(chan struct{}), total: total, all: make(chan struct{})}
	p := newProxy(g, Options{Stderr: new(syncBuffer)})
	t.Cleanup(func() {
		close(g.quit)
		close(p.done)
	})
	go p.pump(p.client, p.toClient)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range total {
			p.send(p.client, &message{Method: "m"})
		}
	}()
	g.gate <- struct{}{}
	for range 3 {
		select {
		case <-time.After(outboxStall / 2):
		case <-done:
			t.Fatal("send finished while the writer was gated")
		}
		g.gate <- struct{}{}
		if err := p.sendError(); err != nil {
			t.Fatalf("stall timer did not restart on delivered frames: %v", err)
		}
	}
	close(g.gate)
	select {
	case <-g.all:
	case <-time.After(waitLimit):
		t.Fatalf("delivered %d of %d frames", g.count(), total)
	}
	select {
	case <-done:
	case <-time.After(waitLimit):
		t.Fatal("send never returned")
	}
	if err := p.sendError(); err != nil {
		t.Fatalf("slow live peer failed the session: %v", err)
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
