// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// TestBridge exercises one full session: a client connects and sends input,
// a simulated page picks up the session, reads standard input, posts output
// and an exit code, and the client observes them.
func TestBridge(t *testing.T) {
	srv := &server{
		token:    "tok",
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}
	ts := httptest.NewServer(srv.mux())
	defer ts.Close()

	clientConn, serverConn := net.Pipe()
	defer clientConn.Close()
	go srv.handleClient(serverConn)

	h, err := json.Marshal(hello{
		Argv: []string{"fips140test.test"},
		Env:  map[string]string{"ACVP_WRAPPER": "1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := writeFrame(clientConn, 'H', h); err != nil {
		t.Fatal(err)
	}
	go func() {
		writeFrame(clientConn, 'I', []byte("ping"))
		writeFrame(clientConn, 'E', nil)
	}()

	// Like a real client, always keep reading server frames.
	type frame struct {
		typ     byte
		payload []byte
	}
	frames := make(chan frame, 16)
	go func() {
		defer close(frames)
		for {
			typ, payload, err := readFrame(clientConn)
			if err != nil {
				return
			}
			frames <- frame{typ, payload}
		}
	}()

	// The page picks up the session.
	resp, err := http.Get(ts.URL + "/tok/session")
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		ID   int               `json:"id"`
		Argv []string          `json:"argv"`
		Env  map[string]string `json:"env"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := meta.Argv[0]; got != "fips140test.test" {
		t.Errorf("argv[0] = %q", got)
	}
	if got := meta.Env["ACVP_WRAPPER"]; got != "1" {
		t.Errorf(`env["ACVP_WRAPPER"] = %q`, got)
	}

	// The page polls standard input a chunk at a time. The first poll returns
	// the client's input; the next, after EOF, sets the Stdin-Eof header and
	// returns an empty body.
	resp, err = http.Get(ts.URL + "/tok/stdin?s=1")
	if err != nil {
		t.Fatal(err)
	}
	stdin, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if !bytes.Equal(stdin, []byte("ping")) {
		t.Errorf("stdin = %q, want %q", stdin, "ping")
	}

	resp, err = http.Get(ts.URL + "/tok/stdin?s=1")
	if err != nil {
		t.Fatal(err)
	}
	eof, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.Header.Get("Stdin-Eof") != "1" || len(eof) != 0 {
		t.Errorf("second stdin poll: Stdin-Eof=%q body=%q, want \"1\" and empty",
			resp.Header.Get("Stdin-Eof"), eof)
	}

	// The page posts output and the exit code...
	resp, err = http.Post(ts.URL+"/tok/io?s=1&fd=1", "application/octet-stream",
		strings.NewReader("pong"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("io: HTTP %d", resp.StatusCode)
	}
	resp, err = http.Post(ts.URL+"/tok/exit?s=1", "application/json",
		strings.NewReader(`{"code":3}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("exit: HTTP %d", resp.StatusCode)
	}

	// ...and the client sees them.
	clientConn.SetReadDeadline(time.Now().Add(5 * time.Second))
	f, ok := <-frames
	if !ok || f.typ != 'O' || !bytes.Equal(f.payload, []byte("pong")) {
		t.Errorf("frame = %c %q, want O pong", f.typ, f.payload)
	}
	f, ok = <-frames
	if !ok || f.typ != 'X' || len(f.payload) != 1 || f.payload[0] != 3 {
		t.Errorf("frame = %c %v, want X [3]", f.typ, f.payload)
	}
	if f, ok := <-frames; ok {
		t.Errorf("unexpected frame after exit: %c %q", f.typ, f.payload)
	}

	// The session is gone.
	resp, err = http.Get(ts.URL + "/tok/stdin?s=1")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("after exit: HTTP %d, want 404", resp.StatusCode)
	}
}

// TestStaleSessionSkipped checks that a session whose client disconnected
// while queued is not handed to the page.
func TestStaleSessionSkipped(t *testing.T) {
	srv := &server{
		token:    "tok",
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}

	// Session 1 was removed (client gone) but is still in the queue.
	srv.queue <- &session{id: 1, argv: []string{"one"}}
	live := &session{id: 2, argv: []string{"two"}, env: map[string]string{}}
	srv.sessions[2] = live
	srv.queue <- live

	ts := httptest.NewServer(srv.mux())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/tok/session")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var meta struct {
		ID int `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		t.Fatal(err)
	}
	if meta.ID != 2 {
		t.Errorf("delivered session %d, want 2", meta.ID)
	}
}

// TestSessionRequeue checks that a session dequeued by a poll whose page
// already went away is requeued rather than dropped.
func TestSessionRequeue(t *testing.T) {
	srv := &server{
		token:    "tok",
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}
	s := &session{id: 1, argv: []string{"one"}, env: map[string]string{}}
	srv.sessions[1] = s

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	for range 20 {
		srv.queue <- s
		r := httptest.NewRequest("GET", "/tok/session", nil).WithContext(ctx)
		srv.handleSession(httptest.NewRecorder(), r)
		select {
		case got := <-srv.queue:
			if got != s {
				t.Fatalf("queued session = %+v, want session 1", got)
			}
		default:
			t.Fatal("session was delivered to a dead page and dropped")
		}
	}
}

func TestCorruptModule(t *testing.T) {
	magic := append([]byte{0xff}, " Go fipsinfo \xff\x00"...)
	prefix := []byte("some wasm bytes ")
	sum := bytes.Repeat([]byte{0xab}, 32)
	suffix := []byte(" more wasm bytes")
	bin := slices.Concat(prefix, magic, sum, suffix)

	got, err := corruptModule(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, sum) {
		t.Errorf("returned sum = %x, want %x", got, sum)
	}
	want := slices.Concat(prefix, magic, bytes.Repeat([]byte("X"), 32), suffix)
	if !bytes.Equal(bin, want) {
		t.Errorf("corrupted module = %q, want %q", bin, want)
	}

	if _, err := corruptModule([]byte("no magic here")); err == nil {
		t.Error("expected error for missing magic")
	}
	if _, err := corruptModule(slices.Concat(magic, sum, magic, sum)); err == nil {
		t.Error("expected error for duplicate magic")
	}
	if _, err := corruptModule(slices.Concat(magic, []byte("short"))); err == nil {
		t.Error("expected error for truncated module")
	}
}

// TestCorruptSession checks that a corrupt session is advertised as such and
// served a module with its checksum overwritten, while a normal session gets
// the unmodified module.
func TestCorruptSession(t *testing.T) {
	magic := append([]byte{0xff}, " Go fipsinfo \xff\x00"...)
	sum := bytes.Repeat([]byte{0xab}, 32)
	module := slices.Concat([]byte("\x00asm"), magic, sum, []byte("tail"))
	path := filepath.Join(t.TempDir(), "module.wasm")
	if err := os.WriteFile(path, module, 0o644); err != nil {
		t.Fatal(err)
	}

	srv := &server{
		token:    "tok",
		wasm:     path,
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}
	srv.sessions[1] = &session{id: 1, env: map[string]string{}}
	srv.sessions[2] = &session{id: 2, env: map[string]string{}, corrupt: true}

	ts := httptest.NewServer(srv.mux())
	defer ts.Close()

	get := func(path string) []byte {
		t.Helper()
		resp, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		return body
	}

	// The page's initial load (no session) gets the real module.
	if got := get("/tok/module.wasm"); !bytes.Equal(got, module) {
		t.Errorf("plain module.wasm was modified")
	}
	// A normal session also gets the real module.
	if got := get("/tok/module.wasm?s=1"); !bytes.Equal(got, module) {
		t.Errorf("normal session served a modified module")
	}
	// A corrupt session gets the module with the checksum overwritten.
	want := slices.Concat([]byte("\x00asm"), magic, bytes.Repeat([]byte("X"), 32), []byte("tail"))
	if got := get("/tok/module.wasm?s=2"); !bytes.Equal(got, want) {
		t.Errorf("corrupt session module = %q, want %q", got, want)
	}

	// The corrupt session advertises itself so the page reloads the module.
	srv.queue <- srv.sessions[2]
	var meta struct {
		ID      int  `json:"id"`
		Corrupt bool `json:"corrupt"`
	}
	if err := json.Unmarshal(get("/tok/session"), &meta); err != nil {
		t.Fatal(err)
	}
	if meta.ID != 2 || !meta.Corrupt {
		t.Errorf("session meta = %+v, want {ID:2 Corrupt:true}", meta)
	}
}

// TestClientInfo checks that a posted client environment is accepted.
func TestClientInfo(t *testing.T) {
	srv := &server{
		token:    "tok",
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}
	ts := httptest.NewServer(srv.mux())
	defer ts.Close()

	body := `{"userAgent":"UA/1.0","inferredBrowser":"Test 1"}`
	resp, err := http.Post(ts.URL+"/tok/clientinfo", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("clientinfo: HTTP %d, want 200", resp.StatusCode)
	}
}

func TestCrossOrigin(t *testing.T) {
	srv := &server{
		token:    "tok",
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}
	ts := httptest.NewServer(srv.mux())
	defer ts.Close()

	req, err := http.NewRequest("POST", ts.URL+"/tok/io?s=1", strings.NewReader("x"))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Origin", "http://evil.example")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("cross-origin POST: HTTP %d, want 403", resp.StatusCode)
	}

	req.Header.Set("Origin", "http://"+req.Host)
	req.Body = io.NopCloser(strings.NewReader("x"))
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound { // no such session, but allowed
		t.Errorf("same-origin POST: HTTP %d, want 404", resp.StatusCode)
	}
}
