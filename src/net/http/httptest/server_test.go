// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httptest

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"sync"
	"testing"
)

type newServerFunc func(http.Handler) *Server

var newServers = map[string]newServerFunc{
	"NewServer":    NewServer,
	"NewTLSServer": NewTLSServer,

	// The manual variants of newServer create a Server manually by only filling
	// in the exported fields of Server.
	"NewServerManual": func(h http.Handler) *Server {
		ts := &Server{Listener: newLocalListener(), Config: &http.Server{Handler: h}}
		ts.Start()
		return ts
	},
	"NewTLSServerManual": func(h http.Handler) *Server {
		ts := &Server{Listener: newLocalListener(), Config: &http.Server{Handler: h}}
		ts.StartTLS()
		return ts
	},
}

func TestServer(t *testing.T) {
	for _, name := range []string{"NewServer", "NewServerManual"} {
		t.Run(name, func(t *testing.T) {
			newServer := newServers[name]
			t.Run("Server", func(t *testing.T) { testServer(t, newServer) })
			t.Run("GetAfterClose", func(t *testing.T) { testGetAfterClose(t, newServer) })
			t.Run("ServerCloseBlocking", func(t *testing.T) { testServerCloseBlocking(t, newServer) })
			t.Run("ServerCloseClientConnections", func(t *testing.T) { testServerCloseClientConnections(t, newServer) })
			t.Run("ServerClientTransportType", func(t *testing.T) { testServerClientTransportType(t, newServer) })
		})
	}
	for _, name := range []string{"NewTLSServer", "NewTLSServerManual"} {
		t.Run(name, func(t *testing.T) {
			newServer := newServers[name]
			t.Run("ServerClient", func(t *testing.T) { testServerClient(t, newServer) })
			t.Run("TLSServerClientTransportType", func(t *testing.T) { testTLSServerClientTransportType(t, newServer) })
		})
	}
}

func testServer(t *testing.T, newServer newServerFunc) {
	ts := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))
	defer ts.Close()
	res, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" {
		t.Errorf("got %q, want hello", string(got))
	}
}

// Issue 12781
func testGetAfterClose(t *testing.T, newServer newServerFunc) {
	ts := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))

	res, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" {
		t.Fatalf("got %q, want hello", string(got))
	}

	ts.Close()

	res, err = http.Get(ts.URL)
	if err == nil {
		body, _ := io.ReadAll(res.Body)
		t.Fatalf("Unexpected response after close: %v, %v, %s", res.Status, res.Header, body)
	}
}

func testServerCloseBlocking(t *testing.T, newServer newServerFunc) {
	ts := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))
	dial := func() net.Conn {
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	// Keep one connection in StateNew (connected, but not sending anything)
	cnew := dial()
	defer cnew.Close()

	// Keep one connection in StateIdle (idle after a request)
	cidle := dial()
	defer cidle.Close()
	cidle.Write([]byte("HEAD / HTTP/1.1\r\nHost: foo\r\n\r\n"))
	_, err := http.ReadResponse(bufio.NewReader(cidle), nil)
	if err != nil {
		t.Fatal(err)
	}

	ts.Close() // test we don't hang here forever.
}

// Issue 14290
func testServerCloseClientConnections(t *testing.T, newServer newServerFunc) {
	var s *Server
	s = newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.CloseClientConnections()
	}))
	defer s.Close()
	res, err := http.Get(s.URL)
	if err == nil {
		res.Body.Close()
		t.Fatalf("Unexpected response: %#v", res)
	}
}

// Tests that the Server.Client method works and returns an http.Client that can hit
// NewTLSServer without cert warnings.
func testServerClient(t *testing.T, newTLSServer newServerFunc) {
	ts := newTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))
	defer ts.Close()
	client := ts.Client()
	res, err := client.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" {
		t.Errorf("got %q, want hello", string(got))
	}
}

// Tests that the Server.Client.Transport interface is implemented
// by a *http.Transport.
func testServerClientTransportType(t *testing.T, newServer newServerFunc) {
	ts := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	}))
	defer ts.Close()
	client := ts.Client()
	if _, ok := client.Transport.(*http.Transport); !ok {
		t.Errorf("got %T, want *http.Transport", client.Transport)
	}
}

// Tests that the TLS Server.Client.Transport interface is implemented
// by a *http.Transport.
func testTLSServerClientTransportType(t *testing.T, newTLSServer newServerFunc) {
	ts := newTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	}))
	defer ts.Close()
	client := ts.Client()
	if _, ok := client.Transport.(*http.Transport); !ok {
		t.Errorf("got %T, want *http.Transport", client.Transport)
	}
}

type onlyCloseListener struct {
	net.Listener
}

func (onlyCloseListener) Close() error { return nil }

// Issue 19729: panic in Server.Close for values created directly
// without a constructor (so the unexported client field is nil).
func TestServerZeroValueClose(t *testing.T) {
	ts := &Server{
		Listener: onlyCloseListener{},
		Config:   &http.Server{},
	}

	ts.Close() // tests that it doesn't panic
}

// Issue 51799: test hijacking a connection and then closing it
// concurrently with closing the server.
func TestCloseHijackedConnection(t *testing.T) {
	hijacked := make(chan net.Conn)
	ts := NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(hijacked)
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("failed to hijack")
		}
		c, _, err := hj.Hijack()
		if err != nil {
			t.Fatal(err)
		}
		hijacked <- c
	}))

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		req, err := http.NewRequest("GET", ts.URL, nil)
		if err != nil {
			t.Log(err)
		}
		// Use a client not associated with the Server.
		var c http.Client
		resp, err := c.Do(req)
		if err != nil {
			t.Log(err)
			return
		}
		resp.Body.Close()
	}()

	wg.Add(1)
	conn := <-hijacked
	go func(conn net.Conn) {
		defer wg.Done()
		// Close the connection and then inform the Server that
		// we closed it.
		conn.Close()
		ts.Config.ConnState(conn, http.StateClosed)
	}(conn)

	wg.Add(1)
	go func() {
		defer wg.Done()
		ts.Close()
	}()
	wg.Wait()
}

func TestTLSServerWithHTTP2(t *testing.T) {
	modes := []struct {
		name      string
		wantProto string
	}{
		{"http1", "HTTP/1.1"},
		{"http2", "HTTP/2.0"},
	}

	for _, tt := range modes {
		t.Run(tt.name, func(t *testing.T) {
			cst := NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("X-Proto", r.Proto)
			}))

			switch tt.name {
			case "http2":
				cst.EnableHTTP2 = true
				cst.StartTLS()
			default:
				cst.Start()
			}

			defer cst.Close()

			res, err := cst.Client().Get(cst.URL)
			if err != nil {
				t.Fatalf("Failed to make request: %v", err)
			}
			if g, w := res.Header.Get("X-Proto"), tt.wantProto; g != w {
				t.Fatalf("X-Proto header mismatch:\n\tgot:  %q\n\twant: %q", g, w)
			}
		})
	}
}
