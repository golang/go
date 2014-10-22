// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	. "net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNextProtoUpgrade(t *testing.T) {
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "path=%s,proto=", r.URL.Path)
		if r.TLS != nil {
			w.Write([]byte(r.TLS.NegotiatedProtocol))
		}
		if r.RemoteAddr == "" {
			t.Error("request with no RemoteAddr")
		}
		if r.Body == nil {
			t.Errorf("request with nil Body")
		}
	}))
	ts.TLS = &tls.Config{
		NextProtos: []string{"unhandled-proto", "tls-0.9"},
	}
	ts.Config.TLSNextProto = map[string]func(*Server, *tls.Conn, Handler){
		"tls-0.9": handleTLSProtocol09,
	}
	ts.StartTLS()
	defer ts.Close()

	tr := newTLSTransport(t, ts)
	defer tr.CloseIdleConnections()
	c := &Client{Transport: tr}

	// Normal request, without NPN.
	{
		res, err := c.Get(ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		body, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		if want := "path=/,proto="; string(body) != want {
			t.Errorf("plain request = %q; want %q", body, want)
		}
	}

	// Request to an advertised but unhandled NPN protocol.
	// Server will hang up.
	{
		tr.CloseIdleConnections()
		tr.TLSClientConfig.NextProtos = []string{"unhandled-proto"}
		_, err := c.Get(ts.URL)
		if err == nil {
			t.Errorf("expected error on unhandled-proto request")
		}
	}

	// Request using the "tls-0.9" protocol, which we register here.
	// It is HTTP/0.9 over TLS.
	{
		tlsConfig := newTLSTransport(t, ts).TLSClientConfig
		tlsConfig.NextProtos = []string{"tls-0.9"}
		conn, err := tls.Dial("tcp", ts.Listener.Addr().String(), tlsConfig)
		if err != nil {
			t.Fatal(err)
		}
		conn.Write([]byte("GET /foo\n"))
		body, err := ioutil.ReadAll(conn)
		if err != nil {
			t.Fatal(err)
		}
		if want := "path=/foo,proto=tls-0.9"; string(body) != want {
			t.Errorf("plain request = %q; want %q", body, want)
		}
	}
}

// handleTLSProtocol09 implements the HTTP/0.9 protocol over TLS, for the
// TestNextProtoUpgrade test.
func handleTLSProtocol09(srv *Server, conn *tls.Conn, h Handler) {
	br := bufio.NewReader(conn)
	line, err := br.ReadString('\n')
	if err != nil {
		return
	}
	line = strings.TrimSpace(line)
	path := strings.TrimPrefix(line, "GET ")
	if path == line {
		return
	}
	req, _ := NewRequest("GET", path, nil)
	req.Proto = "HTTP/0.9"
	req.ProtoMajor = 0
	req.ProtoMinor = 9
	rw := &http09Writer{conn, make(Header)}
	h.ServeHTTP(rw, req)
}

type http09Writer struct {
	io.Writer
	h Header
}

func (w http09Writer) Header() Header  { return w.h }
func (w http09Writer) WriteHeader(int) {} // no headers
