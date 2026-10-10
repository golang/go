// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputil_test

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func connectionWireServer(t *testing.T, headers string) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var handlers sync.WaitGroup
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		for {
			conn, e := ln.Accept()
			if e != nil {
				return
			}
			handlers.Add(1)
			go func() {
				defer handlers.Done()
				defer conn.Close()
				_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
				if _, e := http.ReadRequest(bufio.NewReader(conn)); e != nil {
					return
				}
				_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\n"+headers+"X-Hop: secret\r\nX-Other: private\r\nX-Extension: keep\r\nContent-Length: 2\r\n\r\nok")
			}()
		}
	}()
	t.Cleanup(func() { _ = ln.Close(); <-stopped; handlers.Wait() })
	return "http://" + ln.Addr().String()
}
func TestReverseProxyConnectionResponseFields(t *testing.T) {
	for _, conn := range []string{"X-Hop, close", "close, X-Hop", "x-hOP, ClOsE", "X-Hop, X-Other, close", "X-Hop\r\nConnection: X-Other, close", "\r\nConnection: X-Hop, close", "X-Hop"} {
		for _, h2 := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/frontendH2=%v", strings.ReplaceAll(conn, "\r\n", ";"), h2), func(t *testing.T) {
				upstream := connectionWireServer(t, "Connection: "+conn+"\r\n")
				target, _ := url.Parse(upstream)
				proxy := httputil.NewSingleHostReverseProxy(target)
				tr := http.DefaultTransport.(*http.Transport).Clone()
				proxy.Transport = tr
				t.Cleanup(tr.CloseIdleConnections)
				front := httptest.NewUnstartedServer(proxy)
				front.EnableHTTP2 = h2
				if h2 {
					front.StartTLS()
				} else {
					front.Start()
				}
				t.Cleanup(front.Close)
				res, e := front.Client().Get(front.URL)
				if e != nil {
					t.Fatal(e)
				}
				b, e := io.ReadAll(res.Body)
				_ = res.Body.Close()
				if e != nil || string(b) != "ok" {
					t.Fatalf("body=%q error=%v", b, e)
				}
				if h2 && res.ProtoMajor != 2 {
					t.Fatal("not actual H2")
				}
				if res.Header.Get("X-Hop") != "" {
					t.Errorf("hop leak: X-Hop=%q", res.Header.Get("X-Hop"))
				}
				if strings.Contains(strings.ToLower(conn), "x-other") && res.Header.Get("X-Other") != "" {
					t.Error("X-Other leaked")
				}
				if res.Header.Get("X-Extension") != "keep" {
					t.Error("end-to-end field lost")
				}
			})
		}
	}
}
func TestConnectionResponseEndpoint(t *testing.T) {
	up := connectionWireServer(t, "Connection: X-Hop, close\r\n")
	tr := http.DefaultTransport.(*http.Transport).Clone()
	defer tr.CloseIdleConnections()
	res, e := (&http.Client{Transport: tr}).Get(up)
	if e != nil {
		t.Fatal(e)
	}
	_, e = io.Copy(io.Discard, res.Body)
	_ = res.Body.Close()
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("Connection=%q Close=%v X-Hop=%q", res.Header.Values("Connection"), res.Close, res.Header.Get("X-Hop"))
	if !res.Close || res.Header.Get("Connection") != "X-Hop, close" || res.Header.Get("X-Hop") != "secret" {
		t.Fatal("ordinary endpoint semantics changed")
	}
}
func TestReverseProxyConnectionHTTP2Origin(t *testing.T) {
	up := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor != 2 {
			t.Error("origin not H2")
		}
		w.Header().Set("X-Extension", "keep")
		_, _ = io.WriteString(w, "ok")
	}))
	up.EnableHTTP2 = true
	up.StartTLS()
	defer up.Close()
	target, _ := url.Parse(up.URL)
	p := httputil.NewSingleHostReverseProxy(target)
	p.Transport = up.Client().Transport
	front := httptest.NewUnstartedServer(p)
	front.EnableHTTP2 = true
	front.StartTLS()
	defer front.Close()
	res, e := front.Client().Get(front.URL)
	if e != nil {
		t.Fatal(e)
	}
	body, e := io.ReadAll(res.Body)
	_ = res.Body.Close()
	if e != nil || string(body) != "ok" || res.StatusCode != 200 || res.ProtoMajor != 2 || res.Header.Get("X-Extension") != "keep" {
		t.Fatal("H2 response")
	}
}

func TestReverseProxyConnectionStreamingReuse(t *testing.T) {
	for _, closeConn := range []bool{false, true} {
		for _, h2 := range []bool{false, true} {
			t.Run(fmt.Sprintf("close=%v/frontendH2=%v", closeConn, h2), func(t *testing.T) {
				var connections atomic.Int32
				release := make(chan struct{}, 3)
				up := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					value := "X-Hop"
					if closeConn {
						value += ", close"
					}
					w.Header().Set("Connection", value)
					w.Header().Set("X-Hop", "secret")
					w.Header().Set("X-Extension", "keep")
					w.Header().Set("Trailer", "X-Checksum")
					io.WriteString(w, "first")
					w.(http.Flusher).Flush()
					select {
					case <-r.Context().Done():
						return
					case <-release:
					}
					io.WriteString(w, "last")
					w.Header().Set("X-Checksum", "good")
				}))
				up.Config.ConnState = func(_ net.Conn, state http.ConnState) {
					if state == http.StateNew {
						connections.Add(1)
					}
				}
				up.Start()
				t.Cleanup(up.Close)
				target, _ := url.Parse(up.URL)
				p := httputil.NewSingleHostReverseProxy(target)
				tr := http.DefaultTransport.(*http.Transport).Clone()
				p.Transport = tr
				t.Cleanup(tr.CloseIdleConnections)
				front := httptest.NewUnstartedServer(p)
				front.EnableHTTP2 = h2
				if h2 {
					front.StartTLS()
				} else {
					front.Start()
				}
				t.Cleanup(front.Close)
				front.Client().Timeout = 3 * time.Second
				for i := 0; i < 3; i++ {
					res, err := front.Client().Get(front.URL)
					if err != nil {
						t.Fatal(err)
					}
					first := make([]byte, 5)
					_, err = io.ReadFull(res.Body, first)
					if err != nil || string(first) != "first" {
						t.Fatalf("first chunk = %q, %v", first, err)
					}
					release <- struct{}{}
					rest, err := io.ReadAll(res.Body)
					res.Body.Close()
					if err != nil || string(rest) != "last" {
						t.Fatalf("last chunk = %q, %v", rest, err)
					}
					if res.Header.Get("X-Hop") != "" || res.Header.Get("X-Extension") != "keep" || res.Trailer.Get("X-Checksum") != "good" {
						t.Fatalf("headers=%v trailers=%v", res.Header, res.Trailer)
					}
					if h2 && res.ProtoMajor != 2 {
						t.Fatal("not HTTP/2")
					}
				}
				want := int32(1)
				if closeConn {
					want = 3
				}
				if got := connections.Load(); got != want {
					t.Fatalf("upstream connections=%d want=%d", got, want)
				}
			})
		}
	}
}
