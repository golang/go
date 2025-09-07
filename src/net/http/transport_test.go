// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for transport.go.
//
// More tests are in clientserver_test.go (for things testing both client & server for both
// HTTP/1 and HTTP/2). This

package http_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"errors"
	"fmt"
	"go/token"
	"internal/nettrace"
	"internal/synctest"
	"io"
	"log"
	mrand "math/rand"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/http/httputil"
	"net/http/internal/testcert"
	"net/textproto"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/iotest"
	"time"

	"golang.org/x/net/http/httpguts"
)

// TODO: test 5 pipelined requests with responses: 1) OK, 2) OK, Connection: Close
// and then verify that the final 2 responses get errors back.

// hostPortHandler writes back the client's "host:port".
var hostPortHandler = HandlerFunc(func(w ResponseWriter, r *Request) {
	if r.FormValue("close") == "true" {
		w.Header().Set("Connection", "close")
	}
	w.Header().Set("X-Saw-Close", fmt.Sprint(r.Close))
	w.Write([]byte(r.RemoteAddr))

	// Include the address of the net.Conn in addition to the RemoteAddr,
	// in case kernels reuse source ports quickly (see Issue 52450)
	if c, ok := ResponseWriterConnForTesting(w); ok {
		fmt.Fprintf(w, ", %T %p", c, c)
	}
})

// testCloseConn is a net.Conn tracked by a testConnSet.
type testCloseConn struct {
	net.Conn
	set *testConnSet
}

func (c *testCloseConn) Close() error {
	c.set.remove(c)
	return c.Conn.Close()
}

// testConnSet tracks a set of TCP connections and whether they've
// been closed.
type testConnSet struct {
	t      *testing.T
	mu     sync.Mutex // guards closed and list
	closed map[net.Conn]bool
	list   []net.Conn // in order created
}

func (tcs *testConnSet) insert(c net.Conn) {
	tcs.mu.Lock()
	defer tcs.mu.Unlock()
	tcs.closed[c] = false
	tcs.list = append(tcs.list, c)
}

func (tcs *testConnSet) remove(c net.Conn) {
	tcs.mu.Lock()
	defer tcs.mu.Unlock()
	tcs.closed[c] = true
}

// some tests use this to manage raw tcp connections for later inspection
func makeTestDial(t *testing.T) (*testConnSet, func(n, addr string) (net.Conn, error)) {
	connSet := &testConnSet{
		t:      t,
		closed: make(map[net.Conn]bool),
	}
	dial := func(n, addr string) (net.Conn, error) {
		c, err := net.Dial(n, addr)
		if err != nil {
			return nil, err
		}
		tc := &testCloseConn{c, connSet}
		connSet.insert(tc)
		return tc, nil
	}
	return connSet, dial
}

func (tcs *testConnSet) check(t *testing.T) {
	tcs.mu.Lock()
	defer tcs.mu.Unlock()
	for i := 4; i >= 0; i-- {
		for i, c := range tcs.list {
			if tcs.closed[c] {
				continue
			}
			if i != 0 {
				// TODO(bcmills): What is the Sleep here doing, and why is this
				// Unlock/Sleep/Lock cycle needed at all?
				tcs.mu.Unlock()
				time.Sleep(50 * time.Millisecond)
				tcs.mu.Lock()
				continue
			}
			t.Errorf("TCP connection #%d, %p (of %d total) was not closed", i+1, c, len(tcs.list))
		}
	}
}

func TestReuseRequest(t *testing.T) { run(t, testReuseRequest) }
func testReuseRequest(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("{}"))
	})).ts

	c := ts.Client()
	req, _ := NewRequest("GET", ts.URL, nil)
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	err = res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}

	res, err = c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	err = res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
}

// Two subsequent requests and verify their response is the same.
// The response from the server is our own IP:port
func TestTransportKeepAlives(t *testing.T) { run(t, testTransportKeepAlives, []testMode{http1Mode}) }
func testTransportKeepAlives(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, hostPortHandler).ts

	c := ts.Client()
	for _, disableKeepAlive := range []bool{false, true} {
		c.Transport.(*Transport).DisableKeepAlives = disableKeepAlive
		fetch := func(n int) string {
			res, err := c.Get(ts.URL)
			if err != nil {
				t.Fatalf("error in disableKeepAlive=%v, req #%d, GET: %v", disableKeepAlive, n, err)
			}
			body, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("error in disableKeepAlive=%v, req #%d, ReadAll: %v", disableKeepAlive, n, err)
			}
			return string(body)
		}

		body1 := fetch(1)
		body2 := fetch(2)

		bodiesDiffer := body1 != body2
		if bodiesDiffer != disableKeepAlive {
			t.Errorf("error in disableKeepAlive=%v. unexpected bodiesDiffer=%v; body1=%q; body2=%q",
				disableKeepAlive, bodiesDiffer, body1, body2)
		}
	}
}

func TestTransportConnectionCloseOnResponse(t *testing.T) {
	run(t, testTransportConnectionCloseOnResponse)
}
func testTransportConnectionCloseOnResponse(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, hostPortHandler).ts

	connSet, testDial := makeTestDial(t)

	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.Dial = testDial

	for _, connectionClose := range []bool{false, true} {
		fetch := func(n int) string {
			req := new(Request)
			var err error
			req.URL, err = url.Parse(ts.URL + fmt.Sprintf("/?close=%v", connectionClose))
			if err != nil {
				t.Fatalf("URL parse error: %v", err)
			}
			req.Method = "GET"
			req.Proto = "HTTP/1.1"
			req.ProtoMajor = 1
			req.ProtoMinor = 1

			res, err := c.Do(req)
			if err != nil {
				t.Fatalf("error in connectionClose=%v, req #%d, Do: %v", connectionClose, n, err)
			}
			defer res.Body.Close()
			body, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("error in connectionClose=%v, req #%d, ReadAll: %v", connectionClose, n, err)
			}
			return string(body)
		}

		body1 := fetch(1)
		body2 := fetch(2)
		bodiesDiffer := body1 != body2
		if bodiesDiffer != connectionClose {
			t.Errorf("error in connectionClose=%v. unexpected bodiesDiffer=%v; body1=%q; body2=%q",
				connectionClose, bodiesDiffer, body1, body2)
		}

		tr.CloseIdleConnections()
	}

	connSet.check(t)
}

// TestTransportConnectionCloseOnRequest tests that the Transport's doesn't reuse
// an underlying TCP connection after making an http.Request with Request.Close set.
//
// It tests the behavior by making an HTTP request to a server which
// describes the source connection it got (remote port number +
// address of its net.Conn).
func TestTransportConnectionCloseOnRequest(t *testing.T) {
	run(t, testTransportConnectionCloseOnRequest, []testMode{http1Mode})
}
func testTransportConnectionCloseOnRequest(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, hostPortHandler).ts

	connSet, testDial := makeTestDial(t)

	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.Dial = testDial
	for _, reqClose := range []bool{false, true} {
		fetch := func(n int) string {
			req := new(Request)
			var err error
			req.URL, err = url.Parse(ts.URL)
			if err != nil {
				t.Fatalf("URL parse error: %v", err)
			}
			req.Method = "GET"
			req.Proto = "HTTP/1.1"
			req.ProtoMajor = 1
			req.ProtoMinor = 1
			req.Close = reqClose

			res, err := c.Do(req)
			if err != nil {
				t.Fatalf("error in Request.Close=%v, req #%d, Do: %v", reqClose, n, err)
			}
			if got, want := res.Header.Get("X-Saw-Close"), fmt.Sprint(reqClose); got != want {
				t.Errorf("for Request.Close = %v; handler's X-Saw-Close was %v; want %v",
					reqClose, got, !reqClose)
			}
			body, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("for Request.Close=%v, on request %v/2: ReadAll: %v", reqClose, n, err)
			}
			return string(body)
		}

		body1 := fetch(1)
		body2 := fetch(2)

		got := 1
		if body1 != body2 {
			got++
		}
		want := 1
		if reqClose {
			want = 2
		}
		if got != want {
			t.Errorf("for Request.Close=%v: server saw %v unique connections, wanted %v\n\nbodies were: %q and %q",
				reqClose, got, want, body1, body2)
		}

		tr.CloseIdleConnections()
	}

	connSet.check(t)
}

// if the Transport's DisableKeepAlives is set, all requests should
// send Connection: close.
// HTTP/1-only (Connection: close doesn't exist in h2)
func TestTransportConnectionCloseOnRequestDisableKeepAlive(t *testing.T) {
	run(t, testTransportConnectionCloseOnRequestDisableKeepAlive, []testMode{http1Mode})
}
func testTransportConnectionCloseOnRequestDisableKeepAlive(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, hostPortHandler).ts

	c := ts.Client()
	c.Transport.(*Transport).DisableKeepAlives = true

	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.Header.Get("X-Saw-Close") != "true" {
		t.Errorf("handler didn't see Connection: close ")
	}
}

// Test that Transport only sends one "Connection: close", regardless of
// how "close" was indicated.
func TestTransportRespectRequestWantsClose(t *testing.T) {
	run(t, testTransportRespectRequestWantsClose, []testMode{http1Mode})
}
func testTransportRespectRequestWantsClose(t *testing.T, mode testMode) {
	tests := []struct {
		disableKeepAlives bool
		close             bool
	}{
		{disableKeepAlives: false, close: false},
		{disableKeepAlives: false, close: true},
		{disableKeepAlives: true, close: false},
		{disableKeepAlives: true, close: true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("DisableKeepAlive=%v,RequestClose=%v", tc.disableKeepAlives, tc.close),
			func(t *testing.T) {
				ts := newClientServerTest(t, mode, hostPortHandler).ts

				c := ts.Client()
				c.Transport.(*Transport).DisableKeepAlives = tc.disableKeepAlives
				req, err := NewRequest("GET", ts.URL, nil)
				if err != nil {
					t.Fatal(err)
				}
				count := 0
				trace := &httptrace.ClientTrace{
					WroteHeaderField: func(key string, field []string) {
						if key != "Connection" {
							return
						}
						if httpguts.HeaderValuesContainsToken(field, "close") {
							count += 1
						}
					},
				}
				req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
				req.Close = tc.close
				res, err := c.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				defer res.Body.Close()
				if want := tc.disableKeepAlives || tc.close; count > 1 || (count == 1) != want {
					t.Errorf("expecting want:%v, got 'Connection: close':%d", want, count)
				}
			})
	}

}

func TestTransportIdleCacheKeys(t *testing.T) {
	run(t, testTransportIdleCacheKeys, []testMode{http1Mode})
}
func testTransportIdleCacheKeys(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, hostPortHandler).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)

	if e, g := 0, len(tr.IdleConnKeysForTesting()); e != g {
		t.Errorf("After CloseIdleConnections expected %d idle conn cache keys; got %d", e, g)
	}

	resp, err := c.Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	io.ReadAll(resp.Body)

	keys := tr.IdleConnKeysForTesting()
	if e, g := 1, len(keys); e != g {
		t.Fatalf("After Get expected %d idle conn cache keys; got %d", e, g)
	}

	if e := "|http|" + ts.Listener.Addr().String(); keys[0] != e {
		t.Errorf("Expected idle cache key %q; got %q", e, keys[0])
	}

	tr.CloseIdleConnections()
	if e, g := 0, len(tr.IdleConnKeysForTesting()); e != g {
		t.Errorf("After CloseIdleConnections expected %d idle conn cache keys; got %d", e, g)
	}
}

// Tests that the HTTP transport re-uses connections when a client
// reads to the end of a response Body without closing it.
func TestTransportReadToEndReusesConn(t *testing.T) { run(t, testTransportReadToEndReusesConn) }
func testTransportReadToEndReusesConn(t *testing.T, mode testMode) {
	const msg = "foobar"

	var addrSeen map[string]int
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		addrSeen[r.RemoteAddr]++
		if r.URL.Path == "/chunked/" {
			w.WriteHeader(200)
			w.(Flusher).Flush()
		} else {
			w.Header().Set("Content-Length", strconv.Itoa(len(msg)))
			w.WriteHeader(200)
		}
		w.Write([]byte(msg))
	})).ts

	for pi, path := range []string{"/content-length/", "/chunked/"} {
		wantLen := []int{len(msg), -1}[pi]
		addrSeen = make(map[string]int)
		for i := 0; i < 3; i++ {
			res, err := ts.Client().Get(ts.URL + path)
			if err != nil {
				t.Errorf("Get %s: %v", path, err)
				continue
			}
			// We want to close this body eventually (before the
			// defer afterTest at top runs), but not before the
			// len(addrSeen) check at the bottom of this test,
			// since Closing this early in the loop would risk
			// making connections be re-used for the wrong reason.
			defer res.Body.Close()

			if res.ContentLength != int64(wantLen) {
				t.Errorf("%s res.ContentLength = %d; want %d", path, res.ContentLength, wantLen)
			}
			got, err := io.ReadAll(res.Body)
			if string(got) != msg || err != nil {
				t.Errorf("%s ReadAll(Body) = %q, %v; want %q, nil", path, string(got), err, msg)
			}
		}
		if len(addrSeen) != 1 {
			t.Errorf("for %s, server saw %d distinct client addresses; want 1", path, len(addrSeen))
		}
	}
}

// Tests that the HTTP transport re-uses connections when a client
// early closes a response Body but the content is fully read into the underlying
// buffer. So we can discard the body buffer and reuse the connection.
func TestTransportReusesEarlyCloseButAllReceivedConn(t *testing.T) {
	run(t, testTransportReusesEarlyCloseButAllReceivedConn)
}
func testTransportReusesEarlyCloseButAllReceivedConn(t *testing.T, mode testMode) {
	const msg = "foobar"

	var addrSeen map[string]int
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		addrSeen[r.RemoteAddr]++
		w.Header().Set("Content-Length", strconv.Itoa(len(msg)))
		w.WriteHeader(200)
		w.Write([]byte(msg))
	})).ts

	wantLen := len(msg)
	addrSeen = make(map[string]int)
	total := 5
	for i := 0; i < total; i++ {
		res, err := ts.Client().Get(ts.URL)
		if err != nil {
			t.Errorf("Get /: %v", err)
			continue
		}

		if res.ContentLength != int64(wantLen) {
			t.Errorf("res.ContentLength = %d; want %d", res.ContentLength, wantLen)
		}

		if i+1 < total {
			// Close body directly. The body is small enough, so probably the underlying bufio.Reader
			// has read entire body into buffer. Thus even if the body is not read, the buffer is discarded
			// then connection is reused.
			res.Body.Close()
		} else {
			// when reading body, everything should be same.
			got, err := io.ReadAll(res.Body)
			if string(got) != msg || err != nil {
				t.Errorf("ReadAll(Body) = %q, %v; want %q, nil", string(got), err, msg)
			}
		}
	}

	if len(addrSeen) != 1 {
		t.Errorf("server saw %d distinct client addresses; want 1", len(addrSeen))
	}
}

func TestTransportMaxPerHostIdleConns(t *testing.T) {
	run(t, testTransportMaxPerHostIdleConns, []testMode{http1Mode})
}
func testTransportMaxPerHostIdleConns(t *testing.T, mode testMode) {
	stop := make(chan struct{}) // stop marks the exit of main Test goroutine
	defer close(stop)

	resch := make(chan string)
	gotReq := make(chan bool)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		gotReq <- true
		var msg string
		select {
		case <-stop:
			return
		case msg = <-resch:
		}
		_, err := w.Write([]byte(msg))
		if err != nil {
			t.Errorf("Write: %v", err)
			return
		}
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)
	maxIdleConnsPerHost := 2
	tr.MaxIdleConnsPerHost = maxIdleConnsPerHost

	// Start 3 outstanding requests and wait for the server to get them.
	// Their responses will hang until we write to resch, though.
	donech := make(chan bool)
	doReq := func() {
		defer func() {
			select {
			case <-stop:
				return
			case donech <- t.Failed():
			}
		}()
		resp, err := c.Get(ts.URL)
		if err != nil {
			t.Error(err)
			return
		}
		if _, err := io.ReadAll(resp.Body); err != nil {
			t.Errorf("ReadAll: %v", err)
			return
		}
	}
	go doReq()
	<-gotReq
	go doReq()
	<-gotReq
	go doReq()
	<-gotReq

	if e, g := 0, len(tr.IdleConnKeysForTesting()); e != g {
		t.Fatalf("Before writes, expected %d idle conn cache keys; got %d", e, g)
	}

	resch <- "res1"
	<-donech
	keys := tr.IdleConnKeysForTesting()
	if e, g := 1, len(keys); e != g {
		t.Fatalf("after first response, expected %d idle conn cache keys; got %d", e, g)
	}
	addr := ts.Listener.Addr().String()
	cacheKey := "|http|" + addr
	if keys[0] != cacheKey {
		t.Fatalf("Expected idle cache key %q; got %q", cacheKey, keys[0])
	}
	if e, g := 1, tr.IdleConnCountForTesting("http", addr); e != g {
		t.Errorf("after first response, expected %d idle conns; got %d", e, g)
	}

	resch <- "res2"
	<-donech
	if g, w := tr.IdleConnCountForTesting("http", addr), 2; g != w {
		t.Errorf("after second response, idle conns = %d; want %d", g, w)
	}

	resch <- "res3"
	<-donech
	if g, w := tr.IdleConnCountForTesting("http", addr), maxIdleConnsPerHost; g != w {
		t.Errorf("after third response, idle conns = %d; want %d", g, w)
	}
}

func TestTransportMaxConnsPerHostIncludeDialInProgress(t *testing.T) {
	run(t, testTransportMaxConnsPerHostIncludeDialInProgress)
}
func testTransportMaxConnsPerHostIncludeDialInProgress(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("foo"))
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
	})).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)
	dialStarted := make(chan struct{})
	stallDial := make(chan struct{})
	tr.Dial = func(network, addr string) (net.Conn, error) {
		dialStarted <- struct{}{}
		<-stallDial
		return net.Dial(network, addr)
	}

	tr.DisableKeepAlives = true
	tr.MaxConnsPerHost = 1

	preDial := make(chan struct{})
	reqComplete := make(chan struct{})
	doReq := func(reqId string) {
		req, _ := NewRequest("GET", ts.URL, nil)
		trace := &httptrace.ClientTrace{
			GetConn: func(hostPort string) {
				preDial <- struct{}{}
			},
		}
		req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
		resp, err := tr.RoundTrip(req)
		if err != nil {
			t.Errorf("unexpected error for request %s: %v", reqId, err)
		}
		_, err = io.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("unexpected error for request %s: %v", reqId, err)
		}
		reqComplete <- struct{}{}
	}
	// get req1 to dial-in-progress
	go doReq("req1")
	<-preDial
	<-dialStarted

	// get req2 to waiting on conns per host to go down below max
	go doReq("req2")
	<-preDial
	select {
	case <-dialStarted:
		t.Error("req2 dial started while req1 dial in progress")
		return
	default:
	}

	// let req1 complete
	stallDial <- struct{}{}
	<-reqComplete

	// let req2 complete
	<-dialStarted
	stallDial <- struct{}{}
	<-reqComplete
}

func TestTransportMaxConnsPerHost(t *testing.T) {
	run(t, testTransportMaxConnsPerHost, []testMode{http1Mode, https1Mode, http2Mode})
}
func testTransportMaxConnsPerHost(t *testing.T, mode testMode) {
	CondSkipHTTP2(t)

	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("foo"))
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
	})

	ts := newClientServerTest(t, mode, h).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.MaxConnsPerHost = 1

	mu := sync.Mutex{}
	var conns []net.Conn
	var dialCnt, gotConnCnt, tlsHandshakeCnt int32
	tr.Dial = func(network, addr string) (net.Conn, error) {
		atomic.AddInt32(&dialCnt, 1)
		c, err := net.Dial(network, addr)
		mu.Lock()
		defer mu.Unlock()
		conns = append(conns, c)
		return c, err
	}

	doReq := func() {
		trace := &httptrace.ClientTrace{
			GotConn: func(connInfo httptrace.GotConnInfo) {
				if !connInfo.Reused {
					atomic.AddInt32(&gotConnCnt, 1)
				}
			},
			TLSHandshakeStart: func() {
				atomic.AddInt32(&tlsHandshakeCnt, 1)
			},
		}
		req, _ := NewRequest("GET", ts.URL, nil)
		req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))

		resp, err := c.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()
		_, err = io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("read body failed: %v", err)
		}
	}

	wg := sync.WaitGroup{}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			doReq()
		}()
	}
	wg.Wait()

	expected := int32(tr.MaxConnsPerHost)
	if dialCnt != expected {
		t.Errorf("round 1: too many dials: %d != %d", dialCnt, expected)
	}
	if gotConnCnt != expected {
		t.Errorf("round 1: too many get connections: %d != %d", gotConnCnt, expected)
	}
	if ts.TLS != nil && tlsHandshakeCnt != expected {
		t.Errorf("round 1: too many tls handshakes: %d != %d", tlsHandshakeCnt, expected)
	}

	if t.Failed() {
		t.FailNow()
	}

	mu.Lock()
	for _, c := range conns {
		c.Close()
	}
	conns = nil
	mu.Unlock()
	tr.CloseIdleConnections()

	doReq()
	expected++
	if dialCnt != expected {
		t.Errorf("round 2: too many dials: %d", dialCnt)
	}
	if gotConnCnt != expected {
		t.Errorf("round 2: too many get connections: %d != %d", gotConnCnt, expected)
	}
	if ts.TLS != nil && tlsHandshakeCnt != expected {
		t.Errorf("round 2: too many tls handshakes: %d != %d", tlsHandshakeCnt, expected)
	}
}

func TestTransportMaxConnsPerHostDialCancellation(t *testing.T) {
	run(t, testTransportMaxConnsPerHostDialCancellation,
		testNotParallel, // because test uses SetPendingDialHooks
		[]testMode{http1Mode, https1Mode, http2Mode},
	)
}

func testTransportMaxConnsPerHostDialCancellation(t *testing.T, mode testMode) {
	CondSkipHTTP2(t)

	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("foo"))
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
	})

	cst := newClientServerTest(t, mode, h)
	defer cst.close()
	ts := cst.ts
	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.MaxConnsPerHost = 1

	// This request is canceled when dial is queued, which preempts dialing.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	SetPendingDialHooks(cancel, nil)
	defer SetPendingDialHooks(nil, nil)

	req, _ := NewRequestWithContext(ctx, "GET", ts.URL, nil)
	_, err := c.Do(req)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected error %v, got %v", context.Canceled, err)
	}

	// This request should succeed.
	SetPendingDialHooks(nil, nil)
	req, _ = NewRequest("GET", ts.URL, nil)
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body failed: %v", err)
	}
}

func TestTransportRemovesDeadIdleConnections(t *testing.T) {
	run(t, testTransportRemovesDeadIdleConnections, []testMode{http1Mode})
}
func testTransportRemovesDeadIdleConnections(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, r.RemoteAddr)
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)

	doReq := func(name string) {
		// Do a POST instead of a GET to prevent the Transport's
		// idempotent request retry logic from kicking in...
		res, err := c.Post(ts.URL, "", nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if res.StatusCode != 200 {
			t.Fatalf("%s: %v", name, res.Status)
		}
		defer res.Body.Close()
		slurp, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		t.Logf("%s: ok (%q)", name, slurp)
	}

	doReq("first")
	keys1 := tr.IdleConnKeysForTesting()

	ts.CloseClientConnections()

	var keys2 []string
	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		keys2 = tr.IdleConnKeysForTesting()
		if len(keys2) != 0 {
			if d > 0 {
				t.Logf("Transport hasn't noticed idle connection's death in %v.\nbefore: %q\n after: %q\n", d, keys1, keys2)
			}
			return false
		}
		return true
	})

	doReq("second")
}

// Test that the Transport notices when a server hangs up on its
// unexpectedly (a keep-alive connection is closed).
func TestTransportServerClosingUnexpectedly(t *testing.T) {
	run(t, testTransportServerClosingUnexpectedly, []testMode{http1Mode})
}
func testTransportServerClosingUnexpectedly(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, hostPortHandler).ts
	c := ts.Client()

	fetch := func(n, retries int) string {
		condFatalf := func(format string, arg ...any) {
			if retries <= 0 {
				t.Fatalf(format, arg...)
			}
			t.Logf("retrying shortly after expected error: "+format, arg...)
			time.Sleep(time.Second / time.Duration(retries))
		}
		for retries >= 0 {
			retries--
			res, err := c.Get(ts.URL)
			if err != nil {
				condFatalf("error in req #%d, GET: %v", n, err)
				continue
			}
			body, err := io.ReadAll(res.Body)
			if err != nil {
				condFatalf("error in req #%d, ReadAll: %v", n, err)
				continue
			}
			res.Body.Close()
			return string(body)
		}
		panic("unreachable")
	}

	body1 := fetch(1, 0)
	body2 := fetch(2, 0)

	// Close all the idle connections in a way that's similar to
	// the server hanging up on us. We don't use
	// httptest.Server.CloseClientConnections because it's
	// best-effort and stops blocking after 5 seconds. On a loaded
	// machine running many tests concurrently it's possible for
	// that method to be async and cause the body3 fetch below to
	// run on an old connection. This function is synchronous.
	ExportCloseTransportConnsAbruptly(c.Transport.(*Transport))

	body3 := fetch(3, 5)

	if body1 != body2 {
		t.Errorf("expected body1 and body2 to be equal")
	}
	if body2 == body3 {
		t.Errorf("expected body2 and body3 to be different")
	}
}

// Test for https://golang.org/issue/2616 (appropriate issue number)
// This fails pretty reliably with GOMAXPROCS=100 or something high.
func TestStressSurpriseServerCloses(t *testing.T) {
	run(t, testStressSurpriseServerCloses, []testMode{http1Mode})
}
func testStressSurpriseServerCloses(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "5")
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("Hello"))
		w.(Flusher).Flush()
		conn, buf, _ := w.(Hijacker).Hijack()
		buf.Flush()
		conn.Close()
	})).ts
	c := ts.Client()

	// Do a bunch of traffic from different goroutines. Send to activityc
	// after each request completes, regardless of whether it failed.
	// If these are too high, OS X exhausts its ephemeral ports
	// and hangs waiting for them to transition TCP states. That's
	// not what we want to test. TODO(bradfitz): use an io.Pipe
	// dialer for this test instead?
	const (
		numClients    = 20
		reqsPerClient = 25
	)
	var wg sync.WaitGroup
	wg.Add(numClients * reqsPerClient)
	for i := 0; i < numClients; i++ {
		go func() {
			for i := 0; i < reqsPerClient; i++ {
				res, err := c.Get(ts.URL)
				if err == nil {
					// We expect errors since the server is
					// hanging up on us after telling us to
					// send more requests, so we don't
					// actually care what the error is.
					// But we want to close the body in cases
					// where we won the race.
					res.Body.Close()
				}
				wg.Done()
			}
		}()
	}

	// Make sure all the request come back, one way or another.
	wg.Wait()
}

// TestTransportHeadResponses verifies that we deal with Content-Lengths
// with no bodies properly
func TestTransportHeadResponses(t *testing.T) { run(t, testTransportHeadResponses) }
func testTransportHeadResponses(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method != "HEAD" {
			panic("expected HEAD; got " + r.Method)
		}
		w.Header().Set("Content-Length", "123")
		w.WriteHeader(200)
	})).ts
	c := ts.Client()

	for i := 0; i < 2; i++ {
		res, err := c.Head(ts.URL)
		if err != nil {
			t.Errorf("error on loop %d: %v", i, err)
			continue
		}
		if e, g := "123", res.Header.Get("Content-Length"); e != g {
			t.Errorf("loop %d: expected Content-Length header of %q, got %q", i, e, g)
		}
		if e, g := int64(123), res.ContentLength; e != g {
			t.Errorf("loop %d: expected res.ContentLength of %v, got %v", i, e, g)
		}
		if all, err := io.ReadAll(res.Body); err != nil {
			t.Errorf("loop %d: Body ReadAll: %v", i, err)
		} else if len(all) != 0 {
			t.Errorf("Bogus body %q", all)
		}
	}
}

// TestTransportHeadChunkedResponse verifies that we ignore chunked transfer-encoding
// on responses to HEAD requests.
func TestTransportHeadChunkedResponse(t *testing.T) {
	run(t, testTransportHeadChunkedResponse, []testMode{http1Mode}, testNotParallel)
}
func testTransportHeadChunkedResponse(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method != "HEAD" {
			panic("expected HEAD; got " + r.Method)
		}
		w.Header().Set("Transfer-Encoding", "chunked") // client should ignore
		w.Header().Set("x-client-ipport", r.RemoteAddr)
		w.WriteHeader(200)
	})).ts
	c := ts.Client()

	// Ensure that we wait for the readLoop to complete before
	// calling Head again
	didRead := make(chan bool)
	SetReadLoopBeforeNextReadHook(func() { didRead <- true })
	defer SetReadLoopBeforeNextReadHook(nil)

	res1, err := c.Head(ts.URL)
	<-didRead

	if err != nil {
		t.Fatalf("request 1 error: %v", err)
	}

	res2, err := c.Head(ts.URL)
	<-didRead

	if err != nil {
		t.Fatalf("request 2 error: %v", err)
	}
	if v1, v2 := res1.Header.Get("x-client-ipport"), res2.Header.Get("x-client-ipport"); v1 != v2 {
		t.Errorf("ip/ports differed between head requests: %q vs %q", v1, v2)
	}
}

var roundTripTests = []struct {
	accept       string
	expectAccept string
	compressed   bool
}{
	// Requests with no accept-encoding header use transparent compression
	{"", "gzip", false},
	// Requests with other accept-encoding should pass through unmodified
	{"foo", "foo", false},
	// Requests with accept-encoding == gzip should be passed through
	{"gzip", "gzip", true},
}

// Test that the modification made to the Request by the RoundTripper is cleaned up
func TestRoundTripGzip(t *testing.T) { run(t, testRoundTripGzip) }
func testRoundTripGzip(t *testing.T, mode testMode) {
	const responseBody = "test response body"
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		accept := req.Header.Get("Accept-Encoding")
		if expect := req.FormValue("expect_accept"); accept != expect {
			t.Errorf("in handler, test %v: Accept-Encoding = %q, want %q",
				req.FormValue("testnum"), accept, expect)
		}
		if accept == "gzip" {
			rw.Header().Set("Content-Encoding", "gzip")
			gz := gzip.NewWriter(rw)
			gz.Write([]byte(responseBody))
			gz.Close()
		} else {
			rw.Header().Set("Content-Encoding", accept)
			rw.Write([]byte(responseBody))
		}
	})).ts
	tr := ts.Client().Transport.(*Transport)

	for i, test := range roundTripTests {
		// Test basic request (no accept-encoding)
		req, _ := NewRequest("GET", fmt.Sprintf("%s/?testnum=%d&expect_accept=%s", ts.URL, i, test.expectAccept), nil)
		if test.accept != "" {
			req.Header.Set("Accept-Encoding", test.accept)
		}
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Errorf("%d. RoundTrip: %v", i, err)
			continue
		}
		var body []byte
		if test.compressed {
			var r *gzip.Reader
			r, err = gzip.NewReader(res.Body)
			if err != nil {
				t.Errorf("%d. gzip NewReader: %v", i, err)
				continue
			}
			body, err = io.ReadAll(r)
			res.Body.Close()
		} else {
			body, err = io.ReadAll(res.Body)
		}
		if err != nil {
			t.Errorf("%d. Error: %q", i, err)
			continue
		}
		if g, e := string(body), responseBody; g != e {
			t.Errorf("%d. body = %q; want %q", i, g, e)
		}
		if g, e := req.Header.Get("Accept-Encoding"), test.accept; g != e {
			t.Errorf("%d. Accept-Encoding = %q; want %q (it was mutated, in violation of RoundTrip contract)", i, g, e)
		}
		if g, e := res.Header.Get("Content-Encoding"), test.accept; g != e {
			t.Errorf("%d. Content-Encoding = %q; want %q", i, g, e)
		}
	}

}

func TestTransportGzip(t *testing.T) { run(t, testTransportGzip) }
func testTransportGzip(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("https://go.dev/issue/56020")
	}
	const testString = "The test string aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const nRandBytes = 1024 * 1024
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		if req.Method == "HEAD" {
			if g := req.Header.Get("Accept-Encoding"); g != "" {
				t.Errorf("HEAD request sent with Accept-Encoding of %q; want none", g)
			}
			return
		}
		if g, e := req.Header.Get("Accept-Encoding"), "gzip"; g != e {
			t.Errorf("Accept-Encoding = %q, want %q", g, e)
		}
		rw.Header().Set("Content-Encoding", "gzip")

		var w io.Writer = rw
		var buf bytes.Buffer
		if req.FormValue("chunked") == "0" {
			w = &buf
			defer io.Copy(rw, &buf)
			defer func() {
				rw.Header().Set("Content-Length", strconv.Itoa(buf.Len()))
			}()
		}
		gz := gzip.NewWriter(w)
		gz.Write([]byte(testString))
		if req.FormValue("body") == "large" {
			io.CopyN(gz, rand.Reader, nRandBytes)
		}
		gz.Close()
	})).ts
	c := ts.Client()

	for _, chunked := range []string{"1", "0"} {
		// First fetch something large, but only read some of it.
		res, err := c.Get(ts.URL + "/?body=large&chunked=" + chunked)
		if err != nil {
			t.Fatalf("large get: %v", err)
		}
		buf := make([]byte, len(testString))
		n, err := io.ReadFull(res.Body, buf)
		if err != nil {
			t.Fatalf("partial read of large response: size=%d, %v", n, err)
		}
		if e, g := testString, string(buf); e != g {
			t.Errorf("partial read got %q, expected %q", g, e)
		}
		res.Body.Close()
		// Read on the body, even though it's closed
		n, err = res.Body.Read(buf)
		if n != 0 || err == nil {
			t.Errorf("expected error post-closed large Read; got = %d, %v", n, err)
		}

		// Then something small.
		res, err = c.Get(ts.URL + "/?chunked=" + chunked)
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		if g, e := string(body), testString; g != e {
			t.Fatalf("body = %q; want %q", g, e)
		}
		if g, e := res.Header.Get("Content-Encoding"), ""; g != e {
			t.Fatalf("Content-Encoding = %q; want %q", g, e)
		}

		// Read on the body after it's been fully read:
		n, err = res.Body.Read(buf)
		if n != 0 || err == nil {
			t.Errorf("expected Read error after exhausted reads; got %d, %v", n, err)
		}
		res.Body.Close()
		n, err = res.Body.Read(buf)
		if n != 0 || err == nil {
			t.Errorf("expected Read error after Close; got %d, %v", n, err)
		}
	}

	// And a HEAD request too, because they're always weird.
	res, err := c.Head(ts.URL)
	if err != nil {
		t.Fatalf("Head: %v", err)
	}
	if res.StatusCode != 200 {
		t.Errorf("Head status=%d; want=200", res.StatusCode)
	}
}

// A transport100Continue test exercises Transport behaviors when sending a
// request with an Expect: 100-continue header.
type transport100ContinueTest struct {
	t *testing.T

	reqdone chan struct{}
	resp    *Response
	respErr error

	conn   net.Conn
	reader *bufio.Reader
}

const transport100ContinueTestBody = "request body"

// newTransport100ContinueTest creates a Transport and sends an Expect: 100-continue
// request on it.
func newTransport100ContinueTest(t *testing.T, timeout time.Duration) *transport100ContinueTest {
	ln := newLocalListener(t)
	defer ln.Close()

	test := &transport100ContinueTest{
		t:       t,
		reqdone: make(chan struct{}),
	}

	tr := &Transport{
		ExpectContinueTimeout: timeout,
	}
	go func() {
		defer close(test.reqdone)
		body := strings.NewReader(transport100ContinueTestBody)
		req, _ := NewRequest("PUT", "http://"+ln.Addr().String(), body)
		req.Header.Set("Expect", "100-continue")
		req.ContentLength = int64(len(transport100ContinueTestBody))
		test.resp, test.respErr = tr.RoundTrip(req)
		test.resp.Body.Close()
	}()

	c, err := ln.Accept()
	if err != nil {
		t.Fatalf("Accept: %v", err)
	}
	t.Cleanup(func() {
		c.Close()
	})
	br := bufio.NewReader(c)
	_, err = ReadRequest(br)
	if err != nil {
		t.Fatalf("ReadRequest: %v", err)
	}
	test.conn = c
	test.reader = br
	t.Cleanup(func() {
		<-test.reqdone
		tr.CloseIdleConnections()
		got, _ := io.ReadAll(test.reader)
		if len(got) > 0 {
			t.Fatalf("Transport sent unexpected bytes: %q", got)
		}
	})

	return test
}

// respond sends response lines from the server to the transport.
func (test *transport100ContinueTest) respond(lines ...string) {
	for _, line := range lines {
		if _, err := test.conn.Write([]byte(line + "\r\n")); err != nil {
			test.t.Fatalf("Write: %v", err)
		}
	}
	if _, err := test.conn.Write([]byte("\r\n")); err != nil {
		test.t.Fatalf("Write: %v", err)
	}
}

// wantBodySent ensures the transport has sent the request body to the server.
func (test *transport100ContinueTest) wantBodySent() {
	got, err := io.ReadAll(io.LimitReader(test.reader, int64(len(transport100ContinueTestBody))))
	if err != nil {
		test.t.Fatalf("unexpected error reading body: %v", err)
	}
	if got, want := string(got), transport100ContinueTestBody; got != want {
		test.t.Fatalf("unexpected body: got %q, want %q", got, want)
	}
}

// wantRequestDone ensures the Transport.RoundTrip has completed with the expected status.
func (test *transport100ContinueTest) wantRequestDone(want int) {
	<-test.reqdone
	if test.respErr != nil {
		test.t.Fatalf("unexpected RoundTrip error: %v", test.respErr)
	}
	if got := test.resp.StatusCode; got != want {
		test.t.Fatalf("unexpected response code: got %v, want %v", got, want)
	}
}

func TestTransportExpect100ContinueSent(t *testing.T) {
	test := newTransport100ContinueTest(t, 1*time.Hour)
	// Server sends a 100 Continue response, and the client sends the request body.
	test.respond("HTTP/1.1 100 Continue")
	test.wantBodySent()
	test.respond("HTTP/1.1 200", "Content-Length: 0")
	test.wantRequestDone(200)
}

func TestTransportExpect100Continue200ResponseNoConnClose(t *testing.T) {
	test := newTransport100ContinueTest(t, 1*time.Hour)
	// No 100 Continue response, no Connection: close header.
	test.respond("HTTP/1.1 200", "Content-Length: 0")
	test.wantBodySent()
	test.wantRequestDone(200)
}

func TestTransportExpect100Continue200ResponseWithConnClose(t *testing.T) {
	test := newTransport100ContinueTest(t, 1*time.Hour)
	// No 100 Continue response, Connection: close header set.
	test.respond("HTTP/1.1 200", "Connection: close", "Content-Length: 0")
	test.wantRequestDone(200)
}

func TestTransportExpect100Continue500ResponseNoConnClose(t *testing.T) {
	test := newTransport100ContinueTest(t, 1*time.Hour)
	// No 100 Continue response, no Connection: close header.
	test.respond("HTTP/1.1 500", "Content-Length: 0")
	test.wantBodySent()
	test.wantRequestDone(500)
}

func TestTransportExpect100Continue500ResponseTimeout(t *testing.T) {
	test := newTransport100ContinueTest(t, 5*time.Millisecond) // short timeout
	test.wantBodySent()                                        // after timeout
	test.respond("HTTP/1.1 200", "Content-Length: 0")
	test.wantRequestDone(200)
}

func TestSOCKS5Proxy(t *testing.T) {
	run(t, testSOCKS5Proxy, []testMode{http1Mode, https1Mode, http2Mode})
}
func testSOCKS5Proxy(t *testing.T, mode testMode) {
	ch := make(chan string, 1)
	l := newLocalListener(t)
	defer l.Close()
	defer close(ch)
	proxy := func(t *testing.T) {
		s, err := l.Accept()
		if err != nil {
			t.Errorf("socks5 proxy Accept(): %v", err)
			return
		}
		defer s.Close()
		var buf [22]byte
		if _, err := io.ReadFull(s, buf[:3]); err != nil {
			t.Errorf("socks5 proxy initial read: %v", err)
			return
		}
		if want := []byte{5, 1, 0}; !bytes.Equal(buf[:3], want) {
			t.Errorf("socks5 proxy initial read: got %v, want %v", buf[:3], want)
			return
		}
		if _, err := s.Write([]byte{5, 0}); err != nil {
			t.Errorf("socks5 proxy initial write: %v", err)
			return
		}
		if _, err := io.ReadFull(s, buf[:4]); err != nil {
			t.Errorf("socks5 proxy second read: %v", err)
			return
		}
		if want := []byte{5, 1, 0}; !bytes.Equal(buf[:3], want) {
			t.Errorf("socks5 proxy second read: got %v, want %v", buf[:3], want)
			return
		}
		var ipLen int
		switch buf[3] {
		case 1:
			ipLen = net.IPv4len
		case 4:
			ipLen = net.IPv6len
		default:
			t.Errorf("socks5 proxy second read: unexpected address type %v", buf[4])
			return
		}
		if _, err := io.ReadFull(s, buf[4:ipLen+6]); err != nil {
			t.Errorf("socks5 proxy address read: %v", err)
			return
		}
		ip := net.IP(buf[4 : ipLen+4])
		port := binary.BigEndian.Uint16(buf[ipLen+4 : ipLen+6])
		copy(buf[:3], []byte{5, 0, 0})
		if _, err := s.Write(buf[:ipLen+6]); err != nil {
			t.Errorf("socks5 proxy connect write: %v", err)
			return
		}
		ch <- fmt.Sprintf("proxy for %s:%d", ip, port)

		// Implement proxying.
		targetHost := net.JoinHostPort(ip.String(), strconv.Itoa(int(port)))
		targetConn, err := net.Dial("tcp", targetHost)
		if err != nil {
			t.Errorf("net.Dial failed")
			return
		}
		go io.Copy(targetConn, s)
		io.Copy(s, targetConn) // Wait for the client to close the socket.
		targetConn.Close()
	}

	pu, err := url.Parse("socks5://" + l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	sentinelHeader := "X-Sentinel"
	sentinelValue := "12345"
	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set(sentinelHeader, sentinelValue)
	})
	for _, useTLS := range []bool{false, true} {
		t.Run(fmt.Sprintf("useTLS=%v", useTLS), func(t *testing.T) {
			ts := newClientServerTest(t, mode, h).ts
			go proxy(t)
			c := ts.Client()
			c.Transport.(*Transport).Proxy = ProxyURL(pu)
			r, err := c.Head(ts.URL)
			if err != nil {
				t.Fatal(err)
			}
			if r.Header.Get(sentinelHeader) != sentinelValue {
				t.Errorf("Failed to retrieve sentinel value")
			}
			got := <-ch
			ts.Close()
			tsu, err := url.Parse(ts.URL)
			if err != nil {
				t.Fatal(err)
			}
			want := "proxy for " + tsu.Host
			if got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		})
	}
}

func TestTransportProxy(t *testing.T) {
	defer afterTest(t)
	testCases := []struct{ siteMode, proxyMode testMode }{
		{http1Mode, http1Mode},
		{http1Mode, https1Mode},
		{https1Mode, http1Mode},
		{https1Mode, https1Mode},
	}
	for _, testCase := range testCases {
		siteMode := testCase.siteMode
		proxyMode := testCase.proxyMode
		t.Run(fmt.Sprintf("site=%v/proxy=%v", siteMode, proxyMode), func(t *testing.T) {
			siteCh := make(chan *Request, 1)
			h1 := HandlerFunc(func(w ResponseWriter, r *Request) {
				siteCh <- r
			})
			proxyCh := make(chan *Request, 1)
			h2 := HandlerFunc(func(w ResponseWriter, r *Request) {
				proxyCh <- r
				// Implement an entire CONNECT proxy
				if r.Method == "CONNECT" {
					hijacker, ok := w.(Hijacker)
					if !ok {
						t.Errorf("hijack not allowed")
						return
					}
					clientConn, _, err := hijacker.Hijack()
					if err != nil {
						t.Errorf("hijacking failed")
						return
					}
					res := &Response{
						StatusCode: StatusOK,
						Proto:      "HTTP/1.1",
						ProtoMajor: 1,
						ProtoMinor: 1,
						Header:     make(Header),
					}

					targetConn, err := net.Dial("tcp", r.URL.Host)
					if err != nil {
						t.Errorf("net.Dial(%q) failed: %v", r.URL.Host, err)
						return
					}

					if err := res.Write(clientConn); err != nil {
						t.Errorf("Writing 200 OK failed: %v", err)
						return
					}

					go io.Copy(targetConn, clientConn)
					go func() {
						io.Copy(clientConn, targetConn)
						targetConn.Close()
					}()
				}
			})
			ts := newClientServerTest(t, siteMode, h1).ts
			proxy := newClientServerTest(t, proxyMode, h2).ts

			pu, err := url.Parse(proxy.URL)
			if err != nil {
				t.Fatal(err)
			}

			// If neither server is HTTPS or both are, then c may be derived from either.
			// If only one server is HTTPS, c must be derived from that server in order
			// to ensure that it is configured to use the fake root CA from testcert.go.
			c := proxy.Client()
			if siteMode == https1Mode {
				c = ts.Client()
			}

			c.Transport.(*Transport).Proxy = ProxyURL(pu)
			if _, err := c.Head(ts.URL); err != nil {
				t.Error(err)
			}
			got := <-proxyCh
			c.Transport.(*Transport).CloseIdleConnections()
			ts.Close()
			proxy.Close()
			if siteMode == https1Mode {
				// First message should be a CONNECT, asking for a socket to the real server,
				if got.Method != "CONNECT" {
					t.Errorf("Wrong method for secure proxying: %q", got.Method)
				}
				gotHost := got.URL.Host
				pu, err := url.Parse(ts.URL)
				if err != nil {
					t.Fatal("Invalid site URL")
				}
				if wantHost := pu.Host; gotHost != wantHost {
					t.Errorf("Got CONNECT host %q, want %q", gotHost, wantHost)
				}

				// The next message on the channel should be from the site's server.
				next := <-siteCh
				if next.Method != "HEAD" {
					t.Errorf("Wrong method at destination: %s", next.Method)
				}
				if nextURL := next.URL.String(); nextURL != "/" {
					t.Errorf("Wrong URL at destination: %s", nextURL)
				}
			} else {
				if got.Method != "HEAD" {
					t.Errorf("Wrong method for destination: %q", got.Method)
				}
				gotURL := got.URL.String()
				wantURL := ts.URL + "/"
				if gotURL != wantURL {
					t.Errorf("Got URL %q, want %q", gotURL, wantURL)
				}
			}
		})
	}
}

// Issue 74633: verify that a client will not indefinitely read a response from
// a proxy server that writes an infinite byte of stream, rather than
// responding with 200 OK.
func TestProxyWithInfiniteHeader(t *testing.T) {
	defer afterTest(t)

	ln := newLocalListener(t)
	defer ln.Close()
	cancelc := make(chan struct{})
	defer close(cancelc)

	// Simulate a malicious / misbehaving proxy that writes an unlimited number
	// of bytes rather than responding with 200 OK.
	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("Accept: %v", err)
			return
		}
		defer c.Close()
		// Read the CONNECT request
		br := bufio.NewReader(c)
		cr, err := ReadRequest(br)
		if err != nil {
			t.Errorf("proxy server failed to read CONNECT request")
			return
		}
		if cr.Method != "CONNECT" {
			t.Errorf("unexpected method %q", cr.Method)
			return
		}

		// Keep writing bytes until the test exits.
		for {
			// runtime.Gosched() is needed here. Otherwise, this test might
			// livelock in environments like WASM, where the one single thread
			// we have could be hogged by the infinite loop of writing bytes.
			runtime.Gosched()
			select {
			case <-cancelc:
				return
			default:
				c.Write([]byte("infinite stream of bytes"))
			}
		}
	}()

	c := &Client{
		Transport: &Transport{
			Proxy: func(*Request) (*url.URL, error) {
				return url.Parse("http://" + ln.Addr().String())
			},
			// Limit MaxResponseHeaderBytes so the test returns quicker.
			MaxResponseHeaderBytes: 1024,
		},
	}
	req, err := NewRequest("GET", "https://golang.fake.tld/", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Do(req)
	if err == nil {
		t.Errorf("unexpected Get success")
	}
}

func TestOnProxyConnectResponse(t *testing.T) {

	var tcases = []struct {
		proxyStatusCode int
		err             error
	}{
		{
			StatusOK,
			nil,
		},
		{
			StatusForbidden,
			errors.New("403"),
		},
	}
	for _, tcase := range tcases {
		h1 := HandlerFunc(func(w ResponseWriter, r *Request) {

		})

		h2 := HandlerFunc(func(w ResponseWriter, r *Request) {
			// Implement an entire CONNECT proxy
			if r.Method == "CONNECT" {
				if tcase.proxyStatusCode != StatusOK {
					w.WriteHeader(tcase.proxyStatusCode)
					return
				}
				hijacker, ok := w.(Hijacker)
				if !ok {
					t.Errorf("hijack not allowed")
					return
				}
				clientConn, _, err := hijacker.Hijack()
				if err != nil {
					t.Errorf("hijacking failed")
					return
				}
				res := &Response{
					StatusCode: StatusOK,
					Proto:      "HTTP/1.1",
					ProtoMajor: 1,
					ProtoMinor: 1,
					Header:     make(Header),
				}

				targetConn, err := net.Dial("tcp", r.URL.Host)
				if err != nil {
					t.Errorf("net.Dial(%q) failed: %v", r.URL.Host, err)
					return
				}

				if err := res.Write(clientConn); err != nil {
					t.Errorf("Writing 200 OK failed: %v", err)
					return
				}

				go io.Copy(targetConn, clientConn)
				go func() {
					io.Copy(clientConn, targetConn)
					targetConn.Close()
				}()
			}
		})
		ts := newClientServerTest(t, https1Mode, h1).ts
		proxy := newClientServerTest(t, https1Mode, h2).ts

		pu, err := url.Parse(proxy.URL)
		if err != nil {
			t.Fatal(err)
		}

		c := proxy.Client()

		var (
			dials  atomic.Int32
			closes atomic.Int32
		)
		c.Transport.(*Transport).DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := net.Dial(network, addr)
			if err != nil {
				return nil, err
			}
			dials.Add(1)
			return noteCloseConn{
				Conn: conn,
				closeFunc: func() {
					closes.Add(1)
				},
			}, nil
		}

		c.Transport.(*Transport).Proxy = ProxyURL(pu)
		c.Transport.(*Transport).OnProxyConnectResponse = func(ctx context.Context, proxyURL *url.URL, connectReq *Request, connectRes *Response) error {
			if proxyURL.String() != pu.String() {
				t.Errorf("proxy url got %s, want %s", proxyURL, pu)
			}

			if "https://"+connectReq.URL.String() != ts.URL {
				t.Errorf("connect url got %s, want %s", connectReq.URL, ts.URL)
			}
			return tcase.err
		}
		wantCloses := int32(0)
		if _, err := c.Head(ts.URL); err != nil {
			wantCloses = 1
			if tcase.err != nil && !strings.Contains(err.Error(), tcase.err.Error()) {
				t.Errorf("got %v, want %v", err, tcase.err)
			}
		} else {
			if tcase.err != nil {
				t.Errorf("got %v, want nil", err)
			}
		}
		if got, want := dials.Load(), int32(1); got != want {
			t.Errorf("got %v dials, want %v", got, want)
		}
		// #64804: If OnProxyConnectResponse returns an error, we should close the conn.
		if got, want := closes.Load(), wantCloses; got != want {
			t.Errorf("got %v closes, want %v", got, want)
		}
	}
}

// Issue 28012: verify that the Transport closes its TCP connection to http proxies
// when they're slow to reply to HTTPS CONNECT responses.
func TestTransportProxyHTTPSConnectLeak(t *testing.T) {
	cancelc := make(chan struct{})
	SetTestHookProxyConnectTimeout(t, func(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
		ctx, cancel := context.WithCancel(ctx)
		go func() {
			select {
			case <-cancelc:
			case <-ctx.Done():
			}
			cancel()
		}()
		return ctx, cancel
	})

	defer afterTest(t)

	ln := newLocalListener(t)
	defer ln.Close()
	listenerDone := make(chan struct{})
	go func() {
		defer close(listenerDone)
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("Accept: %v", err)
			return
		}
		defer c.Close()
		// Read the CONNECT request
		br := bufio.NewReader(c)
		cr, err := ReadRequest(br)
		if err != nil {
			t.Errorf("proxy server failed to read CONNECT request")
			return
		}
		if cr.Method != "CONNECT" {
			t.Errorf("unexpected method %q", cr.Method)
			return
		}

		// Now hang and never write a response; instead, cancel the request and wait
		// for the client to close.
		// (Prior to Issue 28012 being fixed, we never closed.)
		close(cancelc)
		var buf [1]byte
		_, err = br.Read(buf[:])
		if err != io.EOF {
			t.Errorf("proxy server Read err = %v; want EOF", err)
		}
		return
	}()

	c := &Client{
		Transport: &Transport{
			Proxy: func(*Request) (*url.URL, error) {
				return url.Parse("http://" + ln.Addr().String())
			},
		},
	}
	req, err := NewRequest("GET", "https://golang.fake.tld/", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Do(req)
	if err == nil {
		t.Errorf("unexpected Get success")
	}

	// Wait unconditionally for the listener goroutine to exit: this should never
	// hang, so if it does we want a full goroutine dump — and that's exactly what
	// the testing package will give us when the test run times out.
	<-listenerDone
}

// Issue 16997: test transport dial preserves typed errors
func TestTransportDialPreservesNetOpProxyError(t *testing.T) {
	defer afterTest(t)

	var errDial = errors.New("some dial error")

	tr := &Transport{
		Proxy: func(*Request) (*url.URL, error) {
			return url.Parse("http://proxy.fake.tld/")
		},
		Dial: func(string, string) (net.Conn, error) {
			return nil, errDial
		},
	}
	defer tr.CloseIdleConnections()

	c := &Client{Transport: tr}
	req, _ := NewRequest("GET", "http://fake.tld", nil)
	res, err := c.Do(req)
	if err == nil {
		res.Body.Close()
		t.Fatal("wanted a non-nil error")
	}

	uerr, ok := err.(*url.Error)
	if !ok {
		t.Fatalf("got %T, want *url.Error", err)
	}
	oe, ok := uerr.Err.(*net.OpError)
	if !ok {
		t.Fatalf("url.Error.Err =  %T; want *net.OpError", uerr.Err)
	}
	want := &net.OpError{
		Op:  "proxyconnect",
		Net: "tcp",
		Err: errDial, // original error, unwrapped.
	}
	if !reflect.DeepEqual(oe, want) {
		t.Errorf("Got error %#v; want %#v", oe, want)
	}
}

// Issue 36431: calls to RoundTrip should not mutate t.ProxyConnectHeader.
//
// (A bug caused dialConn to instead write the per-request Proxy-Authorization
// header through to the shared Header instance, introducing a data race.)
func TestTransportProxyDialDoesNotMutateProxyConnectHeader(t *testing.T) {
	run(t, testTransportProxyDialDoesNotMutateProxyConnectHeader)
}
func testTransportProxyDialDoesNotMutateProxyConnectHeader(t *testing.T, mode testMode) {
	proxy := newClientServerTest(t, mode, NotFoundHandler()).ts
	defer proxy.Close()
	c := proxy.Client()

	tr := c.Transport.(*Transport)
	tr.Proxy = func(*Request) (*url.URL, error) {
		u, _ := url.Parse(proxy.URL)
		u.User = url.UserPassword("aladdin", "opensesame")
		return u, nil
	}
	h := tr.ProxyConnectHeader
	if h == nil {
		h = make(Header)
	}
	tr.ProxyConnectHeader = h.Clone()

	req, err := NewRequest("GET", "https://golang.fake.tld/", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Do(req)
	if err == nil {
		t.Errorf("unexpected Get success")
	}

	if !reflect.DeepEqual(tr.ProxyConnectHeader, h) {
		t.Errorf("tr.ProxyConnectHeader = %v; want %v", tr.ProxyConnectHeader, h)
	}
}

// TestTransportGzipRecursive sends a gzip quine and checks that the
// client gets the same value back. This is more cute than anything,
// but checks that we don't recurse forever, and checks that
// Content-Encoding is removed.
func TestTransportGzipRecursive(t *testing.T) { run(t, testTransportGzipRecursive) }
func testTransportGzipRecursive(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Encoding", "gzip")
		w.Write(rgz)
	})).ts

	c := ts.Client()
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(body, rgz) {
		t.Fatalf("Incorrect result from recursive gz:\nhave=%x\nwant=%x",
			body, rgz)
	}
	if g, e := res.Header.Get("Content-Encoding"), ""; g != e {
		t.Fatalf("Content-Encoding = %q; want %q", g, e)
	}
}

// golang.org/issue/7750: request fails when server replies with
// a short gzip body
func TestTransportGzipShort(t *testing.T) { run(t, testTransportGzipShort) }
func testTransportGzipShort(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Encoding", "gzip")
		w.Write([]byte{0x1f, 0x8b})
	})).ts

	c := ts.Client()
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	_, err = io.ReadAll(res.Body)
	if err == nil {
		t.Fatal("Expect an error from reading a body.")
	}
	if err != io.ErrUnexpectedEOF {
		t.Errorf("ReadAll error = %v; want io.ErrUnexpectedEOF", err)
	}
}

// Wait until number of goroutines is no greater than nmax, or time out.
func waitNumGoroutine(nmax int) int {
	nfinal := runtime.NumGoroutine()
	for ntries := 10; ntries > 0 && nfinal > nmax; ntries-- {
		time.Sleep(50 * time.Millisecond)
		runtime.GC()
		nfinal = runtime.NumGoroutine()
	}
	return nfinal
}

// tests that persistent goroutine connections shut down when no longer desired.
func TestTransportPersistConnLeak(t *testing.T) {
	run(t, testTransportPersistConnLeak, testNotParallel)
}
func testTransportPersistConnLeak(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("flaky in HTTP/2")
	}
	// Not parallel: counts goroutines

	const numReq = 25
	gotReqCh := make(chan bool, numReq)
	unblockCh := make(chan bool, numReq)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		gotReqCh <- true
		<-unblockCh
		w.Header().Set("Content-Length", "0")
		w.WriteHeader(204)
	})).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)

	n0 := runtime.NumGoroutine()

	didReqCh := make(chan bool, numReq)
	failed := make(chan bool, numReq)
	for i := 0; i < numReq; i++ {
		go func() {
			res, err := c.Get(ts.URL)
			didReqCh <- true
			if err != nil {
				t.Logf("client fetch error: %v", err)
				failed <- true
				return
			}
			res.Body.Close()
		}()
	}

	// Wait for all goroutines to be stuck in the Handler.
	for i := 0; i < numReq; i++ {
		select {
		case <-gotReqCh:
			// ok
		case <-failed:
			// Not great but not what we are testing:
			// sometimes an overloaded system will fail to make all the connections.
		}
	}

	nhigh := runtime.NumGoroutine()

	// Tell all handlers to unblock and reply.
	close(unblockCh)

	// Wait for all HTTP clients to be done.
	for i := 0; i < numReq; i++ {
		<-didReqCh
	}

	tr.CloseIdleConnections()
	nfinal := waitNumGoroutine(n0 + 5)

	growth := nfinal - n0

	// We expect 0 or 1 extra goroutine, empirically. Allow up to 5.
	// Previously we were leaking one per numReq.
	if int(growth) > 5 {
		t.Logf("goroutine growth: %d -> %d -> %d (delta: %d)", n0, nhigh, nfinal, growth)
		t.Error("too many new goroutines")
	}
}

// golang.org/issue/4531: Transport leaks goroutines when
// request.ContentLength is explicitly short
func TestTransportPersistConnLeakShortBody(t *testing.T) {
	run(t, testTransportPersistConnLeakShortBody, testNotParallel)
}
func testTransportPersistConnLeakShortBody(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("flaky in HTTP/2")
	}

	// Not parallel: measures goroutines.
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
	})).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)

	n0 := runtime.NumGoroutine()
	body := []byte("Hello")
	for i := 0; i < 20; i++ {
		req, err := NewRequest("POST", ts.URL, bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.ContentLength = int64(len(body) - 2) // explicitly short
		_, err = c.Do(req)
		if err == nil {
			t.Fatal("Expect an error from writing too long of a body.")
		}
	}
	nhigh := runtime.NumGoroutine()
	tr.CloseIdleConnections()
	nfinal := waitNumGoroutine(n0 + 5)

	growth := nfinal - n0

	// We expect 0 or 1 extra goroutine, empirically. Allow up to 5.
	// Previously we were leaking one per numReq.
	t.Logf("goroutine growth: %d -> %d -> %d (delta: %d)", n0, nhigh, nfinal, growth)
	if int(growth) > 5 {
		t.Error("too many new goroutines")
	}
}

// A countedConn is a net.Conn that decrements an atomic counter when finalized.
type countedConn struct {
	net.Conn
}

// A countingDialer dials connections and counts the number that remain reachable.
type countingDialer struct {
	dialer      net.Dialer
	mu          sync.Mutex
	total, live int64
}

func (d *countingDialer) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	conn, err := d.dialer.DialContext(ctx, network, address)
	if err != nil {
		return nil, err
	}

	counted := new(countedConn)
	counted.Conn = conn

	d.mu.Lock()
	defer d.mu.Unlock()
	d.total++
	d.live++

	runtime.AddCleanup(counted, func(dd *countingDialer) { dd.decrement(nil) }, d)
	return counted, nil
}

func (d *countingDialer) decrement(*countedConn) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.live--
}

func (d *countingDialer) Read() (total, live int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.total, d.live
}

func TestTransportPersistConnLeakNeverIdle(t *testing.T) {
	run(t, testTransportPersistConnLeakNeverIdle, []testMode{http1Mode})
}
func testTransportPersistConnLeakNeverIdle(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// Close every connection so that it cannot be kept alive.
		conn, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Errorf("Hijack failed unexpectedly: %v", err)
			return
		}
		conn.Close()
	})).ts

	var d countingDialer
	c := ts.Client()
	c.Transport.(*Transport).DialContext = d.DialContext

	body := []byte("Hello")
	for i := 0; ; i++ {
		total, live := d.Read()
		if live < total {
			break
		}
		if i >= 1<<12 {
			t.Fatalf("Count of live client net.Conns (%d) not lower than total (%d) after %d Do / GC iterations.", live, total, i)
		}

		req, err := NewRequest("POST", ts.URL, bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		_, err = c.Do(req)
		if err == nil {
			t.Fatal("expected broken connection")
		}

		runtime.GC()
	}
}

type countedContext struct {
	context.Context
}

type contextCounter struct {
	mu   sync.Mutex
	live int64
}

func (cc *contextCounter) Track(ctx context.Context) context.Context {
	counted := new(countedContext)
	counted.Context = ctx
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.live++
	runtime.AddCleanup(counted, func(c *contextCounter) { cc.decrement(nil) }, cc)
	return counted
}

func (cc *contextCounter) decrement(*countedContext) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.live--
}

func (cc *contextCounter) Read() (live int64) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.live
}

func TestTransportPersistConnContextLeakMaxConnsPerHost(t *testing.T) {
	run(t, testTransportPersistConnContextLeakMaxConnsPerHost)
}
func testTransportPersistConnContextLeakMaxConnsPerHost(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("https://go.dev/issue/56021")
	}

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		runtime.Gosched()
		w.WriteHeader(StatusOK)
	})).ts

	c := ts.Client()
	c.Transport.(*Transport).MaxConnsPerHost = 1

	ctx := context.Background()
	body := []byte("Hello")
	doPosts := func(cc *contextCounter) {
		var wg sync.WaitGroup
		for n := 64; n > 0; n-- {
			wg.Add(1)
			go func() {
				defer wg.Done()

				ctx := cc.Track(ctx)
				req, err := NewRequest("POST", ts.URL, bytes.NewReader(body))
				if err != nil {
					t.Error(err)
				}

				_, err = c.Do(req.WithContext(ctx))
				if err != nil {
					t.Errorf("Do failed with error: %v", err)
				}
			}()
		}
		wg.Wait()
	}

	var initialCC contextCounter
	doPosts(&initialCC)

	// flushCC exists only to put pressure on the GC to finalize the initialCC
	// contexts: the flushCC allocations should eventually displace the initialCC
	// allocations.
	var flushCC contextCounter
	for i := 0; ; i++ {
		live := initialCC.Read()
		if live == 0 {
			break
		}
		if i >= 100 {
			t.Fatalf("%d Contexts still not finalized after %d GC cycles.", live, i)
		}
		doPosts(&flushCC)
		runtime.GC()
	}
}

// This used to crash; https://golang.org/issue/3266
func TestTransportIdleConnCrash(t *testing.T) { run(t, testTransportIdleConnCrash) }
func testTransportIdleConnCrash(t *testing.T, mode testMode) {
	var tr *Transport

	unblockCh := make(chan bool, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		<-unblockCh
		tr.CloseIdleConnections()
	})).ts
	c := ts.Client()
	tr = c.Transport.(*Transport)

	didreq := make(chan bool)
	go func() {
		res, err := c.Get(ts.URL)
		if err != nil {
			t.Error(err)
		} else {
			res.Body.Close() // returns idle conn
		}
		didreq <- true
	}()
	unblockCh <- true
	<-didreq
}

// Test that the transport doesn't close the TCP connection early,
// before the response body has been read. This was a regression
// which sadly lacked a triggering test. The large response body made
// the old race easier to trigger.
func TestIssue3644(t *testing.T) { run(t, testIssue3644) }
func testIssue3644(t *testing.T, mode testMode) {
	const numFoos = 5000
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Connection", "close")
		for i := 0; i < numFoos; i++ {
			w.Write([]byte("foo "))
		}
	})).ts
	c := ts.Client()
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	bs, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if len(bs) != numFoos*len("foo ") {
		t.Errorf("unexpected response length")
	}
}

// Test that a client receives a server's reply, even if the server doesn't read
// the entire request body.
func TestIssue3595(t *testing.T) {
	// Not parallel: modifies the global rstAvoidanceDelay.
	run(t, testIssue3595, testNotParallel)
}
func testIssue3595(t *testing.T, mode testMode) {
	runTimeSensitiveTest(t, []time.Duration{
		1 * time.Millisecond,
		5 * time.Millisecond,
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		500 * time.Millisecond,
		time.Second,
		5 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		SetRSTAvoidanceDelay(t, timeout)
		t.Logf("set RST avoidance delay to %v", timeout)

		const deniedMsg = "sorry, denied."
		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			Error(w, deniedMsg, StatusUnauthorized)
		}))
		// We need to close cst explicitly here so that in-flight server
		// requests don't race with the call to SetRSTAvoidanceDelay for a retry.
		defer cst.close()
		ts := cst.ts
		c := ts.Client()

		res, err := c.Post(ts.URL, "application/octet-stream", neverEnding('a'))
		if err != nil {
			return fmt.Errorf("Post: %v", err)
		}
		got, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("Body ReadAll: %v", err)
		}
		t.Logf("server response:\n%s", got)
		if !strings.Contains(string(got), deniedMsg) {
			// If we got an RST packet too early, we should have seen an error
			// from io.ReadAll, not a silently-truncated body.
			t.Errorf("Known bug: response %q does not contain %q", got, deniedMsg)
		}
		return nil
	})
}

// From https://golang.org/issue/4454 ,
// "client fails to handle requests with no body and chunked encoding"
func TestChunkedNoContent(t *testing.T) { run(t, testChunkedNoContent) }
func testChunkedNoContent(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusNoContent)
	})).ts

	c := ts.Client()
	for _, closeBody := range []bool{true, false} {
		const n = 4
		for i := 1; i <= n; i++ {
			res, err := c.Get(ts.URL)
			if err != nil {
				t.Errorf("closingBody=%v, req %d/%d: %v", closeBody, i, n, err)
			} else {
				if closeBody {
					res.Body.Close()
				}
			}
		}
	}
}

func TestTransportConcurrency(t *testing.T) {
	run(t, testTransportConcurrency, testNotParallel, []testMode{http1Mode})
}
func testTransportConcurrency(t *testing.T, mode testMode) {
	// Not parallel: uses global test hooks.
	maxProcs, numReqs := 16, 500
	if testing.Short() {
		maxProcs, numReqs = 4, 50
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(maxProcs))
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%v", r.FormValue("echo"))
	})).ts

	var wg sync.WaitGroup
	wg.Add(numReqs)

	// Due to the Transport's "socket late binding" (see
	// idleConnCh in transport.go), the numReqs HTTP requests
	// below can finish with a dial still outstanding. To keep
	// the leak checker happy, keep track of pending dials and
	// wait for them to finish (and be closed or returned to the
	// idle pool) before we close idle connections.
	SetPendingDialHooks(func() { wg.Add(1) }, wg.Done)
	defer SetPendingDialHooks(nil, nil)

	c := ts.Client()
	reqs := make(chan string)
	defer close(reqs)

	for i := 0; i < maxProcs*2; i++ {
		go func() {
			for req := range reqs {
				res, err := c.Get(ts.URL + "/?echo=" + req)
				if err != nil {
					if runtime.GOOS == "netbsd" && strings.HasSuffix(err.Error(), ": connection reset by peer") {
						// https://go.dev/issue/52168: this test was observed to fail with
						// ECONNRESET errors in Dial on various netbsd builders.
						t.Logf("error on req %s: %v", req, err)
						t.Logf("(see https://go.dev/issue/52168)")
					} else {
						t.Errorf("error on req %s: %v", req, err)
					}
					wg.Done()
					continue
				}
				all, err := io.ReadAll(res.Body)
				if err != nil {
					t.Errorf("read error on req %s: %v", req, err)
				} else if string(all) != req {
					t.Errorf("body of req %s = %q; want %q", req, all, req)
				}
				res.Body.Close()
				wg.Done()
			}
		}()
	}
	for i := 0; i < numReqs; i++ {
		reqs <- fmt.Sprintf("request-%d", i)
	}
	wg.Wait()
}

func TestIssue4191_InfiniteGetTimeout(t *testing.T) { run(t, testIssue4191_InfiniteGetTimeout) }
func testIssue4191_InfiniteGetTimeout(t *testing.T, mode testMode) {
	mux := NewServeMux()
	mux.HandleFunc("/get", func(w ResponseWriter, r *Request) {
		io.Copy(w, neverEnding('a'))
	})
	ts := newClientServerTest(t, mode, mux).ts

	connc := make(chan net.Conn, 1)
	c := ts.Client()
	c.Transport.(*Transport).Dial = func(n, addr string) (net.Conn, error) {
		conn, err := net.Dial(n, addr)
		if err != nil {
			return nil, err
		}
		select {
		case connc <- conn:
		default:
		}
		return conn, nil
	}

	res, err := c.Get(ts.URL + "/get")
	if err != nil {
		t.Fatalf("Error issuing GET: %v", err)
	}
	defer res.Body.Close()

	conn := <-connc
	conn.SetDeadline(time.Now().Add(1 * time.Millisecond))
	_, err = io.Copy(io.Discard, res.Body)
	if err == nil {
		t.Errorf("Unexpected successful copy")
	}
}

func TestIssue4191_InfiniteGetToPutTimeout(t *testing.T) {
	run(t, testIssue4191_InfiniteGetToPutTimeout, []testMode{http1Mode})
}
func testIssue4191_InfiniteGetToPutTimeout(t *testing.T, mode testMode) {
	const debug = false
	mux := NewServeMux()
	mux.HandleFunc("/get", func(w ResponseWriter, r *Request) {
		io.Copy(w, neverEnding('a'))
	})
	mux.HandleFunc("/put", func(w ResponseWriter, r *Request) {
		defer r.Body.Close()
		io.Copy(io.Discard, r.Body)
	})
	ts := newClientServerTest(t, mode, mux).ts
	timeout := 100 * time.Millisecond

	c := ts.Client()
	c.Transport.(*Transport).Dial = func(n, addr string) (net.Conn, error) {
		conn, err := net.Dial(n, addr)
		if err != nil {
			return nil, err
		}
		conn.SetDeadline(time.Now().Add(timeout))
		if debug {
			conn = NewLoggingConn("client", conn)
		}
		return conn, nil
	}

	getFailed := false
	nRuns := 5
	if testing.Short() {
		nRuns = 1
	}
	for i := 0; i < nRuns; i++ {
		if debug {
			println("run", i+1, "of", nRuns)
		}
		sres, err := c.Get(ts.URL + "/get")
		if err != nil {
			if !getFailed {
				// Make the timeout longer, once.
				getFailed = true
				t.Logf("increasing timeout")
				i--
				timeout *= 10
				continue
			}
			t.Errorf("Error issuing GET: %v", err)
			break
		}
		req, _ := NewRequest("PUT", ts.URL+"/put", sres.Body)
		_, err = c.Do(req)
		if err == nil {
			sres.Body.Close()
			t.Errorf("Unexpected successful PUT")
			break
		}
		sres.Body.Close()
	}
	if debug {
		println("tests complete; waiting for handlers to finish")
	}
	ts.Close()
}

func TestTransportResponseHeaderTimeout(t *testing.T) { run(t, testTransportResponseHeaderTimeout) }
func testTransportResponseHeaderTimeout(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping timeout test in -short mode")
	}

	timeout := 2 * time.Millisecond
	retry := true
	for retry && !t.Failed() {
		var srvWG sync.WaitGroup
		inHandler := make(chan bool, 1)
		mux := NewServeMux()
		mux.HandleFunc("/fast", func(w ResponseWriter, r *Request) {
			inHandler <- true
			srvWG.Done()
		})
		mux.HandleFunc("/slow", func(w ResponseWriter, r *Request) {
			inHandler <- true
			<-r.Context().Done()
			srvWG.Done()
		})
		ts := newClientServerTest(t, mode, mux).ts

		c := ts.Client()
		c.Transport.(*Transport).ResponseHeaderTimeout = timeout

		retry = false
		srvWG.Add(3)
		tests := []struct {
			path        string
			wantTimeout bool
		}{
			{path: "/fast"},
			{path: "/slow", wantTimeout: true},
			{path: "/fast"},
		}
		for i, tt := range tests {
			req, _ := NewRequest("GET", ts.URL+tt.path, nil)
			req = req.WithT(t)
			res, err := c.Do(req)
			<-inHandler
			if err != nil {
				uerr, ok := err.(*url.Error)
				if !ok {
					t.Errorf("error is not a url.Error; got: %#v", err)
					continue
				}
				nerr, ok := uerr.Err.(net.Error)
				if !ok {
					t.Errorf("error does not satisfy net.Error interface; got: %#v", err)
					continue
				}
				if !nerr.Timeout() {
					t.Errorf("want timeout error; got: %q", nerr)
					continue
				}
				if !tt.wantTimeout {
					if !retry {
						// The timeout may be set too short. Retry with a longer one.
						t.Logf("unexpected timeout for path %q after %v; retrying with longer timeout", tt.path, timeout)
						timeout *= 2
						retry = true
					}
				}
				if !strings.Contains(err.Error(), "timeout awaiting response headers") {
					t.Errorf("%d. unexpected error: %v", i, err)
				}
				continue
			}
			if tt.wantTimeout {
				t.Errorf(`no error for path %q; expected "timeout awaiting response headers"`, tt.path)
				continue
			}
			if res.StatusCode != 200 {
				t.Errorf("%d for path %q status = %d; want 200", i, tt.path, res.StatusCode)
			}
		}

		srvWG.Wait()
		ts.Close()
	}
}

// A cancelTest is a test of request cancellation.
type cancelTest struct {
	mode     testMode
	newReq   func(req *Request) *Request       // prepare the request to cancel
	cancel   func(tr *Transport, req *Request) // cancel the request
	checkErr func(when string, err error)      // verify the expected error
}

// runCancelTestTransport uses Transport.CancelRequest.
func runCancelTestTransport(t *testing.T, mode testMode, f func(t *testing.T, test cancelTest)) {
	t.Run("TransportCancel", func(t *testing.T) {
		f(t, cancelTest{
			mode: mode,
			newReq: func(req *Request) *Request {
				return req
			},
			cancel: func(tr *Transport, req *Request) {
				tr.CancelRequest(req)
			},
			checkErr: func(when string, err error) {
				if !errors.Is(err, ExportErrRequestCanceled) && !errors.Is(err, ExportErrRequestCanceledConn) {
					t.Errorf("%v error = %v, want errRequestCanceled or errRequestCanceledConn", when, err)
				}
			},
		})
	})
}

// runCancelTestChannel uses Request.Cancel.
func runCancelTestChannel(t *testing.T, mode testMode, f func(t *testing.T, test cancelTest)) {
	cancelc := make(chan struct{})
	cancelOnce := sync.OnceFunc(func() { close(cancelc) })
	f(t, cancelTest{
		mode: mode,
		newReq: func(req *Request) *Request {
			req.Cancel = cancelc
			return req
		},
		cancel: func(tr *Transport, req *Request) {
			cancelOnce()
		},
		checkErr: func(when string, err error) {
			if !errors.Is(err, ExportErrRequestCanceled) && !errors.Is(err, ExportErrRequestCanceledConn) {
				t.Errorf("%v error = %v, want errRequestCanceled or errRequestCanceledConn", when, err)
			}
		},
	})
}

// runCancelTestContext uses a request context.
func runCancelTestContext(t *testing.T, mode testMode, f func(t *testing.T, test cancelTest)) {
	ctx, cancel := context.WithCancel(context.Background())
	f(t, cancelTest{
		mode: mode,
		newReq: func(req *Request) *Request {
			return req.WithContext(ctx)
		},
		cancel: func(tr *Transport, req *Request) {
			cancel()
		},
		checkErr: func(when string, err error) {
			if !errors.Is(err, context.Canceled) {
				t.Errorf("%v error = %v, want context.Canceled", when, err)
			}
		},
	})
}

func runCancelTest(t *testing.T, f func(t *testing.T, test cancelTest), opts ...any) {
	run(t, func(t *testing.T, mode testMode) {
		if mode == http1Mode {
			t.Run("TransportCancel", func(t *testing.T) {
				runCancelTestTransport(t, mode, f)
			})
		}
		t.Run("RequestCancel", func(t *testing.T) {
			runCancelTestChannel(t, mode, f)
		})
		t.Run("ContextCancel", func(t *testing.T) {
			runCancelTestContext(t, mode, f)
		})
	}, opts...)
}

func TestTransportCancelRequest(t *testing.T) {
	runCancelTest(t, testTransportCancelRequest)
}
func testTransportCancelRequest(t *testing.T, test cancelTest) {
	if testing.Short() {
		t.Skip("skipping test in -short mode")
	}

	const msg = "Hello"
	unblockc := make(chan bool)
	ts := newClientServerTest(t, test.mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, msg)
		w.(Flusher).Flush() // send headers and some body
		<-unblockc
	})).ts
	defer close(unblockc)

	c := ts.Client()
	tr := c.Transport.(*Transport)

	req, _ := NewRequest("GET", ts.URL, nil)
	req = test.newReq(req)
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, len(msg))
	n, _ := io.ReadFull(res.Body, body)
	if n != len(body) || !bytes.Equal(body, []byte(msg)) {
		t.Errorf("Body = %q; want %q", body[:n], msg)
	}
	test.cancel(tr, req)

	tail, err := io.ReadAll(res.Body)
	res.Body.Close()
	test.checkErr("Body.Read", err)
	if len(tail) > 0 {
		t.Errorf("Spurious bytes from Body.Read: %q", tail)
	}

	// Verify no outstanding requests after readLoop/writeLoop
	// goroutines shut down.
	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		n := tr.NumPendingRequestsForTesting()
		if n > 0 {
			if d > 0 {
				t.Logf("pending requests = %d after %v (want 0)", n, d)
			}
			return false
		}
		return true
	})
}

func testTransportCancelRequestInDo(t *testing.T, test cancelTest, body io.Reader) {
	if testing.Short() {
		t.Skip("skipping test in -short mode")
	}
	unblockc := make(chan bool)
	ts := newClientServerTest(t, test.mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		<-unblockc
	})).ts
	defer close(unblockc)

	c := ts.Client()
	tr := c.Transport.(*Transport)

	donec := make(chan bool)
	req, _ := NewRequest("GET", ts.URL, body)
	req = test.newReq(req)
	go func() {
		defer close(donec)
		c.Do(req)
	}()

	unblockc <- true
	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		test.cancel(tr, req)
		select {
		case <-donec:
			return true
		default:
			if d > 0 {
				t.Logf("Do of canceled request has not returned after %v", d)
			}
			return false
		}
	})
}

func TestTransportCancelRequestInDo(t *testing.T) {
	runCancelTest(t, func(t *testing.T, test cancelTest) {
		testTransportCancelRequestInDo(t, test, nil)
	})
}

func TestTransportCancelRequestWithBodyInDo(t *testing.T) {
	runCancelTest(t, func(t *testing.T, test cancelTest) {
		testTransportCancelRequestInDo(t, test, bytes.NewBuffer([]byte{0}))
	})
}

func TestTransportCancelRequestInDial(t *testing.T) {
	runCancelTest(t, testTransportCancelRequestInDial)
}
func testTransportCancelRequestInDial(t *testing.T, test cancelTest) {
	defer afterTest(t)
	if testing.Short() {
		t.Skip("skipping test in -short mode")
	}
	var logbuf strings.Builder
	eventLog := log.New(&logbuf, "", 0)

	unblockDial := make(chan bool)
	defer close(unblockDial)

	inDial := make(chan bool)
	tr := &Transport{
		Dial: func(network, addr string) (net.Conn, error) {
			eventLog.Println("dial: blocking")
			if !<-inDial {
				return nil, errors.New("main Test goroutine exited")
			}
			<-unblockDial
			return nil, errors.New("nope")
		},
	}
	cl := &Client{Transport: tr}
	gotres := make(chan bool)
	req, _ := NewRequest("GET", "http://something.no-network.tld/", nil)
	req = test.newReq(req)
	go func() {
		_, err := cl.Do(req)
		eventLog.Printf("Get error = %v", err != nil)
		test.checkErr("Get", err)
		gotres <- true
	}()

	inDial <- true

	eventLog.Printf("canceling")
	test.cancel(tr, req)
	test.cancel(tr, req) // used to panic on second call to Transport.Cancel

	if d, ok := t.Deadline(); ok {
		// When the test's deadline is about to expire, log the pending events for
		// better debugging.
		timeout := time.Until(d) * 19 / 20 // Allow 5% for cleanup.
		timer := time.AfterFunc(timeout, func() {
			panic(fmt.Sprintf("hang in %s. events are: %s", t.Name(), logbuf.String()))
		})
		defer timer.Stop()
	}
	<-gotres

	got := logbuf.String()
	want := `dial: blocking
canceling
Get error = true
`
	if got != want {
		t.Errorf("Got events:\n%s\nWant:\n%s", got, want)
	}
}

// Issue 51354
func TestTransportCancelRequestWithBody(t *testing.T) {
	runCancelTest(t, testTransportCancelRequestWithBody)
}
func testTransportCancelRequestWithBody(t *testing.T, test cancelTest) {
	if testing.Short() {
		t.Skip("skipping test in -short mode")
	}

	const msg = "Hello"
	unblockc := make(chan struct{})
	ts := newClientServerTest(t, test.mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, msg)
		w.(Flusher).Flush() // send headers and some body
		<-unblockc
	})).ts
	defer close(unblockc)

	c := ts.Client()
	tr := c.Transport.(*Transport)

	req, _ := NewRequest("POST", ts.URL, strings.NewReader("withbody"))
	req = test.newReq(req)

	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, len(msg))
	n, _ := io.ReadFull(res.Body, body)
	if n != len(body) || !bytes.Equal(body, []byte(msg)) {
		t.Errorf("Body = %q; want %q", body[:n], msg)
	}
	test.cancel(tr, req)

	tail, err := io.ReadAll(res.Body)
	res.Body.Close()
	test.checkErr("Body.Read", err)
	if len(tail) > 0 {
		t.Errorf("Spurious bytes from Body.Read: %q", tail)
	}

	// Verify no outstanding requests after readLoop/writeLoop
	// goroutines shut down.
	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		n := tr.NumPendingRequestsForTesting()
		if n > 0 {
			if d > 0 {
				t.Logf("pending requests = %d after %v (want 0)", n, d)
			}
			return false
		}
		return true
	})
}

func TestTransportCancelRequestBeforeDo(t *testing.T) {
	// We can't cancel a request that hasn't started using Transport.CancelRequest.
	run(t, func(t *testing.T, mode testMode) {
		t.Run("RequestCancel", func(t *testing.T) {
			runCancelTestChannel(t, mode, testTransportCancelRequestBeforeDo)
		})
		t.Run("ContextCancel", func(t *testing.T) {
			runCancelTestContext(t, mode, testTransportCancelRequestBeforeDo)
		})
	})
}
func testTransportCancelRequestBeforeDo(t *testing.T, test cancelTest) {
	unblockc := make(chan bool)
	cst := newClientServerTest(t, test.mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		<-unblockc
	}))
	defer close(unblockc)

	c := cst.ts.Client()

	req, _ := NewRequest("GET", cst.ts.URL, nil)
	req = test.newReq(req)
	test.cancel(cst.tr, req)

	_, err := c.Do(req)
	test.checkErr("Do", err)
}

// Issue 11020. The returned error message should be errRequestCanceled
func TestTransportCancelRequestBeforeResponseHeaders(t *testing.T) {
	runCancelTest(t, testTransportCancelRequestBeforeResponseHeaders, []testMode{http1Mode})
}
func testTransportCancelRequestBeforeResponseHeaders(t *testing.T, test cancelTest) {
	defer afterTest(t)

	serverConnCh := make(chan net.Conn, 1)
	tr := &Transport{
		Dial: func(network, addr string) (net.Conn, error) {
			cc, sc := net.Pipe()
			serverConnCh <- sc
			return cc, nil
		},
	}
	defer tr.CloseIdleConnections()
	errc := make(chan error, 1)
	req, _ := NewRequest("GET", "http://example.com/", nil)
	req = test.newReq(req)
	go func() {
		_, err := tr.RoundTrip(req)
		errc <- err
	}()

	sc := <-serverConnCh
	verb := make([]byte, 3)
	if _, err := io.ReadFull(sc, verb); err != nil {
		t.Errorf("Error reading HTTP verb from server: %v", err)
	}
	if string(verb) != "GET" {
		t.Errorf("server received %q; want GET", verb)
	}
	defer sc.Close()

	test.cancel(tr, req)

	err := <-errc
	if err == nil {
		t.Fatalf("unexpected success from RoundTrip")
	}
	test.checkErr("RoundTrip", err)
}

// golang.org/issue/3672 -- Client can't close HTTP stream
// Calling Close on a Response.Body used to just read until EOF.
// Now it actually closes the TCP connection.
func TestTransportCloseResponseBody(t *testing.T) { run(t, testTransportCloseResponseBody) }
func testTransportCloseResponseBody(t *testing.T, mode testMode) {
	writeErr := make(chan error, 1)
	msg := []byte("young\n")
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		for {
			_, err := w.Write(msg)
			if err != nil {
				writeErr <- err
				return
			}
			w.(Flusher).Flush()
		}
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)

	req, _ := NewRequest("GET", ts.URL, nil)
	defer tr.CancelRequest(req)

	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	const repeats = 3
	buf := make([]byte, len(msg)*repeats)
	want := bytes.Repeat(msg, repeats)

	_, err = io.ReadFull(res.Body, buf)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf, want) {
		t.Fatalf("read %q; want %q", buf, want)
	}

	if err := res.Body.Close(); err != nil {
		t.Errorf("Close = %v", err)
	}

	if err := <-writeErr; err == nil {
		t.Errorf("expected non-nil write error")
	}
}

type fooProto struct{}

func (fooProto) RoundTrip(req *Request) (*Response, error) {
	res := &Response{
		Status:     "200 OK",
		StatusCode: 200,
		Header:     make(Header),
		Body:       io.NopCloser(strings.NewReader("You wanted " + req.URL.String())),
	}
	return res, nil
}

func TestTransportAltProto(t *testing.T) {
	defer afterTest(t)
	tr := &Transport{}
	c := &Client{Transport: tr}
	tr.RegisterProtocol("foo", fooProto{})
	res, err := c.Get("foo://bar.com/path")
	if err != nil {
		t.Fatal(err)
	}
	bodyb, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	body := string(bodyb)
	if e := "You wanted foo://bar.com/path"; body != e {
		t.Errorf("got response %q, want %q", body, e)
	}
}

func TestTransportNoHost(t *testing.T) {
	defer afterTest(t)
	tr := &Transport{}
	_, err := tr.RoundTrip(&Request{
		Header: make(Header),
		URL: &url.URL{
			Scheme: "http",
		},
	})
	want := "http: no Host in request URL"
	if got := fmt.Sprint(err); got != want {
		t.Errorf("error = %v; want %q", err, want)
	}
}

// Issue 13311
func TestTransportEmptyMethod(t *testing.T) {
	req, _ := NewRequest("GET", "http://foo.com/", nil)
	req.Method = ""                                 // docs say "For client requests an empty string means GET"
	got, err := httputil.DumpRequestOut(req, false) // DumpRequestOut uses Transport
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), "GET ") {
		t.Fatalf("expected substring 'GET '; got: %s", got)
	}
}

func TestTransportSocketLateBinding(t *testing.T) { run(t, testTransportSocketLateBinding) }
func testTransportSocketLateBinding(t *testing.T, mode testMode) {
	mux := NewServeMux()
	fooGate := make(chan bool, 1)
	mux.HandleFunc("/foo", func(w ResponseWriter, r *Request) {
		w.Header().Set("foo-ipport", r.RemoteAddr)
		w.(Flusher).Flush()
		<-fooGate
	})
	mux.HandleFunc("/bar", func(w ResponseWriter, r *Request) {
		w.Header().Set("bar-ipport", r.RemoteAddr)
	})
	ts := newClientServerTest(t, mode, mux).ts

	dialGate := make(chan bool, 1)
	dialing := make(chan bool)
	c := ts.Client()
	c.Transport.(*Transport).Dial = func(n, addr string) (net.Conn, error) {
		for {
			select {
			case ok := <-dialGate:
				if !ok {
					return nil, errors.New("manually closed")
				}
				return net.Dial(n, addr)
			case dialing <- true:
			}
		}
	}
	defer close(dialGate)

	dialGate <- true // only allow one dial
	fooRes, err := c.Get(ts.URL + "/foo")
	if err != nil {
		t.Fatal(err)
	}
	fooAddr := fooRes.Header.Get("foo-ipport")
	if fooAddr == "" {
		t.Fatal("No addr on /foo request")
	}

	fooDone := make(chan struct{})
	go func() {
		// We know that the foo Dial completed and reached the handler because we
		// read its header. Wait for the bar request to block in Dial, then
		// let the foo response finish so we can use its connection for /bar.

		if mode == http2Mode {
			// In HTTP/2 mode, the second Dial won't happen because the protocol
			// multiplexes the streams by default. Just sleep for an arbitrary time;
			// the test should pass regardless of how far the bar request gets by this
			// point.
			select {
			case <-dialing:
				t.Errorf("unexpected second Dial in HTTP/2 mode")
			case <-time.After(10 * time.Millisecond):
			}
		} else {
			<-dialing
		}
		fooGate <- true
		io.Copy(io.Discard, fooRes.Body)
		fooRes.Body.Close()
		close(fooDone)
	}()
	defer func() {
		<-fooDone
	}()

	barRes, err := c.Get(ts.URL + "/bar")
	if err != nil {
		t.Fatal(err)
	}
	barAddr := barRes.Header.Get("bar-ipport")
	if barAddr != fooAddr {
		t.Fatalf("/foo came from conn %q; /bar came from %q instead", fooAddr, barAddr)
	}
	barRes.Body.Close()
}

// Issue 2184
func TestTransportReading100Continue(t *testing.T) {
	defer afterTest(t)

	const numReqs = 5
	reqBody := func(n int) string { return fmt.Sprintf("request body %d", n) }
	reqID := func(n int) string { return fmt.Sprintf("REQ-ID-%d", n) }

	send100Response := func(w *io.PipeWriter, r *io.PipeReader) {
		defer w.Close()
		defer r.Close()
		br := bufio.NewReader(r)
		n := 0
		for {
			n++
			req, err := ReadRequest(br)
			if err == io.EOF {
				return
			}
			if err != nil {
				t.Error(err)
				return
			}
			slurp, err := io.ReadAll(req.Body)
			if err != nil {
				t.Errorf("Server request body slurp: %v", err)
				return
			}
			id := req.Header.Get("Request-Id")
			resCode := req.Header.Get("X-Want-Response-Code")
			if resCode == "" {
				resCode = "100 Continue"
				if string(slurp) != reqBody(n) {
					t.Errorf("Server got %q, %v; want %q", slurp, err, reqBody(n))
				}
			}
			body := fmt.Sprintf("Response number %d", n)
			v := []byte(strings.Replace(fmt.Sprintf(`HTTP/1.1 %s
Date: Thu, 28 Feb 2013 17:55:41 GMT

HTTP/1.1 200 OK
Content-Type: text/html
Echo-Request-Id: %s
Content-Length: %d

%s`, resCode, id, len(body), body), "\n", "\r\n", -1))
			w.Write(v)
			if id == reqID(numReqs) {
				return
			}
		}

	}

	tr := &Transport{
		Dial: func(n, addr string) (net.Conn, error) {
			sr, sw := io.Pipe() // server read/write
			cr, cw := io.Pipe() // client read/write
			conn := &rwTestConn{
				Reader: cr,
				Writer: sw,
				closeFunc: func() error {
					sw.Close()
					cw.Close()
					return nil
				},
			}
			go send100Response(cw, sr)
			return conn, nil
		},
		DisableKeepAlives: false,
	}
	defer tr.CloseIdleConnections()
	c := &Client{Transport: tr}

	testResponse := func(req *Request, name string, wantCode int) {
		t.Helper()
		res, err := c.Do(req)
		if err != nil {
			t.Fatalf("%s: Do: %v", name, err)
		}
		if res.StatusCode != wantCode {
			t.Fatalf("%s: Response Statuscode=%d; want %d", name, res.StatusCode, wantCode)
		}
		if id, idBack := req.Header.Get("Request-Id"), res.Header.Get("Echo-Request-Id"); id != "" && id != idBack {
			t.Errorf("%s: response id %q != request id %q", name, idBack, id)
		}
		_, err = io.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("%s: Slurp error: %v", name, err)
		}
	}

	// Few 100 responses, making sure we're not off-by-one.
	for i := 1; i <= numReqs; i++ {
		req, _ := NewRequest("POST", "http://dummy.tld/", strings.NewReader(reqBody(i)))
		req.Header.Set("Request-Id", reqID(i))
		testResponse(req, fmt.Sprintf("100, %d/%d", i, numReqs), 200)
	}
}

// Issue 17739: the HTTP client must ignore any unknown 1xx
// informational responses before the actual response.
func TestTransportIgnore1xxResponses(t *testing.T) {
	run(t, testTransportIgnore1xxResponses, []testMode{http1Mode})
}
func testTransportIgnore1xxResponses(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, buf, _ := w.(Hijacker).Hijack()
		buf.Write([]byte("HTTP/1.1 123 OneTwoThree\r\nFoo: bar\r\n\r\nHTTP/1.1 200 OK\r\nBar: baz\r\nContent-Length: 5\r\n\r\nHello"))
		buf.Flush()
		conn.Close()
	}))
	cst.tr.DisableKeepAlives = true // prevent log spam; our test server is hanging up anyway

	var got strings.Builder

	req, _ := NewRequest("GET", cst.ts.URL, nil)
	req = req.WithContext(httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
		Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
			fmt.Fprintf(&got, "1xx: code=%v, header=%v\n", code, header)
			return nil
		},
	}))
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	res.Write(&got)
	want := "1xx: code=123, header=map[Foo:[bar]]\nHTTP/1.1 200 OK\r\nContent-Length: 5\r\nBar: baz\r\n\r\nHello"
	if got.String() != want {
		t.Errorf(" got: %q\nwant: %q\n", got.String(), want)
	}
}

func TestTransportLimits1xxResponses(t *testing.T) { run(t, testTransportLimits1xxResponses) }
func testTransportLimits1xxResponses(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Add("X-Header", strings.Repeat("a", 100))
		for i := 0; i < 10; i++ {
			w.WriteHeader(123)
		}
		w.WriteHeader(204)
	}))
	cst.tr.DisableKeepAlives = true // prevent log spam; our test server is hanging up anyway
	cst.tr.MaxResponseHeaderBytes = 1000

	res, err := cst.c.Get(cst.ts.URL)
	if err == nil {
		res.Body.Close()
		t.Fatalf("RoundTrip succeeded; want error")
	}
	for _, want := range []string{
		"response headers exceeded",
		"too many 1xx",
		"header list too large",
	} {
		if strings.Contains(err.Error(), want) {
			return
		}
	}
	t.Errorf(`got error %q; want "response headers exceeded" or "too many 1xx"`, err)
}

func TestTransportDoesNotLimitDelivered1xxResponses(t *testing.T) {
	run(t, testTransportDoesNotLimitDelivered1xxResponses)
}
func testTransportDoesNotLimitDelivered1xxResponses(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("skip until x/net/http2 updated")
	}
	const num1xx = 10
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Add("X-Header", strings.Repeat("a", 100))
		for i := 0; i < 10; i++ {
			w.WriteHeader(123)
		}
		w.WriteHeader(204)
	}))
	cst.tr.DisableKeepAlives = true // prevent log spam; our test server is hanging up anyway
	cst.tr.MaxResponseHeaderBytes = 1000

	got1xx := 0
	ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
		Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
			got1xx++
			return nil
		},
	})
	req, _ := NewRequestWithContext(ctx, "GET", cst.ts.URL, nil)
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if got1xx != num1xx {
		t.Errorf("Got %v 1xx responses, want %x", got1xx, num1xx)
	}
}

// Issue 26161: the HTTP client must treat 101 responses
// as the final response.
func TestTransportTreat101Terminal(t *testing.T) {
	run(t, testTransportTreat101Terminal, []testMode{http1Mode})
}
func testTransportTreat101Terminal(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, buf, _ := w.(Hijacker).Hijack()
		buf.Write([]byte("HTTP/1.1 101 Switching Protocols\r\n\r\n"))
		buf.Write([]byte("HTTP/1.1 204 No Content\r\n\r\n"))
		buf.Flush()
		conn.Close()
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != StatusSwitchingProtocols {
		t.Errorf("StatusCode = %v; want 101 Switching Protocols", res.StatusCode)
	}
}

type proxyFromEnvTest struct {
	req string // URL to fetch; blank means "http://example.com"

	env      string // HTTP_PROXY
	httpsenv string // HTTPS_PROXY
	noenv    string // NO_PROXY
	reqmeth  string // REQUEST_METHOD

	want    string
	wanterr error
}

func (t proxyFromEnvTest) String() string {
	var buf strings.Builder
	space := func() {
		if buf.Len() > 0 {
			buf.WriteByte(' ')
		}
	}
	if t.env != "" {
		fmt.Fprintf(&buf, "http_proxy=%q", t.env)
	}
	if t.httpsenv != "" {
		space()
		fmt.Fprintf(&buf, "https_proxy=%q", t.httpsenv)
	}
	if t.noenv != "" {
		space()
		fmt.Fprintf(&buf, "no_proxy=%q", t.noenv)
	}
	if t.reqmeth != "" {
		space()
		fmt.Fprintf(&buf, "request_method=%q", t.reqmeth)
	}
	req := "http://example.com"
	if t.req != "" {
		req = t.req
	}
	space()
	fmt.Fprintf(&buf, "req=%q", req)
	return strings.TrimSpace(buf.String())
}

var proxyFromEnvTests = []proxyFromEnvTest{
	{env: "127.0.0.1:8080", want: "http://127.0.0.1:8080"},
	{env: "cache.corp.example.com:1234", want: "http://cache.corp.example.com:1234"},
	{env: "cache.corp.example.com", want: "http://cache.corp.example.com"},
	{env: "https://cache.corp.example.com", want: "https://cache.corp.example.com"},
	{env: "http://127.0.0.1:8080", want: "http://127.0.0.1:8080"},
	{env: "https://127.0.0.1:8080", want: "https://127.0.0.1:8080"},
	{env: "socks5://127.0.0.1", want: "socks5://127.0.0.1"},
	{env: "socks5h://127.0.0.1", want: "socks5h://127.0.0.1"},

	// Don't use secure for http
	{req: "http://insecure.tld/", env: "http.proxy.tld", httpsenv: "secure.proxy.tld", want: "http://http.proxy.tld"},
	// Use secure for https.
	{req: "https://secure.tld/", env: "http.proxy.tld", httpsenv: "secure.proxy.tld", want: "http://secure.proxy.tld"},
	{req: "https://secure.tld/", env: "http.proxy.tld", httpsenv: "https://secure.proxy.tld", want: "https://secure.proxy.tld"},

	// Issue 16405: don't use HTTP_PROXY in a CGI environment,
	// where HTTP_PROXY can be attacker-controlled.
	{env: "http://10.1.2.3:8080", reqmeth: "POST",
		want:    "<nil>",
		wanterr: errors.New("refusing to use HTTP_PROXY value in CGI environment; see golang.org/s/cgihttpproxy")},

	{want: "<nil>"},

	{noenv: "example.com", req: "http://example.com/", env: "proxy", want: "<nil>"},
	{noenv: ".example.com", req: "http://example.com/", env: "proxy", want: "http://proxy"},
	{noenv: "ample.com", req: "http://example.com/", env: "proxy", want: "http://proxy"},
	{noenv: "example.com", req: "http://foo.example.com/", env: "proxy", want: "<nil>"},
	{noenv: ".foo.com", req: "http://example.com/", env: "proxy", want: "http://proxy"},
}

func testProxyForRequest(t *testing.T, tt proxyFromEnvTest, proxyForRequest func(req *Request) (*url.URL, error)) {
	t.Helper()
	reqURL := tt.req
	if reqURL == "" {
		reqURL = "http://example.com"
	}
	req, _ := NewRequest("GET", reqURL, nil)
	url, err := proxyForRequest(req)
	if g, e := fmt.Sprintf("%v", err), fmt.Sprintf("%v", tt.wanterr); g != e {
		t.Errorf("%v: got error = %q, want %q", tt, g, e)
		return
	}
	if got := fmt.Sprintf("%s", url); got != tt.want {
		t.Errorf("%v: got URL = %q, want %q", tt, url, tt.want)
	}
}

func TestProxyFromEnvironment(t *testing.T) {
	ResetProxyEnv()
	defer ResetProxyEnv()
	for _, tt := range proxyFromEnvTests {
		testProxyForRequest(t, tt, func(req *Request) (*url.URL, error) {
			os.Setenv("HTTP_PROXY", tt.env)
			os.Setenv("HTTPS_PROXY", tt.httpsenv)
			os.Setenv("NO_PROXY", tt.noenv)
			os.Setenv("REQUEST_METHOD", tt.reqmeth)
			ResetCachedEnvironment()
			return ProxyFromEnvironment(req)
		})
	}
}

func TestProxyFromEnvironmentLowerCase(t *testing.T) {
	ResetProxyEnv()
	defer ResetProxyEnv()
	for _, tt := range proxyFromEnvTests {
		testProxyForRequest(t, tt, func(req *Request) (*url.URL, error) {
			os.Setenv("http_proxy", tt.env)
			os.Setenv("https_proxy", tt.httpsenv)
			os.Setenv("no_proxy", tt.noenv)
			os.Setenv("REQUEST_METHOD", tt.reqmeth)
			ResetCachedEnvironment()
			return ProxyFromEnvironment(req)
		})
	}
}

func TestIdleConnChannelLeak(t *testing.T) {
	run(t, testIdleConnChannelLeak, []testMode{http1Mode}, testNotParallel)
}
func testIdleConnChannelLeak(t *testing.T, mode testMode) {
	// Not parallel: uses global test hooks.
	var mu sync.Mutex
	var n int

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		mu.Lock()
		n++
		mu.Unlock()
	})).ts

	const nReqs = 5
	didRead := make(chan bool, nReqs)
	SetReadLoopBeforeNextReadHook(func() { didRead <- true })
	defer SetReadLoopBeforeNextReadHook(nil)

	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.Dial = func(netw, addr string) (net.Conn, error) {
		return net.Dial(netw, ts.Listener.Addr().String())
	}

	// First, without keep-alives.
	for _, disableKeep := range []bool{true, false} {
		tr.DisableKeepAlives = disableKeep
		for i := 0; i < nReqs; i++ {
			_, err := c.Get(fmt.Sprintf("http://foo-host-%d.tld/", i))
			if err != nil {
				t.Fatal(err)
			}
			// Note: no res.Body.Close is needed here, since the
			// response Content-Length is zero. Perhaps the test
			// should be more explicit and use a HEAD, but tests
			// elsewhere guarantee that zero byte responses generate
			// a "Content-Length: 0" instead of chunking.
		}

		// At this point, each of the 5 Transport.readLoop goroutines
		// are scheduling noting that there are no response bodies (see
		// earlier comment), and are then calling putIdleConn, which
		// decrements this count. Usually that happens quickly, which is
		// why this test has seemed to work for ages. But it's still
		// racey: we have wait for them to finish first. See Issue 10427
		for i := 0; i < nReqs; i++ {
			<-didRead
		}

		if got := tr.IdleConnWaitMapSizeForTesting(); got != 0 {
			t.Fatalf("for DisableKeepAlives = %v, map size = %d; want 0", disableKeep, got)
		}
	}
}

// Verify the status quo: that the Client.Post function coerces its
// body into a ReadCloser if it's a Closer, and that the Transport
// then closes it.
func TestTransportClosesRequestBody(t *testing.T) {
	run(t, testTransportClosesRequestBody, []testMode{http1Mode})
}
func testTransportClosesRequestBody(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.Copy(io.Discard, r.Body)
	})).ts

	c := ts.Client()

	closes := 0

	res, err := c.Post(ts.URL, "text/plain", countCloseReader{&closes, strings.NewReader("hello")})
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if closes != 1 {
		t.Errorf("closes = %d; want 1", closes)
	}
}

func TestTransportTLSHandshakeTimeout(t *testing.T) {
	defer afterTest(t)
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	ln := newLocalListener(t)
	defer ln.Close()
	testdonec := make(chan struct{})
	defer close(testdonec)

	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		<-testdonec
		c.Close()
	}()

	tr := &Transport{
		Dial: func(_, _ string) (net.Conn, error) {
			return net.Dial("tcp", ln.Addr().String())
		},
		TLSHandshakeTimeout: 250 * time.Millisecond,
	}
	cl := &Client{Transport: tr}
	_, err := cl.Get("https://dummy.tld/")
	if err == nil {
		t.Error("expected error")
		return
	}
	ue, ok := err.(*url.Error)
	if !ok {
		t.Errorf("expected url.Error; got %#v", err)
		return
	}
	ne, ok := ue.Err.(net.Error)
	if !ok {
		t.Errorf("expected net.Error; got %#v", err)
		return
	}
	if !ne.Timeout() {
		t.Errorf("expected timeout error; got %v", err)
	}
	if !strings.Contains(err.Error(), "handshake timeout") {
		t.Errorf("expected 'handshake timeout' in error; got %v", err)
	}
}

// Trying to repro golang.org/issue/3514
func TestTLSServerClosesConnection(t *testing.T) {
	run(t, testTLSServerClosesConnection, []testMode{https1Mode})
}
func testTLSServerClosesConnection(t *testing.T, mode testMode) {
	closedc := make(chan bool, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if strings.Contains(r.URL.Path, "/keep-alive-then-die") {
			conn, _, _ := w.(Hijacker).Hijack()
			conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nfoo"))
			conn.Close()
			closedc <- true
			return
		}
		fmt.Fprintf(w, "hello")
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)

	var nSuccess = 0
	var errs []error
	const trials = 20
	for i := 0; i < trials; i++ {
		tr.CloseIdleConnections()
		res, err := c.Get(ts.URL + "/keep-alive-then-die")
		if err != nil {
			t.Fatal(err)
		}
		<-closedc
		slurp, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(slurp) != "foo" {
			t.Errorf("Got %q, want foo", slurp)
		}

		// Now try again and see if we successfully
		// pick a new connection.
		res, err = c.Get(ts.URL + "/")
		if err != nil {
			errs = append(errs, err)
			continue
		}
		slurp, err = io.ReadAll(res.Body)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		nSuccess++
	}
	if nSuccess > 0 {
		t.Logf("successes = %d of %d", nSuccess, trials)
	} else {
		t.Errorf("All runs failed:")
	}
	for _, err := range errs {
		t.Logf("  err: %v", err)
	}
}

// byteFromChanReader is an io.Reader that reads a single byte at a
// time from the channel. When the channel is closed, the reader
// returns io.EOF.
type byteFromChanReader chan byte

func (c byteFromChanReader) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return
	}
	b, ok := <-c
	if !ok {
		return 0, io.EOF
	}
	p[0] = b
	return 1, nil
}

// Verifies that the Transport doesn't reuse a connection in the case
// where the server replies before the request has been fully
// written. We still honor that reply (see TestIssue3595), but don't
// send future requests on the connection because it's then in a
// questionable state.
// golang.org/issue/7569
func TestTransportNoReuseAfterEarlyResponse(t *testing.T) {
	run(t, testTransportNoReuseAfterEarlyResponse, []testMode{http1Mode}, testNotParallel)
}
func testTransportNoReuseAfterEarlyResponse(t *testing.T, mode testMode) {
	defer func(d time.Duration) {
		*MaxWriteWaitBeforeConnReuse = d
	}(*MaxWriteWaitBeforeConnReuse)
	*MaxWriteWaitBeforeConnReuse = 10 * time.Millisecond
	var sconn struct {
		sync.Mutex
		c net.Conn
	}
	var getOkay bool
	var copying sync.WaitGroup
	closeConn := func() {
		sconn.Lock()
		defer sconn.Unlock()
		if sconn.c != nil {
			sconn.c.Close()
			sconn.c = nil
			if !getOkay {
				t.Logf("Closed server connection")
			}
		}
	}
	defer func() {
		closeConn()
		copying.Wait()
	}()

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method == "GET" {
			io.WriteString(w, "bar")
			return
		}
		conn, _, _ := w.(Hijacker).Hijack()
		sconn.Lock()
		sconn.c = conn
		sconn.Unlock()
		conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nfoo")) // keep-alive

		copying.Add(1)
		go func() {
			io.Copy(io.Discard, conn)
			copying.Done()
		}()
	})).ts
	c := ts.Client()

	const bodySize = 256 << 10
	finalBit := make(byteFromChanReader, 1)
	req, _ := NewRequest("POST", ts.URL, io.MultiReader(io.LimitReader(neverEnding('x'), bodySize-1), finalBit))
	req.ContentLength = bodySize
	res, err := c.Do(req)
	if err := wantBody(res, err, "foo"); err != nil {
		t.Errorf("POST response: %v", err)
	}

	res, err = c.Get(ts.URL)
	if err := wantBody(res, err, "bar"); err != nil {
		t.Errorf("GET response: %v", err)
		return
	}
	getOkay = true  // suppress test noise
	finalBit <- 'x' // unblock the writeloop of the first Post
	close(finalBit)
}

// Tests that we don't leak Transport persistConn.readLoop goroutines
// when a server hangs up immediately after saying it would keep-alive.
func TestTransportIssue10457(t *testing.T) { run(t, testTransportIssue10457, []testMode{http1Mode}) }
func testTransportIssue10457(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// Send a response with no body, keep-alive
		// (implicit), and then lie and immediately close the
		// connection. This forces the Transport's readLoop to
		// immediately Peek an io.EOF and get to the point
		// that used to hang.
		conn, _, _ := w.(Hijacker).Hijack()
		conn.Write([]byte("HTTP/1.1 200 OK\r\nFoo: Bar\r\nContent-Length: 0\r\n\r\n")) // keep-alive
		conn.Close()
	})).ts
	c := ts.Client()

	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer res.Body.Close()

	// Just a sanity check that we at least get the response. The real
	// test here is that the "defer afterTest" above doesn't find any
	// leaked goroutines.
	if got, want := res.Header.Get("Foo"), "Bar"; got != want {
		t.Errorf("Foo header = %q; want %q", got, want)
	}
}

type closerFunc func() error

func (f closerFunc) Close() error { return f() }

type writerFuncConn struct {
	net.Conn
	write func(p []byte) (n int, err error)
}

func (c writerFuncConn) Write(p []byte) (n int, err error) { return c.write(p) }

// Issues 4677, 18241, and 17844. If we try to reuse a connection that the
// server is in the process of closing, we may end up successfully writing out
// our request (or a portion of our request) only to find a connection error
// when we try to read from (or finish writing to) the socket.
//
// NOTE: we resend a request only if:
//   - we reused a keep-alive connection
//   - we haven't yet received any header data
//   - either we wrote no bytes to the server, or the request is idempotent
//
// This automatically prevents an infinite resend loop because we'll run out of
// the cached keep-alive connections eventually.
func TestRetryRequestsOnError(t *testing.T) {
	run(t, testRetryRequestsOnError, testNotParallel, []testMode{http1Mode})
}
func testRetryRequestsOnError(t *testing.T, mode testMode) {
	newRequest := func(method, urlStr string, body io.Reader) *Request {
		req, err := NewRequest(method, urlStr, body)
		if err != nil {
			t.Fatal(err)
		}
		return req
	}

	testCases := []struct {
		name       string
		failureN   int
		failureErr error
		// Note that we can't just re-use the Request object across calls to c.Do
		// because we need to rewind Body between calls.  (GetBody is only used to
		// rewind Body on failure and redirects, not just because it's done.)
		req       func() *Request
		reqString string
	}{
		{
			name: "IdempotentNoBodySomeWritten",
			// Believe that we've written some bytes to the server, so we know we're
			// not just in the "retry when no bytes sent" case".
			failureN: 1,
			// Use the specific error that shouldRetryRequest looks for with idempotent requests.
			failureErr: ExportErrServerClosedIdle,
			req: func() *Request {
				return newRequest("GET", "http://fake.golang", nil)
			},
			reqString: `GET / HTTP/1.1\r\nHost: fake.golang\r\nUser-Agent: Go-http-client/1.1\r\nAccept-Encoding: gzip\r\n\r\n`,
		},
		{
			name: "IdempotentGetBodySomeWritten",
			// Believe that we've written some bytes to the server, so we know we're
			// not just in the "retry when no bytes sent" case".
			failureN: 1,
			// Use the specific error that shouldRetryRequest looks for with idempotent requests.
			failureErr: ExportErrServerClosedIdle,
			req: func() *Request {
				return newRequest("GET", "http://fake.golang", strings.NewReader("foo\n"))
			},
			reqString: `GET / HTTP/1.1\r\nHost: fake.golang\r\nUser-Agent: Go-http-client/1.1\r\nContent-Length: 4\r\nAccept-Encoding: gzip\r\n\r\nfoo\n`,
		},
		{
			name: "NothingWrittenNoBody",
			// It's key that we return 0 here -- that's what enables Transport to know
			// that nothing was written, even though this is a non-idempotent request.
			failureN:   0,
			failureErr: errors.New("second write fails"),
			req: func() *Request {
				return newRequest("DELETE", "http://fake.golang", nil)
			},
			reqString: `DELETE / HTTP/1.1\r\nHost: fake.golang\r\nUser-Agent: Go-http-client/1.1\r\nAccept-Encoding: gzip\r\n\r\n`,
		},
		{
			name: "NothingWrittenGetBody",
			// It's key that we return 0 here -- that's what enables Transport to know
			// that nothing was written, even though this is a non-idempotent request.
			failureN:   0,
			failureErr: errors.New("second write fails"),
			// Note that NewRequest will set up GetBody for strings.Reader, which is
			// required for the retry to occur
			req: func() *Request {
				return newRequest("POST", "http://fake.golang", strings.NewReader("foo\n"))
			},
			reqString: `POST / HTTP/1.1\r\nHost: fake.golang\r\nUser-Agent: Go-http-client/1.1\r\nContent-Length: 4\r\nAccept-Encoding: gzip\r\n\r\nfoo\n`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				mu     sync.Mutex
				logbuf strings.Builder
			)
			logf := func(format string, args ...any) {
				mu.Lock()
				defer mu.Unlock()
				fmt.Fprintf(&logbuf, format, args...)
				logbuf.WriteByte('\n')
			}

			ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
				logf("Handler")
				w.Header().Set("X-Status", "ok")
			})).ts

			var writeNumAtomic int32
			c := ts.Client()
			c.Transport.(*Transport).Dial = func(network, addr string) (net.Conn, error) {
				logf("Dial")
				c, err := net.Dial(network, ts.Listener.Addr().String())
				if err != nil {
					logf("Dial error: %v", err)
					return nil, err
				}
				return &writerFuncConn{
					Conn: c,
					write: func(p []byte) (n int, err error) {
						if atomic.AddInt32(&writeNumAtomic, 1) == 2 {
							logf("intentional write failure")
							return tc.failureN, tc.failureErr
						}
						logf("Write(%q)", p)
						return c.Write(p)
					},
				}, nil
			}

			SetRoundTripRetried(func() {
				logf("Retried.")
			})
			defer SetRoundTripRetried(nil)

			for i := 0; i < 3; i++ {
				t0 := time.Now()
				req := tc.req()
				res, err := c.Do(req)
				if err != nil {
					if time.Since(t0) < *MaxWriteWaitBeforeConnReuse/2 {
						mu.Lock()
						got := logbuf.String()
						mu.Unlock()
						t.Fatalf("i=%d: Do = %v; log:\n%s", i, err, got)
					}
					t.Skipf("connection likely wasn't recycled within %d, interfering with actual test; skipping", *MaxWriteWaitBeforeConnReuse)
				}
				res.Body.Close()
				if res.Request != req {
					t.Errorf("Response.Request != original request; want identical Request")
				}
			}

			mu.Lock()
			got := logbuf.String()
			mu.Unlock()
			want := fmt.Sprintf(`Dial
Write("%s")
Handler
intentional write failure
Retried.
Dial
Write("%s")
Handler
Write("%s")
Handler
`, tc.reqString, tc.reqString, tc.reqString)
			if got != want {
				t.Errorf("Log of events differs. Got:\n%s\nWant:\n%s", got, want)
			}
		})
	}
}

// Issue 6981
func TestTransportClosesBodyOnError(t *testing.T) { run(t, testTransportClosesBodyOnError) }
func testTransportClosesBodyOnError(t *testing.T, mode testMode) {
	readBody := make(chan error, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := io.ReadAll(r.Body)
		readBody <- err
	})).ts
	c := ts.Client()
	fakeErr := errors.New("fake error")
	didClose := make(chan bool, 1)
	req, _ := NewRequest("POST", ts.URL, struct {
		io.Reader
		io.Closer
	}{
		io.MultiReader(io.LimitReader(neverEnding('x'), 1<<20), iotest.ErrReader(fakeErr)),
		closerFunc(func() error {
			select {
			case didClose <- true:
			default:
			}
			return nil
		}),
	})
	res, err := c.Do(req)
	if res != nil {
		defer res.Body.Close()
	}
	if err == nil || !strings.Contains(err.Error(), fakeErr.Error()) {
		t.Fatalf("Do error = %v; want something containing %q", err, fakeErr.Error())
	}
	if err := <-readBody; err == nil {
		t.Errorf("Unexpected success reading request body from handler; want 'unexpected EOF reading trailer'")
	}
	select {
	case <-didClose:
	default:
		t.Errorf("didn't see Body.Close")
	}
}

func TestTransportDialTLS(t *testing.T) {
	run(t, testTransportDialTLS, []testMode{https1Mode, http2Mode})
}
func testTransportDialTLS(t *testing.T, mode testMode) {
	var mu sync.Mutex // guards following
	var gotReq, didDial bool

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		mu.Lock()
		gotReq = true
		mu.Unlock()
	})).ts
	c := ts.Client()
	c.Transport.(*Transport).DialTLS = func(netw, addr string) (net.Conn, error) {
		mu.Lock()
		didDial = true
		mu.Unlock()
		c, err := tls.Dial(netw, addr, c.Transport.(*Transport).TLSClientConfig)
		if err != nil {
			return nil, err
		}
		return c, c.Handshake()
	}

	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	mu.Lock()
	if !gotReq {
		t.Error("didn't get request")
	}
	if !didDial {
		t.Error("didn't use dial hook")
	}
}

func TestTransportDialContext(t *testing.T) { run(t, testTransportDialContext) }
func testTransportDialContext(t *testing.T, mode testMode) {
	ctxKey := "some-key"
	ctxValue := "some-value"
	var (
		mu          sync.Mutex // guards following
		gotReq      bool
		gotCtxValue any
	)

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		mu.Lock()
		gotReq = true
		mu.Unlock()
	})).ts
	c := ts.Client()
	c.Transport.(*Transport).DialContext = func(ctx context.Context, netw, addr string) (net.Conn, error) {
		mu.Lock()
		gotCtxValue = ctx.Value(ctxKey)
		mu.Unlock()
		return net.Dial(netw, addr)
	}

	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), ctxKey, ctxValue)
	res, err := c.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	mu.Lock()
	if !gotReq {
		t.Error("didn't get request")
	}
	if got, want := gotCtxValue, ctxValue; got != want {
		t.Errorf("got context with value %v, want %v", got, want)
	}
}

func TestTransportDialTLSContext(t *testing.T) {
	run(t, testTransportDialTLSContext, []testMode{https1Mode, http2Mode})
}
func testTransportDialTLSContext(t *testing.T, mode testMode) {
	ctxKey := "some-key"
	ctxValue := "some-value"
	var (
		mu          sync.Mutex // guards following
		gotReq      bool
		gotCtxValue any
	)

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		mu.Lock()
		gotReq = true
		mu.Unlock()
	})).ts
	c := ts.Client()
	c.Transport.(*Transport).DialTLSContext = func(ctx context.Context, netw, addr string) (net.Conn, error) {
		mu.Lock()
		gotCtxValue = ctx.Value(ctxKey)
		mu.Unlock()
		c, err := tls.Dial(netw, addr, c.Transport.(*Transport).TLSClientConfig)
		if err != nil {
			return nil, err
		}
		return c, c.HandshakeContext(ctx)
	}

	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), ctxKey, ctxValue)
	res, err := c.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	mu.Lock()
	if !gotReq {
		t.Error("didn't get request")
	}
	if got, want := gotCtxValue, ctxValue; got != want {
		t.Errorf("got context with value %v, want %v", got, want)
	}
}

// Test for issue 8755
// Ensure that if a proxy returns an error, it is exposed by RoundTrip
func TestRoundTripReturnsProxyError(t *testing.T) {
	badProxy := func(*Request) (*url.URL, error) {
		return nil, errors.New("errorMessage")
	}

	tr := &Transport{Proxy: badProxy}

	req, _ := NewRequest("GET", "http://example.com", nil)

	_, err := tr.RoundTrip(req)

	if err == nil {
		t.Error("Expected proxy error to be returned by RoundTrip")
	}
}

// tests that putting an idle conn after a call to CloseIdleConns does return it
func TestTransportCloseIdleConnsThenReturn(t *testing.T) {
	tr := &Transport{}
	wantIdle := func(when string, n int) bool {
		got := tr.IdleConnCountForTesting("http", "example.com") // key used by PutIdleTestConn
		if got == n {
			return true
		}
		t.Errorf("%s: idle conns = %d; want %d", when, got, n)
		return false
	}
	wantIdle("start", 0)
	if !tr.PutIdleTestConn("http", "example.com") {
		t.Fatal("put failed")
	}
	if !tr.PutIdleTestConn("http", "example.com") {
		t.Fatal("second put failed")
	}
	wantIdle("after put", 2)
	tr.CloseIdleConnections()
	if !tr.IsIdleForTesting() {
		t.Error("should be idle after CloseIdleConnections")
	}
	wantIdle("after close idle", 0)
	if tr.PutIdleTestConn("http", "example.com") {
		t.Fatal("put didn't fail")
	}
	wantIdle("after second put", 0)

	tr.QueueForIdleConnForTesting() // should toggle the transport out of idle mode
	if tr.IsIdleForTesting() {
		t.Error("shouldn't be idle after QueueForIdleConnForTesting")
	}
	if !tr.PutIdleTestConn("http", "example.com") {
		t.Fatal("after re-activation")
	}
	wantIdle("after final put", 1)
}

// Test for issue 34282
// Ensure that getConn doesn't call the GotConn trace hook on an HTTP/2 idle conn
func TestTransportTraceGotConnH2IdleConns(t *testing.T) {
	tr := &Transport{}
	wantIdle := func(when string, n int) bool {
		got := tr.IdleConnCountForTesting("https", "example.com:443") // key used by PutIdleTestConnH2
		if got == n {
			return true
		}
		t.Errorf("%s: idle conns = %d; want %d", when, got, n)
		return false
	}
	wantIdle("start", 0)
	alt := funcRoundTripper(func() {})
	if !tr.PutIdleTestConnH2("https", "example.com:443", alt) {
		t.Fatal("put failed")
	}
	wantIdle("after put", 1)
	ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
		GotConn: func(httptrace.GotConnInfo) {
			// tr.getConn should leave it for the HTTP/2 alt to call GotConn.
			t.Error("GotConn called")
		},
	})
	req, _ := NewRequestWithContext(ctx, MethodGet, "https://example.com", nil)
	_, err := tr.RoundTrip(req)
	if err != errFakeRoundTrip {
		t.Errorf("got error: %v; want %q", err, errFakeRoundTrip)
	}
	wantIdle("after round trip", 1)
}

// https://go.dev/issue/70515
//
// When the first request on a new connection fails, we do not retry the request.
// If the first request on a connection races with IdleConnTimeout,
// we should not fail the request.
func TestTransportIdleConnRacesRequest(t *testing.T) {
	// Use unencrypted HTTP/2, since the *tls.Conn interfers with our ability to
	// block the connection closing.
	runSynctest(t, testTransportIdleConnRacesRequest, []testMode{http1Mode, http2UnencryptedMode})
}
func testTransportIdleConnRacesRequest(t *testing.T, mode testMode) {
	if mode == http2UnencryptedMode {
		t.Skip("remove skip when #70515 is fixed")
	}
	timeout := 1 * time.Millisecond
	trFunc := func(tr *Transport) {
		tr.IdleConnTimeout = timeout
	}
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
	}), trFunc, optFakeNet)
	cst.li.trackConns = true

	// We want to put a connection into the pool which has never had a request made on it.
	//
	// Make a request and cancel it before the dial completes.
	// Then complete the dial.
	dialc := make(chan struct{})
	cst.li.onDial = func() {
		<-dialc
	}
	closec := make(chan struct{})
	cst.li.onClose = func(*fakeNetConn) {
		<-closec
	}
	ctx, cancel := context.WithCancel(context.Background())
	req1c := make(chan error)
	go func() {
		req, _ := NewRequestWithContext(ctx, "GET", cst.ts.URL, nil)
		resp, err := cst.c.Do(req)
		if err == nil {
			resp.Body.Close()
		}
		req1c <- err
	}()
	// Wait for the connection attempt to start.
	synctest.Wait()
	// Cancel the request.
	cancel()
	synctest.Wait()
	if err := <-req1c; err == nil {
		t.Fatal("expected request to fail, but it succeeded")
	}
	// Unblock the dial, placing a new, unused connection into the Transport's pool.
	close(dialc)

	// We want IdleConnTimeout to race with a new request.
	//
	// There's no perfect way to do this, but the following exercises the bug in #70515:
	// Block net.Conn.Close, wait until IdleConnTimeout occurs, and make a request while
	// the connection close is still blocked.
	//
	// First: Wait for IdleConnTimeout. The net.Conn.Close blocks.
	synctest.Wait()
	time.Sleep(timeout)
	synctest.Wait()
	// Make a request, which will use a new connection (since the existing one is closing).
	req2c := make(chan error)
	go func() {
		resp, err := cst.c.Get(cst.ts.URL)
		if err == nil {
			resp.Body.Close()
		}
		req2c <- err
	}()
	// Don't synctest.Wait here: The HTTP/1 transport closes the idle conn
	// with a mutex held, and we'll end up in a deadlock.
	close(closec)
	if err := <-req2c; err != nil {
		t.Fatalf("Get: %v", err)
	}
}

func TestTransportRemovesConnsAfterIdle(t *testing.T) {
	runSynctest(t, testTransportRemovesConnsAfterIdle)
}
func testTransportRemovesConnsAfterIdle(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	timeout := 1 * time.Second
	trFunc := func(tr *Transport) {
		tr.MaxConnsPerHost = 1
		tr.MaxIdleConnsPerHost = 1
		tr.IdleConnTimeout = timeout
	}
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Addr", r.RemoteAddr)
	}), trFunc, optFakeNet)

	// makeRequest returns the local address a request was made from
	// (unique for each connection).
	makeRequest := func() string {
		resp, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatalf("got error: %s", err)
		}
		resp.Body.Close()
		return resp.Header.Get("X-Addr")
	}

	addr1 := makeRequest()

	time.Sleep(timeout / 2)
	synctest.Wait()
	addr2 := makeRequest()
	if addr1 != addr2 {
		t.Fatalf("two requests made within IdleConnTimeout should have used the same conn, but used %v, %v", addr1, addr2)
	}

	time.Sleep(timeout)
	synctest.Wait()
	addr3 := makeRequest()
	if addr1 == addr3 {
		t.Fatalf("two requests made more than IdleConnTimeout apart should have used different conns, but used %v, %v", addr1, addr3)
	}
}

func TestTransportRemovesConnsAfterBroken(t *testing.T) {
	runSynctest(t, testTransportRemovesConnsAfterBroken)
}
func testTransportRemovesConnsAfterBroken(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	trFunc := func(tr *Transport) {
		tr.MaxConnsPerHost = 1
		tr.MaxIdleConnsPerHost = 1
	}
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Addr", r.RemoteAddr)
	}), trFunc, optFakeNet)
	cst.li.trackConns = true

	// makeRequest returns the local address a request was made from
	// (unique for each connection).
	makeRequest := func() string {
		resp, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatalf("got error: %s", err)
		}
		resp.Body.Close()
		return resp.Header.Get("X-Addr")
	}

	addr1 := makeRequest()
	addr2 := makeRequest()
	if addr1 != addr2 {
		t.Fatalf("successive requests should have used the same conn, but used %v, %v", addr1, addr2)
	}

	// The connection breaks.
	synctest.Wait()
	cst.li.conns[0].peer.Close()
	synctest.Wait()
	addr3 := makeRequest()
	if addr1 == addr3 {
		t.Fatalf("successive requests made with conn broken between should have used different conns, but used %v, %v", addr1, addr3)
	}
}

// This tests that a client requesting a content range won't also
// implicitly ask for gzip support. If they want that, they need to do it
// on their own.
// golang.org/issue/8923
func TestTransportRangeAndGzip(t *testing.T) { run(t, testTransportRangeAndGzip) }
func testTransportRangeAndGzip(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			t.Error("Transport advertised gzip support in the Accept header")
		}
		if r.Header.Get("Range") == "" {
			t.Error("no Range in request")
		}
	})).ts
	c := ts.Client()

	req, _ := NewRequest("GET", ts.URL, nil)
	req.Header.Set("Range", "bytes=7-11")
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

// Test for issue 10474
func TestTransportResponseCancelRace(t *testing.T) { run(t, testTransportResponseCancelRace) }
func testTransportResponseCancelRace(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// important that this response has a body.
		var b [1024]byte
		w.Write(b[:])
	})).ts
	tr := ts.Client().Transport.(*Transport)

	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	// If we do an early close, Transport just throws the connection away and
	// doesn't reuse it. In order to trigger the bug, it has to reuse the connection
	// so read the body
	if _, err := io.Copy(io.Discard, res.Body); err != nil {
		t.Fatal(err)
	}

	req2, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	tr.CancelRequest(req)
	res, err = tr.RoundTrip(req2)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

// Test for issue 19248: Content-Encoding's value is case insensitive.
func TestTransportContentEncodingCaseInsensitive(t *testing.T) {
	run(t, testTransportContentEncodingCaseInsensitive)
}
func testTransportContentEncodingCaseInsensitive(t *testing.T, mode testMode) {
	for _, ce := range []string{"gzip", "GZIP"} {
		ce := ce
		t.Run(ce, func(t *testing.T) {
			const encodedString = "Hello Gopher"
			ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
				w.Header().Set("Content-Encoding", ce)
				gz := gzip.NewWriter(w)
				gz.Write([]byte(encodedString))
				gz.Close()
			})).ts

			res, err := ts.Client().Get(ts.URL)
			if err != nil {
				t.Fatal(err)
			}

			body, err := io.ReadAll(res.Body)
			res.Body.Close()
			if err != nil {
				t.Fatal(err)
			}

			if string(body) != encodedString {
				t.Fatalf("Expected body %q, got: %q\n", encodedString, string(body))
			}
		})
	}
}

// https://go.dev/issue/49621
func TestConnClosedBeforeRequestIsWritten(t *testing.T) {
	run(t, testConnClosedBeforeRequestIsWritten, testNotParallel, []testMode{http1Mode})
}
func testConnClosedBeforeRequestIsWritten(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {}),
		func(tr *Transport) {
			tr.DialContext = func(_ context.Context, network, addr string) (net.Conn, error) {
				// Connection immediately returns errors.
				return &funcConn{
					read: func([]byte) (int, error) {
						return 0, errors.New("error")
					},
					write: func([]byte) (int, error) {
						return 0, errors.New("error")
					},
				}, nil
			}
		},
	).ts
	// Set a short delay in RoundTrip to give the persistConn time to notice
	// the connection is broken. We want to exercise the path where writeLoop exits
	// before it reads the request to send. If this delay is too short, we may instead
	// exercise the path where writeLoop accepts the request and then fails to write it.
	// That's fine, so long as we get the desired path often enough.
	SetEnterRoundTripHook(func() {
		time.Sleep(1 * time.Millisecond)
	})
	defer SetEnterRoundTripHook(nil)
	var closes int
	_, err := ts.Client().Post(ts.URL, "text/plain", countCloseReader{&closes, strings.NewReader("hello")})
	if err == nil {
		t.Fatalf("expected request to fail, but it did not")
	}
	if closes != 1 {
		t.Errorf("after RoundTrip, request body was closed %v times; want 1", closes)
	}
}

// logWritesConn is a net.Conn that logs each Write call to writes
// and then proxies to w.
// It proxies Read calls to a reader it receives from rch.
type logWritesConn struct {
	net.Conn // nil. crash on use.

	w io.Writer

	rch <-chan io.Reader
	r   io.Reader // nil until received by rch

	mu     sync.Mutex
	writes []string
}

func (c *logWritesConn) Write(p []byte) (n int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.writes = append(c.writes, string(p))
	return c.w.Write(p)
}

func (c *logWritesConn) Read(p []byte) (n int, err error) {
	if c.r == nil {
		c.r = <-c.rch
	}
	return c.r.Read(p)
}

func (c *logWritesConn) Close() error { return nil }

// Issue 6574
func TestTransportFlushesBodyChunks(t *testing.T) {
	defer afterTest(t)
	resBody := make(chan io.Reader, 1)
	connr, connw := io.Pipe() // connection pipe pair
	lw := &logWritesConn{
		rch: resBody,
		w:   connw,
	}
	tr := &Transport{
		Dial: func(network, addr string) (net.Conn, error) {
			return lw, nil
		},
	}
	bodyr, bodyw := io.Pipe() // body pipe pair
	go func() {
		defer bodyw.Close()
		for i := 0; i < 3; i++ {
			fmt.Fprintf(bodyw, "num%d\n", i)
		}
	}()
	resc := make(chan *Response)
	go func() {
		req, _ := NewRequest("POST", "http://localhost:8080", bodyr)
		req.Header.Set("User-Agent", "x") // known value for test
		res, err := tr.RoundTrip(req)
		if err != nil {
			t.Errorf("RoundTrip: %v", err)
			close(resc)
			return
		}
		resc <- res

	}()
	// Fully consume the request before checking the Write log vs. want.
	req, err := ReadRequest(bufio.NewReader(connr))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, req.Body)

	// Unblock the transport's roundTrip goroutine.
	resBody <- strings.NewReader("HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
	res, ok := <-resc
	if !ok {
		return
	}
	defer res.Body.Close()

	want := []string{
		"POST / HTTP/1.1\r\nHost: localhost:8080\r\nUser-Agent: x\r\nTransfer-Encoding: chunked\r\nAccept-Encoding: gzip\r\n\r\n",
		"5\r\nnum0\n\r\n",
		"5\r\nnum1\n\r\n",
		"5\r\nnum2\n\r\n",
		"0\r\n\r\n",
	}
	if !slices.Equal(lw.writes, want) {
		t.Errorf("Writes differed.\n Got: %q\nWant: %q\n", lw.writes, want)
	}
}

// Issue 22088: flush Transport request headers if we're not sure the body won't block on read.
func TestTransportFlushesRequestHeader(t *testing.T) { run(t, testTransportFlushesRequestHeader) }
func testTransportFlushesRequestHeader(t *testing.T, mode testMode) {
	gotReq := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		close(gotReq)
	}))

	pr, pw := io.Pipe()
	req, err := NewRequest("POST", cst.ts.URL, pr)
	if err != nil {
		t.Fatal(err)
	}
	gotRes := make(chan struct{})
	go func() {
		defer close(gotRes)
		res, err := cst.tr.RoundTrip(req)
		if err != nil {
			t.Error(err)
			return
		}
		res.Body.Close()
	}()

	<-gotReq
	pw.Close()
	<-gotRes
}

type wgReadCloser struct {
	io.Reader
	wg     *sync.WaitGroup
	closed bool
}

func (c *wgReadCloser) Close() error {
	if c.closed {
		return net.ErrClosed
	}
	c.closed = true
	c.wg.Done()
	return nil
}

// Issue 11745.
func TestTransportPrefersResponseOverWriteError(t *testing.T) {
	// Not parallel: modifies the global rstAvoidanceDelay.
	run(t, testTransportPrefersResponseOverWriteError, testNotParallel)
}
func testTransportPrefersResponseOverWriteError(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	runTimeSensitiveTest(t, []time.Duration{
		1 * time.Millisecond,
		5 * time.Millisecond,
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		500 * time.Millisecond,
		time.Second,
		5 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		SetRSTAvoidanceDelay(t, timeout)
		t.Logf("set RST avoidance delay to %v", timeout)

		const contentLengthLimit = 1024 * 1024 // 1MB
		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			if r.ContentLength >= contentLengthLimit {
				w.WriteHeader(StatusBadRequest)
				r.Body.Close()
				return
			}
			w.WriteHeader(StatusOK)
		}))
		// We need to close cst explicitly here so that in-flight server
		// requests don't race with the call to SetRSTAvoidanceDelay for a retry.
		defer cst.close()
		ts := cst.ts
		c := ts.Client()

		count := 100

		bigBody := strings.Repeat("a", contentLengthLimit*2)
		var wg sync.WaitGroup
		defer wg.Wait()
		getBody := func() (io.ReadCloser, error) {
			wg.Add(1)
			body := &wgReadCloser{
				Reader: strings.NewReader(bigBody),
				wg:     &wg,
			}
			return body, nil
		}

		for i := 0; i < count; i++ {
			reqBody, _ := getBody()
			req, err := NewRequest("PUT", ts.URL, reqBody)
			if err != nil {
				reqBody.Close()
				t.Fatal(err)
			}
			req.ContentLength = int64(len(bigBody))
			req.GetBody = getBody

			resp, err := c.Do(req)
			if err != nil {
				return fmt.Errorf("Do %d: %v", i, err)
			} else {
				resp.Body.Close()
				if resp.StatusCode != 400 {
					t.Errorf("Expected status code 400, got %v", resp.Status)
				}
			}
		}
		return nil
	})
}

func TestTransportAutomaticHTTP2(t *testing.T) {
	testTransportAutoHTTP(t, &Transport{}, true)
}

func TestTransportAutomaticHTTP2_DialerAndTLSConfigSupportsHTTP2AndTLSConfig(t *testing.T) {
	testTransportAutoHTTP(t, &Transport{
		ForceAttemptHTTP2: true,
		TLSClientConfig:   new(tls.Config),
	}, true)
}

// golang.org/issue/14391: also check DefaultTransport
func TestTransportAutomaticHTTP2_DefaultTransport(t *testing.T) {
	testTransportAutoHTTP(t, DefaultTransport.(*Transport), true)
}

func TestTransportAutomaticHTTP2_TLSNextProto(t *testing.T) {
	testTransportAutoHTTP(t, &Transport{
		TLSNextProto: make(map[string]func(string, *tls.Conn) RoundTripper),
	}, false)
}

func TestTransportAutomaticHTTP2_TLSConfig(t *testing.T) {
	testTransportAutoHTTP(t, &Transport{
		TLSClientConfig: new(tls.Config),
	}, false)
}

func TestTransportAutomaticHTTP2_ExpectContinueTimeout(t *testing.T) {
	testTransportAutoHTTP(t, &Transport{
		ExpectContinueTimeout: 1 * time.Second,
	}, true)
}

func TestTransportAutomaticHTTP2_Dial(t *testing.T) {
	var d net.Dialer
	testTransportAutoHTTP(t, &Transport{
		Dial: d.Dial,
	}, false)
}

func TestTransportAutomaticHTTP2_DialContext(t *testing.T) {
	var d net.Dialer
	testTransportAutoHTTP(t, &Transport{
		DialContext: d.DialContext,
	}, false)
}

func TestTransportAutomaticHTTP2_DialTLS(t *testing.T) {
	testTransportAutoHTTP(t, &Transport{
		DialTLS: func(network, addr string) (net.Conn, error) {
			panic("unused")
		},
	}, false)
}

func testTransportAutoHTTP(t *testing.T, tr *Transport, wantH2 bool) {
	CondSkipHTTP2(t)
	_, err := tr.RoundTrip(new(Request))
	if err == nil {
		t.Error("expected error from RoundTrip")
	}
	if reg := tr.TLSNextProto["h2"] != nil; reg != wantH2 {
		t.Errorf("HTTP/2 registered = %v; want %v", reg, wantH2)
	}
}

// Issue 13633: there was a race where we returned bodyless responses
// to callers before recycling the persistent connection, which meant
// a client doing two subsequent requests could end up on different
// connections. It's somewhat harmless but enough tests assume it's
// not true in order to test other things that it's worth fixing.
// Plus it's nice to be consistent and not have timing-dependent
// behavior.
func TestTransportReuseConnEmptyResponseBody(t *testing.T) {
	run(t, testTransportReuseConnEmptyResponseBody)
}
func testTransportReuseConnEmptyResponseBody(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Addr", r.RemoteAddr)
		// Empty response body.
	}))
	n := 100
	if testing.Short() {
		n = 10
	}
	var firstAddr string
	for i := 0; i < n; i++ {
		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			log.Fatal(err)
		}
		addr := res.Header.Get("X-Addr")
		if i == 0 {
			firstAddr = addr
		} else if addr != firstAddr {
			t.Fatalf("On request %d, addr %q != original addr %q", i+1, addr, firstAddr)
		}
		res.Body.Close()
	}
}

// Issue 13839
func TestNoCrashReturningTransportAltConn(t *testing.T) {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	ln := newLocalListener(t)
	defer ln.Close()

	var wg sync.WaitGroup
	SetPendingDialHooks(func() { wg.Add(1) }, wg.Done)
	defer SetPendingDialHooks(nil, nil)

	testDone := make(chan struct{})
	defer close(testDone)
	go func() {
		tln := tls.NewListener(ln, &tls.Config{
			NextProtos:   []string{"foo"},
			Certificates: []tls.Certificate{cert},
		})
		sc, err := tln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		if err := sc.(*tls.Conn).Handshake(); err != nil {
			t.Error(err)
			return
		}
		<-testDone
		sc.Close()
	}()

	addr := ln.Addr().String()

	req, _ := NewRequest("GET", "https://fake.tld/", nil)
	cancel := make(chan struct{})
	req.Cancel = cancel

	doReturned := make(chan bool, 1)
	madeRoundTripper := make(chan bool, 1)

	tr := &Transport{
		DisableKeepAlives: true,
		TLSNextProto: map[string]func(string, *tls.Conn) RoundTripper{
			"foo": func(authority string, c *tls.Conn) RoundTripper {
				madeRoundTripper <- true
				return funcRoundTripper(func() {
					t.Error("foo RoundTripper should not be called")
				})
			},
		},
		Dial: func(_, _ string) (net.Conn, error) {
			panic("shouldn't be called")
		},
		DialTLS: func(_, _ string) (net.Conn, error) {
			tc, err := tls.Dial("tcp", addr, &tls.Config{
				InsecureSkipVerify: true,
				NextProtos:         []string{"foo"},
			})
			if err != nil {
				return nil, err
			}
			if err := tc.Handshake(); err != nil {
				return nil, err
			}
			close(cancel)
			<-doReturned
			return tc, nil
		},
	}
	c := &Client{Transport: tr}

	_, err = c.Do(req)
	if ue, ok := err.(*url.Error); !ok || ue.Err != ExportErrRequestCanceledConn {
		t.Fatalf("Do error = %v; want url.Error with errRequestCanceledConn", err)
	}

	doReturned <- true
	<-madeRoundTripper
	wg.Wait()
}

func TestTransportReuseConnection_Gzip_Chunked(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testTransportReuseConnection_Gzip(t, mode, true)
	})
}

func TestTransportReuseConnection_Gzip_ContentLength(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testTransportReuseConnection_Gzip(t, mode, false)
	})
}

// Make sure we re-use underlying TCP connection for gzipped responses too.
func testTransportReuseConnection_Gzip(t *testing.T, mode testMode, chunked bool) {
	addr := make(chan string, 2)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		addr <- r.RemoteAddr
		w.Header().Set("Content-Encoding", "gzip")
		if chunked {
			w.(Flusher).Flush()
		}
		w.Write(rgz) // arbitrary gzip response
	})).ts
	c := ts.Client()

	trace := &httptrace.ClientTrace{
		GetConn:      func(hostPort string) { t.Logf("GetConn(%q)", hostPort) },
		GotConn:      func(ci httptrace.GotConnInfo) { t.Logf("GotConn(%+v)", ci) },
		PutIdleConn:  func(err error) { t.Logf("PutIdleConn(%v)", err) },
		ConnectStart: func(network, addr string) { t.Logf("ConnectStart(%q, %q)", network, addr) },
		ConnectDone:  func(network, addr string, err error) { t.Logf("ConnectDone(%q, %q, %v)", network, addr, err) },
	}
	ctx := httptrace.WithClientTrace(context.Background(), trace)

	for i := 0; i < 2; i++ {
		req, _ := NewRequest("GET", ts.URL, nil)
		req = req.WithContext(ctx)
		res, err := c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		buf := make([]byte, len(rgz))
		if n, err := io.ReadFull(res.Body, buf); err != nil {
			t.Errorf("%d. ReadFull = %v, %v", i, n, err)
		}
		// Note: no res.Body.Close call. It should work without it,
		// since the flate.Reader's internal buffering will hit EOF
		// and that should be sufficient.
	}
	a1, a2 := <-addr, <-addr
	if a1 != a2 {
		t.Fatalf("didn't reuse connection")
	}
}

func TestTransportResponseHeaderLength(t *testing.T) { run(t, testTransportResponseHeaderLength) }
func testTransportResponseHeaderLength(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("HTTP/2 Transport doesn't support MaxResponseHeaderBytes")
	}
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.URL.Path == "/long" {
			w.Header().Set("Long", strings.Repeat("a", 1<<20))
		}
	})).ts
	c := ts.Client()
	c.Transport.(*Transport).MaxResponseHeaderBytes = 512 << 10

	if res, err := c.Get(ts.URL); err != nil {
		t.Fatal(err)
	} else {
		res.Body.Close()
	}

	res, err := c.Get(ts.URL + "/long")
	if err == nil {
		defer res.Body.Close()
		var n int64
		for k, vv := range res.Header {
			for _, v := range vv {
				n += int64(len(k)) + int64(len(v))
			}
		}
		t.Fatalf("Unexpected success. Got %v and %d bytes of response headers", res.Status, n)
	}
	if want := "server response headers exceeded 524288 bytes"; !strings.Contains(err.Error(), want) {
		t.Errorf("got error: %v; want %q", err, want)
	}
}

func TestTransportEventTrace(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testTransportEventTrace(t, mode, false)
	}, testNotParallel)
}

// test a non-nil httptrace.ClientTrace but with all hooks set to zero.
func TestTransportEventTrace_NoHooks(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testTransportEventTrace(t, mode, true)
	}, testNotParallel)
}

func testTransportEventTrace(t *testing.T, mode testMode, noHooks bool) {
	const resBody = "some body"
	gotWroteReqEvent := make(chan struct{}, 500)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method == "GET" {
			// Do nothing for the second request.
			return
		}
		if _, err := io.ReadAll(r.Body); err != nil {
			t.Error(err)
		}
		if !noHooks {
			<-gotWroteReqEvent
		}
		io.WriteString(w, resBody)
	}), func(tr *Transport) {
		if tr.TLSClientConfig != nil {
			tr.TLSClientConfig.InsecureSkipVerify = true
		}
	})
	defer cst.close()

	cst.tr.ExpectContinueTimeout = 1 * time.Second

	var mu sync.Mutex // guards buf
	var buf strings.Builder
	logf := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		fmt.Fprintf(&buf, format, args...)
		buf.WriteByte('\n')
	}

	addrStr := cst.ts.Listener.Addr().String()
	ip, port, err := net.SplitHostPort(addrStr)
	if err != nil {
		t.Fatal(err)
	}

	// Install a fake DNS server.
	ctx := context.WithValue(context.Background(), nettrace.LookupIPAltResolverKey{}, func(ctx context.Context, network, host string) ([]net.IPAddr, error) {
		if host != "dns-is-faked.golang" {
			t.Errorf("unexpected DNS host lookup for %q/%q", network, host)
			return nil, nil
		}
		return []net.IPAddr{{IP: net.ParseIP(ip)}}, nil
	})

	body := "some body"
	req, _ := NewRequest("POST", cst.scheme()+"://dns-is-faked.golang:"+port, strings.NewReader(body))
	req.Header["X-Foo-Multiple-Vals"] = []string{"bar", "baz"}
	trace := &httptrace.ClientTrace{
		GetConn:              func(hostPort string) { logf("Getting conn for %v ...", hostPort) },
		GotConn:              func(ci httptrace.GotConnInfo) { logf("got conn: %+v", ci) },
		GotFirstResponseByte: func() { logf("first response byte") },
		PutIdleConn:          func(err error) { logf("PutIdleConn = %v", err) },
		DNSStart:             func(e httptrace.DNSStartInfo) { logf("DNS start: %+v", e) },
		DNSDone:              func(e httptrace.DNSDoneInfo) { logf("DNS done: %+v", e) },
		ConnectStart:         func(network, addr string) { logf("ConnectStart: Connecting to %s %s ...", network, addr) },
		ConnectDone: func(network, addr string, err error) {
			if err != nil {
				t.Errorf("ConnectDone: %v", err)
			}
			logf("ConnectDone: connected to %s %s = %v", network, addr, err)
		},
		WroteHeaderField: func(key string, value []string) {
			logf("WroteHeaderField: %s: %v", key, value)
		},
		WroteHeaders: func() {
			logf("WroteHeaders")
		},
		Wait100Continue: func() { logf("Wait100Continue") },
		Got100Continue:  func() { logf("Got100Continue") },
		WroteRequest: func(e httptrace.WroteRequestInfo) {
			logf("WroteRequest: %+v", e)
			gotWroteReqEvent <- struct{}{}
		},
	}
	if mode == http2Mode {
		trace.TLSHandshakeStart = func() { logf("tls handshake start") }
		trace.TLSHandshakeDone = func(s tls.ConnectionState, err error) {
			logf("tls handshake done. ConnectionState = %v \n err = %v", s, err)
		}
	}
	if noHooks {
		// zero out all func pointers, trying to get some path to crash
		*trace = httptrace.ClientTrace{}
	}
	req = req.WithContext(httptrace.WithClientTrace(ctx, trace))

	req.Header.Set("Expect", "100-continue")
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	logf("got roundtrip.response")
	slurp, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	logf("consumed body")
	if string(slurp) != resBody || res.StatusCode != 200 {
		t.Fatalf("Got %q, %v; want %q, 200 OK", slurp, res.Status, resBody)
	}
	res.Body.Close()

	if noHooks {
		// Done at this point. Just testing a full HTTP
		// requests can happen with a trace pointing to a zero
		// ClientTrace, full of nil func pointers.
		return
	}

	mu.Lock()
	got := buf.String()
	mu.Unlock()

	wantOnce := func(sub string) {
		if strings.Count(got, sub) != 1 {
			t.Errorf("expected substring %q exactly once in output.", sub)
		}
	}
	wantOnceOrMore := func(sub string) {
		if strings.Count(got, sub) == 0 {
			t.Errorf("expected substring %q at least once in output.", sub)
		}
	}
	wantOnce("Getting conn for dns-is-faked.golang:" + port)
	wantOnce("DNS start: {Host:dns-is-faked.golang}")
	wantOnce("DNS done: {Addrs:[{IP:" + ip + " Zone:}] Err:<nil> Coalesced:false}")
	wantOnce("got conn: {")
	wantOnceOrMore("Connecting to tcp " + addrStr)
	wantOnceOrMore("connected to tcp " + addrStr + " = <nil>")
	wantOnce("Reused:false WasIdle:false IdleTime:0s")
	wantOnce("first response byte")
	if mode == http2Mode {
		wantOnce("tls handshake start")
		wantOnce("tls handshake done")
	} else {
		wantOnce("PutIdleConn = <nil>")
		wantOnce("WroteHeaderField: User-Agent: [Go-http-client/1.1]")
		// TODO(meirf): issue 19761. Make these agnostic to h1/h2. (These are not h1 specific, but the
		// WroteHeaderField hook is not yet implemented in h2.)
		wantOnce(fmt.Sprintf("WroteHeaderField: Host: [dns-is-faked.golang:%s]", port))
		wantOnce(fmt.Sprintf("WroteHeaderField: Content-Length: [%d]", len(body)))
		wantOnce("WroteHeaderField: X-Foo-Multiple-Vals: [bar baz]")
		wantOnce("WroteHeaderField: Accept-Encoding: [gzip]")
	}
	wantOnce("WroteHeaders")
	wantOnce("Wait100Continue")
	wantOnce("Got100Continue")
	wantOnce("WroteRequest: {Err:<nil>}")
	if strings.Contains(got, " to udp ") {
		t.Errorf("should not see UDP (DNS) connections")
	}
	if t.Failed() {
		t.Errorf("Output:\n%s", got)
	}

	// And do a second request:
	req, _ = NewRequest("GET", cst.scheme()+"://dns-is-faked.golang:"+port, nil)
	req = req.WithContext(httptrace.WithClientTrace(ctx, trace))
	res, err = cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 200 {
		t.Fatal(res.Status)
	}
	res.Body.Close()

	mu.Lock()
	got = buf.String()
	mu.Unlock()

	sub := "Getting conn for dns-is-faked.golang:"
	if gotn, want := strings.Count(got, sub), 2; gotn != want {
		t.Errorf("substring %q appeared %d times; want %d. Log:\n%s", sub, gotn, want, got)
	}

}

func TestTransportEventTraceTLSVerify(t *testing.T) {
	run(t, testTransportEventTraceTLSVerify, []testMode{https1Mode, http2Mode})
}
func testTransportEventTraceTLSVerify(t *testing.T, mode testMode) {
	var mu sync.Mutex
	var buf strings.Builder
	logf := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		fmt.Fprintf(&buf, format, args...)
		buf.WriteByte('\n')
	}

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Error("Unexpected request")
	}), func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(funcWriter(func(p []byte) (int, error) {
			logf("%s", p)
			return len(p), nil
		}), "", 0)
	}).ts

	certpool := x509.NewCertPool()
	certpool.AddCert(ts.Certificate())

	c := &Client{Transport: &Transport{
		TLSClientConfig: &tls.Config{
			ServerName: "dns-is-faked.golang",
			RootCAs:    certpool,
		},
	}}

	trace := &httptrace.ClientTrace{
		TLSHandshakeStart: func() { logf("TLSHandshakeStart") },
		TLSHandshakeDone: func(s tls.ConnectionState, err error) {
			logf("TLSHandshakeDone: ConnectionState = %v \n err = %v", s, err)
		},
	}

	req, _ := NewRequest("GET", ts.URL, nil)
	req = req.WithContext(httptrace.WithClientTrace(context.Background(), trace))
	_, err := c.Do(req)
	if err == nil {
		t.Error("Expected request to fail TLS verification")
	}

	mu.Lock()
	got := buf.String()
	mu.Unlock()

	wantOnce := func(sub string) {
		if strings.Count(got, sub) != 1 {
			t.Errorf("expected substring %q exactly once in output.", sub)
		}
	}

	wantOnce("TLSHandshakeStart")
	wantOnce("TLSHandshakeDone")
	wantOnce("err = tls: failed to verify certificate: x509: certificate is valid for example.com")

	if t.Failed() {
		t.Errorf("Output:\n%s", got)
	}
}

var isDNSHijacked = sync.OnceValue(func() bool {
	addrs, _ := net.LookupHost("dns-should-not-resolve.golang")
	return len(addrs) != 0
})

func skipIfDNSHijacked(t *testing.T) {
	// Skip this test if the user is using a shady/ISP
	// DNS server hijacking queries.
	// See issues 16732, 16716.
	if isDNSHijacked() {
		t.Skip("skipping; test requires non-hijacking DNS server")
	}
}

func TestTransportEventTraceRealDNS(t *testing.T) {
	skipIfDNSHijacked(t)
	defer afterTest(t)
	tr := &Transport{}
	defer tr.CloseIdleConnections()
	c := &Client{Transport: tr}

	var mu sync.Mutex // guards buf
	var buf strings.Builder
	logf := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		fmt.Fprintf(&buf, format, args...)
		buf.WriteByte('\n')
	}

	req, _ := NewRequest("GET", "http://dns-should-not-resolve.golang:80", nil)
	trace := &httptrace.ClientTrace{
		DNSStart:     func(e httptrace.DNSStartInfo) { logf("DNSStart: %+v", e) },
		DNSDone:      func(e httptrace.DNSDoneInfo) { logf("DNSDone: %+v", e) },
		ConnectStart: func(network, addr string) { logf("ConnectStart: %s %s", network, addr) },
		ConnectDone:  func(network, addr string, err error) { logf("ConnectDone: %s %s %v", network, addr, err) },
	}
	req = req.WithContext(httptrace.WithClientTrace(context.Background(), trace))

	resp, err := c.Do(req)
	if err == nil {
		resp.Body.Close()
		t.Fatal("expected error during DNS lookup")
	}

	mu.Lock()
	got := buf.String()
	mu.Unlock()

	wantSub := func(sub string) {
		if !strings.Contains(got, sub) {
			t.Errorf("expected substring %q in output.", sub)
		}
	}
	wantSub("DNSStart: {Host:dns-should-not-resolve.golang}")
	wantSub("DNSDone: {Addrs:[] Err:")
	if strings.Contains(got, "ConnectStart") || strings.Contains(got, "ConnectDone") {
		t.Errorf("should not see Connect events")
	}
	if t.Failed() {
		t.Errorf("Output:\n%s", got)
	}
}

// Issue 14353: port can only contain digits.
func TestTransportRejectsAlphaPort(t *testing.T) {
	res, err := Get("http://dummy.tld:123foo/bar")
	if err == nil {
		res.Body.Close()
		t.Fatal("unexpected success")
	}
	ue, ok := err.(*url.Error)
	if !ok {
		t.Fatalf("got %#v; want *url.Error", err)
	}
	got := ue.Err.Error()
	want := `invalid port ":123foo" after host`
	if got != want {
		t.Errorf("got error %q; want %q", got, want)
	}
}

// Test the httptrace.TLSHandshake{Start,Done} hooks with an https http1
// connections. The http2 test is done in TestTransportEventTrace_h2
func TestTLSHandshakeTrace(t *testing.T) {
	run(t, testTLSHandshakeTrace, []testMode{https1Mode, http2Mode})
}
func testTLSHandshakeTrace(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {})).ts

	var mu sync.Mutex
	var start, done bool
	trace := &httptrace.ClientTrace{
		TLSHandshakeStart: func() {
			mu.Lock()
			defer mu.Unlock()
			start = true
		},
		TLSHandshakeDone: func(s tls.ConnectionState, err error) {
			mu.Lock()
			defer mu.Unlock()
			done = true
			if err != nil {
				t.Fatal("Expected error to be nil but was:", err)
			}
		},
	}

	c := ts.Client()
	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal("Unable to construct test request:", err)
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))

	r, err := c.Do(req)
	if err != nil {
		t.Fatal("Unexpected error making request:", err)
	}
	r.Body.Close()
	mu.Lock()
	defer mu.Unlock()
	if !start {
		t.Fatal("Expected TLSHandshakeStart to be called, but wasn't")
	}
	if !done {
		t.Fatal("Expected TLSHandshakeDone to be called, but wasn't")
	}
}

func TestTransportMaxIdleConns(t *testing.T) {
	run(t, testTransportMaxIdleConns, []testMode{http1Mode})
}
func testTransportMaxIdleConns(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// No body for convenience.
	})).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.MaxIdleConns = 4

	ip, port, err := net.SplitHostPort(ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), nettrace.LookupIPAltResolverKey{}, func(ctx context.Context, _, host string) ([]net.IPAddr, error) {
		return []net.IPAddr{{IP: net.ParseIP(ip)}}, nil
	})

	hitHost := func(n int) {
		req, _ := NewRequest("GET", fmt.Sprintf("http://host-%d.dns-is-faked.golang:"+port, n), nil)
		req = req.WithContext(ctx)
		res, err := c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
	}
	for i := 0; i < 4; i++ {
		hitHost(i)
	}
	want := []string{
		"|http|host-0.dns-is-faked.golang:" + port,
		"|http|host-1.dns-is-faked.golang:" + port,
		"|http|host-2.dns-is-faked.golang:" + port,
		"|http|host-3.dns-is-faked.golang:" + port,
	}
	if got := tr.IdleConnKeysForTesting(); !slices.Equal(got, want) {
		t.Fatalf("idle conn keys mismatch.\n got: %q\nwant: %q\n", got, want)
	}

	// Now hitting the 5th host should kick out the first host:
	hitHost(4)
	want = []string{
		"|http|host-1.dns-is-faked.golang:" + port,
		"|http|host-2.dns-is-faked.golang:" + port,
		"|http|host-3.dns-is-faked.golang:" + port,
		"|http|host-4.dns-is-faked.golang:" + port,
	}
	if got := tr.IdleConnKeysForTesting(); !slices.Equal(got, want) {
		t.Fatalf("idle conn keys mismatch after 5th host.\n got: %q\nwant: %q\n", got, want)
	}
}

func TestTransportIdleConnTimeout(t *testing.T) { run(t, testTransportIdleConnTimeout) }
func testTransportIdleConnTimeout(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	timeout := 1 * time.Millisecond
timeoutLoop:
	for {
		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			// No body for convenience.
		}))
		tr := cst.tr
		tr.IdleConnTimeout = timeout
		defer tr.CloseIdleConnections()
		c := &Client{Transport: tr}

		idleConns := func() []string {
			if mode == http2Mode {
				return tr.IdleConnStrsForTesting_h2()
			} else {
				return tr.IdleConnStrsForTesting()
			}
		}

		var conn string
		doReq := func(n int) (timeoutOk bool) {
			req, _ := NewRequest("GET", cst.ts.URL, nil)
			req = req.WithContext(httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
				PutIdleConn: func(err error) {
					if err != nil {
						t.Errorf("failed to keep idle conn: %v", err)
					}
				},
			}))
			res, err := c.Do(req)
			if err != nil {
				if strings.Contains(err.Error(), "use of closed network connection") {
					t.Logf("req %v: connection closed prematurely", n)
					return false
				}
			}
			if err == nil {
				res.Body.Close()
			}
			conns := idleConns()
			if len(conns) != 1 {
				if len(conns) == 0 {
					t.Logf("req %v: no idle conns", n)
					return false
				}
				t.Fatalf("req %v: unexpected number of idle conns: %q", n, conns)
			}
			if conn == "" {
				conn = conns[0]
			}
			if conn != conns[0] {
				t.Logf("req %v: cached connection changed; expected the same one throughout the test", n)
				return false
			}
			return true
		}
		for i := 0; i < 3; i++ {
			if !doReq(i) {
				t.Logf("idle conn timeout %v appears to be too short; retrying with longer", timeout)
				timeout *= 2
				cst.close()
				continue timeoutLoop
			}
			time.Sleep(timeout / 2)
		}

		waitCondition(t, timeout/2, func(d time.Duration) bool {
			if got := idleConns(); len(got) != 0 {
				if d >= timeout*3/2 {
					t.Logf("after %v, idle conns = %q", d, got)
				}
				return false
			}
			return true
		})
		break
	}
}

// Issue 16208: Go 1.7 crashed after Transport.IdleConnTimeout if an
// HTTP/2 connection was established but its caller no longer
// wanted it. (Assuming the connection cache was enabled, which it is
// by default)
//
// This test reproduced the crash by setting the IdleConnTimeout low
// (to make the test reasonable) and then making a request which is
// canceled by the DialTLS hook, which then also waits to return the
// real connection until after the RoundTrip saw the error.  Then we
// know the successful tls.Dial from DialTLS will need to go into the
// idle pool. Then we give it a of time to explode.
func TestIdleConnH2Crash(t *testing.T) { run(t, testIdleConnH2Crash, []testMode{http2Mode}) }
func testIdleConnH2Crash(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// nothing
	}))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sawDoErr := make(chan bool, 1)
	testDone := make(chan struct{})
	defer close(testDone)

	cst.tr.IdleConnTimeout = 5 * time.Millisecond
	cst.tr.DialTLS = func(network, addr string) (net.Conn, error) {
		c, err := tls.Dial(network, addr, &tls.Config{
			InsecureSkipVerify: true,
			NextProtos:         []string{"h2"},
		})
		if err != nil {
			t.Error(err)
			return nil, err
		}
		if cs := c.ConnectionState(); cs.NegotiatedProtocol != "h2" {
			t.Errorf("protocol = %q; want %q", cs.NegotiatedProtocol, "h2")
			c.Close()
			return nil, errors.New("bogus")
		}

		cancel()

		select {
		case <-sawDoErr:
		case <-testDone:
		}
		return c, nil
	}

	req, _ := NewRequest("GET", cst.ts.URL, nil)
	req = req.WithContext(ctx)
	res, err := cst.c.Do(req)
	if err == nil {
		res.Body.Close()
		t.Fatal("unexpected success")
	}
	sawDoErr <- true

	// Wait for the explosion.
	time.Sleep(cst.tr.IdleConnTimeout * 10)
}

type funcConn struct {
	net.Conn
	read  func([]byte) (int, error)
	write func([]byte) (int, error)
}

func (c funcConn) Read(p []byte) (int, error)  { return c.read(p) }
func (c funcConn) Write(p []byte) (int, error) { return c.write(p) }
func (c funcConn) Close() error                { return nil }

// Issue 16465: Transport.RoundTrip should return the raw net.Conn.Read error from Peek
// back to the caller.
func TestTransportReturnsPeekError(t *testing.T) {
	errValue := errors.New("specific error value")

	wrote := make(chan struct{})
	wroteOnce := sync.OnceFunc(func() { close(wrote) })

	tr := &Transport{
		Dial: func(network, addr string) (net.Conn, error) {
			c := funcConn{
				read: func([]byte) (int, error) {
					<-wrote
					return 0, errValue
				},
				write: func(p []byte) (int, error) {
					wroteOnce()
					return len(p), nil
				},
			}
			return c, nil
		},
	}
	_, err := tr.RoundTrip(httptest.NewRequest("GET", "http://fake.tld/", nil))
	if err != errValue {
		t.Errorf("error = %#v; want %v", err, errValue)
	}
}

// Issue 13835: international domain names should work
func TestTransportIDNA(t *testing.T) { run(t, testTransportIDNA) }
func testTransportIDNA(t *testing.T, mode testMode) {
	const uniDomain = "гофер.го"
	const punyDomain = "xn--c1ae0ajs.xn--c1aw"

	var port string
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		want := punyDomain + ":" + port
		if r.Host != want {
			t.Errorf("Host header = %q; want %q", r.Host, want)
		}
		if mode == http2Mode {
			if r.TLS == nil {
				t.Errorf("r.TLS == nil")
			} else if r.TLS.ServerName != punyDomain {
				t.Errorf("TLS.ServerName = %q; want %q", r.TLS.ServerName, punyDomain)
			}
		}
		w.Header().Set("Hit-Handler", "1")
	}), func(tr *Transport) {
		if tr.TLSClientConfig != nil {
			tr.TLSClientConfig.InsecureSkipVerify = true
		}
	})

	ip, port, err := net.SplitHostPort(cst.ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	// Install a fake DNS server.
	ctx := context.WithValue(context.Background(), nettrace.LookupIPAltResolverKey{}, func(ctx context.Context, network, host string) ([]net.IPAddr, error) {
		if host != punyDomain {
			t.Errorf("got DNS host lookup for %q/%q; want %q", network, host, punyDomain)
			return nil, nil
		}
		return []net.IPAddr{{IP: net.ParseIP(ip)}}, nil
	})

	req, _ := NewRequest("GET", cst.scheme()+"://"+uniDomain+":"+port, nil)
	trace := &httptrace.ClientTrace{
		GetConn: func(hostPort string) {
			want := net.JoinHostPort(punyDomain, port)
			if hostPort != want {
				t.Errorf("getting conn for %q; want %q", hostPort, want)
			}
		},
		DNSStart: func(e httptrace.DNSStartInfo) {
			if e.Host != punyDomain {
				t.Errorf("DNSStart Host = %q; want %q", e.Host, punyDomain)
			}
		},
	}
	req = req.WithContext(httptrace.WithClientTrace(ctx, trace))

	res, err := cst.tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.Header.Get("Hit-Handler") != "1" {
		out, err := httputil.DumpResponse(res, true)
		if err != nil {
			t.Fatal(err)
		}
		t.Errorf("Response body wasn't from Handler. Got:\n%s\n", out)
	}
}

// Issue 13290: send User-Agent in proxy CONNECT
func TestTransportProxyConnectHeader(t *testing.T) {
	run(t, testTransportProxyConnectHeader, []testMode{http1Mode})
}
func testTransportProxyConnectHeader(t *testing.T, mode testMode) {
	reqc := make(chan *Request, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method != "CONNECT" {
			t.Errorf("method = %q; want CONNECT", r.Method)
		}
		reqc <- r
		c, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Errorf("Hijack: %v", err)
			return
		}
		c.Close()
	})).ts

	c := ts.Client()
	c.Transport.(*Transport).Proxy = func(r *Request) (*url.URL, error) {
		return url.Parse(ts.URL)
	}
	c.Transport.(*Transport).ProxyConnectHeader = Header{
		"User-Agent": {"foo"},
		"Other":      {"bar"},
	}

	res, err := c.Get("https://dummy.tld/") // https to force a CONNECT
	if err == nil {
		res.Body.Close()
		t.Errorf("unexpected success")
	}

	r := <-reqc
	if got, want := r.Header.Get("User-Agent"), "foo"; got != want {
		t.Errorf("CONNECT request User-Agent = %q; want %q", got, want)
	}
	if got, want := r.Header.Get("Other"), "bar"; got != want {
		t.Errorf("CONNECT request Other = %q; want %q", got, want)
	}
}

func TestTransportProxyGetConnectHeader(t *testing.T) {
	run(t, testTransportProxyGetConnectHeader, []testMode{http1Mode})
}
func testTransportProxyGetConnectHeader(t *testing.T, mode testMode) {
	reqc := make(chan *Request, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method != "CONNECT" {
			t.Errorf("method = %q; want CONNECT", r.Method)
		}
		reqc <- r
		c, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Errorf("Hijack: %v", err)
			return
		}
		c.Close()
	})).ts

	c := ts.Client()
	c.Transport.(*Transport).Proxy = func(r *Request) (*url.URL, error) {
		return url.Parse(ts.URL)
	}
	// These should be ignored:
	c.Transport.(*Transport).ProxyConnectHeader = Header{
		"User-Agent": {"foo"},
		"Other":      {"bar"},
	}
	c.Transport.(*Transport).GetProxyConnectHeader = func(ctx context.Context, proxyURL *url.URL, target string) (Header, error) {
		return Header{
			"User-Agent": {"foo2"},
			"Other":      {"bar2"},
		}, nil
	}

	res, err := c.Get("https://dummy.tld/") // https to force a CONNECT
	if err == nil {
		res.Body.Close()
		t.Errorf("unexpected success")
	}

	r := <-reqc
	if got, want := r.Header.Get("User-Agent"), "foo2"; got != want {
		t.Errorf("CONNECT request User-Agent = %q; want %q", got, want)
	}
	if got, want := r.Header.Get("Other"), "bar2"; got != want {
		t.Errorf("CONNECT request Other = %q; want %q", got, want)
	}
}

var errFakeRoundTrip = errors.New("fake roundtrip")

type funcRoundTripper func()

func (fn funcRoundTripper) RoundTrip(*Request) (*Response, error) {
	fn()
	return nil, errFakeRoundTrip
}

func wantBody(res *Response, err error, want string) error {
	if err != nil {
		return err
	}
	slurp, err := io.ReadAll(res.Body)
	if err != nil {
		return fmt.Errorf("error reading body: %v", err)
	}
	if string(slurp) != want {
		return fmt.Errorf("body = %q; want %q", slurp, want)
	}
	if err := res.Body.Close(); err != nil {
		return fmt.Errorf("body Close = %v", err)
	}
	return nil
}

func newLocalListener(t *testing.T) net.Listener {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

type countCloseReader struct {
	n *int
	io.Reader
}

func (cr countCloseReader) Close() error {
	(*cr.n)++
	return nil
}

// rgz is a gzip quine that uncompresses to itself.
var rgz = []byte{
	0x1f, 0x8b, 0x08, 0x08, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x72, 0x65, 0x63, 0x75, 0x72, 0x73,
	0x69, 0x76, 0x65, 0x00, 0x92, 0xef, 0xe6, 0xe0,
	0x60, 0x00, 0x83, 0xa2, 0xd4, 0xe4, 0xd2, 0xa2,
	0xe2, 0xcc, 0xb2, 0x54, 0x06, 0x00, 0x00, 0x17,
	0x00, 0xe8, 0xff, 0x92, 0xef, 0xe6, 0xe0, 0x60,
	0x00, 0x83, 0xa2, 0xd4, 0xe4, 0xd2, 0xa2, 0xe2,
	0xcc, 0xb2, 0x54, 0x06, 0x00, 0x00, 0x17, 0x00,
	0xe8, 0xff, 0x42, 0x12, 0x46, 0x16, 0x06, 0x00,
	0x05, 0x00, 0xfa, 0xff, 0x42, 0x12, 0x46, 0x16,
	0x06, 0x00, 0x05, 0x00, 0xfa, 0xff, 0x00, 0x05,
	0x00, 0xfa, 0xff, 0x00, 0x14, 0x00, 0xeb, 0xff,
	0x42, 0x12, 0x46, 0x16, 0x06, 0x00, 0x05, 0x00,
	0xfa, 0xff, 0x00, 0x05, 0x00, 0xfa, 0xff, 0x00,
	0x14, 0x00, 0xeb, 0xff, 0x42, 0x88, 0x21, 0xc4,
	0x00, 0x00, 0x14, 0x00, 0xeb, 0xff, 0x42, 0x88,
	0x21, 0xc4, 0x00, 0x00, 0x14, 0x00, 0xeb, 0xff,
	0x42, 0x88, 0x21, 0xc4, 0x00, 0x00, 0x14, 0x00,
	0xeb, 0xff, 0x42, 0x88, 0x21, 0xc4, 0x00, 0x00,
	0x14, 0x00, 0xeb, 0xff, 0x42, 0x88, 0x21, 0xc4,
	0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00,
	0x00, 0xff, 0xff, 0x00, 0x17, 0x00, 0xe8, 0xff,
	0x42, 0x88, 0x21, 0xc4, 0x00, 0x00, 0x00, 0x00,
	0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00,
	0x17, 0x00, 0xe8, 0xff, 0x42, 0x12, 0x46, 0x16,
	0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0x01, 0x08,
	0x00, 0xf7, 0xff, 0x3d, 0xb1, 0x20, 0x85, 0xfa,
	0x00, 0x00, 0x00, 0x42, 0x12, 0x46, 0x16, 0x06,
	0x00, 0x00, 0x00, 0xff, 0xff, 0x01, 0x08, 0x00,
	0xf7, 0xff, 0x3d, 0xb1, 0x20, 0x85, 0xfa, 0x00,
	0x00, 0x00, 0x3d, 0xb1, 0x20, 0x85, 0xfa, 0x00,
	0x00, 0x00,
}

// Ensure that a missing status doesn't make the server panic
// See Issue https://golang.org/issues/21701
func TestMissingStatusNoPanic(t *testing.T) {
	t.Parallel()

	const want = "unknown status code"

	ln := newLocalListener(t)
	addr := ln.Addr().String()
	done := make(chan bool)
	fullAddrURL := fmt.Sprintf("http://%s", addr)
	raw := "HTTP/1.1 400\r\n" +
		"Date: Wed, 30 Aug 2017 19:09:27 GMT\r\n" +
		"Content-Type: text/html; charset=utf-8\r\n" +
		"Content-Length: 10\r\n" +
		"Last-Modified: Wed, 30 Aug 2017 19:02:02 GMT\r\n" +
		"Vary: Accept-Encoding\r\n\r\n" +
		"Aloha Olaa"

	go func() {
		defer close(done)

		conn, _ := ln.Accept()
		if conn != nil {
			io.WriteString(conn, raw)
			io.ReadAll(conn)
			conn.Close()
		}
	}()

	proxyURL, err := url.Parse(fullAddrURL)
	if err != nil {
		t.Fatalf("proxyURL: %v", err)
	}

	tr := &Transport{Proxy: ProxyURL(proxyURL)}

	req, _ := NewRequest("GET", "https://golang.org/", nil)
	res, err, panicked := doFetchCheckPanic(tr, req)
	if panicked {
		t.Error("panicked, expecting an error")
	}
	if res != nil && res.Body != nil {
		io.Copy(io.Discard, res.Body)
		res.Body.Close()
	}

	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("got=%v want=%q", err, want)
	}

	ln.Close()
	<-done
}

func doFetchCheckPanic(tr *Transport, req *Request) (res *Response, err error, panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
		}
	}()
	res, err = tr.RoundTrip(req)
	return
}

// Issue 22330: do not allow the response body to be read when the status code
// forbids a response body.
func TestNoBodyOnChunked304Response(t *testing.T) {
	run(t, testNoBodyOnChunked304Response, []testMode{http1Mode})
}
func testNoBodyOnChunked304Response(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, buf, _ := w.(Hijacker).Hijack()
		buf.Write([]byte("HTTP/1.1 304 NOT MODIFIED\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"))
		buf.Flush()
		conn.Close()
	}))

	// Our test server above is sending back bogus data after the
	// response (the "0\r\n\r\n" part), which causes the Transport
	// code to log spam. Disable keep-alives so we never even try
	// to reuse the connection.
	cst.tr.DisableKeepAlives = true

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	if res.Body != NoBody {
		t.Errorf("Unexpected body on 304 response")
	}
}

type funcWriter func([]byte) (int, error)

func (f funcWriter) Write(p []byte) (int, error) { return f(p) }

type doneContext struct {
	context.Context
	err error
}

func (doneContext) Done() <-chan struct{} {
	c := make(chan struct{})
	close(c)
	return c
}

func (d doneContext) Err() error { return d.err }

// Issue 25852: Transport should check whether Context is done early.
func TestTransportCheckContextDoneEarly(t *testing.T) {
	tr := &Transport{}
	req, _ := NewRequest("GET", "http://fake.example/", nil)
	wantErr := errors.New("some error")
	req = req.WithContext(doneContext{context.Background(), wantErr})
	_, err := tr.RoundTrip(req)
	if err != wantErr {
		t.Errorf("error = %v; want %v", err, wantErr)
	}
}

// Issue 23399: verify that if a client request times out, the Transport's
// conn is closed so that it's not reused.
//
// This is the test variant that times out before the server replies with
// any response headers.
func TestClientTimeoutKillsConn_BeforeHeaders(t *testing.T) {
	run(t, testClientTimeoutKillsConn_BeforeHeaders, []testMode{http1Mode})
}
func testClientTimeoutKillsConn_BeforeHeaders(t *testing.T, mode testMode) {
	timeout := 1 * time.Millisecond
	for {
		inHandler := make(chan bool)
		cancelHandler := make(chan struct{})
		handlerDone := make(chan bool)
		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			<-r.Context().Done()

			select {
			case <-cancelHandler:
				return
			case inHandler <- true:
			}
			defer func() { handlerDone <- true }()

			// Read from the conn until EOF to verify that it was correctly closed.
			conn, _, err := w.(Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			n, err := conn.Read([]byte{0})
			if n != 0 || err != io.EOF {
				t.Errorf("unexpected Read result: %v, %v", n, err)
			}
			conn.Close()
		}))

		cst.c.Timeout = timeout

		_, err := cst.c.Get(cst.ts.URL)
		if err == nil {
			close(cancelHandler)
			t.Fatal("unexpected Get success")
		}

		tooSlow := time.NewTimer(timeout * 10)
		select {
		case <-tooSlow.C:
			// If we didn't get into the Handler, that probably means the builder was
			// just slow and the Get failed in that time but never made it to the
			// server. That's fine; we'll try again with a longer timeout.
			t.Logf("no handler seen in %v; retrying with longer timeout", timeout)
			close(cancelHandler)
			cst.close()
			timeout *= 2
			continue
		case <-inHandler:
			tooSlow.Stop()
			<-handlerDone
		}
		break
	}
}

// Issue 23399: verify that if a client request times out, the Transport's
// conn is closed so that it's not reused.
//
// This is the test variant that has the server send response headers
// first, and time out during the write of the response body.
func TestClientTimeoutKillsConn_AfterHeaders(t *testing.T) {
	run(t, testClientTimeoutKillsConn_AfterHeaders, []testMode{http1Mode})
}
func testClientTimeoutKillsConn_AfterHeaders(t *testing.T, mode testMode) {
	inHandler := make(chan bool)
	cancelHandler := make(chan struct{})
	handlerDone := make(chan bool)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "100")
		w.(Flusher).Flush()

		select {
		case <-cancelHandler:
			return
		case inHandler <- true:
		}
		defer func() { handlerDone <- true }()

		conn, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		conn.Write([]byte("foo"))

		n, err := conn.Read([]byte{0})
		// The error should be io.EOF or "read tcp
		// 127.0.0.1:35827->127.0.0.1:40290: read: connection
		// reset by peer" depending on timing. Really we just
		// care that it returns at all. But if it returns with
		// data, that's weird.
		if n != 0 || err == nil {
			t.Errorf("unexpected Read result: %v, %v", n, err)
		}
		conn.Close()
	}))

	// Set Timeout to something very long but non-zero to exercise
	// the codepaths that check for it. But rather than wait for it to fire
	// (which would make the test slow), we send on the req.Cancel channel instead,
	// which happens to exercise the same code paths.
	cst.c.Timeout = 24 * time.Hour // just to be non-zero, not to hit it.
	req, _ := NewRequest("GET", cst.ts.URL, nil)
	cancelReq := make(chan struct{})
	req.Cancel = cancelReq

	res, err := cst.c.Do(req)
	if err != nil {
		close(cancelHandler)
		t.Fatalf("Get error: %v", err)
	}

	// Cancel the request while the handler is still blocked on sending to the
	// inHandler channel. Then read it until it fails, to verify that the
	// connection is broken before the handler itself closes it.
	close(cancelReq)
	got, err := io.ReadAll(res.Body)
	if err == nil {
		t.Errorf("unexpected success; read %q, nil", got)
	}

	// Now unblock the handler and wait for it to complete.
	<-inHandler
	<-handlerDone
}

func TestTransportResponseBodyWritableOnProtocolSwitch(t *testing.T) {
	run(t, testTransportResponseBodyWritableOnProtocolSwitch, []testMode{http1Mode})
}
func testTransportResponseBodyWritableOnProtocolSwitch(t *testing.T, mode testMode) {
	done := make(chan struct{})
	defer close(done)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		io.WriteString(conn, "HTTP/1.1 101 Switching Protocols Hi\r\nConnection: upgRADe\r\nUpgrade: foo\r\n\r\nSome buffered data\n")
		bs := bufio.NewScanner(conn)
		bs.Scan()
		fmt.Fprintf(conn, "%s\n", strings.ToUpper(bs.Text()))
		<-done
	}))

	req, _ := NewRequest("GET", cst.ts.URL, nil)
	req.Header.Set("Upgrade", "foo")
	req.Header.Set("Connection", "upgrade")
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 101 {
		t.Fatalf("expected 101 switching protocols; got %v, %v", res.Status, res.Header)
	}
	rwc, ok := res.Body.(io.ReadWriteCloser)
	if !ok {
		t.Fatalf("expected a ReadWriteCloser; got a %T", res.Body)
	}
	defer rwc.Close()
	bs := bufio.NewScanner(rwc)
	if !bs.Scan() {
		t.Fatalf("expected readable input")
	}
	if got, want := bs.Text(), "Some buffered data"; got != want {
		t.Errorf("read %q; want %q", got, want)
	}
	io.WriteString(rwc, "echo\n")
	if !bs.Scan() {
		t.Fatalf("expected another line")
	}
	if got, want := bs.Text(), "ECHO"; got != want {
		t.Errorf("read %q; want %q", got, want)
	}
}

func TestTransportCONNECTBidi(t *testing.T) { run(t, testTransportCONNECTBidi, []testMode{http1Mode}) }
func testTransportCONNECTBidi(t *testing.T, mode testMode) {
	const target = "backend:443"
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method != "CONNECT" {
			t.Errorf("unexpected method %q", r.Method)
			w.WriteHeader(500)
			return
		}
		if r.RequestURI != target {
			t.Errorf("unexpected CONNECT target %q", r.RequestURI)
			w.WriteHeader(500)
			return
		}
		nc, brw, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer nc.Close()
		nc.Write([]byte("HTTP/1.1 200 OK\r\n\r\n"))
		// Switch to a little protocol that capitalize its input lines:
		for {
			line, err := brw.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					t.Error(err)
				}
				return
			}
			io.WriteString(brw, strings.ToUpper(line))
			brw.Flush()
		}
	}))
	pr, pw := io.Pipe()
	defer pw.Close()
	req, err := NewRequest("CONNECT", cst.ts.URL, pr)
	if err != nil {
		t.Fatal(err)
	}
	req.URL.Opaque = target
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("status code = %d; want 200", res.StatusCode)
	}
	br := bufio.NewReader(res.Body)
	for _, str := range []string{"foo", "bar", "baz"} {
		fmt.Fprintf(pw, "%s\n", str)
		got, err := br.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		got = strings.TrimSpace(got)
		want := strings.ToUpper(str)
		if got != want {
			t.Fatalf("got %q; want %q", got, want)
		}
	}
}

func TestTransportRequestReplayable(t *testing.T) {
	someBody := io.NopCloser(strings.NewReader(""))
	tests := []struct {
		name string
		req  *Request
		want bool
	}{
		{
			name: "GET",
			req:  &Request{Method: "GET"},
			want: true,
		},
		{
			name: "GET_http.NoBody",
			req:  &Request{Method: "GET", Body: NoBody},
			want: true,
		},
		{
			name: "GET_body",
			req:  &Request{Method: "GET", Body: someBody},
			want: false,
		},
		{
			name: "POST",
			req:  &Request{Method: "POST"},
			want: false,
		},
		{
			name: "POST_idempotency-key",
			req:  &Request{Method: "POST", Header: Header{"Idempotency-Key": {"x"}}},
			want: true,
		},
		{
			name: "POST_x-idempotency-key",
			req:  &Request{Method: "POST", Header: Header{"X-Idempotency-Key": {"x"}}},
			want: true,
		},
		{
			name: "POST_body",
			req:  &Request{Method: "POST", Header: Header{"Idempotency-Key": {"x"}}, Body: someBody},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.req.ExportIsReplayable()
			if got != tt.want {
				t.Errorf("replyable = %v; want %v", got, tt.want)
			}
		})
	}
}

// testMockTCPConn is a mock TCP connection used to test that
// ReadFrom is called when sending the request body.
type testMockTCPConn struct {
	*net.TCPConn

	ReadFromCalled bool
}

func (c *testMockTCPConn) ReadFrom(r io.Reader) (int64, error) {
	c.ReadFromCalled = true
	return c.TCPConn.ReadFrom(r)
}

func TestTransportRequestWriteRoundTrip(t *testing.T) { run(t, testTransportRequestWriteRoundTrip) }
func testTransportRequestWriteRoundTrip(t *testing.T, mode testMode) {
	nBytes := int64(1 << 10)
	newFileFunc := func() (r io.Reader, done func(), err error) {
		f, err := os.CreateTemp("", "net-http-newfilefunc")
		if err != nil {
			return nil, nil, err
		}

		// Write some bytes to the file to enable reading.
		if _, err := io.CopyN(f, rand.Reader, nBytes); err != nil {
			return nil, nil, fmt.Errorf("failed to write data to file: %v", err)
		}
		if _, err := f.Seek(0, 0); err != nil {
			return nil, nil, fmt.Errorf("failed to seek to front: %v", err)
		}

		done = func() {
			f.Close()
			os.Remove(f.Name())
		}

		return f, done, nil
	}

	newBufferFunc := func() (io.Reader, func(), error) {
		return bytes.NewBuffer(make([]byte, nBytes)), func() {}, nil
	}

	cases := []struct {
		name             string
		readerFunc       func() (io.Reader, func(), error)
		contentLength    int64
		expectedReadFrom bool
	}{
		{
			name:             "file, length",
			readerFunc:       newFileFunc,
			contentLength:    nBytes,
			expectedReadFrom: true,
		},
		{
			name:       "file, no length",
			readerFunc: newFileFunc,
		},
		{
			name:          "file, negative length",
			readerFunc:    newFileFunc,
			contentLength: -1,
		},
		{
			name:          "buffer",
			contentLength: nBytes,
			readerFunc:    newBufferFunc,
		},
		{
			name:       "buffer, no length",
			readerFunc: newBufferFunc,
		},
		{
			name:          "buffer, length -1",
			contentLength: -1,
			readerFunc:    newBufferFunc,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, cleanup, err := tc.readerFunc()
			if err != nil {
				t.Fatal(err)
			}
			defer cleanup()

			tConn := &testMockTCPConn{}
			trFunc := func(tr *Transport) {
				tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
					var d net.Dialer
					conn, err := d.DialContext(ctx, network, addr)
					if err != nil {
						return nil, err
					}

					tcpConn, ok := conn.(*net.TCPConn)
					if !ok {
						return nil, fmt.Errorf("%s/%s does not provide a *net.TCPConn", network, addr)
					}

					tConn.TCPConn = tcpConn
					return tConn, nil
				}
			}

			cst := newClientServerTest(
				t,
				mode,
				HandlerFunc(func(w ResponseWriter, r *Request) {
					io.Copy(io.Discard, r.Body)
					r.Body.Close()
					w.WriteHeader(200)
				}),
				trFunc,
			)

			req, err := NewRequest("PUT", cst.ts.URL, r)
			if err != nil {
				t.Fatal(err)
			}
			req.ContentLength = tc.contentLength
			req.Header.Set("Content-Type", "application/octet-stream")
			resp, err := cst.c.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != 200 {
				t.Fatalf("status code = %d; want 200", resp.StatusCode)
			}

			expectedReadFrom := tc.expectedReadFrom
			if mode != http1Mode {
				expectedReadFrom = false
			}
			if !tConn.ReadFromCalled && expectedReadFrom {
				t.Fatalf("did not call ReadFrom")
			}

			if tConn.ReadFromCalled && !expectedReadFrom {
				t.Fatalf("ReadFrom was unexpectedly invoked")
			}
		})
	}
}

func TestTransportClone(t *testing.T) {
	tr := &Transport{
		Proxy: func(*Request) (*url.URL, error) { panic("") },
		OnProxyConnectResponse: func(ctx context.Context, proxyURL *url.URL, connectReq *Request, connectRes *Response) error {
			return nil
		},
		DialContext:            func(ctx context.Context, network, addr string) (net.Conn, error) { panic("") },
		Dial:                   func(network, addr string) (net.Conn, error) { panic("") },
		DialTLS:                func(network, addr string) (net.Conn, error) { panic("") },
		DialTLSContext:         func(ctx context.Context, network, addr string) (net.Conn, error) { panic("") },
		TLSClientConfig:        new(tls.Config),
		TLSHandshakeTimeout:    time.Second,
		DisableKeepAlives:      true,
		DisableCompression:     true,
		MaxIdleConns:           1,
		MaxIdleConnsPerHost:    1,
		MaxConnsPerHost:        1,
		IdleConnTimeout:        time.Second,
		ResponseHeaderTimeout:  time.Second,
		ExpectContinueTimeout:  time.Second,
		ProxyConnectHeader:     Header{},
		GetProxyConnectHeader:  func(context.Context, *url.URL, string) (Header, error) { return nil, nil },
		MaxResponseHeaderBytes: 1,
		ForceAttemptHTTP2:      true,
		HTTP2:                  &HTTP2Config{MaxConcurrentStreams: 1},
		Protocols:              &Protocols{},
		TLSNextProto: map[string]func(authority string, c *tls.Conn) RoundTripper{
			"foo": func(authority string, c *tls.Conn) RoundTripper { panic("") },
		},
		ReadBufferSize:  1,
		WriteBufferSize: 1,
	}
	tr.Protocols.SetHTTP1(true)
	tr.Protocols.SetHTTP2(true)
	tr2 := tr.Clone()
	rv := reflect.ValueOf(tr2).Elem()
	rt := rv.Type()
	for i := 0; i < rt.NumField(); i++ {
		sf := rt.Field(i)
		if !token.IsExported(sf.Name) {
			continue
		}
		if rv.Field(i).IsZero() {
			t.Errorf("cloned field t2.%s is zero", sf.Name)
		}
	}

	if _, ok := tr2.TLSNextProto["foo"]; !ok {
		t.Errorf("cloned Transport lacked TLSNextProto 'foo' key")
	}

	// But test that a nil TLSNextProto is kept nil:
	tr = new(Transport)
	tr2 = tr.Clone()
	if tr2.TLSNextProto != nil {
		t.Errorf("Transport.TLSNextProto unexpected non-nil")
	}
}

func TestIs408(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"HTTP/1.0 408", true},
		{"HTTP/1.1 408", true},
		{"HTTP/1.8 408", true},
		{"HTTP/2.0 408", false}, // maybe h2c would do this? but false for now.
		{"HTTP/1.1 408 ", true},
		{"HTTP/1.1 40", false},
		{"http/1.0 408", false},
		{"HTTP/1-1 408", false},
	}
	for _, tt := range tests {
		if got := Export_is408Message([]byte(tt.in)); got != tt.want {
			t.Errorf("is408Message(%q) = %v; want %v", tt.in, got, tt.want)
		}
	}
}

func TestTransportIgnores408(t *testing.T) {
	run(t, testTransportIgnores408, []testMode{http1Mode}, testNotParallel)
}
func testTransportIgnores408(t *testing.T, mode testMode) {
	// Not parallel. Relies on mutating the log package's global Output.
	defer log.SetOutput(log.Writer())

	var logout strings.Builder
	log.SetOutput(&logout)

	const target = "backend:443"

	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		nc, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer nc.Close()
		nc.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"))
		nc.Write([]byte("HTTP/1.1 408 bye\r\n")) // changing 408 to 409 makes test fail
	}))
	req, err := NewRequest("GET", cst.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	slurp, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	if string(slurp) != "ok" {
		t.Fatalf("got %q; want ok", slurp)
	}

	waitCondition(t, 1*time.Millisecond, func(d time.Duration) bool {
		if n := cst.tr.IdleConnKeyCountForTesting(); n != 0 {
			if d > 0 {
				t.Logf("%v idle conns still present after %v", n, d)
			}
			return false
		}
		return true
	})
	if got := logout.String(); got != "" {
		t.Fatalf("expected no log output; got: %s", got)
	}
}

func TestInvalidHeaderResponse(t *testing.T) {
	run(t, testInvalidHeaderResponse, []testMode{http1Mode})
}
func testInvalidHeaderResponse(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, buf, _ := w.(Hijacker).Hijack()
		buf.Write([]byte("HTTP/1.1 200 OK\r\n" +
			"Date: Wed, 30 Aug 2017 19:09:27 GMT\r\n" +
			"Content-Type: text/html; charset=utf-8\r\n" +
			"Content-Length: 0\r\n" +
			"Foo : bar\r\n\r\n"))
		buf.Flush()
		conn.Close()
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if v := res.Header.Get("Foo"); v != "" {
		t.Errorf(`unexpected "Foo" header: %q`, v)
	}
	if v := res.Header.Get("Foo "); v != "bar" {
		t.Errorf(`bad "Foo " header value: %q, want %q`, v, "bar")
	}
}

type bodyCloser bool

func (bc *bodyCloser) Close() error {
	*bc = true
	return nil
}
func (bc *bodyCloser) Read(b []byte) (n int, err error) {
	return 0, io.EOF
}

// Issue 35015: ensure that Transport closes the body on any error
// with an invalid request, as promised by Client.Do docs.
func TestTransportClosesBodyOnInvalidRequests(t *testing.T) {
	run(t, testTransportClosesBodyOnInvalidRequests)
}
func testTransportClosesBodyOnInvalidRequests(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Errorf("Should not have been invoked")
	})).ts

	u, _ := url.Parse(cst.URL)

	tests := []struct {
		name    string
		req     *Request
		wantErr string
	}{
		{
			name: "invalid method",
			req: &Request{
				Method: " ",
				URL:    u,
			},
			wantErr: `invalid method " "`,
		},
		{
			name: "nil URL",
			req: &Request{
				Method: "GET",
			},
			wantErr: `nil Request.URL`,
		},
		{
			name: "invalid header key",
			req: &Request{
				Method: "GET",
				Header: Header{"💡": {"emoji"}},
				URL:    u,
			},
			wantErr: `invalid header field name "💡"`,
		},
		{
			name: "invalid header value",
			req: &Request{
				Method: "POST",
				Header: Header{"key": {"\x19"}},
				URL:    u,
			},
			wantErr: `invalid header field value for "key"`,
		},
		{
			name: "non HTTP(s) scheme",
			req: &Request{
				Method: "POST",
				URL:    &url.URL{Scheme: "faux"},
			},
			wantErr: `unsupported protocol scheme "faux"`,
		},
		{
			name: "no Host in URL",
			req: &Request{
				Method: "POST",
				URL:    &url.URL{Scheme: "http"},
			},
			wantErr: `no Host in request URL`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var bc bodyCloser
			req := tt.req
			req.Body = &bc
			_, err := cst.Client().Do(tt.req)
			if err == nil {
				t.Fatal("Expected an error")
			}
			if !bc {
				t.Fatal("Expected body to have been closed")
			}
			if g, w := err.Error(), tt.wantErr; !strings.HasSuffix(g, w) {
				t.Fatalf("Error mismatch: %q does not end with %q", g, w)
			}
		})
	}
}

// breakableConn is a net.Conn wrapper with a Write method
// that will fail when its brokenState is true.
type breakableConn struct {
	net.Conn
	*brokenState
}

type brokenState struct {
	sync.Mutex
	broken bool
}

func (w *breakableConn) Write(b []byte) (n int, err error) {
	w.Lock()
	defer w.Unlock()
	if w.broken {
		return 0, errors.New("some write error")
	}
	return w.Conn.Write(b)
}

// Issue 34978: don't cache a broken HTTP/2 connection
func TestDontCacheBrokenHTTP2Conn(t *testing.T) {
	run(t, testDontCacheBrokenHTTP2Conn, []testMode{http2Mode})
}
func testDontCacheBrokenHTTP2Conn(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {}), optQuietLog)

	var brokenState brokenState

	const numReqs = 5
	var numDials, gotConns uint32 // atomic

	cst.tr.Dial = func(netw, addr string) (net.Conn, error) {
		atomic.AddUint32(&numDials, 1)
		c, err := net.Dial(netw, addr)
		if err != nil {
			t.Errorf("unexpected Dial error: %v", err)
			return nil, err
		}
		return &breakableConn{c, &brokenState}, err
	}

	for i := 1; i <= numReqs; i++ {
		brokenState.Lock()
		brokenState.broken = false
		brokenState.Unlock()

		// doBreak controls whether we break the TCP connection after the TLS
		// handshake (before the HTTP/2 handshake). We test a few failures
		// in a row followed by a final success.
		doBreak := i != numReqs

		ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
			GotConn: func(info httptrace.GotConnInfo) {
				t.Logf("got conn: %v, reused=%v, wasIdle=%v, idleTime=%v", info.Conn.LocalAddr(), info.Reused, info.WasIdle, info.IdleTime)
				atomic.AddUint32(&gotConns, 1)
			},
			TLSHandshakeDone: func(cfg tls.ConnectionState, err error) {
				brokenState.Lock()
				defer brokenState.Unlock()
				if doBreak {
					brokenState.broken = true
				}
			},
		})
		req, err := NewRequestWithContext(ctx, "GET", cst.ts.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = cst.c.Do(req)
		if doBreak != (err != nil) {
			t.Errorf("for iteration %d, doBreak=%v; unexpected error %v", i, doBreak, err)
		}
	}
	if got, want := atomic.LoadUint32(&gotConns), 1; int(got) != want {
		t.Errorf("GotConn calls = %v; want %v", got, want)
	}
	if got, want := atomic.LoadUint32(&numDials), numReqs; int(got) != want {
		t.Errorf("Dials = %v; want %v", got, want)
	}
}

// Issue 34941
// When the client has too many concurrent requests on a single connection,
// http.http2noCachedConnError is reported on multiple requests. There should
// only be one decrement regardless of the number of failures.
func TestTransportDecrementConnWhenIdleConnRemoved(t *testing.T) {
	run(t, testTransportDecrementConnWhenIdleConnRemoved, []testMode{http2Mode})
}
func testTransportDecrementConnWhenIdleConnRemoved(t *testing.T, mode testMode) {
	CondSkipHTTP2(t)

	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("foo"))
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
	})

	ts := newClientServerTest(t, mode, h).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.MaxConnsPerHost = 1

	errCh := make(chan error, 300)
	doReq := func() {
		resp, err := c.Get(ts.URL)
		if err != nil {
			errCh <- fmt.Errorf("request failed: %v", err)
			return
		}
		defer resp.Body.Close()
		_, err = io.ReadAll(resp.Body)
		if err != nil {
			errCh <- fmt.Errorf("read body failed: %v", err)
		}
	}

	var wg sync.WaitGroup
	for i := 0; i < 300; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			doReq()
		}()
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("error occurred: %v", err)
	}
}

// Issue 36820
// Test that we use the older backward compatible cancellation protocol
// when a RoundTripper is registered via RegisterProtocol.
func TestAltProtoCancellation(t *testing.T) {
	defer afterTest(t)
	tr := &Transport{}
	c := &Client{
		Transport: tr,
		Timeout:   time.Millisecond,
	}
	tr.RegisterProtocol("cancel", cancelProto{})
	_, err := c.Get("cancel://bar.com/path")
	if err == nil {
		t.Error("request unexpectedly succeeded")
	} else if !strings.Contains(err.Error(), errCancelProto.Error()) {
		t.Errorf("got error %q, does not contain expected string %q", err, errCancelProto)
	}
}

var errCancelProto = errors.New("canceled as expected")

type cancelProto struct{}

func (cancelProto) RoundTrip(req *Request) (*Response, error) {
	<-req.Cancel
	return nil, errCancelProto
}

type roundTripFunc func(r *Request) (*Response, error)

func (f roundTripFunc) RoundTrip(r *Request) (*Response, error) { return f(r) }

// Issue 32441: body is not reset after ErrSkipAltProtocol
func TestIssue32441(t *testing.T) { run(t, testIssue32441, []testMode{http1Mode}) }
func testIssue32441(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if n, _ := io.Copy(io.Discard, r.Body); n == 0 {
			t.Error("body length is zero")
		}
	})).ts
	c := ts.Client()
	c.Transport.(*Transport).RegisterProtocol("http", roundTripFunc(func(r *Request) (*Response, error) {
		// Draining body to trigger failure condition on actual request to server.
		if n, _ := io.Copy(io.Discard, r.Body); n == 0 {
			t.Error("body length is zero during round trip")
		}
		return nil, ErrSkipAltProtocol
	}))
	if _, err := c.Post(ts.URL, "application/octet-stream", bytes.NewBufferString("data")); err != nil {
		t.Error(err)
	}
}

// Issue 39017. Ensure that HTTP/1 transports reject Content-Length headers
// that contain a sign (eg. "+3"), per RFC 2616, Section 14.13.
func TestTransportRejectsSignInContentLength(t *testing.T) {
	run(t, testTransportRejectsSignInContentLength, []testMode{http1Mode})
}
func testTransportRejectsSignInContentLength(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "+3")
		w.Write([]byte("abc"))
	})).ts

	c := cst.Client()
	res, err := c.Get(cst.URL)
	if err == nil || res != nil {
		t.Fatal("Expected a non-nil error and a nil http.Response")
	}
	if got, want := err.Error(), `bad Content-Length "+3"`; !strings.Contains(got, want) {
		t.Fatalf("Error mismatch\nGot: %q\nWanted substring: %q", got, want)
	}
}

// dumpConn is a net.Conn which writes to Writer and reads from Reader
type dumpConn struct {
	io.Writer
	io.Reader
}

func (c *dumpConn) Close() error                       { return nil }
func (c *dumpConn) LocalAddr() net.Addr                { return nil }
func (c *dumpConn) RemoteAddr() net.Addr               { return nil }
func (c *dumpConn) SetDeadline(t time.Time) error      { return nil }
func (c *dumpConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *dumpConn) SetWriteDeadline(t time.Time) error { return nil }

// delegateReader is a reader that delegates to another reader,
// once it arrives on a channel.
type delegateReader struct {
	c chan io.Reader
	r io.Reader // nil until received from c
}

func (r *delegateReader) Read(p []byte) (int, error) {
	if r.r == nil {
		var ok bool
		if r.r, ok = <-r.c; !ok {
			return 0, errors.New("delegate closed")
		}
	}
	return r.r.Read(p)
}

func testTransportRace(req *Request) {
	save := req.Body
	pr, pw := io.Pipe()
	defer pr.Close()
	defer pw.Close()
	dr := &delegateReader{c: make(chan io.Reader)}

	t := &Transport{
		Dial: func(net, addr string) (net.Conn, error) {
			return &dumpConn{pw, dr}, nil
		},
	}
	defer t.CloseIdleConnections()

	quitReadCh := make(chan struct{})
	// Wait for the request before replying with a dummy response:
	go func() {
		defer close(quitReadCh)

		req, err := ReadRequest(bufio.NewReader(pr))
		if err == nil {
			// Ensure all the body is read; otherwise
			// we'll get a partial dump.
			io.Copy(io.Discard, req.Body)
			req.Body.Close()
		}
		select {
		case dr.c <- strings.NewReader("HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n"):
		case quitReadCh <- struct{}{}:
			// Ensure delegate is closed so Read doesn't block forever.
			close(dr.c)
		}
	}()

	t.RoundTrip(req)

	// Ensure the reader returns before we reset req.Body to prevent
	// a data race on req.Body.
	pw.Close()
	<-quitReadCh

	req.Body = save
}

// Issue 37669
// Test that a cancellation doesn't result in a data race due to the writeLoop
// goroutine being left running, if the caller mutates the processed Request
// upon completion.
func TestErrorWriteLoopRace(t *testing.T) {
	if testing.Short() {
		return
	}
	t.Parallel()
	for i := 0; i < 1000; i++ {
		delay := time.Duration(mrand.Intn(5)) * time.Millisecond
		ctx, cancel := context.WithTimeout(context.Background(), delay)
		defer cancel()

		r := bytes.NewBuffer(make([]byte, 10000))
		req, err := NewRequestWithContext(ctx, MethodPost, "http://example.com", r)
		if err != nil {
			t.Fatal(err)
		}

		testTransportRace(req)
	}
}

// Issue 41600
// Test that a new request which uses the connection of an active request
// cannot cause it to be canceled as well.
func TestCancelRequestWhenSharingConnection(t *testing.T) {
	run(t, testCancelRequestWhenSharingConnection, []testMode{http1Mode})
}
func testCancelRequestWhenSharingConnection(t *testing.T, mode testMode) {
	reqc := make(chan chan struct{}, 2)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, req *Request) {
		ch := make(chan struct{}, 1)
		reqc <- ch
		<-ch
		w.Header().Add("Content-Length", "0")
	})).ts

	client := ts.Client()
	transport := client.Transport.(*Transport)
	transport.MaxIdleConns = 1
	transport.MaxConnsPerHost = 1

	var wg sync.WaitGroup

	wg.Add(1)
	putidlec := make(chan chan struct{}, 1)
	reqerrc := make(chan error, 1)
	go func() {
		defer wg.Done()
		ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
			PutIdleConn: func(error) {
				// Signal that the idle conn has been returned to the pool,
				// and wait for the order to proceed.
				ch := make(chan struct{})
				putidlec <- ch
				close(putidlec) // panic if PutIdleConn runs twice for some reason
				<-ch
			},
		})
		req, _ := NewRequestWithContext(ctx, "GET", ts.URL, nil)
		res, err := client.Do(req)
		if err != nil {
			reqerrc <- err
		} else {
			res.Body.Close()
		}
	}()

	// Wait for the first request to receive a response and return the
	// connection to the idle pool.
	select {
	case err := <-reqerrc:
		t.Fatalf("request 1: got err %v, want nil", err)
	case r1c := <-reqc:
		close(r1c)
	}
	var idlec chan struct{}
	select {
	case err := <-reqerrc:
		t.Fatalf("request 1: got err %v, want nil", err)
	case idlec = <-putidlec:
	}

	wg.Add(1)
	cancelctx, cancel := context.WithCancel(context.Background())
	go func() {
		defer wg.Done()
		req, _ := NewRequestWithContext(cancelctx, "GET", ts.URL, nil)
		res, err := client.Do(req)
		if err == nil {
			res.Body.Close()
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("request 2: got err %v, want Canceled", err)
		}

		// Unblock the first request.
		close(idlec)
	}()

	// Wait for the second request to arrive at the server, and then cancel
	// the request context.
	r2c := <-reqc
	cancel()

	<-idlec

	close(r2c)
	wg.Wait()
}

func TestHandlerAbortRacesBodyRead(t *testing.T) { run(t, testHandlerAbortRacesBodyRead) }
func testHandlerAbortRacesBodyRead(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		go io.Copy(io.Discard, req.Body)
		panic(ErrAbortHandler)
	})).ts

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				const reqLen = 6 * 1024 * 1024
				req, _ := NewRequest("POST", ts.URL, &io.LimitedReader{R: neverEnding('x'), N: reqLen})
				req.ContentLength = reqLen
				resp, _ := ts.Client().Transport.RoundTrip(req)
				if resp != nil {
					resp.Body.Close()
				}
			}
		}()
	}
	wg.Wait()
}

func TestRequestSanitization(t *testing.T) { run(t, testRequestSanitization) }
func testRequestSanitization(t *testing.T, mode testMode) {
	if mode == http2Mode {
		// Remove this after updating x/net.
		t.Skip("https://go.dev/issue/60374 test fails when run with HTTP/2")
	}
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		if h, ok := req.Header["X-Evil"]; ok {
			t.Errorf("request has X-Evil header: %q", h)
		}
	})).ts
	req, _ := NewRequest("GET", ts.URL, nil)
	req.Host = "go.dev\r\nX-Evil:evil"
	resp, _ := ts.Client().Do(req)
	if resp != nil {
		resp.Body.Close()
	}
}

func TestProxyAuthHeader(t *testing.T) {
	// Not parallel: Sets an environment variable.
	run(t, testProxyAuthHeader, []testMode{http1Mode}, testNotParallel)
}
func testProxyAuthHeader(t *testing.T, mode testMode) {
	const username = "u"
	const password = "@/?!"
	cst := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		// Copy the Proxy-Authorization header to a new Request,
		// since Request.BasicAuth only parses the Authorization header.
		var r2 Request
		r2.Header = Header{
			"Authorization": req.Header["Proxy-Authorization"],
		}
		gotuser, gotpass, ok := r2.BasicAuth()
		if !ok || gotuser != username || gotpass != password {
			t.Errorf("req.BasicAuth() = %q, %q, %v; want %q, %q, true", gotuser, gotpass, ok, username, password)
		}
	}))
	u, err := url.Parse(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	u.User = url.UserPassword(username, password)
	t.Setenv("HTTP_PROXY", u.String())
	cst.tr.Proxy = ProxyURL(u)
	resp, err := cst.c.Get("http://_/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
}

// Issue 61708
func TestTransportReqCancelerCleanupOnRequestBodyWriteError(t *testing.T) {
	ln := newLocalListener(t)
	addr := ln.Addr().String()

	done := make(chan struct{})
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			t.Errorf("ln.Accept: %v", err)
			return
		}
		// Start reading request before sending response to avoid
		// "Unsolicited response received on idle HTTP channel" RoundTrip error.
		if _, err := io.ReadFull(conn, make([]byte, 1)); err != nil {
			t.Errorf("conn.Read: %v", err)
			return
		}
		io.WriteString(conn, "HTTP/1.1 200\r\nContent-Length: 3\r\n\r\nfoo")
		<-done
		conn.Close()
	}()

	didRead := make(chan bool)
	SetReadLoopBeforeNextReadHook(func() { didRead <- true })
	defer SetReadLoopBeforeNextReadHook(nil)

	tr := &Transport{}

	// Send a request with a body guaranteed to fail on write.
	req, err := NewRequest("POST", "http://"+addr, io.LimitReader(neverEnding('x'), 1<<30))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}

	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("tr.RoundTrip: %v", err)
	}

	close(done)

	// Before closing response body wait for readLoopDone goroutine
	// to complete due to closed connection by writeLoop.
	<-didRead

	resp.Body.Close()

	// Verify no outstanding requests after readLoop/writeLoop
	// goroutines shut down.
	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		n := tr.NumPendingRequestsForTesting()
		if n > 0 {
			if d > 0 {
				t.Logf("pending requests = %d after %v (want 0)", n, d)
			}
			return false
		}
		return true
	})
}

func TestValidateClientRequestTrailers(t *testing.T) {
	run(t, testValidateClientRequestTrailers)
}

func testValidateClientRequestTrailers(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		rw.Write([]byte("Hello"))
	})).ts

	cases := []struct {
		trailer Header
		wantErr string
	}{
		{Header{"Trx": {"x\r\nX-Another-One"}}, `invalid trailer field value for "Trx"`},
		{Header{"\r\nTrx": {"X-Another-One"}}, `invalid trailer field name "\r\nTrx"`},
	}

	for i, tt := range cases {
		testName := fmt.Sprintf("%s%d", mode, i)
		t.Run(testName, func(t *testing.T) {
			req, err := NewRequest("GET", cst.URL, nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Trailer = tt.trailer
			res, err := cst.Client().Do(req)
			if err == nil {
				t.Fatal("Expected an error")
			}
			if g, w := err.Error(), tt.wantErr; !strings.Contains(g, w) {
				t.Fatalf("Mismatched error\n\t%q\ndoes not contain\n\t%q", g, w)
			}
			if res != nil {
				t.Fatal("Unexpected non-nil response")
			}
		})
	}
}

func TestTransportServerProtocols(t *testing.T) {
	CondSkipHTTP2(t)
	DefaultTransport.(*Transport).CloseIdleConnections()

	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	leafCert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	certpool := x509.NewCertPool()
	certpool.AddCert(leafCert)

	for _, test := range []struct {
		name      string
		scheme    string
		setup     func(t *testing.T)
		transport func(*Transport)
		server    func(*Server)
		want      string
	}{{
		name:   "http default",
		scheme: "http",
		want:   "HTTP/1.1",
	}, {
		name:   "https default",
		scheme: "https",
		transport: func(tr *Transport) {
			// Transport default is HTTP/1.
		},
		want: "HTTP/1.1",
	}, {
		name:   "https transport protocols include HTTP2",
		scheme: "https",
		transport: func(tr *Transport) {
			// Server default is to support HTTP/2, so if the Transport enables
			// HTTP/2 we get it.
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
			tr.Protocols.SetHTTP2(true)
		},
		want: "HTTP/2.0",
	}, {
		name:   "https transport protocols only include HTTP1",
		scheme: "https",
		transport: func(tr *Transport) {
			// Explicitly enable only HTTP/1.
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
		},
		want: "HTTP/1.1",
	}, {
		name:   "https transport ForceAttemptHTTP2",
		scheme: "https",
		transport: func(tr *Transport) {
			// Pre-Protocols-field way of enabling HTTP/2.
			tr.ForceAttemptHTTP2 = true
		},
		want: "HTTP/2.0",
	}, {
		name:   "https transport protocols override TLSNextProto",
		scheme: "https",
		transport: func(tr *Transport) {
			// Setting TLSNextProto to an empty map is the historical way
			// of disabling HTTP/2. Explicitly enabling HTTP2 in the Protocols
			// field takes precedence.
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
			tr.Protocols.SetHTTP2(true)
			tr.TLSNextProto = map[string]func(string, *tls.Conn) RoundTripper{}
		},
		want: "HTTP/2.0",
	}, {
		name:   "https server disables HTTP2 with TLSNextProto",
		scheme: "https",
		server: func(srv *Server) {
			// Disable HTTP/2 on the server with TLSNextProto,
			// use default Protocols value.
			srv.TLSNextProto = map[string]func(*Server, *tls.Conn, Handler){}
		},
		want: "HTTP/1.1",
	}, {
		name:   "https server Protocols overrides empty TLSNextProto",
		scheme: "https",
		server: func(srv *Server) {
			// Explicitly enabling HTTP2 in the Protocols field takes precedence
			// over setting an empty TLSNextProto.
			srv.Protocols = &Protocols{}
			srv.Protocols.SetHTTP1(true)
			srv.Protocols.SetHTTP2(true)
			srv.TLSNextProto = map[string]func(*Server, *tls.Conn, Handler){}
		},
		want: "HTTP/2.0",
	}, {
		name:   "https server protocols only include HTTP1",
		scheme: "https",
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetHTTP1(true)
		},
		want: "HTTP/1.1",
	}, {
		name:   "https server protocols include HTTP2",
		scheme: "https",
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetHTTP1(true)
			srv.Protocols.SetHTTP2(true)
		},
		want: "HTTP/2.0",
	}, {
		name:   "GODEBUG disables HTTP2 client",
		scheme: "https",
		setup: func(t *testing.T) {
			t.Setenv("GODEBUG", "http2client=0")
		},
		transport: func(tr *Transport) {
			// Server default is to support HTTP/2, so if the Transport enables
			// HTTP/2 we get it.
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
			tr.Protocols.SetHTTP2(true)
		},
		want: "HTTP/1.1",
	}, {
		name:   "GODEBUG disables HTTP2 server",
		scheme: "https",
		setup: func(t *testing.T) {
			t.Setenv("GODEBUG", "http2server=0")
		},
		transport: func(tr *Transport) {
			// Server default is to support HTTP/2, so if the Transport enables
			// HTTP/2 we get it.
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
			tr.Protocols.SetHTTP2(true)
		},
		want: "HTTP/1.1",
	}, {
		name:   "unencrypted HTTP2 with prior knowledge",
		scheme: "http",
		transport: func(tr *Transport) {
			tr.Protocols = &Protocols{}
			tr.Protocols.SetUnencryptedHTTP2(true)
		},
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetHTTP1(true)
			srv.Protocols.SetUnencryptedHTTP2(true)
		},
		want: "HTTP/2.0",
	}, {
		name:   "unencrypted HTTP2 only on server",
		scheme: "http",
		transport: func(tr *Transport) {
			tr.Protocols = &Protocols{}
			tr.Protocols.SetUnencryptedHTTP2(true)
		},
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetUnencryptedHTTP2(true)
		},
		want: "HTTP/2.0",
	}, {
		name:   "unencrypted HTTP2 with no server support",
		scheme: "http",
		transport: func(tr *Transport) {
			tr.Protocols = &Protocols{}
			tr.Protocols.SetUnencryptedHTTP2(true)
		},
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetHTTP1(true)
		},
		want: "error",
	}, {
		name:   "HTTP1 with no server support",
		scheme: "http",
		transport: func(tr *Transport) {
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
		},
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetUnencryptedHTTP2(true)
		},
		want: "error",
	}, {
		name:   "HTTPS1 with no server support",
		scheme: "https",
		transport: func(tr *Transport) {
			tr.Protocols = &Protocols{}
			tr.Protocols.SetHTTP1(true)
		},
		server: func(srv *Server) {
			srv.Protocols = &Protocols{}
			srv.Protocols.SetHTTP2(true)
		},
		want: "error",
	}} {
		t.Run(test.name, func(t *testing.T) {
			// We don't use httptest here because it makes its own decisions
			// about how to enable/disable HTTP/2.
			srv := &Server{
				TLSConfig: &tls.Config{
					Certificates: []tls.Certificate{cert},
				},
				Handler: HandlerFunc(func(w ResponseWriter, req *Request) {
					w.Header().Set("X-Proto", req.Proto)
				}),
			}
			tr := &Transport{
				TLSClientConfig: &tls.Config{
					RootCAs: certpool,
				},
			}

			if test.setup != nil {
				test.setup(t)
			}
			if test.server != nil {
				test.server(srv)
			}
			if test.transport != nil {
				test.transport(tr)
			} else {
				tr.Protocols = &Protocols{}
				tr.Protocols.SetHTTP1(true)
				tr.Protocols.SetHTTP2(true)
			}

			listener := newLocalListener(t)
			srvc := make(chan error, 1)
			go func() {
				switch test.scheme {
				case "http":
					srvc <- srv.Serve(listener)
				case "https":
					srvc <- srv.ServeTLS(listener, "", "")
				}
			}()
			t.Cleanup(func() {
				srv.Close()
				<-srvc
			})

			client := &Client{Transport: tr}
			resp, err := client.Get(test.scheme + "://" + listener.Addr().String())
			if err != nil {
				if test.want == "error" {
					return
				}
				t.Fatal(err)
			}
			if got := resp.Header.Get("X-Proto"); got != test.want {
				t.Fatalf("request proto %q, want %q", got, test.want)
			}
		})
	}
}
