// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"context"
	"errors"
	"internal/nettest"
	"net"
	"net/http"
	"slices"
	"sync"
	"testing"
	"testing/synctest"
)

// TestHTTP1TransportTest is an example of using http1TransportTest.
func TestHTTP1TransportTest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tt := newHTTP1TransportTest(t)

		// tt.roundTrip immediately returns a testRoundTrip,
		// which we can use to examine the state of the RoundTrip call.
		sentReq, _ := http.NewRequest("GET", "http://example.tld/request/path", nil)
		rt := tt.roundTrip(sentReq)
		if rt.done() {
			t.Fatalf("RoundTrip unexpectedly returned before reading response")
		}

		// Expect that the Transport dials a new connection.
		// dial.connect provides it with a connection, and gives us the other half.
		dial := tt.wantDial("tcp", "example.tld:80")
		conn := dial.connect()

		// Read the request written by the Transport.
		req := conn.readRequest()
		if got, want := req.URL.Path, sentReq.URL.Path; got != want {
			t.Fatalf("read request path %q, want %q", got, want)
		}

		// Respond, finishing the request.
		conn.writeMessage(
			"HTTP/1.1 200 OK",
			"Content-Length: 0",
			"",
		)
		rt.wantStatus(200)
	})
}

// An http1TransportTest tests an HTTP/1 transport using a fake network.
// It must be used in a synctest bubble.
type http1TransportTest struct {
	t  *testing.T
	tr *http.Transport

	dialsMu sync.Mutex
	dials   []*http1TestDial
}

func newHTTP1TransportTest(t *testing.T) *http1TransportTest {
	tt := &http1TransportTest{
		t:  t,
		tr: &http.Transport{},
	}
	tt.tr.DialContext = (*http1TransportTestDialer)(tt).dialContext
	return tt
}

func (tt *http1TransportTest) roundTrip(req *http.Request) *testRoundTrip {
	return newTestRoundTrip(tt.t, tt.tr, req)
}

func newTestRoundTrip(t *testing.T, roundTripper http.RoundTripper, req *http.Request) *testRoundTrip {
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	rt := &testRoundTrip{
		t:      t,
		donec:  make(chan struct{}),
		cancel: cancel,
	}
	go func() {
		defer close(rt.donec)
		rt.resp, rt.respErr = roundTripper.RoundTrip(req)
	}()
	synctest.Wait()

	t.Cleanup(func() {
		if !rt.done() {
			return
		}
		res, _ := rt.result()
		if res != nil {
			res.Body.Close()
		}
	})

	return rt
}

func (tt *http1TransportTest) newClientConn(scheme, address string) (*http.ClientConn, *http1TestConn) {
	t := tt.t
	t.Helper()

	var (
		clientConn *http.ClientConn
		err        = errors.New("still running")
	)
	go func() {
		clientConn, err = tt.tr.NewClientConn(t.Context(), scheme, address)
	}()
	synctest.Wait()
	netConn := tt.wantDial("tcp", address).connect()
	synctest.Wait()
	if err != nil {
		t.Fatalf("NewClientConn: %v (want success)", err)
	}
	t.Cleanup(func() {
		netConn.conn.Close()
		clientConn.Close()
	})
	return clientConn, netConn
}

func (tt *http1TransportTest) wantDial(network, address string) *http1TestDial {
	tt.t.Helper()
	synctest.Wait()
	tt.dialsMu.Lock()
	defer tt.dialsMu.Unlock()
	for i, dial := range tt.dials {
		if dial.network == network && dial.address == address {
			tt.dials = slices.Delete(tt.dials, i, i+1)
			return dial
		}
	}
	if len(tt.dials) == 0 {
		tt.t.Fatalf("want dial for %q, %q; got none", network, address)
	} else {
		tt.t.Fatalf("want dial for %q, %q; got %q, %q", network, address, tt.dials[0].network, tt.dials[0].address)
	}
	return nil
}

type connOrError struct {
	conn net.Conn
	err  error
}

type http1TestDial struct {
	t       *testing.T
	network string
	address string
	resultc chan connOrError
}

func (dial *http1TestDial) connect() *http1TestConn {
	cliConn, srvConn := nettest.NewConnPair()
	dial.t.Cleanup(func() {
		srvConn.Close()
	})
	dial.resultc <- connOrError{conn: cliConn}
	srvConn.SetReadError(errWouldBlock) // effectively make reads non-blocking
	return &http1TestConn{
		t:    dial.t,
		conn: srvConn,
		bufr: bufio.NewReader(srvConn),
	}
}

type http1TransportTestDialer http1TransportTest

func (tt *http1TransportTestDialer) dialContext(ctx context.Context, network, address string) (net.Conn, error) {
	dial := &http1TestDial{
		t:       tt.t,
		network: network,
		address: address,
		resultc: make(chan connOrError, 1),
	}
	tt.dialsMu.Lock()
	tt.dials = append(tt.dials, dial)
	tt.dialsMu.Unlock()
	select {
	case res := <-dial.resultc:
		return res.conn, res.err
	case <-tt.t.Context().Done():
		return nil, errors.New("test ended")
	}
}

// testRoundTrip manages a RoundTrip in progress.
type testRoundTrip struct {
	t       *testing.T
	resp    *http.Response
	respErr error
	donec   chan struct{}
	cancel  context.CancelFunc
}

// done reports whether RoundTrip has returned.
func (rt *testRoundTrip) done() bool {
	synctest.Wait()
	select {
	case <-rt.donec:
		return true
	default:
		return false
	}
}

// result returns the result of the RoundTrip.
func (rt *testRoundTrip) result() (*http.Response, error) {
	t := rt.t
	t.Helper()
	synctest.Wait()
	select {
	case <-rt.donec:
	default:
		t.Fatalf("RoundTrip is not done; want it to be")
	}
	return rt.resp, rt.respErr
}

// response returns the response of a successful RoundTrip.
// If the RoundTrip unexpectedly failed, it calls t.Fatal.
func (rt *testRoundTrip) response() *http.Response {
	t := rt.t
	t.Helper()
	resp, err := rt.result()
	if err != nil {
		t.Fatalf("RoundTrip returned unexpected error: %v", rt.respErr)
	}
	if resp == nil {
		t.Fatalf("RoundTrip returned nil *Response and nil error")
	}
	return resp
}

// err returns the (possibly nil) error result of RoundTrip.
func (rt *testRoundTrip) err() error {
	t := rt.t
	t.Helper()
	_, err := rt.result()
	return err
}

// wantStatus indicates the expected response StatusCode.
func (rt *testRoundTrip) wantStatus(want int) {
	t := rt.t
	t.Helper()
	if got := rt.response().StatusCode; got != want {
		t.Fatalf("got response status %v, want %v", got, want)
	}
}
