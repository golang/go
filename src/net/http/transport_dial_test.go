// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
)

// Successive requests use the same HTTP/1 connection.
func TestTransportPoolConnReusePriorConnection(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode)

		// First request creates a new connection.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/1.1")
		rt1.finish()

		// Second request reuses the first connection.
		rt2 := dt.roundTrip()
		rt2.wantDone(c1, "HTTP/1.1")
		rt2.finish()
	})
}

// Two HTTP/1 requests made at the same time use different connections.
func TestTransportPoolConnCannotReuseConnectionInUse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode)

		// First request creates a new connection.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/1.1")

		// Second request is made while the first request is still using its connection,
		// so it goes on a new connection.
		rt2 := dt.roundTrip()
		c2 := dt.wantDial()
		c2.finish(nil)
		rt2.wantDone(c2, "HTTP/1.1")
	})
}

// When an HTTP/2 connection is at its stream limit
// a new request is made on a new connection.
func TestTransportPoolConnHTTP2OverStreamLimit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http2Mode, func(srv *http.Server) {
			srv.HTTP2 = &http.HTTP2Config{
				MaxConcurrentStreams: 2,
			}
		})

		// First request dials an HTTP/2 connection.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/2.0")

		// Second request uses the existing connection.
		rt2 := dt.roundTrip()
		rt2.wantDone(c1, "HTTP/2.0")

		// Third request creates a new connection
		rt3 := dt.roundTrip()
		c2 := dt.wantDial()
		c2.finish(nil)
		rt3.wantDone(c2, "HTTP/2.0")

		rt1.finish()
		rt2.finish()
		rt3.finish()

		// With slots available on both connections, we prefer the oldest.
		rt4 := dt.roundTrip()
		rt4.wantDone(c1, "HTTP/2.0")
		rt5 := dt.roundTrip()
		rt5.wantDone(c1, "HTTP/2.0")
		rt6 := dt.roundTrip()
		rt6.wantDone(c2, "HTTP/2.0")
		rt4.finish()
		rt5.finish()
		rt6.finish()
	})
}

// A new request made while an HTTP/2 dial is in progress will start a second dial.
func TestTransportPoolConnHTTP2Startup(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http2Mode, func(srv *http.Server) {})

		// Two requests start.
		// Since the second request starts before the first dial finishes, it starts a second dial.
		rt1 := dt.roundTrip()
		rt2 := dt.roundTrip()
		c1 := dt.wantDial()
		c2 := dt.wantDial()

		// Both requests use the conn of the first dial to complete.
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/2.0")
		rt2.wantDone(c1, "HTTP/2.0")

		rt1.finish()
		rt2.finish()
		c2.finish(nil)
	})
}

// When a request finishes using an HTTP/1 connection,
// a pending request attempting to dial a new connection will use the newly-available one.
func TestTransportPoolConnConnectionBecomesAvailableDuringDial(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode)

		// First request creates a new connection.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/1.1")

		// Second request is made while the first request is still using its connection.
		// The first connection completes while the second Dial is in progress, so the
		// second request uses the first connection.
		rt2 := dt.roundTrip()
		c2 := dt.wantDial()
		rt1.finish()
		rt2.wantDone(c1, "HTTP/1.1")

		// This section is a bit overfitted to the current Transport implementation:
		// A third request starts. We have an in-progress dial that was started by rt2,
		// but this new request (rt3) is going to ignore it and make a dial of its own.
		// rt3 will use the first of these dials that completes.
		rt3 := dt.roundTrip()
		c3 := dt.wantDial()
		c2.finish(nil)
		rt3.wantDone(c2, "HTTP/1.1")

		c3.finish(nil)
	})
}

// Connections are not reused when DisableKeepAlives = true.
func TestTransportPoolDisableKeepAlives(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode, func(tr *http.Transport) {
			tr.DisableKeepAlives = true
		})

		// Two requests, each uses a separate connection.
		for range 2 {
			rt := dt.roundTrip()
			c := dt.wantDial()
			c.finish(nil)
			rt.wantDone(c, "HTTP/1.1")
			rt.finish()
		}
	})
}

// Canceling a request before its connection is created returns the conn to the pool.
func TestTransportPoolCancelRequestReusesConn(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode)

		// First request is canceled before its connection is created.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		rt1.cancel()
		rt1.wantError()

		// Second request uses the first connection.
		rt2 := dt.roundTrip()
		c2 := dt.wantDial()
		c1.finish(nil) // first dial finishes
		rt2.wantDone(c1, "HTTP/1.1")
		rt2.finish()

		c2.finish(nil) // second dial finishes
	})
}

// Connections are not reused when DisableKeepAlives = true.
func TestTransportPoolCancelRequestWithDisableKeepAlives(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode, func(tr *http.Transport) {
			tr.DisableKeepAlives = true
		})

		// First request is canceled before its connection is created.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		rt1.cancel()
		rt1.wantError()

		// Dial finishes. DisableKeepAlives = true, so we discard the connection.
		c1.finish(nil)

		// Second request is made on a new connection.
		rt2 := dt.roundTrip()
		c2 := dt.wantDial()
		c2.finish(nil)
		rt2.wantDone(c2, "HTTP/1.1")
		rt2.finish()
	})
}

// Connections are not reused after an error.
func TestTransportPoolConnectionBroken(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode)

		// First request creates a new connection.
		// The connection breaks while sending the response.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/1.1")
		c1.fakeNetConn.Close() // break the connection
		rt1.finish()

		// Second request is made on a new connection, since the first is broken.
		rt2 := dt.roundTrip()
		c2 := dt.wantDial()
		c2.finish(nil)
		rt2.wantDone(c2, "HTTP/1.1")
		c2.fakeNetConn.Close()
		rt2.finish()
	})
}

// MaxIdleConnsPerHost limits the number of idle connections.
func TestTransportPoolClosesConnsPastMaxIdleConnsPerHost(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		dt := newTransportDialTester(t, http1Mode, func(tr *http.Transport) {
			tr.MaxIdleConnsPerHost = 1
		})

		// First request creates a new connection.
		rt1 := dt.roundTrip("host1.fake.tld")
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/1.1")

		// Second request also creates a new connection.
		rt2 := dt.roundTrip("host1.fake.tld")
		c2 := dt.wantDial()
		c2.finish(nil)
		rt2.wantDone(c2, "HTTP/1.1")

		// Third request is to a different host.
		rt3 := dt.roundTrip("host2.fake.tld")
		c3 := dt.wantDial()
		c3.finish(nil)
		rt3.wantDone(c3, "HTTP/1.1")

		// All requests finish. One conn is in excess of MaxIdleConnsPerHost, and is closed.
		rt3.finish()
		rt2.finish()
		rt1.finish()
		c1.wantClosed()

		// Additional requests reuse the remaining connections.
		rt4 := dt.roundTrip("host1.fake.tld")
		rt4.wantDone(c2, "HTTP/1.1")
		rt4.finish()
		rt5 := dt.roundTrip("host2.fake.tld")
		rt5.wantDone(c3, "HTTP/1.1")
		rt5.finish()
	})
}

// Current (but probably wrong) behavior:
// MaxIdleConnsPerHost doesn't apply to HTTP/2 connections.
func TestTransportPoolMaxIdleConnsPerHostHTTP2(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		t.Skip("skipped until h2_bundle.go includes support for MaxConcurrentStreams")

		dt := newTransportDialTester(t, http2Mode, func(srv *http.Server) {
			srv.HTTP2 = &http.HTTP2Config{
				MaxConcurrentStreams: 1,
			}
		}, func(tr *http.Transport) {
			tr.MaxIdleConnsPerHost = 1
		})

		// First request creates a new connection.
		rt1 := dt.roundTrip()
		c1 := dt.wantDial()
		c1.finish(nil)
		rt1.wantDone(c1, "HTTP/2.0")

		// Second request also creates a new connection.
		rt2 := dt.roundTrip()
		c2 := dt.wantDial()
		c2.finish(nil)
		rt2.wantDone(c2, "HTTP/2.0")

		// Both requests finish.
		// We have two idle conns for this host, but we keep them both.
		rt1.finish()
		rt2.finish()

		// Two new requests use the existing connections.
		rt3 := dt.roundTrip()
		rt3.wantDone(c1, "HTTP/2.0")
		rt4 := dt.roundTrip()
		rt4.wantDone(c2, "HTTP/2.0")
	})
}

// A transportDialTester manages a test of a connection's Dials.
type transportDialTester struct {
	t   *testing.T
	cst *clientServerTest

	dialsMu sync.Mutex
	dials   []*transportDialTesterConn

	roundTripCount int
	dialCount      int
}

// A transportDialTesterRoundTrip is a RoundTrip made as part of a dial test.
type transportDialTesterRoundTrip struct {
	t *testing.T

	roundTripID    int                // distinguishes RoundTrips in logs
	cancel         context.CancelFunc // cancels the Request context
	reqBody        io.WriteCloser     // write half of the Request.Body
	respBodyClosed bool               // set when the user calls Response.Body.Close
	returned       bool               // set when RoundTrip returns

	res  *http.Response
	err  error
	conn *transportDialTesterConn
}

// A transportDialTesterConn is a client connection created by the Transport as
// part of a dial test.
type transportDialTesterConn struct {
	t *testing.T

	connID int        // distinguished Dials in logs
	ready  chan error // sent on to complete the Dial
	protos []string
	closed chan struct{}

	*fakeNetConn
}

func newTransportDialTester(t *testing.T, mode testMode, opts ...any) *transportDialTester {
	t.Helper()
	dt := &transportDialTester{
		t: t,
	}
	dialContext := func(_ context.Context, network, address string) (*transportDialTesterConn, error) {
		c := &transportDialTesterConn{
			t:      t,
			ready:  make(chan error),
			closed: make(chan struct{}),
		}
		// Notify the test that a Dial has started,
		// and wait for the test to notify us that it should complete.
		dt.dialsMu.Lock()
		dt.dials = append(dt.dials, c)
		dt.dialsMu.Unlock()

		select {
		case err := <-c.ready:
			if err != nil {
				return nil, err
			}
		case <-t.Context().Done():
			t.Errorf("test finished with dial in progress")
			return nil, errors.New("test finished")
		}

		c.fakeNetConn = dt.cst.li.connect()
		t.Cleanup(func() {
			c.fakeNetConn.Close()
		})
		// Use the *transportDialTesterConn as the net.Conn,
		// to let tests associate requests with connections.
		return c, nil
	}
	dt.cst = newClientServerTest(t, mode, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Write response headers when we receive a request.
		http.NewResponseController(w).EnableFullDuplex()
		w.WriteHeader(200)
		http.NewResponseController(w).Flush()
		// Wait for the client to send the request body,
		// to synchronize with the rest of the test.
		io.ReadAll(r.Body)
	}), append([]any{optFakeNet, func(tr *http.Transport) {
		tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialContext(ctx, network, dt.cst.ts.Listener.Addr().String())
		}
		tr.DialTLSContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := dialContext(ctx, network, dt.cst.ts.Listener.Addr().String())
			if err != nil {
				return nil, err
			}
			config := &tls.Config{
				InsecureSkipVerify: true,
				NextProtos:         []string{"h2", "http/1.1"},
			}
			if conn.protos != nil {
				config.NextProtos = conn.protos
			}
			tc := tls.Client(conn, config)
			if err := tc.Handshake(); err != nil {
				return nil, err
			}
			return tc, nil
		}
	}}, opts...)...)
	return dt
}

// roundTrip starts a RoundTrip.
// It returns immediately, without waiting for the RoundTrip call to complete.
func (dt *transportDialTester) roundTrip(opts ...any) *transportDialTesterRoundTrip {
	dt.t.Helper()
	host := "fake.tld"
	for _, o := range opts {
		switch o := o.(type) {
		case string:
			host = o
		default:
			dt.t.Fatalf("unknown option type %T", o)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()
	dt.roundTripCount++
	rt := &transportDialTesterRoundTrip{
		t:           dt.t,
		roundTripID: dt.roundTripCount,
		reqBody:     pw,
		cancel:      cancel,
	}
	dt.t.Logf("RoundTrip %v: started", rt.roundTripID)
	dt.t.Cleanup(func() {
		rt.cancel()
		rt.finish()
	})
	go func() {
		ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
			GotConn: func(info httptrace.GotConnInfo) {
				c := info.Conn
				if tlsConn, ok := c.(*tls.Conn); ok {
					c = tlsConn.NetConn()
				}
				rt.conn = c.(*transportDialTesterConn)
			},
		})
		proto, _, _ := strings.Cut(dt.cst.ts.URL, ":")
		req, _ := http.NewRequestWithContext(ctx, "POST", proto+"://"+host, pr)
		req.Header.Set("Content-Type", "text/plain")
		rt.res, rt.err = dt.cst.tr.RoundTrip(req)
		dt.t.Logf("RoundTrip %v: done (err:%v)", rt.roundTripID, rt.err)
		rt.returned = true
	}()
	return rt
}

// wantDone indicates that a RoundTrip should have returned.
func (rt *transportDialTesterRoundTrip) wantDone(c *transportDialTesterConn, wantProto string) {
	rt.t.Helper()
	synctest.Wait()
	if !rt.returned {
		rt.t.Fatalf("RoundTrip %v: still running, want to have returned", rt.roundTripID)
	}
	if rt.err != nil {
		rt.t.Fatalf("RoundTrip %v: want success, got err %v", rt.roundTripID, rt.err)
	}
	if rt.conn != c {
		rt.t.Fatalf("RoundTrip %v: want on conn %v, got conn %v", rt.roundTripID, c.connID, rt.conn.connID)
	}
	if got, want := rt.conn, c; got != want {
		rt.t.Fatalf("RoundTrip %v: sent on conn %v, want conn %v", rt.roundTripID, got.connID, want.connID)
	}
	if got, want := rt.res.Proto, wantProto; got != want {
		rt.t.Fatalf("RoundTrip %v: got protocol %q, want %q", rt.roundTripID, got, want)
	}
}

// wantError indicates that a RoundTrip should have returned with an error.
func (rt *transportDialTesterRoundTrip) wantError() {
	rt.t.Helper()
	synctest.Wait()
	if !rt.returned {
		rt.t.Fatalf("RoundTrip %v: still running, want to have returned", rt.roundTripID)
	}
	if rt.err == nil {
		rt.t.Fatalf("RoundTrip %v: success, want error", rt.roundTripID)
	}
}

// finish completes a RoundTrip by sending the request body, consuming the response body,
// and closing the response body.
func (rt *transportDialTesterRoundTrip) finish() {
	rt.t.Helper()

	synctest.Wait()
	if !rt.returned {
		rt.t.Fatalf("RoundTrip %v: still running, want to have returned", rt.roundTripID)
	}
	if rt.err != nil {
		return
	}

	if rt.respBodyClosed {
		return
	}
	rt.respBodyClosed = true
	rt.reqBody.Close()
	io.ReadAll(rt.res.Body)
	rt.res.Body.Close()
	rt.t.Logf("RoundTrip %v: closed request body", rt.roundTripID)
}

// wantDial waits for the Transport to start a Dial.
func (dt *transportDialTester) wantDial() *transportDialTesterConn {
	dt.t.Helper()
	synctest.Wait()
	dt.dialsMu.Lock()
	defer dt.dialsMu.Unlock()
	if len(dt.dials) == 0 {
		dt.t.Fatalf("no dial started, want one")
	}
	c := dt.dials[0]
	dt.dials = dt.dials[1:]
	dt.dialCount++
	c.connID = dt.dialCount
	dt.t.Logf("Dial %v: started", c.connID)
	return c
}

// finish completes a Dial.
func (c *transportDialTesterConn) finish(err error) {
	c.t.Helper()
	c.t.Logf("Dial %v: finished (err:%v)", c.connID, err)
	c.ready <- err
	close(c.ready)
}

func (c *transportDialTesterConn) wantClosed() {
	c.t.Helper()
	<-c.closed
}

func (c *transportDialTesterConn) Close() error {
	select {
	case <-c.closed:
	default:
		c.t.Logf("Conn %v: closed", c.connID)
		close(c.closed)
	}
	return nil
}
