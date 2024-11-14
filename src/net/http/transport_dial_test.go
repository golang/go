// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"testing"
)

func TestTransportPoolConnReusePriorConnection(t *testing.T) {
	dt := newTransportDialTester(t, http1Mode)

	// First request creates a new connection.
	rt1 := dt.roundTrip()
	c1 := dt.wantDial()
	c1.finish(nil)
	rt1.wantDone(c1)
	rt1.finish()

	// Second request reuses the first connection.
	rt2 := dt.roundTrip()
	rt2.wantDone(c1)
	rt2.finish()
}

func TestTransportPoolConnCannotReuseConnectionInUse(t *testing.T) {
	dt := newTransportDialTester(t, http1Mode)

	// First request creates a new connection.
	rt1 := dt.roundTrip()
	c1 := dt.wantDial()
	c1.finish(nil)
	rt1.wantDone(c1)

	// Second request is made while the first request is still using its connection,
	// so it goes on a new connection.
	rt2 := dt.roundTrip()
	c2 := dt.wantDial()
	c2.finish(nil)
	rt2.wantDone(c2)
}

func TestTransportPoolConnConnectionBecomesAvailableDuringDial(t *testing.T) {
	dt := newTransportDialTester(t, http1Mode)

	// First request creates a new connection.
	rt1 := dt.roundTrip()
	c1 := dt.wantDial()
	c1.finish(nil)
	rt1.wantDone(c1)

	// Second request is made while the first request is still using its connection.
	// The first connection completes while the second Dial is in progress, so the
	// second request uses the first connection.
	rt2 := dt.roundTrip()
	c2 := dt.wantDial()
	rt1.finish()
	rt2.wantDone(c1)

	// This section is a bit overfitted to the current Transport implementation:
	// A third request starts. We have an in-progress dial that was started by rt2,
	// but this new request (rt3) is going to ignore it and make a dial of its own.
	// rt3 will use the first of these dials that completes.
	rt3 := dt.roundTrip()
	c3 := dt.wantDial()
	c2.finish(nil)
	rt3.wantDone(c2)

	c3.finish(nil)
}

// A transportDialTester manages a test of a connection's Dials.
type transportDialTester struct {
	t   *testing.T
	cst *clientServerTest

	dials chan *transportDialTesterConn // each new conn is sent to this channel

	roundTripCount int
	dialCount      int
}

// A transportDialTesterRoundTrip is a RoundTrip made as part of a dial test.
type transportDialTesterRoundTrip struct {
	t *testing.T

	roundTripID int                // distinguishes RoundTrips in logs
	cancel      context.CancelFunc // cancels the Request context
	reqBody     io.WriteCloser     // write half of the Request.Body
	finished    bool

	done chan struct{} // closed when RoundTrip returns:w
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

	net.Conn
}

func newTransportDialTester(t *testing.T, mode testMode) *transportDialTester {
	t.Helper()
	dt := &transportDialTester{
		t:     t,
		dials: make(chan *transportDialTesterConn),
	}
	dt.cst = newClientServerTest(t, mode, http.HandlerFunc(func { w, r ->
		// Write response headers when we receive a request.
		http.NewResponseController(w).EnableFullDuplex()
		w.WriteHeader(200)
		http.NewResponseController(w).Flush()
		// Wait for the client to send the request body,
		// to synchronize with the rest of the test.
		io.ReadAll(r.Body)
	}), func { tr ->
		tr.DialContext = func { ctx, network, address ->
			c := &transportDialTesterConn{
				t:     t,
				ready: make(chan error),
			}
			// Notify the test that a Dial has started,
			// and wait for the test to notify us that it should complete.
			dt.dials <- c
			if err := <-c.ready; err != nil {
				return nil, err
			}
			nc, err := net.Dial(network, address)
			if err != nil {
				return nil, err
			}
			// Use the *transportDialTesterConn as the net.Conn,
			// to let tests associate requests with connections.
			c.Conn = nc
			return c, err
		}
	})
	return dt
}

// roundTrip starts a RoundTrip.
// It returns immediately, without waiting for the RoundTrip call to complete.
func (dt *transportDialTester) roundTrip() *transportDialTesterRoundTrip {
	dt.t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()
	rt := &transportDialTesterRoundTrip{
		t:           dt.t,
		roundTripID: dt.roundTripCount,
		done:        make(chan struct{}),
		reqBody:     pw,
		cancel:      cancel,
	}
	dt.roundTripCount++
	dt.t.Logf("RoundTrip %v: started", rt.roundTripID)
	dt.t.Cleanup(func() {
		rt.cancel()
		rt.finish()
	})
	go func() {
		ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
			GotConn: func(info httptrace.GotConnInfo) {
				rt.conn = info.Conn.(*transportDialTesterConn)
			},
		})
		req, _ := http.NewRequestWithContext(ctx, "POST", dt.cst.ts.URL, pr)
		req.Header.Set("Content-Type", "text/plain")
		rt.res, rt.err = dt.cst.tr.RoundTrip(req)
		dt.t.Logf("RoundTrip %v: done (err:%v)", rt.roundTripID, rt.err)
		close(rt.done)
	}()
	return rt
}

// wantDone indicates that a RoundTrip should have returned.
func (rt *transportDialTesterRoundTrip) wantDone(c *transportDialTesterConn) {
	rt.t.Helper()
	<-rt.done
	if rt.err != nil {
		rt.t.Fatalf("RoundTrip %v: want success, got err %v", rt.roundTripID, rt.err)
	}
	if rt.conn != c {
		rt.t.Fatalf("RoundTrip %v: want on conn %v, got conn %v", rt.roundTripID, c.connID, rt.conn.connID)
	}
}

// finish completes a RoundTrip by sending the request body, consuming the response body,
// and closing the response body.
func (rt *transportDialTesterRoundTrip) finish() {
	rt.t.Helper()

	if rt.finished {
		return
	}
	rt.finished = true

	<-rt.done

	if rt.err != nil {
		return
	}
	rt.reqBody.Close()
	io.ReadAll(rt.res.Body)
	rt.res.Body.Close()
	rt.t.Logf("RoundTrip %v: closed request body", rt.roundTripID)
}

// wantDial waits for the Transport to start a Dial.
func (dt *transportDialTester) wantDial() *transportDialTesterConn {
	c := <-dt.dials
	c.connID = dt.dialCount
	dt.dialCount++
	dt.t.Logf("Dial %v: started", c.connID)
	return c
}

// finish completes a Dial.
func (c *transportDialTesterConn) finish(err error) {
	c.t.Logf("Dial %v: finished (err:%v)", c.connID, err)
	c.ready <- err
	close(c.ready)
}
