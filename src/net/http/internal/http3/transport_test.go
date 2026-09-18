// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net"
	"net/http"
	"reflect"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	"golang.org/x/net/quic"
)

// unusablePacketConn is a net.PacketConn whose LocalAddr is not a valid
// UDP address, which makes quic.NewEndpoint fail.
type unusablePacketConn struct {
	net.PacketConn
	closed bool
}

func (c *unusablePacketConn) LocalAddr() net.Addr {
	return &net.UnixAddr{Name: "unusable", Net: "unix"}
}

func (c *unusablePacketConn) Close() error {
	c.closed = true
	return nil
}

// TestTransportInitEndpointError verifies that a transport which fails to
// create its QUIC endpoint reports the error, rather than proceeding with a
// nil endpoint.
func TestTransportInitEndpointError(t *testing.T) {
	conn := &unusablePacketConn{}
	tr := &transport{
		tr1: new(http.Transport), // TLSClientConfig is nil, as by default
		opts: TransportOpts{
			ListenPacket: func(network, addr string) (net.PacketConn, error) {
				return conn, nil
			},
		},
		activeConns: make(map[*clientConn]struct{}),
	}

	if err := tr.initEndpoint(); err == nil {
		t.Fatal("initEndpoint() = nil, want error")
	}
	if tr.endpoint != nil {
		t.Errorf("after failed initEndpoint, transport.endpoint = %v, want nil", tr.endpoint)
	}
	if !conn.closed {
		t.Errorf("after failed initEndpoint, the net.PacketConn was not closed")
	}

	// dial must report the error rather than panicking on the nil endpoint.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := tr.dial(ctx, "127.0.0.1:443", &tls.Config{}, nil); err == nil {
		t.Fatal("dial() = nil error, want error")
	}
}

func TestTransportServerCreatesBidirectionalStream(t *testing.T) {
	// "Clients MUST treat receipt of a server-initiated bidirectional
	// stream as a connection error of type H3_STREAM_CREATION_ERROR [...]"
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.1-3
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()
		st := tc.newStream(streamTypeRequest)
		st.Flush()
		tc.wantClosed("after server creates bidi stream", errH3StreamCreationError)
	})
}

func TestClientConnMethods(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var called bool
		hook := func() {
			if called {
				t.Error("state hook was unexpectedly called")
			}
			called = true
		}
		verifyHookWasCalled := func() {
			if !called {
				t.Error("state hook was unexpectedly not called")
			}
			called = false
		}
		tc := newTestClientConnWithHook(t, hook)
		tc.greet()

		// Initial state after establishing connection.
		if err := tc.cc.Err(); err != nil {
			t.Errorf("cc.Err() = %v, want nil", err)
		}
		if avail := tc.cc.Available(); avail != math.MaxInt {
			t.Errorf("cc.Available() = %v, want %v", avail, math.MaxInt)
		}
		if inFlight := tc.cc.InFlight(); inFlight != 0 {
			t.Errorf("cc.InFlight() = %v, want 0", inFlight)
		}

		// Release, with Reserve before and without.
		for _, reserveBefore := range []bool{true, false} {
			if reserveBefore {
				if err := tc.cc.Reserve(); err != nil {
					t.Fatalf("cc.Reserve() failed: %v", err)
				}
				if avail := tc.cc.Available(); avail != math.MaxInt {
					t.Errorf("after Reserve, cc.Available() = %v, want %v", avail, math.MaxInt)
				}
				if inFlight := tc.cc.InFlight(); inFlight != 1 {
					t.Errorf("after Reserve, cc.InFlight() = %v, want 1", inFlight)
				}
			}
			tc.cc.Release()
			if avail := tc.cc.Available(); avail != math.MaxInt {
				t.Errorf("after Release, cc.Available() = %v, want %v", avail, math.MaxInt)
			}
			if inFlight := tc.cc.InFlight(); inFlight != 0 {
				t.Errorf("after Release, cc.InFlight() = %v, want 0", inFlight)
			}
		}

		// RoundTrip, with Reserve before and without.
		for _, reserveBefore := range []bool{true, false} {
			if reserveBefore {
				if err := tc.cc.Reserve(); err != nil {
					t.Fatalf("cc.Reserve() failed: %v", err)
				}
				if avail := tc.cc.Available(); avail != math.MaxInt {
					t.Errorf("after Reserve, cc.Available() = %v, want %v", avail, math.MaxInt)
				}
				if inFlight := tc.cc.InFlight(); inFlight != 1 {
					t.Errorf("after Reserve, cc.InFlight() = %v, want 1", inFlight)
				}
			}
			req, _ := http.NewRequest("GET", "https://example.com/", nil)
			rt := tc.roundTrip(req)
			if avail := tc.cc.Available(); avail != math.MaxInt {
				t.Errorf("after RoundTrip, cc.Available() = %v, want %v", avail, math.MaxInt)
			}
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)
			st.writeHeaders(http.Header{":status": []string{"200"}})
			resp := rt.response()
			if resp.StatusCode != 200 {
				t.Errorf("resp.StatusCode = %v, want 200", resp.StatusCode)
			}
			if inFlight := tc.cc.InFlight(); inFlight != 1 { // InFlight should decrement only after the body is closed.
				t.Errorf("before body close, cc.InFlight() = %v, want 1", inFlight)
			}
			resp.Body.Close()
			if inFlight := tc.cc.InFlight(); inFlight != 0 {
				t.Errorf("after body close, cc.InFlight() = %v, want 0", inFlight)
			}
			verifyHookWasCalled()
		}

		// Connection closure.
		if err := tc.cc.Reserve(); err != nil {
			t.Fatalf("cc.Reserve() failed: %v", err)
		}
		tc.cc.Close()
		synctest.Wait()
		verifyHookWasCalled()
		if err := tc.cc.Err(); err == nil {
			t.Error("after connection is closed, cc.Err() = nil, want err")
		}
		if avail := tc.cc.Available(); avail != 0 {
			t.Errorf("after connection is closed, cc.Available() = %v, want 0", avail)
		}
		if inFlight := tc.cc.InFlight(); inFlight != 0 {
			t.Errorf("after connection is closed, cc.InFlight() = %v, want 0", inFlight)
		}
	})
}

// A testQUICConn wraps a *quic.Conn and provides methods for inspecting it.
type testQUICConn struct {
	t       testing.TB
	qconn   *quic.Conn
	streams map[streamType][]*testQUICStream
}

func newTestQUICConn(t testing.TB, qconn *quic.Conn) *testQUICConn {
	tq := &testQUICConn{
		t:       t,
		qconn:   qconn,
		streams: make(map[streamType][]*testQUICStream),
	}

	go tq.acceptStreams(t.Context())

	t.Cleanup(func() {
		tq.qconn.Close()
	})
	return tq
}

func (tq *testQUICConn) acceptStreams(ctx context.Context) {
	for {
		qst, err := tq.qconn.AcceptStream(ctx)
		if err != nil {
			return
		}
		st := newStream(qst)
		stype := streamTypeRequest
		if qst.IsReadOnly() {
			v, err := st.readVarint()
			if err != nil {
				tq.t.Errorf("error reading stream type from unidirectional stream: %v", err)
				continue
			}
			stype = streamType(v)
		}
		tq.streams[stype] = append(tq.streams[stype], newTestQUICStream(tq.t, st))
	}
}

func (tq *testQUICConn) newStream(stype streamType) *testQUICStream {
	tq.t.Helper()
	var qs *quic.Stream
	var err error
	if stype == streamTypeRequest {
		qs, err = tq.qconn.NewStream(canceledCtx)
	} else {
		qs, err = tq.qconn.NewSendOnlyStream(canceledCtx)
	}
	if err != nil {
		tq.t.Fatal(err)
	}
	st := newStream(qs)
	if stype != streamTypeRequest {
		st.writeVarint(int64(stype))
		if err := st.Flush(); err != nil {
			tq.t.Fatal(err)
		}
	}
	return newTestQUICStream(tq.t, st)
}

// wantNotClosed asserts that the peer has not closed the connection.
func (tq *testQUICConn) wantNotClosed(reason string) {
	t := tq.t
	t.Helper()
	synctest.Wait()
	err := tq.qconn.Wait(canceledCtx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("%v: want QUIC connection to be alive; closed with error: %v", reason, err)
	}
}

// wantClosed asserts that the peer has closed the connection
// with the provided error code.
func (tq *testQUICConn) wantClosed(reason string, want error) {
	t := tq.t
	t.Helper()
	synctest.Wait()

	if e, ok := want.(http3Error); ok {
		want = &quic.ConnectionCloseError{Code: uint64(e)}
	}
	got := tq.qconn.Wait(canceledCtx)
	if errors.Is(got, context.Canceled) {
		t.Fatalf("%v: want QUIC connection closed, but it is not", reason)
	}
	if !errors.Is(got, want) {
		t.Fatalf("%v: connection closed with error: %v; want %v", reason, got, want)
	}
}

// wantStream asserts that a stream of a given type has been created,
// and returns that stream.
func (tq *testQUICConn) wantStream(stype streamType) *testQUICStream {
	tq.t.Helper()
	synctest.Wait()
	if len(tq.streams[stype]) == 0 {
		tq.t.Fatalf("expected a %v stream to be created, but none were", stype)
	}
	ts := tq.streams[stype][0]
	tq.streams[stype] = tq.streams[stype][1:]
	return ts
}

// testQUICStream wraps a QUIC stream and provides methods for inspecting it.
type testQUICStream struct {
	t testing.TB
	*stream
}

func newTestQUICStream(t testing.TB, st *stream) *testQUICStream {
	st.stream.SetReadContext(canceledCtx)
	st.stream.SetWriteContext(canceledCtx)
	return &testQUICStream{
		t:      t,
		stream: st,
	}
}

func (ts *testQUICStream) wantIdle(reason string) {
	ts.t.Helper()
	synctest.Wait()
	qs := ts.stream.stream
	qs.SetReadContext(canceledCtx)
	if _, err := qs.Read(make([]byte, 1)); !errors.Is(err, context.Canceled) {
		ts.t.Fatalf("%v: want stream to be idle, but stream has content", reason)
	}
	qs.SetReadContext(context.Background())
}

// wantFrameHeader calls readFrameHeader and asserts that the frame is of a given type.
func (ts *testQUICStream) wantFrameHeader(reason string, wantType frameType) {
	ts.t.Helper()
	synctest.Wait()
	gotType, err := ts.readFrameHeader()
	if err != nil {
		ts.t.Fatalf("%v: failed to read frame header: %v", reason, err)
	}
	if gotType != wantType {
		ts.t.Fatalf("%v: got frame type %v, want %v", reason, gotType, wantType)
	}
}

// wantHeaders reads a HEADERS frame.
// If want is nil, the contents of the frame are ignored.
func (ts *testQUICStream) wantHeaders(want http.Header) {
	ts.t.Helper()
	synctest.Wait()
	ftype, err := ts.readFrameHeader()
	if err != nil {
		ts.t.Fatalf("want HEADERS frame, got error: %v", err)
	}
	if ftype != frameTypeHeaders {
		ts.t.Fatalf("want HEADERS frame, got: %v", ftype)
	}

	if want == nil {
		if err := ts.discardFrame(); err != nil {
			ts.t.Fatalf("discardFrame: %v", err)
		}
		return
	}

	got := make(http.Header)
	var dec qpackDecoder
	err = dec.decode(ts.stream, func(_ indexType, name, value string) error {
		got.Add(name, value)
		return nil
	})
	if diff := diffHeaders(got, want); diff != "" {
		ts.t.Fatalf("unexpected response headers:\n%v", diff)
	}
	if err := ts.endFrame(); err != nil {
		ts.t.Fatalf("endFrame: %v", err)
	}
}

// wantSomeHeaders reads a HEADERS frame and asserts that want is a subset of
// the read HEADERS frame.
// This is like wantHeaders, but headers that are in the HEADERS frame but not
// in want are ignored.
func (ts *testQUICStream) wantSomeHeaders(want http.Header) {
	ts.t.Helper()
	synctest.Wait()
	ftype, err := ts.readFrameHeader()
	if err != nil {
		ts.t.Fatalf("want HEADERS frame, got error: %v", err)
	}
	if ftype != frameTypeHeaders {
		ts.t.Fatalf("want HEADERS frame, got: %v", ftype)
	}

	if want == nil {
		panic("use wantHeaders(nil) instead to ignore the content of the frame")
	}

	got := make(http.Header)
	var dec qpackDecoder
	err = dec.decode(ts.stream, func(_ indexType, name, value string) error {
		got.Add(name, value)
		return nil
	})
	for name := range got {
		if _, ok := want[name]; !ok {
			delete(got, name)
		}
	}
	if diff := diffHeaders(got, want); diff != "" {
		ts.t.Fatalf("unexpected response headers:\n%v", diff)
	}
	if err := ts.endFrame(); err != nil {
		ts.t.Fatalf("endFrame: %v", err)
	}
}

func (ts *testQUICStream) encodeHeaders(h http.Header) []byte {
	ts.t.Helper()
	var enc qpackEncoder
	return enc.encode(func(yield func(itype indexType, name, value string)) {
		names := slices.Collect(maps.Keys(h))
		slices.Sort(names)
		for _, k := range names {
			for _, v := range h[k] {
				yield(mayIndex, k, v)
			}
		}
	})
}

func (ts *testQUICStream) writeHeaders(h http.Header) {
	ts.t.Helper()
	headers := ts.encodeHeaders(h)
	ts.writeVarint(int64(frameTypeHeaders))
	ts.writeVarint(int64(len(headers)))
	ts.Write(headers)
	if err := ts.Flush(); err != nil {
		ts.t.Fatalf("flushing HEADERS frame: %v", err)
	}
}

// writeHeadersRaw is just like writeHeaders, but avoids qpackEncoder.encode,
// which will automatically make sure that headers are encoded correctly, e.g.
// making field names all lowercase, skipping non-ASCII names.
// This method can be used to test that we properly reject invalid headers.
func (ts *testQUICStream) writeHeadersRaw(h http.Header) {
	ts.t.Helper()
	var b []byte
	b = appendPrefixedInt(b, 0, 8, 0) // Required Insert Count
	b = appendPrefixedInt(b, 0, 7, 0) // Delta Base

	names := slices.Collect(maps.Keys(h))
	slices.Sort(names)
	for _, k := range names {
		for _, v := range h[k] {
			if i, ok := staticTableByNameValue[tableEntry{k, v}]; ok {
				b = appendIndexedFieldLine(b, staticTable, i)
			} else if i, ok := staticTableByName[k]; ok {
				b = appendLiteralFieldLineWithNameReference(b, staticTable, mayIndex, i, v)
			} else {
				b = appendLiteralFieldLineWithLiteralName(b, mayIndex, k, v)
			}
		}
	}
	headers := b
	ts.writeVarint(int64(frameTypeHeaders))
	ts.writeVarint(int64(len(headers)))
	ts.Write(headers)
	if err := ts.Flush(); err != nil {
		ts.t.Fatalf("flushing HEADERS frame: %v", err)
	}
}

func (ts *testQUICStream) writeData(b []byte) {
	ts.t.Helper()
	ts.writeVarint(int64(frameTypeData))
	ts.writeVarint(int64(len(b)))
	ts.Write(b)
	if err := ts.Flush(); err != nil {
		ts.t.Fatalf("flushing DATA frame: %v", err)
	}
}

func (ts *testQUICStream) wantData(want []byte) {
	ts.t.Helper()
	synctest.Wait()
	ftype, err := ts.readFrameHeader()
	if err != nil {
		ts.t.Fatalf("want DATA frame, got error: %v", err)
	}
	if ftype != frameTypeData {
		ts.t.Fatalf("want DATA frame, got: %v", ftype)
	}
	got, err := ts.readFrameData()
	if err != nil {
		ts.t.Fatalf("error reading DATA frame: %v", err)
	}
	if !bytes.Equal(got, want) {
		ts.t.Fatalf("got data: {%x}, want {%x}", got, want)
	}
	if err := ts.endFrame(); err != nil {
		ts.t.Fatalf("endFrame: %v", err)
	}
}

func (ts *testQUICStream) wantClosed(reason string) {
	ts.t.Helper()
	synctest.Wait()
	ftype, err := ts.readFrameHeader()
	if err != io.EOF {
		ts.t.Fatalf("%v: want io.EOF, got %v %v", reason, ftype, err)
	}
}

func (ts *testQUICStream) wantError(want quic.StreamError) {
	ts.t.Helper()
	synctest.Wait()
	_, err := ts.ReadByte()
	if err == nil {
		ts.t.Fatalf("successfully read from stream; want stream error code %v", want)
	}
	var got quic.StreamError
	if !errors.As(err, &got) {
		ts.t.Fatalf("stream error = %v; want %v", err, want)
	}
	if got != want {
		ts.t.Fatalf("stream error code = %v; want %v", got, want)
	}
}

func (ts *testQUICStream) wantSettings(f func(settingType, value int64) error) {
	ts.t.Helper()
	synctest.Wait()
	if f == nil {
		f = func(settingType, value int64) error { return nil }
	}
	if err := ts.readSettings(f); err != nil {
		ts.t.Fatalf("f returned an error: %v", err)
	}
}

func (ts *testQUICStream) wantGoaway(wantID int64) {
	ts.t.Helper()
	synctest.Wait()
	ftype, err := ts.readFrameHeader()
	if err != nil {
		ts.t.Fatalf("want GOAWAY frame, got error: %v", err)
	}
	if ftype != frameTypeGoaway {
		ts.t.Fatalf("want GOAWAY frame, got: %v", ftype)
	}
	gotID, err := ts.readVarint()
	if err != nil {
		ts.t.Fatalf("failed reading GOAWAY frame, got error: %v", err)
	}
	if gotID != wantID {
		ts.t.Fatalf("got stream ID %v from GOAWAY frame, want %v stream ID", gotID, wantID)
	}
}

func (ts *testQUICStream) writePushPromise(pushID int64, h http.Header) {
	ts.t.Helper()
	headers := ts.encodeHeaders(h)
	ts.writeVarint(int64(frameTypePushPromise))
	ts.writeVarint(int64(sizeVarint(uint64(pushID)) + len(headers)))
	ts.writeVarint(pushID)
	ts.Write(headers)
	if err := ts.Flush(); err != nil {
		ts.t.Fatalf("flushing PUSH_PROMISE frame: %v", err)
	}
}

func diffHeaders(got, want http.Header) string {
	// nil and 0-length non-nil are equal.
	if len(got) == 0 && len(want) == 0 {
		return ""
	}
	// We could do a more sophisticated diff here.
	// DeepEqual is good enough for now.
	if reflect.DeepEqual(got, want) {
		return ""
	}
	return fmt.Sprintf("got:  %v\nwant: %v", got, want)
}

func (ts *testQUICStream) Flush() error {
	err := ts.stream.Flush()
	ts.t.Helper()
	if err != nil {
		ts.t.Errorf("unexpected error flushing stream: %v", err)
	}
	return err
}

// A testClientConn is a ClientConn on a test network.
type testClientConn struct {
	tr *transport
	cc *clientConn

	// *testQUICConn is the server half of the connection.
	*testQUICConn
	control *testQUICStream
}

func newTestClientConnWithHook(t testing.TB, stateHook func()) *testClientConn {
	e1, e2 := newQUICEndpointPair(t)
	tr := &transport{
		endpoint:    e1,
		tr1:         new(http.Transport),
		activeConns: make(map[*clientConn]struct{}),
	}

	cc, err := tr.dial(t.Context(), e2.LocalAddr().String(), testTLSConfig, stateHook)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cc.Close()
	})
	srvConn, err := e2.Accept(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	tc := &testClientConn{
		tr:           tr,
		cc:           cc,
		testQUICConn: newTestQUICConn(t, srvConn),
	}
	synctest.Wait()
	return tc
}

func newTestClientConn(t testing.TB) *testClientConn {
	return newTestClientConnWithHook(t, nil)
}

// greet performs initial connection handshaking with the client.
func (tc *testClientConn) greet() {
	// Client creates a control stream.
	clientControlStream := tc.wantStream(streamTypeControl)
	clientControlStream.wantFrameHeader(
		"client sends SETTINGS frame on control stream",
		frameTypeSettings)
	clientControlStream.discardFrame()

	// Server creates a control stream.
	tc.control = tc.newStream(streamTypeControl)
	tc.control.writeVarint(int64(frameTypeSettings))
	tc.control.writeVarint(0) // size
	tc.control.Flush()

	synctest.Wait()
}

type testRoundTrip struct {
	t       testing.TB
	resp    *http.Response
	respErr error
}

func (rt *testRoundTrip) done() bool {
	synctest.Wait()
	return rt.resp != nil || rt.respErr != nil
}

func (rt *testRoundTrip) result() (*http.Response, error) {
	rt.t.Helper()
	if !rt.done() {
		rt.t.Fatal("RoundTrip is not done; want it to be")
	}
	return rt.resp, rt.respErr
}

func (rt *testRoundTrip) response() *http.Response {
	rt.t.Helper()
	if !rt.done() {
		rt.t.Fatal("RoundTrip is not done; want it to be")
	}
	if rt.respErr != nil {
		rt.t.Fatalf("RoundTrip returned unexpected error: %v", rt.respErr)
	}
	return rt.resp
}

// err returns the (possibly nil) error result of RoundTrip.
func (rt *testRoundTrip) err() error {
	rt.t.Helper()
	_, err := rt.result()
	return err
}

func (rt *testRoundTrip) wantError(reason string) {
	rt.t.Helper()
	synctest.Wait()
	if !rt.done() {
		rt.t.Fatalf("%v: RoundTrip is not done; want it to have returned an error", reason)
	}
	if rt.respErr == nil {
		rt.t.Fatalf("%v: RoundTrip succeeded; want it to have returned an error", reason)
	}
}

// wantStatus indicates the expected response StatusCode.
func (rt *testRoundTrip) wantStatus(want int) {
	rt.t.Helper()
	if got := rt.response().StatusCode; got != want {
		rt.t.Fatalf("got response status %v, want %v", got, want)
	}
}

func (rt *testRoundTrip) wantHeaders(want http.Header) {
	rt.t.Helper()
	if diff := diffHeaders(rt.response().Header, want); diff != "" {
		rt.t.Fatalf("unexpected response headers:\n%v", diff)
	}
}

func (rt *testRoundTrip) wantTrailers(want http.Header) {
	rt.t.Helper()
	if diff := diffHeaders(rt.response().Trailer, want); diff != "" {
		rt.t.Fatalf("unexpected response trailers:\n%v", diff)
	}
}

// readBody reads the contents of the response body.
func (rt *testRoundTrip) readBody() ([]byte, error) {
	t := rt.t
	t.Helper()
	return io.ReadAll(rt.response().Body)
}

// wantBody consumes the a body and asserts that it is as expected.
func (rt *testRoundTrip) wantBody(want []byte) {
	t := rt.t
	t.Helper()
	got, err := rt.readBody()
	if err != nil {
		t.Fatalf("unexpected error reading response body: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("unexpected response body:\ngot:  %q\nwant: %q", got, want)
	}
}

func (tc *testClientConn) roundTrip(req *http.Request) *testRoundTrip {
	rt := &testRoundTrip{t: tc.t}
	go func() {
		rt.resp, rt.respErr = tc.cc.RoundTrip(req)
	}()
	return rt
}
