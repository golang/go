// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"context"
	"fmt"
	"net"
	"net/textproto"
	"sync"
	"testing"
	"time"

	"net/http/internal/httpcommon"

	"golang.org/x/net/http2/hpack"
)

func init() {
	inTests = true
}

const (
	DefaultMaxReadFrameSize     = defaultMaxReadFrameSize
	DefaultMaxStreams           = defaultMaxStreams
	InflowMinRefresh            = inflowMinRefresh
	InitialHeaderTableSize      = initialHeaderTableSize
	InitialMaxConcurrentStreams = initialMaxConcurrentStreams
	InitialWindowSize           = initialWindowSize
	MaxFrameSize                = maxFrameSize
	MaxQueuedControlFrames      = maxQueuedControlFrames
	MinMaxFrameSize             = minMaxFrameSize
)

type (
	ServerConn  = serverConn
	Stream      = stream
	StreamState = streamState

	PseudoHeaderError     = pseudoHeaderError
	HeaderFieldNameError  = headerFieldNameError
	HeaderFieldValueError = headerFieldValueError
)

const (
	StateIdle             = stateIdle
	StateOpen             = stateOpen
	StateHalfClosedLocal  = stateHalfClosedLocal
	StateHalfClosedRemote = stateHalfClosedRemote
	StateClosed           = stateClosed
)

var (
	ErrClientConnForceClosed       = errClientConnForceClosed
	ErrClientConnNotEstablished    = errClientConnNotEstablished
	ErrClientConnUnusable          = errClientConnUnusable
	ErrExtendedConnectNotSupported = errExtendedConnectNotSupported
	ErrReqBodyTooLong              = errReqBodyTooLong
	ErrRequestHeaderListSize       = errRequestHeaderListSize
	ErrResponseHeaderListSize      = errResponseHeaderListSize
)

func (s *Server) TestServeConn(c net.Conn, opts *ServeConnOpts, newf func(*serverConn)) {
	s.serveConn(c, opts, newf)
}

func (sc *serverConn) TestFlowControlConsumed() (consumed int32) {
	conf := configFromServer(sc.hs, sc.srv)
	donec := make(chan struct{})
	sc.sendServeMsg(func(sc *serverConn) {
		defer close(donec)
		initial := int32(conf.MaxReceiveBufferPerConnection)
		avail := sc.inflow.avail + sc.inflow.unsent
		consumed = initial - avail
	})
	<-donec
	return consumed
}

func (sc *serverConn) TestStreamExists(id uint32) bool {
	ch := make(chan bool, 1)
	sc.serveMsgCh <- func(int) {
		ch <- (sc.streams[id] != nil)
	}
	return <-ch
}

func (sc *serverConn) TestStreamState(id uint32) streamState {
	ch := make(chan streamState, 1)
	sc.serveMsgCh <- func(int) {
		state, _ := sc.state(id)
		ch <- state
	}
	return <-ch
}

func (sc *serverConn) StartGracefulShutdown() { sc.startGracefulShutdown() }

func (sc *serverConn) TestHPACKEncoder() *hpack.Encoder {
	return sc.hpackEncoder
}

func (sc *serverConn) TestFramerMaxHeaderStringLen() int {
	return sc.framer.maxHeaderStringLen()
}

func (t *Transport) DialClientConn(ctx context.Context, addr string, singleUse bool) (*ClientConn, error) {
	return t.dialClientConn(ctx, addr, singleUse)
}

func (t *Transport) TestNewClientConn(c net.Conn, singleUse bool, internalStateHook func()) (*ClientConn, error) {
	return t.newClientConn(c, singleUse, internalStateHook)
}

func (t *Transport) TestSetNewClientConnHook(f func(*ClientConn)) {
	t.transportTestHooks = &transportTestHooks{
		newclientconn: f,
	}
}

func (cc *ClientConn) TestNetConn() net.Conn     { return cc.tconn }
func (cc *ClientConn) TestSetNetConn(c net.Conn) { cc.tconn = c }

func (cc *ClientConn) TestRoundTrip(req *ClientRequest, f func(stremaID uint32)) (*ClientResponse, error) {
	return cc.roundTrip(req, func(cs *clientStream) {
		f(cs.ID)
	})
}

func (cc *ClientConn) TestHPACKEncoder() *hpack.Encoder {
	return cc.henc
}

func (cc *ClientConn) TestPeerMaxHeaderTableSize() uint32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.peerMaxHeaderTableSize
}

func (cc *ClientConn) TestInflowWindow(streamID uint32) (int32, error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if streamID == 0 {
		return cc.inflow.avail + cc.inflow.unsent, nil
	}
	cs := cc.streams[streamID]
	if cs == nil {
		return -1, fmt.Errorf("no stream with id %v", streamID)
	}
	return cs.inflow.avail + cs.inflow.unsent, nil
}

func (fr *Framer) TestSetDebugReadLoggerf(f func(string, ...any)) {
	fr.logReads = true
	fr.debugReadLoggerf = f
}

func (fr *Framer) TestSetDebugWriteLoggerf(f func(string, ...any)) {
	fr.logWrites = true
	fr.debugWriteLoggerf = f
}

func SummarizeFrame(f Frame) string {
	return summarizeFrame(f)
}

func SetTestHookGetServerConn(t testing.TB, f func(*serverConn)) {
	SetForTest(t, &testHookGetServerConn, f)
}

func init() {
	testHookOnPanicMu = new(sync.Mutex)
}

func SetTestHookOnPanic(t testing.TB, f func(sc *serverConn, panicVal interface{}) (rePanic bool)) {
	testHookOnPanicMu.Lock()
	defer testHookOnPanicMu.Unlock()
	old := testHookOnPanic
	testHookOnPanic = f
	t.Cleanup(func() {
		testHookOnPanicMu.Lock()
		defer testHookOnPanicMu.Unlock()
		testHookOnPanic = old
	})
}

func SetTestHookGot1xx(t testing.TB, f func(int, textproto.MIMEHeader) error) {
	SetForTest(t, &got1xxFuncForTests, f)
}

func SetDisableExtendedConnectProtocol(t testing.TB, v bool) {
	SetForTest(t, &disableExtendedConnectProtocol, v)
}

func LogFrameReads() bool  { return logFrameReads }
func LogFrameWrites() bool { return logFrameWrites }

const GoAwayTimeout = 25 * time.Millisecond

func init() {
	goAwayTimeout = GoAwayTimeout
}

func EncodeHeaderRaw(t testing.TB, headers ...string) []byte {
	return encodeHeaderRaw(t, headers...)
}

func NewPriorityWriteSchedulerRFC7540(cfg *PriorityWriteSchedulerConfig) WriteScheduler {
	return newPriorityWriteSchedulerRFC7540(cfg)
}

func NewPriorityWriteSchedulerRFC9218() WriteScheduler {
	return newPriorityWriteSchedulerRFC9218()
}

func NewRoundRobinWriteScheduler() WriteScheduler {
	return newRoundRobinWriteScheduler()
}

func DisableGoroutineTracking(t testing.TB) {
	disableDebugGoroutines.Store(true)
	t.Cleanup(func() {
		disableDebugGoroutines.Store(false)
	})
}

func InvalidHTTP1LookingFrameHeader() FrameHeader {
	return invalidHTTP1LookingFrameHeader()
}

func NewNoDialClientConnPool() ClientConnPool {
	return noDialClientConnPool{new(clientConnPool)}
}

func EncodeRequestHeaders(req *ClientRequest, addGzipHeader bool, peerMaxHeaderListSize uint64, headerf func(name, value string)) (httpcommon.EncodeHeadersResult, error) {
	return encodeRequestHeaders(req, addGzipHeader, peerMaxHeaderListSize, headerf)
}
