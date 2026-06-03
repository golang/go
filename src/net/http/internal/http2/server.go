// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: turn off the serve goroutine when idle, so
// an idle conn only has the readFrames goroutine active. (which could
// also be optimized probably to pin less memory in crypto/tls). This
// would involve tracking when the serve goroutine is active (atomic
// int32 read/CAS probably?) and starting it up when frames arrive,
// and shutting it down when all handlers exit. the occasional PING
// packets could use time.AfterFunc to call sc.wakeStartServeLoop()
// (which is a no-op if already running) and then queue the PING write
// as normal. The serve loop would then exit in most cases (if no
// Handlers running) and not be woken up again until the PING packet
// returns.

// TODO (maybe): add a mechanism for Handlers to going into
// half-closed-local mode (rw.(io.Closer) test?) but not exit their
// handler, and continue to be able to read from the
// Request.Body. This would be a somewhat semantic change from HTTP/1
// (or at least what we expose in net/http), so I'd probably want to
// add it there too. For now, this package says that returning from
// the Handler ServeHTTP function means you're both done reading and
// done writing, without a way to stop just one or the other.

package http2

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http/internal"
	"net/http/internal/httpcommon"
	"net/textproto"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/http2/hpack"
)

const (
	prefaceTimeout        = 10 * time.Second
	firstSettingsTimeout  = 2 * time.Second // should be in-flight with preface anyway
	handlerChunkWriteSize = 4 << 10
	defaultMaxStreams     = 250 // TODO: make this 100 as the GFE seems to?

	// maxQueuedControlFrames is the maximum number of control frames like
	// SETTINGS, PING and RST_STREAM that will be queued for writing before
	// the connection is closed to prevent memory exhaustion attacks.
	maxQueuedControlFrames = 10000
)

var (
	errClientDisconnected = errors.New("client disconnected")
	errClosedBody         = errors.New("body closed by handler")
	errHandlerComplete    = errors.New("http2: request body closed due to handler exiting")
	errStreamClosed       = errors.New("http2: stream closed")
)

var responseWriterStatePool = sync.Pool{
	New: func() any {
		rws := &responseWriterState{}
		rws.bw = bufio.NewWriterSize(chunkWriter{rws}, handlerChunkWriteSize)
		return rws
	},
}

// Test hooks.
var (
	testHookOnConn    func()
	testHookOnPanicMu *sync.Mutex // nil except in tests
	testHookOnPanic   func(sc *serverConn, panicVal any) (rePanic bool)
)

// Server is an HTTP/2 server.
type Server struct {
	mu          sync.Mutex
	activeConns map[*serverConn]struct{}

	// Pool of error channels. This is per-Server rather than global
	// because channels can't be reused across synctest bubbles.
	errChanPool sync.Pool
}

func (s *Server) registerConn(sc *serverConn) {
	if s == nil {
		return // if the Server was used without calling ConfigureServer
	}
	s.mu.Lock()
	s.activeConns[sc] = struct{}{}
	s.mu.Unlock()
}

func (s *Server) unregisterConn(sc *serverConn) {
	if s == nil {
		return // if the Server was used without calling ConfigureServer
	}
	s.mu.Lock()
	delete(s.activeConns, sc)
	s.mu.Unlock()
}

func (s *Server) startGracefulShutdown() {
	if s == nil {
		return // if the Server was used without calling ConfigureServer
	}
	s.mu.Lock()
	for sc := range s.activeConns {
		sc.startGracefulShutdown()
	}
	s.mu.Unlock()
}

// Global error channel pool used for uninitialized Servers.
// We use a per-Server pool when possible to avoid using channels across synctest bubbles.
var errChanPool = sync.Pool{
	New: func() any { return make(chan error, 1) },
}

func (s *Server) getErrChan() chan error {
	if s == nil {
		return errChanPool.Get().(chan error) // Server used without calling ConfigureServer
	}
	return s.errChanPool.Get().(chan error)
}

func (s *Server) putErrChan(ch chan error) {
	if s == nil {
		errChanPool.Put(ch) // Server used without calling ConfigureServer
		return
	}
	s.errChanPool.Put(ch)
}

func (s *Server) Configure(conf ServerConfig, tcfg *tls.Config) error {
	s.activeConns = make(map[*serverConn]struct{})
	s.errChanPool = sync.Pool{New: func() any { return make(chan error, 1) }}

	if tcfg.CipherSuites != nil && tcfg.MinVersion < tls.VersionTLS13 {
		// If they already provided a TLS 1.0–1.2 CipherSuite list, return an
		// error if it is missing ECDHE_RSA_WITH_AES_128_GCM_SHA256 or
		// ECDHE_ECDSA_WITH_AES_128_GCM_SHA256.
		haveRequired := false
		for _, cs := range tcfg.CipherSuites {
			switch cs {
			case tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				// Alternative MTI cipher to not discourage ECDSA-only servers.
				// See http://golang.org/cl/30721 for further information.
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:
				haveRequired = true
			}
		}
		if !haveRequired {
			return fmt.Errorf("http2: TLSConfig.CipherSuites is missing an HTTP/2-required AES_128_GCM_SHA256 cipher (need at least one of TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 or TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256)")
		}
	}

	// Note: not setting MinVersion to tls.VersionTLS12,
	// as we don't want to interfere with HTTP/1.1 traffic
	// on the user's server. We enforce TLS 1.2 later once
	// we accept a connection. Ideally this should be done
	// during next-proto selection, but using TLS <1.2 with
	// HTTP/2 is still the client's bug.

	return nil
}

func (s *Server) GracefulShutdown() {
	s.startGracefulShutdown()
}

// ServeConnOpts are options for the Server.ServeConn method.
type ServeConnOpts struct {
	// Context is the base context to use.
	// If nil, context.Background is used.
	Context context.Context

	// BaseConfig optionally sets the base configuration
	// for values. If nil, defaults are used.
	BaseConfig ServerConfig

	// Handler specifies which handler to use for processing
	// requests. If nil, BaseConfig.Handler is used. If BaseConfig
	// or BaseConfig.Handler is nil, http.DefaultServeMux is used.
	Handler Handler

	// Settings is the decoded contents of the HTTP2-Settings header
	// in an h2c upgrade request.
	Settings []byte

	UpgradeRequest *ServerRequest

	// SawClientPreface is set if the HTTP/2 connection preface
	// has already been read from the connection.
	SawClientPreface bool
}

func (o *ServeConnOpts) context() context.Context {
	if o != nil && o.Context != nil {
		return o.Context
	}
	return context.Background()
}

// ServeConn serves HTTP/2 requests on the provided connection and
// blocks until the connection is no longer readable.
//
// ServeConn starts speaking HTTP/2 assuming that c has not had any
// reads or writes. It writes its initial settings frame and expects
// to be able to read the preface and settings frame from the
// client. If c has a ConnectionState method like a *tls.Conn, the
// ConnectionState is used to verify the TLS ciphersuite and to set
// the Request.TLS field in Handlers.
//
// ServeConn does not support h2c by itself. Any h2c support must be
// implemented in terms of providing a suitably-behaving net.Conn.
//
// The opts parameter is optional. If nil, default values are used.
func (s *Server) ServeConn(c net.Conn, opts *ServeConnOpts) {
	if opts == nil {
		opts = &ServeConnOpts{}
	}

	var newf func(*serverConn)
	if inTests {
		// Fetch NewConnContextKey if set, leave newf as nil otherwise.
		newf, _ = opts.Context.Value(NewConnContextKey).(func(*serverConn))
	}

	s.serveConn(c, opts, newf)
}

type contextKey string

var (
	NewConnContextKey         = new("NewConnContextKey")
	ConnectionStateContextKey = new("ConnectionStateContextKey")
)

func (s *Server) serveConn(c net.Conn, opts *ServeConnOpts, newf func(*serverConn)) {
	baseCtx, cancel := serverConnBaseContext(c, opts)
	defer cancel()

	conf := configFromServer(opts.BaseConfig)
	sc := &serverConn{
		srv:                         s,
		hs:                          opts.BaseConfig,
		conn:                        c,
		baseCtx:                     baseCtx,
		remoteAddrStr:               c.RemoteAddr().String(),
		bw:                          newBufferedWriter(c, conf.WriteByteTimeout),
		handler:                     opts.Handler,
		streams:                     make(map[uint32]*stream),
		readFrameCh:                 make(chan readFrameResult),
		wantWriteFrameCh:            make(chan FrameWriteRequest, 8),
		serveMsgCh:                  make(chan any, 8),
		wroteFrameCh:                make(chan frameWriteResult, 1), // buffered; one send in writeFrameAsync
		bodyReadCh:                  make(chan bodyReadMsg),         // buffering doesn't matter either way
		doneServing:                 make(chan struct{}),
		clientMaxStreams:            math.MaxUint32, // Section 6.5.2: "Initially, there is no limit to this value"
		advMaxStreams:               uint32(conf.MaxConcurrentStreams),
		initialStreamSendWindowSize: initialWindowSize,
		initialStreamRecvWindowSize: int32(conf.MaxReceiveBufferPerStream),
		maxFrameSize:                initialMaxFrameSize,
		pingTimeout:                 conf.PingTimeout,
		countErrorFunc:              conf.CountError,
		serveG:                      newGoroutineLock(),
		pushEnabled:                 true,
		sawClientPreface:            opts.SawClientPreface,
	}
	if newf != nil {
		newf(sc)
	}

	s.registerConn(sc)
	defer s.unregisterConn(sc)

	// The net/http package sets the write deadline from the
	// http.Server.WriteTimeout during the TLS handshake, but then
	// passes the connection off to us with the deadline already set.
	// Write deadlines are set per stream in serverConn.newStream.
	// Disarm the net.Conn write deadline here.
	if sc.hs.WriteTimeout() > 0 {
		sc.conn.SetWriteDeadline(time.Time{})
	}

	switch {
	case sc.hs.DisableClientPriority():
		sc.writeSched = newRoundRobinWriteScheduler()
	default:
		sc.writeSched = newPriorityWriteSchedulerRFC9218()
	}

	// These start at the RFC-specified defaults. If there is a higher
	// configured value for inflow, that will be updated when we send a
	// WINDOW_UPDATE shortly after sending SETTINGS.
	sc.flow.add(initialWindowSize)
	sc.inflow.init(initialWindowSize)
	sc.hpackEncoder = hpack.NewEncoder(&sc.headerWriteBuf)
	sc.hpackEncoder.SetMaxDynamicTableSizeLimit(uint32(conf.MaxEncoderHeaderTableSize))

	fr := NewFramer(sc.bw, c)
	if conf.CountError != nil {
		fr.countError = conf.CountError
	}
	fr.ReadMetaHeaders = hpack.NewDecoder(uint32(conf.MaxDecoderHeaderTableSize), nil)
	fr.MaxHeaderListSize = sc.maxHeaderListSize()
	fr.SetMaxReadFrameSize(uint32(conf.MaxReadFrameSize))
	sc.framer = fr

	if tc, ok := c.(connectionStater); ok {
		sc.tlsState = new(tls.ConnectionState)
		*sc.tlsState = tc.ConnectionState()

		// Optionally override the ConnectionState in tests.
		if inTests {
			f, ok := opts.Context.Value(ConnectionStateContextKey).(func() tls.ConnectionState)
			if ok {
				*sc.tlsState = f()
			}
		}

		// 9.2 Use of TLS Features
		// An implementation of HTTP/2 over TLS MUST use TLS
		// 1.2 or higher with the restrictions on feature set
		// and cipher suite described in this section. Due to
		// implementation limitations, it might not be
		// possible to fail TLS negotiation. An endpoint MUST
		// immediately terminate an HTTP/2 connection that
		// does not meet the TLS requirements described in
		// this section with a connection error (Section
		// 5.4.1) of type INADEQUATE_SECURITY.
		if sc.tlsState.Version < tls.VersionTLS12 {
			sc.rejectConn(ErrCodeInadequateSecurity, "TLS version too low")
			return
		}

		if sc.tlsState.ServerName == "" {
			// Client must use SNI, but we don't enforce that anymore,
			// since it was causing problems when connecting to bare IP
			// addresses during development.
			//
			// TODO: optionally enforce? Or enforce at the time we receive
			// a new request, and verify the ServerName matches the :authority?
			// But that precludes proxy situations, perhaps.
			//
			// So for now, do nothing here again.
		}

		if !conf.PermitProhibitedCipherSuites && isBadCipher(sc.tlsState.CipherSuite) {
			// "Endpoints MAY choose to generate a connection error
			// (Section 5.4.1) of type INADEQUATE_SECURITY if one of
			// the prohibited cipher suites are negotiated."
			//
			// We choose that. In my opinion, the spec is weak
			// here. It also says both parties must support at least
			// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 so there's no
			// excuses here. If we really must, we could allow an
			// "AllowInsecureWeakCiphers" option on the server later.
			// Let's see how it plays out first.
			sc.rejectConn(ErrCodeInadequateSecurity, fmt.Sprintf("Prohibited TLS 1.2 Cipher Suite: %x", sc.tlsState.CipherSuite))
			return
		}
	}

	if opts.Settings != nil {
		fr := &SettingsFrame{
			FrameHeader: FrameHeader{valid: true},
			p:           opts.Settings,
		}
		if err := fr.ForeachSetting(sc.processSetting); err != nil {
			sc.rejectConn(ErrCodeProtocol, "invalid settings")
			return
		}
		opts.Settings = nil
	}

	if opts.UpgradeRequest != nil {
		sc.upgradeRequest(opts.UpgradeRequest)
		opts.UpgradeRequest = nil
	}

	sc.serve(conf)
}

func serverConnBaseContext(c net.Conn, opts *ServeConnOpts) (ctx context.Context, cancel func()) {
	return context.WithCancel(opts.context())
}

func (sc *serverConn) rejectConn(err ErrCode, debug string) {
	sc.vlogf("http2: server rejecting conn: %v, %s", err, debug)
	// ignoring errors. hanging up anyway.
	sc.framer.WriteGoAway(0, err, []byte(debug))
	sc.bw.Flush()
	sc.conn.Close()
}

type serverConn struct {
	// Immutable:
	srv              *Server
	hs               ServerConfig
	conn             net.Conn
	bw               *bufferedWriter // writing to conn
	handler          Handler
	baseCtx          context.Context
	framer           *Framer
	doneServing      chan struct{}          // closed when serverConn.serve ends
	readFrameCh      chan readFrameResult   // written by serverConn.readFrames
	wantWriteFrameCh chan FrameWriteRequest // from handlers -> serve
	wroteFrameCh     chan frameWriteResult  // from writeFrameAsync -> serve, tickles more frame writes
	bodyReadCh       chan bodyReadMsg       // from handlers -> serve
	serveMsgCh       chan any               // misc messages & code to send to / run on the serve loop
	flow             outflow                // conn-wide (not stream-specific) outbound flow control
	inflow           inflow                 // conn-wide inbound flow control
	tlsState         *tls.ConnectionState   // shared by all handlers, like net/http
	remoteAddrStr    string
	writeSched       WriteScheduler
	countErrorFunc   func(errType string)

	// Everything following is owned by the serve loop; use serveG.check():
	serveG                      goroutineLock // used to verify funcs are on serve()
	pushEnabled                 bool
	sawClientPreface            bool // preface has already been read, used in h2c upgrade
	sawFirstSettings            bool // got the initial SETTINGS frame after the preface
	needToSendSettingsAck       bool
	unackedSettings             int    // how many SETTINGS have we sent without ACKs?
	queuedControlFrames         int    // control frames in the writeSched queue
	clientMaxStreams            uint32 // SETTINGS_MAX_CONCURRENT_STREAMS from client (our PUSH_PROMISE limit)
	advMaxStreams               uint32 // our SETTINGS_MAX_CONCURRENT_STREAMS advertised the client
	curClientStreams            uint32 // number of open streams initiated by the client
	curPushedStreams            uint32 // number of open streams initiated by server push
	curHandlers                 uint32 // number of running handler goroutines
	maxClientStreamID           uint32 // max ever seen from client (odd), or 0 if there have been no client requests
	maxPushPromiseID            uint32 // ID of the last push promise (even), or 0 if there have been no pushes
	streams                     map[uint32]*stream
	unstartedHandlers           []unstartedHandler
	initialStreamSendWindowSize int32
	initialStreamRecvWindowSize int32
	maxFrameSize                int32
	peerMaxHeaderListSize       uint32            // zero means unknown (default)
	canonHeader                 map[string]string // http2-lower-case -> Go-Canonical-Case
	canonHeaderKeysSize         int               // canonHeader keys size in bytes
	writingFrame                bool              // started writing a frame (on serve goroutine or separate)
	writingFrameAsync           bool              // started a frame on its own goroutine but haven't heard back on wroteFrameCh
	needsFrameFlush             bool              // last frame write wasn't a flush
	inGoAway                    bool              // we've started to or sent GOAWAY
	inFrameScheduleLoop         bool              // whether we're in the scheduleFrameWrite loop
	needToSendGoAway            bool              // we need to schedule a GOAWAY frame write
	pingSent                    bool
	sentPingData                [8]byte
	goAwayCode                  ErrCode
	shutdownTimer               *time.Timer // nil until used
	idleTimer                   *time.Timer // nil if unused
	readIdleTimeout             time.Duration
	pingTimeout                 time.Duration
	readIdleTimer               *time.Timer // nil if unused

	// Owned by the writeFrameAsync goroutine:
	headerWriteBuf bytes.Buffer
	hpackEncoder   *hpack.Encoder

	// Used by startGracefulShutdown.
	shutdownOnce sync.Once

	// Used for RFC 9218 prioritization.
	hasIntermediary bool // connection is done via an intermediary / proxy
	priorityAware   bool // the client has sent priority signal, meaning that it is aware of it.
}

func (sc *serverConn) writeSchedIgnoresRFC7540() bool {
	switch sc.writeSched.(type) {
	case *priorityWriteSchedulerRFC9218:
		return true
	case *roundRobinWriteScheduler:
		return true
	default:
		return false
	}
}

const DefaultMaxHeaderBytes = 1 << 20 // keep this in sync with net/http

func (sc *serverConn) maxHeaderListSize() uint32 {
	n := sc.hs.MaxHeaderBytes()
	if n <= 0 {
		n = DefaultMaxHeaderBytes
	}
	return uint32(adjustHTTP1MaxHeaderSize(int64(n)))
}

func (sc *serverConn) curOpenStreams() uint32 {
	sc.serveG.check()
	return sc.curClientStreams + sc.curPushedStreams
}

// stream represents a stream. This is the minimal metadata needed by
// the serve goroutine. Most of the actual stream state is owned by
// the http.Handler's goroutine in the responseWriter. Because the
// responseWriter's responseWriterState is recycled at the end of a
// handler, this struct intentionally has no pointer to the
// *responseWriter{,State} itself, as the Handler ending nils out the
// responseWriter's state field.
type stream struct {
	// immutable:
	sc        *serverConn
	id        uint32
	body      *pipe       // non-nil if expecting DATA frames
	cw        closeWaiter // closed wait stream transitions to closed state
	ctx       context.Context
	cancelCtx func()

	// owned by serverConn's serve loop:
	bodyBytes        int64   // body bytes seen so far
	declBodyBytes    int64   // or -1 if undeclared
	flow             outflow // limits writing from Handler to client
	inflow           inflow  // what the client is allowed to POST/etc to us
	state            streamState
	resetQueued      bool        // RST_STREAM queued for write; set by sc.resetStream
	gotTrailerHeader bool        // HEADER frame for trailers was seen
	wroteHeaders     bool        // whether we wrote headers (not status 100)
	readDeadline     *time.Timer // nil if unused
	writeDeadline    *time.Timer // nil if unused
	closeErr         error       // set before cw is closed

	trailer    Header // accumulated trailers
	reqTrailer Header // handler's Request.Trailer
}

func (sc *serverConn) Framer() *Framer  { return sc.framer }
func (sc *serverConn) CloseConn() error { return sc.conn.Close() }
func (sc *serverConn) Flush() error     { return sc.bw.Flush() }
func (sc *serverConn) HeaderEncoder() (*hpack.Encoder, *bytes.Buffer) {
	return sc.hpackEncoder, &sc.headerWriteBuf
}

func (sc *serverConn) state(streamID uint32) (streamState, *stream) {
	sc.serveG.check()
	// http://tools.ietf.org/html/rfc7540#section-5.1
	if st, ok := sc.streams[streamID]; ok {
		return st.state, st
	}
	// "The first use of a new stream identifier implicitly closes all
	// streams in the "idle" state that might have been initiated by
	// that peer with a lower-valued stream identifier. For example, if
	// a client sends a HEADERS frame on stream 7 without ever sending a
	// frame on stream 5, then stream 5 transitions to the "closed"
	// state when the first frame for stream 7 is sent or received."
	if streamID%2 == 1 {
		if streamID <= sc.maxClientStreamID {
			return stateClosed, nil
		}
	} else {
		if streamID <= sc.maxPushPromiseID {
			return stateClosed, nil
		}
	}
	return stateIdle, nil
}

// setConnState calls the net/http ConnState hook for this connection, if configured.
// Note that the net/http package does StateNew and StateClosed for us.
// There is currently no plan for StateHijacked or hijacking HTTP/2 connections.
func (sc *serverConn) setConnState(state ConnState) {
	sc.hs.ConnState(sc.conn, state)
}

func (sc *serverConn) vlogf(format string, args ...any) {
	if VerboseLogs {
		sc.logf(format, args...)
	}
}

func (sc *serverConn) logf(format string, args ...any) {
	if lg := sc.hs.ErrorLog(); lg != nil {
		lg.Printf(format, args...)
	} else {
		log.Printf(format, args...)
	}
}

// errno returns v's underlying uintptr, else 0.
//
// TODO: remove this helper function once http2 can use build
// tags. See comment in isClosedConnError.
func errno(v error) uintptr {
	if rv := reflect.ValueOf(v); rv.Kind() == reflect.Uintptr {
		return uintptr(rv.Uint())
	}
	return 0
}

// isClosedConnError reports whether err is an error from use of a closed
// network connection.
func isClosedConnError(err error) bool {
	if err == nil {
		return false
	}

	if errors.Is(err, net.ErrClosed) {
		return true
	}

	// TODO(bradfitz): x/tools/cmd/bundle doesn't really support
	// build tags, so I can't make an http2_windows.go file with
	// Windows-specific stuff. Fix that and move this, once we
	// have a way to bundle this into std's net/http somehow.
	if runtime.GOOS == "windows" {
		if oe, ok := err.(*net.OpError); ok && oe.Op == "read" {
			if se, ok := oe.Err.(*os.SyscallError); ok && se.Syscall == "wsarecv" {
				const WSAECONNABORTED = 10053
				const WSAECONNRESET = 10054
				if n := errno(se.Err); n == WSAECONNRESET || n == WSAECONNABORTED {
					return true
				}
			}
		}
	}
	return false
}

func (sc *serverConn) condlogf(err error, format string, args ...any) {
	if err == nil {
		return
	}
	if err == io.EOF || err == io.ErrUnexpectedEOF || isClosedConnError(err) || err == errPrefaceTimeout {
		// Boring, expected errors.
		sc.vlogf(format, args...)
	} else {
		sc.logf(format, args...)
	}
}

// maxCachedCanonicalHeadersKeysSize is an arbitrarily-chosen limit on the size
// of the entries in the canonHeader cache.
// This should be larger than the size of unique, uncommon header keys likely to
// be sent by the peer, while not so high as to permit unreasonable memory usage
// if the peer sends an unbounded number of unique header keys.
const maxCachedCanonicalHeadersKeysSize = 2048

func (sc *serverConn) canonicalHeader(v string) string {
	sc.serveG.check()
	cv, ok := httpcommon.CachedCanonicalHeader(v)
	if ok {
		return cv
	}
	cv, ok = sc.canonHeader[v]
	if ok {
		return cv
	}
	if sc.canonHeader == nil {
		sc.canonHeader = make(map[string]string)
	}
	cv = textproto.CanonicalMIMEHeaderKey(v)
	size := 100 + len(v)*2 // 100 bytes of map overhead + key + value
	if sc.canonHeaderKeysSize+size <= maxCachedCanonicalHeadersKeysSize {
		sc.canonHeader[v] = cv
		sc.canonHeaderKeysSize += size
	}
	return cv
}

type readFrameResult struct {
	f   Frame // valid until readMore is called
	err error

	// readMore should be called once the consumer no longer needs or
	// retains f. After readMore, f is invalid and more frames can be
	// read.
	readMore func()
}

// readFrames is the loop that reads incoming frames.
// It takes care to only read one frame at a time, blocking until the
// consumer is done with the frame.
// It's run on its own goroutine.
func (sc *serverConn) readFrames() {
	gate := make(chan struct{})
	gateDone := func() { gate <- struct{}{} }
	for {
		f, err := sc.framer.ReadFrame()
		select {
		case sc.readFrameCh <- readFrameResult{f, err, gateDone}:
		case <-sc.doneServing:
			return
		}
		select {
		case <-gate:
		case <-sc.doneServing:
			return
		}
		if terminalReadFrameError(err) {
			return
		}
	}
}

// frameWriteResult is the message passed from writeFrameAsync to the serve goroutine.
type frameWriteResult struct {
	_   incomparable
	wr  FrameWriteRequest // what was written (or attempted)
	err error             // result of the writeFrame call
}

// writeFrameAsync runs in its own goroutine and writes a single frame
// and then reports when it's done.
// At most one goroutine can be running writeFrameAsync at a time per
// serverConn.
func (sc *serverConn) writeFrameAsync(wr FrameWriteRequest, wd *writeData) {
	var err error
	if wd == nil {
		err = wr.write.writeFrame(sc)
	} else {
		err = sc.framer.endWrite()
	}
	sc.wroteFrameCh <- frameWriteResult{wr: wr, err: err}
}

func (sc *serverConn) closeAllStreamsOnConnClose() {
	sc.serveG.check()
	for _, st := range sc.streams {
		sc.closeStream(st, errClientDisconnected)
	}
}

func (sc *serverConn) stopShutdownTimer() {
	sc.serveG.check()
	if t := sc.shutdownTimer; t != nil {
		t.Stop()
	}
}

func (sc *serverConn) notePanic() {
	// Note: this is for serverConn.serve panicking, not http.Handler code.
	if testHookOnPanicMu != nil {
		testHookOnPanicMu.Lock()
		defer testHookOnPanicMu.Unlock()
	}
	if testHookOnPanic != nil {
		if e := recover(); e != nil {
			if testHookOnPanic(sc, e) {
				panic(e)
			}
		}
	}
}

func (sc *serverConn) serve(conf Config) {
	sc.serveG.check()
	defer sc.notePanic()
	defer sc.conn.Close()
	defer sc.closeAllStreamsOnConnClose()
	defer sc.stopShutdownTimer()
	defer close(sc.doneServing) // unblocks handlers trying to send

	if VerboseLogs {
		sc.vlogf("http2: server connection from %v on %p", sc.conn.RemoteAddr(), sc.hs)
	}

	settings := writeSettings{
		{SettingMaxFrameSize, uint32(conf.MaxReadFrameSize)},
		{SettingMaxConcurrentStreams, sc.advMaxStreams},
		{SettingMaxHeaderListSize, sc.maxHeaderListSize()},
		{SettingHeaderTableSize, uint32(conf.MaxDecoderHeaderTableSize)},
		{SettingInitialWindowSize, uint32(sc.initialStreamRecvWindowSize)},
	}
	if !disableExtendedConnectProtocol {
		settings = append(settings, Setting{SettingEnableConnectProtocol, 1})
	}
	if sc.writeSchedIgnoresRFC7540() {
		settings = append(settings, Setting{SettingNoRFC7540Priorities, 1})
	}
	sc.writeFrame(FrameWriteRequest{
		write: settings,
	})
	sc.unackedSettings++

	// Each connection starts with initialWindowSize inflow tokens.
	// If a higher value is configured, we add more tokens.
	if diff := conf.MaxReceiveBufferPerConnection - initialWindowSize; diff > 0 {
		sc.sendWindowUpdate(nil, int(diff))
	}

	if err := sc.readPreface(); err != nil {
		sc.condlogf(err, "http2: server: error reading preface from client %v: %v", sc.conn.RemoteAddr(), err)
		return
	}
	// Now that we've got the preface, get us out of the
	// "StateNew" state. We can't go directly to idle, though.
	// Active means we read some data and anticipate a request. We'll
	// do another Active when we get a HEADERS frame.
	sc.setConnState(ConnStateActive)
	sc.setConnState(ConnStateIdle)

	if idle := sc.hs.IdleTimeout(); idle > 0 {
		sc.idleTimer = time.AfterFunc(idle, sc.onIdleTimer)
		defer sc.idleTimer.Stop()
	}

	if conf.SendPingTimeout > 0 {
		sc.readIdleTimeout = conf.SendPingTimeout
		sc.readIdleTimer = time.AfterFunc(conf.SendPingTimeout, sc.onReadIdleTimer)
		defer sc.readIdleTimer.Stop()
	}

	go sc.readFrames() // closed by defer sc.conn.Close above

	settingsTimer := time.AfterFunc(firstSettingsTimeout, sc.onSettingsTimer)
	defer settingsTimer.Stop()

	lastFrameTime := time.Now()
	loopNum := 0
	for {
		loopNum++
		select {
		case wr := <-sc.wantWriteFrameCh:
			if se, ok := wr.write.(StreamError); ok {
				sc.resetStream(se)
				break
			}
			sc.writeFrame(wr)
		case res := <-sc.wroteFrameCh:
			sc.wroteFrame(res)
		case res := <-sc.readFrameCh:
			lastFrameTime = time.Now()
			// Process any written frames before reading new frames from the client since a
			// written frame could have triggered a new stream to be started.
			if sc.writingFrameAsync {
				select {
				case wroteRes := <-sc.wroteFrameCh:
					sc.wroteFrame(wroteRes)
				default:
				}
			}
			if !sc.processFrameFromReader(res) {
				return
			}
			res.readMore()
			if settingsTimer != nil {
				settingsTimer.Stop()
				settingsTimer = nil
			}
		case m := <-sc.bodyReadCh:
			sc.noteBodyRead(m.st, m.n)
		case msg := <-sc.serveMsgCh:
			switch v := msg.(type) {
			case func(int):
				v(loopNum) // for testing
			case *serverMessage:
				switch v {
				case settingsTimerMsg:
					sc.logf("timeout waiting for SETTINGS frames from %v", sc.conn.RemoteAddr())
					return
				case idleTimerMsg:
					sc.vlogf("connection is idle")
					sc.goAway(ErrCodeNo)
				case readIdleTimerMsg:
					sc.handlePingTimer(lastFrameTime)
				case shutdownTimerMsg:
					sc.vlogf("GOAWAY close timer fired; closing conn from %v", sc.conn.RemoteAddr())
					return
				case gracefulShutdownMsg:
					sc.startGracefulShutdownInternal()
				case handlerDoneMsg:
					sc.handlerDone()
				default:
					panic("unknown timer")
				}
			case *startPushRequest:
				sc.startPush(v)
			case func(*serverConn):
				v(sc)
			default:
				panic(fmt.Sprintf("unexpected type %T", v))
			}
		}

		// If the peer is causing us to generate a lot of control frames,
		// but not reading them from us, assume they are trying to make us
		// run out of memory.
		if sc.queuedControlFrames > maxQueuedControlFrames {
			sc.vlogf("http2: too many control frames in send queue, closing connection")
			return
		}

		// Start the shutdown timer after sending a GOAWAY. When sending GOAWAY
		// with no error code (graceful shutdown), don't start the timer until
		// all open streams have been completed.
		sentGoAway := sc.inGoAway && !sc.needToSendGoAway && !sc.writingFrame
		gracefulShutdownComplete := sc.goAwayCode == ErrCodeNo && sc.curOpenStreams() == 0
		if sentGoAway && sc.shutdownTimer == nil && (sc.goAwayCode != ErrCodeNo || gracefulShutdownComplete) {
			sc.shutDownIn(goAwayTimeout)
		}
	}
}

func (sc *serverConn) handlePingTimer(lastFrameReadTime time.Time) {
	if sc.pingSent {
		sc.logf("timeout waiting for PING response")
		if f := sc.countErrorFunc; f != nil {
			f("conn_close_lost_ping")
		}
		sc.conn.Close()
		return
	}

	pingAt := lastFrameReadTime.Add(sc.readIdleTimeout)
	now := time.Now()
	if pingAt.After(now) {
		// We received frames since arming the ping timer.
		// Reset it for the next possible timeout.
		sc.readIdleTimer.Reset(pingAt.Sub(now))
		return
	}

	sc.pingSent = true
	// Ignore crypto/rand.Read errors: It generally can't fail, and worse case if it does
	// is we send a PING frame containing 0s.
	_, _ = rand.Read(sc.sentPingData[:])
	sc.writeFrame(FrameWriteRequest{
		write: &writePing{data: sc.sentPingData},
	})
	sc.readIdleTimer.Reset(sc.pingTimeout)
}

type serverMessage int

// Message values sent to serveMsgCh.
var (
	settingsTimerMsg    = new(serverMessage)
	idleTimerMsg        = new(serverMessage)
	readIdleTimerMsg    = new(serverMessage)
	shutdownTimerMsg    = new(serverMessage)
	gracefulShutdownMsg = new(serverMessage)
	handlerDoneMsg      = new(serverMessage)
)

func (sc *serverConn) onSettingsTimer() { sc.sendServeMsg(settingsTimerMsg) }
func (sc *serverConn) onIdleTimer()     { sc.sendServeMsg(idleTimerMsg) }
func (sc *serverConn) onReadIdleTimer() { sc.sendServeMsg(readIdleTimerMsg) }
func (sc *serverConn) onShutdownTimer() { sc.sendServeMsg(shutdownTimerMsg) }

func (sc *serverConn) sendServeMsg(msg any) {
	sc.serveG.checkNotOn() // NOT
	select {
	case sc.serveMsgCh <- msg:
	case <-sc.doneServing:
	}
}

var errPrefaceTimeout = errors.New("timeout waiting for client preface")

// readPreface reads the ClientPreface greeting from the peer or
// returns errPrefaceTimeout on timeout, or an error if the greeting
// is invalid.
func (sc *serverConn) readPreface() error {
	if sc.sawClientPreface {
		return nil
	}
	errc := make(chan error, 1)
	go func() {
		// Read the client preface
		buf := make([]byte, len(ClientPreface))
		if _, err := io.ReadFull(sc.conn, buf); err != nil {
			errc <- err
		} else if !bytes.Equal(buf, clientPreface) {
			errc <- fmt.Errorf("bogus greeting %q", buf)
		} else {
			errc <- nil
		}
	}()
	timer := time.NewTimer(prefaceTimeout) // TODO: configurable on *Server?
	defer timer.Stop()
	select {
	case <-timer.C:
		return errPrefaceTimeout
	case err := <-errc:
		if err == nil {
			if VerboseLogs {
				sc.vlogf("http2: server: client %v said hello", sc.conn.RemoteAddr())
			}
		}
		return err
	}
}

var writeDataPool = sync.Pool{
	New: func() any { return new(writeData) },
}

// writeDataFromHandler writes DATA response frames from a handler on
// the given stream.
func (sc *serverConn) writeDataFromHandler(stream *stream, data []byte, endStream bool) error {
	ch := sc.srv.getErrChan()
	writeArg := writeDataPool.Get().(*writeData)
	*writeArg = writeData{stream.id, data, endStream}
	err := sc.writeFrameFromHandler(FrameWriteRequest{
		write:  writeArg,
		stream: stream,
		done:   ch,
	})
	if err != nil {
		return err
	}
	var frameWriteDone bool // the frame write is done (successfully or not)
	select {
	case err = <-ch:
		frameWriteDone = true
	case <-sc.doneServing:
		return errClientDisconnected
	case <-stream.cw:
		// If both ch and stream.cw were ready (as might
		// happen on the final Write after an http.Handler
		// ends), prefer the write result. Otherwise this
		// might just be us successfully closing the stream.
		// The writeFrameAsync and serve goroutines guarantee
		// that the ch send will happen before the stream.cw
		// close.
		select {
		case err = <-ch:
			frameWriteDone = true
		default:
			return errStreamClosed
		}
	}
	sc.srv.putErrChan(ch)
	if frameWriteDone {
		writeDataPool.Put(writeArg)
	}
	return err
}

// writeFrameFromHandler sends wr to sc.wantWriteFrameCh, but aborts
// if the connection has gone away.
//
// This must not be run from the serve goroutine itself, else it might
// deadlock writing to sc.wantWriteFrameCh (which is only mildly
// buffered and is read by serve itself). If you're on the serve
// goroutine, call writeFrame instead.
func (sc *serverConn) writeFrameFromHandler(wr FrameWriteRequest) error {
	sc.serveG.checkNotOn() // NOT
	select {
	case sc.wantWriteFrameCh <- wr:
		return nil
	case <-sc.doneServing:
		// Serve loop is gone.
		// Client has closed their connection to the server.
		return errClientDisconnected
	}
}

// writeFrame schedules a frame to write and sends it if there's nothing
// already being written.
//
// There is no pushback here (the serve goroutine never blocks). It's
// the http.Handlers that block, waiting for their previous frames to
// make it onto the wire
//
// If you're not on the serve goroutine, use writeFrameFromHandler instead.
func (sc *serverConn) writeFrame(wr FrameWriteRequest) {
	sc.serveG.check()

	// If true, wr will not be written and wr.done will not be signaled.
	var ignoreWrite bool

	// We are not allowed to write frames on closed streams. RFC 7540 Section
	// 5.1.1 says: "An endpoint MUST NOT send frames other than PRIORITY on
	// a closed stream." Our server never sends PRIORITY, so that exception
	// does not apply.
	//
	// The serverConn might close an open stream while the stream's handler
	// is still running. For example, the server might close a stream when it
	// receives bad data from the client. If this happens, the handler might
	// attempt to write a frame after the stream has been closed (since the
	// handler hasn't yet been notified of the close). In this case, we simply
	// ignore the frame. The handler will notice that the stream is closed when
	// it waits for the frame to be written.
	//
	// As an exception to this rule, we allow sending RST_STREAM after close.
	// This allows us to immediately reject new streams without tracking any
	// state for those streams (except for the queued RST_STREAM frame). This
	// may result in duplicate RST_STREAMs in some cases, but the client should
	// ignore those.
	if wr.StreamID() != 0 {
		_, isReset := wr.write.(StreamError)
		if state, _ := sc.state(wr.StreamID()); state == stateClosed && !isReset {
			ignoreWrite = true
		}
	}

	// Don't send a 100-continue response if we've already sent headers.
	// See golang.org/issue/14030.
	switch wr.write.(type) {
	case *writeResHeaders:
		wr.stream.wroteHeaders = true
	case write100ContinueHeadersFrame:
		if wr.stream.wroteHeaders {
			// We do not need to notify wr.done because this frame is
			// never written with wr.done != nil.
			if wr.done != nil {
				panic("wr.done != nil for write100ContinueHeadersFrame")
			}
			ignoreWrite = true
		}
	}

	if !ignoreWrite {
		if wr.isControl() {
			sc.queuedControlFrames++
			// For extra safety, detect wraparounds, which should not happen,
			// and pull the plug.
			if sc.queuedControlFrames < 0 {
				sc.conn.Close()
			}
		}
		sc.writeSched.Push(wr)
	}
	sc.scheduleFrameWrite()
}

// startFrameWrite starts a goroutine to write wr (in a separate
// goroutine since that might block on the network), and updates the
// serve goroutine's state about the world, updated from info in wr.
func (sc *serverConn) startFrameWrite(wr FrameWriteRequest) {
	sc.serveG.check()
	if sc.writingFrame {
		panic("internal error: can only be writing one frame at a time")
	}

	st := wr.stream
	if st != nil {
		switch st.state {
		case stateHalfClosedLocal:
			switch wr.write.(type) {
			case StreamError, handlerPanicRST, writeWindowUpdate:
				// RFC 7540 Section 5.1 allows sending RST_STREAM, PRIORITY, and WINDOW_UPDATE
				// in this state. (We never send PRIORITY from the server, so that is not checked.)
			default:
				panic(fmt.Sprintf("internal error: attempt to send frame on a half-closed-local stream: %v", wr))
			}
		case stateClosed:
			panic(fmt.Sprintf("internal error: attempt to send frame on a closed stream: %v", wr))
		}
	}
	if wpp, ok := wr.write.(*writePushPromise); ok {
		var err error
		wpp.promisedID, err = wpp.allocatePromisedID()
		if err != nil {
			sc.writingFrameAsync = false
			wr.replyToWriter(err)
			return
		}
	}

	sc.writingFrame = true
	sc.needsFrameFlush = true
	if wr.write.staysWithinBuffer(sc.bw.Available()) {
		sc.writingFrameAsync = false
		err := wr.write.writeFrame(sc)
		sc.wroteFrame(frameWriteResult{wr: wr, err: err})
	} else if wd, ok := wr.write.(*writeData); ok {
		// Encode the frame in the serve goroutine, to ensure we don't have
		// any lingering asynchronous references to data passed to Write.
		// See https://go.dev/issue/58446.
		sc.framer.startWriteDataPadded(wd.streamID, wd.endStream, wd.p, nil)
		sc.writingFrameAsync = true
		go sc.writeFrameAsync(wr, wd)
	} else {
		sc.writingFrameAsync = true
		go sc.writeFrameAsync(wr, nil)
	}
}

// errHandlerPanicked is the error given to any callers blocked in a read from
// Request.Body when the main goroutine panics. Since most handlers read in the
// main ServeHTTP goroutine, this will show up rarely.
var errHandlerPanicked = errors.New("http2: handler panicked")

// wroteFrame is called on the serve goroutine with the result of
// whatever happened on writeFrameAsync.
func (sc *serverConn) wroteFrame(res frameWriteResult) {
	sc.serveG.check()
	if !sc.writingFrame {
		panic("internal error: expected to be already writing a frame")
	}
	sc.writingFrame = false
	sc.writingFrameAsync = false

	if res.err != nil {
		sc.conn.Close()
	}

	wr := res.wr

	if writeEndsStream(wr.write) {
		st := wr.stream
		if st == nil {
			panic("internal error: expecting non-nil stream")
		}
		switch st.state {
		case stateOpen:
			// Here we would go to stateHalfClosedLocal in
			// theory, but since our handler is done and
			// the net/http package provides no mechanism
			// for closing a ResponseWriter while still
			// reading data (see possible TODO at top of
			// this file), we go into closed state here
			// anyway, after telling the peer we're
			// hanging up on them. We'll transition to
			// stateClosed after the RST_STREAM frame is
			// written.
			st.state = stateHalfClosedLocal
			// Section 8.1: a server MAY request that the client abort
			// transmission of a request without error by sending a
			// RST_STREAM with an error code of NO_ERROR after sending
			// a complete response.
			sc.resetStream(streamError(st.id, ErrCodeNo))
		case stateHalfClosedRemote:
			sc.closeStream(st, errHandlerComplete)
		}
	} else {
		switch v := wr.write.(type) {
		case StreamError:
			// st may be unknown if the RST_STREAM was generated to reject bad input.
			if st, ok := sc.streams[v.StreamID]; ok {
				sc.closeStream(st, v)
			}
		case handlerPanicRST:
			sc.closeStream(wr.stream, errHandlerPanicked)
		}
	}

	// Reply (if requested) to unblock the ServeHTTP goroutine.
	wr.replyToWriter(res.err)

	sc.scheduleFrameWrite()
}

// scheduleFrameWrite tickles the frame writing scheduler.
//
// If a frame is already being written, nothing happens. This will be called again
// when the frame is done being written.
//
// If a frame isn't being written and we need to send one, the best frame
// to send is selected by writeSched.
//
// If a frame isn't being written and there's nothing else to send, we
// flush the write buffer.
func (sc *serverConn) scheduleFrameWrite() {
	sc.serveG.check()
	if sc.writingFrame || sc.inFrameScheduleLoop {
		return
	}
	sc.inFrameScheduleLoop = true
	for !sc.writingFrameAsync {
		if sc.needToSendGoAway {
			sc.needToSendGoAway = false
			sc.startFrameWrite(FrameWriteRequest{
				write: &writeGoAway{
					maxStreamID: sc.maxClientStreamID,
					code:        sc.goAwayCode,
				},
			})
			continue
		}
		if sc.needToSendSettingsAck {
			sc.needToSendSettingsAck = false
			sc.startFrameWrite(FrameWriteRequest{write: writeSettingsAck{}})
			continue
		}
		if !sc.inGoAway || sc.goAwayCode == ErrCodeNo {
			if wr, ok := sc.writeSched.Pop(); ok {
				if wr.isControl() {
					sc.queuedControlFrames--
				}
				sc.startFrameWrite(wr)
				continue
			}
		}
		if sc.needsFrameFlush {
			sc.startFrameWrite(FrameWriteRequest{write: flushFrameWriter{}})
			sc.needsFrameFlush = false // after startFrameWrite, since it sets this true
			continue
		}
		break
	}
	sc.inFrameScheduleLoop = false
}

// startGracefulShutdown gracefully shuts down a connection. This
// sends GOAWAY with ErrCodeNo to tell the client we're gracefully
// shutting down. The connection isn't closed until all current
// streams are done.
//
// startGracefulShutdown returns immediately; it does not wait until
// the connection has shut down.
func (sc *serverConn) startGracefulShutdown() {
	sc.serveG.checkNotOn() // NOT
	sc.shutdownOnce.Do(func() { sc.sendServeMsg(gracefulShutdownMsg) })
}

// After sending GOAWAY with an error code (non-graceful shutdown), the
// connection will close after goAwayTimeout.
//
// If we close the connection immediately after sending GOAWAY, there may
// be unsent data in our kernel receive buffer, which will cause the kernel
// to send a TCP RST on close() instead of a FIN. This RST will abort the
// connection immediately, whether or not the client had received the GOAWAY.
//
// Ideally we should delay for at least 1 RTT + epsilon so the client has
// a chance to read the GOAWAY and stop sending messages. Measuring RTT
// is hard, so we approximate with 1 second. See golang.org/issue/18701.
//
// This is a var so it can be shorter in tests, where all requests uses the
// loopback interface making the expected RTT very small.
//
// TODO: configurable?
var goAwayTimeout = 1 * time.Second

func (sc *serverConn) startGracefulShutdownInternal() {
	sc.goAway(ErrCodeNo)
}

func (sc *serverConn) goAway(code ErrCode) {
	sc.serveG.check()
	if sc.inGoAway {
		if sc.goAwayCode == ErrCodeNo {
			sc.goAwayCode = code
		}
		return
	}
	sc.inGoAway = true
	sc.needToSendGoAway = true
	sc.goAwayCode = code
	sc.scheduleFrameWrite()
}

func (sc *serverConn) shutDownIn(d time.Duration) {
	sc.serveG.check()
	sc.shutdownTimer = time.AfterFunc(d, sc.onShutdownTimer)
}

func (sc *serverConn) resetStream(se StreamError) {
	sc.serveG.check()
	sc.writeFrame(FrameWriteRequest{write: se})
	if st, ok := sc.streams[se.StreamID]; ok {
		st.resetQueued = true
	}
}

// processFrameFromReader processes the serve loop's read from readFrameCh from the
// frame-reading goroutine.
// processFrameFromReader returns whether the connection should be kept open.
func (sc *serverConn) processFrameFromReader(res readFrameResult) bool {
	sc.serveG.check()
	err := res.err
	if err != nil {
		if err == ErrFrameTooLarge {
			sc.goAway(ErrCodeFrameSize)
			return true // goAway will close the loop
		}
		clientGone := err == io.EOF || err == io.ErrUnexpectedEOF || isClosedConnError(err)
		if clientGone {
			// TODO: could we also get into this state if
			// the peer does a half close
			// (e.g. CloseWrite) because they're done
			// sending frames but they're still wanting
			// our open replies?  Investigate.
			// TODO: add CloseWrite to crypto/tls.Conn first
			// so we have a way to test this? I suppose
			// just for testing we could have a non-TLS mode.
			return false
		}
	} else {
		f := res.f
		if VerboseLogs {
			sc.vlogf("http2: server read frame %v", summarizeFrame(f))
		}
		err = sc.processFrame(f)
		if err == nil {
			return true
		}
	}

	switch ev := err.(type) {
	case StreamError:
		sc.resetStream(ev)
		return true
	case goAwayFlowError:
		sc.goAway(ErrCodeFlowControl)
		return true
	case ConnectionError:
		if res.f != nil {
			if id := res.f.Header().StreamID; id > sc.maxClientStreamID {
				sc.maxClientStreamID = id
			}
		}
		sc.logf("http2: server connection error from %v: %v", sc.conn.RemoteAddr(), ev)
		sc.goAway(ErrCode(ev))
		return true // goAway will handle shutdown
	default:
		if res.err != nil {
			sc.vlogf("http2: server closing client connection; error reading frame from client %s: %v", sc.conn.RemoteAddr(), err)
		} else {
			sc.logf("http2: server closing client connection: %v", err)
		}
		return false
	}
}

func (sc *serverConn) processFrame(f Frame) error {
	sc.serveG.check()

	// First frame received must be SETTINGS.
	if !sc.sawFirstSettings {
		if _, ok := f.(*SettingsFrame); !ok {
			return sc.countError("first_settings", ConnectionError(ErrCodeProtocol))
		}
		sc.sawFirstSettings = true
	}

	// Discard frames for streams initiated after the identified last
	// stream sent in a GOAWAY, or all frames after sending an error.
	// We still need to return connection-level flow control for DATA frames.
	// RFC 9113 Section 6.8.
	if sc.inGoAway && (sc.goAwayCode != ErrCodeNo || f.Header().StreamID > sc.maxClientStreamID) {

		if f, ok := f.(*DataFrame); ok {
			if !sc.inflow.take(f.Length) {
				return sc.countError("data_flow", streamError(f.Header().StreamID, ErrCodeFlowControl))
			}
			sc.sendWindowUpdate(nil, int(f.Length)) // conn-level
		}
		return nil
	}

	switch f := f.(type) {
	case *SettingsFrame:
		return sc.processSettings(f)
	case *MetaHeadersFrame:
		return sc.processHeaders(f)
	case *WindowUpdateFrame:
		return sc.processWindowUpdate(f)
	case *PingFrame:
		return sc.processPing(f)
	case *DataFrame:
		return sc.processData(f)
	case *RSTStreamFrame:
		return sc.processResetStream(f)
	case *PriorityFrame:
		return sc.processPriority(f)
	case *GoAwayFrame:
		return sc.processGoAway(f)
	case *PushPromiseFrame:
		// A client cannot push. Thus, servers MUST treat the receipt of a PUSH_PROMISE
		// frame as a connection error (Section 5.4.1) of type PROTOCOL_ERROR.
		return sc.countError("push_promise", ConnectionError(ErrCodeProtocol))
	case *PriorityUpdateFrame:
		return sc.processPriorityUpdate(f)
	default:
		sc.vlogf("http2: server ignoring frame: %v", f.Header())
		return nil
	}
}

func (sc *serverConn) processPing(f *PingFrame) error {
	sc.serveG.check()
	if f.IsAck() {
		if sc.pingSent && sc.sentPingData == f.Data {
			// This is a response to a PING we sent.
			sc.pingSent = false
			sc.readIdleTimer.Reset(sc.readIdleTimeout)
		}
		// 6.7 PING: " An endpoint MUST NOT respond to PING frames
		// containing this flag."
		return nil
	}
	if f.StreamID != 0 {
		// "PING frames are not associated with any individual
		// stream. If a PING frame is received with a stream
		// identifier field value other than 0x0, the recipient MUST
		// respond with a connection error (Section 5.4.1) of type
		// PROTOCOL_ERROR."
		return sc.countError("ping_on_stream", ConnectionError(ErrCodeProtocol))
	}
	sc.writeFrame(FrameWriteRequest{write: writePingAck{f}})
	return nil
}

func (sc *serverConn) processWindowUpdate(f *WindowUpdateFrame) error {
	sc.serveG.check()
	switch {
	case f.StreamID != 0: // stream-level flow control
		state, st := sc.state(f.StreamID)
		if state == stateIdle {
			// Section 5.1: "Receiving any frame other than HEADERS
			// or PRIORITY on a stream in this state MUST be
			// treated as a connection error (Section 5.4.1) of
			// type PROTOCOL_ERROR."
			return sc.countError("stream_idle", ConnectionError(ErrCodeProtocol))
		}
		if st == nil {
			// "WINDOW_UPDATE can be sent by a peer that has sent a
			// frame bearing the END_STREAM flag. This means that a
			// receiver could receive a WINDOW_UPDATE frame on a "half
			// closed (remote)" or "closed" stream. A receiver MUST
			// NOT treat this as an error, see Section 5.1."
			return nil
		}
		if !st.flow.add(int32(f.Increment)) {
			return sc.countError("bad_flow", streamError(f.StreamID, ErrCodeFlowControl))
		}
	default: // connection-level flow control
		if !sc.flow.add(int32(f.Increment)) {
			return goAwayFlowError{}
		}
	}
	sc.scheduleFrameWrite()
	return nil
}

func (sc *serverConn) processResetStream(f *RSTStreamFrame) error {
	sc.serveG.check()

	state, st := sc.state(f.StreamID)
	if state == stateIdle {
		// 6.4 "RST_STREAM frames MUST NOT be sent for a
		// stream in the "idle" state. If a RST_STREAM frame
		// identifying an idle stream is received, the
		// recipient MUST treat this as a connection error
		// (Section 5.4.1) of type PROTOCOL_ERROR.
		return sc.countError("reset_idle_stream", ConnectionError(ErrCodeProtocol))
	}
	if st != nil {
		st.cancelCtx()
		sc.closeStream(st, streamError(f.StreamID, f.ErrCode))
	}
	return nil
}

func (sc *serverConn) closeStream(st *stream, err error) {
	sc.serveG.check()
	if st.state == stateIdle || st.state == stateClosed {
		panic(fmt.Sprintf("invariant; can't close stream in state %v", st.state))
	}
	st.state = stateClosed
	if st.readDeadline != nil {
		st.readDeadline.Stop()
	}
	if st.writeDeadline != nil {
		st.writeDeadline.Stop()
	}
	if st.isPushed() {
		sc.curPushedStreams--
	} else {
		sc.curClientStreams--
	}
	delete(sc.streams, st.id)
	if len(sc.streams) == 0 {
		sc.setConnState(ConnStateIdle)
		idleTimeout := sc.hs.IdleTimeout()
		if idleTimeout > 0 && sc.idleTimer != nil {
			sc.idleTimer.Reset(idleTimeout)
		}
		if h1ServerKeepAlivesDisabled(sc.hs) {
			sc.startGracefulShutdownInternal()
		}
	}
	if p := st.body; p != nil {
		// Return any buffered unread bytes worth of conn-level flow control.
		// See golang.org/issue/16481
		sc.sendWindowUpdate(nil, p.Len())

		p.CloseWithError(err)
	}
	if e, ok := err.(StreamError); ok {
		if e.Cause != nil {
			err = e.Cause
		} else {
			err = errStreamClosed
		}
	}
	st.closeErr = err
	st.cancelCtx()
	st.cw.Close() // signals Handler's CloseNotifier, unblocks writes, etc
	sc.writeSched.CloseStream(st.id)
}

func (sc *serverConn) processSettings(f *SettingsFrame) error {
	sc.serveG.check()
	if f.IsAck() {
		sc.unackedSettings--
		if sc.unackedSettings < 0 {
			// Why is the peer ACKing settings we never sent?
			// The spec doesn't mention this case, but
			// hang up on them anyway.
			return sc.countError("ack_mystery", ConnectionError(ErrCodeProtocol))
		}
		return nil
	}
	if f.NumSettings() > 100 || f.HasDuplicates() {
		// This isn't actually in the spec, but hang up on
		// suspiciously large settings frames or those with
		// duplicate entries.
		return sc.countError("settings_big_or_dups", ConnectionError(ErrCodeProtocol))
	}
	if err := f.ForeachSetting(sc.processSetting); err != nil {
		return err
	}
	// TODO: judging by RFC 7540, Section 6.5.3 each SETTINGS frame should be
	// acknowledged individually, even if multiple are received before the ACK.
	sc.needToSendSettingsAck = true
	sc.scheduleFrameWrite()
	return nil
}

func (sc *serverConn) processSetting(s Setting) error {
	sc.serveG.check()
	if err := s.Valid(); err != nil {
		return err
	}
	if VerboseLogs {
		sc.vlogf("http2: server processing setting %v", s)
	}
	switch s.ID {
	case SettingHeaderTableSize:
		sc.hpackEncoder.SetMaxDynamicTableSize(s.Val)
	case SettingEnablePush:
		sc.pushEnabled = s.Val != 0
	case SettingMaxConcurrentStreams:
		sc.clientMaxStreams = s.Val
	case SettingInitialWindowSize:
		return sc.processSettingInitialWindowSize(s.Val)
	case SettingMaxFrameSize:
		sc.maxFrameSize = int32(s.Val) // the maximum valid s.Val is < 2^31
	case SettingMaxHeaderListSize:
		sc.peerMaxHeaderListSize = s.Val
	case SettingEnableConnectProtocol:
		// Receipt of this parameter by a server does not
		// have any impact
	case SettingNoRFC7540Priorities:
		if s.Val > 1 {
			return ConnectionError(ErrCodeProtocol)
		}
	default:
		// Unknown setting: "An endpoint that receives a SETTINGS
		// frame with any unknown or unsupported identifier MUST
		// ignore that setting."
		if VerboseLogs {
			sc.vlogf("http2: server ignoring unknown setting %v", s)
		}
	}
	return nil
}

func (sc *serverConn) processSettingInitialWindowSize(val uint32) error {
	sc.serveG.check()
	// Note: val already validated to be within range by
	// processSetting's Valid call.

	// "A SETTINGS frame can alter the initial flow control window
	// size for all current streams. When the value of
	// SETTINGS_INITIAL_WINDOW_SIZE changes, a receiver MUST
	// adjust the size of all stream flow control windows that it
	// maintains by the difference between the new value and the
	// old value."
	old := sc.initialStreamSendWindowSize
	sc.initialStreamSendWindowSize = int32(val)
	growth := int32(val) - old // may be negative
	for _, st := range sc.streams {
		if !st.flow.add(growth) {
			// 6.9.2 Initial Flow Control Window Size
			// "An endpoint MUST treat a change to
			// SETTINGS_INITIAL_WINDOW_SIZE that causes any flow
			// control window to exceed the maximum size as a
			// connection error (Section 5.4.1) of type
			// FLOW_CONTROL_ERROR."
			return sc.countError("setting_win_size", ConnectionError(ErrCodeFlowControl))
		}
	}
	return nil
}

func (sc *serverConn) processData(f *DataFrame) error {
	sc.serveG.check()
	id := f.Header().StreamID

	data := f.Data()
	state, st := sc.state(id)
	if id == 0 || state == stateIdle {
		// Section 6.1: "DATA frames MUST be associated with a
		// stream. If a DATA frame is received whose stream
		// identifier field is 0x0, the recipient MUST respond
		// with a connection error (Section 5.4.1) of type
		// PROTOCOL_ERROR."
		//
		// Section 5.1: "Receiving any frame other than HEADERS
		// or PRIORITY on a stream in this state MUST be
		// treated as a connection error (Section 5.4.1) of
		// type PROTOCOL_ERROR."
		return sc.countError("data_on_idle", ConnectionError(ErrCodeProtocol))
	}

	// "If a DATA frame is received whose stream is not in "open"
	// or "half closed (local)" state, the recipient MUST respond
	// with a stream error (Section 5.4.2) of type STREAM_CLOSED."
	if st == nil || state != stateOpen || st.gotTrailerHeader || st.resetQueued {
		// This includes sending a RST_STREAM if the stream is
		// in stateHalfClosedLocal (which currently means that
		// the http.Handler returned, so it's done reading &
		// done writing). Try to stop the client from sending
		// more DATA.

		// But still enforce their connection-level flow control,
		// and return any flow control bytes since we're not going
		// to consume them.
		if !sc.inflow.take(f.Length) {
			return sc.countError("data_flow", streamError(id, ErrCodeFlowControl))
		}
		sc.sendWindowUpdate(nil, int(f.Length)) // conn-level

		if st != nil && st.resetQueued {
			// Already have a stream error in flight. Don't send another.
			return nil
		}
		return sc.countError("closed", streamError(id, ErrCodeStreamClosed))
	}
	if st.body == nil {
		panic("internal error: should have a body in this state")
	}

	// Sender sending more than they'd declared?
	if st.declBodyBytes != -1 && st.bodyBytes+int64(len(data)) > st.declBodyBytes {
		if !sc.inflow.take(f.Length) {
			return sc.countError("data_flow", streamError(id, ErrCodeFlowControl))
		}
		sc.sendWindowUpdate(nil, int(f.Length)) // conn-level

		st.body.CloseWithError(fmt.Errorf("sender tried to send more than declared Content-Length of %d bytes", st.declBodyBytes))
		// RFC 7540, sec 8.1.2.6: A request or response is also malformed if the
		// value of a content-length header field does not equal the sum of the
		// DATA frame payload lengths that form the body.
		return sc.countError("send_too_much", streamError(id, ErrCodeProtocol))
	}
	if f.Length > 0 {
		// Check whether the client has flow control quota.
		if !takeInflows(&sc.inflow, &st.inflow, f.Length) {
			return sc.countError("flow_on_data_length", streamError(id, ErrCodeFlowControl))
		}

		if len(data) > 0 {
			st.bodyBytes += int64(len(data))
			wrote, err := st.body.Write(data)
			if err != nil {
				// The handler has closed the request body.
				// Return the connection-level flow control for the discarded data,
				// but not the stream-level flow control.
				sc.sendWindowUpdate(nil, int(f.Length)-wrote)
				return nil
			}
			if wrote != len(data) {
				panic("internal error: bad Writer")
			}
		}

		// Return any padded flow control now, since we won't
		// refund it later on body reads.
		// Call sendWindowUpdate even if there is no padding,
		// to return buffered flow control credit if the sent
		// window has shrunk.
		pad := int32(f.Length) - int32(len(data))
		sc.sendWindowUpdate32(nil, pad)
		sc.sendWindowUpdate32(st, pad)
	}
	if f.StreamEnded() {
		st.endStream()
	}
	return nil
}

func (sc *serverConn) processGoAway(f *GoAwayFrame) error {
	sc.serveG.check()
	if f.ErrCode != ErrCodeNo {
		sc.logf("http2: received GOAWAY %+v, starting graceful shutdown", f)
	} else {
		sc.vlogf("http2: received GOAWAY %+v, starting graceful shutdown", f)
	}
	sc.startGracefulShutdownInternal()
	// http://tools.ietf.org/html/rfc7540#section-6.8
	// We should not create any new streams, which means we should disable push.
	sc.pushEnabled = false
	return nil
}

// isPushed reports whether the stream is server-initiated.
func (st *stream) isPushed() bool {
	return st.id%2 == 0
}

// endStream closes a Request.Body's pipe. It is called when a DATA
// frame says a request body is over (or after trailers).
func (st *stream) endStream() {
	sc := st.sc
	sc.serveG.check()

	if st.declBodyBytes != -1 && st.declBodyBytes != st.bodyBytes {
		st.body.CloseWithError(fmt.Errorf("request declared a Content-Length of %d but only wrote %d bytes",
			st.declBodyBytes, st.bodyBytes))
	} else {
		st.body.closeWithErrorAndCode(io.EOF, st.copyTrailersToHandlerRequest)
		st.body.CloseWithError(io.EOF)
	}
	st.state = stateHalfClosedRemote
}

// copyTrailersToHandlerRequest is run in the Handler's goroutine in
// its Request.Body.Read just before it gets io.EOF.
func (st *stream) copyTrailersToHandlerRequest() {
	for k, vv := range st.trailer {
		if _, ok := st.reqTrailer[k]; ok {
			// Only copy it over it was pre-declared.
			st.reqTrailer[k] = vv
		}
	}
}

// onReadTimeout is run on its own goroutine (from time.AfterFunc)
// when the stream's ReadTimeout has fired.
func (st *stream) onReadTimeout() {
	if st.body != nil {
		// Wrap the ErrDeadlineExceeded to avoid callers depending on us
		// returning the bare error.
		st.body.CloseWithError(fmt.Errorf("%w", os.ErrDeadlineExceeded))
	}
}

// onWriteTimeout is run on its own goroutine (from time.AfterFunc)
// when the stream's WriteTimeout has fired.
func (st *stream) onWriteTimeout() {
	st.sc.writeFrameFromHandler(FrameWriteRequest{write: StreamError{
		StreamID: st.id,
		Code:     ErrCodeInternal,
		Cause:    os.ErrDeadlineExceeded,
	}})
}

func (sc *serverConn) processHeaders(f *MetaHeadersFrame) error {
	sc.serveG.check()
	id := f.StreamID
	// http://tools.ietf.org/html/rfc7540#section-5.1.1
	// Streams initiated by a client MUST use odd-numbered stream
	// identifiers. [...] An endpoint that receives an unexpected
	// stream identifier MUST respond with a connection error
	// (Section 5.4.1) of type PROTOCOL_ERROR.
	if id%2 != 1 {
		return sc.countError("headers_even", ConnectionError(ErrCodeProtocol))
	}
	// A HEADERS frame can be used to create a new stream or
	// send a trailer for an open one. If we already have a stream
	// open, let it process its own HEADERS frame (trailers at this
	// point, if it's valid).
	if st := sc.streams[f.StreamID]; st != nil {
		if st.resetQueued {
			// We're sending RST_STREAM to close the stream, so don't bother
			// processing this frame.
			return nil
		}
		// RFC 7540, sec 5.1: If an endpoint receives additional frames, other than
		// WINDOW_UPDATE, PRIORITY, or RST_STREAM, for a stream that is in
		// this state, it MUST respond with a stream error (Section 5.4.2) of
		// type STREAM_CLOSED.
		if st.state == stateHalfClosedRemote {
			return sc.countError("headers_half_closed", streamError(id, ErrCodeStreamClosed))
		}
		return st.processTrailerHeaders(f)
	}

	// [...] The identifier of a newly established stream MUST be
	// numerically greater than all streams that the initiating
	// endpoint has opened or reserved. [...]  An endpoint that
	// receives an unexpected stream identifier MUST respond with
	// a connection error (Section 5.4.1) of type PROTOCOL_ERROR.
	if id <= sc.maxClientStreamID {
		return sc.countError("stream_went_down", ConnectionError(ErrCodeProtocol))
	}
	sc.maxClientStreamID = id

	if sc.idleTimer != nil {
		sc.idleTimer.Stop()
	}

	// http://tools.ietf.org/html/rfc7540#section-5.1.2
	// [...] Endpoints MUST NOT exceed the limit set by their peer. An
	// endpoint that receives a HEADERS frame that causes their
	// advertised concurrent stream limit to be exceeded MUST treat
	// this as a stream error (Section 5.4.2) of type PROTOCOL_ERROR
	// or REFUSED_STREAM.
	if sc.curClientStreams+1 > sc.advMaxStreams {
		if sc.unackedSettings == 0 {
			// They should know better.
			return sc.countError("over_max_streams", streamError(id, ErrCodeProtocol))
		}
		// Assume it's a network race, where they just haven't
		// received our last SETTINGS update. But actually
		// this can't happen yet, because we don't yet provide
		// a way for users to adjust server parameters at
		// runtime.
		return sc.countError("over_max_streams_race", streamError(id, ErrCodeRefusedStream))
	}

	initialState := stateOpen
	if f.StreamEnded() {
		initialState = stateHalfClosedRemote
	}

	// We are handling two special cases here:
	// 1. When a request is sent via an intermediary, we force priority to be
	// u=3,i. This is essentially a round-robin behavior, and is done to ensure
	// fairness between, for example, multiple clients using the same proxy.
	// 2. Until a client has shown that it is aware of RFC 9218, we make its
	// streams non-incremental by default. This is done to preserve the
	// historical behavior of handling streams in a round-robin manner, rather
	// than one-by-one to completion.
	initialPriority := defaultRFC9218Priority(sc.priorityAware && !sc.hasIntermediary)
	if _, ok := sc.writeSched.(*priorityWriteSchedulerRFC9218); ok && !sc.hasIntermediary {
		headerPriority, priorityAware, hasIntermediary := f.rfc9218Priority(sc.priorityAware)
		initialPriority = headerPriority
		sc.hasIntermediary = hasIntermediary
		if priorityAware {
			sc.priorityAware = true
		}
	}
	st := sc.newStream(id, 0, initialState, initialPriority)

	if f.HasPriority() {
		if err := sc.checkPriority(f.StreamID, f.Priority); err != nil {
			return err
		}
		if !sc.writeSchedIgnoresRFC7540() {
			sc.writeSched.AdjustStream(st.id, f.Priority)
		}
	}

	rw, req, err := sc.newWriterAndRequest(st, f)
	if err != nil {
		return err
	}
	st.reqTrailer = req.Trailer
	if st.reqTrailer != nil {
		st.trailer = make(Header)
	}
	st.body = req.Body.(*requestBody).pipe // may be nil
	st.declBodyBytes = req.ContentLength

	handler := sc.handler.ServeHTTP
	if f.Truncated {
		// Their header list was too long. Send a 431 error.
		handler = handleHeaderListTooLong
	} else if err := checkValidHTTP2RequestHeaders(req.Header); err != nil {
		handler = serve400Handler{err}.ServeHTTP
	}

	// The net/http package sets the read deadline from the
	// http.Server.ReadTimeout during the TLS handshake, but then
	// passes the connection off to us with the deadline already
	// set. Disarm it here after the request headers are read,
	// similar to how the http1 server works. Here it's
	// technically more like the http1 Server's ReadHeaderTimeout
	// (in Go 1.8), though. That's a more sane option anyway.
	if sc.hs.ReadTimeout() > 0 {
		sc.conn.SetReadDeadline(time.Time{})
		st.readDeadline = time.AfterFunc(sc.hs.ReadTimeout(), st.onReadTimeout)
	}

	return sc.scheduleHandler(id, rw, req, handler)
}

func (sc *serverConn) upgradeRequest(req *ServerRequest) {
	sc.serveG.check()
	id := uint32(1)
	sc.maxClientStreamID = id
	st := sc.newStream(id, 0, stateHalfClosedRemote, defaultRFC9218Priority(sc.priorityAware && !sc.hasIntermediary))
	st.reqTrailer = req.Trailer
	if st.reqTrailer != nil {
		st.trailer = make(Header)
	}
	rw := sc.newResponseWriter(st)
	rw.rws.req = *req
	req = &rw.rws.req

	// Disable any read deadline set by the net/http package
	// prior to the upgrade.
	if sc.hs.ReadTimeout() > 0 {
		sc.conn.SetReadDeadline(time.Time{})
	}

	// This is the first request on the connection,
	// so start the handler directly rather than going
	// through scheduleHandler.
	sc.curHandlers++
	go sc.runHandler(rw, req, sc.handler.ServeHTTP)
}

func (st *stream) processTrailerHeaders(f *MetaHeadersFrame) error {
	sc := st.sc
	sc.serveG.check()
	if st.gotTrailerHeader {
		return sc.countError("dup_trailers", ConnectionError(ErrCodeProtocol))
	}
	st.gotTrailerHeader = true
	if !f.StreamEnded() {
		return sc.countError("trailers_not_ended", streamError(st.id, ErrCodeProtocol))
	}

	if len(f.PseudoFields()) > 0 {
		return sc.countError("trailers_pseudo", streamError(st.id, ErrCodeProtocol))
	}
	if st.trailer != nil {
		for _, hf := range f.RegularFields() {
			key := sc.canonicalHeader(hf.Name)
			if !httpguts.ValidTrailerHeader(key) {
				// TODO: send more details to the peer somehow. But http2 has
				// no way to send debug data at a stream level. Discuss with
				// HTTP folk.
				return sc.countError("trailers_bogus", streamError(st.id, ErrCodeProtocol))
			}
			st.trailer[key] = append(st.trailer[key], hf.Value)
		}
	}
	st.endStream()
	return nil
}

func (sc *serverConn) checkPriority(streamID uint32, p PriorityParam) error {
	if streamID == p.StreamDep {
		// Section 5.3.1: "A stream cannot depend on itself. An endpoint MUST treat
		// this as a stream error (Section 5.4.2) of type PROTOCOL_ERROR."
		// Section 5.3.3 says that a stream can depend on one of its dependencies,
		// so it's only self-dependencies that are forbidden.
		return sc.countError("priority", streamError(streamID, ErrCodeProtocol))
	}
	return nil
}

func (sc *serverConn) processPriority(f *PriorityFrame) error {
	if err := sc.checkPriority(f.StreamID, f.PriorityParam); err != nil {
		return err
	}
	// We need to avoid calling AdjustStream when using the RFC 9218 write
	// scheduler. Otherwise, incremental's zero value in PriorityParam will
	// unexpectedly make all streams non-incremental. This causes us to process
	// streams one-by-one to completion rather than doing it in a round-robin
	// manner (the historical behavior), which might be unexpected to users.
	if sc.writeSchedIgnoresRFC7540() {
		return nil
	}
	sc.writeSched.AdjustStream(f.StreamID, f.PriorityParam)
	return nil
}

func (sc *serverConn) processPriorityUpdate(f *PriorityUpdateFrame) error {
	sc.priorityAware = true
	if _, ok := sc.writeSched.(*priorityWriteSchedulerRFC9218); !ok {
		return nil
	}
	p, ok := parseRFC9218Priority(f.Priority, sc.priorityAware)
	if !ok {
		return sc.countError("unparsable_priority_update", streamError(f.PrioritizedStreamID, ErrCodeProtocol))
	}
	sc.writeSched.AdjustStream(f.PrioritizedStreamID, p)
	return nil
}

func (sc *serverConn) newStream(id, pusherID uint32, state streamState, priority PriorityParam) *stream {
	sc.serveG.check()
	if id == 0 {
		panic("internal error: cannot create stream with id 0")
	}

	ctx, cancelCtx := context.WithCancel(sc.baseCtx)
	st := &stream{
		sc:        sc,
		id:        id,
		state:     state,
		ctx:       ctx,
		cancelCtx: cancelCtx,
	}
	st.cw.Init()
	st.flow.conn = &sc.flow // link to conn-level counter
	st.flow.add(sc.initialStreamSendWindowSize)
	st.inflow.init(sc.initialStreamRecvWindowSize)
	if writeTimeout := sc.hs.WriteTimeout(); writeTimeout > 0 {
		st.writeDeadline = time.AfterFunc(writeTimeout, st.onWriteTimeout)
	}

	sc.streams[id] = st
	sc.writeSched.OpenStream(st.id, OpenStreamOptions{PusherID: pusherID, priority: priority})
	if st.isPushed() {
		sc.curPushedStreams++
	} else {
		sc.curClientStreams++
	}
	if sc.curOpenStreams() == 1 {
		sc.setConnState(ConnStateActive)
	}

	return st
}

func (sc *serverConn) newWriterAndRequest(st *stream, f *MetaHeadersFrame) (*responseWriter, *ServerRequest, error) {
	sc.serveG.check()

	rp := httpcommon.ServerRequestParam{
		Method:    f.PseudoValue("method"),
		Scheme:    f.PseudoValue("scheme"),
		Authority: f.PseudoValue("authority"),
		Path:      f.PseudoValue("path"),
		Protocol:  f.PseudoValue("protocol"),
	}

	// extended connect is disabled, so we should not see :protocol
	if disableExtendedConnectProtocol && rp.Protocol != "" {
		return nil, nil, sc.countError("bad_connect", streamError(f.StreamID, ErrCodeProtocol))
	}

	isConnect := rp.Method == "CONNECT"
	if isConnect {
		if rp.Protocol == "" && (rp.Path != "" || rp.Scheme != "" || rp.Authority == "") {
			return nil, nil, sc.countError("bad_connect", streamError(f.StreamID, ErrCodeProtocol))
		}
	} else if rp.Method == "" || rp.Path == "" || (rp.Scheme != "https" && rp.Scheme != "http") {
		// See 8.1.2.6 Malformed Requests and Responses:
		//
		// Malformed requests or responses that are detected
		// MUST be treated as a stream error (Section 5.4.2)
		// of type PROTOCOL_ERROR."
		//
		// 8.1.2.3 Request Pseudo-Header Fields
		// "All HTTP/2 requests MUST include exactly one valid
		// value for the :method, :scheme, and :path
		// pseudo-header fields"
		return nil, nil, sc.countError("bad_path_method", streamError(f.StreamID, ErrCodeProtocol))
	}

	header := make(Header)
	rp.Header = header
	for _, hf := range f.RegularFields() {
		header.Add(sc.canonicalHeader(hf.Name), hf.Value)
	}
	if rp.Authority == "" {
		rp.Authority = header.Get("Host")
	}
	if rp.Protocol != "" {
		header.Set(":protocol", rp.Protocol)
	}

	rw, req, err := sc.newWriterAndRequestNoBody(st, rp)
	if err != nil {
		return nil, nil, err
	}
	bodyOpen := !f.StreamEnded()
	if vv, ok := rp.Header["Content-Length"]; ok {
		if cl, err := strconv.ParseUint(vv[0], 10, 63); err == nil {
			req.ContentLength = int64(cl)
		} else {
			req.ContentLength = 0
		}
		if !bodyOpen && req.ContentLength != 0 {
			return nil, nil, sc.countError("bodyless_content_length", streamError(f.StreamID, ErrCodeProtocol))
		}
	}
	if bodyOpen {
		if _, ok := rp.Header["Content-Length"]; !ok {
			req.ContentLength = -1
		}
		req.Body.(*requestBody).pipe = &pipe{
			b: &dataBuffer{expected: req.ContentLength},
		}
	}
	return rw, req, nil
}

func (sc *serverConn) newWriterAndRequestNoBody(st *stream, rp httpcommon.ServerRequestParam) (*responseWriter, *ServerRequest, error) {
	sc.serveG.check()

	var tlsState *tls.ConnectionState // nil if not scheme https
	if rp.Scheme == "https" {
		tlsState = sc.tlsState
	}

	res := httpcommon.NewServerRequest(rp)
	if res.InvalidReason != "" {
		return nil, nil, sc.countError(res.InvalidReason, streamError(st.id, ErrCodeProtocol))
	}

	body := &requestBody{
		conn:          sc,
		stream:        st,
		needsContinue: res.NeedsContinue,
	}
	rw := sc.newResponseWriter(st)
	rw.rws.req = ServerRequest{
		Context:    st.ctx,
		Method:     rp.Method,
		URL:        res.URL,
		RemoteAddr: sc.remoteAddrStr,
		Header:     rp.Header,
		RequestURI: res.RequestURI,
		Proto:      "HTTP/2.0",
		ProtoMajor: 2,
		ProtoMinor: 0,
		TLS:        tlsState,
		Host:       rp.Authority,
		Body:       body,
		Trailer:    res.Trailer,
	}
	return rw, &rw.rws.req, nil
}

func (sc *serverConn) newResponseWriter(st *stream) *responseWriter {
	rws := responseWriterStatePool.Get().(*responseWriterState)
	bwSave := rws.bw
	*rws = responseWriterState{} // zero all the fields
	rws.conn = sc
	rws.bw = bwSave
	rws.bw.Reset(chunkWriter{rws})
	rws.stream = st
	return &responseWriter{rws: rws}
}

type unstartedHandler struct {
	streamID uint32
	rw       *responseWriter
	req      *ServerRequest
	handler  func(*ResponseWriter, *ServerRequest)
}

// scheduleHandler starts a handler goroutine,
// or schedules one to start as soon as an existing handler finishes.
func (sc *serverConn) scheduleHandler(streamID uint32, rw *responseWriter, req *ServerRequest, handler func(*ResponseWriter, *ServerRequest)) error {
	sc.serveG.check()
	maxHandlers := sc.advMaxStreams
	if sc.curHandlers < maxHandlers {
		sc.curHandlers++
		go sc.runHandler(rw, req, handler)
		return nil
	}
	if len(sc.unstartedHandlers) > int(4*sc.advMaxStreams) {
		return sc.countError("too_many_early_resets", ConnectionError(ErrCodeEnhanceYourCalm))
	}
	sc.unstartedHandlers = append(sc.unstartedHandlers, unstartedHandler{
		streamID: streamID,
		rw:       rw,
		req:      req,
		handler:  handler,
	})
	return nil
}

func (sc *serverConn) handlerDone() {
	sc.serveG.check()
	sc.curHandlers--
	i := 0
	maxHandlers := sc.advMaxStreams
	for ; i < len(sc.unstartedHandlers); i++ {
		u := sc.unstartedHandlers[i]
		if sc.streams[u.streamID] == nil {
			// This stream was reset before its goroutine had a chance to start.
			continue
		}
		if sc.curHandlers >= maxHandlers {
			break
		}
		sc.curHandlers++
		go sc.runHandler(u.rw, u.req, u.handler)
		sc.unstartedHandlers[i] = unstartedHandler{} // don't retain references
	}
	sc.unstartedHandlers = sc.unstartedHandlers[i:]
	if len(sc.unstartedHandlers) == 0 {
		sc.unstartedHandlers = nil
	}
}

// Run on its own goroutine.
func (sc *serverConn) runHandler(rw *responseWriter, req *ServerRequest, handler func(*ResponseWriter, *ServerRequest)) {
	defer sc.sendServeMsg(handlerDoneMsg)
	didPanic := true
	defer func() {
		rw.rws.stream.cancelCtx()
		if req.MultipartForm != nil {
			req.MultipartForm.RemoveAll()
		}
		if didPanic {
			e := recover()
			sc.writeFrameFromHandler(FrameWriteRequest{
				write:  handlerPanicRST{rw.rws.stream.id},
				stream: rw.rws.stream,
			})
			// Same as net/http:
			if e != nil && e != ErrAbortHandler {
				const size = 64 << 10
				buf := make([]byte, size)
				buf = buf[:runtime.Stack(buf, false)]
				sc.logf("http2: panic serving %v: %v\n%s", sc.conn.RemoteAddr(), e, buf)
			}
			return
		}
		rw.handlerDone()
	}()
	handler(rw, req)
	didPanic = false
}

func handleHeaderListTooLong(w *ResponseWriter, r *ServerRequest) {
	// 10.5.1 Limits on Header Block Size:
	// .. "A server that receives a larger header block than it is
	// willing to handle can send an HTTP 431 (Request Header Fields Too
	// Large) status code"
	const statusRequestHeaderFieldsTooLarge = 431 // only in Go 1.6+
	w.WriteHeader(statusRequestHeaderFieldsTooLarge)
	io.WriteString(w, "<h1>HTTP Error 431</h1><p>Request Header Field(s) Too Large</p>")
}

// called from handler goroutines.
// h may be nil.
func (sc *serverConn) writeHeaders(st *stream, headerData *writeResHeaders) error {
	sc.serveG.checkNotOn() // NOT on
	var errc chan error
	if headerData.h != nil {
		// If there's a header map (which we don't own), so we have to block on
		// waiting for this frame to be written, so an http.Flush mid-handler
		// writes out the correct value of keys, before a handler later potentially
		// mutates it.
		errc = sc.srv.getErrChan()
	}
	if err := sc.writeFrameFromHandler(FrameWriteRequest{
		write:  headerData,
		stream: st,
		done:   errc,
	}); err != nil {
		return err
	}
	if errc != nil {
		select {
		case err := <-errc:
			sc.srv.putErrChan(errc)
			return err
		case <-sc.doneServing:
			return errClientDisconnected
		case <-st.cw:
			return errStreamClosed
		}
	}
	return nil
}

// called from handler goroutines.
func (sc *serverConn) write100ContinueHeaders(st *stream) {
	sc.writeFrameFromHandler(FrameWriteRequest{
		write:  write100ContinueHeadersFrame{st.id},
		stream: st,
	})
}

// A bodyReadMsg tells the server loop that the http.Handler read n
// bytes of the DATA from the client on the given stream.
type bodyReadMsg struct {
	st *stream
	n  int
}

// called from handler goroutines.
// Notes that the handler for the given stream ID read n bytes of its body
// and schedules flow control tokens to be sent.
func (sc *serverConn) noteBodyReadFromHandler(st *stream, n int, err error) {
	sc.serveG.checkNotOn() // NOT on
	if n > 0 {
		select {
		case sc.bodyReadCh <- bodyReadMsg{st, n}:
		case <-sc.doneServing:
		}
	}
}

func (sc *serverConn) noteBodyRead(st *stream, n int) {
	sc.serveG.check()
	sc.sendWindowUpdate(nil, n) // conn-level
	if st.state != stateHalfClosedRemote && st.state != stateClosed {
		// Don't send this WINDOW_UPDATE if the stream is closed
		// remotely.
		sc.sendWindowUpdate(st, n)
	}
}

// st may be nil for conn-level
func (sc *serverConn) sendWindowUpdate32(st *stream, n int32) {
	sc.sendWindowUpdate(st, int(n))
}

// st may be nil for conn-level
func (sc *serverConn) sendWindowUpdate(st *stream, n int) {
	sc.serveG.check()
	var streamID uint32
	var send int32
	if st == nil {
		send = sc.inflow.add(n)
	} else {
		streamID = st.id
		send = st.inflow.add(n)
	}
	if send == 0 {
		return
	}
	sc.writeFrame(FrameWriteRequest{
		write:  writeWindowUpdate{streamID: streamID, n: uint32(send)},
		stream: st,
	})
}

// requestBody is the Handler's Request.Body type.
// Read and Close may be called concurrently.
type requestBody struct {
	_             incomparable
	stream        *stream
	conn          *serverConn
	closeOnce     sync.Once // for use by Close only
	sawEOF        bool      // for use by Read only
	pipe          *pipe     // non-nil if we have an HTTP entity message body
	needsContinue bool      // need to send a 100-continue
}

func (b *requestBody) Close() error {
	b.closeOnce.Do(func() {
		if b.pipe != nil {
			b.pipe.BreakWithError(errClosedBody)
		}
	})
	return nil
}

func (b *requestBody) Read(p []byte) (n int, err error) {
	if b.needsContinue {
		b.needsContinue = false
		b.conn.write100ContinueHeaders(b.stream)
	}
	if b.pipe == nil || b.sawEOF {
		return 0, io.EOF
	}
	n, err = b.pipe.Read(p)
	if err == io.EOF {
		b.sawEOF = true
	}
	if b.conn == nil {
		return
	}
	b.conn.noteBodyReadFromHandler(b.stream, n, err)
	return
}

// responseWriter is the http.ResponseWriter implementation. It's
// intentionally small (1 pointer wide) to minimize garbage. The
// responseWriterState pointer inside is zeroed at the end of a
// request (in handlerDone) and calls on the responseWriter thereafter
// simply crash (caller's mistake), but the much larger responseWriterState
// and buffers are reused between multiple requests.
type responseWriter struct {
	rws *responseWriterState
}

type responseWriterState struct {
	// immutable within a request:
	stream *stream
	req    ServerRequest
	conn   *serverConn

	// TODO: adjust buffer writing sizes based on server config, frame size updates from peer, etc
	bw *bufio.Writer // writing to a chunkWriter{this *responseWriterState}

	// mutated by http.Handler goroutine:
	handlerHeader Header   // nil until called
	snapHeader    Header   // snapshot of handlerHeader at WriteHeader time
	trailers      []string // set in writeChunk
	status        int      // status code passed to WriteHeader
	wroteHeader   bool     // WriteHeader called (explicitly or implicitly). Not necessarily sent to user yet.
	sentHeader    bool     // have we sent the header frame?
	handlerDone   bool     // handler has finished

	sentContentLen int64 // non-zero if handler set a Content-Length header
	wroteBytes     int64

	closeNotifierMu sync.Mutex // guards closeNotifierCh
	closeNotifierCh chan bool  // nil until first used
}

type chunkWriter struct{ rws *responseWriterState }

func (cw chunkWriter) Write(p []byte) (n int, err error) {
	n, err = cw.rws.writeChunk(p)
	if err == errStreamClosed {
		// If writing failed because the stream has been closed,
		// return the reason it was closed.
		err = cw.rws.stream.closeErr
	}
	return n, err
}

func (rws *responseWriterState) hasTrailers() bool { return len(rws.trailers) > 0 }

func (rws *responseWriterState) hasNonemptyTrailers() bool {
	for _, trailer := range rws.trailers {
		if _, ok := rws.handlerHeader[trailer]; ok {
			return true
		}
	}
	return false
}

// declareTrailer is called for each Trailer header when the
// response header is written. It notes that a header will need to be
// written in the trailers at the end of the response.
func (rws *responseWriterState) declareTrailer(k string) {
	k = textproto.CanonicalMIMEHeaderKey(k)
	if !httpguts.ValidTrailerHeader(k) {
		// Forbidden by RFC 7230, section 4.1.2.
		rws.conn.logf("ignoring invalid trailer %q", k)
		return
	}
	if !slices.Contains(rws.trailers, k) {
		rws.trailers = append(rws.trailers, k)
	}
}

const TimeFormat = "Mon, 02 Jan 2006 15:04:05 GMT" // keep in sync with net/http

// writeChunk writes chunks from the bufio.Writer. But because
// bufio.Writer may bypass its chunking, sometimes p may be
// arbitrarily large.
//
// writeChunk is also responsible (on the first chunk) for sending the
// HEADER response.
func (rws *responseWriterState) writeChunk(p []byte) (n int, err error) {
	if !rws.wroteHeader {
		rws.writeHeader(200)
	}

	if rws.handlerDone {
		rws.promoteUndeclaredTrailers()
	}

	isHeadResp := rws.req.Method == "HEAD"
	if !rws.sentHeader {
		rws.sentHeader = true
		var ctype, clen string
		if clen = rws.snapHeader.Get("Content-Length"); clen != "" {
			rws.snapHeader.Del("Content-Length")
			if cl, err := strconv.ParseUint(clen, 10, 63); err == nil {
				rws.sentContentLen = int64(cl)
			} else {
				clen = ""
			}
		}
		_, hasContentLength := rws.snapHeader["Content-Length"]
		if !hasContentLength && clen == "" && rws.handlerDone && bodyAllowedForStatus(rws.status) && (len(p) > 0 || !isHeadResp) {
			clen = strconv.Itoa(len(p))
		}
		_, hasContentType := rws.snapHeader["Content-Type"]
		// If the Content-Encoding is non-blank, we shouldn't
		// sniff the body. See Issue golang.org/issue/31753.
		ce := rws.snapHeader.Get("Content-Encoding")
		hasCE := len(ce) > 0
		if !hasCE && !hasContentType && bodyAllowedForStatus(rws.status) && len(p) > 0 {
			ctype = internal.DetectContentType(p)
		}
		var date string
		if _, ok := rws.snapHeader["Date"]; !ok {
			// TODO(bradfitz): be faster here, like net/http? measure.
			date = time.Now().UTC().Format(TimeFormat)
		}

		for _, v := range rws.snapHeader["Trailer"] {
			foreachHeaderElement(v, rws.declareTrailer)
		}

		// "Connection" headers aren't allowed in HTTP/2 (RFC 7540, 8.1.2.2),
		// but respect "Connection" == "close" to mean sending a GOAWAY and tearing
		// down the TCP connection when idle, like we do for HTTP/1.
		// TODO: remove more Connection-specific header fields here, in addition
		// to "Connection".
		if _, ok := rws.snapHeader["Connection"]; ok {
			v := rws.snapHeader.Get("Connection")
			delete(rws.snapHeader, "Connection")
			if v == "close" {
				rws.conn.startGracefulShutdown()
			}
		}

		endStream := (rws.handlerDone && !rws.hasTrailers() && len(p) == 0) || isHeadResp
		err = rws.conn.writeHeaders(rws.stream, &writeResHeaders{
			streamID:      rws.stream.id,
			httpResCode:   rws.status,
			h:             rws.snapHeader,
			endStream:     endStream,
			contentType:   ctype,
			contentLength: clen,
			date:          date,
		})
		if err != nil {
			return 0, err
		}
		if endStream {
			return 0, nil
		}
	}
	if isHeadResp {
		return len(p), nil
	}
	if len(p) == 0 && !rws.handlerDone {
		return 0, nil
	}

	// only send trailers if they have actually been defined by the
	// server handler.
	hasNonemptyTrailers := rws.hasNonemptyTrailers()
	endStream := rws.handlerDone && !hasNonemptyTrailers
	if len(p) > 0 || endStream {
		// only send a 0 byte DATA frame if we're ending the stream.
		if err := rws.conn.writeDataFromHandler(rws.stream, p, endStream); err != nil {
			return 0, err
		}
	}

	if rws.handlerDone && hasNonemptyTrailers {
		err = rws.conn.writeHeaders(rws.stream, &writeResHeaders{
			streamID:  rws.stream.id,
			h:         rws.handlerHeader,
			trailers:  rws.trailers,
			endStream: true,
		})
		return len(p), err
	}
	return len(p), nil
}

// TrailerPrefix is a magic prefix for ResponseWriter.Header map keys
// that, if present, signals that the map entry is actually for
// the response trailers, and not the response headers. The prefix
// is stripped after the ServeHTTP call finishes and the values are
// sent in the trailers.
//
// This mechanism is intended only for trailers that are not known
// prior to the headers being written. If the set of trailers is fixed
// or known before the header is written, the normal Go trailers mechanism
// is preferred:
//
//	https://golang.org/pkg/net/http/#ResponseWriter
//	https://golang.org/pkg/net/http/#example_ResponseWriter_trailers
const TrailerPrefix = "Trailer:"

// promoteUndeclaredTrailers permits http.Handlers to set trailers
// after the header has already been flushed. Because the Go
// ResponseWriter interface has no way to set Trailers (only the
// Header), and because we didn't want to expand the ResponseWriter
// interface, and because nobody used trailers, and because RFC 7230
// says you SHOULD (but not must) predeclare any trailers in the
// header, the official ResponseWriter rules said trailers in Go must
// be predeclared, and then we reuse the same ResponseWriter.Header()
// map to mean both Headers and Trailers. When it's time to write the
// Trailers, we pick out the fields of Headers that were declared as
// trailers. That worked for a while, until we found the first major
// user of Trailers in the wild: gRPC (using them only over http2),
// and gRPC libraries permit setting trailers mid-stream without
// predeclaring them. So: change of plans. We still permit the old
// way, but we also permit this hack: if a Header() key begins with
// "Trailer:", the suffix of that key is a Trailer. Because ':' is an
// invalid token byte anyway, there is no ambiguity. (And it's already
// filtered out) It's mildly hacky, but not terrible.
//
// This method runs after the Handler is done and promotes any Header
// fields to be trailers.
func (rws *responseWriterState) promoteUndeclaredTrailers() {
	for k, vv := range rws.handlerHeader {
		if !strings.HasPrefix(k, TrailerPrefix) {
			continue
		}
		trailerKey := strings.TrimPrefix(k, TrailerPrefix)
		rws.declareTrailer(trailerKey)
		rws.handlerHeader[textproto.CanonicalMIMEHeaderKey(trailerKey)] = vv
	}

	if len(rws.trailers) > 1 {
		slices.Sort(rws.trailers)
	}
}

func (w *responseWriter) SetReadDeadline(deadline time.Time) error {
	st := w.rws.stream
	if !deadline.IsZero() && deadline.Before(time.Now()) {
		// If we're setting a deadline in the past, reset the stream immediately
		// so writes after SetWriteDeadline returns will fail.
		st.onReadTimeout()
		return nil
	}
	w.rws.conn.sendServeMsg(func(sc *serverConn) {
		if st.readDeadline != nil {
			if !st.readDeadline.Stop() {
				// Deadline already exceeded, or stream has been closed.
				return
			}
		}
		if deadline.IsZero() {
			st.readDeadline = nil
		} else if st.readDeadline == nil {
			st.readDeadline = time.AfterFunc(deadline.Sub(time.Now()), st.onReadTimeout)
		} else {
			st.readDeadline.Reset(deadline.Sub(time.Now()))
		}
	})
	return nil
}

func (w *responseWriter) SetWriteDeadline(deadline time.Time) error {
	st := w.rws.stream
	if !deadline.IsZero() && deadline.Before(time.Now()) {
		// If we're setting a deadline in the past, reset the stream immediately
		// so writes after SetWriteDeadline returns will fail.
		st.onWriteTimeout()
		return nil
	}
	w.rws.conn.sendServeMsg(func(sc *serverConn) {
		if st.writeDeadline != nil {
			if !st.writeDeadline.Stop() {
				// Deadline already exceeded, or stream has been closed.
				return
			}
		}
		if deadline.IsZero() {
			st.writeDeadline = nil
		} else if st.writeDeadline == nil {
			st.writeDeadline = time.AfterFunc(deadline.Sub(time.Now()), st.onWriteTimeout)
		} else {
			st.writeDeadline.Reset(deadline.Sub(time.Now()))
		}
	})
	return nil
}

func (w *responseWriter) EnableFullDuplex() error {
	// We always support full duplex responses, so this is a no-op.
	return nil
}

func (w *responseWriter) Flush() {
	w.FlushError()
}

func (w *responseWriter) FlushError() error {
	rws := w.rws
	if rws == nil {
		panic("Header called after Handler finished")
	}
	var err error
	if rws.bw.Buffered() > 0 {
		err = rws.bw.Flush()
	} else {
		// The bufio.Writer won't call chunkWriter.Write
		// (writeChunk with zero bytes), so we have to do it
		// ourselves to force the HTTP response header and/or
		// final DATA frame (with END_STREAM) to be sent.
		_, err = chunkWriter{rws}.Write(nil)
		if err == nil {
			select {
			case <-rws.stream.cw:
				err = rws.stream.closeErr
			default:
			}
		}
	}
	return err
}

func (w *responseWriter) CloseNotify() <-chan bool {
	rws := w.rws
	if rws == nil {
		panic("CloseNotify called after Handler finished")
	}
	rws.closeNotifierMu.Lock()
	ch := rws.closeNotifierCh
	if ch == nil {
		ch = make(chan bool, 1)
		rws.closeNotifierCh = ch
		cw := rws.stream.cw
		go func() {
			cw.Wait() // wait for close
			ch <- true
		}()
	}
	rws.closeNotifierMu.Unlock()
	return ch
}

func (w *responseWriter) Header() Header {
	rws := w.rws
	if rws == nil {
		panic("Header called after Handler finished")
	}
	if rws.handlerHeader == nil {
		rws.handlerHeader = make(Header)
	}
	return rws.handlerHeader
}

// checkWriteHeaderCode is a copy of net/http's checkWriteHeaderCode.
func checkWriteHeaderCode(code int) {
	// Issue 22880: require valid WriteHeader status codes.
	// For now we only enforce that it's three digits.
	// In the future we might block things over 599 (600 and above aren't defined
	// at http://httpwg.org/specs/rfc7231.html#status.codes).
	// But for now any three digits.
	//
	// We used to send "HTTP/1.1 000 0" on the wire in responses but there's
	// no equivalent bogus thing we can realistically send in HTTP/2,
	// so we'll consistently panic instead and help people find their bugs
	// early. (We can't return an error from WriteHeader even if we wanted to.)
	if code < 100 || code > 999 {
		panic(fmt.Sprintf("invalid WriteHeader code %v", code))
	}
}

func (w *responseWriter) WriteHeader(code int) {
	rws := w.rws
	if rws == nil {
		panic("WriteHeader called after Handler finished")
	}
	rws.writeHeader(code)
}

func (rws *responseWriterState) writeHeader(code int) {
	if rws.wroteHeader {
		return
	}

	checkWriteHeaderCode(code)

	// Handle informational headers
	if code >= 100 && code <= 199 {
		// Per RFC 8297 we must not clear the current header map
		h := rws.handlerHeader

		_, cl := h["Content-Length"]
		_, te := h["Transfer-Encoding"]
		if cl || te {
			h = cloneHeader(h)
			h.Del("Content-Length")
			h.Del("Transfer-Encoding")
		}

		rws.conn.writeHeaders(rws.stream, &writeResHeaders{
			streamID:    rws.stream.id,
			httpResCode: code,
			h:           h,
			endStream:   rws.handlerDone && !rws.hasTrailers(),
		})

		return
	}

	rws.wroteHeader = true
	rws.status = code
	if len(rws.handlerHeader) > 0 {
		rws.snapHeader = cloneHeader(rws.handlerHeader)
	}
}

func cloneHeader(h Header) Header {
	h2 := make(Header, len(h))
	for k, vv := range h {
		vv2 := make([]string, len(vv))
		copy(vv2, vv)
		h2[k] = vv2
	}
	return h2
}

// The Life Of A Write is like this:
//
// * Handler calls w.Write or w.WriteString ->
// * -> rws.bw (*bufio.Writer) ->
// * (Handler might call Flush)
// * -> chunkWriter{rws}
// * -> responseWriterState.writeChunk(p []byte)
// * -> responseWriterState.writeChunk (most of the magic; see comment there)
func (w *responseWriter) Write(p []byte) (n int, err error) {
	return w.write(len(p), p, "")
}

func (w *responseWriter) WriteString(s string) (n int, err error) {
	return w.write(len(s), nil, s)
}

// either dataB or dataS is non-zero.
func (w *responseWriter) write(lenData int, dataB []byte, dataS string) (n int, err error) {
	rws := w.rws
	if rws == nil {
		panic("Write called after Handler finished")
	}
	if !rws.wroteHeader {
		w.WriteHeader(200)
	}
	if !bodyAllowedForStatus(rws.status) {
		return 0, ErrBodyNotAllowed
	}
	rws.wroteBytes += int64(len(dataB)) + int64(len(dataS)) // only one can be set
	if rws.sentContentLen != 0 && rws.wroteBytes > rws.sentContentLen {
		// TODO: send a RST_STREAM
		return 0, errors.New("http2: handler wrote more than declared Content-Length")
	}

	if dataB != nil {
		return rws.bw.Write(dataB)
	} else {
		return rws.bw.WriteString(dataS)
	}
}

func (w *responseWriter) handlerDone() {
	rws := w.rws
	rws.handlerDone = true
	w.Flush()
	w.rws = nil
	responseWriterStatePool.Put(rws)
}

// Push errors.
var (
	ErrRecursivePush    = errors.New("http2: recursive push not allowed")
	ErrPushLimitReached = errors.New("http2: push would exceed peer's SETTINGS_MAX_CONCURRENT_STREAMS")
)

func (w *responseWriter) Push(target, method string, header Header) error {
	st := w.rws.stream
	sc := st.sc
	sc.serveG.checkNotOn()

	// No recursive pushes: "PUSH_PROMISE frames MUST only be sent on a peer-initiated stream."
	// http://tools.ietf.org/html/rfc7540#section-6.6
	if st.isPushed() {
		return ErrRecursivePush
	}

	// Default options.
	if method == "" {
		method = "GET"
	}
	if header == nil {
		header = Header{}
	}
	wantScheme := "http"
	if w.rws.req.TLS != nil {
		wantScheme = "https"
	}

	// Validate the request.
	u, err := url.Parse(target)
	if err != nil {
		return err
	}
	if u.Scheme == "" {
		if !strings.HasPrefix(target, "/") {
			return fmt.Errorf("target must be an absolute URL or an absolute path: %q", target)
		}
		u.Scheme = wantScheme
		u.Host = w.rws.req.Host
	} else {
		if u.Scheme != wantScheme {
			return fmt.Errorf("cannot push URL with scheme %q from request with scheme %q", u.Scheme, wantScheme)
		}
		if u.Host == "" {
			return errors.New("URL must have a host")
		}
	}
	for k := range header {
		if strings.HasPrefix(k, ":") {
			return fmt.Errorf("promised request headers cannot include pseudo header %q", k)
		}
		// These headers are meaningful only if the request has a body,
		// but PUSH_PROMISE requests cannot have a body.
		// http://tools.ietf.org/html/rfc7540#section-8.2
		// Also disallow Host, since the promised URL must be absolute.
		if asciiEqualFold(k, "content-length") ||
			asciiEqualFold(k, "content-encoding") ||
			asciiEqualFold(k, "trailer") ||
			asciiEqualFold(k, "te") ||
			asciiEqualFold(k, "expect") ||
			asciiEqualFold(k, "host") {
			return fmt.Errorf("promised request headers cannot include %q", k)
		}
	}
	if err := checkValidHTTP2RequestHeaders(header); err != nil {
		return err
	}

	// The RFC effectively limits promised requests to GET and HEAD:
	// "Promised requests MUST be cacheable [GET, HEAD, or POST], and MUST be safe [GET or HEAD]"
	// http://tools.ietf.org/html/rfc7540#section-8.2
	if method != "GET" && method != "HEAD" {
		return fmt.Errorf("method %q must be GET or HEAD", method)
	}

	msg := &startPushRequest{
		parent: st,
		method: method,
		url:    u,
		header: cloneHeader(header),
		done:   sc.srv.getErrChan(),
	}

	select {
	case <-sc.doneServing:
		return errClientDisconnected
	case <-st.cw:
		return errStreamClosed
	case sc.serveMsgCh <- msg:
	}

	select {
	case <-sc.doneServing:
		return errClientDisconnected
	case <-st.cw:
		return errStreamClosed
	case err := <-msg.done:
		sc.srv.putErrChan(msg.done)
		return err
	}
}

type startPushRequest struct {
	parent *stream
	method string
	url    *url.URL
	header Header
	done   chan error
}

func (sc *serverConn) startPush(msg *startPushRequest) {
	sc.serveG.check()

	// http://tools.ietf.org/html/rfc7540#section-6.6.
	// PUSH_PROMISE frames MUST only be sent on a peer-initiated stream that
	// is in either the "open" or "half-closed (remote)" state.
	if msg.parent.state != stateOpen && msg.parent.state != stateHalfClosedRemote {
		// responseWriter.Push checks that the stream is peer-initiated.
		msg.done <- errStreamClosed
		return
	}

	// http://tools.ietf.org/html/rfc7540#section-6.6.
	if !sc.pushEnabled {
		msg.done <- ErrNotSupported
		return
	}

	// PUSH_PROMISE frames must be sent in increasing order by stream ID, so
	// we allocate an ID for the promised stream lazily, when the PUSH_PROMISE
	// is written. Once the ID is allocated, we start the request handler.
	allocatePromisedID := func() (uint32, error) {
		sc.serveG.check()

		// Check this again, just in case. Technically, we might have received
		// an updated SETTINGS by the time we got around to writing this frame.
		if !sc.pushEnabled {
			return 0, ErrNotSupported
		}
		// http://tools.ietf.org/html/rfc7540#section-6.5.2.
		if sc.curPushedStreams+1 > sc.clientMaxStreams {
			return 0, ErrPushLimitReached
		}

		// http://tools.ietf.org/html/rfc7540#section-5.1.1.
		// Streams initiated by the server MUST use even-numbered identifiers.
		// A server that is unable to establish a new stream identifier can send a GOAWAY
		// frame so that the client is forced to open a new connection for new streams.
		if sc.maxPushPromiseID+2 >= 1<<31 {
			sc.startGracefulShutdownInternal()
			return 0, ErrPushLimitReached
		}
		sc.maxPushPromiseID += 2
		promisedID := sc.maxPushPromiseID

		// http://tools.ietf.org/html/rfc7540#section-8.2.
		// Strictly speaking, the new stream should start in "reserved (local)", then
		// transition to "half closed (remote)" after sending the initial HEADERS, but
		// we start in "half closed (remote)" for simplicity.
		// See further comments at the definition of stateHalfClosedRemote.
		promised := sc.newStream(promisedID, msg.parent.id, stateHalfClosedRemote, defaultRFC9218Priority(sc.priorityAware && !sc.hasIntermediary))
		rw, req, err := sc.newWriterAndRequestNoBody(promised, httpcommon.ServerRequestParam{
			Method:    msg.method,
			Scheme:    msg.url.Scheme,
			Authority: msg.url.Host,
			Path:      msg.url.RequestURI(),
			Header:    cloneHeader(msg.header), // clone since handler runs concurrently with writing the PUSH_PROMISE
		})
		if err != nil {
			// Should not happen, since we've already validated msg.url.
			panic(fmt.Sprintf("newWriterAndRequestNoBody(%+v): %v", msg.url, err))
		}

		sc.curHandlers++
		go sc.runHandler(rw, req, sc.handler.ServeHTTP)
		return promisedID, nil
	}

	sc.writeFrame(FrameWriteRequest{
		write: &writePushPromise{
			streamID:           msg.parent.id,
			method:             msg.method,
			url:                msg.url,
			h:                  msg.header,
			allocatePromisedID: allocatePromisedID,
		},
		stream: msg.parent,
		done:   msg.done,
	})
}

// foreachHeaderElement splits v according to the "#rule" construction
// in RFC 7230 section 7 and calls fn for each non-empty element.
func foreachHeaderElement(v string, fn func(string)) {
	v = textproto.TrimString(v)
	if v == "" {
		return
	}
	if !strings.Contains(v, ",") {
		fn(v)
		return
	}
	for f := range strings.SplitSeq(v, ",") {
		if f = textproto.TrimString(f); f != "" {
			fn(f)
		}
	}
}

// From http://httpwg.org/specs/rfc7540.html#rfc.section.8.1.2.2
var connHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Connection",
	"Transfer-Encoding",
	"Upgrade",
}

// checkValidHTTP2RequestHeaders checks whether h is a valid HTTP/2 request,
// per RFC 7540 Section 8.1.2.2.
// The returned error is reported to users.
func checkValidHTTP2RequestHeaders(h Header) error {
	for _, k := range connHeaders {
		if _, ok := h[k]; ok {
			return fmt.Errorf("request header %q is not valid in HTTP/2", k)
		}
	}
	te := h["Te"]
	if len(te) > 0 && (len(te) > 1 || (te[0] != "trailers" && te[0] != "")) {
		return errors.New(`request header "TE" may only be "trailers" in HTTP/2`)
	}
	return nil
}

type serve400Handler struct {
	err error
}

func (handler serve400Handler) ServeHTTP(w *ResponseWriter, r *ServerRequest) {
	const statusBadRequest = 400

	// TODO: Dedup with http.Error?
	h := w.Header()
	h.Del("Content-Length")
	h.Set("Content-Type", "text/plain; charset=utf-8")
	h.Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(statusBadRequest)
	fmt.Fprintln(w, handler.err.Error())
}

// h1ServerKeepAlivesDisabled reports whether hs has its keep-alives
// disabled. See comments on h1ServerShutdownChan above for why
// the code is written this way.
func h1ServerKeepAlivesDisabled(hs ServerConfig) bool {
	return !hs.DoKeepAlives()
}

func (sc *serverConn) countError(name string, err error) error {
	if sc == nil || sc.srv == nil {
		return err
	}
	f := sc.countErrorFunc
	if f == nil {
		return err
	}
	var typ string
	var code ErrCode
	switch e := err.(type) {
	case ConnectionError:
		typ = "conn"
		code = ErrCode(e)
	case StreamError:
		typ = "stream"
		code = ErrCode(e.Code)
	default:
		return err
	}
	codeStr := errCodeName[code]
	if codeStr == "" {
		codeStr = strconv.Itoa(int(code))
	}
	f(fmt.Sprintf("%s_%s_%s", typ, codeStr, name))
	return err
}
