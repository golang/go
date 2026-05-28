// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
	_ "unsafe" // for go:linkname

	"net/http/internal/http2"
	. "net/http/internal/http2"
	"net/http/internal/testcert"

	"golang.org/x/net/http2/hpack"
)

var stderrVerbose = flag.Bool("stderr_verbose", false, "Mirror verbosity to stderr, unbuffered")

func stderrv() io.Writer {
	if *stderrVerbose {
		return os.Stderr
	}

	return io.Discard
}

type safeBuffer struct {
	b bytes.Buffer
	m sync.Mutex
}

func (sb *safeBuffer) Write(d []byte) (int, error) {
	sb.m.Lock()
	defer sb.m.Unlock()
	return sb.b.Write(d)
}

func (sb *safeBuffer) Bytes() []byte {
	sb.m.Lock()
	defer sb.m.Unlock()
	return sb.b.Bytes()
}

func (sb *safeBuffer) Len() int {
	sb.m.Lock()
	defer sb.m.Unlock()
	return sb.b.Len()
}

type serverTester struct {
	cc           net.Conn // client conn
	t            testing.TB
	h1server     *http.Server
	h2server     *Server
	serverLogBuf safeBuffer // logger for httptest.Server
	logFilter    []string   // substrings to filter out
	scMu         sync.Mutex // guards sc
	sc           *ServerConn
	wrotePreface bool
	testConnFramer

	callsMu sync.Mutex
	calls   []*serverHandlerCall

	// If http2debug!=2, then we capture Frame debug logs that will be written
	// to t.Log after a test fails. The read and write logs use separate locks
	// and buffers so we don't accidentally introduce synchronization between
	// the read and write goroutines, which may hide data races.
	frameReadLogMu   sync.Mutex
	frameReadLogBuf  bytes.Buffer
	frameWriteLogMu  sync.Mutex
	frameWriteLogBuf bytes.Buffer

	// writing headers:
	headerBuf bytes.Buffer
	hpackEnc  *hpack.Encoder
}

type twriter struct {
	t  testing.TB
	st *serverTester // optional
}

func (w twriter) Write(p []byte) (n int, err error) {
	if w.st != nil {
		ps := string(p)
		for _, phrase := range w.st.logFilter {
			if strings.Contains(ps, phrase) {
				return len(p), nil // no logging
			}
		}
	}
	w.t.Logf("%s", p)
	return len(p), nil
}

func newTestServer(t testing.TB, handler http.HandlerFunc, opts ...any) *httptest.Server {
	t.Helper()
	if handler == nil {
		handler = func(w http.ResponseWriter, req *http.Request) {}
	}
	ts := httptest.NewUnstartedServer(handler)
	ts.EnableHTTP2 = true
	ts.Config.ErrorLog = log.New(twriter{t: t}, "", log.LstdFlags)
	ts.Config.Protocols = protocols("h2")
	for _, opt := range opts {
		switch v := opt.(type) {
		case func(*httptest.Server):
			v(ts)
		case func(*http.Server):
			v(ts.Config)
		case func(*http.HTTP2Config):
			if ts.Config.HTTP2 == nil {
				ts.Config.HTTP2 = &http.HTTP2Config{}
			}
			v(ts.Config.HTTP2)
		default:
			t.Fatalf("unknown newTestServer option type %T", v)
		}
	}

	if ts.Config.Protocols.HTTP2() {
		ts.TLS = testServerTLSConfig
		if ts.Config.TLSConfig != nil {
			ts.TLS = ts.Config.TLSConfig
		}
		ts.StartTLS()
	} else if ts.Config.Protocols.UnencryptedHTTP2() {
		ts.EnableHTTP2 = false // actually just disables HTTP/2 over TLS
		ts.Start()
	} else {
		t.Fatalf("Protocols contains neither HTTP2 nor UnencryptedHTTP2")
	}

	t.Cleanup(func() {
		ts.CloseClientConnections()
		ts.Close()
	})

	return ts
}

type serverTesterOpt string

var optFramerReuseFrames = serverTesterOpt("frame_reuse_frames")

var optQuiet = func(server *http.Server) {
	server.ErrorLog = log.New(io.Discard, "", 0)
}

func newServerTester(t testing.TB, handler http.HandlerFunc, opts ...any) *serverTester {
	t.Helper()

	h1server := &http.Server{}
	var tlsState *tls.ConnectionState
	for _, opt := range opts {
		switch v := opt.(type) {
		case func(*http.Server):
			v(h1server)
		case func(*http.HTTP2Config):
			if h1server.HTTP2 == nil {
				h1server.HTTP2 = &http.HTTP2Config{}
			}
			v(h1server.HTTP2)
		case func(*tls.ConnectionState):
			if tlsState == nil {
				tlsState = &tls.ConnectionState{
					Version:     tls.VersionTLS13,
					ServerName:  "go.dev",
					CipherSuite: tls.TLS_AES_128_GCM_SHA256,
				}
			}
			v(tlsState)
		default:
			t.Fatalf("unknown newServerTester option type %T", v)
		}
	}

	tlsConfig := h1server.TLSConfig
	if tlsConfig == nil {
		cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
		if err != nil {
			t.Fatal(err)
		}
		tlsConfig = &tls.Config{
			Certificates:       []tls.Certificate{cert},
			InsecureSkipVerify: true,
			NextProtos:         []string{"h2"},
		}
		h1server.TLSConfig = tlsConfig
	}

	var cli, srv net.Conn

	cliPipe, srvPipe := synctestNetPipe()

	if h1server.Protocols != nil && h1server.Protocols.UnencryptedHTTP2() {
		cli, srv = cliPipe, srvPipe
	} else {
		cli = tls.Client(cliPipe, &tls.Config{
			InsecureSkipVerify: true,
			NextProtos:         []string{"h2"},
		})
		srv = tls.Server(srvPipe, tlsConfig)
	}

	st := &serverTester{
		t:        t,
		cc:       cli,
		h1server: h1server,
	}
	st.hpackEnc = hpack.NewEncoder(&st.headerBuf)
	if h1server.ErrorLog == nil {
		h1server.ErrorLog = log.New(io.MultiWriter(stderrv(), twriter{t: t, st: st}, &st.serverLogBuf), "", log.LstdFlags)
	}

	if handler == nil {
		handler = serverTesterHandler{st}.ServeHTTP
	}
	h1server.Handler = handler

	t.Cleanup(func() {
		st.Close()
		time.Sleep(GoAwayTimeout) // give server time to shut down
	})

	connc := make(chan *ServerConn)
	h1server.ConnContext = func(ctx context.Context, conn net.Conn) context.Context {
		ctx = context.WithValue(ctx, NewConnContextKey, func(sc *ServerConn) {
			connc <- sc
		})
		if tlsState != nil {
			ctx = context.WithValue(ctx, ConnectionStateContextKey, func() tls.ConnectionState {
				return *tlsState
			})
		}
		return ctx
	}
	go func() {
		li := newOneConnListener(srv)
		t.Cleanup(func() {
			li.Close()
		})
		h1server.Serve(li)
	}()
	if cliTLS, ok := cli.(*tls.Conn); ok {
		if err := cliTLS.Handshake(); err != nil {
			t.Fatalf("client TLS handshake: %v", err)
		}
		cliTLS.SetReadDeadline(time.Now())
	} else {
		// Confusing but difficult to fix: Preface must be written
		// before the conn appears on connc.
		st.writePreface()
		st.wrotePreface = true
		cliPipe.SetReadDeadline(time.Now())
	}
	st.sc = <-connc

	st.fr = NewFramer(st.cc, st.cc)
	st.testConnFramer = testConnFramer{
		t:   t,
		fr:  NewFramer(cli, cli),
		dec: hpack.NewDecoder(InitialHeaderTableSize, nil),
	}
	synctest.Wait()
	return st
}

type netConnWithConnectionState struct {
	net.Conn
	state tls.ConnectionState
}

func (c *netConnWithConnectionState) ConnectionState() tls.ConnectionState {
	return c.state
}

func (c *netConnWithConnectionState) HandshakeContext() tls.ConnectionState {
	return c.state
}

type serverTesterHandler struct {
	st *serverTester
}

func (h serverTesterHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	call := &serverHandlerCall{
		w:   w,
		req: req,
		ch:  make(chan func()),
	}
	h.st.t.Cleanup(call.exit)
	h.st.callsMu.Lock()
	h.st.calls = append(h.st.calls, call)
	h.st.callsMu.Unlock()
	for f := range call.ch {
		f()
	}
}

// serverHandlerCall is a call to the server handler's ServeHTTP method.
type serverHandlerCall struct {
	w         http.ResponseWriter
	req       *http.Request
	closeOnce sync.Once
	ch        chan func()
}

// do executes f in the handler's goroutine.
func (call *serverHandlerCall) do(f func(http.ResponseWriter, *http.Request)) {
	donec := make(chan struct{})
	call.ch <- func() {
		defer close(donec)
		f(call.w, call.req)
	}
	<-donec
}

// exit causes the handler to return.
func (call *serverHandlerCall) exit() {
	call.closeOnce.Do(func() {
		close(call.ch)
	})
}

// sync waits for all goroutines to idle.
func (st *serverTester) sync() {
	synctest.Wait()
}

// advance advances synthetic time by a duration.
func (st *serverTester) advance(d time.Duration) {
	time.Sleep(d)
	synctest.Wait()
}

func (st *serverTester) authority() string {
	return "dummy.tld"
}

func (st *serverTester) addLogFilter(phrase string) {
	st.logFilter = append(st.logFilter, phrase)
}

func (st *serverTester) nextHandlerCall() *serverHandlerCall {
	st.t.Helper()
	synctest.Wait()
	st.callsMu.Lock()
	defer st.callsMu.Unlock()
	if len(st.calls) == 0 {
		st.t.Fatal("expected server handler call, got none")
	}
	call := st.calls[0]
	st.calls = st.calls[1:]
	return call
}

func (st *serverTester) streamExists(id uint32) bool {
	return st.sc.TestStreamExists(id)
}

func (st *serverTester) streamState(id uint32) StreamState {
	return st.sc.TestStreamState(id)
}

func (st *serverTester) Close() {
	if st.t.Failed() {
		st.frameReadLogMu.Lock()
		if st.frameReadLogBuf.Len() > 0 {
			st.t.Logf("Framer read log:\n%s", st.frameReadLogBuf.String())
		}
		st.frameReadLogMu.Unlock()

		st.frameWriteLogMu.Lock()
		if st.frameWriteLogBuf.Len() > 0 {
			st.t.Logf("Framer write log:\n%s", st.frameWriteLogBuf.String())
		}
		st.frameWriteLogMu.Unlock()

		// If we failed already (and are likely in a Fatal,
		// unwindowing), force close the connection, so the
		// httptest.Server doesn't wait forever for the conn
		// to close.
		if st.cc != nil {
			st.cc.Close()
		}
	}
	if st.cc != nil {
		st.cc.Close()
	}
	log.SetOutput(os.Stderr)
}

// greet initiates the client's HTTP/2 connection into a state where
// frames may be sent.
func (st *serverTester) greet() {
	st.t.Helper()
	st.greetAndCheckSettings(func(Setting) error { return nil })
}

func (st *serverTester) greetAndCheckSettings(checkSetting func(s Setting) error) {
	st.t.Helper()
	st.writePreface()
	st.writeSettings()
	st.sync()
	readFrame[*SettingsFrame](st.t, st).ForeachSetting(checkSetting)
	st.writeSettingsAck()

	// The initial WINDOW_UPDATE and SETTINGS ACK can come in any order.
	var gotSettingsAck bool
	var gotWindowUpdate bool

	for range 2 {
		f := st.readFrame()
		if f == nil {
			st.t.Fatal("wanted a settings ACK and window update, got none")
		}
		switch f := f.(type) {
		case *SettingsFrame:
			if !f.Header().Flags.Has(FlagSettingsAck) {
				st.t.Fatal("Settings Frame didn't have ACK set")
			}
			gotSettingsAck = true

		case *WindowUpdateFrame:
			if f.FrameHeader.StreamID != 0 {
				st.t.Fatalf("WindowUpdate StreamID = %d; want 0", f.FrameHeader.StreamID)
			}
			gotWindowUpdate = true

		default:
			st.t.Fatalf("Wanting a settings ACK or window update, received a %T", f)
		}
	}

	if !gotSettingsAck {
		st.t.Fatalf("Didn't get a settings ACK")
	}
	if !gotWindowUpdate {
		st.t.Fatalf("Didn't get a window update")
	}
}

func (st *serverTester) writePreface() {
	if st.wrotePreface {
		return
	}
	n, err := st.cc.Write([]byte(ClientPreface))
	if err != nil {
		st.t.Fatalf("Error writing client preface: %v", err)
	}
	if n != len(ClientPreface) {
		st.t.Fatalf("Writing client preface, wrote %d bytes; want %d", n, len(ClientPreface))
	}
}

func (st *serverTester) encodeHeaderField(k, v string) {
	err := st.hpackEnc.WriteField(hpack.HeaderField{Name: k, Value: v})
	if err != nil {
		st.t.Fatalf("HPACK encoding error for %q/%q: %v", k, v, err)
	}
}

// encodeHeaderRaw is the magic-free version of encodeHeader.
// It takes 0 or more (k, v) pairs and encodes them.
func (st *serverTester) encodeHeaderRaw(headers ...string) []byte {
	if len(headers)%2 == 1 {
		panic("odd number of kv args")
	}
	st.headerBuf.Reset()
	for len(headers) > 0 {
		k, v := headers[0], headers[1]
		st.encodeHeaderField(k, v)
		headers = headers[2:]
	}
	return st.headerBuf.Bytes()
}

// encodeHeader encodes headers and returns their HPACK bytes. headers
// must contain an even number of key/value pairs. There may be
// multiple pairs for keys (e.g. "cookie").  The :method, :path, and
// :scheme headers default to GET, / and https. The :authority header
// defaults to st.ts.Listener.Addr().
func (st *serverTester) encodeHeader(headers ...string) []byte {
	if len(headers)%2 == 1 {
		panic("odd number of kv args")
	}

	st.headerBuf.Reset()
	defaultAuthority := st.authority()

	if len(headers) == 0 {
		// Fast path, mostly for benchmarks, so test code doesn't pollute
		// profiles when we're looking to improve server allocations.
		st.encodeHeaderField(":method", "GET")
		st.encodeHeaderField(":scheme", "https")
		st.encodeHeaderField(":authority", defaultAuthority)
		st.encodeHeaderField(":path", "/")
		return st.headerBuf.Bytes()
	}

	if len(headers) == 2 && headers[0] == ":method" {
		// Another fast path for benchmarks.
		st.encodeHeaderField(":method", headers[1])
		st.encodeHeaderField(":scheme", "https")
		st.encodeHeaderField(":authority", defaultAuthority)
		st.encodeHeaderField(":path", "/")
		return st.headerBuf.Bytes()
	}

	pseudoCount := map[string]int{}
	keys := []string{":method", ":scheme", ":authority", ":path"}
	vals := map[string][]string{
		":method":    {"GET"},
		":scheme":    {"https"},
		":authority": {defaultAuthority},
		":path":      {"/"},
	}
	for len(headers) > 0 {
		k, v := headers[0], headers[1]
		headers = headers[2:]
		if _, ok := vals[k]; !ok {
			keys = append(keys, k)
		}
		if strings.HasPrefix(k, ":") {
			pseudoCount[k]++
			if pseudoCount[k] == 1 {
				vals[k] = []string{v}
			} else {
				// Allows testing of invalid headers w/ dup pseudo fields.
				vals[k] = append(vals[k], v)
			}
		} else {
			vals[k] = append(vals[k], v)
		}
	}
	for _, k := range keys {
		for _, v := range vals[k] {
			st.encodeHeaderField(k, v)
		}
	}
	return st.headerBuf.Bytes()
}

// bodylessReq1 writes a HEADERS frames with StreamID 1 and EndStream and EndHeaders set.
func (st *serverTester) bodylessReq1(headers ...string) {
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1, // clients send odd numbers
		BlockFragment: st.encodeHeader(headers...),
		EndStream:     true,
		EndHeaders:    true,
	})
}

func (st *serverTester) wantConnFlowControlConsumed(consumed int32) {
	if got, want := st.sc.TestFlowControlConsumed(), consumed; got != want {
		st.t.Errorf("connection flow control consumed: %v, want %v", got, want)
	}
}

func TestServer(t *testing.T) { synctestTest(t, testServer) }
func testServer(t testing.TB) {
	gotReq := make(chan bool, 1)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Foo", "Bar")
		gotReq <- true
	})
	defer st.Close()

	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1, // clients send odd numbers
		BlockFragment: st.encodeHeader(),
		EndStream:     true, // no DATA frames
		EndHeaders:    true,
	})

	<-gotReq
}

func TestServer_Request_Get(t *testing.T) { synctestTest(t, testServer_Request_Get) }
func testServer_Request_Get(t testing.TB) {
	testServerRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader("foo-bar", "some-value"),
			EndStream:     true, // no DATA frames
			EndHeaders:    true,
		})
	}, func(r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("Method = %q; want GET", r.Method)
		}
		if r.URL.Path != "/" {
			t.Errorf("URL.Path = %q; want /", r.URL.Path)
		}
		if r.ContentLength != 0 {
			t.Errorf("ContentLength = %v; want 0", r.ContentLength)
		}
		if r.Close {
			t.Error("Close = true; want false")
		}
		if !strings.Contains(r.RemoteAddr, ":") {
			t.Errorf("RemoteAddr = %q; want something with a colon", r.RemoteAddr)
		}
		if r.Proto != "HTTP/2.0" || r.ProtoMajor != 2 || r.ProtoMinor != 0 {
			t.Errorf("Proto = %q Major=%v,Minor=%v; want HTTP/2.0", r.Proto, r.ProtoMajor, r.ProtoMinor)
		}
		wantHeader := http.Header{
			"Foo-Bar": []string{"some-value"},
		}
		if !reflect.DeepEqual(r.Header, wantHeader) {
			t.Errorf("Header = %#v; want %#v", r.Header, wantHeader)
		}
		if n, err := r.Body.Read([]byte(" ")); err != io.EOF || n != 0 {
			t.Errorf("Read = %d, %v; want 0, EOF", n, err)
		}
	})
}

func TestServer_Request_Get_PathSlashes(t *testing.T) {
	synctestTest(t, testServer_Request_Get_PathSlashes)
}
func testServer_Request_Get_PathSlashes(t testing.TB) {
	testServerRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":path", "/%2f/"),
			EndStream:     true, // no DATA frames
			EndHeaders:    true,
		})
	}, func(r *http.Request) {
		if r.RequestURI != "/%2f/" {
			t.Errorf("RequestURI = %q; want /%%2f/", r.RequestURI)
		}
		if r.URL.Path != "///" {
			t.Errorf("URL.Path = %q; want ///", r.URL.Path)
		}
	})
}

// TODO: add a test with EndStream=true on the HEADERS but setting a
// Content-Length anyway. Should we just omit it and force it to
// zero?

func TestServer_Request_Post_NoContentLength_EndStream(t *testing.T) {
	synctestTest(t, testServer_Request_Post_NoContentLength_EndStream)
}
func testServer_Request_Post_NoContentLength_EndStream(t testing.TB) {
	testServerRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":method", "POST"),
			EndStream:     true,
			EndHeaders:    true,
		})
	}, func(r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Method = %q; want POST", r.Method)
		}
		if r.ContentLength != 0 {
			t.Errorf("ContentLength = %v; want 0", r.ContentLength)
		}
		if n, err := r.Body.Read([]byte(" ")); err != io.EOF || n != 0 {
			t.Errorf("Read = %d, %v; want 0, EOF", n, err)
		}
	})
}

func TestServer_Request_Post_Body_ImmediateEOF(t *testing.T) {
	synctestTest(t, testServer_Request_Post_Body_ImmediateEOF)
}
func testServer_Request_Post_Body_ImmediateEOF(t testing.TB) {
	testBodyContents(t, -1, "", func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":method", "POST"),
			EndStream:     false, // to say DATA frames are coming
			EndHeaders:    true,
		})
		st.writeData(1, true, nil) // just kidding. empty body.
	})
}

func TestServer_Request_Post_Body_OneData(t *testing.T) {
	synctestTest(t, testServer_Request_Post_Body_OneData)
}
func testServer_Request_Post_Body_OneData(t testing.TB) {
	const content = "Some content"
	testBodyContents(t, -1, content, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":method", "POST"),
			EndStream:     false, // to say DATA frames are coming
			EndHeaders:    true,
		})
		st.writeData(1, true, []byte(content))
	})
}

func TestServer_Request_Post_Body_TwoData(t *testing.T) {
	synctestTest(t, testServer_Request_Post_Body_TwoData)
}
func testServer_Request_Post_Body_TwoData(t testing.TB) {
	const content = "Some content"
	testBodyContents(t, -1, content, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":method", "POST"),
			EndStream:     false, // to say DATA frames are coming
			EndHeaders:    true,
		})
		st.writeData(1, false, []byte(content[:5]))
		st.writeData(1, true, []byte(content[5:]))
	})
}

func TestServer_Request_Post_Body_ContentLength_Correct(t *testing.T) {
	synctestTest(t, testServer_Request_Post_Body_ContentLength_Correct)
}
func testServer_Request_Post_Body_ContentLength_Correct(t testing.TB) {
	const content = "Some content"
	testBodyContents(t, int64(len(content)), content, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1, // clients send odd numbers
			BlockFragment: st.encodeHeader(
				":method", "POST",
				"content-length", strconv.Itoa(len(content)),
			),
			EndStream:  false, // to say DATA frames are coming
			EndHeaders: true,
		})
		st.writeData(1, true, []byte(content))
	})
}

func TestServer_Request_Post_Body_ContentLength_TooLarge(t *testing.T) {
	synctestTest(t, testServer_Request_Post_Body_ContentLength_TooLarge)
}
func testServer_Request_Post_Body_ContentLength_TooLarge(t testing.TB) {
	testBodyContentsFail(t, 3, "request declared a Content-Length of 3 but only wrote 2 bytes",
		func(st *serverTester) {
			st.writeHeaders(HeadersFrameParam{
				StreamID: 1, // clients send odd numbers
				BlockFragment: st.encodeHeader(
					":method", "POST",
					"content-length", "3",
				),
				EndStream:  false, // to say DATA frames are coming
				EndHeaders: true,
			})
			st.writeData(1, true, []byte("12"))
		})
}

func TestServer_Request_Post_Body_ContentLength_EndStream(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1, // clients send odd numbers
			BlockFragment: st.encodeHeader(
				":method", "POST",
				"content-length", "3",
			),
			EndStream:  true,
			EndHeaders: true,
		})
	})
}

func TestServer_Request_Post_Body_ContentLength_TooSmall(t *testing.T) {
	synctestTest(t, testServer_Request_Post_Body_ContentLength_TooSmall)
}
func testServer_Request_Post_Body_ContentLength_TooSmall(t testing.TB) {
	testBodyContentsFail(t, 4, "sender tried to send more than declared Content-Length of 4 bytes",
		func(st *serverTester) {
			st.writeHeaders(HeadersFrameParam{
				StreamID: 1, // clients send odd numbers
				BlockFragment: st.encodeHeader(
					":method", "POST",
					"content-length", "4",
				),
				EndStream:  false, // to say DATA frames are coming
				EndHeaders: true,
			})
			st.writeData(1, true, []byte("12345"))
			// Return flow control bytes back, since the data handler closed
			// the stream.
			st.wantRSTStream(1, ErrCodeProtocol)
			st.wantConnFlowControlConsumed(0)
		})
}

func testBodyContents(t testing.TB, wantContentLength int64, wantBody string, write func(st *serverTester)) {
	testServerRequest(t, write, func(r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Method = %q; want POST", r.Method)
		}
		if r.ContentLength != wantContentLength {
			t.Errorf("ContentLength = %v; want %d", r.ContentLength, wantContentLength)
		}
		all, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(all) != wantBody {
			t.Errorf("Read = %q; want %q", all, wantBody)
		}
		if err := r.Body.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	})
}

func testBodyContentsFail(t testing.TB, wantContentLength int64, wantReadError string, write func(st *serverTester)) {
	testServerRequest(t, write, func(r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Method = %q; want POST", r.Method)
		}
		if r.ContentLength != wantContentLength {
			t.Errorf("ContentLength = %v; want %d", r.ContentLength, wantContentLength)
		}
		all, err := io.ReadAll(r.Body)
		if err == nil {
			t.Fatalf("expected an error (%q) reading from the body. Successfully read %q instead.",
				wantReadError, all)
		}
		if !strings.Contains(err.Error(), wantReadError) {
			t.Fatalf("Body.Read = %v; want substring %q", err, wantReadError)
		}
		if err := r.Body.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	})
}

// Using a Host header, instead of :authority
func TestServer_Request_Get_Host(t *testing.T) { synctestTest(t, testServer_Request_Get_Host) }
func testServer_Request_Get_Host(t testing.TB) {
	const host = "example.com"
	testServerRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":authority", "", "host", host),
			EndStream:     true,
			EndHeaders:    true,
		})
	}, func(r *http.Request) {
		if r.Host != host {
			t.Errorf("Host = %q; want %q", r.Host, host)
		}
	})
}

// Using an :authority pseudo-header, instead of Host
func TestServer_Request_Get_Authority(t *testing.T) {
	synctestTest(t, testServer_Request_Get_Authority)
}
func testServer_Request_Get_Authority(t testing.TB) {
	const host = "example.com"
	testServerRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":authority", host),
			EndStream:     true,
			EndHeaders:    true,
		})
	}, func(r *http.Request) {
		if r.Host != host {
			t.Errorf("Host = %q; want %q", r.Host, host)
		}
	})
}

func TestServer_Request_WithContinuation(t *testing.T) {
	synctestTest(t, testServer_Request_WithContinuation)
}
func testServer_Request_WithContinuation(t testing.TB) {
	wantHeader := http.Header{
		"Foo-One":   []string{"value-one"},
		"Foo-Two":   []string{"value-two"},
		"Foo-Three": []string{"value-three"},
	}
	testServerRequest(t, func(st *serverTester) {
		fullHeaders := st.encodeHeader(
			"foo-one", "value-one",
			"foo-two", "value-two",
			"foo-three", "value-three",
		)
		remain := fullHeaders
		chunks := 0
		for len(remain) > 0 {
			const maxChunkSize = 5
			chunk := remain
			if len(chunk) > maxChunkSize {
				chunk = chunk[:maxChunkSize]
			}
			remain = remain[len(chunk):]

			if chunks == 0 {
				st.writeHeaders(HeadersFrameParam{
					StreamID:      1, // clients send odd numbers
					BlockFragment: chunk,
					EndStream:     true,  // no DATA frames
					EndHeaders:    false, // we'll have continuation frames
				})
			} else {
				err := st.fr.WriteContinuation(1, len(remain) == 0, chunk)
				if err != nil {
					t.Fatal(err)
				}
			}
			chunks++
		}
		if chunks < 2 {
			t.Fatal("too few chunks")
		}
	}, func(r *http.Request) {
		if !reflect.DeepEqual(r.Header, wantHeader) {
			t.Errorf("Header = %#v; want %#v", r.Header, wantHeader)
		}
	})
}

// Concatenated cookie headers. ("8.1.2.5 Compressing the Cookie Header Field")
func TestServer_Request_CookieConcat(t *testing.T) { synctestTest(t, testServer_Request_CookieConcat) }
func testServer_Request_CookieConcat(t testing.TB) {
	const host = "example.com"
	testServerRequest(t, func(st *serverTester) {
		st.bodylessReq1(
			":authority", host,
			"cookie", "a=b",
			"cookie", "c=d",
			"cookie", "e=f",
		)
	}, func(r *http.Request) {
		const want = "a=b; c=d; e=f"
		if got := r.Header.Get("Cookie"); got != want {
			t.Errorf("Cookie = %q; want %q", got, want)
		}
	})
}

func TestServer_Request_Reject_CapitalHeader(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("UPPER", "v") })
}

func TestServer_Request_Reject_HeaderFieldNameColon(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("has:colon", "v") })
}

func TestServer_Request_Reject_HeaderFieldNameNULL(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("has\x00null", "v") })
}

func TestServer_Request_Reject_HeaderFieldNameEmpty(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("", "v") })
}

func TestServer_Request_Reject_HeaderFieldValueNewline(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("foo", "has\nnewline") })
}

func TestServer_Request_Reject_HeaderFieldValueCR(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("foo", "has\rcarriage") })
}

func TestServer_Request_Reject_HeaderFieldValueDEL(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1("foo", "has\x7fdel") })
}

func TestServer_Request_Reject_Pseudo_Missing_method(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1(":method", "") })
}

func TestServer_Request_Reject_Pseudo_ExactlyOne(t *testing.T) {
	// 8.1.2.3 Request Pseudo-Header Fields
	// "All HTTP/2 requests MUST include exactly one valid value" ...
	testRejectRequest(t, func(st *serverTester) {
		st.addLogFilter("duplicate pseudo-header")
		st.bodylessReq1(":method", "GET", ":method", "POST")
	})
}

func TestServer_Request_Reject_Pseudo_AfterRegular(t *testing.T) {
	// 8.1.2.3 Request Pseudo-Header Fields
	// "All pseudo-header fields MUST appear in the header block
	// before regular header fields. Any request or response that
	// contains a pseudo-header field that appears in a header
	// block after a regular header field MUST be treated as
	// malformed (Section 8.1.2.6)."
	testRejectRequest(t, func(st *serverTester) {
		st.addLogFilter("pseudo-header after regular header")
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		enc.WriteField(hpack.HeaderField{Name: ":method", Value: "GET"})
		enc.WriteField(hpack.HeaderField{Name: "regular", Value: "foobar"})
		enc.WriteField(hpack.HeaderField{Name: ":path", Value: "/"})
		enc.WriteField(hpack.HeaderField{Name: ":scheme", Value: "https"})
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: buf.Bytes(),
			EndStream:     true,
			EndHeaders:    true,
		})
	})
}

func TestServer_Request_Reject_Pseudo_Missing_path(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1(":path", "") })
}

func TestServer_Request_Reject_Pseudo_Missing_scheme(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1(":scheme", "") })
}

func TestServer_Request_Reject_Pseudo_scheme_invalid(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) { st.bodylessReq1(":scheme", "bogus") })
}

func TestServer_Request_Reject_Pseudo_Unknown(t *testing.T) {
	testRejectRequest(t, func(st *serverTester) {
		st.addLogFilter(`invalid pseudo-header ":unknown_thing"`)
		st.bodylessReq1(":unknown_thing", "")
	})
}

func TestServer_Request_Reject_Authority_Userinfo(t *testing.T) {
	// "':authority' MUST NOT include the deprecated userinfo subcomponent
	// for "http" or "https" schemed URIs."
	// https://www.rfc-editor.org/rfc/rfc9113.html#section-8.3.1-2.3.8
	testRejectRequest(t, func(st *serverTester) {
		var buf bytes.Buffer
		enc := hpack.NewEncoder(&buf)
		enc.WriteField(hpack.HeaderField{Name: ":authority", Value: "userinfo@example.tld"})
		enc.WriteField(hpack.HeaderField{Name: ":method", Value: "GET"})
		enc.WriteField(hpack.HeaderField{Name: ":path", Value: "/"})
		enc.WriteField(hpack.HeaderField{Name: ":scheme", Value: "https"})
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: buf.Bytes(),
			EndStream:     true,
			EndHeaders:    true,
		})
	})
}

func testRejectRequest(t *testing.T, send func(*serverTester)) {
	synctestTest(t, func(t testing.TB) {
		st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
			t.Error("server request made it to handler; should've been rejected")
		})
		defer st.Close()

		st.greet()
		send(st)
		st.wantRSTStream(1, ErrCodeProtocol)
	})
}

func newServerTesterForError(t testing.TB) *serverTester {
	t.Helper()
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("server request made it to handler; should've been rejected")
	}, optQuiet)
	st.greet()
	return st
}

// Section 5.1, on idle connections: "Receiving any frame other than
// HEADERS or PRIORITY on a stream in this state MUST be treated as a
// connection error (Section 5.4.1) of type PROTOCOL_ERROR."
func TestRejectFrameOnIdle_WindowUpdate(t *testing.T) {
	synctestTest(t, testRejectFrameOnIdle_WindowUpdate)
}
func testRejectFrameOnIdle_WindowUpdate(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.WriteWindowUpdate(123, 456)
	st.wantGoAway(123, ErrCodeProtocol)
}
func TestRejectFrameOnIdle_Data(t *testing.T) { synctestTest(t, testRejectFrameOnIdle_Data) }
func testRejectFrameOnIdle_Data(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.WriteData(123, true, nil)
	st.wantGoAway(123, ErrCodeProtocol)
}
func TestRejectFrameOnIdle_RSTStream(t *testing.T) { synctestTest(t, testRejectFrameOnIdle_RSTStream) }
func testRejectFrameOnIdle_RSTStream(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.WriteRSTStream(123, ErrCodeCancel)
	st.wantGoAway(123, ErrCodeProtocol)
}

func TestServer_Request_Connect(t *testing.T) { synctestTest(t, testServer_Request_Connect) }
func testServer_Request_Connect(t testing.TB) {
	testServerRequest(t, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1,
			BlockFragment: st.encodeHeaderRaw(
				":method", "CONNECT",
				":authority", "example.com:123",
			),
			EndStream:  true,
			EndHeaders: true,
		})
	}, func(r *http.Request) {
		if g, w := r.Method, "CONNECT"; g != w {
			t.Errorf("Method = %q; want %q", g, w)
		}
		if g, w := r.RequestURI, "example.com:123"; g != w {
			t.Errorf("RequestURI = %q; want %q", g, w)
		}
		if g, w := r.URL.Host, "example.com:123"; g != w {
			t.Errorf("URL.Host = %q; want %q", g, w)
		}
	})
}

func TestServer_Request_Connect_InvalidPath(t *testing.T) {
	synctestTest(t, testServer_Request_Connect_InvalidPath)
}
func testServer_Request_Connect_InvalidPath(t testing.TB) {
	testServerRejectsStream(t, ErrCodeProtocol, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1,
			BlockFragment: st.encodeHeaderRaw(
				":method", "CONNECT",
				":authority", "example.com:123",
				":path", "/bogus",
			),
			EndStream:  true,
			EndHeaders: true,
		})
	})
}

func TestServer_Request_Connect_InvalidScheme(t *testing.T) {
	synctestTest(t, testServer_Request_Connect_InvalidScheme)
}
func testServer_Request_Connect_InvalidScheme(t testing.TB) {
	testServerRejectsStream(t, ErrCodeProtocol, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1,
			BlockFragment: st.encodeHeaderRaw(
				":method", "CONNECT",
				":authority", "example.com:123",
				":scheme", "https",
			),
			EndStream:  true,
			EndHeaders: true,
		})
	})
}

func TestServer_Ping(t *testing.T) { synctestTest(t, testServer_Ping) }
func testServer_Ping(t testing.TB) {
	st := newServerTester(t, nil)
	defer st.Close()
	st.greet()

	// Server should ignore this one, since it has ACK set.
	ackPingData := [8]byte{1, 2, 4, 8, 16, 32, 64, 128}
	if err := st.fr.WritePing(true, ackPingData); err != nil {
		t.Fatal(err)
	}

	// But the server should reply to this one, since ACK is false.
	pingData := [8]byte{1, 2, 3, 4, 5, 6, 7, 8}
	if err := st.fr.WritePing(false, pingData); err != nil {
		t.Fatal(err)
	}

	pf := readFrame[*PingFrame](t, st)
	if !pf.Flags.Has(FlagPingAck) {
		t.Error("response ping doesn't have ACK set")
	}
	if pf.Data != pingData {
		t.Errorf("response ping has data %q; want %q", pf.Data, pingData)
	}
}

type filterListener struct {
	net.Listener
	accept func(conn net.Conn) (net.Conn, error)
}

func (l *filterListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return l.accept(c)
}

func TestServer_MaxQueuedControlFrames(t *testing.T) {
	synctestTest(t, testServer_MaxQueuedControlFrames)
}
func testServer_MaxQueuedControlFrames(t testing.TB) {
	// Goroutine debugging makes this test very slow.
	DisableGoroutineTracking(t)

	st := newServerTester(t, nil)
	st.greet()

	st.cc.(*tls.Conn).NetConn().(*synctestNetConn).SetReadBufferSize(0) // all writes block

	// Send maxQueuedControlFrames pings, plus a few extra
	// to account for ones that enter the server's write buffer.
	const extraPings = 2
	for range MaxQueuedControlFrames + extraPings {
		pingData := [8]byte{1, 2, 3, 4, 5, 6, 7, 8}
		st.fr.WritePing(false, pingData)
	}
	synctest.Wait()

	// Unblock the server.
	// It should have closed the connection after exceeding the control frame limit.
	st.cc.(*tls.Conn).NetConn().(*synctestNetConn).SetReadBufferSize(math.MaxInt)

	st.advance(GoAwayTimeout)
	// Some frames may have persisted in the server's buffers.
	for range 10 {
		if st.readFrame() == nil {
			break
		}
	}
	st.wantClosed()
}

func TestServer_RejectsLargeFrames(t *testing.T) { synctestTest(t, testServer_RejectsLargeFrames) }
func testServer_RejectsLargeFrames(t testing.TB) {
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" || runtime.GOOS == "zos" {
		t.Skip("see golang.org/issue/13434, golang.org/issue/37321")
	}
	st := newServerTester(t, nil)
	defer st.Close()
	st.greet()

	// Write too large of a frame (too large by one byte)
	// We ignore the return value because it's expected that the server
	// will only read the first 9 bytes (the headre) and then disconnect.
	st.fr.WriteRawFrame(0xff, 0, 0, make([]byte, DefaultMaxReadFrameSize+1))

	st.wantGoAway(0, ErrCodeFrameSize)
	st.advance(GoAwayTimeout)
	st.wantClosed()
}

func TestServer_Handler_Sends_WindowUpdate(t *testing.T) {
	synctestTest(t, testServer_Handler_Sends_WindowUpdate)
}
func testServer_Handler_Sends_WindowUpdate(t testing.TB) {
	// Need to set this to at least twice the initial window size,
	// or st.greet gets stuck waiting for a WINDOW_UPDATE.
	//
	// This also needs to be less than MAX_FRAME_SIZE.
	const windowSize = 65535 * 2
	st := newServerTester(t, nil, func(h2 *http.HTTP2Config) {
		h2.MaxReceiveBufferPerConnection = windowSize
		h2.MaxReceiveBufferPerStream = windowSize
	})
	defer st.Close()

	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1, // clients send odd numbers
		BlockFragment: st.encodeHeader(":method", "POST"),
		EndStream:     false, // data coming
		EndHeaders:    true,
	})
	call := st.nextHandlerCall()

	// Write less than half the max window of data and consume it.
	// The server doesn't return flow control yet, buffering the 1024 bytes to
	// combine with a future update.
	data := make([]byte, windowSize)
	st.writeData(1, false, data[:1024])
	call.do(readBodyHandler(t, string(data[:1024])))

	// Write up to the window limit.
	// The server returns the buffered credit.
	st.writeData(1, false, data[1024:])
	st.wantWindowUpdate(0, 1024)
	st.wantWindowUpdate(1, 1024)

	// The handler consumes the data and the server returns credit.
	call.do(readBodyHandler(t, string(data[1024:])))
	st.wantWindowUpdate(0, windowSize-1024)
	st.wantWindowUpdate(1, windowSize-1024)
}

// the version of the TestServer_Handler_Sends_WindowUpdate with padding.
// See golang.org/issue/16556
func TestServer_Handler_Sends_WindowUpdate_Padding(t *testing.T) {
	synctestTest(t, testServer_Handler_Sends_WindowUpdate_Padding)
}
func testServer_Handler_Sends_WindowUpdate_Padding(t testing.TB) {
	const windowSize = 65535 * 2
	st := newServerTester(t, nil, func(h2 *http.HTTP2Config) {
		h2.MaxReceiveBufferPerConnection = windowSize
		h2.MaxReceiveBufferPerStream = windowSize
	})
	defer st.Close()

	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(":method", "POST"),
		EndStream:     false,
		EndHeaders:    true,
	})
	call := st.nextHandlerCall()

	// Write half a window of data, with some padding.
	// The server doesn't return the padding yet, buffering the 5 bytes to combine
	// with a future update.
	data := make([]byte, windowSize/2)
	pad := make([]byte, 4)
	st.writeDataPadded(1, false, data, pad)

	// The handler consumes the body.
	// The server returns flow control for the body and padding
	// (4 bytes of padding + 1 byte of length).
	call.do(readBodyHandler(t, string(data)))
	st.wantWindowUpdate(0, uint32(len(data)+1+len(pad)))
	st.wantWindowUpdate(1, uint32(len(data)+1+len(pad)))
}

func TestServer_Send_GoAway_After_Bogus_WindowUpdate(t *testing.T) {
	synctestTest(t, testServer_Send_GoAway_After_Bogus_WindowUpdate)
}
func testServer_Send_GoAway_After_Bogus_WindowUpdate(t testing.TB) {
	st := newServerTester(t, nil)
	defer st.Close()
	st.greet()
	if err := st.fr.WriteWindowUpdate(0, 1<<31-1); err != nil {
		t.Fatal(err)
	}
	st.wantGoAway(0, ErrCodeFlowControl)
}

func TestServer_Send_RstStream_After_Bogus_WindowUpdate(t *testing.T) {
	synctestTest(t, testServer_Send_RstStream_After_Bogus_WindowUpdate)
}
func testServer_Send_RstStream_After_Bogus_WindowUpdate(t testing.TB) {
	inHandler := make(chan bool)
	blockHandler := make(chan bool)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		inHandler <- true
		<-blockHandler
	})
	defer st.Close()
	defer close(blockHandler)
	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(":method", "POST"),
		EndStream:     false, // keep it open
		EndHeaders:    true,
	})
	<-inHandler
	// Send a bogus window update:
	if err := st.fr.WriteWindowUpdate(1, 1<<31-1); err != nil {
		t.Fatal(err)
	}
	st.wantRSTStream(1, ErrCodeFlowControl)
}

// testServerPostUnblock sends a hanging POST with unsent data to handler,
// then runs fn once in the handler, and verifies that the error returned from
// handler is acceptable. It fails if takes over 5 seconds for handler to exit.
func testServerPostUnblock(t testing.TB,
	handler func(http.ResponseWriter, *http.Request) error,
	fn func(*serverTester),
	checkErr func(error),
	otherHeaders ...string) {
	inHandler := make(chan bool)
	errc := make(chan error, 1)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		inHandler <- true
		errc <- handler(w, r)
	})
	defer st.Close()
	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(append([]string{":method", "POST"}, otherHeaders...)...),
		EndStream:     false, // keep it open
		EndHeaders:    true,
	})
	<-inHandler
	fn(st)
	err := <-errc
	if checkErr != nil {
		checkErr(err)
	}
}

func TestServer_RSTStream_Unblocks_Read(t *testing.T) {
	synctestTest(t, testServer_RSTStream_Unblocks_Read)
}
func testServer_RSTStream_Unblocks_Read(t testing.TB) {
	testServerPostUnblock(t,
		func(w http.ResponseWriter, r *http.Request) (err error) {
			_, err = r.Body.Read(make([]byte, 1))
			return
		},
		func(st *serverTester) {
			if err := st.fr.WriteRSTStream(1, ErrCodeCancel); err != nil {
				t.Fatal(err)
			}
		},
		func(err error) {
			want := StreamError{StreamID: 0x1, Code: 0x8}
			if !reflect.DeepEqual(err, want) {
				t.Errorf("Read error = %v; want %v", err, want)
			}
		},
	)
}

func TestServer_RSTStream_Unblocks_Header_Write(t *testing.T) {
	// Run this test a bunch, because it doesn't always
	// deadlock. But with a bunch, it did.
	n := 50
	if testing.Short() {
		n = 5
	}
	for i := 0; i < n; i++ {
		synctestTest(t, testServer_RSTStream_Unblocks_Header_Write)
	}
}

func testServer_RSTStream_Unblocks_Header_Write(t testing.TB) {
	inHandler := make(chan bool, 1)
	unblockHandler := make(chan bool, 1)
	headerWritten := make(chan bool, 1)
	wroteRST := make(chan bool, 1)

	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		inHandler <- true
		<-wroteRST
		w.Header().Set("foo", "bar")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		headerWritten <- true
		<-unblockHandler
	})
	defer st.Close()

	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(":method", "POST"),
		EndStream:     false, // keep it open
		EndHeaders:    true,
	})
	<-inHandler
	if err := st.fr.WriteRSTStream(1, ErrCodeCancel); err != nil {
		t.Fatal(err)
	}
	wroteRST <- true
	synctest.Wait()
	<-headerWritten
	unblockHandler <- true
}

func TestServer_DeadConn_Unblocks_Read(t *testing.T) {
	synctestTest(t, testServer_DeadConn_Unblocks_Read)
}
func testServer_DeadConn_Unblocks_Read(t testing.TB) {
	testServerPostUnblock(t,
		func(w http.ResponseWriter, r *http.Request) (err error) {
			_, err = r.Body.Read(make([]byte, 1))
			return
		},
		func(st *serverTester) { st.cc.Close() },
		func(err error) {
			if err == nil {
				t.Error("unexpected nil error from Request.Body.Read")
			}
		},
	)
}

var blockUntilClosed = func(w http.ResponseWriter, r *http.Request) error {
	<-w.(http.CloseNotifier).CloseNotify()
	return nil
}

func TestServer_CloseNotify_After_RSTStream(t *testing.T) {
	synctestTest(t, testServer_CloseNotify_After_RSTStream)
}
func testServer_CloseNotify_After_RSTStream(t testing.TB) {
	testServerPostUnblock(t, blockUntilClosed, func(st *serverTester) {
		if err := st.fr.WriteRSTStream(1, ErrCodeCancel); err != nil {
			t.Fatal(err)
		}
	}, nil)
}

func TestServer_CloseNotify_After_ConnClose(t *testing.T) {
	synctestTest(t, testServer_CloseNotify_After_ConnClose)
}
func testServer_CloseNotify_After_ConnClose(t testing.TB) {
	testServerPostUnblock(t, blockUntilClosed, func(st *serverTester) { st.cc.Close() }, nil)
}

// that CloseNotify unblocks after a stream error due to the client's
// problem that's unrelated to them explicitly canceling it (which is
// TestServer_CloseNotify_After_RSTStream above)
func TestServer_CloseNotify_After_StreamError(t *testing.T) {
	synctestTest(t, testServer_CloseNotify_After_StreamError)
}
func testServer_CloseNotify_After_StreamError(t testing.TB) {
	testServerPostUnblock(t, blockUntilClosed, func(st *serverTester) {
		// data longer than declared Content-Length => stream error
		st.writeData(1, true, []byte("1234"))
	}, nil, "content-length", "3")
}

func TestServer_StateTransitions(t *testing.T) { synctestTest(t, testServer_StateTransitions) }
func testServer_StateTransitions(t testing.TB) {
	var st *serverTester
	inHandler := make(chan bool)
	writeData := make(chan bool)
	leaveHandler := make(chan bool)
	st = newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		inHandler <- true
		if !st.streamExists(1) {
			t.Errorf("stream 1 does not exist in handler")
		}
		if got, want := st.streamState(1), StateOpen; got != want {
			t.Errorf("in handler, state is %v; want %v", got, want)
		}
		writeData <- true
		if n, err := r.Body.Read(make([]byte, 1)); n != 0 || err != io.EOF {
			t.Errorf("body read = %d, %v; want 0, EOF", n, err)
		}
		if got, want := st.streamState(1), StateHalfClosedRemote; got != want {
			t.Errorf("in handler, state is %v; want %v", got, want)
		}

		<-leaveHandler
	})
	st.greet()
	if st.streamExists(1) {
		t.Fatal("stream 1 should be empty")
	}
	if got := st.streamState(1); got != StateIdle {
		t.Fatalf("stream 1 should be idle; got %v", got)
	}

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(":method", "POST"),
		EndStream:     false, // keep it open
		EndHeaders:    true,
	})
	<-inHandler
	<-writeData
	st.writeData(1, true, nil)

	leaveHandler <- true
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
	})

	if got, want := st.streamState(1), StateClosed; got != want {
		t.Errorf("at end, state is %v; want %v", got, want)
	}
	if st.streamExists(1) {
		t.Fatal("at end, stream 1 should be gone")
	}
}

// test HEADERS w/o EndHeaders + another HEADERS (should get rejected)
func TestServer_Rejects_HeadersNoEnd_Then_Headers(t *testing.T) {
	synctestTest(t, testServer_Rejects_HeadersNoEnd_Then_Headers)
}
func testServer_Rejects_HeadersNoEnd_Then_Headers(t testing.TB) {
	st := newServerTesterForError(t)
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    false,
	})
	st.writeHeaders(HeadersFrameParam{ // Not a continuation.
		StreamID:      3, // different stream.
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantGoAway(0, ErrCodeProtocol)
}

// test HEADERS w/o EndHeaders + PING (should get rejected)
func TestServer_Rejects_HeadersNoEnd_Then_Ping(t *testing.T) {
	synctestTest(t, testServer_Rejects_HeadersNoEnd_Then_Ping)
}
func testServer_Rejects_HeadersNoEnd_Then_Ping(t testing.TB) {
	st := newServerTesterForError(t)
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    false,
	})
	if err := st.fr.WritePing(false, [8]byte{}); err != nil {
		t.Fatal(err)
	}
	st.wantGoAway(0, ErrCodeProtocol)
}

// test HEADERS w/ EndHeaders + a continuation HEADERS (should get rejected)
func TestServer_Rejects_HeadersEnd_Then_Continuation(t *testing.T) {
	synctestTest(t, testServer_Rejects_HeadersEnd_Then_Continuation)
}
func testServer_Rejects_HeadersEnd_Then_Continuation(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {}, optQuiet)
	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
	})
	if err := st.fr.WriteContinuation(1, true, EncodeHeaderRaw(t, "foo", "bar")); err != nil {
		t.Fatal(err)
	}
	st.wantGoAway(1, ErrCodeProtocol)
}

// test HEADERS w/o EndHeaders + a continuation HEADERS on wrong stream ID
func TestServer_Rejects_HeadersNoEnd_Then_ContinuationWrongStream(t *testing.T) {
	synctestTest(t, testServer_Rejects_HeadersNoEnd_Then_ContinuationWrongStream)
}
func testServer_Rejects_HeadersNoEnd_Then_ContinuationWrongStream(t testing.TB) {
	st := newServerTesterForError(t)
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    false,
	})
	if err := st.fr.WriteContinuation(3, true, EncodeHeaderRaw(t, "foo", "bar")); err != nil {
		t.Fatal(err)
	}
	st.wantGoAway(0, ErrCodeProtocol)
}

// No HEADERS on stream 0.
func TestServer_Rejects_Headers0(t *testing.T) { synctestTest(t, testServer_Rejects_Headers0) }
func testServer_Rejects_Headers0(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.AllowIllegalWrites = true
	st.writeHeaders(HeadersFrameParam{
		StreamID:      0,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantGoAway(0, ErrCodeProtocol)
}

// No CONTINUATION on stream 0.
func TestServer_Rejects_Continuation0(t *testing.T) {
	synctestTest(t, testServer_Rejects_Continuation0)
}
func testServer_Rejects_Continuation0(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.AllowIllegalWrites = true
	if err := st.fr.WriteContinuation(0, true, st.encodeHeader()); err != nil {
		t.Fatal(err)
	}
	st.wantGoAway(0, ErrCodeProtocol)
}

// No PRIORITY on stream 0.
func TestServer_Rejects_Priority0(t *testing.T) { synctestTest(t, testServer_Rejects_Priority0) }
func testServer_Rejects_Priority0(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.AllowIllegalWrites = true
	st.writePriority(0, PriorityParam{StreamDep: 1})
	st.wantGoAway(0, ErrCodeProtocol)
}

// PRIORITY_UPDATE only accepts non-zero ID for the prioritized stream ID in
// its payload.
func TestServer_Rejects_PriorityUpdate0(t *testing.T) {
	synctestTest(t, testServer_Rejects_PriorityUpdate0)
}
func testServer_Rejects_PriorityUpdate0(t testing.TB) {
	st := newServerTesterForError(t)
	st.fr.AllowIllegalWrites = true
	st.writePriorityUpdate(0, "")
	st.wantGoAway(0, ErrCodeProtocol)
}

// PRIORITY_UPDATE with unparsable priority parameters may be rejected.
func TestServer_Rejects_PriorityUpdateUnparsable(t *testing.T) {
	synctestTest(t, testServer_Rejects_PriorityUnparsable)
}
func testServer_Rejects_PriorityUnparsable(t testing.TB) {
	st := newServerTester(t, nil)
	defer st.Close()
	st.greet()
	st.writePriorityUpdate(1, "Invalid dictionary: ((((")
	st.wantRSTStream(1, ErrCodeProtocol)
}

// No HEADERS frame with a self-dependence.
func TestServer_Rejects_HeadersSelfDependence(t *testing.T) {
	synctestTest(t, testServer_Rejects_HeadersSelfDependence)
}
func testServer_Rejects_HeadersSelfDependence(t testing.TB) {
	testServerRejectsStream(t, ErrCodeProtocol, func(st *serverTester) {
		st.fr.AllowIllegalWrites = true
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
			Priority:      PriorityParam{StreamDep: 1},
		})
	})
}

// No PRIORITY frame with a self-dependence.
func TestServer_Rejects_PrioritySelfDependence(t *testing.T) {
	synctestTest(t, testServer_Rejects_PrioritySelfDependence)
}
func testServer_Rejects_PrioritySelfDependence(t testing.TB) {
	testServerRejectsStream(t, ErrCodeProtocol, func(st *serverTester) {
		st.fr.AllowIllegalWrites = true
		st.writePriority(1, PriorityParam{StreamDep: 1})
	})
}

func TestServer_Rejects_PushPromise(t *testing.T) { synctestTest(t, testServer_Rejects_PushPromise) }
func testServer_Rejects_PushPromise(t testing.TB) {
	st := newServerTesterForError(t)
	pp := PushPromiseParam{
		StreamID:  1,
		PromiseID: 3,
	}
	if err := st.fr.WritePushPromise(pp); err != nil {
		t.Fatal(err)
	}
	st.wantGoAway(1, ErrCodeProtocol)
}

// testServerRejectsStream tests that the server sends a RST_STREAM with the provided
// error code after a client sends a bogus request.
func testServerRejectsStream(t testing.TB, code ErrCode, writeReq func(*serverTester)) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {})
	defer st.Close()
	st.greet()
	writeReq(st)
	st.wantRSTStream(1, code)
}

// testServerRequest sets up an idle HTTP/2 connection and lets you
// write a single request with writeReq, and then verify that the
// *http.Request is built correctly in checkReq.
func testServerRequest(t testing.TB, writeReq func(*serverTester), checkReq func(*http.Request)) {
	gotReq := make(chan bool, 1)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Body == nil {
			t.Fatal("nil Body")
		}
		checkReq(r)
		gotReq <- true
	})
	defer st.Close()

	st.greet()
	writeReq(st)
	<-gotReq
}

func getSlash(st *serverTester) { st.bodylessReq1() }

func TestServer_Response_NoData(t *testing.T) { synctestTest(t, testServer_Response_NoData) }
func testServer_Response_NoData(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		// Nothing.
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: true,
		})
	})
}

func TestServer_Response_NoData_Header_FooBar(t *testing.T) {
	synctestTest(t, testServer_Response_NoData_Header_FooBar)
}
func testServer_Response_NoData_Header_FooBar(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("Foo-Bar", "some-value")
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: true,
			header: http.Header{
				":status":        []string{"200"},
				"foo-bar":        []string{"some-value"},
				"content-length": []string{"0"},
			},
		})
	})
}

// Reject content-length headers containing a sign.
// See https://golang.org/issue/39017
func TestServerIgnoresContentLengthSignWhenWritingChunks(t *testing.T) {
	synctestTest(t, testServerIgnoresContentLengthSignWhenWritingChunks)
}
func testServerIgnoresContentLengthSignWhenWritingChunks(t testing.TB) {
	tests := []struct {
		name   string
		cl     string
		wantCL string
	}{
		{
			name:   "proper content-length",
			cl:     "3",
			wantCL: "3",
		},
		{
			name:   "ignore cl with plus sign",
			cl:     "+3",
			wantCL: "0",
		},
		{
			name:   "ignore cl with minus sign",
			cl:     "-3",
			wantCL: "0",
		},
		{
			name:   "max int64, for safe uint64->int64 conversion",
			cl:     "9223372036854775807",
			wantCL: "9223372036854775807",
		},
		{
			name:   "overflows int64, so ignored",
			cl:     "9223372036854775808",
			wantCL: "0",
		},
	}

	for _, tt := range tests {
		testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
			w.Header().Set("content-length", tt.cl)
			return nil
		}, func(st *serverTester) {
			getSlash(st)
			st.wantHeaders(wantHeader{
				streamID:  1,
				endStream: true,
				header: http.Header{
					":status":        []string{"200"},
					"content-length": []string{tt.wantCL},
				},
			})
		})
	}
}

// Reject content-length headers containing a sign.
// See https://golang.org/issue/39017
func TestServerRejectsContentLengthWithSignNewRequests(t *testing.T) {
	tests := []struct {
		name   string
		cl     string
		wantCL int64
	}{
		{
			name:   "proper content-length",
			cl:     "3",
			wantCL: 3,
		},
		{
			name:   "ignore cl with plus sign",
			cl:     "+3",
			wantCL: 0,
		},
		{
			name:   "ignore cl with minus sign",
			cl:     "-3",
			wantCL: 0,
		},
		{
			name:   "max int64, for safe uint64->int64 conversion",
			cl:     "9223372036854775807",
			wantCL: 9223372036854775807,
		},
		{
			name:   "overflows int64, so ignored",
			cl:     "9223372036854775808",
			wantCL: 0,
		},
	}

	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t testing.TB) {
			writeReq := func(st *serverTester) {
				st.writeHeaders(HeadersFrameParam{
					StreamID:      1, // clients send odd numbers
					BlockFragment: st.encodeHeader("content-length", tt.cl),
					EndStream:     false,
					EndHeaders:    true,
				})
				st.writeData(1, false, []byte(""))
			}
			checkReq := func(r *http.Request) {
				if r.ContentLength != tt.wantCL {
					t.Fatalf("Got: %d\nWant: %d", r.ContentLength, tt.wantCL)
				}
			}
			testServerRequest(t, writeReq, checkReq)
		})
	}
}

func TestServer_Response_Data_Sniff_DoesntOverride(t *testing.T) {
	synctestTest(t, testServer_Response_Data_Sniff_DoesntOverride)
}
func testServer_Response_Data_Sniff_DoesntOverride(t testing.TB) {
	const msg = "<html>this is HTML."
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("Content-Type", "foo/bar")
		io.WriteString(w, msg)
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"content-type":   []string{"foo/bar"},
				"content-length": []string{strconv.Itoa(len(msg))},
			},
		})
		st.wantData(wantData{
			streamID:  1,
			endStream: true,
			data:      []byte(msg),
		})
	})
}

func TestServer_Response_TransferEncoding_chunked(t *testing.T) {
	synctestTest(t, testServer_Response_TransferEncoding_chunked)
}
func testServer_Response_TransferEncoding_chunked(t testing.TB) {
	const msg = "hi"
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("Transfer-Encoding", "chunked") // should be stripped
		io.WriteString(w, msg)
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"content-type":   []string{"text/plain; charset=utf-8"},
				"content-length": []string{strconv.Itoa(len(msg))},
			},
		})
	})
}

// Header accessed only after the initial write.
func TestServer_Response_Data_IgnoreHeaderAfterWrite_After(t *testing.T) {
	synctestTest(t, testServer_Response_Data_IgnoreHeaderAfterWrite_After)
}
func testServer_Response_Data_IgnoreHeaderAfterWrite_After(t testing.TB) {
	const msg = "<html>this is HTML."
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		io.WriteString(w, msg)
		w.Header().Set("foo", "should be ignored")
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"content-type":   []string{"text/html; charset=utf-8"},
				"content-length": []string{strconv.Itoa(len(msg))},
			},
		})
	})
}

// Header accessed before the initial write and later mutated.
func TestServer_Response_Data_IgnoreHeaderAfterWrite_Overwrite(t *testing.T) {
	synctestTest(t, testServer_Response_Data_IgnoreHeaderAfterWrite_Overwrite)
}
func testServer_Response_Data_IgnoreHeaderAfterWrite_Overwrite(t testing.TB) {
	const msg = "<html>this is HTML."
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("foo", "proper value")
		io.WriteString(w, msg)
		w.Header().Set("foo", "should be ignored")
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"foo":            []string{"proper value"},
				"content-type":   []string{"text/html; charset=utf-8"},
				"content-length": []string{strconv.Itoa(len(msg))},
			},
		})
	})
}

func TestServer_Response_Data_SniffLenType(t *testing.T) {
	synctestTest(t, testServer_Response_Data_SniffLenType)
}
func testServer_Response_Data_SniffLenType(t testing.TB) {
	const msg = "<html>this is HTML."
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		io.WriteString(w, msg)
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"content-type":   []string{"text/html; charset=utf-8"},
				"content-length": []string{strconv.Itoa(len(msg))},
			},
		})
		st.wantData(wantData{
			streamID:  1,
			endStream: true,
			data:      []byte(msg),
		})
	})
}

func TestServer_Response_Header_Flush_MidWrite(t *testing.T) {
	synctestTest(t, testServer_Response_Header_Flush_MidWrite)
}
func testServer_Response_Header_Flush_MidWrite(t testing.TB) {
	const msg = "<html>this is HTML"
	const msg2 = ", and this is the next chunk"
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		io.WriteString(w, msg)
		w.(http.Flusher).Flush()
		io.WriteString(w, msg2)
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":      []string{"200"},
				"content-type": []string{"text/html; charset=utf-8"}, // sniffed
				// and no content-length
			},
		})
		st.wantData(wantData{
			streamID:  1,
			endStream: false,
			data:      []byte(msg),
		})
		st.wantData(wantData{
			streamID:  1,
			endStream: true,
			data:      []byte(msg2),
		})
	})
}

func TestServer_Response_LargeWrite(t *testing.T) { synctestTest(t, testServer_Response_LargeWrite) }
func testServer_Response_LargeWrite(t testing.TB) {
	const size = 1 << 20
	const maxFrameSize = 16 << 10
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		n, err := w.Write(bytes.Repeat([]byte("a"), size))
		if err != nil {
			return fmt.Errorf("Write error: %v", err)
		}
		if n != size {
			return fmt.Errorf("wrong size %d from Write", n)
		}
		return nil
	}, func(st *serverTester) {
		if err := st.fr.WriteSettings(
			Setting{SettingInitialWindowSize, 0},
			Setting{SettingMaxFrameSize, maxFrameSize},
		); err != nil {
			t.Fatal(err)
		}
		st.wantSettingsAck()

		getSlash(st) // make the single request

		// Give the handler quota to write:
		if err := st.fr.WriteWindowUpdate(1, size); err != nil {
			t.Fatal(err)
		}
		// Give the handler quota to write to connection-level
		// window as well
		if err := st.fr.WriteWindowUpdate(0, size); err != nil {
			t.Fatal(err)
		}
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":      []string{"200"},
				"content-type": []string{"text/plain; charset=utf-8"}, // sniffed
				// and no content-length
			},
		})
		var bytes, frames int
		for {
			df := readFrame[*DataFrame](t, st)
			bytes += len(df.Data())
			frames++
			for _, b := range df.Data() {
				if b != 'a' {
					t.Fatal("non-'a' byte seen in DATA")
				}
			}
			if df.StreamEnded() {
				break
			}
		}
		if bytes != size {
			t.Errorf("Got %d bytes; want %d", bytes, size)
		}
		if want := int(size / maxFrameSize); frames < want || frames > want*2 {
			t.Errorf("Got %d frames; want %d", frames, size)
		}
	})
}

// Test that the handler can't write more than the client allows
func TestServer_Response_LargeWrite_FlowControlled(t *testing.T) {
	synctestTest(t, testServer_Response_LargeWrite_FlowControlled)
}
func testServer_Response_LargeWrite_FlowControlled(t testing.TB) {
	// Make these reads. Before each read, the client adds exactly enough
	// flow-control to satisfy the read. Numbers chosen arbitrarily.
	reads := []int{123, 1, 13, 127}
	size := 0
	for _, n := range reads {
		size += n
	}

	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.(http.Flusher).Flush()
		n, err := w.Write(bytes.Repeat([]byte("a"), size))
		if err != nil {
			return fmt.Errorf("Write error: %v", err)
		}
		if n != size {
			return fmt.Errorf("wrong size %d from Write", n)
		}
		return nil
	}, func(st *serverTester) {
		// Set the window size to something explicit for this test.
		// It's also how much initial data we expect.
		if err := st.fr.WriteSettings(Setting{SettingInitialWindowSize, uint32(reads[0])}); err != nil {
			t.Fatal(err)
		}
		st.wantSettingsAck()

		getSlash(st) // make the single request

		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
		})

		st.wantData(wantData{
			streamID:  1,
			endStream: false,
			size:      reads[0],
		})

		for i, quota := range reads[1:] {
			if err := st.fr.WriteWindowUpdate(1, uint32(quota)); err != nil {
				t.Fatal(err)
			}
			st.wantData(wantData{
				streamID:  1,
				endStream: i == len(reads[1:])-1,
				size:      quota,
			})
		}
	})
}

// Test that the handler blocked in a Write is unblocked if the server sends a RST_STREAM.
func TestServer_Response_RST_Unblocks_LargeWrite(t *testing.T) {
	synctestTest(t, testServer_Response_RST_Unblocks_LargeWrite)
}
func testServer_Response_RST_Unblocks_LargeWrite(t testing.TB) {
	const size = 1 << 20
	const maxFrameSize = 16 << 10
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.(http.Flusher).Flush()
		_, err := w.Write(bytes.Repeat([]byte("a"), size))
		if err == nil {
			return errors.New("unexpected nil error from Write in handler")
		}
		return nil
	}, func(st *serverTester) {
		if err := st.fr.WriteSettings(
			Setting{SettingInitialWindowSize, 0},
			Setting{SettingMaxFrameSize, maxFrameSize},
		); err != nil {
			t.Fatal(err)
		}
		st.wantSettingsAck()

		getSlash(st) // make the single request

		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
		})

		if err := st.fr.WriteRSTStream(1, ErrCodeCancel); err != nil {
			t.Fatal(err)
		}
	})
}

func TestServer_Response_Empty_Data_Not_FlowControlled(t *testing.T) {
	synctestTest(t, testServer_Response_Empty_Data_Not_FlowControlled)
}
func testServer_Response_Empty_Data_Not_FlowControlled(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.(http.Flusher).Flush()
		// Nothing; send empty DATA
		return nil
	}, func(st *serverTester) {
		// Handler gets no data quota:
		if err := st.fr.WriteSettings(Setting{SettingInitialWindowSize, 0}); err != nil {
			t.Fatal(err)
		}
		st.wantSettingsAck()

		getSlash(st) // make the single request

		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
		})

		st.wantData(wantData{
			streamID:  1,
			endStream: true,
			size:      0,
		})
	})
}

func TestServer_Response_Automatic100Continue(t *testing.T) {
	synctestTest(t, testServer_Response_Automatic100Continue)
}
func testServer_Response_Automatic100Continue(t testing.TB) {
	const msg = "foo"
	const reply = "bar"
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		if v := r.Header.Get("Expect"); v != "" {
			t.Errorf("Expect header = %q; want empty", v)
		}
		buf := make([]byte, len(msg))
		// This read should trigger the 100-continue being sent.
		if n, err := io.ReadFull(r.Body, buf); err != nil || n != len(msg) || string(buf) != msg {
			return fmt.Errorf("ReadFull = %q, %v; want %q, nil", buf[:n], err, msg)
		}
		_, err := io.WriteString(w, reply)
		return err
	}, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader(":method", "POST", "expect", "100-Continue"),
			EndStream:     false,
			EndHeaders:    true,
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status": []string{"100"},
			},
		})

		// Okay, they sent status 100, so we can send our
		// gigantic and/or sensitive "foo" payload now.
		st.writeData(1, true, []byte(msg))

		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"content-type":   []string{"text/plain; charset=utf-8"},
				"content-length": []string{strconv.Itoa(len(reply))},
			},
		})

		st.wantData(wantData{
			streamID:  1,
			endStream: true,
			data:      []byte(reply),
		})
	})
}

func TestServer_HandlerWriteErrorOnDisconnect(t *testing.T) {
	synctestTest(t, testServer_HandlerWriteErrorOnDisconnect)
}
func testServer_HandlerWriteErrorOnDisconnect(t testing.TB) {
	errc := make(chan error, 1)
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		p := []byte("some data.\n")
		for {
			_, err := w.Write(p)
			if err != nil {
				errc <- err
				return nil
			}
		}
	}, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1,
			BlockFragment: st.encodeHeader(),
			EndStream:     false,
			EndHeaders:    true,
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
		})
		// Close the connection and wait for the handler to (hopefully) notice.
		st.cc.Close()
		_ = <-errc
	})
}

func TestServer_Rejects_Too_Many_Streams(t *testing.T) {
	synctestTest(t, testServer_Rejects_Too_Many_Streams)
}
func testServer_Rejects_Too_Many_Streams(t testing.TB) {
	st := newServerTester(t, nil)
	st.greet()
	nextStreamID := uint32(1)
	streamID := func() uint32 {
		defer func() { nextStreamID += 2 }()
		return nextStreamID
	}
	sendReq := func(id uint32) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: id,
			BlockFragment: st.encodeHeader(
				":path", fmt.Sprintf("/%v", id),
			),
			EndStream:  true,
			EndHeaders: true,
		})
	}
	var calls []*serverHandlerCall
	for range DefaultMaxStreams {
		sendReq(streamID())
		calls = append(calls, st.nextHandlerCall())
	}

	// And this one should cross the limit:
	// (It's also sent as a CONTINUATION, to verify we still track the decoder context,
	// even if we're rejecting it)
	rejectID := streamID()
	headerBlock := st.encodeHeader(":path", fmt.Sprintf("/%v", rejectID))
	frag1, frag2 := headerBlock[:3], headerBlock[3:]
	st.writeHeaders(HeadersFrameParam{
		StreamID:      rejectID,
		BlockFragment: frag1,
		EndStream:     true,
		EndHeaders:    false, // CONTINUATION coming
	})
	if err := st.fr.WriteContinuation(rejectID, true, frag2); err != nil {
		t.Fatal(err)
	}
	st.sync()
	st.wantRSTStream(rejectID, ErrCodeProtocol)

	// But let a handler finish:
	calls[0].exit()
	st.sync()
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
	})

	// And now another stream should be able to start:
	goodID := streamID()
	sendReq(goodID)
	call := st.nextHandlerCall()
	if got, want := call.req.URL.Path, fmt.Sprintf("/%d", goodID); got != want {
		t.Errorf("Got request for %q, want %q", got, want)
	}
}

// So many response headers that the server needs to use CONTINUATION frames:
func TestServer_Response_ManyHeaders_With_Continuation(t *testing.T) {
	synctestTest(t, testServer_Response_ManyHeaders_With_Continuation)
}
func testServer_Response_ManyHeaders_With_Continuation(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		h := w.Header()
		for i := range 5000 {
			h.Set(fmt.Sprintf("x-header-%d", i), fmt.Sprintf("x-value-%d", i))
		}
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		hf := readFrame[*HeadersFrame](t, st)
		if hf.HeadersEnded() {
			t.Fatal("got unwanted END_HEADERS flag")
		}
		n := 0
		for {
			n++
			cf := readFrame[*ContinuationFrame](t, st)
			if cf.HeadersEnded() {
				break
			}
		}
		if n < 5 {
			t.Errorf("Only got %d CONTINUATION frames; expected 5+ (currently 6)", n)
		}
	})
}

// This previously crashed (reported by Mathieu Lonjaret as observed
// while using Camlistore) because we got a DATA frame from the client
// after the handler exited and our logic at the time was wrong,
// keeping a stream in the map in stateClosed, which tickled an
// invariant check later when we tried to remove that stream (via
// defer sc.closeAllStreamsOnConnClose) when the serverConn serve loop
// ended.
func TestServer_NoCrash_HandlerClose_Then_ClientClose(t *testing.T) {
	synctestTest(t, testServer_NoCrash_HandlerClose_Then_ClientClose)
}
func testServer_NoCrash_HandlerClose_Then_ClientClose(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		// nothing
		return nil
	}, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1,
			BlockFragment: st.encodeHeader(),
			EndStream:     false, // DATA is coming
			EndHeaders:    true,
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: true,
		})

		// Sent when the a Handler closes while a client has
		// indicated it's still sending DATA:
		st.wantRSTStream(1, ErrCodeNo)

		// Now the handler has ended, so it's ended its
		// stream, but the client hasn't closed its side
		// (stateClosedLocal).  So send more data and verify
		// it doesn't crash with an internal invariant panic, like
		// it did before.
		st.writeData(1, true, []byte("foo"))

		// Sent after a peer sends data anyway (admittedly the
		// previous RST_STREAM might've still been in-flight),
		// but they'll get the more friendly 'cancel' code
		// first.
		st.wantRSTStream(1, ErrCodeStreamClosed)

		// We should have our flow control bytes back,
		// since the handler didn't get them.
		st.wantConnFlowControlConsumed(0)

		// Set up a bunch of machinery to record the panic we saw
		// previously.
		var (
			panMu    sync.Mutex
			panicVal any
		)

		SetTestHookOnPanic(t, func(sc *ServerConn, pv any) bool {
			panMu.Lock()
			panicVal = pv
			panMu.Unlock()
			return true
		})

		// Now force the serve loop to end, via closing the connection.
		st.cc.Close()
		synctest.Wait()

		panMu.Lock()
		got := panicVal
		panMu.Unlock()
		if got != nil {
			t.Errorf("Got panic: %v", got)
		}
	})
}

func TestServer_Rejects_TLS10(t *testing.T) { testRejectTLS(t, tls.VersionTLS10) }
func TestServer_Rejects_TLS11(t *testing.T) { testRejectTLS(t, tls.VersionTLS11) }

func testRejectTLS(t *testing.T, version uint16) {
	synctestTest(t, func(t testing.TB) {
		st := newServerTester(t, nil, func(state *tls.ConnectionState) {
			// As of 1.18 the default minimum Go TLS version is
			// 1.2. In order to test rejection of lower versions,
			// manually set the version to 1.0
			state.Version = version
		})
		defer st.Close()
		st.wantGoAway(0, ErrCodeInadequateSecurity)
	})
}

func TestServer_Rejects_TLSBadCipher(t *testing.T) { synctestTest(t, testServer_Rejects_TLSBadCipher) }
func testServer_Rejects_TLSBadCipher(t testing.TB) {
	st := newServerTester(t, nil, func(state *tls.ConnectionState) {
		state.Version = tls.VersionTLS12
		state.CipherSuite = tls.TLS_RSA_WITH_RC4_128_SHA
	})
	defer st.Close()
	st.wantGoAway(0, ErrCodeInadequateSecurity)
}

func TestServer_Advertises_Common_Cipher(t *testing.T) {
	synctestTest(t, testServer_Advertises_Common_Cipher)
}
func testServer_Advertises_Common_Cipher(t testing.TB) {
	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
	}, func(srv *http.Server) {
		// Have the server configured with no specific cipher suites.
		// This tests that Go's defaults include the required one.
		srv.TLSConfig = nil
	})

	// Have the client only support the one required by the spec.
	const requiredSuite = tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	tlsConfig := tlsConfigInsecure.Clone()
	tlsConfig.MaxVersion = tls.VersionTLS12
	tlsConfig.CipherSuites = []uint16{requiredSuite}
	tr := &http.Transport{
		TLSClientConfig: tlsConfig,
		Protocols:       protocols("h2"),
	}
	defer tr.CloseIdleConnections()

	req, err := http.NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

// testServerResponse sets up an idle HTTP/2 connection. The client function should
// write a single request that must be handled by the handler.
func testServerResponse(t testing.TB,
	handler func(http.ResponseWriter, *http.Request) error,
	client func(*serverTester),
) {
	errc := make(chan error, 1)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Body == nil {
			t.Fatal("nil Body")
		}
		err := handler(w, r)
		select {
		case errc <- err:
		default:
			t.Errorf("unexpected duplicate request")
		}
	})
	defer st.Close()

	st.greet()
	client(st)

	if err := <-errc; err != nil {
		t.Fatalf("Error in handler: %v", err)
	}
}

// readBodyHandler returns an http Handler func that reads len(want)
// bytes from r.Body and fails t if the contents read were not
// the value of want.
func readBodyHandler(t testing.TB, want string) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, len(want))
		_, err := io.ReadFull(r.Body, buf)
		if err != nil {
			t.Error(err)
			return
		}
		if string(buf) != want {
			t.Errorf("read %q; want %q", buf, want)
		}
	}
}

func TestServer_MaxDecoderHeaderTableSize(t *testing.T) {
	synctestTest(t, testServer_MaxDecoderHeaderTableSize)
}
func testServer_MaxDecoderHeaderTableSize(t testing.TB) {
	wantHeaderTableSize := uint32(InitialHeaderTableSize * 2)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {}, func(h2 *http.HTTP2Config) {
		h2.MaxDecoderHeaderTableSize = int(wantHeaderTableSize)
	})
	defer st.Close()

	var advHeaderTableSize *uint32
	st.greetAndCheckSettings(func(s Setting) error {
		switch s.ID {
		case SettingHeaderTableSize:
			advHeaderTableSize = &s.Val
		}
		return nil
	})

	if advHeaderTableSize == nil {
		t.Errorf("server didn't advertise a header table size")
	} else if got, want := *advHeaderTableSize, wantHeaderTableSize; got != want {
		t.Errorf("server advertised a header table size of %d, want %d", got, want)
	}
}

func TestServer_MaxEncoderHeaderTableSize(t *testing.T) {
	synctestTest(t, testServer_MaxEncoderHeaderTableSize)
}
func testServer_MaxEncoderHeaderTableSize(t testing.TB) {
	wantHeaderTableSize := uint32(InitialHeaderTableSize / 2)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {}, func(h2 *http.HTTP2Config) {
		h2.MaxEncoderHeaderTableSize = int(wantHeaderTableSize)
	})
	defer st.Close()

	st.greet()

	if got, want := st.sc.TestHPACKEncoder().MaxDynamicTableSize(), wantHeaderTableSize; got != want {
		t.Errorf("server encoder is using a header table size of %d, want %d", got, want)
	}
}

// Issue 12843
func TestServerDoS_MaxHeaderListSize(t *testing.T) { synctestTest(t, testServerDoS_MaxHeaderListSize) }
func testServerDoS_MaxHeaderListSize(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {})
	defer st.Close()

	// shake hands
	frameSize := DefaultMaxReadFrameSize
	var advHeaderListSize *uint32
	st.greetAndCheckSettings(func(s Setting) error {
		switch s.ID {
		case SettingMaxFrameSize:
			if s.Val < MinMaxFrameSize {
				frameSize = MinMaxFrameSize
			} else if s.Val > MaxFrameSize {
				frameSize = MaxFrameSize
			} else {
				frameSize = int(s.Val)
			}
		case SettingMaxHeaderListSize:
			advHeaderListSize = &s.Val
		}
		return nil
	})

	if advHeaderListSize == nil {
		t.Errorf("server didn't advertise a max header list size")
	} else if *advHeaderListSize == 0 {
		t.Errorf("server advertised a max header list size of 0")
	}

	st.encodeHeaderField(":method", "GET")
	st.encodeHeaderField(":path", "/")
	st.encodeHeaderField(":scheme", "https")
	cookie := strings.Repeat("*", 4058)
	st.encodeHeaderField("cookie", cookie)
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.headerBuf.Bytes(),
		EndStream:     true,
		EndHeaders:    false,
	})

	// Capture the short encoding of a duplicate ~4K cookie, now
	// that we've already sent it once.
	st.headerBuf.Reset()
	st.encodeHeaderField("cookie", cookie)

	// Now send 1MB of it.
	const size = 1 << 20
	b := bytes.Repeat(st.headerBuf.Bytes(), size/st.headerBuf.Len())
	for len(b) > 0 {
		chunk := b
		if len(chunk) > frameSize {
			chunk = chunk[:frameSize]
		}
		b = b[len(chunk):]
		st.fr.WriteContinuation(1, len(b) == 0, chunk)
	}

	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: false,
		header: http.Header{
			":status":        []string{"431"},
			"content-type":   []string{"text/html; charset=utf-8"},
			"content-length": []string{"63"},
		},
	})
}

func TestServer_Response_Stream_With_Missing_Trailer(t *testing.T) {
	synctestTest(t, testServer_Response_Stream_With_Missing_Trailer)
}
func testServer_Response_Stream_With_Missing_Trailer(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("Trailer", "test-trailer")
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
		})
		st.wantData(wantData{
			streamID:  1,
			endStream: true,
			size:      0,
		})
	})
}

func TestCompressionErrorOnWrite(t *testing.T) { synctestTest(t, testCompressionErrorOnWrite) }
func testCompressionErrorOnWrite(t testing.TB) {
	const maxStrLen = 8 << 10
	var serverConfig *http.Server
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		// No response body.
	}, func(s *http.Server) {
		serverConfig = s
		serverConfig.MaxHeaderBytes = maxStrLen
	})
	st.addLogFilter("connection error: COMPRESSION_ERROR")
	defer st.Close()
	st.greet()

	maxAllowed := st.sc.TestFramerMaxHeaderStringLen()

	// Crank this up, now that we have a conn connected with the
	// hpack.Decoder's max string length set has been initialized
	// from the earlier low ~8K value. We want this higher so don't
	// hit the max header list size. We only want to test hitting
	// the max string size.
	serverConfig.MaxHeaderBytes = 1 << 20

	// First a request with a header that's exactly the max allowed size
	// for the hpack compression. It's still too long for the header list
	// size, so we'll get the 431 error, but that keeps the compression
	// context still valid.
	hbf := st.encodeHeader("foo", strings.Repeat("a", maxAllowed))

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: hbf,
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: false,
		header: http.Header{
			":status":        []string{"431"},
			"content-type":   []string{"text/html; charset=utf-8"},
			"content-length": []string{"63"},
		},
	})
	df := readFrame[*DataFrame](t, st)
	if !strings.Contains(string(df.Data()), "HTTP Error 431") {
		t.Errorf("Unexpected data body: %q", df.Data())
	}
	if !df.StreamEnded() {
		t.Fatalf("expect data stream end")
	}

	// And now send one that's just one byte too big.
	hbf = st.encodeHeader("bar", strings.Repeat("b", maxAllowed+1))
	st.writeHeaders(HeadersFrameParam{
		StreamID:      3,
		BlockFragment: hbf,
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantGoAway(3, ErrCodeCompression)
}

func TestCompressionErrorOnClose(t *testing.T) { synctestTest(t, testCompressionErrorOnClose) }
func testCompressionErrorOnClose(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		// No response body.
	})
	st.addLogFilter("connection error: COMPRESSION_ERROR")
	defer st.Close()
	st.greet()

	hbf := st.encodeHeader("foo", "bar")
	hbf = hbf[:len(hbf)-1] // truncate one byte from the end, so hpack.Decoder.Close fails.
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: hbf,
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantGoAway(1, ErrCodeCompression)
}

// test that a server handler can read trailers from a client
func TestServerReadsTrailers(t *testing.T) { synctestTest(t, testServerReadsTrailers) }
func testServerReadsTrailers(t testing.TB) {
	const testBody = "some test body"
	writeReq := func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1, // clients send odd numbers
			BlockFragment: st.encodeHeader("trailer", "Foo, Bar", "trailer", "Baz"),
			EndStream:     false,
			EndHeaders:    true,
		})
		st.writeData(1, false, []byte(testBody))
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1, // clients send odd numbers
			BlockFragment: st.encodeHeaderRaw(
				"foo", "foov",
				"bar", "barv",
				"baz", "bazv",
				"surprise", "wasn't declared; shouldn't show up",
			),
			EndStream:  true,
			EndHeaders: true,
		})
	}
	checkReq := func(r *http.Request) {
		wantTrailer := http.Header{
			"Foo": nil,
			"Bar": nil,
			"Baz": nil,
		}
		if !reflect.DeepEqual(r.Trailer, wantTrailer) {
			t.Errorf("initial Trailer = %v; want %v", r.Trailer, wantTrailer)
		}
		slurp, err := io.ReadAll(r.Body)
		if string(slurp) != testBody {
			t.Errorf("read body %q; want %q", slurp, testBody)
		}
		if err != nil {
			t.Fatalf("Body slurp: %v", err)
		}
		wantTrailerAfter := http.Header{
			"Foo": {"foov"},
			"Bar": {"barv"},
			"Baz": {"bazv"},
		}
		if !reflect.DeepEqual(r.Trailer, wantTrailerAfter) {
			t.Errorf("final Trailer = %v; want %v", r.Trailer, wantTrailerAfter)
		}
	}
	testServerRequest(t, writeReq, checkReq)
}

// test that a server handler can send trailers
func TestServerWritesTrailers_WithFlush(t *testing.T) {
	synctestTest(t, func(t testing.TB) {
		testServerWritesTrailers(t, true)
	})
}
func TestServerWritesTrailers_WithoutFlush(t *testing.T) {
	synctestTest(t, func(t testing.TB) {
		testServerWritesTrailers(t, false)
	})
}

func testServerWritesTrailers(t testing.TB, withFlush bool) {
	// See https://httpwg.github.io/specs/rfc7540.html#rfc.section.8.1.3
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("Trailer", "Server-Trailer-A, Server-Trailer-B")
		w.Header().Add("Trailer", "Server-Trailer-C")
		w.Header().Add("Trailer", "Transfer-Encoding, Content-Length, Trailer") // filtered

		// Regular headers:
		w.Header().Set("Foo", "Bar")
		w.Header().Set("Content-Length", "5") // len("Hello")

		io.WriteString(w, "Hello")
		if withFlush {
			w.(http.Flusher).Flush()
		}
		w.Header().Set("Server-Trailer-A", "valuea")
		w.Header().Set("Server-Trailer-C", "valuec") // skipping B
		// After a flush, random keys like Server-Surprise shouldn't show up:
		w.Header().Set("Server-Surpise", "surprise! this isn't predeclared!")
		// But we do permit promoting keys to trailers after a
		// flush if they start with the magic
		// otherwise-invalid "Trailer:" prefix:
		w.Header().Set("Trailer:Post-Header-Trailer", "hi1")
		w.Header().Set("Trailer:post-header-trailer2", "hi2")
		w.Header().Set("Trailer:Range", "invalid")
		w.Header().Set("Trailer:Foo\x01Bogus", "invalid")
		w.Header().Set("Transfer-Encoding", "should not be included; Forbidden by RFC 7230 4.1.2")
		w.Header().Set("Content-Length", "should not be included; Forbidden by RFC 7230 4.1.2")
		w.Header().Set("Trailer", "should not be included; Forbidden by RFC 7230 4.1.2")
		return nil
	}, func(st *serverTester) {
		// Ignore errors from writing invalid trailers.
		st.h1server.ErrorLog = log.New(io.Discard, "", 0)
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status": []string{"200"},
				"foo":     []string{"Bar"},
				"trailer": []string{
					"Server-Trailer-A, Server-Trailer-B",
					"Server-Trailer-C",
					"Transfer-Encoding, Content-Length, Trailer",
				},
				"content-type":   []string{"text/plain; charset=utf-8"},
				"content-length": []string{"5"},
			},
		})
		st.wantData(wantData{
			streamID:  1,
			endStream: false,
			data:      []byte("Hello"),
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: true,
			header: http.Header{
				"post-header-trailer":  []string{"hi1"},
				"post-header-trailer2": []string{"hi2"},
				"server-trailer-a":     []string{"valuea"},
				"server-trailer-c":     []string{"valuec"},
			},
		})
	})
}

func TestServerWritesUndeclaredTrailers(t *testing.T) {
	synctestTest(t, testServerWritesUndeclaredTrailers)
}
func testServerWritesUndeclaredTrailers(t testing.TB) {
	const trailer = "Trailer-Header"
	const value = "hi1"
	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(http.TrailerPrefix+trailer, value)
	})

	tr := &http.Transport{
		TLSClientConfig: tlsConfigInsecure,
		Protocols:       protocols("h2"),
	}
	defer tr.CloseIdleConnections()

	cl := &http.Client{Transport: tr}
	resp, err := cl.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if got, want := resp.Trailer.Get(trailer), value; got != want {
		t.Errorf("trailer %v = %q, want %q", trailer, got, want)
	}
}

// validate transmitted header field names & values
// golang.org/issue/14048
func TestServerDoesntWriteInvalidHeaders(t *testing.T) {
	synctestTest(t, testServerDoesntWriteInvalidHeaders)
}
func testServerDoesntWriteInvalidHeaders(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Add("OK1", "x")
		w.Header().Add("Bad:Colon", "x") // colon (non-token byte) in key
		w.Header().Add("Bad1\x00", "x")  // null in key
		w.Header().Add("Bad2", "x\x00y") // null in value
		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: true,
			header: http.Header{
				":status":        []string{"200"},
				"ok1":            []string{"x"},
				"content-length": []string{"0"},
			},
		})
	})
}

func TestIssue53(t *testing.T) { synctestTest(t, testIssue53) }
func testIssue53(t testing.TB) {
	const data = "PRI * HTTP/2.0\r\n\r\nSM" +
		"\r\n\r\n\x00\x00\x00\x01\ainfinfin\ad"
	st := newServerTester(t, func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte("hello"))
	})

	st.cc.Write([]byte(data))
	st.wantFrameType(FrameSettings)
	st.wantFrameType(FrameWindowUpdate)
	st.wantFrameType(FrameGoAway)
	time.Sleep(GoAwayTimeout)
	st.wantClosed()
}

func TestServerServeNoBannedCiphers(t *testing.T) {
	tests := []struct {
		name      string
		tlsConfig *tls.Config
		wantErr   string
	}{
		{
			name:      "empty CipherSuites",
			tlsConfig: &tls.Config{},
		},
		{
			name: "bad CipherSuites but MinVersion TLS 1.3",
			tlsConfig: &tls.Config{
				MinVersion:   tls.VersionTLS13,
				CipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
			},
		},
		{
			name: "just the required cipher suite",
			tlsConfig: &tls.Config{
				CipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			},
		},
		{
			name: "just the alternative required cipher suite",
			tlsConfig: &tls.Config{
				CipherSuites: []uint16{tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			},
		},
		{
			name: "missing required cipher suite",
			tlsConfig: &tls.Config{
				CipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
			},
			wantErr: "is missing an HTTP/2-required",
		},
		{
			name: "required after bad",
			tlsConfig: &tls.Config{
				CipherSuites: []uint16{tls.TLS_RSA_WITH_RC4_128_SHA, tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			},
		},
		{
			name: "bad after required",
			tlsConfig: &tls.Config{
				CipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, tls.TLS_RSA_WITH_RC4_128_SHA},
			},
		},
	}
	for _, tt := range tests {
		tt.tlsConfig.Certificates = testServerTLSConfig.Certificates

		srv := &http.Server{
			TLSConfig: tt.tlsConfig,
			Protocols: protocols("h2"),
		}

		err := srv.ServeTLS(errListener{}, "", "")
		if (err != net.ErrClosed) != (tt.wantErr != "") {
			if tt.wantErr != "" {
				t.Errorf("%s: success, but want error", tt.name)
			} else {
				t.Errorf("%s: unexpected error: %v", tt.name, err)
			}
		}
		if err != nil && tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr) {
			t.Errorf("%s: err = %v; want substring %q", tt.name, err, tt.wantErr)
		}
		if err == nil && !srv.TLSConfig.PreferServerCipherSuites {
			t.Errorf("%s: PreferServerCipherSuite is false; want true", tt.name)
		}
	}
}

type errListener struct{}

func (li errListener) Accept() (net.Conn, error) { return nil, net.ErrClosed }
func (li errListener) Close() error              { return nil }
func (li errListener) Addr() net.Addr            { return nil }

func TestServerNoAutoContentLengthOnHead(t *testing.T) {
	synctestTest(t, testServerNoAutoContentLengthOnHead)
}
func testServerNoAutoContentLengthOnHead(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		// No response body. (or smaller than one frame)
	})
	defer st.Close()
	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1, // clients send odd numbers
		BlockFragment: st.encodeHeader(":method", "HEAD"),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
		header: http.Header{
			":status": []string{"200"},
		},
	})
}

// golang.org/issue/13495
func TestServerNoDuplicateContentType(t *testing.T) {
	synctestTest(t, testServerNoDuplicateContentType)
}
func testServerNoDuplicateContentType(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header()["Content-Type"] = []string{""}
		fmt.Fprintf(w, "<html><head></head><body>hi</body></html>")
	})
	defer st.Close()
	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: false,
		header: http.Header{
			":status":        []string{"200"},
			"content-type":   []string{""},
			"content-length": []string{"41"},
		},
	})
}

func TestServerContentLengthCanBeDisabled(t *testing.T) {
	synctestTest(t, testServerContentLengthCanBeDisabled)
}
func testServerContentLengthCanBeDisabled(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header()["Content-Length"] = nil
		fmt.Fprintf(w, "OK")
	})
	defer st.Close()
	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: false,
		header: http.Header{
			":status":      []string{"200"},
			"content-type": []string{"text/plain; charset=utf-8"},
		},
	})
}

// golang.org/issue/14214
func TestServer_Rejects_ConnHeaders(t *testing.T) { synctestTest(t, testServer_Rejects_ConnHeaders) }
func testServer_Rejects_ConnHeaders(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("should not get to Handler")
	})
	defer st.Close()
	st.greet()
	st.bodylessReq1("connection", "foo")
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: false,
		header: http.Header{
			":status":                []string{"400"},
			"content-type":           []string{"text/plain; charset=utf-8"},
			"x-content-type-options": []string{"nosniff"},
			"content-length":         []string{"51"},
		},
	})
}

type hpackEncoder struct {
	enc *hpack.Encoder
	buf bytes.Buffer
}

func (he *hpackEncoder) encodeHeaderRaw(t testing.TB, headers ...string) []byte {
	if len(headers)%2 == 1 {
		panic("odd number of kv args")
	}
	he.buf.Reset()
	if he.enc == nil {
		he.enc = hpack.NewEncoder(&he.buf)
	}
	for len(headers) > 0 {
		k, v := headers[0], headers[1]
		err := he.enc.WriteField(hpack.HeaderField{Name: k, Value: v})
		if err != nil {
			t.Fatalf("HPACK encoding error for %q/%q: %v", k, v, err)
		}
		headers = headers[2:]
	}
	return he.buf.Bytes()
}

// golang.org/issue/14030
func TestExpect100ContinueAfterHandlerWrites(t *testing.T) {
	synctestTest(t, testExpect100ContinueAfterHandlerWrites)
}
func testExpect100ContinueAfterHandlerWrites(t testing.TB) {
	const msg = "Hello"
	const msg2 = "World"

	doRead := make(chan bool, 1)
	defer close(doRead) // fallback cleanup

	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, msg)
		w.(http.Flusher).Flush()

		// Do a read, which might force a 100-continue status to be sent.
		<-doRead
		r.Body.Read(make([]byte, 10))

		io.WriteString(w, msg2)
	})

	tr := &http.Transport{
		TLSClientConfig: tlsConfigInsecure,
		Protocols:       protocols("h2"),
	}
	defer tr.CloseIdleConnections()

	req, _ := http.NewRequest("POST", ts.URL, io.LimitReader(neverEnding('A'), 2<<20))
	req.Header.Set("Expect", "100-continue")

	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	buf := make([]byte, len(msg))
	if _, err := io.ReadFull(res.Body, buf); err != nil {
		t.Fatal(err)
	}
	if string(buf) != msg {
		t.Fatalf("msg = %q; want %q", buf, msg)
	}

	doRead <- true

	if _, err := io.ReadFull(res.Body, buf); err != nil {
		t.Fatal(err)
	}
	if string(buf) != msg2 {
		t.Fatalf("second msg = %q; want %q", buf, msg2)
	}
}

type funcReader func([]byte) (n int, err error)

func (f funcReader) Read(p []byte) (n int, err error) { return f(p) }

// golang.org/issue/16481 -- return flow control when streams close with unread data.
// (The Server version of the bug. See also TestUnreadFlowControlReturned_Transport)
func TestUnreadFlowControlReturned_Server(t *testing.T) {
	for _, tt := range []struct {
		name  string
		reqFn func(r *http.Request)
	}{
		{
			"body-open",
			func(r *http.Request) {},
		},
		{
			"body-closed",
			func(r *http.Request) {
				r.Body.Close()
			},
		},
		{
			"read-1-byte-and-close",
			func(r *http.Request) {
				b := make([]byte, 1)
				r.Body.Read(b)
				r.Body.Close()
			},
		},
	} {
		synctestSubtest(t, tt.name, func(t testing.TB) {
			unblock := make(chan bool, 1)
			defer close(unblock)

			ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				// Don't read the 16KB request body. Wait until the client's
				// done sending it and then return. This should cause the Server
				// to then return those 16KB of flow control to the client.
				tt.reqFn(r)
				<-unblock
			})

			tr := &http.Transport{
				TLSClientConfig: tlsConfigInsecure,
				Protocols:       protocols("h2"),
			}
			defer tr.CloseIdleConnections()

			// This previously hung on the 4th iteration.
			iters := 100
			if testing.Short() {
				iters = 20
			}
			for i := 0; i < iters; i++ {
				body := io.MultiReader(
					io.LimitReader(neverEnding('A'), 16<<10),
					funcReader(func([]byte) (n int, err error) {
						unblock <- true
						return 0, io.EOF
					}),
				)
				req, _ := http.NewRequest("POST", ts.URL, body)
				res, err := tr.RoundTrip(req)
				if err != nil {
					t.Fatal(tt.name, err)
				}
				res.Body.Close()
			}
		})
	}
}

func TestServerReturnsStreamAndConnFlowControlOnBodyClose(t *testing.T) {
	synctestTest(t, testServerReturnsStreamAndConnFlowControlOnBodyClose)
}
func testServerReturnsStreamAndConnFlowControlOnBodyClose(t testing.TB) {
	unblockHandler := make(chan struct{})
	defer close(unblockHandler)

	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		r.Body.Close()
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		<-unblockHandler
	})
	defer st.Close()

	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndHeaders:    true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: false,
	})
	const size = InflowMinRefresh // enough to trigger flow control return
	st.writeData(1, false, make([]byte, size))
	st.wantWindowUpdate(0, size) // conn-level flow control is returned
	unblockHandler <- struct{}{}
	st.wantData(wantData{
		streamID:  1,
		endStream: true,
	})
}

func TestServerIdleTimeout(t *testing.T) { synctestTest(t, testServerIdleTimeout) }
func testServerIdleTimeout(t testing.TB) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
	}, func(s *http.Server) {
		s.IdleTimeout = 500 * time.Millisecond
	})
	defer st.Close()

	st.greet()
	st.advance(500 * time.Millisecond)
	st.wantGoAway(0, ErrCodeNo)
}

func TestServerIdleTimeout_AfterRequest(t *testing.T) {
	synctestTest(t, testServerIdleTimeout_AfterRequest)
}
func testServerIdleTimeout_AfterRequest(t testing.TB) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	const (
		requestTimeout = 2 * time.Second
		idleTimeout    = 1 * time.Second
	)

	var st *serverTester
	st = newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(requestTimeout)
	}, func(s *http.Server) {
		s.IdleTimeout = idleTimeout
	})
	defer st.Close()

	st.greet()

	// Send a request which takes twice the timeout. Verifies the
	// idle timeout doesn't fire while we're in a request:
	st.bodylessReq1()
	st.advance(requestTimeout)
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
	})

	// But the idle timeout should be rearmed after the request
	// is done:
	st.advance(idleTimeout)
	st.wantGoAway(1, ErrCodeNo)
}

// grpc-go closes the Request.Body currently with a Read.
// Verify that it doesn't race.
// See https://github.com/grpc/grpc-go/pull/938
func TestRequestBodyReadCloseRace(t *testing.T) { synctestTest(t, testRequestBodyReadCloseRace) }
func testRequestBodyReadCloseRace(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		go r.Body.Close()
		io.Copy(io.Discard, r.Body)
	})
	st.greet()

	data := make([]byte, 1024)
	for i := range 100 {
		streamID := uint32(1 + (i * 2)) // clients send odd numbers
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader(),
			EndHeaders:    true,
		})
		st.writeData(1, false, data)

		for {
			// Look for a RST_STREAM frame.
			// Skip over anything else (HEADERS and WINDOW_UPDATE).
			fr := st.readFrame()
			if fr == nil {
				t.Fatalf("got no RSTStreamFrame, want one")
			}
			rst, ok := fr.(*RSTStreamFrame)
			if !ok {
				continue
			}
			// We can get NO or STREAM_CLOSED depending on scheduling.
			if rst.ErrCode != ErrCodeNo && rst.ErrCode != ErrCodeStreamClosed {
				t.Fatalf("got RSTStreamFrame with error code %v, want ErrCodeNo or ErrCodeStreamClosed", rst.ErrCode)
			}
			break
		}
	}
}

func TestIssue20704Race(t *testing.T) { synctestTest(t, testIssue20704Race) }
func testIssue20704Race(t testing.TB) {
	if testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		t.Skip("skipping in short mode")
	}
	const (
		itemSize  = 1 << 10
		itemCount = 100
	)

	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		for range itemCount {
			_, err := w.Write(make([]byte, itemSize))
			if err != nil {
				return
			}
		}
	})

	tr := &http.Transport{
		TLSClientConfig: tlsConfigInsecure,
		Protocols:       protocols("h2"),
	}
	defer tr.CloseIdleConnections()
	cl := &http.Client{Transport: tr}

	for range 1000 {
		resp, err := cl.Get(ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		// Force a RST stream to the server by closing without
		// reading the body:
		resp.Body.Close()
	}
}

func TestServer_Rejects_TooSmall(t *testing.T) { synctestTest(t, testServer_Rejects_TooSmall) }
func testServer_Rejects_TooSmall(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		io.ReadAll(r.Body)
		return nil
	}, func(st *serverTester) {
		st.writeHeaders(HeadersFrameParam{
			StreamID: 1, // clients send odd numbers
			BlockFragment: st.encodeHeader(
				":method", "POST",
				"content-length", "4",
			),
			EndStream:  false, // to say DATA frames are coming
			EndHeaders: true,
		})
		st.writeData(1, true, []byte("12345"))
		st.wantRSTStream(1, ErrCodeProtocol)
		st.wantConnFlowControlConsumed(0)
	})
}

// Tests that a handler setting "Connection: close" results in a GOAWAY being sent,
// and the connection still completing.
func TestServerHandlerConnectionClose(t *testing.T) {
	synctestTest(t, testServerHandlerConnectionClose)
}
func testServerHandlerConnectionClose(t testing.TB) {
	unblockHandler := make(chan bool, 1)
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.Header().Set("Connection", "close")
		w.Header().Set("Foo", "bar")
		w.(http.Flusher).Flush()
		<-unblockHandler
		return nil
	}, func(st *serverTester) {
		defer close(unblockHandler) // backup; in case of errors
		st.writeHeaders(HeadersFrameParam{
			StreamID:      1,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
		var sawGoAway bool
		var sawRes bool
		var sawWindowUpdate bool
		for {
			f := st.readFrame()
			if f == nil {
				break
			}
			switch f := f.(type) {
			case *GoAwayFrame:
				sawGoAway = true
				if f.LastStreamID != 1 || f.ErrCode != ErrCodeNo {
					t.Errorf("unexpected GOAWAY frame: %v", SummarizeFrame(f))
				}
				// Create a stream and reset it.
				// The server should ignore the stream.
				st.writeHeaders(HeadersFrameParam{
					StreamID:      3,
					BlockFragment: st.encodeHeader(),
					EndStream:     false,
					EndHeaders:    true,
				})
				st.fr.WriteRSTStream(3, ErrCodeCancel)
				// Create a stream and send data to it.
				// The server should return flow control, even though it
				// does not process the stream.
				st.writeHeaders(HeadersFrameParam{
					StreamID:      5,
					BlockFragment: st.encodeHeader(),
					EndStream:     false,
					EndHeaders:    true,
				})
				// Write enough data to trigger a window update.
				st.writeData(5, true, make([]byte, 1<<19))
			case *HeadersFrame:
				goth := st.decodeHeader(f.HeaderBlockFragment())
				wanth := [][2]string{
					{":status", "200"},
					{"foo", "bar"},
				}
				if !reflect.DeepEqual(goth, wanth) {
					t.Errorf("got headers %v; want %v", goth, wanth)
				}
				sawRes = true
			case *DataFrame:
				if f.StreamID != 1 || !f.StreamEnded() || len(f.Data()) != 0 {
					t.Errorf("unexpected DATA frame: %v", SummarizeFrame(f))
				}
			case *WindowUpdateFrame:
				if !sawGoAway {
					t.Errorf("unexpected WINDOW_UPDATE frame: %v", SummarizeFrame(f))
					return
				}
				if f.StreamID != 0 {
					st.t.Fatalf("WindowUpdate StreamID = %d; want 5", f.FrameHeader.StreamID)
					return
				}
				sawWindowUpdate = true
				unblockHandler <- true
				st.sync()
				st.advance(GoAwayTimeout)
			default:
				t.Logf("unexpected frame: %v", SummarizeFrame(f))
			}
		}
		if !sawGoAway {
			t.Errorf("didn't see GOAWAY")
		}
		if !sawRes {
			t.Errorf("didn't see response")
		}
		if !sawWindowUpdate {
			t.Errorf("didn't see WINDOW_UPDATE")
		}
	})
}

func TestServer_Headers_HalfCloseRemote(t *testing.T) {
	synctestTest(t, testServer_Headers_HalfCloseRemote)
}
func testServer_Headers_HalfCloseRemote(t testing.TB) {
	var st *serverTester
	writeData := make(chan bool)
	writeHeaders := make(chan bool)
	leaveHandler := make(chan bool)
	st = newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		if !st.streamExists(1) {
			t.Errorf("stream 1 does not exist in handler")
		}
		if got, want := st.streamState(1), StateOpen; got != want {
			t.Errorf("in handler, state is %v; want %v", got, want)
		}
		writeData <- true
		if n, err := r.Body.Read(make([]byte, 1)); n != 0 || err != io.EOF {
			t.Errorf("body read = %d, %v; want 0, EOF", n, err)
		}
		if got, want := st.streamState(1), StateHalfClosedRemote; got != want {
			t.Errorf("in handler, state is %v; want %v", got, want)
		}
		writeHeaders <- true

		<-leaveHandler
	})
	st.greet()

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     false, // keep it open
		EndHeaders:    true,
	})
	<-writeData
	st.writeData(1, true, nil)

	<-writeHeaders

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     false, // keep it open
		EndHeaders:    true,
	})

	defer close(leaveHandler)

	st.wantRSTStream(1, ErrCodeStreamClosed)
}

func TestServerGracefulShutdown(t *testing.T) { synctestTest(t, testServerGracefulShutdown) }
func testServerGracefulShutdown(t testing.TB) {
	handlerDone := make(chan struct{})
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		<-handlerDone
		w.Header().Set("x-foo", "bar")
	})
	defer st.Close()

	st.greet()
	st.bodylessReq1()

	st.sync()

	shutdownc := make(chan struct{})
	go func() {
		defer close(shutdownc)
		st.h1server.Shutdown(context.Background())
	}()

	st.wantGoAway(1, ErrCodeNo)

	close(handlerDone)
	st.sync()

	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
		header: http.Header{
			":status":        []string{"200"},
			"x-foo":          []string{"bar"},
			"content-length": []string{"0"},
		},
	})

	n, err := st.cc.Read([]byte{0})
	if n != 0 || err == nil {
		t.Errorf("Read = %v, %v; want 0, non-nil", n, err)
	}

	// Shutdown happens after GoAwayTimeout and net/http.Server polling delay.
	<-shutdownc
}

// Issue 31753: don't sniff when Content-Encoding is set
func TestContentEncodingNoSniffing(t *testing.T) {
	type resp struct {
		name string
		body []byte
		// setting Content-Encoding as an interface instead of a string
		// directly, so as to differentiate between 3 states:
		//    unset, empty string "" and set string "foo/bar".
		contentEncoding any
		wantContentType string
	}

	resps := []*resp{
		{
			name:            "gzip content-encoding, gzipped", // don't sniff.
			contentEncoding: "application/gzip",
			wantContentType: "",
			body: func() []byte {
				buf := new(bytes.Buffer)
				gzw := gzip.NewWriter(buf)
				gzw.Write([]byte("doctype html><p>Hello</p>"))
				gzw.Close()
				return buf.Bytes()
			}(),
		},
		{
			name:            "zlib content-encoding, zlibbed", // don't sniff.
			contentEncoding: "application/zlib",
			wantContentType: "",
			body: func() []byte {
				buf := new(bytes.Buffer)
				zw := zlib.NewWriter(buf)
				zw.Write([]byte("doctype html><p>Hello</p>"))
				zw.Close()
				return buf.Bytes()
			}(),
		},
		{
			name:            "no content-encoding", // must sniff.
			wantContentType: "application/x-gzip",
			body: func() []byte {
				buf := new(bytes.Buffer)
				gzw := gzip.NewWriter(buf)
				gzw.Write([]byte("doctype html><p>Hello</p>"))
				gzw.Close()
				return buf.Bytes()
			}(),
		},
		{
			name:            "phony content-encoding", // don't sniff.
			contentEncoding: "foo/bar",
			body:            []byte("doctype html><p>Hello</p>"),
		},
		{
			name:            "empty but set content-encoding",
			contentEncoding: "",
			wantContentType: "audio/mpeg",
			body:            []byte("ID3"),
		},
	}

	for _, tt := range resps {
		synctestSubtest(t, tt.name, func(t testing.TB) {
			ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				if tt.contentEncoding != nil {
					w.Header().Set("Content-Encoding", tt.contentEncoding.(string))
				}
				w.Write(tt.body)
			})

			tr := &http.Transport{
				TLSClientConfig: tlsConfigInsecure,
				Protocols:       protocols("h2"),
			}
			defer tr.CloseIdleConnections()

			req, _ := http.NewRequest("GET", ts.URL, nil)
			res, err := tr.RoundTrip(req)
			if err != nil {
				t.Fatalf("GET %s: %v", ts.URL, err)
			}
			defer res.Body.Close()

			g := res.Header.Get("Content-Encoding")
			t.Logf("%s: Content-Encoding: %s", ts.URL, g)

			if w := tt.contentEncoding; g != w {
				if w != nil { // The case where contentEncoding was set explicitly.
					t.Errorf("Content-Encoding mismatch\n\tgot:  %q\n\twant: %q", g, w)
				} else if g != "" { // "" should be the equivalent when the contentEncoding is unset.
					t.Errorf("Unexpected Content-Encoding %q", g)
				}
			}

			g = res.Header.Get("Content-Type")
			if w := tt.wantContentType; g != w {
				t.Errorf("Content-Type mismatch\n\tgot:  %q\n\twant: %q", g, w)
			}
			t.Logf("%s: Content-Type: %s", ts.URL, g)
		})
	}
}

func TestServerWindowUpdateOnBodyClose(t *testing.T) {
	synctestTest(t, testServerWindowUpdateOnBodyClose)
}
func testServerWindowUpdateOnBodyClose(t testing.TB) {
	const windowSize = 65535 * 2
	content := make([]byte, windowSize)
	errc := make(chan error)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 4)
		n, err := io.ReadFull(r.Body, buf)
		if err != nil {
			errc <- err
			return
		}
		if n != len(buf) {
			errc <- fmt.Errorf("too few bytes read: %d", n)
			return
		}
		r.Body.Close()
		errc <- nil
	}, func(h2 *http.HTTP2Config) {
		h2.MaxReceiveBufferPerConnection = windowSize
		h2.MaxReceiveBufferPerStream = windowSize
	})
	defer st.Close()

	st.greet()
	st.writeHeaders(HeadersFrameParam{
		StreamID: 1, // clients send odd numbers
		BlockFragment: st.encodeHeader(
			":method", "POST",
			"content-length", strconv.Itoa(len(content)),
		),
		EndStream:  false, // to say DATA frames are coming
		EndHeaders: true,
	})
	st.writeData(1, false, content[:windowSize/2])
	if err := <-errc; err != nil {
		t.Fatal(err)
	}

	// Wait for flow control credit for the portion of the request written so far.
	increments := windowSize / 2
	for {
		f := st.readFrame()
		if f == nil {
			break
		}
		if wu, ok := f.(*WindowUpdateFrame); ok && wu.StreamID == 0 {
			increments -= int(wu.Increment)
			if increments == 0 {
				break
			}
		}
	}

	// Writing data after the stream is reset immediately returns flow control credit.
	st.writeData(1, false, content[windowSize/2:])
	st.wantWindowUpdate(0, windowSize/2)
}

func TestNoErrorLoggedOnPostAfterGOAWAY(t *testing.T) {
	synctestTest(t, testNoErrorLoggedOnPostAfterGOAWAY)
}
func testNoErrorLoggedOnPostAfterGOAWAY(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {})
	defer st.Close()

	st.greet()

	content := "some content"
	st.writeHeaders(HeadersFrameParam{
		StreamID: 1,
		BlockFragment: st.encodeHeader(
			":method", "POST",
			"content-length", strconv.Itoa(len(content)),
		),
		EndStream:  false,
		EndHeaders: true,
	})
	st.wantHeaders(wantHeader{
		streamID:  1,
		endStream: true,
	})

	st.sc.StartGracefulShutdown()
	st.wantRSTStream(1, ErrCodeNo)
	st.wantGoAway(1, ErrCodeNo)

	st.writeData(1, true, []byte(content))
	st.Close()

	if bytes.Contains(st.serverLogBuf.Bytes(), []byte("PROTOCOL_ERROR")) {
		t.Error("got protocol error")
	}
}

func TestServerSendsProcessing(t *testing.T) { synctestTest(t, testServerSendsProcessing) }
func testServerSendsProcessing(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		w.WriteHeader(http.StatusProcessing)
		w.Write([]byte("stuff"))

		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status": []string{"102"},
			},
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status":        []string{"200"},
				"content-type":   []string{"text/plain; charset=utf-8"},
				"content-length": []string{"5"},
			},
		})
	})
}

func TestServerSendsEarlyHints(t *testing.T) { synctestTest(t, testServerSendsEarlyHints) }
func testServerSendsEarlyHints(t testing.TB) {
	testServerResponse(t, func(w http.ResponseWriter, r *http.Request) error {
		h := w.Header()
		h.Add("Content-Length", "123")
		h.Add("Link", "</style.css>; rel=preload; as=style")
		h.Add("Link", "</script.js>; rel=preload; as=script")
		w.WriteHeader(http.StatusEarlyHints)

		h.Add("Link", "</foo.js>; rel=preload; as=script")
		w.WriteHeader(http.StatusEarlyHints)

		w.Write([]byte("stuff"))

		return nil
	}, func(st *serverTester) {
		getSlash(st)
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status": []string{"103"},
				"link": []string{
					"</style.css>; rel=preload; as=style",
					"</script.js>; rel=preload; as=script",
				},
			},
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status": []string{"103"},
				"link": []string{
					"</style.css>; rel=preload; as=style",
					"</script.js>; rel=preload; as=script",
					"</foo.js>; rel=preload; as=script",
				},
			},
		})
		st.wantHeaders(wantHeader{
			streamID:  1,
			endStream: false,
			header: http.Header{
				":status": []string{"200"},
				"link": []string{
					"</style.css>; rel=preload; as=style",
					"</script.js>; rel=preload; as=script",
					"</foo.js>; rel=preload; as=script",
				},
				"content-type":   []string{"text/plain; charset=utf-8"},
				"content-length": []string{"123"},
			},
		})
	})
}

func TestProtocolErrorAfterGoAway(t *testing.T) { synctestTest(t, testProtocolErrorAfterGoAway) }
func testProtocolErrorAfterGoAway(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
	})
	defer st.Close()

	st.greet()
	content := "some content"
	st.writeHeaders(HeadersFrameParam{
		StreamID: 1,
		BlockFragment: st.encodeHeader(
			":method", "POST",
			"content-length", strconv.Itoa(len(content)),
		),
		EndStream:  false,
		EndHeaders: true,
	})
	st.writeData(1, false, []byte(content[:5]))

	// Send a GOAWAY with ErrCodeNo, followed by a bogus window update.
	// The server should close the connection.
	if err := st.fr.WriteGoAway(1, ErrCodeNo, nil); err != nil {
		t.Fatal(err)
	}
	if err := st.fr.WriteWindowUpdate(0, 1<<31-1); err != nil {
		t.Fatal(err)
	}

	st.advance(GoAwayTimeout)
	st.wantGoAway(1, ErrCodeNo)
	st.wantClosed()
}

func TestServerInitialFlowControlWindow(t *testing.T) {
	for _, want := range []int32{
		65535,
		1 << 19,
		1 << 21,
		// For MaxUploadBufferPerConnection values in the range
		// (65535, 65535*2), we don't send an initial WINDOW_UPDATE
		// because we only send flow control when the window drops
		// below half of the maximum. Perhaps it would be nice to
		// test this case, but we currently do not.
		65535 * 2,
	} {
		synctestSubtest(t, fmt.Sprint(want), func(t testing.TB) {

			st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
			}, func(h2 *http.HTTP2Config) {
				h2.MaxReceiveBufferPerConnection = int(want)
			})
			st.writePreface()
			st.writeSettings()
			_ = readFrame[*SettingsFrame](t, st)
			st.writeSettingsAck()
			st.writeHeaders(HeadersFrameParam{
				StreamID:      1,
				BlockFragment: st.encodeHeader(),
				EndStream:     true,
				EndHeaders:    true,
			})
			window := 65535
		Frames:
			for {
				f := st.readFrame()
				switch f := f.(type) {
				case *WindowUpdateFrame:
					if f.FrameHeader.StreamID != 0 {
						t.Errorf("WindowUpdate StreamID = %d; want 0", f.FrameHeader.StreamID)
						return
					}
					window += int(f.Increment)
				case *HeadersFrame:
					break Frames
				case nil:
					break Frames
				default:
				}
			}
			if window != int(want) {
				t.Errorf("got initial flow control window = %v, want %v", window, want)
			}
		})
	}
}

// TestServerWriteDoesNotRetainBufferAfterReturn checks for access to
// the slice passed to ResponseWriter.Write after Write returns.
//
// Terminating the request stream on the client causes Write to return.
// We should not access the slice after this point.
func TestServerWriteDoesNotRetainBufferAfterReturn(t *testing.T) {
	synctestTest(t, testServerWriteDoesNotRetainBufferAfterReturn)
}
func testServerWriteDoesNotRetainBufferAfterReturn(t testing.TB) {
	donec := make(chan struct{})
	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		defer close(donec)
		buf := make([]byte, 1<<20)
		var i byte
		for {
			i++
			_, err := w.Write(buf)
			for j := range buf {
				buf[j] = byte(i) // trigger race detector
			}
			if err != nil {
				return
			}
		}
	})

	tr := &http.Transport{
		TLSClientConfig: tlsConfigInsecure,
		Protocols:       protocols("h2"),
	}
	defer tr.CloseIdleConnections()

	req, _ := http.NewRequest("GET", ts.URL, nil)
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	<-donec
}

// TestServerWriteDoesNotRetainBufferAfterServerClose checks for access to
// the slice passed to ResponseWriter.Write after Write returns.
//
// Shutting down the Server causes Write to return.
// We should not access the slice after this point.
func TestServerWriteDoesNotRetainBufferAfterServerClose(t *testing.T) {
	synctestTest(t, testServerWriteDoesNotRetainBufferAfterServerClose)
}
func testServerWriteDoesNotRetainBufferAfterServerClose(t testing.TB) {
	donec := make(chan struct{}, 1)
	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		donec <- struct{}{}
		defer close(donec)
		buf := make([]byte, 1<<20)
		var i byte
		for {
			i++
			_, err := w.Write(buf)
			for j := range buf {
				buf[j] = byte(i)
			}
			if err != nil {
				return
			}
		}
	})

	tr := &http.Transport{
		TLSClientConfig: tlsConfigInsecure,
		Protocols:       protocols("h2"),
	}
	defer tr.CloseIdleConnections()

	req, _ := http.NewRequest("GET", ts.URL, nil)
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	<-donec
	ts.Config.Close()
	<-donec
}

func TestServerMaxHandlerGoroutines(t *testing.T) { synctestTest(t, testServerMaxHandlerGoroutines) }
func testServerMaxHandlerGoroutines(t testing.TB) {
	const maxHandlers = 10
	handlerc := make(chan chan bool)
	donec := make(chan struct{})
	defer close(donec)
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		stopc := make(chan bool, 1)
		select {
		case handlerc <- stopc:
		case <-donec:
		}
		select {
		case shouldPanic := <-stopc:
			if shouldPanic {
				panic(http.ErrAbortHandler)
			}
		case <-donec:
		}
	}, func(h2 *http.HTTP2Config) {
		h2.MaxConcurrentStreams = maxHandlers
	})
	defer st.Close()

	st.greet()

	// Make maxHandlers concurrent requests.
	// Reset them all, but only after the handler goroutines have started.
	var stops []chan bool
	streamID := uint32(1)
	for range maxHandlers {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
		stops = append(stops, <-handlerc)
		st.fr.WriteRSTStream(streamID, ErrCodeCancel)
		streamID += 2
	}

	// Start another request, and immediately reset it.
	st.writeHeaders(HeadersFrameParam{
		StreamID:      streamID,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	st.fr.WriteRSTStream(streamID, ErrCodeCancel)
	streamID += 2

	// Start another two requests. Don't reset these.
	for range 2 {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
		streamID += 2
	}

	// The initial maxHandlers handlers are still executing,
	// so the last two requests don't start any new handlers.
	select {
	case <-handlerc:
		t.Errorf("handler unexpectedly started while maxHandlers are already running")
	case <-time.After(1 * time.Millisecond):
	}

	// Tell two handlers to exit.
	// The pending requests which weren't reset start handlers.
	stops[0] <- false // normal exit
	stops[1] <- true  // panic
	stops = stops[2:]
	stops = append(stops, <-handlerc)
	stops = append(stops, <-handlerc)

	// Make a bunch more requests.
	// Eventually, the server tells us to go away.
	for range 5 * maxHandlers {
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
		st.fr.WriteRSTStream(streamID, ErrCodeCancel)
		streamID += 2
	}
	fr := readFrame[*GoAwayFrame](t, st)
	if fr.ErrCode != ErrCodeEnhanceYourCalm {
		t.Errorf("err code = %v; want %v", fr.ErrCode, ErrCodeEnhanceYourCalm)
	}

	for _, s := range stops {
		close(s)
	}
}

func TestServerContinuationFlood(t *testing.T) { synctestTest(t, testServerContinuationFlood) }
func testServerContinuationFlood(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header)
	}, func(s *http.Server) {
		s.MaxHeaderBytes = 4096
	})
	defer st.Close()

	st.greet()

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
	})
	for i := range 1000 {
		st.fr.WriteContinuation(1, false, st.encodeHeaderRaw(
			fmt.Sprintf("x-%v", i), "1234567890",
		))
	}
	st.fr.WriteContinuation(1, true, st.encodeHeaderRaw(
		"x-last-header", "1",
	))

	for {
		f := st.readFrame()
		if f == nil {
			break
		}
		switch f := f.(type) {
		case *HeadersFrame:
			t.Fatalf("received HEADERS frame; want GOAWAY and a closed connection")
		case *GoAwayFrame:
			// We might not see the GOAWAY (see below), but if we do it should
			// indicate that the server processed this request so the client doesn't
			// attempt to retry it.
			if got, want := f.LastStreamID, uint32(1); got != want {
				t.Errorf("received GOAWAY with LastStreamId %v, want %v", got, want)
			}

		}
	}
	// We expect to have seen a GOAWAY before the connection closes,
	// but the server will close the connection after one second
	// whether or not it has finished sending the GOAWAY. On windows-amd64-race
	// builders, this fairly consistently results in the connection closing without
	// the GOAWAY being sent.
	//
	// Since the server's behavior is inherently racy here and the important thing
	// is that the connection is closed, don't check for the GOAWAY having been sent.
}

func TestServerContinuationAfterInvalidHeader(t *testing.T) {
	synctestTest(t, testServerContinuationAfterInvalidHeader)
}
func testServerContinuationAfterInvalidHeader(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header)
	})
	defer st.Close()

	st.greet()

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
	})
	st.fr.WriteContinuation(1, false, st.encodeHeaderRaw(
		"x-invalid-header", "\x00",
	))
	st.fr.WriteContinuation(1, true, st.encodeHeaderRaw(
		"x-valid-header", "1",
	))

	var sawGoAway bool
	for {
		f := st.readFrame()
		if f == nil {
			break
		}
		switch f.(type) {
		case *GoAwayFrame:
			sawGoAway = true
		case *HeadersFrame:
			t.Fatalf("received HEADERS frame; want GOAWAY")
		}
	}
	if !sawGoAway {
		t.Errorf("connection closed with no GOAWAY frame; want one")
	}
}

// Issue 67036: A stream error should result in the handler's request context being canceled.
func TestServerRequestCancelOnError(t *testing.T) { synctestTest(t, testServerRequestCancelOnError) }
func testServerRequestCancelOnError(t testing.TB) {
	recvc := make(chan struct{}) // handler has started
	donec := make(chan struct{}) // handler has finished
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		close(recvc)
		<-r.Context().Done()
		close(donec)
	})
	defer st.Close()

	st.greet()

	// Client sends request headers, handler starts.
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	<-recvc

	// Client sends an invalid second set of request headers.
	// The stream is reset.
	// The handler's context is canceled, and the handler exits.
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})
	<-donec
}

func TestServerSetReadWriteDeadlineRace(t *testing.T) {
	synctestTest(t, testServerSetReadWriteDeadlineRace)
}
func testServerSetReadWriteDeadlineRace(t testing.TB) {
	ts := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		ctl := http.NewResponseController(w)
		ctl.SetReadDeadline(time.Now().Add(3600 * time.Second))
		ctl.SetWriteDeadline(time.Now().Add(3600 * time.Second))
	})
	resp, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
}

func TestServerWriteByteTimeout(t *testing.T) { synctestTest(t, testServerWriteByteTimeout) }
func testServerWriteByteTimeout(t testing.TB) {
	const timeout = 1 * time.Second
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, 100))
	}, func(s *http.Server) {
		// Use unencrypted HTTP/2, so a byte written by the server corresponds
		// to a byte read by the test. Using TLS adds another layer of buffering
		// and timeout management, which aren't really relevant to the test.
		s.Protocols = protocols("h2c")
	}, func(h2 *http.HTTP2Config) {
		h2.WriteByteTimeout = timeout
	})
	st.greet()

	st.cc.(*synctestNetConn).SetReadBufferSize(1) // write one byte at a time
	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     true,
		EndHeaders:    true,
	})

	// Read a few bytes, staying just under WriteByteTimeout.
	for i := range 10 {
		st.advance(timeout - 1)
		if n, err := st.cc.Read(make([]byte, 1)); n != 1 || err != nil {
			t.Fatalf("read %v: %v, %v; want 1, nil", i, n, err)
		}
	}

	// Wait for WriteByteTimeout.
	// The connection should close.
	st.advance(1 * time.Second) // timeout after writing one byte
	st.advance(1 * time.Second) // timeout after failing to write any more bytes
	st.wantClosed()
}

func TestServerPingSent(t *testing.T) { synctestTest(t, testServerPingSent) }
func testServerPingSent(t testing.TB) {
	const sendPingTimeout = 15 * time.Second
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
	}, func(h2 *http.HTTP2Config) {
		h2.SendPingTimeout = sendPingTimeout
	})
	st.greet()

	st.wantIdle()

	st.advance(sendPingTimeout)
	_ = readFrame[*PingFrame](t, st)
	st.wantIdle()

	st.advance(14 * time.Second)
	st.wantIdle()
	st.advance(1 * time.Second)
	st.wantClosed()
}

func TestServerPingResponded(t *testing.T) { synctestTest(t, testServerPingResponded) }
func testServerPingResponded(t testing.TB) {
	const sendPingTimeout = 15 * time.Second
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
	}, func(h2 *http.HTTP2Config) {
		h2.SendPingTimeout = sendPingTimeout
	})
	st.greet()

	st.wantIdle()

	st.advance(sendPingTimeout)
	pf := readFrame[*PingFrame](t, st)
	st.wantIdle()

	st.advance(14 * time.Second)
	st.wantIdle()

	st.writePing(true, pf.Data)

	st.advance(2 * time.Second)
	st.wantIdle()
}

// golang.org/issue/15425: test that a handler closing the request
// body doesn't terminate the stream to the peer. (It just stops
// readability from the handler's side, and eventually the client
// runs out of flow control tokens)
func TestServerSendDataAfterRequestBodyClose(t *testing.T) {
	synctestTest(t, testServerSendDataAfterRequestBodyClose)
}
func testServerSendDataAfterRequestBodyClose(t testing.TB) {
	st := newServerTester(t, nil)
	st.greet()

	st.writeHeaders(HeadersFrameParam{
		StreamID:      1,
		BlockFragment: st.encodeHeader(),
		EndStream:     false,
		EndHeaders:    true,
	})

	// Handler starts writing the response body.
	call := st.nextHandlerCall()
	call.do(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte("one"))
		http.NewResponseController(w).Flush()
	})
	st.wantFrameType(FrameHeaders)
	st.wantData(wantData{
		streamID:  1,
		endStream: false,
		data:      []byte("one"),
	})
	st.wantIdle()

	// Handler closes the request body.
	// This is not observable by the client.
	call.do(func(w http.ResponseWriter, req *http.Request) {
		req.Body.Close()
	})
	st.wantIdle()

	// The client can still send request data, which is discarded.
	st.writeData(1, false, []byte("client-sent data"))
	st.wantIdle()

	// Handler can still write more response body,
	// which is sent to the client.
	call.do(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte("two"))
		http.NewResponseController(w).Flush()
	})
	st.wantData(wantData{
		streamID:  1,
		endStream: false,
		data:      []byte("two"),
	})
	st.wantIdle()
}

func TestServerSettingNoRFC7540Priorities(t *testing.T) {
	synctestTest(t, testServerSettingNoRFC7540Priorities)
}
func testServerSettingNoRFC7540Priorities(t testing.TB) {
	const wantNoRFC7540Setting = true
	st := newServerTester(t, nil)
	defer st.Close()

	var gotNoRFC7540Setting bool
	st.greetAndCheckSettings(func(s Setting) error {
		if s.ID != SettingNoRFC7540Priorities {
			return nil
		}
		gotNoRFC7540Setting = s.Val == 1
		return nil
	})
	if wantNoRFC7540Setting != gotNoRFC7540Setting {
		t.Errorf("want SETTINGS_NO_RFC7540_PRIORITIES to be %v, got %v", wantNoRFC7540Setting, gotNoRFC7540Setting)
	}
}

func TestServerSettingNoRFC7540PrioritiesInvalid(t *testing.T) {
	synctestTest(t, testServerSettingNoRFC7540PrioritiesInvalid)
}
func testServerSettingNoRFC7540PrioritiesInvalid(t testing.TB) {
	st := newServerTester(t, nil)
	defer st.Close()

	st.writePreface()
	st.writeSettings(Setting{ID: SettingNoRFC7540Priorities, Val: 2})
	synctest.Wait()
	st.readFrame() // SETTINGS frame
	st.readFrame() // WINDOW_UPDATE frame
	st.wantGoAway(0, ErrCodeProtocol)
}

// This test documents current behavior, rather than ideal behavior that we
// would necessarily like to see. Refer to go.dev/issues/75936 for details.
func TestServerRFC9218PrioritySmallPayload(t *testing.T) {
	synctestTest(t, testServerRFC9218PrioritySmallPayload)
}
func testServerRFC9218PrioritySmallPayload(t testing.TB) {
	endTest := false
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		for !endTest {
			w.Write([]byte("a"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}, func(s *http.Server) {
		s.Protocols = protocols("h2c")
	})
	st.greet()
	if syncConn, ok := st.cc.(*synctestNetConn); ok {
		syncConn.SetReadBufferSize(1)
	} else {
		t.Fatal("Server connection is not synctestNetConn")
	}
	defer st.Close()
	defer func() { endTest = true }()

	// Create 5 streams with urgency of 0, and another 5 streams with urgency
	// of 7.
	// Since each stream receives an infinite number of bytes, we should expect
	// to see that almost all of the response we get are for the streams with
	// urgency of 0.
	for i := 1; i <= 19; i += 2 {
		urgency := uint8(0)
		if i > 10 {
			urgency = 7
		}
		st.writeHeaders(HeadersFrameParam{
			StreamID:      uint32(i),
			BlockFragment: st.encodeHeader("priority", fmt.Sprintf("u=%d", urgency)),
			EndStream:     true,
			EndHeaders:    true,
		})
		synctest.Wait()
	}

	// In the current implementation however, the response we get are
	// distributed equally amongst all the streams, regardless of weight.
	streamWriteCount := make(map[uint32]int)
	totalWriteCount := 10000
	for range totalWriteCount {
		f := st.readFrame()
		if f == nil {
			break
		}
		streamWriteCount[f.Header().StreamID] += 1
	}
	for streamID, writeCount := range streamWriteCount {
		expectedWriteCount := totalWriteCount / len(streamWriteCount)
		errorMargin := expectedWriteCount / 100
		if writeCount >= expectedWriteCount+errorMargin || writeCount <= expectedWriteCount-errorMargin {
			t.Errorf("Expected stream %v to receive %v±%v writes, got %v", streamID, expectedWriteCount, errorMargin, writeCount)
		}
	}
}

func TestServerRFC9218Priority(t *testing.T) {
	synctestTest(t, testServerRFC9218Priority)
}
func testServerRFC9218Priority(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(slices.Repeat([]byte("a"), 16<<20))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}, func(s *http.Server) {
		s.Protocols = protocols("h2c")
	})
	defer st.Close()
	st.greet()
	if syncConn, ok := st.cc.(*synctestNetConn); ok {
		syncConn.SetReadBufferSize(1)
	} else {
		t.Fatal("Server connection is not synctestNetConn")
	}
	st.writeWindowUpdate(0, 1<<30)
	synctest.Wait()

	// Create 8 streams, where streams with larger ID has lower urgency value
	// (i.e. more urgent).
	for i := range 8 {
		streamID := uint32(i*2 + 1)
		urgency := 7 - i
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader("priority", fmt.Sprintf("u=%d", urgency)),
			EndStream:     true,
			EndHeaders:    true,
		})
	}
	synctest.Wait()

	// Keep track of the last frame seen for each stream, indicating that they
	// are done being processed.
	lastFrame := make(map[uint32]int)
	for i := 0; ; i++ {
		f := st.readFrame()
		if f == nil {
			break
		}
		lastFrame[f.Header().StreamID] = i
	}
	for i := range 7 {
		streamID := uint32(i*2 + 1)
		nextStreamID := streamID + 2
		if lastFrame[streamID] < lastFrame[nextStreamID] {
			t.Errorf("stream %d finished before stream %d unexpectedly", streamID, nextStreamID)
		}
	}
}

func TestServerRFC9218PriorityIgnoredWhenProxied(t *testing.T) {
	synctestTest(t, testServerRFC9218PriorityIgnoredWhenProxied)
}
func testServerRFC9218PriorityIgnoredWhenProxied(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(slices.Repeat([]byte("a"), 16<<20))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}, func(s *http.Server) {
		s.Protocols = protocols("h2c")
	})
	defer st.Close()
	st.greet()
	if syncConn, ok := st.cc.(*synctestNetConn); ok {
		syncConn.SetReadBufferSize(1)
	} else {
		t.Fatal("Server connection is not synctestNetConn")
	}
	st.writeWindowUpdate(0, 1<<30)
	synctest.Wait()

	// Create 8 streams, where streams with larger ID has lower urgency value
	// (i.e. more urgent). These should be ignored since the requests are
	// coming through a proxy.
	for i := range 8 {
		streamID := uint32(i*2 + 1)
		urgency := 7 - i
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader("priority", fmt.Sprintf("u=%d", urgency), "via", "a proxy"),
			EndStream:     true,
			EndHeaders:    true,
		})
	}
	synctest.Wait()
	var streamFrameOrder []uint32
	for f := st.readFrame(); f != nil; f = st.readFrame() {
		streamFrameOrder = append(streamFrameOrder, f.Header().StreamID)
	}
	// Only check the middle-half of the frame processing order, since the
	// beginning and end can be not perfectly round-robin (e.g. stream 1 gets
	// processed a few times while waiting before other streams are opened).
	half := streamFrameOrder[len(streamFrameOrder)/4 : len(streamFrameOrder)*3/4]
	if !slices.Equal(slices.Compact(half), half) {
		t.Errorf("want stream to be processed in round-robin manner when proxied, got: %v", streamFrameOrder)
	}
}

func TestServerRFC9218PriorityAware(t *testing.T) {
	synctestTest(t, testServerRFC9218PriorityAware)
}
func testServerRFC9218PriorityAware(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(slices.Repeat([]byte("a"), 16<<20))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}, func(s *http.Server) {
		s.Protocols = protocols("h2c")
	})
	defer st.Close()
	st.greet()
	if syncConn, ok := st.cc.(*synctestNetConn); ok {
		syncConn.SetReadBufferSize(1)
	} else {
		t.Fatal("Server connection is not synctestNetConn")
	}
	st.writeWindowUpdate(0, 1<<30)
	synctest.Wait()

	// When there is no indication that the client is aware of RFC 9218
	// priority, it should process streams in a round-robin manner.
	streamCount := 10
	for i := range streamCount {
		streamID := uint32(i*2 + 1)
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
	}
	synctest.Wait()
	var streamFrameOrder []uint32
	for f := st.readFrame(); f != nil; f = st.readFrame() {
		streamFrameOrder = append(streamFrameOrder, f.Header().StreamID)
	}
	// Only check the middle-half of the frame processing order, since the
	// beginning and end can be not perfectly round-robin (e.g. stream 1 gets
	// processed a few times while waiting before other streams are opened).
	half := streamFrameOrder[len(streamFrameOrder)/4 : len(streamFrameOrder)*3/4]
	if !slices.Equal(slices.Compact(half), half) {
		t.Errorf("want stream to be processed in round-robin manner when unaware of priority, got: %v", streamFrameOrder)
	}

	// Send a PRIORITY_UPDATE frame for stream 1 which would have finished by
	// now. So, this is a no-op, but makes it so that the server is aware that
	// the client is aware of RFC 9218 priority.
	st.writePriorityUpdate(1, "")
	synctest.Wait()

	// Now that the server knows that the client is aware of RFC 9218 priority,
	// streams should be processed one-by-one to completion when no explicit
	// priority is given as they all have the same urgency and are
	// non-incremental.
	streamFrameOrder = []uint32{}
	for i := range streamCount {
		i += streamCount
		streamID := uint32(i*2 + 1)
		st.writeHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: st.encodeHeader(),
			EndStream:     true,
			EndHeaders:    true,
		})
	}
	for f := st.readFrame(); f != nil; f = st.readFrame() {
		streamFrameOrder = append(streamFrameOrder, f.Header().StreamID)
	}
	if !slices.Equal(slices.Compact(half), half) {
		t.Errorf("want stream to be processed one-by-one to completion when aware of priority, got: %v", streamFrameOrder)
	}
}

func TestServerInvalidPathHeader(t *testing.T) {
	synctestTest(t, testServerInvalidPathHeader)
}
func testServerInvalidPathHeader(t testing.TB) {
	for _, path := range []string{
		"",
		"\x00",
		"https://example.com/",
	} {
		testServerRejectsStream(t, ErrCodeProtocol, func(st *serverTester) {
			st.fr.AllowIllegalWrites = true
			st.writeHeaders(HeadersFrameParam{
				StreamID: 1,
				BlockFragment: st.encodeHeader(
					":path", path,
				),
				EndStream:  true,
				EndHeaders: true,
			})
		})
	}
}

func TestServerPathInitialSlashes(t *testing.T) {
	synctestTest(t, testServerPathInitialSlashes)
}
func testServerPathInitialSlashes(t testing.TB) {
	st := newServerTester(t, nil)
	st.greet()

	// This path should be passed through unchanged,
	// and not interpreted as a protocol-relative URL or have initial /s stripped.
	const path = "//narf.com/path"
	st.writeHeaders(HeadersFrameParam{
		StreamID: 1,
		BlockFragment: st.encodeHeader(
			":path", path,
		),
		EndStream:  true,
		EndHeaders: true,
	})

	call := st.nextHandlerCall()
	if got, want := call.req.URL.Host, ""; got != want {
		t.Errorf("got req.URL.Host %q, want %q", got, want)
	}
	if got, want := call.req.URL.Path, path; got != want {
		t.Errorf("got req.URL.Path %q, want %q", got, want)
	}
}

func TestConsistentConstants(t *testing.T) {
	if h1, h2 := http.DefaultMaxHeaderBytes, http2.DefaultMaxHeaderBytes; h1 != h2 {
		t.Errorf("DefaultMaxHeaderBytes: http (%v) != http2 (%v)", h1, h2)
	}
	if h1, h2 := http.TimeFormat, http2.TimeFormat; h1 != h2 {
		t.Errorf("TimeFormat: http (%v) != http2 (%v)", h1, h2)
	}
}

var (
	testServerTLSConfig *tls.Config
	testClientTLSConfig *tls.Config
)

func init() {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		panic(err)
	}
	testServerTLSConfig = &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h2"},
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		panic(err)
	}
	certpool := x509.NewCertPool()
	certpool.AddCert(x509Cert)
	testClientTLSConfig = &tls.Config{
		InsecureSkipVerify: true,
		RootCAs:            certpool,
		NextProtos:         []string{"h2"},
	}
}

func protocols(protos ...string) *http.Protocols {
	p := new(http.Protocols)
	for _, s := range protos {
		switch s {
		case "h1":
			p.SetHTTP1(true)
		case "h2":
			p.SetHTTP2(true)
		case "h2c":
			p.SetUnencryptedHTTP2(true)
		default:
			panic("unknown protocol: " + s)
		}
	}
	return p
}

//go:linkname transportFromH1Transport
func transportFromH1Transport(tr *http.Transport) any
