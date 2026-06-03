// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of Server

package httptest

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"fmt"
	"internal/nettest"
	"log"
	"net"
	"net/http"
	"net/http/internal/testcert"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	_ "unsafe" // for linkname
)

// A Server is an HTTP server for use in end-to-end HTTP tests.
//
// Most tests should create a server with [NewTestServer].
// The [Server.Client] method returns a client which sends requests to the test server.
//
//	// Create a test server and send a request to it.
//	server := httptest.NewTestServer(t, handler)
//	resp, err := server.Client().Get("http://www.example.com/")
//
// # Configuration
//
// Tests may change a Server's configuration prior to using it.
// The configuration must not be changed after the first call to
// [Server.Client], [Server.Start], or [Server.StartTLS].
//
//	// Configure a test server before using.
//	server := httptest.NewTestServer(t, handler)
//	server.Config.MaxHeaderBytes = 1024
//	resp, err := server.Client().Get("http://www.example.com/")
//
// # Tests
//
// Servers created with [NewTestServer] will:
//
//   - Fail the test if the server handler panics with
//     any value other than [http.ErrAbortHandler].
//   - Register a Cleanup function to shut down the server at the end of the test.
//
// Servers created in any other way must be manually shut down with [Server.Close].
//
// # In-Memory Network
//
// A Server may use an in-memory network implementation or
// listen on a local network loopback interface.
// Most tests should use the in-memory network,
// which avoids port exhaustion and other transient networking issues
// and is suitable for use with the [testing/synctest] package.
//
// To use the in-memory network, create a server with [NewTestServer].
// Do not call [Server.Start] or [Server.StartTLS].
//
// When using the in-memory network, the [http.Client] returned by [Server.Client]
// is configured to send all requests to the server.
// The client will direct HTTP and HTTPS requests,
// regardless of destination address or hostname, to the server.
// Requests do not need to use [Server.URL] as the base URL.
//
//	server := httptest.NewTestServer(t, handler)
//	client := server.Client()
//
//	// All of these requests are sent to the test server.
//	// https:// requests use TLS over the in-memory network.
//	_, _ = client.Get("http://www.example.com/")
//	_, _ = client.Get("https://go.dev/")
//	_, _ = client.Get("http://10.0.0.1/")
//
// The [Server.Listener] field is not set when using the in-memory network.
//
// # Loopback Network
//
// To listen on a loopback interface, call [Server.Start] or [Server.StartTLS].
// The server will listen on a system-chosen port.
//
// Loopback servers serve one of HTTP (when started with [Server.Start])
// or HTTPS (when started with [Server.StartTLS]).
//
// When using the loopback network, the [http.Client] returned by [Server.Client]
// is configured to send requests with a hostname of "example.com" or a subdomain
// of ".example.com" to the server.
//
// Requests may also be sent to the server's loopback address.
// The [Server.URL] field is set to a base URL containing the server's address.
//
//	server := httptest.NewTestServer(t, handler)
//	server.Start()
//	client := server.Client()
//
//	// This request is sent to the test server.
//	_, _ = server.Client().Get(server.URL + "/")
//
//	// This request (using http.DefaultClient) is also sent to the test server,
//	// since server.URL contains the server's local IP address.
//	_, _ = http.Get(server.URL + "/")
type Server struct {
	// URL is the base URL of the server, of the form http://address:port
	// with no trailing slash.
	//
	// It is set by the first call to Client, Start, or StartTLS.
	//
	// For servers listening on loopback, the address is the loopback IP address
	// of the server.
	//
	// For servers using the in-memory network, this address is "example.com".
	// Requests sent to servers using the in-memory network may use any address.
	// It is not necessary to use this base URL.
	URL string

	// Listener is the network listener for servers listening on loopback.
	// It is not set for servers using the in-memory network.
	Listener net.Listener

	// EnableHTTP2 controls whether HTTP/2 is enabled on the server.
	// It must be set before calling Client, Start, or StartTLS.
	EnableHTTP2 bool

	// TLS is the optional TLS configuration, populated with a new config
	// after TLS is started. If set on an unstarted server before StartTLS
	// is called, existing fields are copied into the new config.
	TLS *tls.Config

	// Config may be changed before calling Client, Start, or StartTLS.
	Config *http.Server

	t testing.TB

	// certificate is a parsed version of the TLS config certificate, if present.
	certificate *x509.Certificate

	// startOnce is used to start fakenet servers once.
	startOnce sync.Once

	// started indicates whether the server has been started.
	started bool

	// Fake network listeners, one for HTTP and one for HTTPS.
	fakeListener    *nettest.Listener
	fakeTLSListener *nettest.Listener

	// wg counts the number of outstanding HTTP requests on this server.
	// Close blocks until all requests are finished.
	wg sync.WaitGroup

	mu     sync.Mutex // guards closed and conns
	closed bool
	conns  map[net.Conn]http.ConnState // except terminal states

	// client is configured for use with the server.
	// Its transport is automatically closed when Close is called.
	client *http.Client
}

// NewTestServer returns a new [Server] for a test.
// The server will use an in-memory network implementation by default.
//
// If the handler is nil, the server will serve 500 responses to all requests.
// It will not use [http.DefaultServeMux].
//
// See the [Server] documentation for more details.
func NewTestServer(t testing.TB, handler http.Handler) *Server {
	s := &Server{
		t:      t,
		Config: &http.Server{Handler: testServerHandler{t: t, h: handler}},
	}
	t.Cleanup(func() {
		s.Close()
	})
	return s
}

type testServerHandler struct {
	t testing.TB
	h http.Handler
}

func (h testServerHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	defer func() {
		if err := recover(); err != nil {
			if err != http.ErrAbortHandler {
				// This is the same logging http.Server would do,
				// but we can put it into the test output rather than stderr.
				const size = 64 << 10
				buf := make([]byte, size)
				buf = buf[:runtime.Stack(buf, false)]
				h.t.Errorf("httptest: panic in server handler: %v\n%s", err, buf)
			}
			// Convert panic to ErrAbortHandler to suppress http.Server's logging.
			panic(http.ErrAbortHandler)
		}
	}()
	if h.h != nil {
		h.h.ServeHTTP(w, req)
	} else {
		w.WriteHeader(500)
	}
}

func newLocalListener() net.Listener {
	if serveFlag != "" {
		l, err := net.Listen("tcp", serveFlag)
		if err != nil {
			panic(fmt.Sprintf("httptest: failed to listen on %v: %v", serveFlag, err))
		}
		return l
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		if l, err = net.Listen("tcp6", "[::1]:0"); err != nil {
			panic(fmt.Sprintf("httptest: failed to listen on a port: %v", err))
		}
	}
	return l
}

// When debugging a particular http server-based test,
// this flag lets you run
//
//	go test -run='^BrokenTest$' -httptest.serve=127.0.0.1:8000
//
// to start the broken server so you can interact with it manually.
// We only register this flag if it looks like the caller knows about it
// and is trying to use it as we don't want to pollute flags and this
// isn't really part of our API. Don't depend on this.
var serveFlag string

func init() {
	if strSliceContainsPrefix(os.Args, "-httptest.serve=") || strSliceContainsPrefix(os.Args, "--httptest.serve=") {
		flag.StringVar(&serveFlag, "httptest.serve", "", "if non-empty, httptest.NewServer serves on this address and blocks.")
	}
}

func strSliceContainsPrefix(v []string, pre string) bool {
	for _, s := range v {
		if strings.HasPrefix(s, pre) {
			return true
		}
	}
	return false
}

// NewServer starts and returns a new [Server] listening on a
// local network loopback interface.
// This is equivalent to calling [NewUnstartedServer] followed by [Server.Start].
//
// The caller should call [Server.Close] when finished, to shut it down.
//
// Most users should use [NewTestServer] instead.
// See the [Server] documentation for details.
func NewServer(handler http.Handler) *Server {
	ts := NewUnstartedServer(handler)
	ts.Start()
	return ts
}

// NewUnstartedServer returns a new [Server] listening on a
// local network loopback interface. It does not start the server.
//
// After changing the server's configuration, the caller should
// call [Server.Start] or [Server.StartTLS].
//
// The caller should call [Server.Close] when finished, to shut it down.
//
// Most users should use [NewTestServer] instead.
// See the [Server] documentation for details.
func NewUnstartedServer(handler http.Handler) *Server {
	return &Server{
		Listener: newLocalListener(),
		Config:   &http.Server{Handler: handler},
	}
}

func (s *Server) startCommon(useLoopback bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		panic("Server already started")
	}
	if s.closed {
		panic("Start of closed Server")
	}
	s.started = true
	if s.t != nil && useLoopback {
		// We're being called from Start or StartTLS.
		// Don't try to start the server again when Client is called.
		s.startOnce.Do(func() {})

		// NewTestServer servers create their listener at start time.
		//
		// We might want to permit the user to provide their own Listener
		// in the future. For now, we panic.
		if s.Listener != nil {
			panic("Server.Listener is unexpectedly set")
		}
		s.Listener = newLocalListener()
	}
	s.wrap()
}

// Start starts a server on a local loopback network interface.
//
// The server should have been created by [NewTestServer] or [NewUnstartedServer].
func (s *Server) Start() {
	s.startCommon(true)

	tr := &http.Transport{}
	s.client = &http.Client{Transport: tr}
	if s.Listener == nil {
		return
	}
	dialer := net.Dialer{}
	// User code may set either of Dial or DialContext, with DialContext taking precedence.
	// We set DialContext here to preserve any context values that are passed in,
	// but fall back to Dial if the user has set it.
	tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		if tr.Dial != nil {
			return tr.Dial(network, addr)
		}
		if addr == "example.com:80" || strings.HasSuffix(addr, ".example.com:80") {
			addr = s.Listener.Addr().String()
		}
		return dialer.DialContext(ctx, network, addr)
	}
	s.URL = "http://" + s.Listener.Addr().String()
	s.goServe(s.Listener)
	if serveFlag != "" {
		fmt.Fprintln(os.Stderr, "httptest: serving on", s.URL)
		select {}
	}
}

func (s *Server) initTLS() (tlsClientConfig *tls.Config, err error) {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		return nil, err
	}

	existingConfig := s.TLS
	if existingConfig != nil {
		s.TLS = existingConfig.Clone()
	} else {
		s.TLS = new(tls.Config)
	}
	if s.TLS.NextProtos == nil {
		nextProtos := []string{"http/1.1"}
		if s.EnableHTTP2 {
			nextProtos = []string{"h2"}
		}
		s.TLS.NextProtos = nextProtos
	}
	if len(s.TLS.Certificates) == 0 {
		s.TLS.Certificates = []tls.Certificate{cert}
	}
	s.certificate, err = x509.ParseCertificate(s.TLS.Certificates[0].Certificate[0])
	if err != nil {
		return nil, err
	}
	certpool := x509.NewCertPool()
	certpool.AddCert(s.certificate)
	return &tls.Config{
		RootCAs: certpool,
	}, nil
}

// Start starts TLS on a server on a local loopback network interface.
//
// The server should have been created by [NewTestServer] or [NewUnstartedServer].
func (s *Server) StartTLS() {
	s.startCommon(true)

	s.client = &http.Client{}

	tlsClientConfig, err := s.initTLS()
	if err != nil {
		panic(fmt.Sprintf("httptest: NewTLSServer: %v", err))
	}

	tr := &http.Transport{
		TLSClientConfig:   tlsClientConfig,
		ForceAttemptHTTP2: s.EnableHTTP2,
	}
	s.client.Transport = tr

	if s.Listener == nil {
		return
	}
	dialer := net.Dialer{}
	tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		if tr.Dial != nil {
			return tr.Dial(network, addr)
		}
		if addr == "example.com:443" || strings.HasSuffix(addr, ".example.com:443") {
			addr = s.Listener.Addr().String()
		}
		return dialer.DialContext(ctx, network, addr)
	}
	s.Listener = tls.NewListener(s.Listener, s.TLS)
	s.URL = "https://" + s.Listener.Addr().String()
	s.goServe(s.Listener)
}

func (s *Server) startFakeNet() {
	s.startCommon(false)

	s.client = &http.Client{}

	tlsClientConfig, err := s.initTLS()
	if err != nil {
		panic(fmt.Sprintf("httptest: NewTestServer: %v", err))
	}

	tr := &http.Transport{
		TLSClientConfig:   tlsClientConfig,
		ForceAttemptHTTP2: s.EnableHTTP2,
	}
	s.client.Transport = tr

	s.fakeListener = nettest.NewListener()
	s.fakeTLSListener = nettest.NewListener()

	// Set InsecureSkipVerify rather than depending on a specific server hostname.
	tr.TLSClientConfig.InsecureSkipVerify = true
	tr.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		return s.fakeListener.NewConn(), nil
	}
	tr.DialTLSContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		return tls.Client(s.fakeTLSListener.NewConn(), tr.TLSClientConfig), nil
	}
	s.URL = "http://example.com"
	s.goServe(s.fakeListener)
	s.goServe(tls.NewListener(s.fakeTLSListener, s.TLS))
}

// NewTLSServer starts and returns a new [Server] using TLS and listening on a
// local network loopback interface.
// This is equivalent to calling [NewUnstartedServer] followed by [Server.StartTLS].
//
// The caller should call [Server.Close] when finished, to shut it down.
//
// Most users should use [NewTestServer] instead.
// See the [Server] documentation for details.
func NewTLSServer(handler http.Handler) *Server {
	ts := NewUnstartedServer(handler)
	ts.StartTLS()
	return ts
}

type closeIdleTransport interface {
	CloseIdleConnections()
}

// Close shuts down the server and blocks until all outstanding
// requests on this server have completed.
func (s *Server) Close() {
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		if s.Listener != nil {
			s.Listener.Close()
		}
		if s.fakeListener != nil {
			s.fakeListener.Close()
			s.fakeTLSListener.Close()
		}
		s.Config.SetKeepAlivesEnabled(false)
		for c, st := range s.conns {
			// Force-close any idle connections (those between
			// requests) and new connections (those which connected
			// but never sent a request). StateNew connections are
			// super rare and have only been seen (in
			// previously-flaky tests) in the case of
			// socket-late-binding races from the http Client
			// dialing this server and then getting an idle
			// connection before the dial completed. There is thus
			// a connected connection in StateNew with no
			// associated Request. We only close StateIdle and
			// StateNew because they're not doing anything. It's
			// possible StateNew is about to do something in a few
			// milliseconds, but a previous CL to check again in a
			// few milliseconds wasn't liked (early versions of
			// https://golang.org/cl/15151) so now we just
			// forcefully close StateNew. The docs for Server.Close say
			// we wait for "outstanding requests", so we don't close things
			// in StateActive.
			if st == http.StateIdle || st == http.StateNew {
				s.closeConn(c)
			}
		}
		// If this server doesn't shut down in 5 seconds, tell the user why.
		t := time.AfterFunc(5*time.Second, s.logCloseHangDebugInfo)
		defer t.Stop()
	}
	s.mu.Unlock()

	// Not part of httptest.Server's correctness, but assume most
	// users of httptest.Server will be using the standard
	// transport, so help them out and close any idle connections for them.
	if t, ok := http.DefaultTransport.(closeIdleTransport); ok {
		t.CloseIdleConnections()
	}

	// Also close the client idle connections.
	if s.client != nil {
		if t, ok := s.client.Transport.(closeIdleTransport); ok {
			t.CloseIdleConnections()
		}
	}
	s.wg.Wait()
}

func (s *Server) logCloseHangDebugInfo() {
	s.mu.Lock()
	defer s.mu.Unlock()
	var buf strings.Builder
	buf.WriteString("httptest.Server blocked in Close after 5 seconds, waiting for connections:\n")
	for c, st := range s.conns {
		fmt.Fprintf(&buf, "  %T %p %v in state %v\n", c, c, c.RemoteAddr(), st)
	}
	log.Print(buf.String())
}

// CloseClientConnections closes any open HTTP connections to the test Server.
func (s *Server) CloseClientConnections() {
	s.mu.Lock()
	nconn := len(s.conns)
	ch := make(chan struct{}, nconn)
	for c := range s.conns {
		go s.closeConnChan(c, ch)
	}
	s.mu.Unlock()

	// Wait for outstanding closes to finish.
	//
	// Out of paranoia for making a late change in Go 1.6, we
	// bound how long this can wait, since golang.org/issue/14291
	// isn't fully understood yet. At least this should only be used
	// in tests.
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	for i := 0; i < nconn; i++ {
		select {
		case <-ch:
		case <-timer.C:
			// Too slow. Give up.
			return
		}
	}
}

// Certificate returns the certificate used by the server, or nil if
// the server doesn't use TLS.
func (s *Server) Certificate() *x509.Certificate {
	return s.certificate
}

// Client returns an HTTP client configured for making requests to the server.
// It is configured to trust the server's TLS test certificate and will
// close its idle connections on [Server.Close].
func (s *Server) Client() *http.Client {
	if s.t != nil {
		s.startOnce.Do(s.startFakeNet)
	}
	return s.client
}

func (s *Server) goServe(li net.Listener) {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.Config.Serve(li)
	}()
}

// wrap installs the connection state-tracking hook to know which
// connections are idle.
func (s *Server) wrap() {
	oldHook := s.Config.ConnState
	s.Config.ConnState = func(c net.Conn, cs http.ConnState) {
		s.mu.Lock()
		defer s.mu.Unlock()

		switch cs {
		case http.StateNew:
			if _, exists := s.conns[c]; exists {
				panic("invalid state transition")
			}
			if s.conns == nil {
				s.conns = make(map[net.Conn]http.ConnState)
			}
			// Add c to the set of tracked conns and increment it to the
			// waitgroup.
			s.wg.Add(1)
			s.conns[c] = cs
			if s.closed {
				// Probably just a socket-late-binding dial from
				// the default transport that lost the race (and
				// thus this connection is now idle and will
				// never be used).
				s.closeConn(c)
			}
		case http.StateActive:
			if oldState, ok := s.conns[c]; ok {
				if oldState != http.StateNew && oldState != http.StateIdle {
					panic("invalid state transition")
				}
				s.conns[c] = cs
			}
		case http.StateIdle:
			if oldState, ok := s.conns[c]; ok {
				if oldState != http.StateActive {
					panic("invalid state transition")
				}
				s.conns[c] = cs
			}
			if s.closed {
				s.closeConn(c)
			}
		case http.StateHijacked, http.StateClosed:
			// Remove c from the set of tracked conns and decrement it from the
			// waitgroup, unless it was previously removed.
			if _, ok := s.conns[c]; ok {
				delete(s.conns, c)
				// Keep Close from returning until the user's ConnState hook
				// (if any) finishes.
				defer s.wg.Done()
			}
		}
		if oldHook != nil {
			oldHook(c, cs)
		}
	}
}

// closeConn closes c.
// s.mu must be held.
func (s *Server) closeConn(c net.Conn) { s.closeConnChan(c, nil) }

// closeConnChan is like closeConn, but takes an optional channel to receive a value
// when the goroutine closing c is done.
func (s *Server) closeConnChan(c net.Conn, done chan<- struct{}) {
	c.Close()
	if done != nil {
		done <- struct{}{}
	}
}
