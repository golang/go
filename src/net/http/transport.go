// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP client implementation. See RFC 2616.
//
// This is the low-level Transport implementation of RoundTripper.
// The high-level interface is in client.go.

package http

import (
	"bufio"
	"compress/gzip"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// DefaultTransport is the default implementation of Transport and is
// used by DefaultClient. It establishes network connections as needed
// and caches them for reuse by subsequent calls. It uses HTTP proxies
// as directed by the $HTTP_PROXY and $NO_PROXY (or $http_proxy and
// $no_proxy) environment variables.
var DefaultTransport RoundTripper = &Transport{
	Proxy: ProxyFromEnvironment,
	Dial: (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}).Dial,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: 1 * time.Second,
}

// DefaultMaxIdleConnsPerHost is the default value of Transport's
// MaxIdleConnsPerHost.
const DefaultMaxIdleConnsPerHost = 2

// Transport is an implementation of RoundTripper that supports HTTP,
// HTTPS, and HTTP proxies (for either HTTP or HTTPS with CONNECT).
//
// By default, Transport caches connections for future re-use.
// This may leave many open connections when accessing many hosts.
// This behavior can be managed using Transport's CloseIdleConnections method
// and the MaxIdleConnsPerHost and DisableKeepAlives fields.
//
// Transports should be reused instead of created as needed.
// Transports are safe for concurrent use by multiple goroutines.
//
// A Transport is a low-level primitive for making HTTP and HTTPS requests.
// For high-level functionality, such as cookies and redirects, see Client.
//
// Transport uses HTTP/1.1 for HTTP URLs and either HTTP/1.1 or HTTP/2
// for HTTPS URLs, depending on whether the server supports HTTP/2.
// See the package docs for more about HTTP/2.
type Transport struct {
	idleMu     sync.Mutex
	wantIdle   bool // user has requested to close all idle conns
	idleConn   map[connectMethodKey][]*persistConn
	idleConnCh map[connectMethodKey]chan *persistConn

	reqMu       sync.Mutex
	reqCanceler map[*Request]func()

	altMu    sync.RWMutex
	altProto map[string]RoundTripper // nil or map of URI scheme => RoundTripper

	// Proxy specifies a function to return a proxy for a given
	// Request. If the function returns a non-nil error, the
	// request is aborted with the provided error.
	// If Proxy is nil or returns a nil *URL, no proxy is used.
	Proxy func(*Request) (*url.URL, error)

	// Dial specifies the dial function for creating unencrypted
	// TCP connections.
	// If Dial is nil, net.Dial is used.
	Dial func(network, addr string) (net.Conn, error)

	// DialTLS specifies an optional dial function for creating
	// TLS connections for non-proxied HTTPS requests.
	//
	// If DialTLS is nil, Dial and TLSClientConfig are used.
	//
	// If DialTLS is set, the Dial hook is not used for HTTPS
	// requests and the TLSClientConfig and TLSHandshakeTimeout
	// are ignored. The returned net.Conn is assumed to already be
	// past the TLS handshake.
	DialTLS func(network, addr string) (net.Conn, error)

	// TLSClientConfig specifies the TLS configuration to use with
	// tls.Client. If nil, the default configuration is used.
	TLSClientConfig *tls.Config

	// TLSHandshakeTimeout specifies the maximum amount of time waiting to
	// wait for a TLS handshake. Zero means no timeout.
	TLSHandshakeTimeout time.Duration

	// DisableKeepAlives, if true, prevents re-use of TCP connections
	// between different HTTP requests.
	DisableKeepAlives bool

	// DisableCompression, if true, prevents the Transport from
	// requesting compression with an "Accept-Encoding: gzip"
	// request header when the Request contains no existing
	// Accept-Encoding value. If the Transport requests gzip on
	// its own and gets a gzipped response, it's transparently
	// decoded in the Response.Body. However, if the user
	// explicitly requested gzip it is not automatically
	// uncompressed.
	DisableCompression bool

	// MaxIdleConnsPerHost, if non-zero, controls the maximum idle
	// (keep-alive) to keep per-host.  If zero,
	// DefaultMaxIdleConnsPerHost is used.
	MaxIdleConnsPerHost int

	// ResponseHeaderTimeout, if non-zero, specifies the amount of
	// time to wait for a server's response headers after fully
	// writing the request (including its body, if any). This
	// time does not include the time to read the response body.
	ResponseHeaderTimeout time.Duration

	// ExpectContinueTimeout, if non-zero, specifies the amount of
	// time to wait for a server's first response headers after fully
	// writing the request headers if the request has an
	// "Expect: 100-continue" header. Zero means no timeout.
	// This time does not include the time to send the request header.
	ExpectContinueTimeout time.Duration

	// TLSNextProto specifies how the Transport switches to an
	// alternate protocol (such as HTTP/2) after a TLS NPN/ALPN
	// protocol negotiation.  If Transport dials an TLS connection
	// with a non-empty protocol name and TLSNextProto contains a
	// map entry for that key (such as "h2"), then the func is
	// called with the request's authority (such as "example.com"
	// or "example.com:1234") and the TLS connection. The function
	// must return a RoundTripper that then handles the request.
	// If TLSNextProto is nil, HTTP/2 support is enabled automatically.
	TLSNextProto map[string]func(authority string, c *tls.Conn) RoundTripper

	// nextProtoOnce guards initialization of TLSNextProto and
	// h2transport (via onceSetNextProtoDefaults)
	nextProtoOnce sync.Once
	h2transport   *http2Transport // non-nil if http2 wired up

	// TODO: tunable on global max cached connections
	// TODO: tunable on timeout on cached connections
	// TODO: tunable on max per-host TCP dials in flight (Issue 13957)
}

// onceSetNextProtoDefaults initializes TLSNextProto.
// It must be called via t.nextProtoOnce.Do.
func (t *Transport) onceSetNextProtoDefaults() {
	if strings.Contains(os.Getenv("GODEBUG"), "http2client=0") {
		return
	}
	if t.TLSNextProto != nil {
		// This is the documented way to disable http2 on a
		// Transport.
		return
	}
	if t.TLSClientConfig != nil {
		// Be conservative for now (for Go 1.6) at least and
		// don't automatically enable http2 if they've
		// specified a custom TLS config. Let them opt-in
		// themselves via http2.ConfigureTransport so we don't
		// surprise them by modifying their tls.Config.
		// Issue 14275.
		return
	}
	if t.ExpectContinueTimeout != 0 {
		// Unsupported in http2, so disable http2 for now.
		// Issue 13851.
		return
	}
	t2, err := http2configureTransport(t)
	if err != nil {
		log.Printf("Error enabling Transport HTTP/2 support: %v", err)
	} else {
		t.h2transport = t2
	}
}

// ProxyFromEnvironment returns the URL of the proxy to use for a
// given request, as indicated by the environment variables
// HTTP_PROXY, HTTPS_PROXY and NO_PROXY (or the lowercase versions
// thereof). HTTPS_PROXY takes precedence over HTTP_PROXY for https
// requests.
//
// The environment values may be either a complete URL or a
// "host[:port]", in which case the "http" scheme is assumed.
// An error is returned if the value is a different form.
//
// A nil URL and nil error are returned if no proxy is defined in the
// environment, or a proxy should not be used for the given request,
// as defined by NO_PROXY.
//
// As a special case, if req.URL.Host is "localhost" (with or without
// a port number), then a nil URL and nil error will be returned.
func ProxyFromEnvironment(req *Request) (*url.URL, error) {
	var proxy string
	if req.URL.Scheme == "https" {
		proxy = httpsProxyEnv.Get()
	}
	if proxy == "" {
		proxy = httpProxyEnv.Get()
	}
	if proxy == "" {
		return nil, nil
	}
	if !useProxy(canonicalAddr(req.URL)) {
		return nil, nil
	}
	proxyURL, err := url.Parse(proxy)
	if err != nil || !strings.HasPrefix(proxyURL.Scheme, "http") {
		// proxy was bogus. Try prepending "http://" to it and
		// see if that parses correctly. If not, we fall
		// through and complain about the original one.
		if proxyURL, err := url.Parse("http://" + proxy); err == nil {
			return proxyURL, nil
		}
	}
	if err != nil {
		return nil, fmt.Errorf("invalid proxy address %q: %v", proxy, err)
	}
	return proxyURL, nil
}

// ProxyURL returns a proxy function (for use in a Transport)
// that always returns the same URL.
func ProxyURL(fixedURL *url.URL) func(*Request) (*url.URL, error) {
	return func(*Request) (*url.URL, error) {
		return fixedURL, nil
	}
}

// transportRequest is a wrapper around a *Request that adds
// optional extra headers to write.
type transportRequest struct {
	*Request        // original request, not to be mutated
	extra    Header // extra headers to write, or nil
}

func (tr *transportRequest) extraHeaders() Header {
	if tr.extra == nil {
		tr.extra = make(Header)
	}
	return tr.extra
}

// RoundTrip implements the RoundTripper interface.
//
// For higher-level HTTP client support (such as handling of cookies
// and redirects), see Get, Post, and the Client type.
func (t *Transport) RoundTrip(req *Request) (*Response, error) {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)
	if req.URL == nil {
		req.closeBody()
		return nil, errors.New("http: nil Request.URL")
	}
	if req.Header == nil {
		req.closeBody()
		return nil, errors.New("http: nil Request.Header")
	}
	// TODO(bradfitz): switch to atomic.Value for this map instead of RWMutex
	t.altMu.RLock()
	altRT := t.altProto[req.URL.Scheme]
	t.altMu.RUnlock()
	if altRT != nil {
		if resp, err := altRT.RoundTrip(req); err != ErrSkipAltProtocol {
			return resp, err
		}
	}
	if s := req.URL.Scheme; s != "http" && s != "https" {
		req.closeBody()
		return nil, &badStringError{"unsupported protocol scheme", s}
	}
	if req.Method != "" && !validMethod(req.Method) {
		return nil, fmt.Errorf("net/http: invalid method %q", req.Method)
	}
	if req.URL.Host == "" {
		req.closeBody()
		return nil, errors.New("http: no Host in request URL")
	}

	for {
		// treq gets modified by roundTrip, so we need to recreate for each retry.
		treq := &transportRequest{Request: req}
		cm, err := t.connectMethodForRequest(treq)
		if err != nil {
			req.closeBody()
			return nil, err
		}

		// Get the cached or newly-created connection to either the
		// host (for http or https), the http proxy, or the http proxy
		// pre-CONNECTed to https server.  In any case, we'll be ready
		// to send it requests.
		pconn, err := t.getConn(req, cm)
		if err != nil {
			t.setReqCanceler(req, nil)
			req.closeBody()
			return nil, err
		}

		var resp *Response
		if pconn.alt != nil {
			// HTTP/2 path.
			t.setReqCanceler(req, nil) // not cancelable with CancelRequest
			resp, err = pconn.alt.RoundTrip(req)
		} else {
			resp, err = pconn.roundTrip(treq)
		}
		if err == nil {
			return resp, nil
		}
		if err := checkTransportResend(err, req, pconn); err != nil {
			return nil, err
		}
		testHookRoundTripRetried()
	}
}

// checkTransportResend checks whether a failed HTTP request can be
// resent on a new connection. The non-nil input error is the error from
// roundTrip, which might be wrapped in a beforeRespHeaderError error.
//
// The return value is err or the unwrapped error inside a
// beforeRespHeaderError.
func checkTransportResend(err error, req *Request, pconn *persistConn) error {
	brhErr, ok := err.(beforeRespHeaderError)
	if !ok {
		return err
	}
	err = brhErr.error // unwrap the custom error in case we return it
	if err != errMissingHost && pconn.isReused() && req.isReplayable() {
		// If we try to reuse a connection that the server is in the process of
		// closing, we may end up successfully writing out our request (or a
		// portion of our request) only to find a connection error when we try to
		// read from (or finish writing to) the socket.

		// There can be a race between the socket pool checking whether a socket
		// is still connected, receiving the FIN, and sending/reading data on a
		// reused socket. If we receive the FIN between the connectedness check
		// and writing/reading from the socket, we may first learn the socket is
		// disconnected when we get a ERR_SOCKET_NOT_CONNECTED. This will most
		// likely happen when trying to retrieve its IP address. See
		// http://crbug.com/105824 for more details.

		// We resend a request only if we reused a keep-alive connection and did
		// not yet receive any header data. This automatically prevents an
		// infinite resend loop because we'll run out of the cached keep-alive
		// connections eventually.
		return nil
	}
	return err
}

// ErrSkipAltProtocol is a sentinel error value defined by Transport.RegisterProtocol.
var ErrSkipAltProtocol = errors.New("net/http: skip alternate protocol")

// RegisterProtocol registers a new protocol with scheme.
// The Transport will pass requests using the given scheme to rt.
// It is rt's responsibility to simulate HTTP request semantics.
//
// RegisterProtocol can be used by other packages to provide
// implementations of protocol schemes like "ftp" or "file".
//
// If rt.RoundTrip returns ErrSkipAltProtocol, the Transport will
// handle the RoundTrip itself for that one request, as if the
// protocol were not registered.
func (t *Transport) RegisterProtocol(scheme string, rt RoundTripper) {
	t.altMu.Lock()
	defer t.altMu.Unlock()
	if t.altProto == nil {
		t.altProto = make(map[string]RoundTripper)
	}
	if _, exists := t.altProto[scheme]; exists {
		panic("protocol " + scheme + " already registered")
	}
	t.altProto[scheme] = rt
}

// CloseIdleConnections closes any connections which were previously
// connected from previous requests but are now sitting idle in
// a "keep-alive" state. It does not interrupt any connections currently
// in use.
func (t *Transport) CloseIdleConnections() {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)
	t.idleMu.Lock()
	m := t.idleConn
	t.idleConn = nil
	t.idleConnCh = nil
	t.wantIdle = true
	t.idleMu.Unlock()
	for _, conns := range m {
		for _, pconn := range conns {
			pconn.close(errCloseIdleConns)
		}
	}
	if t2 := t.h2transport; t2 != nil {
		t2.CloseIdleConnections()
	}
}

// CancelRequest cancels an in-flight request by closing its connection.
// CancelRequest should only be called after RoundTrip has returned.
//
// Deprecated: Use Request.Cancel instead. CancelRequest can not cancel
// HTTP/2 requests.
func (t *Transport) CancelRequest(req *Request) {
	t.reqMu.Lock()
	cancel := t.reqCanceler[req]
	delete(t.reqCanceler, req)
	t.reqMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

//
// Private implementation past this point.
//

var (
	httpProxyEnv = &envOnce{
		names: []string{"HTTP_PROXY", "http_proxy"},
	}
	httpsProxyEnv = &envOnce{
		names: []string{"HTTPS_PROXY", "https_proxy"},
	}
	noProxyEnv = &envOnce{
		names: []string{"NO_PROXY", "no_proxy"},
	}
)

// envOnce looks up an environment variable (optionally by multiple
// names) once. It mitigates expensive lookups on some platforms
// (e.g. Windows).
type envOnce struct {
	names []string
	once  sync.Once
	val   string
}

func (e *envOnce) Get() string {
	e.once.Do(e.init)
	return e.val
}

func (e *envOnce) init() {
	for _, n := range e.names {
		e.val = os.Getenv(n)
		if e.val != "" {
			return
		}
	}
}

// reset is used by tests
func (e *envOnce) reset() {
	e.once = sync.Once{}
	e.val = ""
}

func (t *Transport) connectMethodForRequest(treq *transportRequest) (cm connectMethod, err error) {
	cm.targetScheme = treq.URL.Scheme
	cm.targetAddr = canonicalAddr(treq.URL)
	if t.Proxy != nil {
		cm.proxyURL, err = t.Proxy(treq.Request)
	}
	return cm, err
}

// proxyAuth returns the Proxy-Authorization header to set
// on requests, if applicable.
func (cm *connectMethod) proxyAuth() string {
	if cm.proxyURL == nil {
		return ""
	}
	if u := cm.proxyURL.User; u != nil {
		username := u.Username()
		password, _ := u.Password()
		return "Basic " + basicAuth(username, password)
	}
	return ""
}

// error values for debugging and testing, not seen by users.
var (
	errKeepAlivesDisabled = errors.New("http: putIdleConn: keep alives disabled")
	errConnBroken         = errors.New("http: putIdleConn: connection is in bad state")
	errWantIdle           = errors.New("http: putIdleConn: CloseIdleConnections was called")
	errTooManyIdle        = errors.New("http: putIdleConn: too many idle connections")
	errCloseIdleConns     = errors.New("http: CloseIdleConnections called")
	errReadLoopExiting    = errors.New("http: persistConn.readLoop exiting")
	errServerClosedIdle   = errors.New("http: server closed idle conn")
)

func (t *Transport) putOrCloseIdleConn(pconn *persistConn) {
	if err := t.tryPutIdleConn(pconn); err != nil {
		pconn.close(err)
	}
}

// tryPutIdleConn adds pconn to the list of idle persistent connections awaiting
// a new request.
// If pconn is no longer needed or not in a good state, tryPutIdleConn returns
// an error explaining why it wasn't registered.
// tryPutIdleConn does not close pconn. Use putOrCloseIdleConn instead for that.
func (t *Transport) tryPutIdleConn(pconn *persistConn) error {
	if t.DisableKeepAlives || t.MaxIdleConnsPerHost < 0 {
		return errKeepAlivesDisabled
	}
	if pconn.isBroken() {
		return errConnBroken
	}
	key := pconn.cacheKey
	max := t.MaxIdleConnsPerHost
	if max == 0 {
		max = DefaultMaxIdleConnsPerHost
	}
	pconn.markReused()
	t.idleMu.Lock()

	waitingDialer := t.idleConnCh[key]
	select {
	case waitingDialer <- pconn:
		// We're done with this pconn and somebody else is
		// currently waiting for a conn of this type (they're
		// actively dialing, but this conn is ready
		// first). Chrome calls this socket late binding.  See
		// https://insouciant.org/tech/connection-management-in-chromium/
		t.idleMu.Unlock()
		return nil
	default:
		if waitingDialer != nil {
			// They had populated this, but their dial won
			// first, so we can clean up this map entry.
			delete(t.idleConnCh, key)
		}
	}
	if t.wantIdle {
		t.idleMu.Unlock()
		return errWantIdle
	}
	if t.idleConn == nil {
		t.idleConn = make(map[connectMethodKey][]*persistConn)
	}
	if len(t.idleConn[key]) >= max {
		t.idleMu.Unlock()
		return errTooManyIdle
	}
	for _, exist := range t.idleConn[key] {
		if exist == pconn {
			log.Fatalf("dup idle pconn %p in freelist", pconn)
		}
	}
	t.idleConn[key] = append(t.idleConn[key], pconn)
	t.idleMu.Unlock()
	return nil
}

// getIdleConnCh returns a channel to receive and return idle
// persistent connection for the given connectMethod.
// It may return nil, if persistent connections are not being used.
func (t *Transport) getIdleConnCh(cm connectMethod) chan *persistConn {
	if t.DisableKeepAlives {
		return nil
	}
	key := cm.key()
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	t.wantIdle = false
	if t.idleConnCh == nil {
		t.idleConnCh = make(map[connectMethodKey]chan *persistConn)
	}
	ch, ok := t.idleConnCh[key]
	if !ok {
		ch = make(chan *persistConn)
		t.idleConnCh[key] = ch
	}
	return ch
}

func (t *Transport) getIdleConn(cm connectMethod) (pconn *persistConn) {
	key := cm.key()
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	if t.idleConn == nil {
		return nil
	}
	for {
		pconns, ok := t.idleConn[key]
		if !ok {
			return nil
		}
		if len(pconns) == 1 {
			pconn = pconns[0]
			delete(t.idleConn, key)
		} else {
			// 2 or more cached connections; pop last
			// TODO: queue?
			pconn = pconns[len(pconns)-1]
			t.idleConn[key] = pconns[:len(pconns)-1]
		}
		if !pconn.isBroken() {
			return
		}
	}
}

func (t *Transport) setReqCanceler(r *Request, fn func()) {
	t.reqMu.Lock()
	defer t.reqMu.Unlock()
	if t.reqCanceler == nil {
		t.reqCanceler = make(map[*Request]func())
	}
	if fn != nil {
		t.reqCanceler[r] = fn
	} else {
		delete(t.reqCanceler, r)
	}
}

// replaceReqCanceler replaces an existing cancel function. If there is no cancel function
// for the request, we don't set the function and return false.
// Since CancelRequest will clear the canceler, we can use the return value to detect if
// the request was canceled since the last setReqCancel call.
func (t *Transport) replaceReqCanceler(r *Request, fn func()) bool {
	t.reqMu.Lock()
	defer t.reqMu.Unlock()
	_, ok := t.reqCanceler[r]
	if !ok {
		return false
	}
	if fn != nil {
		t.reqCanceler[r] = fn
	} else {
		delete(t.reqCanceler, r)
	}
	return true
}

func (t *Transport) dial(network, addr string) (net.Conn, error) {
	if t.Dial != nil {
		c, err := t.Dial(network, addr)
		if c == nil && err == nil {
			err = errors.New("net/http: Transport.Dial hook returned (nil, nil)")
		}
		return c, err
	}
	return net.Dial(network, addr)
}

// getConn dials and creates a new persistConn to the target as
// specified in the connectMethod.  This includes doing a proxy CONNECT
// and/or setting up TLS.  If this doesn't return an error, the persistConn
// is ready to write requests to.
func (t *Transport) getConn(req *Request, cm connectMethod) (*persistConn, error) {
	if pc := t.getIdleConn(cm); pc != nil {
		// set request canceler to some non-nil function so we
		// can detect whether it was cleared between now and when
		// we enter roundTrip
		t.setReqCanceler(req, func() {})
		return pc, nil
	}

	type dialRes struct {
		pc  *persistConn
		err error
	}
	dialc := make(chan dialRes)

	// Copy these hooks so we don't race on the postPendingDial in
	// the goroutine we launch. Issue 11136.
	testHookPrePendingDial := testHookPrePendingDial
	testHookPostPendingDial := testHookPostPendingDial

	handlePendingDial := func() {
		testHookPrePendingDial()
		go func() {
			if v := <-dialc; v.err == nil {
				t.putOrCloseIdleConn(v.pc)
			}
			testHookPostPendingDial()
		}()
	}

	cancelc := make(chan struct{})
	t.setReqCanceler(req, func() { close(cancelc) })

	go func() {
		pc, err := t.dialConn(cm)
		dialc <- dialRes{pc, err}
	}()

	idleConnCh := t.getIdleConnCh(cm)
	select {
	case v := <-dialc:
		// Our dial finished.
		return v.pc, v.err
	case pc := <-idleConnCh:
		// Another request finished first and its net.Conn
		// became available before our dial. Or somebody
		// else's dial that they didn't use.
		// But our dial is still going, so give it away
		// when it finishes:
		handlePendingDial()
		return pc, nil
	case <-req.Cancel:
		handlePendingDial()
		return nil, errRequestCanceledConn
	case <-cancelc:
		handlePendingDial()
		return nil, errRequestCanceledConn
	}
}

func (t *Transport) dialConn(cm connectMethod) (*persistConn, error) {
	pconn := &persistConn{
		t:          t,
		cacheKey:   cm.key(),
		reqch:      make(chan requestAndChan, 1),
		writech:    make(chan writeRequest, 1),
		closech:    make(chan struct{}),
		writeErrCh: make(chan error, 1),
	}
	tlsDial := t.DialTLS != nil && cm.targetScheme == "https" && cm.proxyURL == nil
	if tlsDial {
		var err error
		pconn.conn, err = t.DialTLS("tcp", cm.addr())
		if err != nil {
			return nil, err
		}
		if pconn.conn == nil {
			return nil, errors.New("net/http: Transport.DialTLS returned (nil, nil)")
		}
		if tc, ok := pconn.conn.(*tls.Conn); ok {
			// Handshake here, in case DialTLS didn't. TLSNextProto below
			// depends on it for knowing the connection state.
			if err := tc.Handshake(); err != nil {
				go pconn.conn.Close()
				return nil, err
			}
			cs := tc.ConnectionState()
			pconn.tlsState = &cs
		}
	} else {
		conn, err := t.dial("tcp", cm.addr())
		if err != nil {
			if cm.proxyURL != nil {
				err = fmt.Errorf("http: error connecting to proxy %s: %v", cm.proxyURL, err)
			}
			return nil, err
		}
		pconn.conn = conn
	}

	// Proxy setup.
	switch {
	case cm.proxyURL == nil:
		// Do nothing. Not using a proxy.
	case cm.targetScheme == "http":
		pconn.isProxy = true
		if pa := cm.proxyAuth(); pa != "" {
			pconn.mutateHeaderFunc = func(h Header) {
				h.Set("Proxy-Authorization", pa)
			}
		}
	case cm.targetScheme == "https":
		conn := pconn.conn
		connectReq := &Request{
			Method: "CONNECT",
			URL:    &url.URL{Opaque: cm.targetAddr},
			Host:   cm.targetAddr,
			Header: make(Header),
		}
		if pa := cm.proxyAuth(); pa != "" {
			connectReq.Header.Set("Proxy-Authorization", pa)
		}
		connectReq.Write(conn)

		// Read response.
		// Okay to use and discard buffered reader here, because
		// TLS server will not speak until spoken to.
		br := bufio.NewReader(conn)
		resp, err := ReadResponse(br, connectReq)
		if err != nil {
			conn.Close()
			return nil, err
		}
		if resp.StatusCode != 200 {
			f := strings.SplitN(resp.Status, " ", 2)
			conn.Close()
			return nil, errors.New(f[1])
		}
	}

	if cm.targetScheme == "https" && !tlsDial {
		// Initiate TLS and check remote host name against certificate.
		cfg := cloneTLSClientConfig(t.TLSClientConfig)
		if cfg.ServerName == "" {
			cfg.ServerName = cm.tlsHost()
		}
		plainConn := pconn.conn
		tlsConn := tls.Client(plainConn, cfg)
		errc := make(chan error, 2)
		var timer *time.Timer // for canceling TLS handshake
		if d := t.TLSHandshakeTimeout; d != 0 {
			timer = time.AfterFunc(d, func() {
				errc <- tlsHandshakeTimeoutError{}
			})
		}
		go func() {
			err := tlsConn.Handshake()
			if timer != nil {
				timer.Stop()
			}
			errc <- err
		}()
		if err := <-errc; err != nil {
			plainConn.Close()
			return nil, err
		}
		if !cfg.InsecureSkipVerify {
			if err := tlsConn.VerifyHostname(cfg.ServerName); err != nil {
				plainConn.Close()
				return nil, err
			}
		}
		cs := tlsConn.ConnectionState()
		pconn.tlsState = &cs
		pconn.conn = tlsConn
	}

	if s := pconn.tlsState; s != nil && s.NegotiatedProtocolIsMutual && s.NegotiatedProtocol != "" {
		if next, ok := t.TLSNextProto[s.NegotiatedProtocol]; ok {
			return &persistConn{alt: next(cm.targetAddr, pconn.conn.(*tls.Conn))}, nil
		}
	}

	pconn.br = bufio.NewReader(noteEOFReader{pconn.conn, &pconn.sawEOF})
	pconn.bw = bufio.NewWriter(pconn.conn)
	go pconn.readLoop()
	go pconn.writeLoop()
	return pconn, nil
}

// useProxy reports whether requests to addr should use a proxy,
// according to the NO_PROXY or no_proxy environment variable.
// addr is always a canonicalAddr with a host and port.
func useProxy(addr string) bool {
	if len(addr) == 0 {
		return true
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host == "localhost" {
		return false
	}
	if ip := net.ParseIP(host); ip != nil {
		if ip.IsLoopback() {
			return false
		}
	}

	no_proxy := noProxyEnv.Get()
	if no_proxy == "*" {
		return false
	}

	addr = strings.ToLower(strings.TrimSpace(addr))
	if hasPort(addr) {
		addr = addr[:strings.LastIndex(addr, ":")]
	}

	for _, p := range strings.Split(no_proxy, ",") {
		p = strings.ToLower(strings.TrimSpace(p))
		if len(p) == 0 {
			continue
		}
		if hasPort(p) {
			p = p[:strings.LastIndex(p, ":")]
		}
		if addr == p {
			return false
		}
		if p[0] == '.' && (strings.HasSuffix(addr, p) || addr == p[1:]) {
			// no_proxy ".foo.com" matches "bar.foo.com" or "foo.com"
			return false
		}
		if p[0] != '.' && strings.HasSuffix(addr, p) && addr[len(addr)-len(p)-1] == '.' {
			// no_proxy "foo.com" matches "bar.foo.com"
			return false
		}
	}
	return true
}

// connectMethod is the map key (in its String form) for keeping persistent
// TCP connections alive for subsequent HTTP requests.
//
// A connect method may be of the following types:
//
// Cache key form                Description
// -----------------             -------------------------
// |http|foo.com                 http directly to server, no proxy
// |https|foo.com                https directly to server, no proxy
// http://proxy.com|https|foo.com  http to proxy, then CONNECT to foo.com
// http://proxy.com|http           http to proxy, http to anywhere after that
//
// Note: no support to https to the proxy yet.
//
type connectMethod struct {
	proxyURL     *url.URL // nil for no proxy, else full proxy URL
	targetScheme string   // "http" or "https"
	targetAddr   string   // Not used if proxy + http targetScheme (4th example in table)
}

func (cm *connectMethod) key() connectMethodKey {
	proxyStr := ""
	targetAddr := cm.targetAddr
	if cm.proxyURL != nil {
		proxyStr = cm.proxyURL.String()
		if cm.targetScheme == "http" {
			targetAddr = ""
		}
	}
	return connectMethodKey{
		proxy:  proxyStr,
		scheme: cm.targetScheme,
		addr:   targetAddr,
	}
}

// addr returns the first hop "host:port" to which we need to TCP connect.
func (cm *connectMethod) addr() string {
	if cm.proxyURL != nil {
		return canonicalAddr(cm.proxyURL)
	}
	return cm.targetAddr
}

// tlsHost returns the host name to match against the peer's
// TLS certificate.
func (cm *connectMethod) tlsHost() string {
	h := cm.targetAddr
	if hasPort(h) {
		h = h[:strings.LastIndex(h, ":")]
	}
	return h
}

// connectMethodKey is the map key version of connectMethod, with a
// stringified proxy URL (or the empty string) instead of a pointer to
// a URL.
type connectMethodKey struct {
	proxy, scheme, addr string
}

func (k connectMethodKey) String() string {
	// Only used by tests.
	return fmt.Sprintf("%s|%s|%s", k.proxy, k.scheme, k.addr)
}

// persistConn wraps a connection, usually a persistent one
// (but may be used for non-keep-alive requests as well)
type persistConn struct {
	// alt optionally specifies the TLS NextProto RoundTripper.
	// This is used for HTTP/2 today and future protocol laters.
	// If it's non-nil, the rest of the fields are unused.
	alt RoundTripper

	t        *Transport
	cacheKey connectMethodKey
	conn     net.Conn
	tlsState *tls.ConnectionState
	br       *bufio.Reader       // from conn
	sawEOF   bool                // whether we've seen EOF from conn; owned by readLoop
	bw       *bufio.Writer       // to conn
	reqch    chan requestAndChan // written by roundTrip; read by readLoop
	writech  chan writeRequest   // written by roundTrip; read by writeLoop
	closech  chan struct{}       // closed when conn closed
	isProxy  bool
	// writeErrCh passes the request write error (usually nil)
	// from the writeLoop goroutine to the readLoop which passes
	// it off to the res.Body reader, which then uses it to decide
	// whether or not a connection can be reused. Issue 7569.
	writeErrCh chan error

	lk                   sync.Mutex // guards following fields
	numExpectedResponses int
	closed               error // set non-nil when conn is closed, before closech is closed
	broken               bool  // an error has happened on this connection; marked broken so it's not reused.
	canceled             bool  // whether this conn was broken due a CancelRequest
	reused               bool  // whether conn has had successful request/response and is being reused.
	// mutateHeaderFunc is an optional func to modify extra
	// headers on each outbound request before it's written. (the
	// original Request given to RoundTrip is not modified)
	mutateHeaderFunc func(Header)
}

// isBroken reports whether this connection is in a known broken state.
func (pc *persistConn) isBroken() bool {
	pc.lk.Lock()
	b := pc.broken
	pc.lk.Unlock()
	return b
}

// isCanceled reports whether this connection was closed due to CancelRequest.
func (pc *persistConn) isCanceled() bool {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	return pc.canceled
}

// isReused reports whether this connection is in a known broken state.
func (pc *persistConn) isReused() bool {
	pc.lk.Lock()
	r := pc.reused
	pc.lk.Unlock()
	return r
}

func (pc *persistConn) cancelRequest() {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	pc.canceled = true
	pc.closeLocked(errRequestCanceled)
}

func (pc *persistConn) readLoop() {
	closeErr := errReadLoopExiting // default value, if not changed below
	defer func() { pc.close(closeErr) }()

	tryPutIdleConn := func() bool {
		if err := pc.t.tryPutIdleConn(pc); err != nil {
			closeErr = err
			return false
		}
		return true
	}

	// eofc is used to block caller goroutines reading from Response.Body
	// at EOF until this goroutines has (potentially) added the connection
	// back to the idle pool.
	eofc := make(chan struct{})
	defer close(eofc) // unblock reader on errors

	// Read this once, before loop starts. (to avoid races in tests)
	testHookMu.Lock()
	testHookReadLoopBeforeNextRead := testHookReadLoopBeforeNextRead
	testHookMu.Unlock()

	alive := true
	for alive {
		_, err := pc.br.Peek(1)
		if err != nil {
			err = beforeRespHeaderError{err}
		}

		pc.lk.Lock()
		if pc.numExpectedResponses == 0 {
			pc.readLoopPeekFailLocked(err)
			pc.lk.Unlock()
			return
		}
		pc.lk.Unlock()

		rc := <-pc.reqch

		var resp *Response
		if err == nil {
			resp, err = pc.readResponse(rc)
		}

		if err != nil {
			// If we won't be able to retry this request later (from the
			// roundTrip goroutine), mark it as done now.
			// BEFORE the send on rc.ch, as the client might re-use the
			// same *Request pointer, and we don't want to set call
			// t.setReqCanceler from this persistConn while the Transport
			// potentially spins up a different persistConn for the
			// caller's subsequent request.
			if checkTransportResend(err, rc.req, pc) != nil {
				pc.t.setReqCanceler(rc.req, nil)
			}
			select {
			case rc.ch <- responseAndError{err: err}:
			case <-rc.callerGone:
				return
			}
			return
		}

		pc.lk.Lock()
		pc.numExpectedResponses--
		pc.lk.Unlock()

		hasBody := rc.req.Method != "HEAD" && resp.ContentLength != 0

		if resp.Close || rc.req.Close || resp.StatusCode <= 199 {
			// Don't do keep-alive on error if either party requested a close
			// or we get an unexpected informational (1xx) response.
			// StatusCode 100 is already handled above.
			alive = false
		}

		if !hasBody {
			pc.t.setReqCanceler(rc.req, nil)

			// Put the idle conn back into the pool before we send the response
			// so if they process it quickly and make another request, they'll
			// get this same conn. But we use the unbuffered channel 'rc'
			// to guarantee that persistConn.roundTrip got out of its select
			// potentially waiting for this persistConn to close.
			// but after
			alive = alive &&
				!pc.sawEOF &&
				pc.wroteRequest() &&
				tryPutIdleConn()

			select {
			case rc.ch <- responseAndError{res: resp}:
			case <-rc.callerGone:
				return
			}

			// Now that they've read from the unbuffered channel, they're safely
			// out of the select that also waits on this goroutine to die, so
			// we're allowed to exit now if needed (if alive is false)
			testHookReadLoopBeforeNextRead()
			continue
		}

		if rc.addedGzip {
			maybeUngzipResponse(resp)
		}
		resp.Body = &bodyEOFSignal{body: resp.Body}

		waitForBodyRead := make(chan bool, 2)
		resp.Body.(*bodyEOFSignal).earlyCloseFn = func() error {
			waitForBodyRead <- false
			return nil
		}
		resp.Body.(*bodyEOFSignal).fn = func(err error) error {
			isEOF := err == io.EOF
			waitForBodyRead <- isEOF
			if isEOF {
				<-eofc // see comment above eofc declaration
			} else if err != nil && pc.isCanceled() {
				return errRequestCanceled
			}
			return err
		}

		select {
		case rc.ch <- responseAndError{res: resp}:
		case <-rc.callerGone:
			return
		}

		// Before looping back to the top of this function and peeking on
		// the bufio.Reader, wait for the caller goroutine to finish
		// reading the response body. (or for cancelation or death)
		select {
		case bodyEOF := <-waitForBodyRead:
			pc.t.setReqCanceler(rc.req, nil) // before pc might return to idle pool
			alive = alive &&
				bodyEOF &&
				!pc.sawEOF &&
				pc.wroteRequest() &&
				tryPutIdleConn()
			if bodyEOF {
				eofc <- struct{}{}
			}
		case <-rc.req.Cancel:
			alive = false
			pc.t.CancelRequest(rc.req)
		case <-pc.closech:
			alive = false
		}

		testHookReadLoopBeforeNextRead()
	}
}

func maybeUngzipResponse(resp *Response) {
	if resp.Header.Get("Content-Encoding") == "gzip" {
		resp.Header.Del("Content-Encoding")
		resp.Header.Del("Content-Length")
		resp.ContentLength = -1
		resp.Body = &gzipReader{body: resp.Body}
	}
}

func (pc *persistConn) readLoopPeekFailLocked(peekErr error) {
	if pc.closed != nil {
		return
	}
	if n := pc.br.Buffered(); n > 0 {
		buf, _ := pc.br.Peek(n)
		log.Printf("Unsolicited response received on idle HTTP channel starting with %q; err=%v", buf, peekErr)
	}
	if peekErr == io.EOF {
		// common case.
		pc.closeLocked(errServerClosedIdle)
	} else {
		pc.closeLocked(fmt.Errorf("readLoopPeekFailLocked: %v", peekErr))
	}
}

// readResponse reads an HTTP response (or two, in the case of "Expect:
// 100-continue") from the server. It returns the final non-100 one.
func (pc *persistConn) readResponse(rc requestAndChan) (resp *Response, err error) {
	resp, err = ReadResponse(pc.br, rc.req)
	if err != nil {
		return
	}
	if rc.continueCh != nil {
		if resp.StatusCode == 100 {
			rc.continueCh <- struct{}{}
		} else {
			close(rc.continueCh)
		}
	}
	if resp.StatusCode == 100 {
		resp, err = ReadResponse(pc.br, rc.req)
		if err != nil {
			return
		}
	}
	resp.TLS = pc.tlsState
	return
}

// waitForContinue returns the function to block until
// any response, timeout or connection close. After any of them,
// the function returns a bool which indicates if the body should be sent.
func (pc *persistConn) waitForContinue(continueCh <-chan struct{}) func() bool {
	if continueCh == nil {
		return nil
	}
	return func() bool {
		timer := time.NewTimer(pc.t.ExpectContinueTimeout)
		defer timer.Stop()

		select {
		case _, ok := <-continueCh:
			return ok
		case <-timer.C:
			return true
		case <-pc.closech:
			return false
		}
	}
}

func (pc *persistConn) writeLoop() {
	for {
		select {
		case wr := <-pc.writech:
			if pc.isBroken() {
				wr.ch <- errors.New("http: can't write HTTP request on broken connection")
				continue
			}
			err := wr.req.Request.write(pc.bw, pc.isProxy, wr.req.extra, pc.waitForContinue(wr.continueCh))
			if err == nil {
				err = pc.bw.Flush()
			}
			if err != nil {
				pc.markBroken()
				wr.req.Request.closeBody()
			}
			pc.writeErrCh <- err // to the body reader, which might recycle us
			wr.ch <- err         // to the roundTrip function
		case <-pc.closech:
			return
		}
	}
}

// wroteRequest is a check before recycling a connection that the previous write
// (from writeLoop above) happened and was successful.
func (pc *persistConn) wroteRequest() bool {
	select {
	case err := <-pc.writeErrCh:
		// Common case: the write happened well before the response, so
		// avoid creating a timer.
		return err == nil
	default:
		// Rare case: the request was written in writeLoop above but
		// before it could send to pc.writeErrCh, the reader read it
		// all, processed it, and called us here. In this case, give the
		// write goroutine a bit of time to finish its send.
		//
		// Less rare case: We also get here in the legitimate case of
		// Issue 7569, where the writer is still writing (or stalled),
		// but the server has already replied. In this case, we don't
		// want to wait too long, and we want to return false so this
		// connection isn't re-used.
		select {
		case err := <-pc.writeErrCh:
			return err == nil
		case <-time.After(50 * time.Millisecond):
			return false
		}
	}
}

// responseAndError is how the goroutine reading from an HTTP/1 server
// communicates with the goroutine doing the RoundTrip.
type responseAndError struct {
	res *Response // else use this response (see res method)
	err error
}

type requestAndChan struct {
	req *Request
	ch  chan responseAndError // unbuffered; always send in select on callerGone

	// did the Transport (as opposed to the client code) add an
	// Accept-Encoding gzip header? only if it we set it do
	// we transparently decode the gzip.
	addedGzip bool

	// Optional blocking chan for Expect: 100-continue (for send).
	// If the request has an "Expect: 100-continue" header and
	// the server responds 100 Continue, readLoop send a value
	// to writeLoop via this chan.
	continueCh chan<- struct{}

	callerGone <-chan struct{} // closed when roundTrip caller has returned
}

// A writeRequest is sent by the readLoop's goroutine to the
// writeLoop's goroutine to write a request while the read loop
// concurrently waits on both the write response and the server's
// reply.
type writeRequest struct {
	req *transportRequest
	ch  chan<- error

	// Optional blocking chan for Expect: 100-continue (for recieve).
	// If not nil, writeLoop blocks sending request body until
	// it receives from this chan.
	continueCh <-chan struct{}
}

type httpError struct {
	err     string
	timeout bool
}

func (e *httpError) Error() string   { return e.err }
func (e *httpError) Timeout() bool   { return e.timeout }
func (e *httpError) Temporary() bool { return true }

var errTimeout error = &httpError{err: "net/http: timeout awaiting response headers", timeout: true}
var errClosed error = &httpError{err: "net/http: server closed connection before response was received"}
var errRequestCanceled = errors.New("net/http: request canceled")
var errRequestCanceledConn = errors.New("net/http: request canceled while waiting for connection") // TODO: unify?

func nop() {}

// testHooks. Always non-nil.
var (
	testHookEnterRoundTrip   = nop
	testHookWaitResLoop      = nop
	testHookRoundTripRetried = nop
	testHookPrePendingDial   = nop
	testHookPostPendingDial  = nop

	testHookMu                     sync.Locker = fakeLocker{} // guards following
	testHookReadLoopBeforeNextRead             = nop
)

// beforeRespHeaderError is used to indicate when an IO error has occurred before
// any header data was received.
type beforeRespHeaderError struct {
	error
}

func (pc *persistConn) roundTrip(req *transportRequest) (resp *Response, err error) {
	testHookEnterRoundTrip()
	if !pc.t.replaceReqCanceler(req.Request, pc.cancelRequest) {
		pc.t.putOrCloseIdleConn(pc)
		return nil, errRequestCanceled
	}
	pc.lk.Lock()
	pc.numExpectedResponses++
	headerFn := pc.mutateHeaderFunc
	pc.lk.Unlock()

	if headerFn != nil {
		headerFn(req.extraHeaders())
	}

	// Ask for a compressed version if the caller didn't set their
	// own value for Accept-Encoding. We only attempt to
	// uncompress the gzip stream if we were the layer that
	// requested it.
	requestedGzip := false
	if !pc.t.DisableCompression &&
		req.Header.Get("Accept-Encoding") == "" &&
		req.Header.Get("Range") == "" &&
		req.Method != "HEAD" {
		// Request gzip only, not deflate. Deflate is ambiguous and
		// not as universally supported anyway.
		// See: http://www.gzip.org/zlib/zlib_faq.html#faq38
		//
		// Note that we don't request this for HEAD requests,
		// due to a bug in nginx:
		//   http://trac.nginx.org/nginx/ticket/358
		//   https://golang.org/issue/5522
		//
		// We don't request gzip if the request is for a range, since
		// auto-decoding a portion of a gzipped document will just fail
		// anyway. See https://golang.org/issue/8923
		requestedGzip = true
		req.extraHeaders().Set("Accept-Encoding", "gzip")
	}

	var continueCh chan struct{}
	if req.ProtoAtLeast(1, 1) && req.Body != nil && req.expectsContinue() {
		continueCh = make(chan struct{}, 1)
	}

	if pc.t.DisableKeepAlives {
		req.extraHeaders().Set("Connection", "close")
	}

	gone := make(chan struct{})
	defer close(gone)

	// Write the request concurrently with waiting for a response,
	// in case the server decides to reply before reading our full
	// request body.
	writeErrCh := make(chan error, 1)
	pc.writech <- writeRequest{req, writeErrCh, continueCh}

	resc := make(chan responseAndError)
	pc.reqch <- requestAndChan{
		req:        req.Request,
		ch:         resc,
		addedGzip:  requestedGzip,
		continueCh: continueCh,
		callerGone: gone,
	}

	var re responseAndError
	var respHeaderTimer <-chan time.Time
	cancelChan := req.Request.Cancel
WaitResponse:
	for {
		testHookWaitResLoop()
		select {
		case err := <-writeErrCh:
			if err != nil {
				if pc.isCanceled() {
					err = errRequestCanceled
				}
				re = responseAndError{err: beforeRespHeaderError{err}}
				pc.close(fmt.Errorf("write error: %v", err))
				break WaitResponse
			}
			if d := pc.t.ResponseHeaderTimeout; d > 0 {
				timer := time.NewTimer(d)
				defer timer.Stop() // prevent leaks
				respHeaderTimer = timer.C
			}
		case <-pc.closech:
			var err error
			if pc.isCanceled() {
				err = errRequestCanceled
			} else {
				err = beforeRespHeaderError{fmt.Errorf("net/http: HTTP/1 transport connection broken: %v", pc.closed)}
			}
			re = responseAndError{err: err}
			break WaitResponse
		case <-respHeaderTimer:
			pc.close(errTimeout)
			re = responseAndError{err: errTimeout}
			break WaitResponse
		case re = <-resc:
			if re.err != nil && pc.isCanceled() {
				re.err = errRequestCanceled
			}
			break WaitResponse
		case <-cancelChan:
			pc.t.CancelRequest(req.Request)
			cancelChan = nil
		}
	}

	if re.err != nil {
		pc.t.setReqCanceler(req.Request, nil)
	}
	if (re.res == nil) == (re.err == nil) {
		panic("internal error: exactly one of res or err should be set")
	}
	return re.res, re.err
}

// markBroken marks a connection as broken (so it's not reused).
// It differs from close in that it doesn't close the underlying
// connection for use when it's still being read.
func (pc *persistConn) markBroken() {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	pc.broken = true
}

// markReused marks this connection as having been successfully used for a
// request and response.
func (pc *persistConn) markReused() {
	pc.lk.Lock()
	pc.reused = true
	pc.lk.Unlock()
}

// close closes the underlying TCP connection and closes
// the pc.closech channel.
//
// The provided err is only for testing and debugging; in normal
// circumstances it should never be seen by users.
func (pc *persistConn) close(err error) {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	pc.closeLocked(err)
}

func (pc *persistConn) closeLocked(err error) {
	if err == nil {
		panic("nil error")
	}
	pc.broken = true
	if pc.closed == nil {
		pc.closed = err
		if pc.alt != nil {
			// Do nothing; can only get here via getConn's
			// handlePendingDial's putOrCloseIdleConn when
			// it turns out the abandoned connection in
			// flight ended up negotiating an alternate
			// protocol.  We don't use the connection
			// freelist for http2. That's done by the
			// alternate protocol's RoundTripper.
		} else {
			pc.conn.Close()
			close(pc.closech)
		}
	}
	pc.mutateHeaderFunc = nil
}

var portMap = map[string]string{
	"http":  "80",
	"https": "443",
}

// canonicalAddr returns url.Host but always with a ":port" suffix
func canonicalAddr(url *url.URL) string {
	addr := url.Host
	if !hasPort(addr) {
		return addr + ":" + portMap[url.Scheme]
	}
	return addr
}

// bodyEOFSignal wraps a ReadCloser but runs fn (if non-nil) at most
// once, right before its final (error-producing) Read or Close call
// returns. fn should return the new error to return from Read or Close.
//
// If earlyCloseFn is non-nil and Close is called before io.EOF is
// seen, earlyCloseFn is called instead of fn, and its return value is
// the return value from Close.
type bodyEOFSignal struct {
	body         io.ReadCloser
	mu           sync.Mutex        // guards following 4 fields
	closed       bool              // whether Close has been called
	rerr         error             // sticky Read error
	fn           func(error) error // err will be nil on Read io.EOF
	earlyCloseFn func() error      // optional alt Close func used if io.EOF not seen
}

func (es *bodyEOFSignal) Read(p []byte) (n int, err error) {
	es.mu.Lock()
	closed, rerr := es.closed, es.rerr
	es.mu.Unlock()
	if closed {
		return 0, errors.New("http: read on closed response body")
	}
	if rerr != nil {
		return 0, rerr
	}

	n, err = es.body.Read(p)
	if err != nil {
		es.mu.Lock()
		defer es.mu.Unlock()
		if es.rerr == nil {
			es.rerr = err
		}
		err = es.condfn(err)
	}
	return
}

func (es *bodyEOFSignal) Close() error {
	es.mu.Lock()
	defer es.mu.Unlock()
	if es.closed {
		return nil
	}
	es.closed = true
	if es.earlyCloseFn != nil && es.rerr != io.EOF {
		return es.earlyCloseFn()
	}
	err := es.body.Close()
	return es.condfn(err)
}

// caller must hold es.mu.
func (es *bodyEOFSignal) condfn(err error) error {
	if es.fn == nil {
		return err
	}
	err = es.fn(err)
	es.fn = nil
	return err
}

// gzipReader wraps a response body so it can lazily
// call gzip.NewReader on the first call to Read
type gzipReader struct {
	body io.ReadCloser // underlying Response.Body
	zr   io.Reader     // lazily-initialized gzip reader
}

func (gz *gzipReader) Read(p []byte) (n int, err error) {
	if gz.zr == nil {
		gz.zr, err = gzip.NewReader(gz.body)
		if err != nil {
			return 0, err
		}
	}
	return gz.zr.Read(p)
}

func (gz *gzipReader) Close() error {
	return gz.body.Close()
}

type readerAndCloser struct {
	io.Reader
	io.Closer
}

type tlsHandshakeTimeoutError struct{}

func (tlsHandshakeTimeoutError) Timeout() bool   { return true }
func (tlsHandshakeTimeoutError) Temporary() bool { return true }
func (tlsHandshakeTimeoutError) Error() string   { return "net/http: TLS handshake timeout" }

type noteEOFReader struct {
	r      io.Reader
	sawEOF *bool
}

func (nr noteEOFReader) Read(p []byte) (n int, err error) {
	n, err = nr.r.Read(p)
	if err == io.EOF {
		*nr.sawEOF = true
	}
	return
}

// fakeLocker is a sync.Locker which does nothing. It's used to guard
// test-only fields when not under test, to avoid runtime atomic
// overhead.
type fakeLocker struct{}

func (fakeLocker) Lock()   {}
func (fakeLocker) Unlock() {}

func isNetWriteError(err error) bool {
	switch e := err.(type) {
	case *url.Error:
		return isNetWriteError(e.Err)
	case *net.OpError:
		return e.Op == "write"
	default:
		return false
	}
}

// cloneTLSConfig returns a shallow clone of the exported
// fields of cfg, ignoring the unexported sync.Once, which
// contains a mutex and must not be copied.
//
// The cfg must not be in active use by tls.Server, or else
// there can still be a race with tls.Server updating SessionTicketKey
// and our copying it, and also a race with the server setting
// SessionTicketsDisabled=false on failure to set the random
// ticket key.
//
// If cfg is nil, a new zero tls.Config is returned.
func cloneTLSConfig(cfg *tls.Config) *tls.Config {
	if cfg == nil {
		return &tls.Config{}
	}
	return &tls.Config{
		Rand:                     cfg.Rand,
		Time:                     cfg.Time,
		Certificates:             cfg.Certificates,
		NameToCertificate:        cfg.NameToCertificate,
		GetCertificate:           cfg.GetCertificate,
		RootCAs:                  cfg.RootCAs,
		NextProtos:               cfg.NextProtos,
		ServerName:               cfg.ServerName,
		ClientAuth:               cfg.ClientAuth,
		ClientCAs:                cfg.ClientCAs,
		InsecureSkipVerify:       cfg.InsecureSkipVerify,
		CipherSuites:             cfg.CipherSuites,
		PreferServerCipherSuites: cfg.PreferServerCipherSuites,
		SessionTicketsDisabled:   cfg.SessionTicketsDisabled,
		SessionTicketKey:         cfg.SessionTicketKey,
		ClientSessionCache:       cfg.ClientSessionCache,
		MinVersion:               cfg.MinVersion,
		MaxVersion:               cfg.MaxVersion,
		CurvePreferences:         cfg.CurvePreferences,
	}
}

// cloneTLSClientConfig is like cloneTLSConfig but omits
// the fields SessionTicketsDisabled and SessionTicketKey.
// This makes it safe to call cloneTLSClientConfig on a config
// in active use by a server.
func cloneTLSClientConfig(cfg *tls.Config) *tls.Config {
	if cfg == nil {
		return &tls.Config{}
	}
	return &tls.Config{
		Rand:                     cfg.Rand,
		Time:                     cfg.Time,
		Certificates:             cfg.Certificates,
		NameToCertificate:        cfg.NameToCertificate,
		GetCertificate:           cfg.GetCertificate,
		RootCAs:                  cfg.RootCAs,
		NextProtos:               cfg.NextProtos,
		ServerName:               cfg.ServerName,
		ClientAuth:               cfg.ClientAuth,
		ClientCAs:                cfg.ClientCAs,
		InsecureSkipVerify:       cfg.InsecureSkipVerify,
		CipherSuites:             cfg.CipherSuites,
		PreferServerCipherSuites: cfg.PreferServerCipherSuites,
		ClientSessionCache:       cfg.ClientSessionCache,
		MinVersion:               cfg.MinVersion,
		MaxVersion:               cfg.MaxVersion,
		CurvePreferences:         cfg.CurvePreferences,
	}
}
