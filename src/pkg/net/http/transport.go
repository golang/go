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
	"encoding/base64"
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
var DefaultTransport RoundTripper = &Transport{Proxy: ProxyFromEnvironment}

// DefaultMaxIdleConnsPerHost is the default value of Transport's
// MaxIdleConnsPerHost.
const DefaultMaxIdleConnsPerHost = 2

// Transport is an implementation of RoundTripper that supports http,
// https, and http proxies (for either http or https with CONNECT).
// Transport can also cache connections for future re-use.
type Transport struct {
	idleMu     sync.Mutex
	idleConn   map[string][]*persistConn
	idleConnCh map[string]chan *persistConn
	reqMu      sync.Mutex
	reqConn    map[*Request]*persistConn
	altMu      sync.RWMutex
	altProto   map[string]RoundTripper // nil or map of URI scheme => RoundTripper

	// Proxy specifies a function to return a proxy for a given
	// Request. If the function returns a non-nil error, the
	// request is aborted with the provided error.
	// If Proxy is nil or returns a nil *URL, no proxy is used.
	Proxy func(*Request) (*url.URL, error)

	// Dial specifies the dial function for creating TCP
	// connections.
	// If Dial is nil, net.Dial is used.
	Dial func(network, addr string) (net.Conn, error)

	// TLSClientConfig specifies the TLS configuration to use with
	// tls.Client. If nil, the default configuration is used.
	TLSClientConfig *tls.Config

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

	// TODO: tunable on global max cached connections
	// TODO: tunable on timeout on cached connections
}

// ProxyFromEnvironment returns the URL of the proxy to use for a
// given request, as indicated by the environment variables
// $HTTP_PROXY and $NO_PROXY (or $http_proxy and $no_proxy).
// An error is returned if the proxy environment is invalid.
// A nil URL and nil error are returned if no proxy is defined in the
// environment, or a proxy should not be used for the given request.
func ProxyFromEnvironment(req *Request) (*url.URL, error) {
	proxy := getenvEitherCase("HTTP_PROXY")
	if proxy == "" {
		return nil, nil
	}
	if !useProxy(canonicalAddr(req.URL)) {
		return nil, nil
	}
	proxyURL, err := url.Parse(proxy)
	if err != nil || !strings.HasPrefix(proxyURL.Scheme, "http") {
		if u, err := url.Parse("http://" + proxy); err == nil {
			proxyURL = u
			err = nil
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
func (t *Transport) RoundTrip(req *Request) (resp *Response, err error) {
	if req.URL == nil {
		return nil, errors.New("http: nil Request.URL")
	}
	if req.Header == nil {
		return nil, errors.New("http: nil Request.Header")
	}
	if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
		t.altMu.RLock()
		var rt RoundTripper
		if t.altProto != nil {
			rt = t.altProto[req.URL.Scheme]
		}
		t.altMu.RUnlock()
		if rt == nil {
			return nil, &badStringError{"unsupported protocol scheme", req.URL.Scheme}
		}
		return rt.RoundTrip(req)
	}
	if req.URL.Host == "" {
		return nil, errors.New("http: no Host in request URL")
	}
	treq := &transportRequest{Request: req}
	cm, err := t.connectMethodForRequest(treq)
	if err != nil {
		return nil, err
	}

	// Get the cached or newly-created connection to either the
	// host (for http or https), the http proxy, or the http proxy
	// pre-CONNECTed to https server.  In any case, we'll be ready
	// to send it requests.
	pconn, err := t.getConn(cm)
	if err != nil {
		return nil, err
	}

	return pconn.roundTrip(treq)
}

// RegisterProtocol registers a new protocol with scheme.
// The Transport will pass requests using the given scheme to rt.
// It is rt's responsibility to simulate HTTP request semantics.
//
// RegisterProtocol can be used by other packages to provide
// implementations of protocol schemes like "ftp" or "file".
func (t *Transport) RegisterProtocol(scheme string, rt RoundTripper) {
	if scheme == "http" || scheme == "https" {
		panic("protocol " + scheme + " already registered")
	}
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
	t.idleMu.Lock()
	m := t.idleConn
	t.idleConn = nil
	t.idleMu.Unlock()
	if m == nil {
		return
	}
	for _, conns := range m {
		for _, pconn := range conns {
			pconn.close()
		}
	}
}

// CancelRequest cancels an in-flight request by closing its
// connection.
func (t *Transport) CancelRequest(req *Request) {
	t.reqMu.Lock()
	pc := t.reqConn[req]
	t.reqMu.Unlock()
	if pc != nil {
		pc.conn.Close()
	}
}

//
// Private implementation past this point.
//

func getenvEitherCase(k string) string {
	if v := os.Getenv(strings.ToUpper(k)); v != "" {
		return v
	}
	return os.Getenv(strings.ToLower(k))
}

func (t *Transport) connectMethodForRequest(treq *transportRequest) (*connectMethod, error) {
	cm := &connectMethod{
		targetScheme: treq.URL.Scheme,
		targetAddr:   canonicalAddr(treq.URL),
	}
	if t.Proxy != nil {
		var err error
		cm.proxyURL, err = t.Proxy(treq.Request)
		if err != nil {
			return nil, err
		}
	}
	return cm, nil
}

// proxyAuth returns the Proxy-Authorization header to set
// on requests, if applicable.
func (cm *connectMethod) proxyAuth() string {
	if cm.proxyURL == nil {
		return ""
	}
	if u := cm.proxyURL.User; u != nil {
		return "Basic " + base64.URLEncoding.EncodeToString([]byte(u.String()))
	}
	return ""
}

// putIdleConn adds pconn to the list of idle persistent connections awaiting
// a new request.
// If pconn is no longer needed or not in a good state, putIdleConn
// returns false.
func (t *Transport) putIdleConn(pconn *persistConn) bool {
	if t.DisableKeepAlives || t.MaxIdleConnsPerHost < 0 {
		pconn.close()
		return false
	}
	if pconn.isBroken() {
		return false
	}
	key := pconn.cacheKey
	max := t.MaxIdleConnsPerHost
	if max == 0 {
		max = DefaultMaxIdleConnsPerHost
	}
	t.idleMu.Lock()
	select {
	case t.idleConnCh[key] <- pconn:
		// We're done with this pconn and somebody else is
		// currently waiting for a conn of this type (they're
		// actively dialing, but this conn is ready
		// first). Chrome calls this socket late binding.  See
		// https://insouciant.org/tech/connection-management-in-chromium/
		t.idleMu.Unlock()
		return true
	default:
	}
	if t.idleConn == nil {
		t.idleConn = make(map[string][]*persistConn)
	}
	if len(t.idleConn[key]) >= max {
		t.idleMu.Unlock()
		pconn.close()
		return false
	}
	for _, exist := range t.idleConn[key] {
		if exist == pconn {
			log.Fatalf("dup idle pconn %p in freelist", pconn)
		}
	}
	t.idleConn[key] = append(t.idleConn[key], pconn)
	t.idleMu.Unlock()
	return true
}

func (t *Transport) getIdleConnCh(cm *connectMethod) chan *persistConn {
	key := cm.key()
	t.idleMu.Lock()
	defer t.idleMu.Unlock()
	if t.idleConnCh == nil {
		t.idleConnCh = make(map[string]chan *persistConn)
	}
	ch, ok := t.idleConnCh[key]
	if !ok {
		ch = make(chan *persistConn)
		t.idleConnCh[key] = ch
	}
	return ch
}

func (t *Transport) getIdleConn(cm *connectMethod) (pconn *persistConn) {
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
			t.idleConn[key] = pconns[0 : len(pconns)-1]
		}
		if !pconn.isBroken() {
			return
		}
	}
}

func (t *Transport) setReqConn(r *Request, pc *persistConn) {
	t.reqMu.Lock()
	defer t.reqMu.Unlock()
	if t.reqConn == nil {
		t.reqConn = make(map[*Request]*persistConn)
	}
	if pc != nil {
		t.reqConn[r] = pc
	} else {
		delete(t.reqConn, r)
	}
}

func (t *Transport) dial(network, addr string) (c net.Conn, err error) {
	if t.Dial != nil {
		return t.Dial(network, addr)
	}
	return net.Dial(network, addr)
}

// getConn dials and creates a new persistConn to the target as
// specified in the connectMethod.  This includes doing a proxy CONNECT
// and/or setting up TLS.  If this doesn't return an error, the persistConn
// is ready to write requests to.
func (t *Transport) getConn(cm *connectMethod) (*persistConn, error) {
	if pc := t.getIdleConn(cm); pc != nil {
		return pc, nil
	}

	type dialRes struct {
		pc  *persistConn
		err error
	}
	dialc := make(chan dialRes)
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
		go func() {
			if v := <-dialc; v.err == nil {
				t.putIdleConn(v.pc)
			}
		}()
		return pc, nil
	}
}

func (t *Transport) dialConn(cm *connectMethod) (*persistConn, error) {
	conn, err := t.dial("tcp", cm.addr())
	if err != nil {
		if cm.proxyURL != nil {
			err = fmt.Errorf("http: error connecting to proxy %s: %v", cm.proxyURL, err)
		}
		return nil, err
	}

	pa := cm.proxyAuth()

	pconn := &persistConn{
		t:        t,
		cacheKey: cm.key(),
		conn:     conn,
		reqch:    make(chan requestAndChan, 50),
		writech:  make(chan writeRequest, 50),
		closech:  make(chan struct{}),
	}

	switch {
	case cm.proxyURL == nil:
		// Do nothing.
	case cm.targetScheme == "http":
		pconn.isProxy = true
		if pa != "" {
			pconn.mutateHeaderFunc = func(h Header) {
				h.Set("Proxy-Authorization", pa)
			}
		}
	case cm.targetScheme == "https":
		connectReq := &Request{
			Method: "CONNECT",
			URL:    &url.URL{Opaque: cm.targetAddr},
			Host:   cm.targetAddr,
			Header: make(Header),
		}
		if pa != "" {
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

	if cm.targetScheme == "https" {
		// Initiate TLS and check remote host name against certificate.
		cfg := t.TLSClientConfig
		if cfg == nil || cfg.ServerName == "" {
			host := cm.tlsHost()
			if cfg == nil {
				cfg = &tls.Config{ServerName: host}
			} else {
				clone := *cfg // shallow clone
				clone.ServerName = host
				cfg = &clone
			}
		}
		conn = tls.Client(conn, cfg)
		if err = conn.(*tls.Conn).Handshake(); err != nil {
			return nil, err
		}
		if t.TLSClientConfig == nil || !t.TLSClientConfig.InsecureSkipVerify {
			if err = conn.(*tls.Conn).VerifyHostname(cm.tlsHost()); err != nil {
				return nil, err
			}
		}
		pconn.conn = conn
	}

	pconn.br = bufio.NewReader(pconn.conn)
	pconn.bw = bufio.NewWriter(pconn.conn)
	go pconn.readLoop()
	go pconn.writeLoop()
	return pconn, nil
}

// useProxy returns true if requests to addr should use a proxy,
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

	no_proxy := getenvEitherCase("NO_PROXY")
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
// ||http|foo.com                http directly to server, no proxy
// ||https|foo.com               https directly to server, no proxy
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

func (ck *connectMethod) key() string {
	return ck.String() // TODO: use a struct type instead
}

func (ck *connectMethod) String() string {
	proxyStr := ""
	targetAddr := ck.targetAddr
	if ck.proxyURL != nil {
		proxyStr = ck.proxyURL.String()
		if ck.targetScheme == "http" {
			targetAddr = ""
		}
	}
	return strings.Join([]string{proxyStr, ck.targetScheme, targetAddr}, "|")
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

// persistConn wraps a connection, usually a persistent one
// (but may be used for non-keep-alive requests as well)
type persistConn struct {
	t        *Transport
	cacheKey string // its connectMethod.String()
	conn     net.Conn
	closed   bool                // whether conn has been closed
	br       *bufio.Reader       // from conn
	bw       *bufio.Writer       // to conn
	reqch    chan requestAndChan // written by roundTrip; read by readLoop
	writech  chan writeRequest   // written by roundTrip; read by writeLoop
	closech  chan struct{}       // broadcast close when readLoop (TCP connection) closes
	isProxy  bool

	lk                   sync.Mutex // guards following 3 fields
	numExpectedResponses int
	broken               bool // an error has happened on this connection; marked broken so it's not reused.
	// mutateHeaderFunc is an optional func to modify extra
	// headers on each outbound request before it's written. (the
	// original Request given to RoundTrip is not modified)
	mutateHeaderFunc func(Header)
}

func (pc *persistConn) isBroken() bool {
	pc.lk.Lock()
	b := pc.broken
	pc.lk.Unlock()
	return b
}

var remoteSideClosedFunc func(error) bool // or nil to use default

func remoteSideClosed(err error) bool {
	if err == io.EOF {
		return true
	}
	if remoteSideClosedFunc != nil {
		return remoteSideClosedFunc(err)
	}
	return false
}

func (pc *persistConn) readLoop() {
	defer close(pc.closech)
	alive := true

	for alive {
		pb, err := pc.br.Peek(1)

		pc.lk.Lock()
		if pc.numExpectedResponses == 0 {
			pc.closeLocked()
			pc.lk.Unlock()
			if len(pb) > 0 {
				log.Printf("Unsolicited response received on idle HTTP channel starting with %q; err=%v",
					string(pb), err)
			}
			return
		}
		pc.lk.Unlock()

		rc := <-pc.reqch

		var resp *Response
		if err == nil {
			resp, err = ReadResponse(pc.br, rc.req)
			if err == nil && resp.StatusCode == 100 {
				// Skip any 100-continue for now.
				// TODO(bradfitz): if rc.req had "Expect: 100-continue",
				// actually block the request body write and signal the
				// writeLoop now to begin sending it. (Issue 2184) For now we
				// eat it, since we're never expecting one.
				resp, err = ReadResponse(pc.br, rc.req)
			}
		}
		hasBody := resp != nil && rc.req.Method != "HEAD" && resp.ContentLength != 0

		if err != nil {
			pc.close()
		} else {
			if rc.addedGzip && hasBody && resp.Header.Get("Content-Encoding") == "gzip" {
				resp.Header.Del("Content-Encoding")
				resp.Header.Del("Content-Length")
				resp.ContentLength = -1
				gzReader, zerr := gzip.NewReader(resp.Body)
				if zerr != nil {
					pc.close()
					err = zerr
				} else {
					resp.Body = &readerAndCloser{gzReader, resp.Body}
				}
			}
			resp.Body = &bodyEOFSignal{body: resp.Body}
		}

		if err != nil || resp.Close || rc.req.Close || resp.StatusCode <= 199 {
			// Don't do keep-alive on error if either party requested a close
			// or we get an unexpected informational (1xx) response.
			// StatusCode 100 is already handled above.
			alive = false
		}

		var waitForBodyRead chan bool
		if hasBody {
			waitForBodyRead = make(chan bool, 2)
			resp.Body.(*bodyEOFSignal).earlyCloseFn = func() error {
				// Sending false here sets alive to
				// false and closes the connection
				// below.
				waitForBodyRead <- false
				return nil
			}
			resp.Body.(*bodyEOFSignal).fn = func(err error) {
				alive1 := alive
				if err != nil {
					alive1 = false
				}
				if alive1 && !pc.t.putIdleConn(pc) {
					alive1 = false
				}
				if !alive1 || pc.isBroken() {
					pc.close()
				}
				waitForBodyRead <- alive1
			}
		}

		if alive && !hasBody {
			if !pc.t.putIdleConn(pc) {
				alive = false
			}
		}

		rc.ch <- responseAndError{resp, err}

		// Wait for the just-returned response body to be fully consumed
		// before we race and peek on the underlying bufio reader.
		if waitForBodyRead != nil {
			alive = <-waitForBodyRead
		}

		pc.t.setReqConn(rc.req, nil)

		if !alive {
			pc.close()
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
			err := wr.req.Request.write(pc.bw, pc.isProxy, wr.req.extra)
			if err == nil {
				err = pc.bw.Flush()
			}
			if err != nil {
				pc.markBroken()
			}
			wr.ch <- err
		case <-pc.closech:
			return
		}
	}
}

type responseAndError struct {
	res *Response
	err error
}

type requestAndChan struct {
	req *Request
	ch  chan responseAndError

	// did the Transport (as opposed to the client code) add an
	// Accept-Encoding gzip header? only if it we set it do
	// we transparently decode the gzip.
	addedGzip bool
}

// A writeRequest is sent by the readLoop's goroutine to the
// writeLoop's goroutine to write a request while the read loop
// concurrently waits on both the write response and the server's
// reply.
type writeRequest struct {
	req *transportRequest
	ch  chan<- error
}

func (pc *persistConn) roundTrip(req *transportRequest) (resp *Response, err error) {
	pc.t.setReqConn(req.Request, pc)
	pc.lk.Lock()
	pc.numExpectedResponses++
	headerFn := pc.mutateHeaderFunc
	pc.lk.Unlock()

	if headerFn != nil {
		headerFn(req.extraHeaders())
	}

	// Ask for a compressed version if the caller didn't set their
	// own value for Accept-Encoding. We only attempted to
	// uncompress the gzip stream if we were the layer that
	// requested it.
	requestedGzip := false
	if !pc.t.DisableCompression && req.Header.Get("Accept-Encoding") == "" {
		// Request gzip only, not deflate. Deflate is ambiguous and
		// not as universally supported anyway.
		// See: http://www.gzip.org/zlib/zlib_faq.html#faq38
		requestedGzip = true
		req.extraHeaders().Set("Accept-Encoding", "gzip")
	}

	// Write the request concurrently with waiting for a response,
	// in case the server decides to reply before reading our full
	// request body.
	writeErrCh := make(chan error, 1)
	pc.writech <- writeRequest{req, writeErrCh}

	resc := make(chan responseAndError, 1)
	pc.reqch <- requestAndChan{req.Request, resc, requestedGzip}

	var re responseAndError
	var pconnDeadCh = pc.closech
	var failTicker <-chan time.Time
	var respHeaderTimer <-chan time.Time
WaitResponse:
	for {
		select {
		case err := <-writeErrCh:
			if err != nil {
				re = responseAndError{nil, err}
				pc.close()
				break WaitResponse
			}
			if d := pc.t.ResponseHeaderTimeout; d > 0 {
				respHeaderTimer = time.After(d)
			}
		case <-pconnDeadCh:
			// The persist connection is dead. This shouldn't
			// usually happen (only with Connection: close responses
			// with no response bodies), but if it does happen it
			// means either a) the remote server hung up on us
			// prematurely, or b) the readLoop sent us a response &
			// closed its closech at roughly the same time, and we
			// selected this case first, in which case a response
			// might still be coming soon.
			//
			// We can't avoid the select race in b) by using a unbuffered
			// resc channel instead, because then goroutines can
			// leak if we exit due to other errors.
			pconnDeadCh = nil                               // avoid spinning
			failTicker = time.After(100 * time.Millisecond) // arbitrary time to wait for resc
		case <-failTicker:
			re = responseAndError{err: errors.New("net/http: transport closed before response was received")}
			break WaitResponse
		case <-respHeaderTimer:
			pc.close()
			re = responseAndError{err: errors.New("net/http: timeout awaiting response headers")}
			break WaitResponse
		case re = <-resc:
			break WaitResponse
		}
	}

	pc.lk.Lock()
	pc.numExpectedResponses--
	pc.lk.Unlock()

	if re.err != nil {
		pc.t.setReqConn(req.Request, nil)
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

func (pc *persistConn) close() {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	pc.closeLocked()
}

func (pc *persistConn) closeLocked() {
	pc.broken = true
	if !pc.closed {
		pc.conn.Close()
		pc.closed = true
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
// returns. If earlyCloseFn is non-nil and Close is called before
// io.EOF is seen, earlyCloseFn is called instead of fn, and its
// return value is the return value from Close.
type bodyEOFSignal struct {
	body         io.ReadCloser
	mu           sync.Mutex   // guards following 4 fields
	closed       bool         // whether Close has been called
	rerr         error        // sticky Read error
	fn           func(error)  // error will be nil on Read io.EOF
	earlyCloseFn func() error // optional alt Close func used if io.EOF not seen
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
		es.condfn(err)
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
	es.condfn(err)
	return err
}

// caller must hold es.mu.
func (es *bodyEOFSignal) condfn(err error) {
	if es.fn == nil {
		return
	}
	if err == io.EOF {
		err = nil
	}
	es.fn(err)
	es.fn = nil
}

type readerAndCloser struct {
	io.Reader
	io.Closer
}
