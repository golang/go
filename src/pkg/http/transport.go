// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync"
)

// DefaultTransport is the default implementation of Transport and is
// used by DefaultClient.  It establishes a new network connection for
// each call to Do and uses HTTP proxies as directed by the
// $HTTP_PROXY and $NO_PROXY (or $http_proxy and $no_proxy)
// environment variables.
var DefaultTransport RoundTripper = &Transport{}

// DefaultMaxIdleConnsPerHost is the default value of Transport's
// MaxIdleConnsPerHost.
const DefaultMaxIdleConnsPerHost = 2

// Transport is an implementation of RoundTripper that supports http,
// https, and http proxies (for either http or https with CONNECT).
// Transport can also cache connections for future re-use.
type Transport struct {
	lk       sync.Mutex
	idleConn map[string][]*persistConn

	// TODO: tunable on global max cached connections
	// TODO: tunable on timeout on cached connections
	// TODO: optional pipelining

	IgnoreEnvironment  bool // don't look at environment variables for proxy configuration
	DisableKeepAlives  bool
	DisableCompression bool

	// MaxIdleConnsPerHost, if non-zero, controls the maximum idle
	// (keep-alive) to keep to keep per-host.  If zero,
	// DefaultMaxIdleConnsPerHost is used.
	MaxIdleConnsPerHost int
}

// RoundTrip implements the RoundTripper interface.
func (t *Transport) RoundTrip(req *Request) (resp *Response, err os.Error) {
	if req.URL == nil {
		if req.URL, err = ParseURL(req.RawURL); err != nil {
			return
		}
	}
	if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
		return nil, &badStringError{"unsupported protocol scheme", req.URL.Scheme}
	}

	cm, err := t.connectMethodForRequest(req)
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

	return pconn.roundTrip(req)
}

// CloseIdleConnections closes any connections which were previously
// connected from previous requests but are now sitting idle in
// a "keep-alive" state. It does not interrupt any connections currently
// in use.
func (t *Transport) CloseIdleConnections() {
	t.lk.Lock()
	defer t.lk.Unlock()
	if t.idleConn == nil {
		return
	}
	for _, conns := range t.idleConn {
		for _, pconn := range conns {
			pconn.close()
		}
	}
	t.idleConn = nil
}

//
// Private implementation past this point.
//

func (t *Transport) getenvEitherCase(k string) string {
	if t.IgnoreEnvironment {
		return ""
	}
	if v := t.getenv(strings.ToUpper(k)); v != "" {
		return v
	}
	return t.getenv(strings.ToLower(k))
}

func (t *Transport) getenv(k string) string {
	if t.IgnoreEnvironment {
		return ""
	}
	return os.Getenv(k)
}

func (t *Transport) connectMethodForRequest(req *Request) (*connectMethod, os.Error) {
	cm := &connectMethod{
		targetScheme: req.URL.Scheme,
		targetAddr:   canonicalAddr(req.URL),
	}

	proxy := t.getenvEitherCase("HTTP_PROXY")
	if proxy != "" && t.useProxy(cm.targetAddr) {
		proxyURL, err := ParseRequestURL(proxy)
		if err != nil {
			return nil, os.ErrorString("invalid proxy address")
		}
		if proxyURL.Host == "" {
			proxyURL, err = ParseRequestURL("http://" + proxy)
			if err != nil {
				return nil, os.ErrorString("invalid proxy address")
			}
		}
		cm.proxyURL = proxyURL
	}
	return cm, nil
}

// proxyAuth returns the Proxy-Authorization header to set
// on requests, if applicable.
func (cm *connectMethod) proxyAuth() string {
	if cm.proxyURL == nil {
		return ""
	}
	proxyInfo := cm.proxyURL.RawUserinfo
	if proxyInfo != "" {
		enc := base64.URLEncoding
		encoded := make([]byte, enc.EncodedLen(len(proxyInfo)))
		enc.Encode(encoded, []byte(proxyInfo))
		return "Basic " + string(encoded)
	}
	return ""
}

func (t *Transport) putIdleConn(pconn *persistConn) {
	t.lk.Lock()
	defer t.lk.Unlock()
	if t.DisableKeepAlives || t.MaxIdleConnsPerHost < 0 {
		pconn.close()
		return
	}
	if pconn.isBroken() {
		return
	}
	key := pconn.cacheKey
	max := t.MaxIdleConnsPerHost
	if max == 0 {
		max = DefaultMaxIdleConnsPerHost
	}
	if len(t.idleConn[key]) >= max {
		pconn.close()
		return
	}
	t.idleConn[key] = append(t.idleConn[key], pconn)
}

func (t *Transport) getIdleConn(cm *connectMethod) (pconn *persistConn) {
	t.lk.Lock()
	defer t.lk.Unlock()
	if t.idleConn == nil {
		t.idleConn = make(map[string][]*persistConn)
	}
	key := cm.String()
	for {
		pconns, ok := t.idleConn[key]
		if !ok {
			return nil
		}
		if len(pconns) == 1 {
			pconn = pconns[0]
			t.idleConn[key] = nil, false
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
	return
}

// getConn dials and creates a new persistConn to the target as
// specified in the connectMethod.  This includes doing a proxy CONNECT
// and/or setting up TLS.  If this doesn't return an error, the persistConn
// is ready to write requests to.
func (t *Transport) getConn(cm *connectMethod) (*persistConn, os.Error) {
	if pc := t.getIdleConn(cm); pc != nil {
		return pc, nil
	}

	conn, err := net.Dial("tcp", cm.addr())
	if err != nil {
		if cm.proxyURL != nil {
			err = fmt.Errorf("http: error connecting to proxy %s: %v", cm.proxyURL, err)
		}
		return nil, err
	}

	pa := cm.proxyAuth()

	pconn := &persistConn{
		t:        t,
		cacheKey: cm.String(),
		conn:     conn,
		reqch:    make(chan requestAndChan, 50),
	}
	newClientConnFunc := NewClientConn

	switch {
	case cm.proxyURL == nil:
		// Do nothing.
	case cm.targetScheme == "http":
		newClientConnFunc = NewProxyClientConn
		if pa != "" {
			pconn.mutateRequestFunc = func(req *Request) {
				if req.Header == nil {
					req.Header = make(Header)
				}
				req.Header.Set("Proxy-Authorization", pa)
			}
		}
	case cm.targetScheme == "https":
		fmt.Fprintf(conn, "CONNECT %s HTTP/1.1\r\n", cm.targetAddr)
		fmt.Fprintf(conn, "Host: %s\r\n", cm.targetAddr)
		if pa != "" {
			fmt.Fprintf(conn, "Proxy-Authorization: %s\r\n", pa)
		}
		fmt.Fprintf(conn, "\r\n")

		// Read response.
		// Okay to use and discard buffered reader here, because
		// TLS server will not speak until spoken to.
		br := bufio.NewReader(conn)
		resp, err := ReadResponse(br, "CONNECT")
		if err != nil {
			conn.Close()
			return nil, err
		}
		if resp.StatusCode != 200 {
			f := strings.Split(resp.Status, " ", 2)
			conn.Close()
			return nil, os.ErrorString(f[1])
		}
	}

	if cm.targetScheme == "https" {
		// Initiate TLS and check remote host name against certificate.
		conn = tls.Client(conn, nil)
		if err = conn.(*tls.Conn).Handshake(); err != nil {
			return nil, err
		}
		if err = conn.(*tls.Conn).VerifyHostname(cm.tlsHost()); err != nil {
			return nil, err
		}
		pconn.conn = conn
	}

	pconn.br = bufio.NewReader(pconn.conn)
	pconn.cc = newClientConnFunc(conn, pconn.br)
	pconn.cc.readRes = readResponseWithEOFSignal
	go pconn.readLoop()
	return pconn, nil
}

// useProxy returns true if requests to addr should use a proxy,
// according to the NO_PROXY or no_proxy environment variable.
// addr is always a canonicalAddr with a host and port.
func (t *Transport) useProxy(addr string) bool {
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
		if ip4 := ip.To4(); ip4 != nil && ip4[0] == 127 {
			// 127.0.0.0/8 loopback isn't proxied.
			return false
		}
		if bytes.Equal(ip, net.IPv6loopback) {
			return false
		}
	}

	no_proxy := t.getenvEitherCase("NO_PROXY")
	if no_proxy == "*" {
		return false
	}

	addr = strings.ToLower(strings.TrimSpace(addr))
	if hasPort(addr) {
		addr = addr[:strings.LastIndex(addr, ":")]
	}

	for _, p := range strings.Split(no_proxy, ",", -1) {
		p = strings.ToLower(strings.TrimSpace(p))
		if len(p) == 0 {
			continue
		}
		if hasPort(p) {
			p = p[:strings.LastIndex(p, ":")]
		}
		if addr == p || (p[0] == '.' && (strings.HasSuffix(addr, p) || addr == p[1:])) {
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
	proxyURL     *URL   // "" for no proxy, else full proxy URL
	targetScheme string // "http" or "https"
	targetAddr   string // Not used if proxy + http targetScheme (4th example in table)
}

func (ck *connectMethod) String() string {
	proxyStr := ""
	if ck.proxyURL != nil {
		proxyStr = ck.proxyURL.String()
	}
	return strings.Join([]string{proxyStr, ck.targetScheme, ck.targetAddr}, "|")
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

type readResult struct {
	res *Response // either res or err will be set
	err os.Error
}

type writeRequest struct {
	// Set by client (in pc.roundTrip)
	req   *Request
	resch chan *readResult

	// Set by writeLoop if an error writing headers.
	writeErr os.Error
}

// persistConn wraps a connection, usually a persistent one
// (but may be used for non-keep-alive requests as well)
type persistConn struct {
	t                 *Transport
	cacheKey          string // its connectMethod.String()
	conn              net.Conn
	cc                *ClientConn
	br                *bufio.Reader
	reqch             chan requestAndChan // written by roundTrip(); read by readLoop()
	mutateRequestFunc func(*Request)      // nil or func to modify each outbound request

	lk                   sync.Mutex // guards numExpectedResponses and broken
	numExpectedResponses int
	broken               bool // an error has happened on this connection; marked broken so it's not reused.
}

func (pc *persistConn) isBroken() bool {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	return pc.broken
}

func (pc *persistConn) expectingResponse() bool {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	return pc.numExpectedResponses > 0
}

func (pc *persistConn) readLoop() {
	alive := true
	for alive {
		pb, err := pc.br.Peek(1)
		if err != nil {
			if (err == os.EOF || err == os.EINVAL) && !pc.expectingResponse() {
				// Remote side closed on us.  (We probably hit their
				// max idle timeout)
				pc.close()
				return
			}
		}
		if !pc.expectingResponse() {
			log.Printf("Unsolicited response received on idle HTTP channel starting with %q; err=%v",
				string(pb), err)
			pc.close()
			return
		}

		rc := <-pc.reqch
		resp, err := pc.cc.Read(rc.req)

		if err == ErrPersistEOF {
			// Succeeded, but we can't send any more
			// persistent connections on this again.  We
			// hide this error to upstream callers.
			alive = false
			err = nil
		} else if err != nil || rc.req.Close {
			alive = false
		}

		hasBody := resp != nil && resp.ContentLength != 0
		var waitForBodyRead chan bool
		if alive {
			if hasBody {
				waitForBodyRead = make(chan bool)
				resp.Body.(*bodyEOFSignal).fn = func() {
					pc.t.putIdleConn(pc)
					waitForBodyRead <- true
				}
			} else {
				pc.t.putIdleConn(pc)
			}
		}

		rc.ch <- responseAndError{resp, err}

		// Wait for the just-returned response body to be fully consumed
		// before we race and peek on the underlying bufio reader.
		if waitForBodyRead != nil {
			<-waitForBodyRead
		}
	}
}

type responseAndError struct {
	res *Response
	err os.Error
}

type requestAndChan struct {
	req *Request
	ch  chan responseAndError
}

func (pc *persistConn) roundTrip(req *Request) (resp *Response, err os.Error) {
	if pc.mutateRequestFunc != nil {
		pc.mutateRequestFunc(req)
	}

	// Ask for a compressed version if the caller didn't set their
	// own value for Accept-Encoding. We only attempted to
	// uncompress the gzip stream if we were the layer that
	// requested it.
	requestedGzip := false
	if !pc.t.DisableCompression && req.Header.Get("Accept-Encoding") == "" {
		// Request gzip only, not deflate. Deflate is ambiguous and 
		// as universally supported anyway.
		// See: http://www.gzip.org/zlib/zlib_faq.html#faq38
		requestedGzip = true
		req.Header.Set("Accept-Encoding", "gzip")
	}

	pc.lk.Lock()
	pc.numExpectedResponses++
	pc.lk.Unlock()

	err = pc.cc.Write(req)
	if err != nil {
		pc.close()
		return
	}

	ch := make(chan responseAndError, 1)
	pc.reqch <- requestAndChan{req, ch}
	re := <-ch
	pc.lk.Lock()
	pc.numExpectedResponses--
	pc.lk.Unlock()

	if re.err == nil && requestedGzip && re.res.Header.Get("Content-Encoding") == "gzip" {
		re.res.Header.Del("Content-Encoding")
		re.res.Header.Del("Content-Length")
		re.res.ContentLength = -1
		esb := re.res.Body.(*bodyEOFSignal)
		gzReader, err := gzip.NewReader(esb.body)
		if err != nil {
			pc.close()
			return nil, err
		}
		esb.body = &readFirstCloseBoth{gzReader, esb.body}
	}

	return re.res, re.err
}

func (pc *persistConn) close() {
	pc.lk.Lock()
	defer pc.lk.Unlock()
	pc.broken = true
	pc.cc.Close()
	pc.conn.Close()
	pc.mutateRequestFunc = nil
}

var portMap = map[string]string{
	"http":  "80",
	"https": "443",
}

// canonicalAddr returns url.Host but always with a ":port" suffix
func canonicalAddr(url *URL) string {
	addr := url.Host
	if !hasPort(addr) {
		return addr + ":" + portMap[url.Scheme]
	}
	return addr
}

func responseIsKeepAlive(res *Response) bool {
	// TODO: implement.  for now just always shutting down the connection.
	return false
}

// readResponseWithEOFSignal is a wrapper around ReadResponse that replaces
// the response body with a bodyEOFSignal-wrapped version.
func readResponseWithEOFSignal(r *bufio.Reader, requestMethod string) (resp *Response, err os.Error) {
	resp, err = ReadResponse(r, requestMethod)
	if err == nil && resp.ContentLength != 0 {
		resp.Body = &bodyEOFSignal{body: resp.Body}
	}
	return
}

// bodyEOFSignal wraps a ReadCloser but runs fn (if non-nil) at most
// once, right before the final Read() or Close() call returns, but after
// EOF has been seen.
type bodyEOFSignal struct {
	body     io.ReadCloser
	fn       func()
	isClosed bool
}

func (es *bodyEOFSignal) Read(p []byte) (n int, err os.Error) {
	n, err = es.body.Read(p)
	if es.isClosed && n > 0 {
		panic("http: unexpected bodyEOFSignal Read after Close; see issue 1725")
	}
	if err == os.EOF && es.fn != nil {
		es.fn()
		es.fn = nil
	}
	return
}

func (es *bodyEOFSignal) Close() (err os.Error) {
	es.isClosed = true
	err = es.body.Close()
	if err == nil && es.fn != nil {
		es.fn()
		es.fn = nil
	}
	return
}

type readFirstCloseBoth struct {
	io.ReadCloser
	io.Closer
}

func (r *readFirstCloseBoth) Close() os.Error {
	if err := r.ReadCloser.Close(); err != nil {
		r.Closer.Close()
		return err
	}
	if err := r.Closer.Close(); err != nil {
		return err
	}
	return nil
}
