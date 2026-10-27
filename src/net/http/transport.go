// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP client implementation. See RFC 7230 through 7235.
//
// This is the low-level Transport implementation of RoundTripper.
// The high-level interface is in client.go.

package http

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"internal/godebug"
	"io"
	"log"
	"maps"
	"net"
	"net/http/httptrace"
	"net/http/internal"
	"net/http/internal/ascii"
	"net/http/internal/httpcommon"
	"net/textproto"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	_ "unsafe"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/http/httpproxy"
)

// DefaultTransport is the default implementation of [Transport] and is
// used by [DefaultClient]. It establishes network connections as needed
// and caches them for reuse by subsequent calls. It uses HTTP proxies
// as directed by the environment variables HTTP_PROXY, HTTPS_PROXY
// and NO_PROXY (or the lowercase versions thereof, which take
// precedence over the uppercase versions).
var DefaultTransport RoundTripper = &Transport{
	Proxy: ProxyFromEnvironment,
	DialContext: defaultTransportDialContext(&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}),
	ForceAttemptHTTP2:     true,
	MaxIdleConns:          100,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: 1 * time.Second,
}

// DefaultMaxIdleConnsPerHost is the default value of [Transport]'s
// MaxIdleConnsPerHost.
const DefaultMaxIdleConnsPerHost = 2

// Transport is an implementation of [RoundTripper] that supports HTTP,
// HTTPS, and HTTP proxies (for either HTTP or HTTPS with CONNECT).
//
// By default, Transport caches connections for future re-use.
// This may leave many open connections when accessing many hosts.
// This behavior can be managed using [Transport.CloseIdleConnections] method
// and the [Transport.MaxIdleConnsPerHost] and [Transport.DisableKeepAlives] fields.
//
// Transports should be reused instead of created as needed.
// Transports are safe for concurrent use by multiple goroutines.
//
// A Transport is a low-level primitive for making HTTP and HTTPS requests.
// For high-level functionality, such as cookies and redirects, see [Client].
//
// Transport uses HTTP/1.1 for HTTP URLs and either HTTP/1.1 or HTTP/2
// for HTTPS URLs, depending on whether the server supports HTTP/2,
// and how the Transport is configured. The [DefaultTransport] supports HTTP/2.
// To explicitly enable HTTP/2 on a transport, set [Transport.Protocols].
//
// Responses with status codes in the 1xx range are either handled
// automatically (100 expect-continue) or ignored. The one
// exception is HTTP status code 101 (Switching Protocols), which is
// considered a terminal status and returned by [Transport.RoundTrip]. To see the
// ignored 1xx responses, use the httptrace trace package's
// ClientTrace.Got1xxResponse.
//
// Transport only retries a request upon encountering a network error
// if the connection has already been used successfully and if the
// request is idempotent and either has no body or has its [Request.GetBody]
// defined. HTTP requests are considered idempotent if they have HTTP methods
// GET, HEAD, OPTIONS, TRACE, or QUERY; or if their [Header] map contains an
// "Idempotency-Key" or "X-Idempotency-Key" entry. If the idempotency key
// value is a zero-length slice, the request is treated as idempotent but the
// header is not sent on the wire.
type Transport struct {
	pool clientConnPool

	altMu    sync.Mutex   // guards changing altProto only
	altProto atomic.Value // of nil or map[string]RoundTripper, key is URI scheme

	// Proxy specifies a function to return a proxy for a given
	// Request. If the function returns a non-nil error, the
	// request is aborted with the provided error.
	//
	// The proxy type is determined by the URL scheme. "http",
	// "https", "socks5", and "socks5h" are supported. If the scheme is empty,
	// "http" is assumed.
	// "socks5" is treated the same as "socks5h".
	//
	// If the proxy URL contains a userinfo subcomponent,
	// the proxy request will pass the username and password
	// in a Proxy-Authorization header.
	//
	// If Proxy is nil or returns a nil *URL, no proxy is used.
	Proxy func(*Request) (*url.URL, error)

	// OnProxyConnectResponse is called when the Transport gets an HTTP response from
	// a proxy for a CONNECT request. It's called before the check for a 200 OK response.
	// If it returns an error, the request fails with that error.
	OnProxyConnectResponse func(ctx context.Context, proxyURL *url.URL, connectReq *Request, connectRes *Response) error

	// DialContext specifies the dial function for creating unencrypted TCP connections.
	// If DialContext is nil (and the deprecated Dial below is also nil),
	// then the transport dials using package net.
	//
	// DialContext runs concurrently with calls to RoundTrip.
	// A RoundTrip call that initiates a dial may end up using
	// a connection dialed previously when the earlier connection
	// becomes idle before the later DialContext completes.
	DialContext func(ctx context.Context, network, addr string) (net.Conn, error)

	// Dial specifies the dial function for creating unencrypted TCP connections.
	//
	// Dial runs concurrently with calls to RoundTrip.
	// A RoundTrip call that initiates a dial may end up using
	// a connection dialed previously when the earlier connection
	// becomes idle before the later Dial completes.
	//
	// Deprecated: Use DialContext instead, which allows the transport
	// to cancel dials as soon as they are no longer needed.
	// If both are set, DialContext takes priority.
	Dial func(network, addr string) (net.Conn, error)

	// DialTLSContext specifies an optional dial function for creating
	// TLS connections for non-proxied HTTPS requests.
	//
	// If DialTLSContext is nil (and the deprecated DialTLS below is also nil),
	// DialContext and TLSClientConfig are used.
	//
	// If DialTLSContext is set, the Dial and DialContext hooks are not used for HTTPS
	// requests and the TLSClientConfig and TLSHandshakeTimeout
	// are ignored. The returned net.Conn is assumed to already be
	// past the TLS handshake.
	//
	// To support ALPN protocol negotiation, the returned net.Conn should be
	// a *tls.Conn or implement the same ConnectionState method as *tls.Conn.
	DialTLSContext func(ctx context.Context, network, addr string) (net.Conn, error)

	// DialTLS specifies an optional dial function for creating
	// TLS connections for non-proxied HTTPS requests.
	//
	// Deprecated: Use DialTLSContext instead, which allows the transport
	// to cancel dials as soon as they are no longer needed.
	// If both are set, DialTLSContext takes priority.
	DialTLS func(network, addr string) (net.Conn, error)

	// TLSClientConfig specifies the TLS configuration to use with
	// tls.Client.
	// If nil, the default configuration is used.
	// If non-nil, HTTP/2 support may not be enabled by default.
	TLSClientConfig *tls.Config

	// TLSHandshakeTimeout specifies the maximum amount of time to
	// wait for a TLS handshake. Zero means no timeout.
	TLSHandshakeTimeout time.Duration

	// DisableKeepAlives, if true, disables HTTP keep-alives and
	// will only use the connection to the server for a single
	// HTTP request.
	//
	// This is unrelated to the similarly named TCP keep-alives.
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

	// MaxIdleConns controls the maximum number of idle (keep-alive)
	// connections across all hosts. Zero means no limit.
	MaxIdleConns int

	// MaxIdleConnsPerHost, if non-zero, controls the maximum idle
	// (keep-alive) connections to keep per-host. If zero,
	// DefaultMaxIdleConnsPerHost is used.
	MaxIdleConnsPerHost int

	// MaxConnsPerHost optionally limits the total number of
	// connections per host, including connections in the dialing,
	// active, and idle states. On limit violation, dials will block.
	//
	// Zero means no limit.
	MaxConnsPerHost int

	// IdleConnTimeout is the maximum amount of time an idle
	// (keep-alive) connection will remain idle before closing
	// itself.
	// Zero means no limit.
	IdleConnTimeout time.Duration

	// ResponseHeaderTimeout, if non-zero, specifies the amount of
	// time to wait for a server's response headers after fully
	// writing the request (including its body, if any). This
	// time does not include the time to read the response body.
	ResponseHeaderTimeout time.Duration

	// ExpectContinueTimeout, if non-zero, specifies the amount of
	// time to wait for a server's first response headers after fully
	// writing the request headers if the request has an
	// "Expect: 100-continue" header. Zero means no timeout and
	// causes the body to be sent immediately, without
	// waiting for the server to approve.
	// This time does not include the time to send the request header.
	ExpectContinueTimeout time.Duration

	// TLSNextProto specifies how the Transport switches to an
	// alternate protocol (such as HTTP/2) after a TLS ALPN
	// protocol negotiation. If Transport dials a TLS connection
	// with a non-empty protocol name and TLSNextProto contains a
	// map entry for that key (such as "h2"), then the func is
	// called with the request's authority (such as "example.com"
	// or "example.com:1234") and the TLS connection. The function
	// must return a RoundTripper that then handles the request.
	// If TLSNextProto is not nil, HTTP/2 support is not enabled
	// automatically.
	//
	// Historically, TLSNextProto was used to disable HTTP/2 support.
	// The Transport.Protocols field now provides a simpler way to do this.
	TLSNextProto map[string]func(authority string, c *tls.Conn) RoundTripper

	// ProxyConnectHeader optionally specifies headers to send to
	// proxies during CONNECT requests.
	// To set the header dynamically, see GetProxyConnectHeader.
	ProxyConnectHeader Header

	// GetProxyConnectHeader optionally specifies a func to return
	// headers to send to proxyURL during a CONNECT request to the
	// ip:port target.
	// If it returns an error, the Transport's RoundTrip fails with
	// that error. It can return (nil, nil) to not add headers.
	// If GetProxyConnectHeader is non-nil, ProxyConnectHeader is
	// ignored.
	GetProxyConnectHeader func(ctx context.Context, proxyURL *url.URL, target string) (Header, error)

	// MaxResponseHeaderBytes specifies a limit on how many
	// response bytes are allowed in the server's response
	// header.
	//
	// Zero means to use a default limit.
	MaxResponseHeaderBytes int64

	// WriteBufferSize specifies the size of the write buffer used
	// when writing to the transport.
	// If zero, a default (currently 4KB) is used.
	WriteBufferSize int

	// ReadBufferSize specifies the size of the read buffer used
	// when reading from the transport.
	// If zero, a default (currently 4KB) is used.
	ReadBufferSize int

	// nextProtoOnce guards initialization of TLSNextProto and
	// h2Transport (via onceSetNextProtoDefaults)
	nextProtoOnce      sync.Once
	closeIdleFunc      closeIdleConnectionser // non-nil if http2 wired up
	h2Transport        *http2Transport
	h2Config           http2ExternalTransportConfig
	h3Transport        http3Transport // non-nil if http3 wired up
	tlsNextProtoWasNil bool           // whether TLSNextProto was nil when the Once fired

	// ForceAttemptHTTP2 controls whether HTTP/2 is enabled when a non-zero
	// Dial, DialTLS, or DialContext func or TLSClientConfig is provided.
	// By default, use of any those fields conservatively disables HTTP/2.
	// To use a custom dialer or TLS config and still attempt HTTP/2
	// upgrades, set this to true.
	ForceAttemptHTTP2 bool

	// HTTP2 configures HTTP/2 connections.
	HTTP2 *HTTP2Config

	// Protocols is the set of protocols supported by the transport.
	//
	// If Protocols includes UnencryptedHTTP2 and does not include HTTP1,
	// the transport will use unencrypted HTTP/2 for requests for http:// URLs.
	//
	// If Protocols is nil, the default is usually HTTP/1 only.
	// If ForceAttemptHTTP2 is true, or if TLSNextProto contains an "h2" entry,
	// the default is HTTP/1 and HTTP/2.
	Protocols *Protocols
}

func (t *Transport) writeBufferSize() int {
	if t.WriteBufferSize > 0 {
		return t.WriteBufferSize
	}
	return 4 << 10
}

func (t *Transport) readBufferSize() int {
	if t.ReadBufferSize > 0 {
		return t.ReadBufferSize
	}
	return 4 << 10
}

func (t *Transport) maxHeaderResponseSize() int64 {
	if t.MaxResponseHeaderBytes > 0 {
		return t.MaxResponseHeaderBytes
	}
	return 10 << 20 // conservative default; same as http2
}

// Clone returns a deep copy of t's exported fields.
func (t *Transport) Clone() *Transport {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)
	t2 := &Transport{
		Proxy:                  t.Proxy,
		OnProxyConnectResponse: t.OnProxyConnectResponse,
		DialContext:            t.DialContext,
		Dial:                   t.Dial,
		DialTLS:                t.DialTLS,
		DialTLSContext:         t.DialTLSContext,
		TLSHandshakeTimeout:    t.TLSHandshakeTimeout,
		DisableKeepAlives:      t.DisableKeepAlives,
		DisableCompression:     t.DisableCompression,
		MaxIdleConns:           t.MaxIdleConns,
		MaxIdleConnsPerHost:    t.MaxIdleConnsPerHost,
		MaxConnsPerHost:        t.MaxConnsPerHost,
		IdleConnTimeout:        t.IdleConnTimeout,
		ResponseHeaderTimeout:  t.ResponseHeaderTimeout,
		ExpectContinueTimeout:  t.ExpectContinueTimeout,
		ProxyConnectHeader:     t.ProxyConnectHeader.Clone(),
		GetProxyConnectHeader:  t.GetProxyConnectHeader,
		MaxResponseHeaderBytes: t.MaxResponseHeaderBytes,
		ForceAttemptHTTP2:      t.ForceAttemptHTTP2,
		WriteBufferSize:        t.WriteBufferSize,
		ReadBufferSize:         t.ReadBufferSize,
	}
	if t.TLSClientConfig != nil {
		t2.TLSClientConfig = t.TLSClientConfig.Clone()
	}
	if t.HTTP2 != nil {
		t2.HTTP2 = &HTTP2Config{}
		*t2.HTTP2 = *t.HTTP2
	}
	if t.Protocols != nil {
		t2.Protocols = &Protocols{}
		*t2.Protocols = *t.Protocols
	}
	if !t.tlsNextProtoWasNil {
		npm := maps.Clone(t.TLSNextProto)
		if npm == nil {
			npm = make(map[string]func(authority string, c *tls.Conn) RoundTripper)
		}
		t2.TLSNextProto = npm
	}
	return t2
}

type closeIdleConnectionser interface {
	// CloseIdleConnections is called by Transport.CloseIdleConnections.
	//
	// We expect to use this on transports supplied by x/net/http2 or
	// internal/http3.
	//
	// The transport will close idle connections created with DialClientConn
	// before calling this method. The HTTP/3 transport should not attempt to
	// close idle connections, but may clean up shared resources such as UDP
	// sockets if no connections remain.
	CloseIdleConnections()
}

func (t *Transport) hasCustomTLSDialer() bool {
	return t.DialTLS != nil || t.DialTLSContext != nil
}

var http2client = godebug.New("http2client")

// onceSetNextProtoDefaults initializes TLSNextProto.
// It must be called via t.nextProtoOnce.Do.
func (t *Transport) onceSetNextProtoDefaults() {
	t.tlsNextProtoWasNil = (t.TLSNextProto == nil)
	if http2client.Value() == "0" {
		http2client.IncNonDefault()
		return
	}

	// If they've already configured http2 with
	// golang.org/x/net/http2 instead of the bundled copy, try to
	// get at its http2.Transport value (via the "https"
	// altproto map) so we can call CloseIdleConnections on it if
	// requested. (Issue 22891)
	altProto, _ := t.altProto.Load().(map[string]RoundTripper)
	if rv := reflect.ValueOf(altProto["https"]); rv.IsValid() && rv.Type().Kind() == reflect.Struct && rv.Type().NumField() == 1 {
		if v := rv.Field(0); v.CanInterface() {
			if h2i, ok := v.Interface().(closeIdleConnectionser); ok {
				t.closeIdleFunc = h2i
				return
			}
		}
	}

	if _, ok := t.TLSNextProto["h2"]; ok {
		// There's an existing HTTP/2 implementation installed.
		return
	}
	protocols := t.protocols()
	if !protocols.HTTP2() && !protocols.UnencryptedHTTP2() {
		return
	}
	if omitHTTP2Client {
		return
	}

	t.configureHTTP2(protocols)
}

func (t *Transport) protocols() Protocols {
	if t.Protocols != nil {
		return *t.Protocols // user-configured set
	}
	var p Protocols
	p.SetHTTP1(true) // default always includes HTTP/1
	switch {
	case t.TLSNextProto != nil:
		// Setting TLSNextProto to an empty map is a documented way
		// to disable HTTP/2 on a Transport.
		if t.TLSNextProto["h2"] != nil {
			p.SetHTTP2(true)
		}
	case !t.ForceAttemptHTTP2 && (t.TLSClientConfig != nil || t.Dial != nil || t.DialContext != nil || t.hasCustomTLSDialer()):
		// Be conservative and don't automatically enable
		// http2 if they've specified a custom TLS config or
		// custom dialers. Let them opt-in themselves via
		// Transport.Protocols.SetHTTP2(true) so we don't surprise them
		// by modifying their tls.Config. Issue 14275.
		// However, if ForceAttemptHTTP2 is true, it overrides the above checks.
	case http2client.Value() == "0":
	default:
		p.SetHTTP2(true)
	}
	return p
}

// ProxyFromEnvironment returns the URL of the proxy to use for a
// given request, as indicated by the environment variables
// HTTP_PROXY, HTTPS_PROXY and NO_PROXY (or the lowercase versions
// thereof, which take precedence over the uppercase versions).
// Requests use the proxy from the environment variable
// matching their scheme, unless excluded by NO_PROXY.
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
	return envProxyFunc()(req.URL)
}

// ProxyURL returns a proxy function (for use in a [Transport])
// that always returns the same URL.
func ProxyURL(fixedURL *url.URL) func(*Request) (*url.URL, error) {
	return func(*Request) (*url.URL, error) {
		return fixedURL, nil
	}
}

// transportRequest is a wrapper around a *Request that adds
// optional extra headers to write and stores any error to return
// from roundTrip.
type transportRequest struct {
	*Request                        // original request, not to be mutated
	extra    Header                 // extra headers to write, or nil
	trace    *httptrace.ClientTrace // optional

	ctx    context.Context // canceled when we are done with the request
	cancel context.CancelCauseFunc

	mu  sync.Mutex // guards err
	err error      // first setError value for mapRoundTripError to consider
}

func (tr *transportRequest) extraHeaders() Header {
	if tr.extra == nil {
		tr.extra = make(Header)
	}
	return tr.extra
}

func (tr *transportRequest) setError(err error) {
	tr.mu.Lock()
	if tr.err == nil {
		tr.err = err
	}
	tr.mu.Unlock()
}

// useRegisteredProtocol reports whether an alternate protocol (as registered
// with Transport.RegisterProtocol) should be respected for this request.
func (t *Transport) useRegisteredProtocol(req *Request) bool {
	if req.URL.Scheme == "https" && req.requiresHTTP1() {
		// If this request requires HTTP/1, don't use the
		// "https" alternate protocol, which is used by the
		// HTTP/2 code to take over requests if there's an
		// existing cached HTTP/2 connection.
		return false
	}
	return true
}

// alternateRoundTripper returns the alternate RoundTripper to use
// for this request if the Request's URL scheme requires one,
// or nil for the normal case of using the Transport.
func (t *Transport) alternateRoundTripper(req *Request) RoundTripper {
	if !t.useRegisteredProtocol(req) {
		return nil
	}
	if req.URL.Scheme == "https" && t.h2Config != nil && t.h2Config.ExternalRoundTrip() {
		// This Transport has been configured to use an x/net/http2 Transport
		// with a user-provided ClientConnPool. We're going to pass off the
		// RoundTrip to x/net/http2 so it can use that pool.
		//
		// The ClientConnPool API is deprecated, but we're doing our best here
		// to continue supporting any users who are using it.
		return t.h2Config
	}
	altProto, _ := t.altProto.Load().(map[string]RoundTripper)
	return altProto[req.URL.Scheme]
}

func validateHeaders(hdrs Header) string {
	for k, vv := range hdrs {
		if !httpguts.ValidHeaderFieldName(k) {
			return fmt.Sprintf("field name %q", k)
		}
		for _, v := range vv {
			if !httpguts.ValidHeaderFieldValue(v) {
				// Don't include the value in the error,
				// because it may be sensitive.
				return fmt.Sprintf("field value for %q", k)
			}
		}
	}
	return ""
}

// roundTrip implements a RoundTripper over HTTP.
func (t *Transport) roundTrip(req *Request) (_ *Response, err error) {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)
	ctx := req.Context()

	if req.URL == nil {
		req.closeBody()
		return nil, errors.New("http: nil Request.URL")
	}
	if req.Header == nil {
		req.closeBody()
		return nil, errors.New("http: nil Request.Header")
	}
	scheme := req.URL.Scheme
	isHTTP := scheme == "http" || scheme == "https"
	if isHTTP {
		// Validate the outgoing headers.
		if err := validateHeaders(req.Header); err != "" {
			req.closeBody()
			return nil, fmt.Errorf("net/http: invalid header %s", err)
		}

		// Validate the outgoing trailers too.
		if err := validateHeaders(req.Trailer); err != "" {
			req.closeBody()
			return nil, fmt.Errorf("net/http: invalid trailer %s", err)
		}
	}

	origReq := req
	req = setupRewindBody(req)

	if altRT := t.alternateRoundTripper(req); altRT != nil {
		if resp, err := altRT.RoundTrip(req); err != ErrSkipAltProtocol {
			return resp, err
		}
		var err error
		req, err = rewindBody(req)
		if err != nil {
			return nil, err
		}
	}
	if !isHTTP {
		req.closeBody()
		return nil, badStringError("unsupported protocol scheme", scheme)
	}
	if req.Method != "" && !validMethod(req.Method) {
		req.closeBody()
		return nil, fmt.Errorf("net/http: invalid method %q", req.Method)
	}
	if req.URL.Host == "" {
		req.closeBody()
		return nil, errors.New("http: no Host in request URL")
	}

	// Transport request context.
	//
	// RoundTrip cancels this context before returning.
	// On the HTTP/1 path, persistConn.RoundTrip creates its own context
	// that persistConn.readLoop cancels after reading the response body.
	ctx, cancel := context.WithCancelCause(req.Context())

	// Convert Request.Cancel into context cancellation.
	if origReq.Cancel != nil {
		go awaitLegacyCancel(ctx, cancel, origReq)
	}

	defer func() {
		if err != nil {
			cancel(err)
		} else {
			cancel(errRequestDone)
		}
	}()

	for {
		select {
		case <-ctx.Done():
			req.closeBody()
			return nil, context.Cause(ctx)
		default:
		}

		cm, err := t.connectMethodForRequest(req)
		if err != nil {
			req.closeBody()
			return nil, err
		}

		// Get the cached or newly-created connection to either the
		// host (for http or https), the http proxy, or the http proxy
		// pre-CONNECTed to https server. In any case, we'll be ready
		// to send it requests.
		pconn, err := t.getConn(ctx, cm)
		if err != nil {
			req.closeBody()
			return nil, err
		}

		resp, err := pconn.cc.RoundTrip(req)
		if err == nil {
			resp.Request = origReq
			return resp, nil
		}

		// Failed. Clean up and determine whether to retry.
		if http2isNoCachedConnError(err) {
			t.poolRemoveConn(pconn)
		} else if !shouldRetryRequest(pconn, req, err) {
			// Issue 16465: return underlying net.Conn.Read error from peek,
			// as we've historically done.
			if e, ok := err.(nothingWrittenError); ok {
				err = e.error
			}
			if e, ok := err.(transportReadFromServerError); ok {
				err = e.err
			}
			if b, ok := req.Body.(*readTrackingBody); ok && !b.didClose.Load() {
				// Issue 49621: Close the request body if pconn.roundTrip
				// didn't do so already. This can happen if the pconn
				// write loop exits without reading the write request.
				req.closeBody()
			}
			return nil, err
		}
		testHookRoundTripRetried()

		// Rewind the body if we're able to.
		req, err = rewindBody(req)
		if err != nil {
			return nil, err
		}
	}
}

func http2isNoCachedConnError(err error) bool {
	_, ok := err.(interface{ IsHTTP2NoCachedConnError() })
	return ok
}

func awaitLegacyCancel(ctx context.Context, cancel context.CancelCauseFunc, req *Request) {
	select {
	case <-req.Cancel:
		cancel(errRequestCanceled)
	case <-ctx.Done():
	}
}

var errCannotRewind = errors.New("net/http: cannot rewind body after connection loss")

type readTrackingBody struct {
	io.ReadCloser
	didRead  bool // not atomic.Bool because only one goroutine (the user's) should be accessing
	didClose atomic.Bool
}

func (r *readTrackingBody) Read(data []byte) (int, error) {
	r.didRead = true
	return r.ReadCloser.Read(data)
}

func (r *readTrackingBody) Close() error {
	if !r.didClose.CompareAndSwap(false, true) {
		return nil
	}
	return r.ReadCloser.Close()
}

// setupRewindBody returns a new request with a custom body wrapper
// that can report whether the body needs rewinding.
// This lets rewindBody avoid an error result when the request
// does not have GetBody but the body hasn't been read at all yet.
func setupRewindBody(req *Request) *Request {
	if req.Body == nil || req.Body == NoBody {
		return req
	}
	newReq := *req
	newReq.Body = &readTrackingBody{ReadCloser: req.Body}
	return &newReq
}

// rewindBody returns a new request with the body rewound.
// It returns req unmodified if the body does not need rewinding.
// rewindBody takes care of closing req.Body when appropriate
// (in all cases except when rewindBody returns req unmodified).
func rewindBody(req *Request) (rewound *Request, err error) {
	if req.Body == nil || req.Body == NoBody || (!req.Body.(*readTrackingBody).didRead && !req.Body.(*readTrackingBody).didClose.Load()) {
		return req, nil // nothing to rewind
	}
	if !req.Body.(*readTrackingBody).didClose.Load() {
		req.closeBody()
	}
	if req.GetBody == nil {
		return nil, errCannotRewind
	}
	body, err := req.GetBody()
	if err != nil {
		return nil, err
	}
	newReq := *req
	newReq.Body = &readTrackingBody{ReadCloser: body}
	return &newReq, nil
}

// shouldRetryRequest reports whether we should retry sending a failed
// HTTP request on a new connection. The non-nil input error is the
// error from roundTrip.
func shouldRetryRequest(pconn *poolConn, req *Request, err error) bool {
	if http2isNoCachedConnError(err) {
		// Issue 16582: if the user started a bunch of
		// requests at once, they can all pick the same conn
		// and violate the server's max concurrent streams.
		// Instead, match the HTTP/1 behavior for now and dial
		// again to get a new TCP connection, rather than failing
		// this request.
		return true
	}
	if err == errMissingHost {
		// User error.
		return false
	}
	if !pconn.cc.IsReused() {
		// This was a fresh connection. There's no reason the server
		// should've hung up on us.
		//
		// Also, if we retried now, we could loop forever
		// creating new connections and retrying if the server
		// is just hanging up on us because it doesn't like
		// our request (as opposed to sending an error).
		return false
	}
	if _, ok := err.(nothingWrittenError); ok {
		// We never wrote anything, so it's safe to retry, if there's no body or we
		// can "rewind" the body with GetBody.
		return req.outgoingLength() == 0 || req.GetBody != nil
	}
	if !req.isReplayable() {
		// Don't retry non-idempotent requests.
		return false
	}
	if _, ok := err.(transportReadFromServerError); ok {
		// We got some non-EOF net.Conn.Read failure reading
		// the 1st response byte from the server.
		return true
	}
	if err == errServerClosedIdle {
		// The server replied with io.EOF while we were trying to
		// read the response. Probably an unfortunately keep-alive
		// timeout, just as the client was writing a request.
		return true
	}
	return false // conservatively
}

// ErrSkipAltProtocol is a sentinel error value defined by Transport.RegisterProtocol.
var ErrSkipAltProtocol = internal.ErrSkipAltProtocol

// RegisterProtocol registers a new protocol with scheme.
// The [Transport] will pass requests using the given scheme to rt.
// It is rt's responsibility to simulate HTTP request semantics.
//
// RegisterProtocol can be used by other packages to provide
// implementations of protocol schemes like "ftp" or "file".
//
// If rt.RoundTrip returns [ErrSkipAltProtocol], the Transport will
// handle the [Transport.RoundTrip] itself for that one request, as if the
// protocol were not registered.
func (t *Transport) RegisterProtocol(scheme string, rt RoundTripper) {
	if err := t.registerProtocol(scheme, rt); err != nil {
		panic(err)
	}
}

func (t *Transport) registerProtocol(scheme string, rt RoundTripper) error {
	t.altMu.Lock()
	defer t.altMu.Unlock()

	if scheme == "http/2" {
		if t.h2Config != nil {
			panic("http: HTTP/2 Transport already registered")
		}
		var ok bool
		if t.h2Config, ok = rt.(http2ExternalTransportConfig); !ok {
			panic("http: HTTP/2 configuration does not implement ExternalTransportConfig")
		}
		t.h2Config.Registered(t)
	}

	if scheme == "http/3" {
		if t.h3Transport != nil {
			panic("http: HTTP/3 Transport already registered")
		}
		var ok bool
		if t.h3Transport, ok = rt.(http3Transport); !ok {
			panic("http: HTTP/3 RoundTripper does not implement DialClientConn")
		}
		// Notify the HTTP/3 transport of successful registration.
		// (Since RegisterProtocol doesn't return anything, we call a method here.)
		if r, ok := rt.(interface {
			Registered(*Transport)
		}); ok {
			r.Registered(t)
		}
		return nil
	}

	oldMap, _ := t.altProto.Load().(map[string]RoundTripper)
	if _, exists := oldMap[scheme]; exists {
		return errors.New("protocol " + scheme + " already registered")
	}
	newMap := maps.Clone(oldMap)
	if newMap == nil {
		newMap = make(map[string]RoundTripper)
	}
	newMap[scheme] = rt
	t.altProto.Store(newMap)
	return nil
}

// CloseIdleConnections closes any connections which were previously
// connected from previous requests but are now sitting idle in
// a "keep-alive" state. It does not interrupt any connections currently
// in use.
func (t *Transport) CloseIdleConnections() {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)

	t.poolCloseIdleConnections()

	// Tell various associated transports to close their connections.

	// net/http/internal/http2 transport. This is the common case for HTTP/2 users.
	if tr2 := t.h2Transport; tr2 != nil {
		tr2.CloseIdleConnections()
	}
	// Probably an older x/net/http2 transport registered via Transport.RegisterProtocol.
	// This is a legacy path; modern users just use internal/http2.
	// (Note that we don't use this path when x/net/http2 wraps the net/http transport;
	// this is supporting pre-wrapping x/net/http2.)
	if t2 := t.closeIdleFunc; t2 != nil {
		t2.CloseIdleConnections()
	}
	// HTTP/3 transport, probably from internal/http3.
	if cc, ok := t.h3Transport.(closeIdleConnectionser); ok {
		cc.CloseIdleConnections()
	}
}

// CancelRequest is obsolete and does nothing.
//
// Deprecated: Use [NewRequestWithContext] to create a request with a
// cancelable context instead.
func (t *Transport) CancelRequest(req *Request) {
}

//
// Private implementation past this point.
//

var (
	envProxyOnce      sync.Once
	envProxyFuncValue func(*url.URL) (*url.URL, error)
)

// envProxyFunc returns a function that reads the
// environment variable to determine the proxy address.
func envProxyFunc() func(*url.URL) (*url.URL, error) {
	envProxyOnce.Do(func() {
		envProxyFuncValue = httpproxy.FromEnvironment().ProxyFunc()
	})
	return envProxyFuncValue
}

// resetProxyConfig is used by tests.
func resetProxyConfig() {
	envProxyOnce = sync.Once{}
	envProxyFuncValue = nil
}

func (t *Transport) connectMethodForRequest(req *Request) (cm connectMethod, err error) {
	cm.targetScheme = req.URL.Scheme
	cm.targetAddr = canonicalAddr(req.URL)
	if t.Proxy != nil {
		cm.proxyURL, err = t.Proxy(req)
	}
	cm.onlyH1 = req.requiresHTTP1()
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
	errCloseIdle          = errors.New("http: putIdleConn: CloseIdleConnections was called")
	errTooManyIdle        = errors.New("http: putIdleConn: too many idle connections")
	errTooManyIdleHost    = errors.New("http: putIdleConn: too many idle connections for host")
	errNotInPool          = errors.New("http: putIdleConn: connection is not in a pool")
	errCloseIdleConns     = errors.New("http: CloseIdleConnections called")
	errReadLoopExiting    = errors.New("http: persistConn.readLoop exiting")
	errIdleConnTimeout    = errors.New("http: idle connection timeout")

	// errServerClosedIdle is not seen by users for idempotent requests, but may be
	// seen by a user if the server shuts down an idle connection and sends its FIN
	// in flight with already-written POST body bytes from the client.
	// See https://github.com/golang/go/issues/19943#issuecomment-355607646
	errServerClosedIdle = errors.New("http: server closed idle connection")
)

// transportReadFromServerError is used by Transport.readLoop when the
// 1 byte peek read fails and we're actually anticipating a response.
// Usually this is just due to the inherent keep-alive shut down race,
// where the server closed the connection at the same time the client
// wrote. The underlying err field is usually io.EOF or some
// ECONNRESET sort of thing which varies by platform. But it might be
// the user's custom net.Conn.Read error too, so we carry it along for
// them to return from Transport.RoundTrip.
type transportReadFromServerError struct {
	err error
}

func (e transportReadFromServerError) Unwrap() error { return e.err }

func (e transportReadFromServerError) Error() string {
	return fmt.Sprintf("net/http: Transport failed to read from server: %v", e.err)
}

func (t *Transport) maxIdleConnsPerHost() int {
	if v := t.MaxIdleConnsPerHost; v != 0 {
		return v
	}
	return DefaultMaxIdleConnsPerHost
}

func (t *Transport) keepAlivesDisabled() bool {
	return t.DisableKeepAlives || t.MaxIdleConnsPerHost < 0 || t.MaxIdleConns < 0
}

// tryPutIdleConn adds pconn to the list of idle persistent connections awaiting
// a new request.
// If pconn is no longer needed or not in a good state, tryPutIdleConn returns
// an error explaining why it wasn't registered.
// tryPutIdleConn does not close pconn.
func (t *Transport) tryPutIdleConn(pconn *persistConn) error {
	if t.keepAlivesDisabled() {
		return errKeepAlivesDisabled
	}
	if pconn.isBroken() {
		return errConnBroken
	}
	pconn.markReused()
	if pconn.isClientConn {
		// internalStateHook is always set for conns created by NewClientConn.
		defer pconn.internalStateHook()
		pconn.mu.Lock()
		defer pconn.mu.Unlock()
		if !pconn.inFlight {
			panic("pconn is not in flight")
		}
		pconn.inFlight = false
		select {
		case pconn.availch <- struct{}{}:
		default:
			panic("unable to make pconn available")
		}
		return nil
	}
	return t.poolPutIdleConn(pconn.poolConn)
}

// removeIdleConn marks pconn as dead.
func (t *Transport) removeIdleConn(pconn *persistConn) {
	t.poolRemoveConn(pconn.poolConn)
}

var zeroDialer net.Dialer

func (t *Transport) dial(ctx context.Context, network, addr string) (net.Conn, error) {
	if t.DialContext != nil {
		c, err := t.DialContext(ctx, network, addr)
		if c == nil && err == nil {
			err = errors.New("net/http: Transport.DialContext hook returned (nil, nil)")
		}
		return c, err
	}
	if t.Dial != nil {
		c, err := t.Dial(network, addr)
		if c == nil && err == nil {
			err = errors.New("net/http: Transport.Dial hook returned (nil, nil)")
		}
		return c, err
	}
	return zeroDialer.DialContext(ctx, network, addr)
}

func (t *Transport) customDialTLS(ctx context.Context, network, addr string) (conn net.Conn, err error) {
	if t.DialTLSContext != nil {
		conn, err = t.DialTLSContext(ctx, network, addr)
	} else {
		conn, err = t.DialTLS(network, addr)
	}
	if conn == nil && err == nil {
		err = errors.New("net/http: Transport.DialTLS or DialTLSContext returned (nil, nil)")
	}
	return
}

// getConn dials and creates a new connection to the target as
// specified in the connectMethod. This includes doing a proxy CONNECT
// and/or setting up TLS. If this doesn't return an error, the connection
// is ready to write requests to.
func (t *Transport) getConn(ctx context.Context, cm connectMethod) (_ *poolConn, err error) {
	trace := httptrace.ContextClientTrace(ctx)
	if trace != nil && trace.GetConn != nil {
		trace.GetConn(cm.addr())
	}

	pconn, err := t.poolGetConn(ctx, cm)
	if err != nil {
		return nil, err
	}
	if !pconn.cc.IsAlt() && trace != nil && trace.GotConn != nil {
		info := httptrace.GotConnInfo{
			Conn:   pconn.cc.NetConn(),
			Reused: pconn.cc.IsReused(),
		}
		if !pconn.idleAt.IsZero() {
			info.WasIdle = true
			info.IdleTime = time.Since(pconn.idleAt)
		}
		trace.GotConn(info)
	}
	return pconn, nil
}

func (t *Transport) tlsConfigForDial(host string) (*tls.Config, error) {
	firstTLSHost, _, err := net.SplitHostPort(host)
	if err != nil {
		return nil, err
	}
	cfg := cloneTLSConfig(t.TLSClientConfig)
	if cfg.ServerName == "" {
		cfg.ServerName = firstTLSHost
	}
	return cfg, nil
}

// Add TLS to a persistent connection, i.e. negotiate a TLS session. If pconn is already a TLS
// tunnel, this function establishes a nested TLS session inside the encrypted channel.
// The remote endpoint's name may be overridden by TLSClientConfig.ServerName.
func (pconn *persistConn) addTLS(ctx context.Context, addr string, trace *httptrace.ClientTrace) error {
	cfg, err := pconn.t.tlsConfigForDial(addr)
	if err != nil {
		pconn.conn.Close()
		return err
	}
	if pconn.cacheKey.onlyH1 {
		cfg.NextProtos = nil
	}
	plainConn := pconn.conn
	tlsConn := tls.Client(plainConn, cfg)
	errc := make(chan error, 2)
	var timer *time.Timer // for canceling TLS handshake
	if d := pconn.t.TLSHandshakeTimeout; d != 0 {
		timer = time.AfterFunc(d, func() {
			errc <- tlsHandshakeTimeoutError{}
		})
	}
	go func() {
		if trace != nil && trace.TLSHandshakeStart != nil {
			trace.TLSHandshakeStart()
		}
		err := tlsConn.HandshakeContext(ctx)
		if timer != nil {
			timer.Stop()
		}
		errc <- err
	}()
	if err := <-errc; err != nil {
		plainConn.Close()
		if err == (tlsHandshakeTimeoutError{}) {
			// Now that we have closed the connection,
			// wait for the call to HandshakeContext to return.
			<-errc
		}
		if trace != nil && trace.TLSHandshakeDone != nil {
			trace.TLSHandshakeDone(tls.ConnectionState{}, err)
		}
		return err
	}
	cs := tlsConn.ConnectionState()
	if trace != nil && trace.TLSHandshakeDone != nil {
		trace.TLSHandshakeDone(cs, nil)
	}
	pconn.tlsState = &cs
	pconn.conn = tlsConn
	return nil
}

type erringRoundTripper interface {
	RoundTripErr() error
}

var testHookProxyConnectTimeout = context.WithTimeout

func (t *Transport) dialConn(ctx context.Context, cm connectMethod, isClientConn bool, poolConn *poolConn, internalStateHook func()) (_ genericClientConn, err error) {
	// TODO: actually support HTTP/3. Among other things:
	// - make HTTP/3 play well with proxy.
	// - implement happy eyeball between HTTP/3 and HTTP/1 & HTTP/2.
	// - clean up the connection pooling logic.
	if p := t.protocols(); p.http3() {
		if p.HTTP1() || p.HTTP2() || p.UnencryptedHTTP2() {
			return nil, errors.New("http: when using HTTP3, Transport.Protocols must contain only HTTP3")
		}
		if t.h3Transport == nil {
			return nil, errors.New("http: Transport.Protocols contains HTTP3, but Transport does not support HTTP/3")
		}
		tlsConfig, err := t.tlsConfigForDial(cm.addr())
		if err != nil {
			return nil, err
		}
		tlsConfig.NextProtos = []string{"h3"}
		rt, err := t.h3Transport.DialClientConn(ctx, cm.addr(), cm.proxyURL, tlsConfig, internalStateHook)
		if err != nil {
			return nil, err
		}
		return rt.(genericClientConn), nil
	}

	// TODO: Don't allocate persistConn until we know that we're using HTTP/1.
	pconn := &persistConn{
		t:                 t,
		cacheKey:          cm.key(),
		poolConn:          poolConn,
		reqch:             make(chan requestAndChan, 1),
		writech:           make(chan writeRequest, 1),
		closech:           make(chan struct{}),
		writeErrCh:        make(chan error, 1),
		writeLoopDone:     make(chan struct{}),
		isClientConn:      isClientConn,
		internalStateHook: internalStateHook,
	}
	trace := httptrace.ContextClientTrace(ctx)
	wrapErr := func(err error) error {
		if cm.proxyURL != nil {
			// Return a typed error, per Issue 16997
			return &net.OpError{Op: "proxyconnect", Net: "tcp", Err: err}
		}
		return err
	}

	if cc, err := t.http2ExternalDial(ctx, cm); err != errors.ErrUnsupported {
		if err != nil {
			return nil, err
		}
		return cc, nil
	}

	if cm.scheme() == "https" && t.hasCustomTLSDialer() {
		var err error
		pconn.conn, err = t.customDialTLS(ctx, "tcp", cm.addr())
		if err != nil {
			return nil, wrapErr(err)
		}
		type connectionStater interface {
			ConnectionState() tls.ConnectionState
		}
		type handshaker interface {
			HandshakeContext(context.Context) error
		}
		if cstater, ok := pconn.conn.(connectionStater); ok {
			if trace != nil && trace.TLSHandshakeStart != nil {
				trace.TLSHandshakeStart()
			}
			if handshaker, ok := cstater.(handshaker); ok {
				// Handshake here, in case DialTLS didn't. TLSNextProto below
				// depends on it for knowing the connection state.
				if err := handshaker.HandshakeContext(ctx); err != nil {
					go pconn.conn.Close()
					if trace != nil && trace.TLSHandshakeDone != nil {
						trace.TLSHandshakeDone(tls.ConnectionState{}, err)
					}
					return nil, err
				}
			}
			cs := cstater.ConnectionState()
			if trace != nil && trace.TLSHandshakeDone != nil {
				trace.TLSHandshakeDone(cs, nil)
			}
			pconn.tlsState = &cs
		}
	} else {
		conn, err := t.dial(ctx, "tcp", cm.addr())
		if err != nil {
			return nil, wrapErr(err)
		}
		pconn.conn = conn
		if cm.scheme() == "https" {
			if err = pconn.addTLS(ctx, cm.addr(), trace); err != nil {
				return nil, wrapErr(err)
			}
		}
	}

	// Proxy setup.
	switch {
	case cm.proxyURL == nil:
		// Do nothing. Not using a proxy.
	case cm.proxyURL.Scheme == "socks5" || cm.proxyURL.Scheme == "socks5h":
		conn := pconn.conn
		d := socksNewDialer("tcp", conn.RemoteAddr().String())
		if u := cm.proxyURL.User; u != nil {
			auth := &socksUsernamePassword{
				Username: u.Username(),
			}
			auth.Password, _ = u.Password()
			d.AuthMethods = []socksAuthMethod{
				socksAuthMethodNotRequired,
				socksAuthMethodUsernamePassword,
			}
			d.Authenticate = auth.Authenticate
		}
		if _, err := d.DialWithConn(ctx, conn, "tcp", cm.targetAddr); err != nil {
			conn.Close()
			return nil, err
		}
	case cm.targetScheme == "http":
		pconn.isProxy = true
		if pa := cm.proxyAuth(); pa != "" {
			pconn.mutateHeaderFunc = func(h Header) {
				h.Set("Proxy-Authorization", pa)
			}
		}
	case cm.targetScheme == "https":
		conn := pconn.conn
		var hdr Header
		if t.GetProxyConnectHeader != nil {
			var err error
			hdr, err = t.GetProxyConnectHeader(ctx, cm.proxyURL, cm.targetAddr)
			if err != nil {
				conn.Close()
				return nil, err
			}
		} else {
			hdr = t.ProxyConnectHeader
		}
		if hdr == nil {
			hdr = make(Header)
		}
		if pa := cm.proxyAuth(); pa != "" {
			hdr = hdr.Clone()
			hdr.Set("Proxy-Authorization", pa)
		}
		connectReq := &Request{
			Method: "CONNECT",
			URL:    &url.URL{Opaque: cm.targetAddr},
			Host:   cm.targetAddr,
			Header: hdr,
		}

		// Set a (long) timeout here to make sure we don't block forever
		// and leak a goroutine if the connection stops replying after
		// the TCP connect.
		connectCtx, cancel := testHookProxyConnectTimeout(ctx, 1*time.Minute)
		defer cancel()

		didReadResponse := make(chan struct{}) // closed after CONNECT write+read is done or fails
		var (
			resp *Response
			err  error // write or read error
		)
		// Write the CONNECT request & read the response.
		go func() {
			defer close(didReadResponse)
			err = connectReq.Write(conn)
			if err != nil {
				return
			}
			// Okay to use and discard buffered reader here, because
			// TLS server will not speak until spoken to.
			br := bufio.NewReader(&io.LimitedReader{R: conn, N: t.maxHeaderResponseSize()})
			resp, err = ReadResponse(br, connectReq)
		}()
		select {
		case <-connectCtx.Done():
			conn.Close()
			<-didReadResponse
			return nil, connectCtx.Err()
		case <-didReadResponse:
			// resp or err now set
		}
		if err != nil {
			conn.Close()
			return nil, err
		}

		if t.OnProxyConnectResponse != nil {
			err = t.OnProxyConnectResponse(ctx, cm.proxyURL, connectReq, resp)
			if err != nil {
				conn.Close()
				return nil, err
			}
		}

		if resp.StatusCode != 200 {
			_, text, ok := strings.Cut(resp.Status, " ")
			conn.Close()
			if !ok {
				return nil, errors.New("unknown status code")
			}
			return nil, errors.New(text)
		}
	}

	if cm.proxyURL != nil && cm.targetScheme == "https" {
		if err := pconn.addTLS(ctx, cm.targetAddr, trace); err != nil {
			return nil, err
		}
	}

	// Possible unencrypted HTTP/2 with prior knowledge.
	unencryptedHTTP2 := pconn.tlsState == nil &&
		t.Protocols != nil &&
		t.Protocols.UnencryptedHTTP2() &&
		!t.Protocols.HTTP1()

	http2 := unencryptedHTTP2 ||
		(pconn.tlsState != nil && pconn.tlsState.NegotiatedProtocol == "h2")

	if http2 && t.h2Transport != nil {
		// HTTP/2, backed by internal/http2.
		if isClientConn {
			// NewClientConn, creating a connection.
			cc, err := t.http2NewClientConn(pconn.conn, internalStateHook)
			if err == nil {
				return cc, nil
			}
			if err != errors.ErrUnsupported {
				return nil, err
			}
		} else {
			// RoundTrip, adding a net.Conn to internal/http2.Transport's pool.
			// The "connection" returned by http2AddConn is just a reference
			// to the http2.Transport.
			cc, err := t.http2AddConn(cm.targetScheme, cm.targetAddr, pconn.conn)
			if err == nil {
				return cc, nil
			}
			if err != errors.ErrUnsupported {
				return nil, err
			}
		}
	}

	if isClientConn && (unencryptedHTTP2 || (pconn.tlsState != nil && pconn.tlsState.NegotiatedProtocol == "h2")) {
		// NewClientConn, using an HTTP/2 implementation installed via the legacy
		// RegisterProtocol path. This shouldn't really be used by anything any more.
		altProto, _ := t.altProto.Load().(map[string]RoundTripper)
		h2, ok := altProto["https"].(newClientConner)
		if !ok {
			return nil, errors.New("http: HTTP/2 implementation does not support NewClientConn (update golang.org/x/net?)")
		}
		rt, err := h2.NewClientConn(pconn.conn, internalStateHook)
		if err != nil {
			pconn.conn.Close()
			return nil, err
		}
		cc, ok := rt.(externalClientConn)
		if !ok {
			return nil, errors.New("http: NewClientConn returned something that is not a ClientConn")
		}
		return externalConn{cc}, nil
	}

	wrapTLSNextProtoRoundTripper := func(rt RoundTripper) (genericClientConn, error) {
		if e, ok := rt.(erringRoundTripper); ok {
			// pconn.conn was closed by next (http2configureTransports.upgradeFn).
			return nil, e.RoundTripErr()
		}
		return &altClientConn{rt: rt}, nil
	}
	if unencryptedHTTP2 {
		// Unencrypted HTTP/2 using the legacy TLSNextProto.
		next, ok := t.TLSNextProto[nextProtoUnencryptedHTTP2]
		if !ok {
			return nil, errors.New("http: Transport does not support unencrypted HTTP/2")
		}
		alt := next(cm.targetAddr, unencryptedTLSConn(pconn.conn))
		return wrapTLSNextProtoRoundTripper(alt)
	}

	if s := pconn.tlsState; s != nil && s.NegotiatedProtocolIsMutual && s.NegotiatedProtocol != "" {
		// Encrypted HTTP/2 using the legacy TLSNextProto.
		tlsConn, tlsConnOK := pconn.conn.(*tls.Conn)
		if next, ok := t.TLSNextProto[s.NegotiatedProtocol]; tlsConnOK && ok {
			alt := next(cm.targetAddr, tlsConn)
			return wrapTLSNextProtoRoundTripper(alt)
		}
	}

	// All HTTP/1 connections.
	if isClientConn {
		pconn.availch = make(chan struct{}, 1)
		pconn.availch <- struct{}{}
	}
	pconn.br = bufio.NewReaderSize(pconn, t.readBufferSize())
	pconn.bw = bufio.NewWriterSize(persistConnWriter{pconn}, t.writeBufferSize())

	go pconn.readLoop()
	go pconn.writeLoop()
	return pconn, nil
}

// persistConnWriter is the io.Writer written to by pc.bw.
// It accumulates the number of bytes written to the underlying conn,
// so the retry logic can determine whether any bytes made it across
// the wire.
// This is exactly 1 pointer field wide so it can go into an interface
// without allocation.
type persistConnWriter struct {
	pc *persistConn
}

func (w persistConnWriter) Write(p []byte) (n int, err error) {
	n, err = w.pc.conn.Write(p)
	w.pc.nwrite += int64(n)
	return
}

// ReadFrom exposes persistConnWriter's underlying Conn to io.Copy and if
// the Conn implements io.ReaderFrom, it can take advantage of optimizations
// such as sendfile.
func (w persistConnWriter) ReadFrom(r io.Reader) (n int64, err error) {
	n, err = io.Copy(w.pc.conn, r)
	w.pc.nwrite += n
	return
}

var _ io.ReaderFrom = (*persistConnWriter)(nil)

// connectMethod is the map key (in its String form) for keeping persistent
// TCP connections alive for subsequent HTTP requests.
//
// A connect method may be of the following types:
//
//	connectMethod.key().String()      Description
//	------------------------------    -------------------------
//	|http|foo.com                     http directly to server, no proxy
//	|https|foo.com                    https directly to server, no proxy
//	|https,h1|foo.com                 https directly to server w/o HTTP/2, no proxy
//	http://proxy.com|https|foo.com    http to proxy, then CONNECT to foo.com
//	http://proxy.com|http             http to proxy, http to anywhere after that
//	socks5://proxy.com|http|foo.com   socks5 to proxy, then http to foo.com
//	socks5://proxy.com|https|foo.com  socks5 to proxy, then https to foo.com
//	https://proxy.com|https|foo.com   https to proxy, then CONNECT to foo.com
//	https://proxy.com|http            https to proxy, http to anywhere after that
type connectMethod struct {
	_            incomparable
	proxyURL     *url.URL // nil for no proxy, else full proxy URL
	targetScheme string   // "http" or "https"
	// If proxyURL specifies an http or https proxy, and targetScheme is http (not https),
	// then targetAddr is not included in the connect method key, because the socket can
	// be reused for different targetAddr values.
	targetAddr string
	onlyH1     bool // whether to disable HTTP/2 and force HTTP/1
}

func (cm *connectMethod) key() connectMethodKey {
	proxyStr := ""
	targetAddr := cm.targetAddr
	if cm.proxyURL != nil {
		proxyStr = cm.proxyURL.String()
		if (cm.proxyURL.Scheme == "http" || cm.proxyURL.Scheme == "https") && cm.targetScheme == "http" {
			targetAddr = ""
		}
	}
	return connectMethodKey{
		proxy:  proxyStr,
		scheme: cm.targetScheme,
		addr:   targetAddr,
		onlyH1: cm.onlyH1,
	}
}

// scheme returns the first hop scheme: http, https, or socks5
func (cm *connectMethod) scheme() string {
	if cm.proxyURL != nil {
		return cm.proxyURL.Scheme
	}
	return cm.targetScheme
}

// addr returns the first hop "host:port" to which we need to TCP connect.
func (cm *connectMethod) addr() string {
	if cm.proxyURL != nil {
		return canonicalAddr(cm.proxyURL)
	}
	return cm.targetAddr
}

// connectMethodKey is the map key version of connectMethod, with a
// stringified proxy URL (or the empty string) instead of a pointer to
// a URL.
type connectMethodKey struct {
	proxy, scheme, addr string
	onlyH1              bool
}

func (k connectMethodKey) String() string {
	// Only used by tests.
	var h1 string
	if k.onlyH1 {
		h1 = ",h1"
	}
	return fmt.Sprintf("%s|%s%s|%s", k.proxy, k.scheme, h1, k.addr)
}

// persistConn wraps an HTTP/1 client connection, usually a persistent one
// (but may be used for non-keep-alive requests as well).
type persistConn struct {
	poolConn *poolConn

	t            *Transport
	cacheKey     connectMethodKey
	conn         net.Conn
	tlsState     *tls.ConnectionState
	br           *bufio.Reader       // from conn
	bw           *bufio.Writer       // to conn
	nwrite       int64               // bytes written
	reqch        chan requestAndChan // written by roundTrip; read by readLoop
	writech      chan writeRequest   // written by roundTrip; read by writeLoop
	closech      chan struct{}       // closed when conn closed
	availch      chan struct{}       // ClientConn only: contains a value when conn is usable
	isProxy      bool
	sawEOF       bool  // whether we've seen EOF from conn; owned by readLoop
	isClientConn bool  // whether this is a ClientConn (outside any pool)
	readLimit    int64 // bytes allowed to be read; owned by readLoop
	// writeErrCh passes the request write error (usually nil)
	// from the writeLoop goroutine to the readLoop which passes
	// it off to the res.Body reader, which then uses it to decide
	// whether or not a connection can be reused. Issue 7569.
	writeErrCh chan error

	writeLoopDone chan struct{} // closed when write loop ends

	mu                   sync.Mutex // guards following fields
	numExpectedResponses int
	closed               error  // set non-nil when conn is closed, before closech is closed
	canceledErr          error  // set non-nil if conn is canceled
	reused               bool   // whether conn has had successful request/response and is being reused.
	reserved             bool   // ClientConn only: concurrency slot reserved
	inFlight             bool   // ClientConn only: request is in flight
	internalStateHook    func() // ClientConn state hook

	// mutateHeaderFunc is an optional func to modify extra
	// headers on each outbound request before it's written. (the
	// original Request given to RoundTrip is not modified)
	mutateHeaderFunc func(Header)
}

func (pc *persistConn) maxHeaderResponseSize() int64 {
	return pc.t.maxHeaderResponseSize()
}

func (pc *persistConn) Read(p []byte) (n int, err error) {
	if pc.readLimit <= 0 {
		return 0, fmt.Errorf("read limit of %d bytes exhausted", pc.maxHeaderResponseSize())
	}
	if int64(len(p)) > pc.readLimit {
		p = p[:pc.readLimit]
	}
	n, err = pc.conn.Read(p)
	if err == io.EOF {
		pc.sawEOF = true
	}
	pc.readLimit -= int64(n)
	return
}

// isBroken reports whether this connection is in a known broken state.
func (pc *persistConn) isBroken() bool {
	pc.mu.Lock()
	b := pc.closed != nil
	pc.mu.Unlock()
	return b
}

// canceled returns non-nil if the connection was closed due to context cancellation.
func (pc *persistConn) canceled() error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.canceledErr
}

// IsReused reports whether this connection has been used before.
func (pc *persistConn) IsReused() bool {
	pc.mu.Lock()
	r := pc.reused
	pc.mu.Unlock()
	return r
}

func (pc *persistConn) cancelRequest(err error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.canceledErr = err
	pc.closeLocked(errRequestCanceled)
}

// mapRoundTripError returns the appropriate error value for
// persistConn.roundTrip.
//
// The provided err is the first error that (*persistConn).roundTrip
// happened to receive from its select statement.
//
// The startBytesWritten value should be the value of pc.nwrite before the roundTrip
// started writing the request.
func (pc *persistConn) mapRoundTripError(req *transportRequest, startBytesWritten int64, err error) error {
	if err == nil {
		return nil
	}

	// Wait for the writeLoop goroutine to terminate to avoid data
	// races on callers who mutate the request on failure.
	//
	// When resc in pc.roundTrip and hence rc.ch receives a responseAndError
	// with a non-nil error it implies that the persistConn is either closed
	// or closing. Waiting on pc.writeLoopDone is hence safe as all callers
	// close closech which in turn ensures writeLoop returns.
	<-pc.writeLoopDone

	// If the request was canceled, that's better than network
	// failures that were likely the result of tearing down the
	// connection.
	if cerr := pc.canceled(); cerr != nil {
		return cerr
	}

	// See if an error was set explicitly.
	req.mu.Lock()
	reqErr := req.err
	req.mu.Unlock()
	if reqErr != nil {
		return reqErr
	}

	if err == errServerClosedIdle {
		// Don't decorate
		return err
	}

	if _, ok := err.(transportReadFromServerError); ok {
		if pc.nwrite == startBytesWritten {
			return nothingWrittenError{err}
		}
		// Don't decorate
		return err
	}
	if pc.isBroken() {
		if pc.nwrite == startBytesWritten {
			return nothingWrittenError{err}
		}
		return fmt.Errorf("net/http: HTTP/1.x transport connection broken: %w", err)
	}
	return err
}

// errCallerOwnsConn is an internal sentinel error used when we hand
// off a writable response.Body to the caller. We use this to prevent
// closing a net.Conn that is now owned by the caller.
var errCallerOwnsConn = errors.New("read loop ending; caller owns writable underlying conn")

// maxPostCloseReadBytes is the max number of bytes that a client is willing to
// read when draining the response body of any unread bytes after it has been
// closed. This number is chosen for consistency with maxPostHandlerReadBytes.
const maxPostCloseReadBytes = 256 << 10

// maxPostCloseReadTime defines the maximum amount of time that a client is
// willing to spend on draining a response body of any unread bytes after it
// has been closed.
const maxPostCloseReadTime = 50 * time.Millisecond

func maybeDrainBody(r io.Reader) bool {
	drainedCh := make(chan bool, 1)
	go func() {
		// When we drain the body and (hopefully) reach EOF, we might
		// potentially need to deal with trailers. Make sure they are discarded
		// so the connection can actually be reused.
		if b, ok := r.(*body); ok {
			b.discardTrailer()
		}
		if _, err := io.CopyN(io.Discard, r, maxPostCloseReadBytes+1); err == io.EOF {
			drainedCh <- true
		} else {
			drainedCh <- false
		}
	}()
	select {
	case drained := <-drainedCh:
		return drained
	case <-time.After(maxPostCloseReadTime):
		return false
	}
}

type bodyReadStatus uint8

const (
	bodyReadEOF bodyReadStatus = iota
	bodyReadClosedEarly
	bodyReadError
)

func (pc *persistConn) readLoop() {
	closeErr := errReadLoopExiting // default value, if not changed below
	defer func() {
		pc.CloseWithError(closeErr)
		pc.t.removeIdleConn(pc)
		if pc.internalStateHook != nil {
			pc.internalStateHook()
		}
	}()

	tryPutIdleConn := func(treq *transportRequest) bool {
		trace := treq.trace
		if err := pc.t.tryPutIdleConn(pc); err != nil {
			closeErr = err
			if trace != nil && trace.PutIdleConn != nil && err != errKeepAlivesDisabled {
				trace.PutIdleConn(err)
			}
			return false
		}
		if trace != nil && trace.PutIdleConn != nil {
			trace.PutIdleConn(nil)
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
		pc.readLimit = pc.maxHeaderResponseSize()
		_, err := pc.br.Peek(1)

		pc.mu.Lock()
		if pc.numExpectedResponses == 0 {
			pc.readLoopPeekFailLocked(err)
			pc.mu.Unlock()
			return
		}
		pc.mu.Unlock()

		rc := <-pc.reqch
		trace := rc.treq.trace

		var resp *Response
		if err == nil {
			resp, err = pc.readResponse(rc, trace)
		} else {
			err = transportReadFromServerError{err}
			closeErr = err
		}

		if err != nil {
			if pc.readLimit <= 0 {
				err = fmt.Errorf("net/http: server response headers exceeded %d bytes; aborted", pc.maxHeaderResponseSize())
			}

			select {
			case rc.ch <- responseAndError{err: err}:
			case <-rc.callerGone:
				return
			}
			return
		}
		pc.readLimit = maxInt64 // effectively no limit for response bodies

		pc.mu.Lock()
		pc.numExpectedResponses--
		pc.mu.Unlock()

		bodyWritable := resp.bodyIsWritable()
		isConnect := rc.treq.Request.Method == "CONNECT"
		hasBody := rc.treq.Request.Method != "HEAD" && resp.ContentLength != 0

		if resp.Close || rc.treq.Request.Close || resp.StatusCode <= 199 || bodyWritable || isConnect {
			// Don't do keep-alive on error if either party requested a close
			// or we get an unexpected informational (1xx) response.
			// StatusCode 100 is already handled above.
			//
			// Don't do keep-alive after sending a CONNECT request.
			// Only a 2xx response converts the connection into a tunnel,
			// but for safety we'll drop the connection even after getting a non-2xx.
			alive = false
		}

		if !hasBody || bodyWritable {
			// Put the idle conn back into the pool before we send the response
			// so if they process it quickly and make another request, they'll
			// get this same conn. But we use the unbuffered channel 'rc'
			// to guarantee that persistConn.roundTrip got out of its select
			// potentially waiting for this persistConn to close.
			alive = alive &&
				!pc.sawEOF &&
				pc.wroteRequest() &&
				tryPutIdleConn(rc.treq)

			if bodyWritable {
				closeErr = errCallerOwnsConn
			}

			select {
			case rc.ch <- responseAndError{res: resp}:
			case <-rc.callerGone:
				return
			}

			rc.treq.cancel(errRequestDone)

			// Now that they've read from the unbuffered channel, they're safely
			// out of the select that also waits on this goroutine to die, so
			// we're allowed to exit now if needed (if alive is false)
			testHookReadLoopBeforeNextRead()
			continue
		}

		waitForBodyRead := make(chan bodyReadStatus, 1)
		body := &bodyEOFSignal{
			body: resp.Body,
			earlyCloseFn: func() error {
				waitForBodyRead <- bodyReadClosedEarly
				<-eofc // will be closed by deferred call at the end of the function
				return nil
			},
			fn: func(err error) error {
				if err == io.EOF {
					waitForBodyRead <- bodyReadEOF
					<-eofc // see comment above eofc declaration
				} else {
					waitForBodyRead <- bodyReadError
					if err != nil {
						if cerr := pc.canceled(); cerr != nil {
							return cerr
						}
					}
				}
				return err
			},
		}

		resp.Body = body
		if rc.addedGzip && ascii.EqualFold(resp.Header.Get("Content-Encoding"), "gzip") {
			resp.Body = &httpcommon.GzipReader{Body: body}
			resp.Header.Del("Content-Encoding")
			resp.Header.Del("Content-Length")
			resp.ContentLength = -1
			resp.Uncompressed = true
		}

		select {
		case rc.ch <- responseAndError{res: resp}:
		case <-rc.callerGone:
			return
		}

		// Before looping back to the top of this function and peeking on
		// the bufio.Reader, wait for the caller goroutine to finish
		// reading the response body. (or for cancellation or death)
		select {
		case status := <-waitForBodyRead:
			tryPutIdle := func() {
				alive = alive &&
					!pc.sawEOF &&
					pc.wroteRequest() &&
					tryPutIdleConn(rc.treq)
			}
			switch status {
			case bodyReadEOF:
				tryPutIdle()
				eofc <- struct{}{}
			case bodyReadClosedEarly:
				// Read resp before signaling eofc: the send lets the caller's
				// Close return, and resp belongs to the caller after that.
				tryDrain := alive && !pc.t.keepAlivesDisabled() && resp.ContentLength <= maxPostCloseReadBytes
				eofc <- struct{}{}
				if tryDrain && maybeDrainBody(body.body) {
					tryPutIdle()
				} else {
					alive = false
				}
			case bodyReadError:
				alive = false
			}
		case <-rc.treq.ctx.Done():
			alive = false
			pc.cancelRequest(context.Cause(rc.treq.ctx))
		case <-pc.closech:
			alive = false
		}

		rc.treq.cancel(errRequestDone)
		testHookReadLoopBeforeNextRead()
	}
}

func (pc *persistConn) readLoopPeekFailLocked(peekErr error) {
	if pc.closed != nil {
		return
	}
	if n := pc.br.Buffered(); n > 0 {
		buf, _ := pc.br.Peek(n)
		if is408Message(buf) {
			pc.closeLocked(errServerClosedIdle)
			return
		} else {
			log.Printf("Unsolicited response received on idle HTTP channel starting with %q; err=%v", buf, peekErr)
		}
	}
	if peekErr == io.EOF {
		// common case.
		pc.closeLocked(errServerClosedIdle)
	} else {
		pc.closeLocked(fmt.Errorf("readLoopPeekFailLocked: %w", peekErr))
	}
}

// is408Message reports whether buf has the prefix of an
// HTTP 408 Request Timeout response.
// See golang.org/issue/32310.
func is408Message(buf []byte) bool {
	if len(buf) < len("HTTP/1.x 408") {
		return false
	}
	if string(buf[:7]) != "HTTP/1." {
		return false
	}
	return string(buf[8:12]) == " 408"
}

// readResponse reads an HTTP response (or two, in the case of "Expect:
// 100-continue") from the server. It returns the final non-100 one.
// trace is optional.
func (pc *persistConn) readResponse(rc requestAndChan, trace *httptrace.ClientTrace) (resp *Response, err error) {
	if trace != nil && trace.GotFirstResponseByte != nil {
		if peek, err := pc.br.Peek(1); err == nil && len(peek) == 1 {
			trace.GotFirstResponseByte()
		}
	}

	continueCh := rc.continueCh
	for {
		resp, err = ReadResponse(pc.br, rc.treq.Request)
		if err != nil {
			return
		}
		resCode := resp.StatusCode
		if continueCh != nil && resCode == StatusContinue {
			if trace != nil && trace.Got100Continue != nil {
				trace.Got100Continue()
			}
			continueCh <- struct{}{}
			continueCh = nil
		}
		is1xx := 100 <= resCode && resCode <= 199
		// treat 101 as a terminal status, see issue 26161
		is1xxNonTerminal := is1xx && resCode != StatusSwitchingProtocols
		if is1xxNonTerminal {
			if trace != nil && trace.Got1xxResponse != nil {
				if err := trace.Got1xxResponse(resCode, textproto.MIMEHeader(resp.Header)); err != nil {
					return nil, err
				}
				// If the 1xx response was delivered to the user,
				// then they're responsible for limiting the number of
				// responses. Reset the header limit.
				//
				// If the user didn't examine the 1xx response, then we
				// limit the size of all headers (including both 1xx
				// and the final response) to maxHeaderResponseSize.
				pc.readLimit = pc.maxHeaderResponseSize() // reset the limit
			}
			continue
		}
		break
	}
	if resp.isProtocolSwitch() {
		resp.Body = newReadWriteCloserBody(pc.br, pc.conn)
	}
	if continueCh != nil {
		// We send an "Expect: 100-continue" header, but the server
		// responded with a terminal status and no 100 Continue.
		//
		// If we're going to keep using the connection, we need to send the request body.
		// Tell writeLoop to skip sending the body if we're going to close the connection,
		// or to send it otherwise.
		//
		// The case where we receive a 101 Switching Protocols response is a bit
		// ambiguous, since we don't know what protocol we're switching to.
		// Conceivably, it's one that doesn't need us to send the body.
		// Given that we'll send the body if ExpectContinueTimeout expires,
		// be consistent and always send it if we aren't closing the connection.
		if resp.Close || rc.treq.Request.Close {
			close(continueCh) // don't send the body; the connection will close
		} else {
			continueCh <- struct{}{} // send the body
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

func newReadWriteCloserBody(br *bufio.Reader, rwc io.ReadWriteCloser) io.ReadWriteCloser {
	body := &readWriteCloserBody{ReadWriteCloser: rwc}
	if br.Buffered() != 0 {
		body.br = br
	}
	return body
}

// readWriteCloserBody is the Response.Body type used when we want to
// give users write access to the Body through the underlying
// connection (TCP, unless using custom dialers). This is then
// the concrete type for a Response.Body on the 101 Switching
// Protocols response, as used by WebSockets, h2c, etc.
type readWriteCloserBody struct {
	_  incomparable
	br *bufio.Reader // used until empty
	io.ReadWriteCloser
}

func (b *readWriteCloserBody) Read(p []byte) (n int, err error) {
	if b.br != nil {
		if n := b.br.Buffered(); len(p) > n {
			p = p[:n]
		}
		n, err = b.br.Read(p)
		if b.br.Buffered() == 0 {
			b.br = nil
		}
		return n, err
	}
	return b.ReadWriteCloser.Read(p)
}

func (b *readWriteCloserBody) CloseWrite() error {
	if cw, ok := b.ReadWriteCloser.(interface{ CloseWrite() error }); ok {
		return cw.CloseWrite()
	}
	return fmt.Errorf("CloseWrite: %w", ErrNotSupported)
}

// nothingWrittenError wraps a write errors which ended up writing zero bytes.
type nothingWrittenError struct {
	error
}

func (nwe nothingWrittenError) Unwrap() error {
	return nwe.error
}

func (pc *persistConn) writeLoop() {
	defer close(pc.writeLoopDone)
	for {
		select {
		case wr := <-pc.writech:
			startBytesWritten := pc.nwrite
			err := wr.req.Request.write(pc.bw, pc.isProxy, wr.req.extra, pc.waitForContinue(wr.continueCh))
			if bre, ok := err.(requestBodyReadError); ok {
				err = bre.error
				// Errors reading from the user's
				// Request.Body are high priority.
				// Set it here before sending on the
				// channels below or calling
				// pc.close() which tears down
				// connections and causes other
				// errors.
				wr.req.setError(err)
			}
			if err == nil {
				err = pc.bw.Flush()
			}
			if err != nil {
				if pc.nwrite == startBytesWritten {
					err = nothingWrittenError{err}
				}
			}
			pc.writeErrCh <- err // to the body reader, which might recycle us
			wr.ch <- err         // to the roundTrip function
			if err != nil {
				// The conn will be closed by either roundTrip or readLoop.
				return
			}
		case <-pc.closech:
			return
		}
	}
}

// maxWriteWaitBeforeConnReuse is how long the a Transport RoundTrip
// will wait to see the Request's Body.Write result after getting a
// response from the server. See comments in (*persistConn).wroteRequest.
//
// In tests, we set this to a large value to avoid flakiness from inconsistent
// recycling of connections.
var maxWriteWaitBeforeConnReuse = 50 * time.Millisecond

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
		t := time.NewTimer(maxWriteWaitBeforeConnReuse)
		defer t.Stop()
		select {
		case err := <-pc.writeErrCh:
			return err == nil
		case <-t.C:
			return false
		}
	}
}

// responseAndError is how the goroutine reading from an HTTP/1 server
// communicates with the goroutine doing the RoundTrip.
type responseAndError struct {
	_   incomparable
	res *Response // else use this response (see res method)
	err error
}

type requestAndChan struct {
	_    incomparable
	treq *transportRequest
	ch   chan responseAndError // unbuffered; always send in select on callerGone

	// whether the Transport (as opposed to the user client code)
	// added the Accept-Encoding gzip header. If the Transport
	// set it, only then do we transparently decode the gzip.
	addedGzip bool

	// Optional blocking chan for Expect: 100-continue (for send).
	// If the request has an "Expect: 100-continue" header and
	// the server responds 100 Continue, readLoop send a value
	// to writeLoop via this chan.
	continueCh chan<- struct{}

	callerGone <-chan struct{} // closed when roundTrip caller has returned
}

// A writeRequest is sent by the caller's goroutine to the
// writeLoop's goroutine to write a request while the read loop
// concurrently waits on both the write response and the server's
// reply.
type writeRequest struct {
	req *transportRequest
	ch  chan<- error

	// Optional blocking chan for Expect: 100-continue (for receive).
	// If not nil, writeLoop blocks sending request body until
	// it receives from this chan.
	continueCh <-chan struct{}
}

// httpTimeoutError represents a timeout.
// It implements net.Error and wraps context.DeadlineExceeded.
type timeoutError struct {
	err string
}

func (e *timeoutError) Error() string     { return e.err }
func (e *timeoutError) Timeout() bool     { return true }
func (e *timeoutError) Temporary() bool   { return true }
func (e *timeoutError) Is(err error) bool { return err == context.DeadlineExceeded }

var errTimeout error = &timeoutError{"net/http: timeout awaiting response headers"}

// errRequestCanceled is set to be identical to the one from h2 to facilitate
// testing.
var errRequestCanceled = internal.ErrRequestCanceled
var errRequestCanceledConn = errors.New("net/http: request canceled while waiting for connection") // TODO: unify?

// errRequestDone is used to cancel the round trip Context after a request is successfully done.
// It should not be seen by the user.
var errRequestDone = errors.New("net/http: request completed")

func nop() {}

// testHooks. Always non-nil.
var (
	testHookEnterRoundTrip   = nop
	testHookWaitResLoop      = nop
	testHookRoundTripRetried = nop

	testHookMu                     sync.Locker = fakeLocker{} // guards following
	testHookReadLoopBeforeNextRead             = nop
)

var (
	testHookPrePendingDial  func()
	testHookPostPendingDial func()
)

func (pc *persistConn) waitForAvailability(ctx context.Context) error {
	select {
	case <-pc.availch:
		return nil
	case <-pc.closech:
		return pc.closed
	case <-ctx.Done():
		return ctx.Err()
	}
}

const postRequestWriteErrorWaitTime = 50 * time.Millisecond

func (pc *persistConn) roundTrip(req *transportRequest) (resp *Response, err error) {
	testHookEnterRoundTrip()

	pc.mu.Lock()
	if pc.isClientConn {
		if !pc.reserved {
			pc.mu.Unlock()
			if err := pc.waitForAvailability(req.ctx); err != nil {
				return nil, err
			}
			pc.mu.Lock()
		}
		pc.reserved = false
		pc.inFlight = true
	}
	pc.numExpectedResponses++
	headerFn := pc.mutateHeaderFunc
	pc.mu.Unlock()

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
		// See: https://zlib.net/zlib_faq.html#faq39
		//
		// Note that we don't request this for HEAD requests,
		// due to a bug in nginx:
		//   https://trac.nginx.org/nginx/ticket/358
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

	if pc.t.DisableKeepAlives &&
		!req.wantsClose() &&
		!isProtocolSwitchHeader(req.Header) {
		req.extraHeaders().Set("Connection", "close")
	}

	gone := make(chan struct{})
	defer close(gone)

	const debugRoundTrip = false

	// Write the request concurrently with waiting for a response,
	// in case the server decides to reply before reading our full
	// request body.
	startBytesWritten := pc.nwrite
	writeErrCh := make(chan error, 1)
	pc.writech <- writeRequest{req, writeErrCh, continueCh}

	resc := make(chan responseAndError)
	pc.reqch <- requestAndChan{
		treq:       req,
		ch:         resc,
		addedGzip:  requestedGzip,
		continueCh: continueCh,
		callerGone: gone,
	}

	handleResponse := func(re responseAndError) (*Response, error) {
		if (re.res == nil) == (re.err == nil) {
			panic(fmt.Sprintf("internal error: exactly one of res or err should be set; nil=%v", re.res == nil))
		}
		if debugRoundTrip {
			req.logf("resc recv: %p, %T/%#v", re.res, re.err, re.err)
		}
		if re.err != nil {
			return nil, pc.mapRoundTripError(req, startBytesWritten, re.err)
		}
		return re.res, nil
	}

	var writeErr error
	var respHeaderTimer <-chan time.Time
	ctxDoneChan := req.ctx.Done()
	pcClosed := pc.closech
	for {
		testHookWaitResLoop()
		select {
		case err := <-writeErrCh:
			if debugRoundTrip {
				req.logf("writeErrCh recv: %T/%#v", err, err)
			}
			timeout := pc.t.ResponseHeaderTimeout
			if err != nil {
				// If this is something other than a network write error
				// (presumably an error reading from the user-provided body),
				// then return immediately.
				//
				// Otherwise, it's possible that the server sent a response
				// and immediately closed the connection. Wait for a short
				// time to see if a response shows up.
				//
				// See #11745.
				if _, ok := errors.AsType[*net.OpError](err); !ok {
					pc.CloseWithError(fmt.Errorf("write error: %w", err))
					return nil, pc.mapRoundTripError(req, startBytesWritten, err)
				}
				writeErr = err
				if timeout == 0 {
					timeout = postRequestWriteErrorWaitTime
				} else {
					timeout = min(timeout, postRequestWriteErrorWaitTime)
				}
			}
			if timeout > 0 {
				if debugRoundTrip {
					req.logf("starting timer for %v", timeout)
				}
				timer := time.NewTimer(timeout)
				respHeaderTimer = timer.C
			}
		case <-pcClosed:
			select {
			case re := <-resc:
				// The pconn closing raced with the response to the request,
				// probably after the server wrote a response and immediately
				// closed the connection. Use the response.
				return handleResponse(re)
			default:
			}
			if debugRoundTrip {
				req.logf("closech recv: %T %#v", pc.closed, pc.closed)
			}
			return nil, pc.mapRoundTripError(req, startBytesWritten, pc.closed)
		case <-respHeaderTimer:
			if debugRoundTrip {
				req.logf("timeout waiting for response headers.")
			}
			if writeErr != nil {
				// Error writing request body, and we haven't read response
				// headers within postRequestWriteErrorWaitTime.
				pc.CloseWithError(fmt.Errorf("write error: %w", writeErr))
				return nil, pc.mapRoundTripError(req, startBytesWritten, writeErr)
			}
			pc.CloseWithError(errTimeout)
			return nil, errTimeout
		case re := <-resc:
			return handleResponse(re)
		case <-ctxDoneChan:
			select {
			case re := <-resc:
				// readLoop is responsible for canceling req.ctx after
				// it reads the response body. Check for a response racing
				// the context close, and use the response if available.
				return handleResponse(re)
			default:
			}
			pc.cancelRequest(context.Cause(req.ctx))
		}
	}
}

// tLogKey is a context WithValue key for test debugging contexts containing
// a t.Logf func. See export_test.go's Request.WithT method.
type tLogKey struct{}

func (tr *transportRequest) logf(format string, args ...any) {
	if logf, ok := tr.Request.Context().Value(tLogKey{}).(func(string, ...any)); ok {
		logf(time.Now().Format(time.RFC3339Nano)+": "+format, args...)
	}
}

// markReused marks this connection as having been successfully used for a
// request and response.
func (pc *persistConn) markReused() {
	pc.mu.Lock()
	pc.reused = true
	pc.mu.Unlock()
}

// CloseWithError closes the underlying TCP connection and closes
// the pc.closech channel.
//
// The provided err is only for testing and debugging; in normal
// circumstances it should never be seen by users.
func (pc *persistConn) CloseWithError(err error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.closeLocked(err)
}

func (pc *persistConn) closeLocked(err error) {
	if err == nil {
		panic("nil error")
	}
	if pc.closed == nil {
		pc.closed = err
		if err != errCallerOwnsConn {
			pc.conn.Close()
		}
		close(pc.closech)
	}
	pc.mutateHeaderFunc = nil
}

func schemePort(scheme string) string {
	switch scheme {
	case "http":
		return "80"
	case "https":
		return "443"
	case "socks5", "socks5h":
		return "1080"
	default:
		return ""
	}
}

func idnaASCIIFromURL(url *url.URL) string {
	addr := url.Hostname()
	if v, err := idnaASCII(addr); err == nil && v != "" {
		addr = v
	}
	return addr
}

// canonicalAddr returns url.Host but always with a ":port" suffix.
func canonicalAddr(url *url.URL) string {
	port := url.Port()
	if port == "" {
		port = schemePort(url.Scheme)
	}
	return net.JoinHostPort(idnaASCIIFromURL(url), port)
}

// bodyEOFSignal is used by the HTTP/1 transport when reading response
// bodies to make sure we see the end of a response body before
// proceeding and reading on the connection again.
//
// It wraps a ReadCloser but runs fn (if non-nil) at most
// once, right before its final (error-producing) Read or Close call
// returns. fn should return the new error to return from Read or Close.
//
// If earlyCloseFn is non-nil and Close is called before any final error from
// Read is seen, earlyCloseFn is called instead of fn, and its return value is
// the return value from Close.
type bodyEOFSignal struct {
	body         io.ReadCloser
	mu           sync.Mutex        // guards following 4 fields
	closed       bool              // whether Close has been called
	rerr         error             // sticky Read error
	fn           func(error) error // called on final body.Read non-nil error (or body.Close if earlyCloseFn is not run)
	earlyCloseFn func() error      // called if body.Close is called before body.Read ever returns a non-nil error
}

var errReadOnClosedResBody = errors.New("http: read on closed response body")

func (es *bodyEOFSignal) Read(p []byte) (n int, err error) {
	es.mu.Lock()
	closed, rerr := es.closed, es.rerr
	es.mu.Unlock()
	if closed {
		return 0, errReadOnClosedResBody
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
	if es.earlyCloseFn != nil && es.rerr == nil {
		earlyCloseFn := es.earlyCloseFn
		es.earlyCloseFn = nil
		es.fn = nil
		return earlyCloseFn()
	}
	if es.rerr != nil && es.rerr != io.EOF {
		// Read already returned this error and readLoop gave up the
		// connection. Draining would only read the same error again.
		return nil
	}
	err := es.body.Close()
	return es.condfn(err)
}

// caller must hold es.mu.
func (es *bodyEOFSignal) condfn(err error) error {
	if es.fn == nil {
		return err
	}
	fn := es.fn
	es.fn = nil
	es.earlyCloseFn = nil
	return fn(err)
}

type tlsHandshakeTimeoutError struct{}

func (tlsHandshakeTimeoutError) Timeout() bool   { return true }
func (tlsHandshakeTimeoutError) Temporary() bool { return true }
func (tlsHandshakeTimeoutError) Error() string   { return "net/http: TLS handshake timeout" }

// fakeLocker is a sync.Locker which does nothing. It's used to guard
// test-only fields when not under test, to avoid runtime atomic
// overhead.
type fakeLocker struct{}

func (fakeLocker) Lock()   {}
func (fakeLocker) Unlock() {}

// cloneTLSConfig returns a shallow clone of cfg, or a new zero tls.Config if
// cfg is nil. This is safe to call even if cfg is in active use by a TLS
// client or server.
//
// cloneTLSConfig should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/searKing/golang
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname cloneTLSConfig
func cloneTLSConfig(cfg *tls.Config) *tls.Config {
	if cfg == nil {
		return &tls.Config{}
	}
	return cfg.Clone()
}
