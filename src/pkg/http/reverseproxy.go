// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP reverse proxy handler

package http

import (
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"time"
	"url"
)

// ReverseProxy is an HTTP Handler that takes an incoming request and
// sends it to another server, proxying the response back to the
// client.
type ReverseProxy struct {
	// Director must be a function which modifies
	// the request into a new request to be sent
	// using Transport. Its response is then copied
	// back to the original client unmodified.
	Director func(*Request)

	// The Transport used to perform proxy requests.
	// If nil, DefaultTransport is used.
	Transport RoundTripper

	// FlushInterval specifies the flush interval, in
	// nanoseconds, to flush to the client while
	// coping the response body.
	// If zero, no periodic flushing is done.
	FlushInterval int64
}

func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}

// NewSingleHostReverseProxy returns a new ReverseProxy that rewrites
// URLs to the scheme, host, and base path provided in target. If the
// target's path is "/base" and the incoming request was for "/dir",
// the target request will be for /base/dir.
func NewSingleHostReverseProxy(target *url.URL) *ReverseProxy {
	director := func(req *Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.URL.Path = singleJoiningSlash(target.Path, req.URL.Path)
		if q := req.URL.RawQuery; q != "" {
			req.URL.RawPath = req.URL.Path + "?" + q
		} else {
			req.URL.RawPath = req.URL.Path
		}
		req.URL.RawQuery = target.RawQuery
	}
	return &ReverseProxy{Director: director}
}

func copyHeader(dst, src Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func (p *ReverseProxy) ServeHTTP(rw ResponseWriter, req *Request) {
	transport := p.Transport
	if transport == nil {
		transport = DefaultTransport
	}

	outreq := new(Request)
	*outreq = *req // includes shallow copies of maps, but okay

	p.Director(outreq)
	outreq.Proto = "HTTP/1.1"
	outreq.ProtoMajor = 1
	outreq.ProtoMinor = 1
	outreq.Close = false

	// Remove the connection header to the backend.  We want a
	// persistent connection, regardless of what the client sent
	// to us.  This is modifying the same underlying map from req
	// (shallow copied above) so we only copy it if necessary.
	if outreq.Header.Get("Connection") != "" {
		outreq.Header = make(Header)
		copyHeader(outreq.Header, req.Header)
		outreq.Header.Del("Connection")
	}

	if clientIp, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
		outreq.Header.Set("X-Forwarded-For", clientIp)
	}

	res, err := transport.RoundTrip(outreq)
	if err != nil {
		log.Printf("http: proxy error: %v", err)
		rw.WriteHeader(StatusInternalServerError)
		return
	}

	copyHeader(rw.Header(), res.Header)

	rw.WriteHeader(res.StatusCode)

	if res.Body != nil {
		var dst io.Writer = rw
		if p.FlushInterval != 0 {
			if wf, ok := rw.(writeFlusher); ok {
				dst = &maxLatencyWriter{dst: wf, latency: p.FlushInterval}
			}
		}
		io.Copy(dst, res.Body)
	}
}

type writeFlusher interface {
	io.Writer
	Flusher
}

type maxLatencyWriter struct {
	dst     writeFlusher
	latency int64 // nanos

	lk   sync.Mutex // protects init of done, as well Write + Flush
	done chan bool
}

func (m *maxLatencyWriter) Write(p []byte) (n int, err os.Error) {
	m.lk.Lock()
	defer m.lk.Unlock()
	if m.done == nil {
		m.done = make(chan bool)
		go m.flushLoop()
	}
	n, err = m.dst.Write(p)
	if err != nil {
		m.done <- true
	}
	return
}

func (m *maxLatencyWriter) flushLoop() {
	t := time.NewTicker(m.latency)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			m.lk.Lock()
			m.dst.Flush()
			m.lk.Unlock()
		case <-m.done:
			return
		}
	}
	panic("unreached")
}
