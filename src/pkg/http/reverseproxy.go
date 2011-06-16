// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP reverse proxy handler

package http

import (
	"io"
	"log"
	"net"
	"strings"
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
func NewSingleHostReverseProxy(target *URL) *ReverseProxy {
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

	if clientIp, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
		outreq.Header.Set("X-Forwarded-For", clientIp)
	}

	res, err := transport.RoundTrip(outreq)
	if err != nil {
		log.Printf("http: proxy error: %v", err)
		rw.WriteHeader(StatusInternalServerError)
		return
	}

	hdr := rw.Header()
	for k, vv := range res.Header {
		for _, v := range vv {
			hdr.Add(k, v)
		}
	}

	rw.WriteHeader(res.StatusCode)

	if res.Body != nil {
		io.Copy(rw, res.Body)
	}
}
