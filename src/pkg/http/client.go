// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Primitive HTTP client. See RFC 2616.

package http

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
)

// Given a string of the form "host", "host:port", or "[ipv6::address]:port",
// return true if the string includes a port.
func hasPort(s string) bool { return strings.LastIndex(s, ":") > strings.LastIndex(s, "]") }

// Used in Send to implement io.ReadCloser by bundling together the
// bufio.Reader through which we read the response, and the underlying
// network connection.
type readClose struct {
	io.Reader
	io.Closer
}

// Send issues an HTTP request.  Caller should close resp.Body when done reading it.
//
// TODO: support persistent connections (multiple requests on a single connection).
// send() method is nonpublic because, when we refactor the code for persistent
// connections, it may no longer make sense to have a method with this signature.
func send(req *Request) (resp *Response, err os.Error) {
	if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
		return nil, &badStringError{"unsupported protocol scheme", req.URL.Scheme}
	}

	addr := req.URL.Host
	if !hasPort(addr) {
		addr += ":" + req.URL.Scheme
	}
	info := req.URL.RawUserinfo
	if len(info) > 0 {
		enc := base64.URLEncoding
		encoded := make([]byte, enc.EncodedLen(len(info)))
		enc.Encode(encoded, []byte(info))
		if req.Header == nil {
			req.Header = make(map[string]string)
		}
		req.Header["Authorization"] = "Basic " + string(encoded)
	}

	var conn io.ReadWriteCloser
	if req.URL.Scheme == "http" {
		conn, err = net.Dial("tcp", "", addr)
		if err != nil {
			return nil, err
		}
	} else { // https
		conn, err = tls.Dial("tcp", "", addr, nil)
		if err != nil {
			return nil, err
		}
		h := req.URL.Host
		if hasPort(h) {
			h = h[0:strings.LastIndex(h, ":")]
		}
		if err := conn.(*tls.Conn).VerifyHostname(h); err != nil {
			return nil, err
		}
	}

	err = req.Write(conn)
	if err != nil {
		conn.Close()
		return nil, err
	}

	reader := bufio.NewReader(conn)
	resp, err = ReadResponse(reader, req.Method)
	if err != nil {
		conn.Close()
		return nil, err
	}

	resp.Body = readClose{resp.Body, conn}

	return
}

// True if the specified HTTP status code is one for which the Get utility should
// automatically redirect.
func shouldRedirect(statusCode int) bool {
	switch statusCode {
	case StatusMovedPermanently, StatusFound, StatusSeeOther, StatusTemporaryRedirect:
		return true
	}
	return false
}

// Get issues a GET to the specified URL.  If the response is one of the following
// redirect codes, it follows the redirect, up to a maximum of 10 redirects:
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
//
// finalURL is the URL from which the response was fetched -- identical to the
// input URL unless redirects were followed.
//
// Caller should close r.Body when done reading it.
func Get(url string) (r *Response, finalURL string, err os.Error) {
	// TODO: if/when we add cookie support, the redirected request shouldn't
	// necessarily supply the same cookies as the original.
	// TODO: set referrer header on redirects.
	var base *URL
	for redirect := 0; ; redirect++ {
		if redirect >= 10 {
			err = os.ErrorString("stopped after 10 redirects")
			break
		}

		var req Request
		if base == nil {
			req.URL, err = ParseURL(url)
		} else {
			req.URL, err = base.ParseURL(url)
		}
		if err != nil {
			break
		}
		url = req.URL.String()
		if r, err = send(&req); err != nil {
			break
		}
		if shouldRedirect(r.StatusCode) {
			r.Body.Close()
			if url = r.GetHeader("Location"); url == "" {
				err = os.ErrorString(fmt.Sprintf("%d response missing Location header", r.StatusCode))
				break
			}
			base = req.URL
			continue
		}
		finalURL = url
		return
	}

	err = &URLError{"Get", url, err}
	return
}

// Post issues a POST to the specified URL.
//
// Caller should close r.Body when done reading it.
func Post(url string, bodyType string, body io.Reader) (r *Response, err os.Error) {
	var req Request
	req.Method = "POST"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = true
	req.Body = nopCloser{body}
	req.Header = map[string]string{
		"Content-Type": bodyType,
	}
	req.TransferEncoding = []string{"chunked"}

	req.URL, err = ParseURL(url)
	if err != nil {
		return nil, err
	}

	return send(&req)
}

// PostForm issues a POST to the specified URL, 
// with data's keys and values urlencoded as the request body.
//
// Caller should close r.Body when done reading it.
func PostForm(url string, data map[string]string) (r *Response, err os.Error) {
	var req Request
	req.Method = "POST"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = true
	body := urlencode(data)
	req.Body = nopCloser{body}
	req.Header = map[string]string{
		"Content-Type":   "application/x-www-form-urlencoded",
		"Content-Length": strconv.Itoa(body.Len()),
	}
	req.ContentLength = int64(body.Len())

	req.URL, err = ParseURL(url)
	if err != nil {
		return nil, err
	}

	return send(&req)
}

// TODO: remove this function when PostForm takes a multimap.
func urlencode(data map[string]string) (b *bytes.Buffer) {
	m := make(map[string][]string, len(data))
	for k, v := range data {
		m[k] = []string{v}
	}
	return bytes.NewBuffer([]byte(EncodeQuery(m)))
}

// Head issues a HEAD to the specified URL.
func Head(url string) (r *Response, err os.Error) {
	var req Request
	req.Method = "HEAD"
	if req.URL, err = ParseURL(url); err != nil {
		return
	}
	url = req.URL.String()
	if r, err = send(&req); err != nil {
		return
	}
	return
}

type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() os.Error { return nil }
