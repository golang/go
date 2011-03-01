// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Primitive HTTP client. See RFC 2616.

package http

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// A Client is an HTTP client. Its zero value (DefaultClient) is a usable client
// that uses DefaultTransport.
// Client is not yet very configurable.
type Client struct {
	Transport ClientTransport // if nil, DefaultTransport is used
}

// DefaultClient is the default Client and is used by Get, Head, and Post.
var DefaultClient = &Client{}

// ClientTransport is an interface representing the ability to execute a
// single HTTP transaction, obtaining the Response for a given Request.
type ClientTransport interface {
	// Do executes a single HTTP transaction, returning the Response for the
	// request req.  Do should not attempt to interpret the response.
	// In particular, Do must return err == nil if it obtained a response,
	// regardless of the response's HTTP status code.  A non-nil err should
	// be reserved for failure to obtain a response.  Similarly, Do should
	// not attempt to handle higher-level protocol details such as redirects,
	// authentication, or cookies.
	Do(req *Request) (resp *Response, err os.Error)
}

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

// matchNoProxy returns true if requests to addr should not use a proxy,
// according to the NO_PROXY or no_proxy environment variable.
func matchNoProxy(addr string) bool {
	if len(addr) == 0 {
		return false
	}
	no_proxy := os.Getenv("NO_PROXY")
	if len(no_proxy) == 0 {
		no_proxy = os.Getenv("no_proxy")
	}
	if no_proxy == "*" {
		return true
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
			return true
		}
	}
	return false
}

// Do sends an HTTP request and returns an HTTP response, following
// policy (e.g. redirects, cookies, auth) as configured on the client.
//
// Callers should close resp.Body when done reading from it.
//
// Generally Get, Post, or PostForm will be used instead of Do.
func (c *Client) Do(req *Request) (resp *Response, err os.Error) {
	return send(req, c.Transport)
}


// send issues an HTTP request.  Caller should close resp.Body when done reading from it.
//
// TODO: support persistent connections (multiple requests on a single connection).
// send() method is nonpublic because, when we refactor the code for persistent
// connections, it may no longer make sense to have a method with this signature.
func send(req *Request, t ClientTransport) (resp *Response, err os.Error) {
	if t == nil {
		t = DefaultTransport
		if t == nil {
			err = os.NewError("no http.Client.Transport or http.DefaultTransport")
			return
		}
	}
	info := req.URL.RawUserinfo
	if len(info) > 0 {
		enc := base64.URLEncoding
		encoded := make([]byte, enc.EncodedLen(len(info)))
		enc.Encode(encoded, []byte(info))
		if req.Header == nil {
			req.Header = make(Header)
		}
		req.Header.Set("Authorization", "Basic "+string(encoded))
	}
	return t.Do(req)
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
// Caller should close r.Body when done reading from it.
//
// Get is a convenience wrapper around DefaultClient.Get.
func Get(url string) (r *Response, finalURL string, err os.Error) {
	return DefaultClient.Get(url)
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
// Caller should close r.Body when done reading from it.
func (c *Client) Get(url string) (r *Response, finalURL string, err os.Error) {
	// TODO: if/when we add cookie support, the redirected request shouldn't
	// necessarily supply the same cookies as the original.
	// TODO: set referrer header on redirects.
	var base *URL
	// TODO: remove this hard-coded 10 and use the Client's policy
	// (ClientConfig) instead.
	for redirect := 0; ; redirect++ {
		if redirect >= 10 {
			err = os.ErrorString("stopped after 10 redirects")
			break
		}

		var req Request
		req.Method = "GET"
		req.ProtoMajor = 1
		req.ProtoMinor = 1
		if base == nil {
			req.URL, err = ParseURL(url)
		} else {
			req.URL, err = base.ParseURL(url)
		}
		if err != nil {
			break
		}
		url = req.URL.String()
		if r, err = send(&req, c.Transport); err != nil {
			break
		}
		if shouldRedirect(r.StatusCode) {
			r.Body.Close()
			if url = r.Header.Get("Location"); url == "" {
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
// Caller should close r.Body when done reading from it.
//
// Post is a wrapper around DefaultClient.Post
func Post(url string, bodyType string, body io.Reader) (r *Response, err os.Error) {
	return DefaultClient.Post(url, bodyType, body)
}

// Post issues a POST to the specified URL.
//
// Caller should close r.Body when done reading from it.
func (c *Client) Post(url string, bodyType string, body io.Reader) (r *Response, err os.Error) {
	var req Request
	req.Method = "POST"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = true
	req.Body = nopCloser{body}
	req.Header = Header{
		"Content-Type": {bodyType},
	}
	req.TransferEncoding = []string{"chunked"}

	req.URL, err = ParseURL(url)
	if err != nil {
		return nil, err
	}

	return send(&req, c.Transport)
}

// PostForm issues a POST to the specified URL, 
// with data's keys and values urlencoded as the request body.
//
// Caller should close r.Body when done reading from it.
//
// PostForm is a wrapper around DefaultClient.PostForm
func PostForm(url string, data map[string]string) (r *Response, err os.Error) {
	return DefaultClient.PostForm(url, data)
}

// PostForm issues a POST to the specified URL, 
// with data's keys and values urlencoded as the request body.
//
// Caller should close r.Body when done reading from it.
func (c *Client) PostForm(url string, data map[string]string) (r *Response, err os.Error) {
	var req Request
	req.Method = "POST"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = true
	body := urlencode(data)
	req.Body = nopCloser{body}
	req.Header = Header{
		"Content-Type":   {"application/x-www-form-urlencoded"},
		"Content-Length": {strconv.Itoa(body.Len())},
	}
	req.ContentLength = int64(body.Len())

	req.URL, err = ParseURL(url)
	if err != nil {
		return nil, err
	}

	return send(&req, c.Transport)
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
//
// Head is a wrapper around DefaultClient.Head
func Head(url string) (r *Response, err os.Error) {
	return DefaultClient.Head(url)
}

// Head issues a HEAD to the specified URL.
func (c *Client) Head(url string) (r *Response, err os.Error) {
	var req Request
	req.Method = "HEAD"
	if req.URL, err = ParseURL(url); err != nil {
		return
	}
	return send(&req, c.Transport)
}

type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() os.Error { return nil }
