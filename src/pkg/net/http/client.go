// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP client. See RFC 2616.
// 
// This is the high-level Client interface.
// The low-level implementation is in transport.go.

package http

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"
)

// A Client is an HTTP client. Its zero value (DefaultClient) is a usable client
// that uses DefaultTransport.
//
// The Client's Transport typically has internal state (cached
// TCP connections), so Clients should be reused instead of created as
// needed. Clients are safe for concurrent use by multiple goroutines.
//
// Client is not yet very configurable.
type Client struct {
	Transport RoundTripper // if nil, DefaultTransport is used

	// If CheckRedirect is not nil, the client calls it before
	// following an HTTP redirect. The arguments req and via
	// are the upcoming request and the requests made already,
	// oldest first. If CheckRedirect returns an error, the client
	// returns that error instead of issue the Request req.
	//
	// If CheckRedirect is nil, the Client uses its default policy,
	// which is to stop after 10 consecutive requests.
	CheckRedirect func(req *Request, via []*Request) error

	// Jar specifies the cookie jar. 
	// If Jar is nil, cookies are not sent in requests and ignored 
	// in responses.
	Jar CookieJar
}

// DefaultClient is the default Client and is used by Get, Head, and Post.
var DefaultClient = &Client{}

// RoundTripper is an interface representing the ability to execute a
// single HTTP transaction, obtaining the Response for a given Request.
//
// A RoundTripper must be safe for concurrent use by multiple
// goroutines.
type RoundTripper interface {
	// RoundTrip executes a single HTTP transaction, returning
	// the Response for the request req.  RoundTrip should not
	// attempt to interpret the response.  In particular,
	// RoundTrip must return err == nil if it obtained a response,
	// regardless of the response's HTTP status code.  A non-nil
	// err should be reserved for failure to obtain a response.
	// Similarly, RoundTrip should not attempt to handle
	// higher-level protocol details such as redirects,
	// authentication, or cookies.
	//
	// RoundTrip should not modify the request, except for
	// consuming the Body.  The request's URL and Header fields
	// are guaranteed to be initialized.
	RoundTrip(*Request) (*Response, error)
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

// Do sends an HTTP request and returns an HTTP response, following
// policy (e.g. redirects, cookies, auth) as configured on the client.
//
// A non-nil response always contains a non-nil resp.Body.
//
// Callers should close resp.Body when done reading from it. If
// resp.Body is not closed, the Client's underlying RoundTripper
// (typically Transport) may not be able to re-use a persistent TCP
// connection to the server for a subsequent "keep-alive" request.
//
// Generally Get, Post, or PostForm will be used instead of Do.
func (c *Client) Do(req *Request) (resp *Response, err error) {
	if req.Method == "GET" || req.Method == "HEAD" {
		return c.doFollowingRedirects(req)
	}
	return send(req, c.Transport)
}

// send issues an HTTP request.  Caller should close resp.Body when done reading from it.
func send(req *Request, t RoundTripper) (resp *Response, err error) {
	if t == nil {
		t = DefaultTransport
		if t == nil {
			err = errors.New("http: no Client.Transport or DefaultTransport")
			return
		}
	}

	if req.URL == nil {
		return nil, errors.New("http: nil Request.URL")
	}

	// Most the callers of send (Get, Post, et al) don't need
	// Headers, leaving it uninitialized.  We guarantee to the
	// Transport that this has been initialized, though.
	if req.Header == nil {
		req.Header = make(Header)
	}

	info := req.URL.RawUserinfo
	if len(info) > 0 {
		req.Header.Set("Authorization", "Basic "+base64.URLEncoding.EncodeToString([]byte(info)))
	}
	return t.RoundTrip(req)
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
// redirect codes, Get follows the redirect, up to a maximum of 10 redirects:
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
//
// Caller should close r.Body when done reading from it.
//
// Get is a wrapper around DefaultClient.Get.
func Get(url string) (r *Response, err error) {
	return DefaultClient.Get(url)
}

// Get issues a GET to the specified URL.  If the response is one of the
// following redirect codes, Get follows the redirect after calling the
// Client's CheckRedirect function.
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
//
// Caller should close r.Body when done reading from it.
func (c *Client) Get(url string) (r *Response, err error) {
	req, err := NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	return c.doFollowingRedirects(req)
}

func (c *Client) doFollowingRedirects(ireq *Request) (r *Response, err error) {
	// TODO: if/when we add cookie support, the redirected request shouldn't
	// necessarily supply the same cookies as the original.
	var base *url.URL
	redirectChecker := c.CheckRedirect
	if redirectChecker == nil {
		redirectChecker = defaultCheckRedirect
	}
	var via []*Request

	if ireq.URL == nil {
		return nil, errors.New("http: nil Request.URL")
	}

	jar := c.Jar
	if jar == nil {
		jar = blackHoleJar{}
	}

	req := ireq
	urlStr := "" // next relative or absolute URL to fetch (after first request)
	for redirect := 0; ; redirect++ {
		if redirect != 0 {
			req = new(Request)
			req.Method = ireq.Method
			req.Header = make(Header)
			req.URL, err = base.Parse(urlStr)
			if err != nil {
				break
			}
			if len(via) > 0 {
				// Add the Referer header.
				lastReq := via[len(via)-1]
				if lastReq.URL.Scheme != "https" {
					req.Header.Set("Referer", lastReq.URL.String())
				}

				err = redirectChecker(req, via)
				if err != nil {
					break
				}
			}
			for _, cookie := range jar.Cookies(req.URL) {
				req.AddCookie(cookie)
			}
		}

		urlStr = req.URL.String()
		if r, err = send(req, c.Transport); err != nil {
			break
		}
		if c := r.Cookies(); len(c) > 0 {
			jar.SetCookies(req.URL, c)
		}

		if shouldRedirect(r.StatusCode) {
			r.Body.Close()
			if urlStr = r.Header.Get("Location"); urlStr == "" {
				err = errors.New(fmt.Sprintf("%d response missing Location header", r.StatusCode))
				break
			}
			base = req.URL
			via = append(via, req)
			continue
		}
		return
	}

	method := ireq.Method
	err = &url.Error{method[0:1] + strings.ToLower(method[1:]), urlStr, err}
	return
}

func defaultCheckRedirect(req *Request, via []*Request) error {
	if len(via) >= 10 {
		return errors.New("stopped after 10 redirects")
	}
	return nil
}

// Post issues a POST to the specified URL.
//
// Caller should close r.Body when done reading from it.
//
// Post is a wrapper around DefaultClient.Post
func Post(url string, bodyType string, body io.Reader) (r *Response, err error) {
	return DefaultClient.Post(url, bodyType, body)
}

// Post issues a POST to the specified URL.
//
// Caller should close r.Body when done reading from it.
func (c *Client) Post(url string, bodyType string, body io.Reader) (r *Response, err error) {
	req, err := NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", bodyType)
	return send(req, c.Transport)
}

// PostForm issues a POST to the specified URL, 
// with data's keys and values urlencoded as the request body.
//
// Caller should close r.Body when done reading from it.
//
// PostForm is a wrapper around DefaultClient.PostForm
func PostForm(url string, data url.Values) (r *Response, err error) {
	return DefaultClient.PostForm(url, data)
}

// PostForm issues a POST to the specified URL, 
// with data's keys and values urlencoded as the request body.
//
// Caller should close r.Body when done reading from it.
func (c *Client) PostForm(url string, data url.Values) (r *Response, err error) {
	return c.Post(url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
}

// Head issues a HEAD to the specified URL.  If the response is one of the
// following redirect codes, Head follows the redirect after calling the
// Client's CheckRedirect function.
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
//
// Head is a wrapper around DefaultClient.Head
func Head(url string) (r *Response, err error) {
	return DefaultClient.Head(url)
}

// Head issues a HEAD to the specified URL.  If the response is one of the
// following redirect codes, Head follows the redirect after calling the
// Client's CheckRedirect function.
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
func (c *Client) Head(url string) (r *Response, err error) {
	req, err := NewRequest("HEAD", url, nil)
	if err != nil {
		return nil, err
	}
	return c.doFollowingRedirects(req)
}
