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
	"io/ioutil"
	"log"
	"net/url"
	"strings"
	"sync"
	"time"
)

// A Client is an HTTP client. Its zero value (DefaultClient) is a
// usable client that uses DefaultTransport.
//
// The Client's Transport typically has internal state (cached TCP
// connections), so Clients should be reused instead of created as
// needed. Clients are safe for concurrent use by multiple goroutines.
//
// A Client is higher-level than a RoundTripper (such as Transport)
// and additionally handles HTTP details such as cookies and
// redirects.
type Client struct {
	// Transport specifies the mechanism by which individual
	// HTTP requests are made.
	// If nil, DefaultTransport is used.
	Transport RoundTripper

	// CheckRedirect specifies the policy for handling redirects.
	// If CheckRedirect is not nil, the client calls it before
	// following an HTTP redirect. The arguments req and via are
	// the upcoming request and the requests made already, oldest
	// first. If CheckRedirect returns an error, the Client's Get
	// method returns both the previous Response and
	// CheckRedirect's error (wrapped in a url.Error) instead of
	// issuing the Request req.
	//
	// If CheckRedirect is nil, the Client uses its default policy,
	// which is to stop after 10 consecutive requests.
	CheckRedirect func(req *Request, via []*Request) error

	// Jar specifies the cookie jar.
	// If Jar is nil, cookies are not sent in requests and ignored
	// in responses.
	Jar CookieJar

	// Timeout specifies a time limit for requests made by this
	// Client. The timeout includes connection time, any
	// redirects, and reading the response body. The timer remains
	// running after Get, Head, Post, or Do return and will
	// interrupt reading of the Response.Body.
	//
	// A Timeout of zero means no timeout.
	//
	// The Client's Transport must support the CancelRequest
	// method or Client will return errors when attempting to make
	// a request with Get, Head, Post, or Do. Client's default
	// Transport (DefaultTransport) supports CancelRequest.
	Timeout time.Duration
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
	// consuming and closing the Body, including on errors. The
	// request's URL and Header fields are guaranteed to be
	// initialized.
	RoundTrip(*Request) (*Response, error)
}

// Given a string of the form "host", "host:port", or "[ipv6::address]:port",
// return true if the string includes a port.
func hasPort(s string) bool { return strings.LastIndex(s, ":") > strings.LastIndex(s, "]") }

// refererForURL returns a referer without any authentication info or
// an empty string if lastReq scheme is https and newReq scheme is http.
func refererForURL(lastReq, newReq *url.URL) string {
	// https://tools.ietf.org/html/rfc7231#section-5.5.2
	//   "Clients SHOULD NOT include a Referer header field in a
	//    (non-secure) HTTP request if the referring page was
	//    transferred with a secure protocol."
	if lastReq.Scheme == "https" && newReq.Scheme == "http" {
		return ""
	}
	referer := lastReq.String()
	if lastReq.User != nil {
		// This is not very efficient, but is the best we can
		// do without:
		// - introducing a new method on URL
		// - creating a race condition
		// - copying the URL struct manually, which would cause
		//   maintenance problems down the line
		auth := lastReq.User.String() + "@"
		referer = strings.Replace(referer, auth, "", 1)
	}
	return referer
}

// Used in Send to implement io.ReadCloser by bundling together the
// bufio.Reader through which we read the response, and the underlying
// network connection.
type readClose struct {
	io.Reader
	io.Closer
}

func (c *Client) send(req *Request) (*Response, error) {
	if c.Jar != nil {
		for _, cookie := range c.Jar.Cookies(req.URL) {
			req.AddCookie(cookie)
		}
	}
	resp, err := send(req, c.transport())
	if err != nil {
		return nil, err
	}
	if c.Jar != nil {
		if rc := resp.Cookies(); len(rc) > 0 {
			c.Jar.SetCookies(req.URL, rc)
		}
	}
	return resp, err
}

// Do sends an HTTP request and returns an HTTP response, following
// policy (e.g. redirects, cookies, auth) as configured on the client.
//
// An error is returned if caused by client policy (such as
// CheckRedirect), or if there was an HTTP protocol error.
// A non-2xx response doesn't cause an error.
//
// When err is nil, resp always contains a non-nil resp.Body.
//
// Callers should close resp.Body when done reading from it. If
// resp.Body is not closed, the Client's underlying RoundTripper
// (typically Transport) may not be able to re-use a persistent TCP
// connection to the server for a subsequent "keep-alive" request.
//
// The request Body, if non-nil, will be closed by the underlying
// Transport, even on errors.
//
// Generally Get, Post, or PostForm will be used instead of Do.
func (c *Client) Do(req *Request) (resp *Response, err error) {
	if req.Method == "GET" || req.Method == "HEAD" {
		return c.doFollowingRedirects(req, shouldRedirectGet)
	}
	if req.Method == "POST" || req.Method == "PUT" {
		return c.doFollowingRedirects(req, shouldRedirectPost)
	}
	return c.send(req)
}

func (c *Client) transport() RoundTripper {
	if c.Transport != nil {
		return c.Transport
	}
	return DefaultTransport
}

// send issues an HTTP request.
// Caller should close resp.Body when done reading from it.
func send(req *Request, t RoundTripper) (resp *Response, err error) {
	if t == nil {
		req.closeBody()
		return nil, errors.New("http: no Client.Transport or DefaultTransport")
	}

	if req.URL == nil {
		req.closeBody()
		return nil, errors.New("http: nil Request.URL")
	}

	if req.RequestURI != "" {
		req.closeBody()
		return nil, errors.New("http: Request.RequestURI can't be set in client requests.")
	}

	// Most the callers of send (Get, Post, et al) don't need
	// Headers, leaving it uninitialized.  We guarantee to the
	// Transport that this has been initialized, though.
	if req.Header == nil {
		req.Header = make(Header)
	}

	if u := req.URL.User; u != nil {
		username := u.Username()
		password, _ := u.Password()
		req.Header.Set("Authorization", "Basic "+basicAuth(username, password))
	}
	resp, err = t.RoundTrip(req)
	if err != nil {
		if resp != nil {
			log.Printf("RoundTripper returned a response & error; ignoring response")
		}
		return nil, err
	}
	return resp, nil
}

// See 2 (end of page 4) http://www.ietf.org/rfc/rfc2617.txt
// "To receive authorization, the client sends the userid and password,
// separated by a single colon (":") character, within a base64
// encoded string in the credentials."
// It is not meant to be urlencoded.
func basicAuth(username, password string) string {
	auth := username + ":" + password
	return base64.StdEncoding.EncodeToString([]byte(auth))
}

// True if the specified HTTP status code is one for which the Get utility should
// automatically redirect.
func shouldRedirectGet(statusCode int) bool {
	switch statusCode {
	case StatusMovedPermanently, StatusFound, StatusSeeOther, StatusTemporaryRedirect:
		return true
	}
	return false
}

// True if the specified HTTP status code is one for which the Post utility should
// automatically redirect.
func shouldRedirectPost(statusCode int) bool {
	switch statusCode {
	case StatusFound, StatusSeeOther:
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
// An error is returned if there were too many redirects or if there
// was an HTTP protocol error. A non-2xx response doesn't cause an
// error.
//
// When err is nil, resp always contains a non-nil resp.Body.
// Caller should close resp.Body when done reading from it.
//
// Get is a wrapper around DefaultClient.Get.
func Get(url string) (resp *Response, err error) {
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
// An error is returned if the Client's CheckRedirect function fails
// or if there was an HTTP protocol error. A non-2xx response doesn't
// cause an error.
//
// When err is nil, resp always contains a non-nil resp.Body.
// Caller should close resp.Body when done reading from it.
func (c *Client) Get(url string) (resp *Response, err error) {
	req, err := NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	return c.doFollowingRedirects(req, shouldRedirectGet)
}

func (c *Client) doFollowingRedirects(ireq *Request, shouldRedirect func(int) bool) (resp *Response, err error) {
	var base *url.URL
	redirectChecker := c.CheckRedirect
	if redirectChecker == nil {
		redirectChecker = defaultCheckRedirect
	}
	var via []*Request

	if ireq.URL == nil {
		ireq.closeBody()
		return nil, errors.New("http: nil Request.URL")
	}

	var reqmu sync.Mutex // guards req
	req := ireq

	var timer *time.Timer
	if c.Timeout > 0 {
		type canceler interface {
			CancelRequest(*Request)
		}
		tr, ok := c.transport().(canceler)
		if !ok {
			return nil, fmt.Errorf("net/http: Client Transport of type %T doesn't support CancelRequest; Timeout not supported", c.transport())
		}
		timer = time.AfterFunc(c.Timeout, func() {
			reqmu.Lock()
			defer reqmu.Unlock()
			tr.CancelRequest(req)
		})
	}

	urlStr := "" // next relative or absolute URL to fetch (after first request)
	redirectFailed := false
	for redirect := 0; ; redirect++ {
		if redirect != 0 {
			nreq := new(Request)
			nreq.Method = ireq.Method
			if ireq.Method == "POST" || ireq.Method == "PUT" {
				nreq.Method = "GET"
			}
			nreq.Header = make(Header)
			nreq.URL, err = base.Parse(urlStr)
			if err != nil {
				break
			}
			if len(via) > 0 {
				// Add the Referer header.
				lastReq := via[len(via)-1]
				if ref := refererForURL(lastReq.URL, nreq.URL); ref != "" {
					nreq.Header.Set("Referer", ref)
				}

				err = redirectChecker(nreq, via)
				if err != nil {
					redirectFailed = true
					break
				}
			}
			reqmu.Lock()
			req = nreq
			reqmu.Unlock()
		}

		urlStr = req.URL.String()
		if resp, err = c.send(req); err != nil {
			break
		}

		if shouldRedirect(resp.StatusCode) {
			// Read the body if small so underlying TCP connection will be re-used.
			// No need to check for errors: if it fails, Transport won't reuse it anyway.
			const maxBodySlurpSize = 2 << 10
			if resp.ContentLength == -1 || resp.ContentLength <= maxBodySlurpSize {
				io.CopyN(ioutil.Discard, resp.Body, maxBodySlurpSize)
			}
			resp.Body.Close()
			if urlStr = resp.Header.Get("Location"); urlStr == "" {
				err = errors.New(fmt.Sprintf("%d response missing Location header", resp.StatusCode))
				break
			}
			base = req.URL
			via = append(via, req)
			continue
		}
		if timer != nil {
			resp.Body = &cancelTimerBody{timer, resp.Body}
		}
		return resp, nil
	}

	method := ireq.Method
	urlErr := &url.Error{
		Op:  method[0:1] + strings.ToLower(method[1:]),
		URL: urlStr,
		Err: err,
	}

	if redirectFailed {
		// Special case for Go 1 compatibility: return both the response
		// and an error if the CheckRedirect function failed.
		// See http://golang.org/issue/3795
		return resp, urlErr
	}

	if resp != nil {
		resp.Body.Close()
	}
	return nil, urlErr
}

func defaultCheckRedirect(req *Request, via []*Request) error {
	if len(via) >= 10 {
		return errors.New("stopped after 10 redirects")
	}
	return nil
}

// Post issues a POST to the specified URL.
//
// Caller should close resp.Body when done reading from it.
//
// Post is a wrapper around DefaultClient.Post
func Post(url string, bodyType string, body io.Reader) (resp *Response, err error) {
	return DefaultClient.Post(url, bodyType, body)
}

// Post issues a POST to the specified URL.
//
// Caller should close resp.Body when done reading from it.
//
// If the provided body is also an io.Closer, it is closed after the
// request.
func (c *Client) Post(url string, bodyType string, body io.Reader) (resp *Response, err error) {
	req, err := NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", bodyType)
	return c.doFollowingRedirects(req, shouldRedirectPost)
}

// PostForm issues a POST to the specified URL, with data's keys and
// values URL-encoded as the request body.
//
// When err is nil, resp always contains a non-nil resp.Body.
// Caller should close resp.Body when done reading from it.
//
// PostForm is a wrapper around DefaultClient.PostForm
func PostForm(url string, data url.Values) (resp *Response, err error) {
	return DefaultClient.PostForm(url, data)
}

// PostForm issues a POST to the specified URL,
// with data's keys and values urlencoded as the request body.
//
// When err is nil, resp always contains a non-nil resp.Body.
// Caller should close resp.Body when done reading from it.
func (c *Client) PostForm(url string, data url.Values) (resp *Response, err error) {
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
func Head(url string) (resp *Response, err error) {
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
func (c *Client) Head(url string) (resp *Response, err error) {
	req, err := NewRequest("HEAD", url, nil)
	if err != nil {
		return nil, err
	}
	return c.doFollowingRedirects(req, shouldRedirectGet)
}

type cancelTimerBody struct {
	t  *time.Timer
	rc io.ReadCloser
}

func (b *cancelTimerBody) Read(p []byte) (n int, err error) {
	n, err = b.rc.Read(p)
	if err == io.EOF {
		b.t.Stop()
	}
	return
}

func (b *cancelTimerBody) Close() error {
	err := b.rc.Close()
	b.t.Stop()
	return err
}
