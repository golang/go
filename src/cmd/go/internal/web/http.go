// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap

// This code is compiled into the real 'go' binary, but it is not
// compiled into the binary that is built during all.bash, so as
// to avoid needing to build net (and thus use cgo) during the
// bootstrap process.

package web

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	urlpkg "net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/auth"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/web/intercept"
	"cmd/internal/browser"
)

const userAgent = "GoCommand/1 (+https://go.dev/cmd/go)"

// impatientInsecureHTTPClient is used with GOINSECURE,
// when we're connecting to https servers that might not be there
// or might be using self-signed certificates.
var impatientInsecureHTTPClient = &http.Client{
	CheckRedirect: checkRedirect,
	Timeout:       5 * time.Second,
	Transport: &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	},
}

var securityPreservingDefaultClient = securityPreservingHTTPClient(http.DefaultClient)

// securityPreservingHTTPClient returns a client that is like the original
// but rejects redirects to plain-HTTP URLs if the original URL was secure.
func securityPreservingHTTPClient(original *http.Client) *http.Client {
	c := new(http.Client)
	*c = *original
	c.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) > 0 && via[0].URL.Scheme == "https" && req.URL.Scheme != "https" {
			lastHop := via[len(via)-1].URL
			return fmt.Errorf("redirected from secure URL %s to insecure URL %s", lastHop, req.URL)
		}
		return checkRedirect(req, via)
	}
	return c
}

func checkRedirect(req *http.Request, via []*http.Request) error {
	// Go's http.DefaultClient allows 10 redirects before returning an error.
	// Mimic that behavior here.
	if len(via) >= 10 {
		return errors.New("stopped after 10 redirects")
	}
	hasGoGet1 := via[len(via)-1].URL.Query().Get("go-get") == "1"
	if hasGoGet1 {
		if len(req.URL.RawQuery) > 0 {
			req.URL.RawQuery += "&"
		}
		req.URL.RawQuery += "go-get=1"
	}

	intercept.Request(req)
	return nil
}

func get(security SecurityMode, url *urlpkg.URL) (*Response, error) {
	start := time.Now()

	if url.Scheme == "file" {
		return getFile(url)
	}

	if intercept.TestHooksEnabled {
		switch url.Host {
		case "localhost.localdev":
			return nil, fmt.Errorf("no such host localhost.localdev")

		default:
			if os.Getenv("TESTGONETWORK") == "panic" {
				if _, ok := intercept.URL(url); !ok {
					host := url.Host
					if h, _, err := net.SplitHostPort(url.Host); err == nil && h != "" {
						host = h
					}
					addr := net.ParseIP(host)
					if addr == nil || (!addr.IsLoopback() && !addr.IsUnspecified()) {
						panic("use of network: " + url.String())
					}
				}
			}
		}
	}

	var (
		fetched *urlpkg.URL
		res     *http.Response
		err     error
	)
	if url.Scheme == "" || url.Scheme == "https" {
		secure := new(urlpkg.URL)
		*secure = *url
		secure.Scheme = "https"

		res, err = fetch(security, secure, 0, "")
		if err == nil {
			fetched = secure
		} else {
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: %v\n", secure.Redacted(), err)
			}
			if security != Insecure || url.Scheme == "https" {
				// HTTPS failed, and we can't fall back to plain HTTP.
				// Report the error from the HTTPS attempt.
				return nil, err
			}
		}
	}

	if res == nil {
		switch url.Scheme {
		case "http":
			if security == SecureOnly {
				if cfg.BuildX {
					fmt.Fprintf(os.Stderr, "# get %s: insecure\n", url.Redacted())
				}
				return nil, fmt.Errorf("insecure URL: %s", url.Redacted())
			}
		case "":
			if security != Insecure {
				panic("should have returned after HTTPS failure")
			}
		default:
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: unsupported\n", url.Redacted())
			}
			return nil, fmt.Errorf("unsupported scheme: %s", url.Redacted())
		}

		insecure := new(urlpkg.URL)
		*insecure = *url
		insecure.Scheme = "http"
		if insecure.User != nil && security != Insecure {
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: insecure credentials\n", insecure.Redacted())
			}
			return nil, fmt.Errorf("refusing to pass credentials to insecure URL: %s", insecure.Redacted())
		}

		res, err = fetch(security, insecure, 0, "")
		if err == nil {
			fetched = insecure
		} else {
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: %v\n", insecure.Redacted(), err)
			}
			// HTTP failed, and we already tried HTTPS if applicable.
			// Report the error from the HTTP attempt.
			return nil, err
		}
	}

	// Note: accepting a non-200 OK here, so people can serve a
	// meta import in their http 404 page.
	if cfg.BuildX {
		fmt.Fprintf(os.Stderr, "# get %s: %v (%.3fs)\n", fetched.Redacted(), res.Status, time.Since(start).Seconds())
	}

	r := &Response{
		URL:        fetched.Redacted(),
		Status:     res.Status,
		StatusCode: res.StatusCode,
		Header:     map[string][]string(res.Header),
		Body:       res.Body,
	}

	switch res.StatusCode {
	case http.StatusOK:
		r.Body = newRetryBody(security, fetched, res)
	default:
		contentType := res.Header.Get("Content-Type")
		if mediaType, params, _ := mime.ParseMediaType(contentType); mediaType == "text/plain" {
			switch charset := strings.ToLower(params["charset"]); charset {
			case "us-ascii", "utf-8", "":
				// Body claims to be plain text in UTF-8 or a subset thereof.
				// Try to extract a useful error message from it.
				r.errorDetail.r = res.Body
				r.Body = &r.errorDetail
			}
		}
	}

	return r, nil
}

func fetch(security SecurityMode, url *urlpkg.URL, offset int64, ifRange string) (*http.Response, error) {
	// Note: The -v build flag does not mean "print logging information",
	// despite its historical misuse for this in GOPATH-based go get.
	// We print extra logging in -x mode instead, which traces what
	// commands are executed.
	if cfg.BuildX {
		if offset == 0 {
			fmt.Fprintf(os.Stderr, "# get %s\n", url.Redacted())
		} else {
			fmt.Fprintf(os.Stderr, "# get %s (offset %d)\n", url.Redacted(), offset)
		}
	}

	req, err := http.NewRequest("GET", url.String(), nil)
	if err != nil {
		return nil, err
	}
	t, intercepted := intercept.URL(req.URL)
	var client *http.Client
	if security == Insecure && url.Scheme == "https" {
		client = impatientInsecureHTTPClient
	} else if intercepted && t.Client != nil {
		client = securityPreservingHTTPClient(t.Client)
	} else {
		client = securityPreservingDefaultClient
	}
	if url.Scheme == "https" {
		// Use initial GOAUTH credentials.
		auth.AddCredentials(client, req, nil, "")
	}
	if intercepted {
		req.Host = req.URL.Host
		req.URL.Host = t.ToHost
	}
	req.Header.Set("User-Agent", userAgent)

	setRangeHeaders := func(req *http.Request) {
		if offset <= 0 {
			return
		}
		// Make a conditional range request: The server will only respond with
		// 206 Partial Content if the resource hasn't changed since our last GET.
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		req.Header.Set("If-Range", ifRange)
	}
	setRangeHeaders(req)

	release, err := base.AcquireNet()
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil && release != nil {
			release()
		}
	}()
	res, err := client.Do(req)
	// If the initial request fails with a 4xx client error and the
	// response body didn't satisfy the request
	// (e.g. a valid <meta name="go-import"> tag),
	// retry the request with credentials obtained by invoking GOAUTH
	// with the request URL.
	if url.Scheme == "https" && err == nil && res.StatusCode >= 400 && res.StatusCode < 500 {
		// Close the body of the previous response since we
		// are discarding it and creating a new one.
		res.Body.Close()
		req, err = http.NewRequest("GET", url.String(), nil)
		if err != nil {
			return nil, err
		}
		auth.AddCredentials(client, req, res, url.String())
		if intercepted {
			intercept.Request(req)
		}
		setRangeHeaders(req)
		res, err = client.Do(req)
	}

	if err != nil {
		// Per the docs for [net/http.Client.Do], “On error, any Response can be
		// ignored. A non-nil Response with a non-nil error only occurs when
		// CheckRedirect fails, and even then the returned Response.Body is
		// already closed.”
		return nil, err
	}

	// “If the returned error is nil, the Response will contain a non-nil Body
	// which the user is expected to close.”
	body := res.Body
	res.Body = hookCloser{
		ReadCloser: body,
		afterClose: release,
	}
	return res, nil
}

type retryBody struct {
	security SecurityMode
	url      *urlpkg.URL
	ifRange  string

	mu          sync.Mutex
	closed      bool
	restarts    int
	startOffset int64
	offset      int64
	size        int64
	err         error
	lastReadErr error
	body        io.ReadCloser
}

func newRetryBody(security SecurityMode, u *urlpkg.URL, res *http.Response) io.ReadCloser {
	if res.Uncompressed {
		// The http Transport automatically added an Accept-Encoding: gzip header,
		// and the server responded with Content-Encoding: gzip. We can't resume
		// a broken download, because we don't know the correct content offset to
		// resume at--a Range request will specify a location in the compressed
		// content, and res.Body contains the uncompressed content.
		//
		// We can avoid this case by setting Transport.DisableCompression,
		// but that requires modifying the Transport and the module proxy never
		// responds with Content-Encoding: gzip anyway. Check here just in case,
		// but this should never happen.
		//
		// (If we implement #81200, we can disable automatic decompression
		// on a per-request basis and should do that instead.)
		return res.Body
	}

	if res.ContentLength < 0 {
		// The server didn't send us a Content-Length header.
		// It probably can't handle Range requests if it doesn't know
		// the size of the files it serves.
		return res.Body
	}

	// We need a strong ETag or Last-Modified header to retry with a Range request.
	// If we don't have one, just use the original body.
	ifRange := res.Header.Get("ETag")
	if !isStrongETag(ifRange) {
		ifRange = res.Header.Get("Last-Modified")
	}
	if ifRange == "" {
		return res.Body
	}

	return &retryBody{
		// We could use res.Request.URL for the retry URL,
		// which would avoid re-following redirects.
		// On the other hand, following redirects might send us to a healthier destination.
		// Probably doesn't actually matter either way in practice.
		url: u,

		security: security,
		ifRange:  ifRange,
		size:     res.ContentLength,
		body:     res.Body,
	}
}

func (b *retryBody) Close() error {
	// This is the same mutex Read takes, so Close can't interrupt a Read.
	// Not a problem in our current usage.
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true

	// If we hit an error prior to reading to EOF, always return an error from Close.
	// Prefer the error from b.body.Close, if we have an open body and closing it fails.
	var closeErr error
	if b.err != nil && b.err != io.EOF {
		closeErr = fmt.Errorf("fetch error: %v", b.err)
	}
	if b.body != nil {
		if err := b.body.Close(); err != nil {
			closeErr = err
		}
		b.body = nil
	}
	return closeErr
}

func (b *retryBody) Read(p []byte) (n int, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return 0, errors.New("read from closed body")
	}
	if b.err != nil {
		return 0, b.err
	}

	for {
		if b.body == nil {
			// The previous read returned an error,
			// and we want to try resuming the download with a Range request.
			if err := b.resume(); err != nil {
				if cfg.BuildX {
					fmt.Fprintf(os.Stderr, "# get %s: resume: %v\n", b.url.Redacted(), err)
				}
				b.err = b.lastReadErr
				return n, b.lastReadErr
			}
		}

		n, err = b.body.Read(p)
		if n > 0 {
			b.offset += int64(n)
			if b.size >= 0 && b.offset > b.size {
				// Can't retry this (we're past the end of the expected body).
				b.err = fmt.Errorf("read %v bytes from %v-byte response",
					b.offset, b.size)
				return 0, b.err
			}
		}
		if err == io.EOF && b.size >= 0 && b.offset != b.size {
			err = io.ErrUnexpectedEOF
		}
		b.err = err

		const maxRestarts = 2 // 3 total: 1 initial + 2 restarts
		switch {
		case err == nil || err == io.EOF:
			return n, err // success
		case err != nil && b.offset == b.size:
			return n, io.EOF // non-EOF error at the exact end of file, call it EOF
		case b.offset == b.startOffset:
			return n, err // no progress
		case b.restarts >= maxRestarts:
			return n, err // too many restarts
		}

		// We've hit an error while downloading,
		// and we can try to resume.
		if cfg.BuildX {
			fmt.Fprintf(os.Stderr, "# get %s: interrupted\n", b.url.Redacted())
		}
		b.lastReadErr = err
		b.err = nil
		b.body.Close()
		b.body = nil
		b.startOffset = b.offset
		b.restarts++
		if n != 0 {
			// We did get some data, so return it before resuming.
			return n, nil
		}
	}
}

func (b *retryBody) resume() error {
	res, err := fetch(b.security, b.url, b.offset, b.ifRange)
	if err != nil {
		return err
	}
	defer func() {
		if res.Body != nil {
			res.Body.Close()
		}
	}()
	if res.StatusCode != http.StatusPartialContent {
		// We could handle a 200 response (which resends the entire response)
		// and skip to where we left off. Don't bother for now.
		return fmt.Errorf("non-206 response code: %v", res.StatusCode)
	}
	cr := res.Header.Get("Content-Range")
	first, _, size, ok := parseContentRange(cr)
	if !ok || first != b.offset || (b.size != -1 && size != -1 && b.size != size) {
		return fmt.Errorf("invalid Content-Range: %q", cr)
	}
	b.body = res.Body
	res.Body = nil
	return nil
}

func isStrongETag(s string) bool {
	return s != "" && !strings.HasPrefix(s, "W/")
}

// parseContentRange parses a Content-Range header.
func parseContentRange(s string) (first, last, total int64, ok bool) {
	// "bytes NNNN-NNNN/NNNN"
	s = strings.ToLower(strings.TrimSpace(s))
	s, ok = strings.CutPrefix(s, "bytes ")
	if !ok {
		return 0, 0, 0, false
	}
	s = strings.TrimSpace(s)
	// "NNNN-NNNN/NNNN"
	first, s, ok = cutInt63(s, "-")
	if !ok {
		return 0, 0, 0, false
	}
	// "NNNN/NNNN"
	last, s, ok = cutInt63(s, "/")
	if !ok || first > last {
		return 0, 0, 0, false
	}
	// "NNNN"
	if s == "*" {
		return first, last, -1, true // "bytes NNNN-NNNN/*"
	}
	total, err := parseInt63(s)
	if err != nil {
		return 0, 0, 0, false
	}
	return first, last, total, true
}

func cutInt63(s, sep string) (int64, string, bool) {
	part, rest, ok := strings.Cut(s, sep)
	if !ok {
		return 0, rest, false
	}
	n, err := parseInt63(part)
	if err != nil {
		return 0, rest, false
	}
	return n, rest, true
}

func parseInt63(s string) (int64, error) {
	n, err := strconv.ParseUint(s, 10, 63)
	if err != nil {
		return 0, err
	}
	return int64(n), err
}

func getFile(u *urlpkg.URL) (*Response, error) {
	path, err := urlToFilePath(u)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)

	if os.IsNotExist(err) {
		return &Response{
			URL:        u.Redacted(),
			Status:     http.StatusText(http.StatusNotFound),
			StatusCode: http.StatusNotFound,
			Body:       http.NoBody,
			fileErr:    err,
		}, nil
	}

	if os.IsPermission(err) {
		return &Response{
			URL:        u.Redacted(),
			Status:     http.StatusText(http.StatusForbidden),
			StatusCode: http.StatusForbidden,
			Body:       http.NoBody,
			fileErr:    err,
		}, nil
	}

	if err != nil {
		return nil, err
	}

	return &Response{
		URL:        u.Redacted(),
		Status:     http.StatusText(http.StatusOK),
		StatusCode: http.StatusOK,
		Body:       f,
	}, nil
}

func openBrowser(url string) bool { return browser.Open(url) }

func isLocalHost(u *urlpkg.URL) bool {
	// VCSTestRepoURL itself is secure, and it may redirect requests to other
	// ports (such as a port serving the "svn" protocol) which should also be
	// considered secure.
	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		host = u.Host
	}
	if host == "localhost" {
		return true
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return true
	}
	return false
}

type hookCloser struct {
	io.ReadCloser
	afterClose func()
}

func (c hookCloser) Close() error {
	err := c.ReadCloser.Close()
	c.afterClose()
	return err
}
