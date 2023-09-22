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
	"strings"
	"time"

	"cmd/go/internal/auth"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/internal/browser"
)

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

	interceptRequest(req)
	return nil
}

type Interceptor struct {
	Scheme   string
	FromHost string
	ToHost   string
	Client   *http.Client
}

func EnableTestHooks(interceptors []Interceptor) error {
	if enableTestHooks {
		return errors.New("web: test hooks already enabled")
	}

	for _, t := range interceptors {
		if t.FromHost == "" {
			panic("EnableTestHooks: missing FromHost")
		}
		if t.ToHost == "" {
			panic("EnableTestHooks: missing ToHost")
		}
	}

	testInterceptors = interceptors
	enableTestHooks = true
	return nil
}

func DisableTestHooks() {
	if !enableTestHooks {
		panic("web: test hooks not enabled")
	}
	enableTestHooks = false
	testInterceptors = nil
}

var (
	enableTestHooks  = false
	testInterceptors []Interceptor
)

func interceptURL(u *urlpkg.URL) (*Interceptor, bool) {
	if !enableTestHooks {
		return nil, false
	}
	for i, t := range testInterceptors {
		if u.Host == t.FromHost && (u.Scheme == "" || u.Scheme == t.Scheme) {
			return &testInterceptors[i], true
		}
	}
	return nil, false
}

func interceptRequest(req *http.Request) {
	if t, ok := interceptURL(req.URL); ok {
		req.Host = req.URL.Host
		req.URL.Host = t.ToHost
	}
}

func get(security SecurityMode, url *urlpkg.URL) (*Response, error) {
	start := time.Now()

	if url.Scheme == "file" {
		return getFile(url)
	}

	if enableTestHooks {
		switch url.Host {
		case "proxy.golang.org":
			if os.Getenv("TESTGOPROXY404") == "1" {
				res := &Response{
					URL:        url.Redacted(),
					Status:     "404 testing",
					StatusCode: 404,
					Header:     make(map[string][]string),
					Body:       http.NoBody,
				}
				if cfg.BuildX {
					fmt.Fprintf(os.Stderr, "# get %s: %v (%.3fs)\n", url.Redacted(), res.Status, time.Since(start).Seconds())
				}
				return res, nil
			}

		case "localhost.localdev":
			return nil, fmt.Errorf("no such host localhost.localdev")

		default:
			if os.Getenv("TESTGONETWORK") == "panic" {
				if _, ok := interceptURL(url); !ok {
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

	fetch := func(url *urlpkg.URL) (*http.Response, error) {
		// Note: The -v build flag does not mean "print logging information",
		// despite its historical misuse for this in GOPATH-based go get.
		// We print extra logging in -x mode instead, which traces what
		// commands are executed.
		if cfg.BuildX {
			fmt.Fprintf(os.Stderr, "# get %s\n", url.Redacted())
		}

		req, err := http.NewRequest("GET", url.String(), nil)
		if err != nil {
			return nil, err
		}
		if url.Scheme == "https" {
			auth.AddCredentials(req)
		}
		t, intercepted := interceptURL(req.URL)
		if intercepted {
			req.Host = req.URL.Host
			req.URL.Host = t.ToHost
		}

		release, err := base.AcquireNet()
		if err != nil {
			return nil, err
		}

		var res *http.Response
		if security == Insecure && url.Scheme == "https" { // fail earlier
			res, err = impatientInsecureHTTPClient.Do(req)
		} else {
			if intercepted && t.Client != nil {
				client := securityPreservingHTTPClient(t.Client)
				res, err = client.Do(req)
			} else {
				res, err = securityPreservingDefaultClient.Do(req)
			}
		}

		if err != nil {
			// Per the docs for [net/http.Client.Do], “On error, any Response can be
			// ignored. A non-nil Response with a non-nil error only occurs when
			// CheckRedirect fails, and even then the returned Response.Body is
			// already closed.”
			release()
			return nil, err
		}

		// “If the returned error is nil, the Response will contain a non-nil Body
		// which the user is expected to close.”
		body := res.Body
		res.Body = hookCloser{
			ReadCloser: body,
			afterClose: release,
		}
		return res, err
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

		res, err = fetch(secure)
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

		res, err = fetch(insecure)
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

	if res.StatusCode != http.StatusOK {
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
