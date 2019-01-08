// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cmd_go_bootstrap

// This code is compiled into the real 'go' binary, but it is not
// compiled into the binary that is built during all.bash, so as
// to avoid needing to build net (and thus use cgo) during the
// bootstrap process.

package web

import (
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"time"

	"cmd/go/internal/cfg"
	"cmd/internal/browser"
)

// impatientInsecureHTTPClient is used in -insecure mode,
// when we're connecting to https servers that might not be there
// or might be using self-signed certificates.
var impatientInsecureHTTPClient = &http.Client{
	Timeout: 5 * time.Second,
	Transport: &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	},
}

// securityPreservingHTTPClient is like the default HTTP client, but rejects
// redirects to plain-HTTP URLs if the original URL was secure.
var securityPreservingHTTPClient = &http.Client{
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) > 0 && via[0].URL.Scheme == "https" && req.URL.Scheme != "https" {
			lastHop := via[len(via)-1].URL
			return fmt.Errorf("redirected from secure URL %s to insecure URL %s", lastHop, req.URL)
		}
		return nil
	},
}

type HTTPError struct {
	status     string
	StatusCode int
	url        string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("%s: %s", e.url, e.status)
}

// Get returns the data from an HTTP GET request for the given URL.
func Get(url string) ([]byte, error) {
	resp, err := securityPreservingHTTPClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		err := &HTTPError{status: resp.Status, StatusCode: resp.StatusCode, url: url}

		return nil, err
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", url, err)
	}
	return b, nil
}

// GetMaybeInsecure returns the body of either the importPath's
// https resource or, if unavailable and permitted by the security mode, the http resource.
func GetMaybeInsecure(importPath string, security SecurityMode) (urlStr string, body io.ReadCloser, err error) {
	fetch := func(scheme string) (urlStr string, res *http.Response, err error) {
		u, err := url.Parse(scheme + "://" + importPath)
		if err != nil {
			return "", nil, err
		}
		u.RawQuery = "go-get=1"
		urlStr = u.String()
		if cfg.BuildV {
			log.Printf("Fetching %s", urlStr)
		}
		if security == Insecure && scheme == "https" { // fail earlier
			res, err = impatientInsecureHTTPClient.Get(urlStr)
		} else {
			res, err = securityPreservingHTTPClient.Get(urlStr)
		}
		return
	}
	closeBody := func(res *http.Response) {
		if res != nil {
			res.Body.Close()
		}
	}
	urlStr, res, err := fetch("https")
	if err != nil {
		if cfg.BuildV {
			log.Printf("https fetch failed: %v", err)
		}
		if security == Insecure {
			closeBody(res)
			urlStr, res, err = fetch("http")
		}
	}
	if err != nil {
		closeBody(res)
		return "", nil, err
	}
	// Note: accepting a non-200 OK here, so people can serve a
	// meta import in their http 404 page.
	if cfg.BuildV {
		log.Printf("Parsing meta tags from %s (status code %d)", urlStr, res.StatusCode)
	}
	return urlStr, res.Body, nil
}

func QueryEscape(s string) string { return url.QueryEscape(s) }
func OpenBrowser(url string) bool { return browser.Open(url) }
