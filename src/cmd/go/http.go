// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cmd_go_bootstrap

// This code is compiled into the real 'go' binary, but it is not
// compiled into the binary that is built during all.bash, so as
// to avoid needing to build net (and thus use cgo) during the
// bootstrap process.

package main

import (
	"cmd/internal/browser"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"time"
)

// httpClient is the default HTTP client, but a variable so it can be
// changed by tests, without modifying http.DefaultClient.
var httpClient = http.DefaultClient

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

type httpError struct {
	status     string
	statusCode int
	url        string
}

func (e *httpError) Error() string {
	return fmt.Sprintf("%s: %s", e.url, e.status)
}

// httpGET returns the data from an HTTP GET request for the given URL.
func httpGET(url string) ([]byte, error) {
	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		err := &httpError{status: resp.Status, statusCode: resp.StatusCode, url: url}

		return nil, err
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", url, err)
	}
	return b, nil
}

// httpsOrHTTP returns the body of either the importPath's
// https resource or, if unavailable, the http resource.
func httpsOrHTTP(importPath string, security securityMode) (urlStr string, body io.ReadCloser, err error) {
	fetch := func(scheme string) (urlStr string, res *http.Response, err error) {
		u, err := url.Parse(scheme + "://" + importPath)
		if err != nil {
			return "", nil, err
		}
		u.RawQuery = "go-get=1"
		urlStr = u.String()
		if buildV {
			log.Printf("Fetching %s", urlStr)
		}
		if security == insecure && scheme == "https" { // fail earlier
			res, err = impatientInsecureHTTPClient.Get(urlStr)
		} else {
			res, err = httpClient.Get(urlStr)
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
		if buildV {
			log.Printf("https fetch failed: %v", err)
		}
		if security == insecure {
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
	if buildV {
		log.Printf("Parsing meta tags from %s (status code %d)", urlStr, res.StatusCode)
	}
	return urlStr, res.Body, nil
}

func queryEscape(s string) string { return url.QueryEscape(s) }
func openBrowser(url string) bool { return browser.Open(url) }
