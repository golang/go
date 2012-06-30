// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cmd_go_bootstrap

// This code is compiled into the real 'go' binary, but it is not
// compiled into the binary that is built during all.bash, so as
// to avoid needing to build net (and thus use cgo) during the
// bootstrap process.

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
)

// httpClient is the default HTTP client, but a variable so it can be
// changed by tests, without modifying http.DefaultClient.
var httpClient = http.DefaultClient

// httpGET returns the data from an HTTP GET request for the given URL.
func httpGET(url string) ([]byte, error) {
	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%s: %s", url, resp.Status)
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", url, err)
	}
	return b, nil
}

// httpsOrHTTP returns the body of either the importPath's
// https resource or, if unavailable, the http resource.
func httpsOrHTTP(importPath string) (urlStr string, body io.ReadCloser, err error) {
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
		res, err = httpClient.Get(urlStr)
		return
	}
	closeBody := func(res *http.Response) {
		if res != nil {
			res.Body.Close()
		}
	}
	urlStr, res, err := fetch("https")
	if err != nil || res.StatusCode != 200 {
		if buildV {
			if err != nil {
				log.Printf("https fetch failed.")
			} else {
				log.Printf("ignoring https fetch with status code %d", res.StatusCode)
			}
		}
		closeBody(res)
		urlStr, res, err = fetch("http")
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
