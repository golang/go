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
	"mime"
	"net/http"
	urlpkg "net/url"
	"os"
	"strings"
	"time"

	"cmd/go/internal/auth"
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

func get(security SecurityMode, url *urlpkg.URL) (*Response, error) {
	start := time.Now()

	if url.Scheme == "file" {
		return getFile(url)
	}

	if os.Getenv("TESTGOPROXY404") == "1" && url.Host == "proxy.golang.org" {
		res := &Response{
			URL:        Redacted(url),
			Status:     "404 testing",
			StatusCode: 404,
			Header:     make(map[string][]string),
			Body:       http.NoBody,
		}
		if cfg.BuildX {
			fmt.Fprintf(os.Stderr, "# get %s: %v (%.3fs)\n", Redacted(url), res.Status, time.Since(start).Seconds())
		}
		return res, nil
	}

	fetch := func(url *urlpkg.URL) (*urlpkg.URL, *http.Response, error) {
		// Note: The -v build flag does not mean "print logging information",
		// despite its historical misuse for this in GOPATH-based go get.
		// We print extra logging in -x mode instead, which traces what
		// commands are executed.
		if cfg.BuildX {
			fmt.Fprintf(os.Stderr, "# get %s\n", Redacted(url))
		}

		req, err := http.NewRequest("GET", url.String(), nil)
		if err != nil {
			return nil, nil, err
		}
		if url.Scheme == "https" {
			auth.AddCredentials(req)
		}

		var res *http.Response
		if security == Insecure && url.Scheme == "https" { // fail earlier
			res, err = impatientInsecureHTTPClient.Do(req)
		} else {
			res, err = securityPreservingHTTPClient.Do(req)
		}
		return url, res, err
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

		fetched, res, err = fetch(secure)
		if err != nil {
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: %v\n", Redacted(secure), err)
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
					fmt.Fprintf(os.Stderr, "# get %s: insecure\n", Redacted(url))
				}
				return nil, fmt.Errorf("insecure URL: %s", Redacted(url))
			}
		case "":
			if security != Insecure {
				panic("should have returned after HTTPS failure")
			}
		default:
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: unsupported\n", Redacted(url))
			}
			return nil, fmt.Errorf("unsupported scheme: %s", Redacted(url))
		}

		insecure := new(urlpkg.URL)
		*insecure = *url
		insecure.Scheme = "http"
		if insecure.User != nil && security != Insecure {
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: insecure credentials\n", Redacted(insecure))
			}
			return nil, fmt.Errorf("refusing to pass credentials to insecure URL: %s", Redacted(insecure))
		}

		fetched, res, err = fetch(insecure)
		if err != nil {
			if cfg.BuildX {
				fmt.Fprintf(os.Stderr, "# get %s: %v\n", Redacted(insecure), err)
			}
			// HTTP failed, and we already tried HTTPS if applicable.
			// Report the error from the HTTP attempt.
			return nil, err
		}
	}

	// Note: accepting a non-200 OK here, so people can serve a
	// meta import in their http 404 page.
	if cfg.BuildX {
		fmt.Fprintf(os.Stderr, "# get %s: %v (%.3fs)\n", Redacted(fetched), res.Status, time.Since(start).Seconds())
	}

	r := &Response{
		URL:        Redacted(fetched),
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
			URL:        Redacted(u),
			Status:     http.StatusText(http.StatusNotFound),
			StatusCode: http.StatusNotFound,
			Body:       http.NoBody,
			fileErr:    err,
		}, nil
	}

	if os.IsPermission(err) {
		return &Response{
			URL:        Redacted(u),
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
		URL:        Redacted(u),
		Status:     http.StatusText(http.StatusOK),
		StatusCode: http.StatusOK,
		Body:       f,
	}, nil
}

func openBrowser(url string) bool { return browser.Open(url) }
