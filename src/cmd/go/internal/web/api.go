// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package web defines minimal helper routines for accessing HTTP/HTTPS
// resources without requiring external dependenicies on the net package.
//
// If the cmd_go_bootstrap build tag is present, web avoids the use of the net
// package and returns errors for all network operations.
package web

import (
	"fmt"
	"io"
	"io/ioutil"
	urlpkg "net/url"
)

// SecurityMode specifies whether a function should make network
// calls using insecure transports (eg, plain text HTTP).
// The zero value is "secure".
type SecurityMode int

const (
	SecureOnly      SecurityMode = iota // Reject plain HTTP; validate HTTPS.
	DefaultSecurity                     // Allow plain HTTP if explicit; validate HTTPS.
	Insecure                            // Allow plain HTTP if not explicitly HTTPS; skip HTTPS validation.
)

type HTTPError struct {
	status     string
	StatusCode int
	url        *urlpkg.URL
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("%s: %s", e.url, e.status)
}

// GetBytes returns the body of the requested resource, or an error if the
// response status was not http.StatusOk.
//
// GetBytes is a convenience wrapper around Get.
func GetBytes(url *urlpkg.URL) ([]byte, error) {
	url, resp, err := Get(DefaultSecurity, url)
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

type Response struct {
	Status     string
	StatusCode int
	Header     map[string][]string
	Body       io.ReadCloser
}

// Get returns the body of the HTTP or HTTPS resource specified at the given URL.
//
// If the URL does not include an explicit scheme, Get first tries "https".
// If the server does not respond under that scheme and the security mode is
// Insecure, Get then tries "http".
// The returned URL indicates which scheme was actually used.
//
// For the "https" scheme only, credentials are attached using the
// cmd/go/internal/auth package. If the URL itself includes a username and
// password, it will not be attempted under the "http" scheme unless the
// security mode is Insecure.
//
// Get returns a non-nil error only if the request did not receive a response
// under any applicable scheme. (A non-2xx response does not cause an error.)
func Get(security SecurityMode, url *urlpkg.URL) (*urlpkg.URL, *Response, error) {
	return get(security, url)
}

// PasswordRedacted returns url directly if it does not encode a password,
// or else a copy of url with the password redacted.
func PasswordRedacted(url *urlpkg.URL) *urlpkg.URL {
	if url.User != nil {
		if _, ok := url.User.Password(); ok {
			redacted := *url
			redacted.User = urlpkg.UserPassword(url.User.Username(), "[redacted]")
			return &redacted
		}
	}
	return url
}

// OpenBrowser attempts to open the requested URL in a web browser.
func OpenBrowser(url string) (opened bool) {
	return openBrowser(url)
}
