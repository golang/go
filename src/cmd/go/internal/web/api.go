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
	"net/url"
	"os"
	"strings"
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

// An HTTPError describes an HTTP error response (non-200 result).
type HTTPError struct {
	URL        string // redacted
	Status     string
	StatusCode int
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("reading %s: %v", e.URL, e.Status)
}

func (e *HTTPError) Is(target error) bool {
	return target == os.ErrNotExist && (e.StatusCode == 404 || e.StatusCode == 410)
}

// GetBytes returns the body of the requested resource, or an error if the
// response status was not http.StatusOK.
//
// GetBytes is a convenience wrapper around Get and Response.Err.
func GetBytes(u *url.URL) ([]byte, error) {
	resp, err := Get(DefaultSecurity, u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if err := resp.Err(); err != nil {
		return nil, err
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %v", Redacted(u), err)
	}
	return b, nil
}

type Response struct {
	URL        string // redacted
	Status     string
	StatusCode int
	Header     map[string][]string
	Body       io.ReadCloser
}

// Err returns an *HTTPError corresponding to the response r.
// It returns nil if the response r has StatusCode 200 or 0 (unset).
func (r *Response) Err() error {
	if r.StatusCode == 200 || r.StatusCode == 0 {
		return nil
	}
	return &HTTPError{URL: r.URL, Status: r.Status, StatusCode: r.StatusCode}
}

// Get returns the body of the HTTP or HTTPS resource specified at the given URL.
//
// If the URL does not include an explicit scheme, Get first tries "https".
// If the server does not respond under that scheme and the security mode is
// Insecure, Get then tries "http".
// The URL included in the response indicates which scheme was actually used,
// and it is a redacted URL suitable for use in error messages.
//
// For the "https" scheme only, credentials are attached using the
// cmd/go/internal/auth package. If the URL itself includes a username and
// password, it will not be attempted under the "http" scheme unless the
// security mode is Insecure.
//
// Get returns a non-nil error only if the request did not receive a response
// under any applicable scheme. (A non-2xx response does not cause an error.)
func Get(security SecurityMode, u *url.URL) (*Response, error) {
	return get(security, u)
}

// Redacted returns a redacted string form of the URL,
// suitable for printing in error messages.
// The string form replaces any non-empty password
// in the original URL with "[redacted]".
func Redacted(u *url.URL) string {
	if u.User != nil {
		if _, ok := u.User.Password(); ok {
			redacted := *u
			redacted.User = url.UserPassword(u.User.Username(), "[redacted]")
			u = &redacted
		}
	}
	return u.String()
}

// OpenBrowser attempts to open the requested URL in a web browser.
func OpenBrowser(url string) (opened bool) {
	return openBrowser(url)
}

// Join returns the result of adding the slash-separated
// path elements to the end of u's path.
func Join(u *url.URL, path string) *url.URL {
	j := *u
	if path == "" {
		return &j
	}
	j.Path = strings.TrimSuffix(u.Path, "/") + "/" + strings.TrimPrefix(path, "/")
	j.RawPath = strings.TrimSuffix(u.RawPath, "/") + "/" + strings.TrimPrefix(path, "/")
	return &j
}
