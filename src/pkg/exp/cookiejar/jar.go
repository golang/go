// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cookiejar implements an RFC 6265-compliant http.CookieJar.
//
// TODO: example code to create a memory-backed cookie jar with the default
// public suffix list.
package cookiejar

import (
	"net/http"
	"net/url"
)

// PublicSuffixList provides the public suffix of a domain. For example:
//      - the public suffix of "example.com" is "com",
//      - the public suffix of "foo1.foo2.foo3.co.uk" is "co.uk", and
//      - the public suffix of "bar.pvt.k12.wy.us" is "pvt.k12.wy.us".
//
// Implementations of PublicSuffixList must be safe for concurrent use by
// multiple goroutines.
//
// An implementation that always returns "" is valid and may be useful for
// testing but it is not secure: it means that the HTTP server for foo.com can
// set a cookie for bar.com.
type PublicSuffixList interface {
	// PublicSuffix returns the public suffix of domain.
	//
	// TODO: specify which of the caller and callee is responsible for IP
	// addresses, for leading and trailing dots, for case sensitivity, and
	// for IDN/Punycode.
	PublicSuffix(domain string) string

	// String returns a description of the source of this public suffix list.
	// A Jar will store its PublicSuffixList's description in its storage,
	// and update the stored cookies if its list has a different description
	// than the stored list. The description will typically contain something
	// like a time stamp or version number.
	String() string
}

// Options are the options for creating a new Jar.
type Options struct {
	// Storage is the cookie jar storage. It may not be nil.
	Storage Storage

	// PublicSuffixList is the public suffix list that determines whether an
	// HTTP server can set a cookie for a domain. It may not be nil.
	PublicSuffixList PublicSuffixList

	// TODO: ErrorFunc for handling storage errors?
}

// Jar implements the http.CookieJar interface from the net/http package.
type Jar struct {
	storage Storage
	psList  PublicSuffixList
}

// New returns a new cookie jar.
func New(o *Options) *Jar {
	return &Jar{
		storage: o.Storage,
		psList:  o.PublicSuffixList,
	}
}

// TODO(nigeltao): how do we reject HttpOnly cookies? Do we post-process the
// return value from Jar.Cookies?
//
// HttpOnly cookies are those for regular HTTP(S) requests but should not be
// visible from JavaScript. The HttpOnly bit mitigates XSS attacks; it's not
// for HTTP vs HTTPS vs FTP transports.

// Cookies implements the Cookies method of the http.CookieJar interface.
//
// It returns an empty slice if the URL's scheme is not HTTP or HTTPS.
func (j *Jar) Cookies(u *url.URL) []*http.Cookie {
	// TODO.
	return nil
}

// SetCookies implements the SetCookies method of the http.CookieJar interface.
//
// It does nothing if the URL's scheme is not HTTP or HTTPS.
func (j *Jar) SetCookies(u *url.URL, cookies []*http.Cookie) {
	// TODO.
}
