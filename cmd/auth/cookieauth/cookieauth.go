// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cookieauth uses a “Netscape cookie file” to implement the GOAUTH protocol
// described in https://golang.org/issue/26232.
// It expects the location of the file as the first command-line argument.
//
// Example GOAUTH usage:
// 	export GOAUTH="cookieauth $(git config --get http.cookieFile)"
//
// See http://www.cookiecentral.com/faq/#3.5 for a description of the Netscape
// cookie file format.
package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s COOKIEFILE [URL]\n", os.Args[0])
		os.Exit(2)
	}

	log.SetPrefix("cookieauth: ")

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatalf("failed to read cookie file: %v\n", os.Args[1])
		os.Exit(1)
	}
	defer f.Close()

	var (
		targetURL  *url.URL
		targetURLs = map[string]*url.URL{}
	)
	if len(os.Args) == 3 {
		targetURL, err = url.ParseRequestURI(os.Args[2])
		if err != nil {
			log.Fatalf("invalid request URI (%v): %q\n", err, os.Args[2])
		}
		targetURLs[targetURL.String()] = targetURL
	} else if len(os.Args) > 3 {
		// Extra arguments were passed: maybe the protocol was expanded?
		// We don't know how to interpret the request, so ignore it.
		return
	}

	entries, err := parseCookieFile(f.Name(), f)
	if err != nil {
		log.Fatalf("error reading cookie file: %v\n", f.Name())
	}

	jar, err := cookiejar.New(nil)
	if err != nil {
		log.Fatalf("failed to initialize cookie jar: %v\n", err)
	}

	for _, e := range entries {
		u := &url.URL{
			Scheme: "https",
			Host:   e.Host,
			Path:   e.Cookie.Path,
		}

		if targetURL == nil {
			targetURLs[u.String()] = u
		}

		jar.SetCookies(u, []*http.Cookie{&e.Cookie})
	}

	for _, u := range targetURLs {
		req := &http.Request{URL: u, Header: make(http.Header)}
		for _, c := range jar.Cookies(req.URL) {
			req.AddCookie(c)
		}
		fmt.Printf("%s\n\n", u)
		req.Header.Write(os.Stdout)
		fmt.Println()
	}
}

type Entry struct {
	Host   string
	Cookie http.Cookie
}

// parseCookieFile parses a Netscape cookie file as described in
// http://www.cookiecentral.com/faq/#3.5.
func parseCookieFile(name string, r io.Reader) ([]*Entry, error) {
	var entries []*Entry
	s := bufio.NewScanner(r)
	line := 0
	for s.Scan() {
		line++
		text := strings.TrimSpace(s.Text())
		if len(text) < 2 || (text[0] == '#' && unicode.IsSpace(rune(text[1]))) {
			continue
		}

		e, err := parseCookieLine(text)
		if err != nil {
			log.Printf("%s:%d: %v\n", name, line, err)
			continue
		}
		entries = append(entries, e)
	}
	return entries, s.Err()
}

func parseCookieLine(line string) (*Entry, error) {
	f := strings.Fields(line)
	if len(f) < 7 {
		return nil, fmt.Errorf("found %d columns; want 7", len(f))
	}

	e := new(Entry)
	c := &e.Cookie

	if domain := f[0]; strings.HasPrefix(domain, "#HttpOnly_") {
		c.HttpOnly = true
		e.Host = strings.TrimPrefix(domain[10:], ".")
	} else {
		e.Host = strings.TrimPrefix(domain, ".")
	}

	isDomain, err := strconv.ParseBool(f[1])
	if err != nil {
		return nil, fmt.Errorf("non-boolean domain flag: %v", err)
	}
	if isDomain {
		c.Domain = e.Host
	}

	c.Path = f[2]

	c.Secure, err = strconv.ParseBool(f[3])
	if err != nil {
		return nil, fmt.Errorf("non-boolean secure flag: %v", err)
	}

	expiration, err := strconv.ParseInt(f[4], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("malformed expiration: %v", err)
	}
	c.Expires = time.Unix(expiration, 0)

	c.Name = f[5]
	c.Value = f[6]

	return e, nil
}
