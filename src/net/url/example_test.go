// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package url_test

import (
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
)

func ExampleValues() {
	v := url.Values{}
	v.Set("name", "Ava")
	v.Add("friend", "Jess")
	v.Add("friend", "Sarah")
	v.Add("friend", "Zoe")
	// v.Encode() == "name=Ava&friend=Jess&friend=Sarah&friend=Zoe"
	fmt.Println(v.Get("name"))
	fmt.Println(v.Get("friend"))
	fmt.Println(v["friend"])
	// Output:
	// Ava
	// Jess
	// [Jess Sarah Zoe]
}

func ExampleURL() {
	u, err := url.Parse("http://bing.com/search?q=dotnet")
	if err != nil {
		log.Fatal(err)
	}
	u.Scheme = "https"
	u.Host = "google.com"
	q := u.Query()
	q.Set("q", "golang")
	u.RawQuery = q.Encode()
	fmt.Println(u)
	// Output: https://google.com/search?q=golang
}

func ExampleURL_roundtrip() {
	// Parse + String preserve the original encoding.
	u, err := url.Parse("https://example.com/foo%2fbar")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.Path)
	fmt.Println(u.RawPath)
	fmt.Println(u.String())
	// Output:
	// /foo/bar
	// /foo%2fbar
	// https://example.com/foo%2fbar
}

func ExampleURL_opaque() {
	// Sending a literal '%' in an HTTP request's Path
	req := &http.Request{
		Method: "GET",
		Host:   "example.com", // takes precedence over URL.Host
		URL: &url.URL{
			Host:   "ignored",
			Scheme: "https",
			Opaque: "/%2f/",
		},
		Header: http.Header{
			"User-Agent": {"godoc-example/0.1"},
		},
	}
	out, err := httputil.DumpRequestOut(req, true)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(strings.Replace(string(out), "\r", "", -1))
	// Output:
	// GET /%2f/ HTTP/1.1
	// Host: example.com
	// User-Agent: godoc-example/0.1
	// Accept-Encoding: gzip
	//
}

func ExampleURL_ResolveReference() {
	u, err := url.Parse("../../..//search?q=dotnet")
	if err != nil {
		log.Fatal(err)
	}
	base, err := url.Parse("http://example.com/directory/")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(base.ResolveReference(u))
	// Output:
	// http://example.com/search?q=dotnet
}
