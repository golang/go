// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !golangorg

package main

import (
	"errors"
	"flag"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
)

func handleRemoteSearch() {
	// Command-line queries.
	for i := 0; i < flag.NArg(); i++ {
		res, err := remoteSearch(flag.Arg(i))
		if err != nil {
			log.Fatalf("remoteSearch: %s", err)
		}
		io.Copy(os.Stdout, res.Body)
	}
	return
}

// remoteSearchURL returns the search URL for a given query as needed by
// remoteSearch. If html is set, an html result is requested; otherwise
// the result is in textual form.
// Adjust this function as necessary if modeNames or FormValue parameters
// change.
func remoteSearchURL(query string, html bool) string {
	s := "/search?m=text&q="
	if html {
		s = "/search?q="
	}
	return s + url.QueryEscape(query)
}

func remoteSearch(query string) (res *http.Response, err error) {
	// list of addresses to try
	var addrs []string
	if *serverAddr != "" {
		// explicit server address - only try this one
		addrs = []string{*serverAddr}
	} else {
		addrs = []string{
			defaultAddr,
			"golang.org",
		}
	}

	// remote search
	search := remoteSearchURL(query, *html)
	for _, addr := range addrs {
		url := "http://" + addr + search
		res, err = http.Get(url)
		if err == nil && res.StatusCode == http.StatusOK {
			break
		}
	}

	if err == nil && res.StatusCode != http.StatusOK {
		err = errors.New(res.Status)
	}

	return
}
