// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regression tests to run against a production instance of godoc.

package main_test

import (
	"bytes"
	"flag"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"
)

var host = flag.String("regtest.host", "", "host to run regression test against")

func init() {
	flag.Parse()
	*host = strings.TrimSuffix(*host, "/")
}

func TestLiveServer(t *testing.T) {
	if *host == "" {
		t.Skip("regtest.host flag missing.")
	}
	substringTests := []struct {
		Message     string
		Path        string
		Substring   string
		Regexp      string
		NoAnalytics bool // expect the response to not contain GA.
		PostBody    string
	}{
		{
			Path:      "/doc/faq",
			Substring: "What is the purpose of the project",
		},
		{
			Path:      "/pkg/",
			Substring: "Package tar",
		},
		{
			Path:      "/pkg/os/",
			Substring: "func Open",
		},
		{
			Path:      "/pkg/net/http/",
			Substring: `title="Added in Go 1.11"`,
			Message:   "version information not present - failed InitVersionInfo?",
		},
		{
			Path:        "/robots.txt",
			Substring:   "Disallow: /search",
			Message:     "robots not present - not deployed from Dockerfile?",
			NoAnalytics: true,
		},
		{
			Path:        "/change/75944e2e3a63",
			Substring:   "bdb10cf",
			Message:     "no change redirect - hg to git mapping not registered?",
			NoAnalytics: true,
		},
		{
			Path:      "/dl/",
			Substring: "go1.11.windows-amd64.msi",
			Message:   "missing data on dl page - misconfiguration of datastore?",
		},
		{
			Path:        "/dl/?mode=json",
			Substring:   ".windows-amd64.msi",
			NoAnalytics: true,
		},
		{
			Message:     "broken shortlinks - misconfiguration of datastore or memcache?",
			Path:        "/s/go2design",
			Regexp:      "proposal.*Found",
			NoAnalytics: true,
		},
		{
			Message:   "incorrect search result - broken index?",
			Path:      "/search?q=IsDir",
			Substring: "src/os/types.go",
		},
		{
			Path:        "/compile",
			PostBody:    "body=" + url.QueryEscape("package main; func main() { print(6*7); }"),
			Regexp:      `^{"compile_errors":"","output":"42"}$`,
			NoAnalytics: true,
		},
		{
			Path:        "/compile",
			PostBody:    "body=" + url.QueryEscape("//empty"),
			Substring:   "expected 'package', found 'EOF'",
			NoAnalytics: true,
		},
		{
			Path:        "/compile",
			PostBody:    "version=2&body=package+main%3Bimport+(%22fmt%22%3B%22time%22)%3Bfunc+main()%7Bfmt.Print(%22A%22)%3Btime.Sleep(time.Second)%3Bfmt.Print(%22B%22)%7D",
			Regexp:      `^{"Errors":"","Events":\[{"Message":"A","Kind":"stdout","Delay":0},{"Message":"B","Kind":"stdout","Delay":1000000000}\]}$`,
			NoAnalytics: true,
		},
	}

	for _, tc := range substringTests {
		t.Run(tc.Path, func(t *testing.T) {
			method := "GET"
			var reqBody io.Reader
			if tc.PostBody != "" {
				method = "POST"
				reqBody = strings.NewReader(tc.PostBody)
			}
			req, err := http.NewRequest(method, *host+tc.Path, reqBody)
			if err != nil {
				t.Fatalf("NewRequest: %v", err)
			}
			if reqBody != nil {
				req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			}
			resp, err := http.DefaultTransport.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip: %v", err)
			}
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("ReadAll: %v", err)
			}

			const googleAnalyticsID = "UA-11222381-2" // golang.org analytics ID
			if !tc.NoAnalytics && !bytes.Contains(body, []byte(googleAnalyticsID)) {
				t.Errorf("want response to contain analytics tracking ID")
			}

			if tc.Substring != "" {
				tc.Regexp = regexp.QuoteMeta(tc.Substring)
			}
			re := regexp.MustCompile(tc.Regexp)

			if !re.Match(body) {
				t.Log("------ actual output -------")
				t.Log(string(body))
				t.Log("----------------------------")
				if tc.Message != "" {
					t.Log(tc.Message)
				}
				t.Fatalf("wanted response to match %s", tc.Regexp)
			}
		})
	}
}
