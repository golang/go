// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fetch provides an extensible mechanism to fetch a profile
// from a data source.
package fetch

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"cmd/internal/pprof/plugin"
	"cmd/internal/pprof/profile"
)

// FetchProfile reads from a data source (network, file) and generates a
// profile.
func FetchProfile(source string, timeout time.Duration) (*profile.Profile, error) {
	return Fetcher(source, timeout, plugin.StandardUI())
}

// Fetcher is the plugin.Fetcher version of FetchProfile.
func Fetcher(source string, timeout time.Duration, ui plugin.UI) (*profile.Profile, error) {
	var f io.ReadCloser
	var err error

	url, err := url.Parse(source)
	if err == nil && url.Host != "" {
		f, err = FetchURL(source, timeout)
	} else {
		f, err = os.Open(source)
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return profile.Parse(f)
}

// FetchURL fetches a profile from a URL using HTTP.
func FetchURL(source string, timeout time.Duration) (io.ReadCloser, error) {
	resp, err := httpGet(source, timeout)
	if err != nil {
		return nil, fmt.Errorf("http fetch %s: %v", source, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server response: %s", resp.Status)
	}

	return resp.Body, nil
}

// PostURL issues a POST to a URL over HTTP.
func PostURL(source, post string) ([]byte, error) {
	resp, err := http.Post(source, "application/octet-stream", strings.NewReader(post))
	if err != nil {
		return nil, fmt.Errorf("http post %s: %v", source, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server response: %s", resp.Status)
	}
	defer resp.Body.Close()
	return ioutil.ReadAll(resp.Body)
}

// httpGet is a wrapper around http.Get; it is defined as a variable
// so it can be redefined during for testing.
var httpGet = func(url string, timeout time.Duration) (*http.Response, error) {
	client := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: timeout + 5*time.Second,
		},
	}
	return client.Get(url)
}
