// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fetch provides an extensible mechanism to fetch a profile
// from a data source.
package fetch

import (
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"cmd/pprof/internal/plugin"
	"internal/pprof/profile"
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
		return nil, fmt.Errorf("http fetch: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, statusCodeError(resp)
	}

	return resp.Body, nil
}

// PostURL issues a POST to a URL over HTTP.
func PostURL(source, post string) ([]byte, error) {
	resp, err := http.Post(source, "application/octet-stream", strings.NewReader(post))
	if err != nil {
		return nil, fmt.Errorf("http post %s: %v", source, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, statusCodeError(resp)
	}
	return ioutil.ReadAll(resp.Body)
}

func statusCodeError(resp *http.Response) error {
	if resp.Header.Get("X-Go-Pprof") != "" && strings.Contains(resp.Header.Get("Content-Type"), "text/plain") {
		// error is from pprof endpoint
		body, err := ioutil.ReadAll(resp.Body)
		if err == nil {
			return fmt.Errorf("server response: %s - %s", resp.Status, body)
		}
	}
	return fmt.Errorf("server response: %s", resp.Status)
}

// httpGet is a wrapper around http.Get; it is defined as a variable
// so it can be redefined during for testing.
var httpGet = func(source string, timeout time.Duration) (*http.Response, error) {
	url, err := url.Parse(source)
	if err != nil {
		return nil, err
	}

	var tlsConfig *tls.Config
	if url.Scheme == "https+insecure" {
		tlsConfig = &tls.Config{
			InsecureSkipVerify: true,
		}
		url.Scheme = "https"
		source = url.String()
	}

	client := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: timeout + 5*time.Second,
			TLSClientConfig:       tlsConfig,
		},
	}
	return client.Get(source)
}
