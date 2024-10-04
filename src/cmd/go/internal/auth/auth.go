// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package auth provides access to user-provided authentication credentials.
package auth

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"net/http"
	"path"
	"slices"
	"strings"
	"sync"
)

var (
	credentialCache sync.Map // prefix → http.Header
	authOnce        sync.Once
)

// AddCredentials populates the request header with the user's credentials
// as specified by the GOAUTH environment variable.
// It returns whether any matching credentials were found.
// req must use HTTPS or this function will panic.
func AddCredentials(req *http.Request) bool {
	if req.URL.Scheme != "https" {
		panic("GOAUTH called without https")
	}
	if cfg.GOAUTH == "off" {
		return false
	}
	authOnce.Do(runGoAuth)
	currentPrefix := strings.TrimPrefix(req.URL.String(), "https://")
	// Iteratively try prefixes, moving up the path hierarchy.
	for currentPrefix != "/" && currentPrefix != "." && currentPrefix != "" {
		if loadCredential(req, currentPrefix) {
			return true
		}

		// Move to the parent directory.
		currentPrefix = path.Dir(currentPrefix)
	}
	return false
}

// runGoAuth executes authentication commands specified by the GOAUTH
// environment variable handling 'off', 'netrc', and 'git' methods specially,
// and storing retrieved credentials for future access.
func runGoAuth() {
	// The GOAUTH commands are processed in reverse order to prioritize
	// credentials in the order they were specified.
	goAuthCmds := strings.Split(cfg.GOAUTH, ";")
	slices.Reverse(goAuthCmds)
	for _, cmdStr := range goAuthCmds {
		cmdStr = strings.TrimSpace(cmdStr)
		switch {
		case cmdStr == "off":
			if len(goAuthCmds) != 1 {
				base.Fatalf("GOAUTH=off cannot be combined with other authentication commands (GOAUTH=%s)", cfg.GOAUTH)
			}
			return
		case cmdStr == "netrc":
			lines, err := readNetrc()
			if err != nil {
				base.Fatalf("could not parse netrc (GOAUTH=%s): %v", cfg.GOAUTH, err)
			}
			for _, l := range lines {
				r := http.Request{Header: make(http.Header)}
				r.SetBasicAuth(l.login, l.password)
				storeCredential([]string{l.machine}, r.Header)
			}
		case strings.HasPrefix(cmdStr, "git"):
			base.Fatalf("unimplemented: %s", cmdStr)
		default:
			base.Fatalf("unimplemented: %s", cmdStr)
		}
	}
}

// loadCredential retrieves cached credentials for the given url prefix and adds
// them to the request headers.
func loadCredential(req *http.Request, prefix string) bool {
	headers, ok := credentialCache.Load(prefix)
	if !ok {
		return false
	}
	for key, values := range headers.(http.Header) {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	return true
}

// storeCredential caches or removes credentials (represented by HTTP headers)
// associated with given URL prefixes.
func storeCredential(prefixes []string, header http.Header) {
	for _, prefix := range prefixes {
		if len(header) == 0 {
			credentialCache.Delete(prefix)
		} else {
			credentialCache.Store(prefix, header)
		}
	}
}
