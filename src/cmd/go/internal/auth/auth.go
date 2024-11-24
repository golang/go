// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package auth provides access to user-provided authentication credentials.
package auth

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"fmt"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
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
// res is used for the custom GOAUTH command's stdin.
func AddCredentials(client *http.Client, req *http.Request, res *http.Response, url string) bool {
	if req.URL.Scheme != "https" {
		panic("GOAUTH called without https")
	}
	if cfg.GOAUTH == "off" {
		return false
	}
	// Run all GOAUTH commands at least once.
	authOnce.Do(func() {
		runGoAuth(client, res, "")
	})
	if url != "" {
		// First fetch must have failed; re-invoke GOAUTH commands with url.
		runGoAuth(client, res, url)
	}
	return loadCredential(req, req.URL.String())
}

// runGoAuth executes authentication commands specified by the GOAUTH
// environment variable handling 'off', 'netrc', and 'git' methods specially,
// and storing retrieved credentials for future access.
func runGoAuth(client *http.Client, res *http.Response, url string) {
	var cmdErrs []error // store GOAUTH command errors to log later.
	goAuthCmds := strings.Split(cfg.GOAUTH, ";")
	// The GOAUTH commands are processed in reverse order to prioritize
	// credentials in the order they were specified.
	slices.Reverse(goAuthCmds)
	for _, command := range goAuthCmds {
		command = strings.TrimSpace(command)
		words := strings.Fields(command)
		if len(words) == 0 {
			base.Fatalf("go: GOAUTH encountered an empty command (GOAUTH=%s)", cfg.GOAUTH)
		}
		switch words[0] {
		case "off":
			if len(goAuthCmds) != 1 {
				base.Fatalf("go: GOAUTH=off cannot be combined with other authentication commands (GOAUTH=%s)", cfg.GOAUTH)
			}
			return
		case "netrc":
			lines, err := readNetrc()
			if err != nil {
				base.Fatalf("go: could not parse netrc (GOAUTH=%s): %v", cfg.GOAUTH, err)
			}
			for _, l := range lines {
				r := http.Request{Header: make(http.Header)}
				r.SetBasicAuth(l.login, l.password)
				storeCredential(l.machine, r.Header)
			}
		case "git":
			if len(words) != 2 {
				base.Fatalf("go: GOAUTH=git dir method requires an absolute path to the git working directory")
			}
			dir := words[1]
			if !filepath.IsAbs(dir) {
				base.Fatalf("go: GOAUTH=git dir method requires an absolute path to the git working directory, dir is not absolute")
			}
			fs, err := os.Stat(dir)
			if err != nil {
				base.Fatalf("go: GOAUTH=git encountered an error; cannot stat %s: %v", dir, err)
			}
			if !fs.IsDir() {
				base.Fatalf("go: GOAUTH=git dir method requires an absolute path to the git working directory, dir is not a directory")
			}

			if url == "" {
				// Skip the initial GOAUTH run since we need to provide an
				// explicit url to runGitAuth.
				continue
			}
			prefix, header, err := runGitAuth(client, dir, url)
			if err != nil {
				// Save the error, but don't print it yet in case another
				// GOAUTH command might succeed.
				cmdErrs = append(cmdErrs, fmt.Errorf("GOAUTH=%s: %v", command, err))
			} else {
				storeCredential(prefix, header)
			}
		default:
			credentials, err := runAuthCommand(command, url, res)
			if err != nil {
				// Save the error, but don't print it yet in case another
				// GOAUTH command might succeed.
				cmdErrs = append(cmdErrs, fmt.Errorf("GOAUTH=%s: %v", command, err))
				continue
			}
			for prefix := range credentials {
				storeCredential(prefix, credentials[prefix])
			}
		}
	}
	// If no GOAUTH command provided a credential for the given url
	// and an error occurred, log the error.
	if cfg.BuildX && url != "" {
		if ok := loadCredential(&http.Request{}, url); !ok && len(cmdErrs) > 0 {
			log.Printf("GOAUTH encountered errors for %s:", url)
			for _, err := range cmdErrs {
				log.Printf("  %v", err)
			}
		}
	}
}

// loadCredential retrieves cached credentials for the given url and adds
// them to the request headers.
func loadCredential(req *http.Request, url string) bool {
	currentPrefix := strings.TrimPrefix(url, "https://")
	// Iteratively try prefixes, moving up the path hierarchy.
	for currentPrefix != "/" && currentPrefix != "." && currentPrefix != "" {
		headers, ok := credentialCache.Load(currentPrefix)
		if !ok {
			// Move to the parent directory.
			currentPrefix = path.Dir(currentPrefix)
			continue
		}
		for key, values := range headers.(http.Header) {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
		return true
	}
	return false
}

// storeCredential caches or removes credentials (represented by HTTP headers)
// associated with given URL prefixes.
func storeCredential(prefix string, header http.Header) {
	// Trim "https://" prefix to match the format used in .netrc files.
	prefix = strings.TrimPrefix(prefix, "https://")
	if len(header) == 0 {
		credentialCache.Delete(prefix)
	} else {
		credentialCache.Store(prefix, header)
	}
}
