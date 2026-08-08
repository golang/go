// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package auth provides access to user-provided authentication credentials.
package auth

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/web/intercept"
	"cmd/internal/quoted"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
)

var (
	credentialCache        sync.Map // prefix → http.Header
	clientCertificateCache sync.Map // origin → ClientCertificate
	authOnce               sync.Once
)

// A ClientCertificate describes a certificate and private key to use for
// HTTPS requests to Origin. CertFile may contain both the certificate and key,
// in which case KeyFile is equal to CertFile.
type ClientCertificate struct {
	Origin   string
	CertFile string
	KeyFile  string
}

// ClientCertificateForRequest returns the client certificate configured for
// req's HTTPS origin, as derived from req.URL. When test hooks are enabled,
// req.Host takes precedence over req.URL.Host so that test interceptors,
// which rewrite req.URL to point at a local test server, preserve the logical
// request origin. req.Host must never influence the origin otherwise: a
// certificate must not be selected by a Host header that differs from the
// host the connection is made to.
func ClientCertificateForRequest(req *http.Request) (ClientCertificate, bool) {
	host := req.URL.Host
	if intercept.TestHooksEnabled && req.Host != "" {
		host = req.Host
	}
	origin, err := canonicalHTTPSOrigin(&url.URL{Scheme: req.URL.Scheme, Host: host})
	if err != nil {
		return ClientCertificate{}, false
	}
	cert, ok := clientCertificateCache.Load(origin)
	if !ok {
		return ClientCertificate{}, false
	}
	return cert.(ClientCertificate), true
}

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
	found := loadCredential(req, req.URL.String())
	if _, ok := ClientCertificateForRequest(req); ok {
		found = true
	}
	return found
}

// runGoAuth executes authentication commands specified by the GOAUTH
// environment variable handling 'off', 'netrc', 'git', and 'mtls' methods
// specially, and storing retrieved credentials for future access.
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
				cmdErrs = append(cmdErrs, fmt.Errorf("GOAUTH=%s: %v", command, err))
				continue
			}
			// Process lines in reverse so that if the same machine is listed
			// multiple times, we end up saving the earlier one
			// (overwriting later ones). This matches the way the go command
			// worked before GOAUTH.
			for i := len(lines) - 1; i >= 0; i-- {
				l := lines[i]
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
		case "mtls":
			words, err := quoted.Split(command)
			if err != nil {
				base.Fatalf("go: cannot parse GOAUTH=mtls command %q: %v", command, err)
			}
			cert, err := parseClientCertificate(words)
			if err != nil {
				base.Fatalf("go: GOAUTH=mtls: %v", err)
			}
			clientCertificateCache.Store(cert.Origin, cert)
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
		req, err := http.NewRequest("GET", url, nil)
		hasCredential := err == nil && loadCredential(req, url)
		if err == nil {
			_, hasClientCertificate := ClientCertificateForRequest(req)
			hasCredential = hasCredential || hasClientCertificate
		}
		if !hasCredential && len(cmdErrs) > 0 {
			log.Printf("GOAUTH encountered errors for %s:", url)
			for _, err := range cmdErrs {
				log.Printf("  %v", err)
			}
		}
	}
}

func parseClientCertificate(words []string) (ClientCertificate, error) {
	if len(words) != 3 && len(words) != 4 {
		return ClientCertificate{}, fmt.Errorf("usage: mtls https-origin cert-file [key-file]")
	}
	u, err := url.ParseRequestURI(words[1])
	if err != nil {
		return ClientCertificate{}, fmt.Errorf("invalid HTTPS origin %q: %v", words[1], err)
	}
	origin, err := canonicalHTTPSOrigin(u)
	if err != nil {
		return ClientCertificate{}, err
	}
	if !filepath.IsAbs(words[2]) {
		return ClientCertificate{}, fmt.Errorf("certificate file must be an absolute path")
	}
	keyFile := words[2]
	if len(words) == 4 {
		keyFile = words[3]
		if !filepath.IsAbs(keyFile) {
			return ClientCertificate{}, fmt.Errorf("key file must be an absolute path")
		}
	}
	return ClientCertificate{Origin: origin, CertFile: words[2], KeyFile: keyFile}, nil
}

func canonicalHTTPSOrigin(u *url.URL) (string, error) {
	if u.Scheme != "https" || u.Host == "" || u.User != nil || (u.Path != "" && u.Path != "/") || u.ForceQuery || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("origin must be an HTTPS URL without user information, a non-root path, query, or fragment")
	}
	host := strings.TrimSuffix(u.Hostname(), ".")
	if host == "" {
		return "", fmt.Errorf("HTTPS origin is missing a hostname")
	}
	for i := 0; i < len(host); i++ {
		if host[i] >= 0x80 {
			return "", fmt.Errorf("HTTPS origin hostname must use ASCII or Punycode")
		}
	}
	host = strings.ToLower(host)
	port := u.Port()
	if port == "" {
		port = "443"
	}
	return "https://" + net.JoinHostPort(host, port), nil
}

// loadCredential retrieves cached credentials for the given url and adds
// them to the request headers.
func loadCredential(req *http.Request, url string) bool {
	currentPrefix := strings.TrimPrefix(url, "https://")
	currentPrefix = strings.TrimSuffix(currentPrefix, "/")

	// Iteratively try prefixes, moving up the path hierarchy.
	// E.g. example.com/foo/bar, example.com/foo, example.com
	for {
		headers, ok := credentialCache.Load(currentPrefix)
		if !ok {
			lastSlash := strings.LastIndexByte(currentPrefix, '/')
			if lastSlash == -1 {
				return false
			}
			currentPrefix = currentPrefix[:lastSlash]
			continue
		}
		for key, values := range headers.(http.Header) {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
		return true
	}
}

// storeCredential caches or removes credentials (represented by HTTP headers)
// associated with given URL prefixes.
func storeCredential(prefix string, header http.Header) {
	// Trim "https://" prefix to match the format used in .netrc files.
	prefix = strings.TrimPrefix(prefix, "https://")
	prefix = strings.TrimSuffix(prefix, "/")
	if len(header) == 0 {
		credentialCache.Delete(prefix)
	} else {
		credentialCache.Store(prefix, header)
	}
}
