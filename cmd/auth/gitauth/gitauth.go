// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gitauth uses 'git credential' to implement the GOAUTH protocol described in
// https://golang.org/issue/26232. It expects an absolute path to the working
// directory for the 'git' command as the first command-line argument.
//
// Example GOAUTH usage:
// 	export GOAUTH="gitauth $HOME"
//
// See https://git-scm.com/docs/gitcredentials or run 'man gitcredentials' for
// information on how to configure 'git credential'.
package main

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	if len(os.Args) < 2 || !filepath.IsAbs(os.Args[1]) {
		fmt.Fprintf(os.Stderr, "usage: %s WORKDIR [URL]", os.Args[0])
		os.Exit(2)
	}

	log.SetPrefix("gitauth: ")

	if len(os.Args) != 3 {
		// No explicit URL was passed on the command line, but 'git credential'
		// provides no way to enumerate existing credentials.
		// Wait for a request for a specific URL.
		return
	}

	u, err := url.ParseRequestURI(os.Args[2])
	if err != nil {
		log.Fatalf("invalid request URI (%v): %q\n", err, os.Args[1])
	}

	var (
		prefix     *url.URL
		lastHeader http.Header
		lastStatus = http.StatusUnauthorized
	)
	for lastStatus == http.StatusUnauthorized {
		cmd := exec.Command("git", "credential", "fill")

		// We don't want to execute a 'git' command in an arbitrary directory, since
		// that opens up a number of config-injection attacks (for example,
		// https://golang.org/issue/29230). Instead, we have the user configure a
		// directory explicitly on the command line.
		cmd.Dir = os.Args[1]

		cmd.Stdin = strings.NewReader(fmt.Sprintf("url=%s\n", u))
		cmd.Stderr = os.Stderr
		out, err := cmd.Output()
		if err != nil {
			log.Fatalf("'git credential fill' failed: %v\n", err)
		}

		prefix = new(url.URL)
		var username, password string
		lines := strings.Split(string(out), "\n")
		for _, line := range lines {
			frags := strings.SplitN(line, "=", 2)
			if len(frags) != 2 {
				continue // Ignore unrecognized response lines.
			}
			switch strings.TrimSpace(frags[0]) {
			case "protocol":
				prefix.Scheme = frags[1]
			case "host":
				prefix.Host = frags[1]
			case "path":
				prefix.Path = frags[1]
			case "username":
				username = frags[1]
			case "password":
				password = frags[1]
			case "url":
				// Write to a local variable instead of updating prefix directly:
				// if the url field is malformed, we don't want to invalidate
				// information parsed from the protocol, host, and path fields.
				u, err := url.ParseRequestURI(frags[1])
				if err == nil {
					prefix = u
				} else {
					log.Printf("malformed URL from 'git credential fill' (%v): %q\n", err, frags[1])
					// Proceed anyway: we might be able to parse the prefix from other fields of the response.
				}
			}
		}

		// Double-check that the URL Git gave us is a prefix of the one we requested.
		if !strings.HasPrefix(u.String(), prefix.String()) {
			log.Fatalf("requested a credential for %q, but 'git credential fill' provided one for %q\n", u, prefix)
		}

		// Send a HEAD request to try to detect whether the credential is valid.
		// If the user just typed in a correct password and has caching enabled,
		// we don't want to nag them for it again the next time they run a 'go' command.
		req, err := http.NewRequest("HEAD", u.String(), nil)
		if err != nil {
			log.Fatalf("internal error constructing HTTP HEAD request: %v\n", err)
		}
		req.SetBasicAuth(username, password)
		lastHeader = req.Header
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			log.Printf("HTTPS HEAD request failed to connect: %v\n", err)
			// Couldn't verify the credential, but we have no evidence that it is invalid either.
			// Proceed, but don't update git's credential cache.
			break
		}
		lastStatus = resp.StatusCode

		if resp.StatusCode != http.StatusOK {
			log.Printf("%s: %v %s\n", u, resp.StatusCode, http.StatusText(resp.StatusCode))
		}

		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusUnauthorized {
			// We learned something about the credential: it either worked or it was invalid.
			// Approve or reject the credential (on a best-effort basis)
			// so that the git credential helper can update its cache as appropriate.
			action := "approve"
			if resp.StatusCode != http.StatusOK {
				action = "reject"
			}
			cmd = exec.Command("git", "credential", action)
			cmd.Stderr = os.Stderr
			cmd.Stdout = os.Stderr
			cmd.Stdin = bytes.NewReader(out)
			_ = cmd.Run()
		}
	}

	// Write out the credential in the format expected by the 'go' command.
	fmt.Printf("%s\n\n", prefix)
	lastHeader.Write(os.Stdout)
	fmt.Println()
}
