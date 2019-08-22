// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command server serves get.golang.org, redirecting users to the appropriate
// getgo installer based on the request path.
package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	base             = "https://dl.google.com/go/getgo/"
	windowsInstaller = base + "installer.exe"
	linuxInstaller   = base + "installer_linux"
	macInstaller     = base + "installer_darwin"
)

// substring-based redirects.
var stringMatch = map[string]string{
	// via uname, from bash
	"MINGW":  windowsInstaller, // Reported as MINGW64_NT-10.0 in git bash
	"Linux":  linuxInstaller,
	"Darwin": macInstaller,
}

func main() {
	http.HandleFunc("/", handler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
		fmt.Printf("Defaulting to port %s", port)
	}

	fmt.Printf("Listening on port %s", port)
	if err := http.ListenAndServe(fmt.Sprintf(":%s", port), nil); err != nil {
		fmt.Fprintf(os.Stderr, "http.ListenAndServe: %v", err)
	}
}

func handler(w http.ResponseWriter, r *http.Request) {
	if containsIgnoreCase(r.URL.Path, "installer.exe") {
		// cache bust
		http.Redirect(w, r, windowsInstaller+cacheBust(), http.StatusFound)
		return
	}

	for match, redirect := range stringMatch {
		if containsIgnoreCase(r.URL.Path, match) {
			http.Redirect(w, r, redirect, http.StatusFound)
			return
		}
	}

	http.NotFound(w, r)
}

func containsIgnoreCase(s, substr string) bool {
	return strings.Contains(
		strings.ToLower(s),
		strings.ToLower(substr),
	)
}

func cacheBust() string {
	return fmt.Sprintf("?%d", time.Now().Nanosecond())
}
