// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command server serves get.golang.org, redirecting users to the appropriate
// getgo installer based on the request path.
package server

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

const (
	base             = "https://storage.googleapis.com/golang/getgo/"
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

func init() {
	http.HandleFunc("/", handler)
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
