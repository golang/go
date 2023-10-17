// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"log"
	"net/http"
)

// insecureHandler redirects requests to the same host and path but using the
// "http" scheme instead of "https".
type insecureHandler struct{}

func (h *insecureHandler) Available() bool { return true }

func (h *insecureHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	// The insecure-redirect handler implementation doesn't depend or dir or env.
	//
	// The only effect of the directory is to determine which prefix the caller
	// will strip from the request before passing it on to this handler.
	return h, nil
}

func (h *insecureHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Host == "" && req.URL.Host == "" {
		http.Error(w, "no Host provided in request", http.StatusBadRequest)
		return
	}

	// Note that if the handler is wrapped with http.StripPrefix, the prefix
	// will remain stripped in the redirected URL, preventing redirect loops
	// if the scheme is already "http".

	u := *req.URL
	u.Scheme = "http"
	u.User = nil
	u.Host = req.Host

	http.Redirect(w, req, u.String(), http.StatusFound)
}
