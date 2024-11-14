// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"log"
	"net/http"
	"net/http/cgi"
	"os/exec"
	"runtime"
	"slices"
	"sync"
)

type gitHandler struct {
	once       sync.Once
	gitPath    string
	gitPathErr error
}

func (h *gitHandler) Available() bool {
	if runtime.GOOS == "plan9" {
		// The Git command is usually not the real Git on Plan 9.
		// See https://golang.org/issues/29640.
		return false
	}
	h.once.Do(func() {
		h.gitPath, h.gitPathErr = exec.LookPath("git")
	})
	return h.gitPathErr == nil
}

func (h *gitHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	if !h.Available() {
		return nil, ServerNotInstalledError{name: "git"}
	}

	baseEnv := append(slices.Clip(env),
		"GIT_PROJECT_ROOT="+dir,
		"GIT_HTTP_EXPORT_ALL=1",
	)

	handler := http.HandlerFunc(func { w, req ->
		// The Git client sends the requested Git protocol version as a
		// "Git-Protocol" HTTP request header, which the CGI host then converts
		// to an environment variable (HTTP_GIT_PROTOCOL).
		//
		// However, versions of Git older that 2.34.0 don't recognize the
		// HTTP_GIT_PROTOCOL variable, and instead need that value to be set in the
		// GIT_PROTOCOL variable. We do so here so that vcweb can work reliably
		// with older Git releases. (As of the time of writing, the Go project's
		// builders were on Git version 2.30.2.)
		env := slices.Clip(baseEnv)
		if p := req.Header.Get("Git-Protocol"); p != "" {
			env = append(env, "GIT_PROTOCOL="+p)
		}

		h := &cgi.Handler{
			Path:   h.gitPath,
			Logger: logger,
			Args:   []string{"http-backend"},
			Dir:    dir,
			Env:    env,
		}
		h.ServeHTTP(w, req)
	})

	return handler, nil
}
