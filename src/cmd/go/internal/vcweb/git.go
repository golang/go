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

	handler := &cgi.Handler{
		Path:   h.gitPath,
		Logger: logger,
		Args:   []string{"http-backend"},
		Dir:    dir,
		Env: append(slices.Clip(env),
			"GIT_PROJECT_ROOT="+dir,
			"GIT_HTTP_EXPORT_ALL=1",
		),
	}

	return handler, nil
}
