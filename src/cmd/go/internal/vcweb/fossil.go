// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"fmt"
	"log"
	"net/http"
	"net/http/cgi"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

type fossilHandler struct {
	once          sync.Once
	fossilPath    string
	fossilPathErr error
}

func (h *fossilHandler) Available() bool {
	h.once.Do(func() {
		h.fossilPath, h.fossilPathErr = exec.LookPath("fossil")
	})
	return h.fossilPathErr == nil
}

func (h *fossilHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	if !h.Available() {
		return nil, ServerNotInstalledError{name: "fossil"}
	}

	name := filepath.Base(dir)
	db := filepath.Join(dir, name+".fossil")

	cgiPath := db + ".cgi"
	cgiScript := fmt.Sprintf("#!%s\nrepository: %s\n", h.fossilPath, db)
	if err := os.WriteFile(cgiPath, []byte(cgiScript), 0755); err != nil {
		return nil, err
	}

	handler := http.HandlerFunc(func { w, req ->
		if _, err := os.Stat(db); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		ch := &cgi.Handler{
			Env:    env,
			Logger: logger,
			Path:   h.fossilPath,
			Args:   []string{cgiPath},
			Dir:    dir,
		}
		ch.ServeHTTP(w, req)
	})

	return handler, nil
}
