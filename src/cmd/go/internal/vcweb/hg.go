// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"bufio"
	"context"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"time"
)

type hgHandler struct {
	once      sync.Once
	hgPath    string
	hgPathErr error
}

func (h *hgHandler) Available() bool {
	h.once.Do(func() {
		h.hgPath, h.hgPathErr = exec.LookPath("hg")
	})
	return h.hgPathErr == nil
}

func (h *hgHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	if !h.Available() {
		return nil, ServerNotInstalledError{name: "hg"}
	}

	handler := http.HandlerFunc(func { w, req ->
		// Mercurial has a CGI server implementation (called hgweb). In theory we
		// could use that — however, assuming that hgweb is even installed, the
		// configuration for hgweb varies by Python version (2 vs 3), and we would
		// rather not go rooting around trying to find the right Python version to
		// run.
		//
		// Instead, we'll take a somewhat more roundabout approach: we assume that
		// if "hg" works at all then "hg serve" works too, and we'll execute that as
		// a subprocess, using a reverse proxy to forward the request and response.

		ctx, cancel := context.WithCancel(req.Context())
		defer cancel()

		cmd := exec.CommandContext(ctx, h.hgPath, "serve", "--port", "0", "--address", "localhost", "--accesslog", os.DevNull, "--name", "vcweb", "--print-url")
		cmd.Dir = dir
		cmd.Env = append(slices.Clip(env), "PWD="+dir)

		cmd.Cancel = func {
			err := cmd.Process.Signal(os.Interrupt)
			if err != nil && !errors.Is(err, os.ErrProcessDone) {
				err = cmd.Process.Kill()
			}
			return err
		}
		// This WaitDelay is arbitrary. After 'hg serve' prints its URL, any further
		// I/O is only for debugging. (The actual output goes through the HTTP URL,
		// not the standard I/O streams.)
		cmd.WaitDelay = 10 * time.Second

		stderr := new(strings.Builder)
		cmd.Stderr = stderr

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		if err := cmd.Start(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var wg sync.WaitGroup
		defer func() {
			cancel()
			err := cmd.Wait()
			if out := strings.TrimSuffix(stderr.String(), "interrupted!\n"); out != "" {
				logger.Printf("%v: %v\n%s", cmd, err, out)
			} else {
				logger.Printf("%v", cmd)
			}
			wg.Wait()
		}()

		r := bufio.NewReader(stdout)
		line, err := r.ReadString('\n')
		if err != nil {
			return
		}
		// We have read what should be the server URL. 'hg serve' shouldn't need to
		// write anything else to stdout, but it's not a big deal if it does anyway.
		// Keep the stdout pipe open so that 'hg serve' won't get a SIGPIPE, but
		// actively discard its output so that it won't hang on a blocking write.
		wg.Add(1)
		go func() {
			io.Copy(io.Discard, r)
			wg.Done()
		}()

		u, err := url.Parse(strings.TrimSpace(line))
		if err != nil {
			logger.Printf("%v: %v", cmd, err)
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		logger.Printf("proxying hg request to %s", u)
		httputil.NewSingleHostReverseProxy(u).ServeHTTP(w, req)
	})

	return handler, nil
}
