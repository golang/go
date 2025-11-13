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

	mu     sync.Mutex
	wg     sync.WaitGroup
	ctx    context.Context
	cancel func()
	cmds   []*exec.Cmd
	url    map[string]*url.URL
}

func (h *hgHandler) Available() bool {
	h.once.Do(func() {
		h.hgPath, h.hgPathErr = exec.LookPath("hg")
	})
	return h.hgPathErr == nil
}

func (h *hgHandler) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cancel == nil {
		return nil
	}

	h.cancel()
	for _, cmd := range h.cmds {
		h.wg.Add(1)
		go func() {
			cmd.Wait()
			h.wg.Done()
		}()
	}
	h.wg.Wait()
	h.url = nil
	h.cmds = nil
	h.ctx = nil
	h.cancel = nil
	return nil
}

func (h *hgHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	if !h.Available() {
		return nil, ServerNotInstalledError{name: "hg"}
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Mercurial has a CGI server implementation (called hgweb). In theory we
		// could use that — however, assuming that hgweb is even installed, the
		// configuration for hgweb varies by Python version (2 vs 3), and we would
		// rather not go rooting around trying to find the right Python version to
		// run.
		//
		// Instead, we'll take a somewhat more roundabout approach: we assume that
		// if "hg" works at all then "hg serve" works too, and we'll execute that as
		// a subprocess, using a reverse proxy to forward the request and response.

		h.mu.Lock()

		if h.ctx == nil {
			h.ctx, h.cancel = context.WithCancel(context.Background())
		}

		// Cache the hg server subprocess globally, because hg is too slow
		// to start a new one for each request. There are under a dozen different
		// repos we serve, so leaving a dozen processes around is not a big deal.
		u := h.url[dir]
		if u != nil {
			h.mu.Unlock()
			logger.Printf("proxying hg request to %s", u)
			httputil.NewSingleHostReverseProxy(u).ServeHTTP(w, req)
			return
		}

		logger.Printf("starting hg serve for %s", dir)
		cmd := exec.CommandContext(h.ctx, h.hgPath, "serve", "--port", "0", "--address", "localhost", "--accesslog", os.DevNull, "--name", "vcweb", "--print-url")
		cmd.Dir = dir
		cmd.Env = append(slices.Clip(env), "PWD="+dir)

		cmd.Cancel = func() error {
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
			h.mu.Unlock()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		if err := cmd.Start(); err != nil {
			h.mu.Unlock()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		r := bufio.NewReader(stdout)
		line, err := r.ReadString('\n')
		if err != nil {
			h.mu.Unlock()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// We have read what should be the server URL. 'hg serve' shouldn't need to
		// write anything else to stdout, but it's not a big deal if it does anyway.
		// Keep the stdout pipe open so that 'hg serve' won't get a SIGPIPE, but
		// actively discard its output so that it won't hang on a blocking write.
		h.wg.Add(1)
		go func() {
			io.Copy(io.Discard, r)
			h.wg.Done()
		}()

		// On some systems,
		// hg serve --address=localhost --print-url prints in-addr.arpa hostnames
		// even though they cannot be looked up.
		// Replace them with IP literals.
		line = strings.ReplaceAll(line, "//1.0.0.127.in-addr.arpa", "//127.0.0.1")
		line = strings.ReplaceAll(line, "//1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa", "//[::1]")

		u, err = url.Parse(strings.TrimSpace(line))
		if err != nil {
			h.mu.Unlock()
			logger.Printf("%v: %v", cmd, err)
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}

		if h.url == nil {
			h.url = make(map[string]*url.URL)
		}
		h.url[dir] = u
		h.cmds = append(h.cmds, cmd)
		h.mu.Unlock()

		logger.Printf("proxying hg request to %s", u)
		httputil.NewSingleHostReverseProxy(u).ServeHTTP(w, req)
	})

	return handler, nil
}
