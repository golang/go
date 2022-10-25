// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"bufio"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
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

		cmd := exec.Command(h.hgPath, "serve", "--port", "0", "--address", "localhost", "--accesslog", os.DevNull, "--name", "vcweb", "--print-url")
		cmd.Dir = dir
		cmd.Env = append(env[:len(env):len(env)], "PWD="+dir)

		stderr := new(strings.Builder)
		cmd.Stderr = stderr

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		readDone := make(chan struct{})
		defer func() {
			stdout.Close()
			<-readDone
		}()

		hgURL := make(chan *url.URL, 1)
		hgURLError := make(chan error, 1)
		go func() {
			defer close(readDone)
			r := bufio.NewReader(stdout)
			for {
				line, err := r.ReadString('\n')
				if err != nil {
					return
				}
				u, err := url.Parse(strings.TrimSpace(line))
				if err == nil {
					hgURL <- u
				} else {
					hgURLError <- err
				}
				break
			}
			io.Copy(io.Discard, r)
		}()

		if err := cmd.Start(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() {
			if err := cmd.Process.Signal(os.Interrupt); err != nil && !errors.Is(err, os.ErrProcessDone) {
				cmd.Process.Kill()
			}
			err := cmd.Wait()
			if out := strings.TrimSuffix(stderr.String(), "interrupted!\n"); out != "" {
				logger.Printf("%v: %v\n%s", cmd, err, out)
			} else {
				logger.Printf("%v", cmd)
			}
		}()

		select {
		case <-req.Context().Done():
			logger.Printf("%v: %v", req.Context().Err(), cmd)
			http.Error(w, req.Context().Err().Error(), http.StatusBadGateway)
			return
		case err := <-hgURLError:
			logger.Printf("%v: %v", cmd, err)
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		case url := <-hgURL:
			logger.Printf("proxying hg request to %s", url)
			httputil.NewSingleHostReverseProxy(url).ServeHTTP(w, req)
		}
	})

	return handler, nil
}
