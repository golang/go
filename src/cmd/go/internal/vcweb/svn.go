// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"io"
	"log"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"sync"
)

// An svnHandler serves requests for Subversion repos.
//
// Unlike the other vcweb handlers, svnHandler does not serve the Subversion
// protocol directly over the HTTP connection. Instead, it opens a separate port
// that serves the (non-HTTP) 'svn' protocol. The test binary can retrieve the
// URL for that port by sending an HTTP request with the query parameter
// "vcwebsvn=1".
//
// We take this approach because the 'svn' protocol is implemented by a
// lightweight 'svnserve' binary that is usually packaged along with the 'svn'
// client binary, whereas only known implementation of the Subversion HTTP
// protocol is the mod_dav_svn apache2 module. Apache2 has a lot of dependencies
// and also seems to rely on global configuration via well-known file paths, so
// implementing a hermetic test using apache2 would require the test to run in a
// complicated container environment, which wouldn't be nearly as
// straightforward for Go contributors to set up and test against on their local
// machine.
type svnHandler struct {
	svnRoot string // a directory containing all svn repos to be served
	logger  *log.Logger

	pathOnce     sync.Once
	svnservePath string // the path to the 'svnserve' executable
	svnserveErr  error

	listenOnce sync.Once
	s          chan *svnState // 1-buffered
}

// An svnState describes the state of a port serving the 'svn://' protocol.
type svnState struct {
	listener  net.Listener
	listenErr error
	conns     map[net.Conn]struct{}
	closing   bool
	done      chan struct{}
}

func (h *svnHandler) Available() bool {
	h.pathOnce.Do(func() {
		h.svnservePath, h.svnserveErr = exec.LookPath("svnserve")
	})
	return h.svnserveErr == nil
}

// Handler returns an http.Handler that checks for the "vcwebsvn" query
// parameter and then serves the 'svn://' URL for the repository at the
// requested path.
// The HTTP client is expected to read that URL and pass it to the 'svn' client.
func (h *svnHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	if !h.Available() {
		return nil, ServerNotInstalledError{name: "svn"}
	}

	// Go ahead and start the listener now, so that if it fails (for example, due
	// to port exhaustion) we can return an error from the Handler method instead
	// of serving an error for each individual HTTP request.
	h.listenOnce.Do(func() {
		h.s = make(chan *svnState, 1)
		l, err := net.Listen("tcp", "localhost:0")
		done := make(chan struct{})

		h.s <- &svnState{
			listener:  l,
			listenErr: err,
			conns:     map[net.Conn]struct{}{},
			done:      done,
		}
		if err != nil {
			close(done)
			return
		}

		h.logger.Printf("serving svn on svn://%v", l.Addr())

		go func() {
			for {
				c, err := l.Accept()

				s := <-h.s
				if err != nil {
					s.listenErr = err
					if len(s.conns) == 0 {
						close(s.done)
					}
					h.s <- s
					return
				}
				if s.closing {
					c.Close()
				} else {
					s.conns[c] = struct{}{}
					go h.serve(c)
				}
				h.s <- s
			}
		}()
	})

	s := <-h.s
	addr := ""
	if s.listener != nil {
		addr = s.listener.Addr().String()
	}
	err := s.listenErr
	h.s <- s
	if err != nil {
		return nil, err
	}

	handler := http.HandlerFunc(func { w, req ->
		if req.FormValue("vcwebsvn") != "" {
			w.Header().Add("Content-Type", "text/plain; charset=UTF-8")
			io.WriteString(w, "svn://"+addr+"\n")
			return
		}
		http.NotFound(w, req)
	})

	return handler, nil
}

// serve serves a single 'svn://' connection on c.
func (h *svnHandler) serve(c net.Conn) {
	defer func() {
		c.Close()

		s := <-h.s
		delete(s.conns, c)
		if len(s.conns) == 0 && s.listenErr != nil {
			close(s.done)
		}
		h.s <- s
	}()

	// The "--inetd" flag causes svnserve to speak the 'svn' protocol over its
	// stdin and stdout streams as if invoked by the Unix "inetd" service.
	// We aren't using inetd, but we are implementing essentially the same
	// approach: using a host process to listen for connections and spawn
	// subprocesses to serve them.
	cmd := exec.Command(h.svnservePath, "--read-only", "--root="+h.svnRoot, "--inetd")
	cmd.Stdin = c
	cmd.Stdout = c
	stderr := new(strings.Builder)
	cmd.Stderr = stderr
	err := cmd.Run()

	var errFrag any = "ok"
	if err != nil {
		errFrag = err
	}
	stderrFrag := ""
	if stderr.Len() > 0 {
		stderrFrag = "\n" + stderr.String()
	}
	h.logger.Printf("%v: %s%s", cmd, errFrag, stderrFrag)
}

// Close stops accepting new svn:// connections and terminates the existing
// ones, then waits for the 'svnserve' subprocesses to complete.
func (h *svnHandler) Close() error {
	h.listenOnce.Do(func() {})
	if h.s == nil {
		return nil
	}

	var err error
	s := <-h.s
	s.closing = true
	if s.listener == nil {
		err = s.listenErr
	} else {
		err = s.listener.Close()
	}
	for c := range s.conns {
		c.Close()
	}
	done := s.done
	h.s <- s

	<-done
	return err
}
