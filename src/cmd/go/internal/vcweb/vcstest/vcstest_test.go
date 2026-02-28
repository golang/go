// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcstest_test

import (
	"cmd/go/internal/vcweb"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var (
	dir  = flag.String("dir", "../../../testdata/vcstest", "directory containing scripts to serve")
	host = flag.String("host", "localhost", "hostname on which to serve HTTP")
	port = flag.Int("port", -1, "port on which to serve HTTP; if nonnegative, skips running tests")
)

func TestMain(m *testing.M) {
	flag.Parse()

	if *port >= 0 {
		err := serveStandalone(*host, *port)
		if err != nil {
			log.Fatal(err)
		}
		os.Exit(0)
	}

	m.Run()
}

// serveStandalone serves the vcweb testdata in a standalone HTTP server.
func serveStandalone(host string, port int) (err error) {
	scriptDir, err := filepath.Abs(*dir)
	if err != nil {
		return err
	}
	work, err := os.MkdirTemp("", "vcweb")
	if err != nil {
		return err
	}
	defer func() {
		if rmErr := os.RemoveAll(work); err == nil {
			err = rmErr
		}
	}()

	log.Printf("running scripts in %s", work)

	v, err := vcweb.NewServer(scriptDir, work, log.Default())
	if err != nil {
		return err
	}

	l, err := net.Listen("tcp", fmt.Sprintf("%s:%d", host, port))
	if err != nil {
		return err
	}
	log.Printf("serving on http://%s:%d/", host, l.Addr().(*net.TCPAddr).Port)

	return http.Serve(l, v)
}

// TestScripts verifies that the VCS setup scripts in cmd/go/testdata/vcstest
// run successfully.
func TestScripts(t *testing.T) {
	scriptDir, err := filepath.Abs(*dir)
	if err != nil {
		t.Fatal(err)
	}
	s, err := vcweb.NewServer(scriptDir, t.TempDir(), log.Default())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s)

	// To check for data races in the handler, run the root handler to produce an
	// overview of the script status at an arbitrary point during the test.
	// (We ignore the output because the expected failure mode is a friendly stack
	// dump from the race detector.)
	t.Run("overview", func(t *testing.T) {
		t.Parallel()

		time.Sleep(1 * time.Millisecond) // Give the other handlers time to race.

		resp, err := http.Get(srv.URL)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		} else {
			t.Error(err)
		}
	})

	t.Cleanup(func() {
		// The subtests spawned by WalkDir run in parallel. When they complete, this
		// Cleanup callback will run. At that point we fetch the root URL (which
		// contains a status page), both to test that the root handler runs without
		// crashing and to display a nice summary of the server's view of the test
		// coverage.
		resp, err := http.Get(srv.URL)
		if err == nil {
			var body []byte
			body, err = io.ReadAll(resp.Body)
			if err == nil && testing.Verbose() {
				t.Logf("GET %s:\n%s", srv.URL, body)
			}
			resp.Body.Close()
		}
		if err != nil {
			t.Error(err)
		}

		srv.Close()
	})

	err = filepath.WalkDir(scriptDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}

		rel, err := filepath.Rel(scriptDir, path)
		if err != nil {
			return err
		}
		if rel == "README" {
			return nil
		}

		t.Run(filepath.ToSlash(rel), func(t *testing.T) {
			t.Parallel()

			buf := new(strings.Builder)
			logger := log.New(buf, "", log.LstdFlags)
			// Load the script but don't try to serve the results:
			// different VCS tools have different handler protocols,
			// and the tests that actually use these repos will ensure
			// that they are served correctly as a side effect anyway.
			err := s.HandleScript(rel, logger, func(http.Handler) {})
			if buf.Len() > 0 {
				t.Log(buf)
			}
			if err != nil {
				if _, ok := errors.AsType[vcweb.ServerNotInstalledError](err); ok || errors.Is(err, exec.ErrNotFound) {
					t.Skip(err)
				}
				if skip, ok := errors.AsType[vcweb.SkipError](err); ok {
					if skip.Msg == "" {
						t.Skip("SKIP")
					} else {
						t.Skipf("SKIP: %v", skip.Msg)
					}
				}
				t.Error(err)
			}
		})
		return nil
	})

	if err != nil {
		t.Error(err)
	}
}
