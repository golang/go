// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap

package doc

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
)

// pickUnusedPort finds an unused port by trying to listen on port 0
// and letting the OS pick a port, then closing that connection and
// returning that port number.
// This is inherently racy.
func pickUnusedPort() (int, error) {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		return 0, err
	}
	return port, nil
}

func doPkgsite(urlPath, fragment string) error {
	port, err := pickUnusedPort()
	if err != nil {
		return fmt.Errorf("failed to find port for documentation server: %v", err)
	}
	addr := fmt.Sprintf("localhost:%d", port)
	path, err := url.JoinPath("http://"+addr, urlPath)
	if err != nil {
		return fmt.Errorf("internal error: failed to construct url: %v", err)
	}
	if fragment != "" {
		path += "#" + fragment
	}

	// Turn off the default signal handler for SIGINT (and SIGQUIT on Unix)
	// and instead wait for the child process to handle the signal and
	// exit before exiting ourselves.
	signal.Ignore(signalsToIgnore...)

	// Prepend the local download cache to GOPROXY to get around deprecation checks.
	env := os.Environ()
	vars, err := runCmd(env, goCmd(), "env", "GOPROXY", "GOMODCACHE")
	fields := strings.Fields(vars)
	if err == nil && len(fields) == 2 {
		goproxy, gomodcache := fields[0], fields[1]
		gomodcache = filepath.Join(gomodcache, "cache", "download")
		// Convert absolute path to file URL. pkgsite will not accept
		// Windows absolute paths because they look like a host:path remote.
		// TODO(golang.org/issue/32456): use url.FromFilePath when implemented.
		if strings.HasPrefix(gomodcache, "/") {
			gomodcache = "file://" + gomodcache
		} else {
			gomodcache = "file:///" + filepath.ToSlash(gomodcache)
		}
		env = append(env, "GOPROXY="+gomodcache+","+goproxy)
	}

	const version = "v0.0.0-20250714212547-01b046e81fe7"
	cmd := exec.Command(goCmd(), "run", "golang.org/x/pkgsite/cmd/internal/doc@"+version,
		"-gorepo", buildCtx.GOROOT,
		"-http", addr,
		"-open", path)
	cmd.Env = env
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			// Exit with the same exit status as pkgsite to avoid
			// printing of "exit status" error messages.
			// Any relevant messages have already been printed
			// to stdout or stderr.
			os.Exit(ee.ExitCode())
		}
		return err
	}

	return nil
}
