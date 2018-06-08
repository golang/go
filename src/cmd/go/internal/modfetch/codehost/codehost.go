// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package codehost defines the interface implemented by a code hosting source,
// along with support code for use by implementations.
package codehost

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/str"
)

// Downloaded size limits.
const (
	MaxGoMod   = 16 << 20  // maximum size of go.mod file
	MaxLICENSE = 16 << 20  // maximum size of LICENSE file
	MaxZipFile = 500 << 20 // maximum size of downloaded zip file
)

// A Repo represents a code hosting source.
// Typical implementations include local version control repositories,
// remote version control servers, and code hosting sites.
type Repo interface {
	// Root returns the import path of the root directory of the repository.
	Root() string

	// List lists all tags with the given prefix.
	Tags(prefix string) (tags []string, err error)

	// Stat returns information about the revision rev.
	// A revision can be any identifier known to the underlying service:
	// commit hash, branch, tag, and so on.
	Stat(rev string) (*RevInfo, error)

	// Latest returns the latest revision on the default branch,
	// whatever that means in the underlying implementation.
	Latest() (*RevInfo, error)

	// ReadFile reads the given file in the file tree corresponding to revision rev.
	// It should refuse to read more than maxSize bytes.
	ReadFile(rev, file string, maxSize int64) (data []byte, err error)

	// ReadZip downloads a zip file for the subdir subdirectory
	// of the given revision to a new file in a given temporary directory.
	// It should refuse to read more than maxSize bytes.
	// It returns a ReadCloser for a streamed copy of the zip file,
	// along with the actual subdirectory (possibly shorter than subdir)
	// contained in the zip file. All files in the zip file are expected to be
	// nested in a single top-level directory, whose name is not specified.
	ReadZip(rev, subdir string, maxSize int64) (zip io.ReadCloser, actualSubdir string, err error)
}

// A Rev describes a single revision in a source code repository.
type RevInfo struct {
	Name    string    // complete ID in underlying repository
	Short   string    // shortened ID, for use in pseudo-version
	Version string    // TODO what is this?
	Time    time.Time // commit time
}

// AllHex reports whether the revision rev is entirely lower-case hexadecimal digits.
func AllHex(rev string) bool {
	for i := 0; i < len(rev); i++ {
		c := rev[i]
		if '0' <= c && c <= '9' || 'a' <= c && c <= 'f' {
			continue
		}
		return false
	}
	return true
}

// ShortenSHA1 shortens a SHA1 hash (40 hex digits) to the canonical length
// used in pseudo-versions (12 hex digits).
func ShortenSHA1(rev string) string {
	if AllHex(rev) && len(rev) == 40 {
		return rev[:12]
	}
	return rev
}

// WorkRoot is the root of the cached work directory.
// It is set by cmd/go/internal/vgo.InitMod.
var WorkRoot string

// WorkDir returns the name of the cached work directory to use for the
// given repository type and name.
func WorkDir(typ, name string) (string, error) {
	if WorkRoot == "" {
		return "", fmt.Errorf("codehost.WorkRoot not set")
	}

	// We name the work directory for the SHA256 hash of the type and name.
	// We intentionally avoid the actual name both because of possible
	// conflicts with valid file system paths and because we want to ensure
	// that one checkout is never nested inside another. That nesting has
	// led to security problems in the past.
	if strings.Contains(typ, ":") {
		return "", fmt.Errorf("codehost.WorkDir: type cannot contain colon")
	}
	key := typ + ":" + name
	dir := filepath.Join(WorkRoot, fmt.Sprintf("%x", sha256.Sum256([]byte(key))))
	data, err := ioutil.ReadFile(dir + ".info")
	if err == nil {
		have := strings.TrimSuffix(string(data), "\n")
		if have != key {
			return "", fmt.Errorf("%s exists with wrong content (have %q want %q)", dir+".info", have, key)
		}
		_, err := os.Stat(dir)
		if err != nil {
			return "", fmt.Errorf("%s exists but %s does not", dir+".info", dir)
		}
		if cfg.BuildX {
			fmt.Fprintf(os.Stderr, "# %s for %s %s\n", dir, typ, name)
		}
		return dir, nil
	}

	if cfg.BuildX {
		fmt.Fprintf(os.Stderr, "mkdir -p %s # %s %s\n", dir, typ, name)
	}
	os.RemoveAll(dir)
	if err := os.MkdirAll(dir, 0777); err != nil {
		return "", err
	}
	if err := ioutil.WriteFile(dir+".info", []byte(key), 0666); err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	return dir, nil
}

type RunError struct {
	Cmd    string
	Err    error
	Stderr []byte
}

func (e *RunError) Error() string {
	text := e.Cmd + ": " + e.Err.Error()
	stderr := bytes.TrimRight(e.Stderr, "\n")
	if len(stderr) > 0 {
		text += ":\n\t" + strings.Replace(string(stderr), "\n", "\n\t", -1)
	}
	return text
}

// Run runs the command line in the given directory
// (an empty dir means the current directory).
// It returns the standard output and, for a non-zero exit,
// a *RunError indicating the command, exit status, and standard error.
// Standard error is unavailable for commands that exit successfully.
func Run(dir string, cmdline ...interface{}) ([]byte, error) {
	cmd := str.StringList(cmdline...)
	if cfg.BuildX {
		var cd string
		if dir != "" {
			cd = "cd " + dir + "; "
		}
		fmt.Fprintf(os.Stderr, "%s%s\n", cd, strings.Join(cmd, " "))
	}
	// TODO: Impose limits on command output size.
	// TODO: Set environment to get English error messages.
	var stderr bytes.Buffer
	var stdout bytes.Buffer
	c := exec.Command(cmd[0], cmd[1:]...)
	c.Dir = dir
	c.Stderr = &stderr
	c.Stdout = &stdout
	err := c.Run()
	if err != nil {
		err = &RunError{Cmd: strings.Join(cmd, " ") + " in " + dir, Stderr: stderr.Bytes(), Err: err}
	}
	return stdout.Bytes(), err
}
