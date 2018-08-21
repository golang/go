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
	"sync"
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
// A Repo must be safe for simultaneous use by multiple goroutines.
type Repo interface {
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
	//
	// If the requested file does not exist it should return an error for which
	// os.IsNotExist(err) returns true.
	ReadFile(rev, file string, maxSize int64) (data []byte, err error)

	// ReadFileRevs reads a single file at multiple versions.
	// It should refuse to read more than maxSize bytes.
	// The result is a map from each requested rev strings
	// to the associated FileRev. The map must have a non-nil
	// entry for every requested rev (unless ReadFileRevs returned an error).
	// A file simply being missing or even corrupted in revs[i]
	// should be reported only in files[revs[i]].Err, not in the error result
	// from ReadFileRevs.
	// The overall call should return an error (and no map) only
	// in the case of a problem with obtaining the data, such as
	// a network failure.
	// Implementations may assume that revs only contain tags,
	// not direct commit hashes.
	ReadFileRevs(revs []string, file string, maxSize int64) (files map[string]*FileRev, err error)

	// ReadZip downloads a zip file for the subdir subdirectory
	// of the given revision to a new file in a given temporary directory.
	// It should refuse to read more than maxSize bytes.
	// It returns a ReadCloser for a streamed copy of the zip file,
	// along with the actual subdirectory (possibly shorter than subdir)
	// contained in the zip file. All files in the zip file are expected to be
	// nested in a single top-level directory, whose name is not specified.
	ReadZip(rev, subdir string, maxSize int64) (zip io.ReadCloser, actualSubdir string, err error)

	// RecentTag returns the most recent tag at or before the given rev
	// with the given prefix. It should make a best-effort attempt to
	// find a tag that is a valid semantic version (following the prefix),
	// or else the result is not useful to the caller, but it need not
	// incur great expense in doing so. For example, the git implementation
	// of RecentTag limits git's search to tags matching the glob expression
	// "v[0-9]*.[0-9]*.[0-9]*" (after the prefix).
	RecentTag(rev, prefix string) (tag string, err error)
}

// A Rev describes a single revision in a source code repository.
type RevInfo struct {
	Name    string    // complete ID in underlying repository
	Short   string    // shortened ID, for use in pseudo-version
	Version string    // version used in lookup
	Time    time.Time // commit time
	Tags    []string  // known tags for commit
}

// A FileRev describes the result of reading a file at a given revision.
type FileRev struct {
	Rev  string // requested revision
	Data []byte // file data
	Err  error  // error if any; os.IsNotExist(Err)==true if rev exists but file does not exist in that rev
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
// It is set by cmd/go/internal/modload.InitMod.
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
	info, err2 := os.Stat(dir)
	if err == nil && err2 == nil && info.IsDir() {
		// Info file and directory both already exist: reuse.
		have := strings.TrimSuffix(string(data), "\n")
		if have != key {
			return "", fmt.Errorf("%s exists with wrong content (have %q want %q)", dir+".info", have, key)
		}
		if cfg.BuildX {
			fmt.Fprintf(os.Stderr, "# %s for %s %s\n", dir, typ, name)
		}
		return dir, nil
	}

	// Info file or directory missing. Start from scratch.
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

var dirLock sync.Map

// Run runs the command line in the given directory
// (an empty dir means the current directory).
// It returns the standard output and, for a non-zero exit,
// a *RunError indicating the command, exit status, and standard error.
// Standard error is unavailable for commands that exit successfully.
func Run(dir string, cmdline ...interface{}) ([]byte, error) {
	return RunWithStdin(dir, nil, cmdline...)
}

// bashQuoter escapes characters that have special meaning in double-quoted strings in the bash shell.
// See https://www.gnu.org/software/bash/manual/html_node/Double-Quotes.html.
var bashQuoter = strings.NewReplacer(`"`, `\"`, `$`, `\$`, "`", "\\`", `\`, `\\`)

func RunWithStdin(dir string, stdin io.Reader, cmdline ...interface{}) ([]byte, error) {
	if dir != "" {
		muIface, ok := dirLock.Load(dir)
		if !ok {
			muIface, _ = dirLock.LoadOrStore(dir, new(sync.Mutex))
		}
		mu := muIface.(*sync.Mutex)
		mu.Lock()
		defer mu.Unlock()
	}

	cmd := str.StringList(cmdline...)
	if cfg.BuildX {
		text := new(strings.Builder)
		if dir != "" {
			text.WriteString("cd ")
			text.WriteString(dir)
			text.WriteString("; ")
		}
		for i, arg := range cmd {
			if i > 0 {
				text.WriteByte(' ')
			}
			switch {
			case strings.ContainsAny(arg, "'"):
				// Quote args that could be mistaken for quoted args.
				text.WriteByte('"')
				text.WriteString(bashQuoter.Replace(arg))
				text.WriteByte('"')
			case strings.ContainsAny(arg, "$`\\*?[\"\t\n\v\f\r \u0085\u00a0"):
				// Quote args that contain special characters, glob patterns, or spaces.
				text.WriteByte('\'')
				text.WriteString(arg)
				text.WriteByte('\'')
			default:
				text.WriteString(arg)
			}
		}
		fmt.Fprintf(os.Stderr, "%s\n", text)
		start := time.Now()
		defer func() {
			fmt.Fprintf(os.Stderr, "%.3fs # %s\n", time.Since(start).Seconds(), text)
		}()
	}
	// TODO: Impose limits on command output size.
	// TODO: Set environment to get English error messages.
	var stderr bytes.Buffer
	var stdout bytes.Buffer
	c := exec.Command(cmd[0], cmd[1:]...)
	c.Dir = dir
	c.Stdin = stdin
	c.Stderr = &stderr
	c.Stdout = &stdout
	err := c.Run()
	if err != nil {
		err = &RunError{Cmd: strings.Join(cmd, " ") + " in " + dir, Stderr: stderr.Bytes(), Err: err}
	}
	return stdout.Bytes(), err
}
