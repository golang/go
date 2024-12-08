// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package codehost defines the interface implemented by a code hosting source,
// along with support code for use by implementations.
package codehost

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/str"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
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
//
// A Repo must be safe for simultaneous use by multiple goroutines,
// and callers must not modify returned values, which may be cached and shared.
type Repo interface {
	// CheckReuse checks whether the old origin information
	// remains up to date. If so, whatever cached object it was
	// taken from can be reused.
	// The subdir gives subdirectory name where the module root is expected to be found,
	// "" for the root or "sub/dir" for a subdirectory (no trailing slash).
	CheckReuse(ctx context.Context, old *Origin, subdir string) error

	// Tags lists all tags with the given prefix.
	Tags(ctx context.Context, prefix string) (*Tags, error)

	// Stat returns information about the revision rev.
	// A revision can be any identifier known to the underlying service:
	// commit hash, branch, tag, and so on.
	Stat(ctx context.Context, rev string) (*RevInfo, error)

	// Latest returns the latest revision on the default branch,
	// whatever that means in the underlying implementation.
	Latest(ctx context.Context) (*RevInfo, error)

	// ReadFile reads the given file in the file tree corresponding to revision rev.
	// It should refuse to read more than maxSize bytes.
	//
	// If the requested file does not exist it should return an error for which
	// os.IsNotExist(err) returns true.
	ReadFile(ctx context.Context, rev, file string, maxSize int64) (data []byte, err error)

	// ReadZip downloads a zip file for the subdir subdirectory
	// of the given revision to a new file in a given temporary directory.
	// It should refuse to read more than maxSize bytes.
	// It returns a ReadCloser for a streamed copy of the zip file.
	// All files in the zip file are expected to be
	// nested in a single top-level directory, whose name is not specified.
	ReadZip(ctx context.Context, rev, subdir string, maxSize int64) (zip io.ReadCloser, err error)

	// RecentTag returns the most recent tag on rev or one of its predecessors
	// with the given prefix. allowed may be used to filter out unwanted versions.
	RecentTag(ctx context.Context, rev, prefix string, allowed func(tag string) bool) (tag string, err error)

	// DescendsFrom reports whether rev or any of its ancestors has the given tag.
	//
	// DescendsFrom must return true for any tag returned by RecentTag for the
	// same revision.
	DescendsFrom(ctx context.Context, rev, tag string) (bool, error)
}

// An Origin describes the provenance of a given repo method result.
// It can be passed to CheckReuse (usually in a different go command invocation)
// to see whether the result remains up-to-date.
type Origin struct {
	VCS    string `json:",omitempty"` // "git" etc
	URL    string `json:",omitempty"` // URL of repository
	Subdir string `json:",omitempty"` // subdirectory in repo

	Hash string `json:",omitempty"` // commit hash or ID

	// If TagSum is non-empty, then the resolution of this module version
	// depends on the set of tags present in the repo, specifically the tags
	// of the form TagPrefix + a valid semver version.
	// If the matching repo tags and their commit hashes still hash to TagSum,
	// the Origin is still valid (at least as far as the tags are concerned).
	// The exact checksum is up to the Repo implementation; see (*gitRepo).Tags.
	TagPrefix string `json:",omitempty"`
	TagSum    string `json:",omitempty"`

	// If Ref is non-empty, then the resolution of this module version
	// depends on Ref resolving to the revision identified by Hash.
	// If Ref still resolves to Hash, the Origin is still valid (at least as far as Ref is concerned).
	// For Git, the Ref is a full ref like "refs/heads/main" or "refs/tags/v1.2.3",
	// and the Hash is the Git object hash the ref maps to.
	// Other VCS might choose differently, but the idea is that Ref is the name
	// with a mutable meaning while Hash is a name with an immutable meaning.
	Ref string `json:",omitempty"`

	// If RepoSum is non-empty, then the resolution of this module version
	// failed due to the repo being available but the version not being present.
	// This depends on the entire state of the repo, which RepoSum summarizes.
	// For Git, this is a hash of all the refs and their hashes.
	RepoSum string `json:",omitempty"`
}

// A Tags describes the available tags in a code repository.
type Tags struct {
	Origin *Origin
	List   []Tag
}

// A Tag describes a single tag in a code repository.
type Tag struct {
	Name string
	Hash string // content hash identifying tag's content, if available
}

// isOriginTag reports whether tag should be preserved
// in the Tags method's Origin calculation.
// We can safely ignore tags that are not look like pseudo-versions,
// because ../coderepo.go's (*codeRepo).Versions ignores them too.
// We can also ignore non-semver tags, but we have to include semver
// tags with extra suffixes, because the pseudo-version base finder uses them.
func isOriginTag(tag string) bool {
	// modfetch.(*codeRepo).Versions uses Canonical == tag,
	// but pseudo-version calculation has a weaker condition that
	// the canonical is a prefix of the tag.
	// Include those too, so that if any new one appears, we'll invalidate the cache entry.
	// This will lead to spurious invalidation of version list results,
	// but tags of this form being created should be fairly rare
	// (and invalidate pseudo-version results anyway).
	c := semver.Canonical(tag)
	return c != "" && strings.HasPrefix(tag, c) && !module.IsPseudoVersion(tag)
}

// A RevInfo describes a single revision in a source code repository.
type RevInfo struct {
	Origin  *Origin
	Name    string    // complete ID in underlying repository
	Short   string    // shortened ID, for use in pseudo-version
	Version string    // version used in lookup
	Time    time.Time // commit time
	Tags    []string  // known tags for commit
}

// UnknownRevisionError is an error equivalent to fs.ErrNotExist, but for a
// revision rather than a file.
type UnknownRevisionError struct {
	Rev string
}

func (e *UnknownRevisionError) Error() string {
	return "unknown revision " + e.Rev
}
func (UnknownRevisionError) Is(err error) bool {
	return err == fs.ErrNotExist
}

// ErrNoCommits is an error equivalent to fs.ErrNotExist indicating that a given
// repository or module contains no commits.
var ErrNoCommits error = noCommitsError{}

type noCommitsError struct{}

func (noCommitsError) Error() string {
	return "no commits"
}
func (noCommitsError) Is(err error) bool {
	return err == fs.ErrNotExist
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

// WorkDir returns the name of the cached work directory to use for the
// given repository type and name.
func WorkDir(ctx context.Context, typ, name string) (dir, lockfile string, err error) {
	if cfg.GOMODCACHE == "" {
		return "", "", fmt.Errorf("neither GOPATH nor GOMODCACHE are set")
	}

	// We name the work directory for the SHA256 hash of the type and name.
	// We intentionally avoid the actual name both because of possible
	// conflicts with valid file system paths and because we want to ensure
	// that one checkout is never nested inside another. That nesting has
	// led to security problems in the past.
	if strings.Contains(typ, ":") {
		return "", "", fmt.Errorf("codehost.WorkDir: type cannot contain colon")
	}
	key := typ + ":" + name
	dir = filepath.Join(cfg.GOMODCACHE, "cache/vcs", fmt.Sprintf("%x", sha256.Sum256([]byte(key))))

	xLog, buildX := cfg.BuildXWriter(ctx)
	if buildX {
		fmt.Fprintf(xLog, "mkdir -p %s # %s %s\n", filepath.Dir(dir), typ, name)
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0777); err != nil {
		return "", "", err
	}

	lockfile = dir + ".lock"
	if buildX {
		fmt.Fprintf(xLog, "# lock %s\n", lockfile)
	}

	unlock, err := lockedfile.MutexAt(lockfile).Lock()
	if err != nil {
		return "", "", fmt.Errorf("codehost.WorkDir: can't find or create lock file: %v", err)
	}
	defer unlock()

	data, err := os.ReadFile(dir + ".info")
	info, err2 := os.Stat(dir)
	if err == nil && err2 == nil && info.IsDir() {
		// Info file and directory both already exist: reuse.
		have := strings.TrimSuffix(string(data), "\n")
		if have != key {
			return "", "", fmt.Errorf("%s exists with wrong content (have %q want %q)", dir+".info", have, key)
		}
		if buildX {
			fmt.Fprintf(xLog, "# %s for %s %s\n", dir, typ, name)
		}
		return dir, lockfile, nil
	}

	// Info file or directory missing. Start from scratch.
	if xLog != nil {
		fmt.Fprintf(xLog, "mkdir -p %s # %s %s\n", dir, typ, name)
	}
	os.RemoveAll(dir)
	if err := os.MkdirAll(dir, 0777); err != nil {
		return "", "", err
	}
	if err := os.WriteFile(dir+".info", []byte(key), 0666); err != nil {
		os.RemoveAll(dir)
		return "", "", err
	}
	return dir, lockfile, nil
}

type RunError struct {
	Cmd      string
	Err      error
	Stderr   []byte
	HelpText string
}

func (e *RunError) Error() string {
	text := e.Cmd + ": " + e.Err.Error()
	stderr := bytes.TrimRight(e.Stderr, "\n")
	if len(stderr) > 0 {
		text += ":\n\t" + strings.ReplaceAll(string(stderr), "\n", "\n\t")
	}
	if len(e.HelpText) > 0 {
		text += "\n" + e.HelpText
	}
	return text
}

var dirLock sync.Map

type RunArgs struct {
	cmdline []any    // the command to run
	dir     string   // the directory to run the command in
	local   bool     // true if the VCS information is local
	env     []string // environment variables for the command
	stdin   io.Reader
}

// Run runs the command line in the given directory
// (an empty dir means the current directory).
// It returns the standard output and, for a non-zero exit,
// a *RunError indicating the command, exit status, and standard error.
// Standard error is unavailable for commands that exit successfully.
func Run(ctx context.Context, dir string, cmdline ...any) ([]byte, error) {
	return run(ctx, RunArgs{cmdline: cmdline, dir: dir})
}

// RunWithArgs is the same as Run but it also accepts additional arguments.
func RunWithArgs(ctx context.Context, args RunArgs) ([]byte, error) {
	return run(ctx, args)
}

// bashQuoter escapes characters that have special meaning in double-quoted strings in the bash shell.
// See https://www.gnu.org/software/bash/manual/html_node/Double-Quotes.html.
var bashQuoter = strings.NewReplacer(`"`, `\"`, `$`, `\$`, "`", "\\`", `\`, `\\`)

func run(ctx context.Context, args RunArgs) ([]byte, error) {
	if args.dir != "" {
		muIface, ok := dirLock.Load(args.dir)
		if !ok {
			muIface, _ = dirLock.LoadOrStore(args.dir, new(sync.Mutex))
		}
		mu := muIface.(*sync.Mutex)
		mu.Lock()
		defer mu.Unlock()
	}

	cmd := str.StringList(args.cmdline...)
	if os.Getenv("TESTGOVCSREMOTE") == "panic" && !args.local {
		panic(fmt.Sprintf("use of remote vcs: %v", cmd))
	}
	if xLog, ok := cfg.BuildXWriter(ctx); ok {
		text := new(strings.Builder)
		if args.dir != "" {
			text.WriteString("cd ")
			text.WriteString(args.dir)
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
		fmt.Fprintf(xLog, "%s\n", text)
		start := time.Now()
		defer func() {
			fmt.Fprintf(xLog, "%.3fs # %s\n", time.Since(start).Seconds(), text)
		}()
	}
	// TODO: Impose limits on command output size.
	// TODO: Set environment to get English error messages.
	var stderr bytes.Buffer
	var stdout bytes.Buffer
	c := exec.CommandContext(ctx, cmd[0], cmd[1:]...)
	c.Cancel = func() error { return c.Process.Signal(os.Interrupt) }
	c.Dir = args.dir
	c.Stdin = args.stdin
	c.Stderr = &stderr
	c.Stdout = &stdout
	c.Env = append(c.Environ(), args.env...)
	err := c.Run()
	if err != nil {
		err = &RunError{Cmd: strings.Join(cmd, " ") + " in " + args.dir, Stderr: stderr.Bytes(), Err: err}
	}
	return stdout.Bytes(), err
}
