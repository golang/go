// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"bytes"
	"context"
	"fmt"
	"internal/txtar"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// A State encapsulates the current state of a running script engine,
// including the script environment and any running background commands.
type State struct {
	engine *Engine // the engine currently executing the script, if any

	ctx    context.Context
	cancel context.CancelFunc
	file   string
	log    bytes.Buffer

	workdir string            // initial working directory
	pwd     string            // current working directory during execution
	env     []string          // environment list (for os/exec)
	envMap  map[string]string // environment mapping (matches env)
	stdout  string            // standard output from last 'go' command; for 'stdout' command
	stderr  string            // standard error from last 'go' command; for 'stderr' command

	background []backgroundCmd
}

type backgroundCmd struct {
	*command
	wait WaitFunc
}

// NewState returns a new State permanently associated with ctx, with its
// initial working directory in workdir and its initial environment set to
// initialEnv (or os.Environ(), if initialEnv is nil).
//
// The new State also contains pseudo-environment-variables for
// ${/} and ${:} (for the platform's path and list separators respectively),
// but does not pass those to subprocesses.
func NewState(ctx context.Context, workdir string, initialEnv []string) (*State, error) {
	absWork, err := filepath.Abs(workdir)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(ctx)

	// Make a fresh copy of the env slice to avoid aliasing bugs if we ever
	// start modifying it in place; this also establishes the invariant that
	// s.env contains no duplicates.
	env := cleanEnv(initialEnv, absWork)

	envMap := make(map[string]string, len(env))

	// Add entries for ${:} and ${/} to make it easier to write platform-independent
	// paths in scripts.
	envMap["/"] = string(os.PathSeparator)
	envMap[":"] = string(os.PathListSeparator)

	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok {
			envMap[k] = v
		}
	}

	s := &State{
		ctx:     ctx,
		cancel:  cancel,
		workdir: absWork,
		pwd:     absWork,
		env:     env,
		envMap:  envMap,
	}
	s.Setenv("PWD", absWork)
	return s, nil
}

// CloseAndWait cancels the State's Context and waits for any background commands to
// finish. If any remaining background command ended in an unexpected state,
// Close returns a non-nil error.
func (s *State) CloseAndWait(log io.Writer) error {
	s.cancel()
	wait, err := Wait().Run(s)
	if wait != nil {
		panic("script: internal error: Wait unexpectedly returns its own WaitFunc")
	}
	if flushErr := s.flushLog(log); err == nil {
		err = flushErr
	}
	return err
}

// Chdir changes the State's working directory to the given path.
func (s *State) Chdir(path string) error {
	dir := s.Path(path)
	if _, err := os.Stat(dir); err != nil {
		return &fs.PathError{Op: "Chdir", Path: dir, Err: err}
	}
	s.pwd = dir
	s.Setenv("PWD", dir)
	return nil
}

// Context returns the Context with which the State was created.
func (s *State) Context() context.Context {
	return s.ctx
}

// Environ returns a copy of the current script environment,
// in the form "key=value".
func (s *State) Environ() []string {
	return append([]string(nil), s.env...)
}

// ExpandEnv replaces ${var} or $var in the string according to the values of
// the environment variables in s. References to undefined variables are
// replaced by the empty string.
func (s *State) ExpandEnv(str string, inRegexp bool) string {
	return os.Expand(str, func { key ->
		e := s.envMap[key]
		if inRegexp {
			// Quote to literal strings: we want paths like C:\work\go1.4 to remain
			// paths rather than regular expressions.
			e = regexp.QuoteMeta(e)
		}
		return e
	})
}

// ExtractFiles extracts the files in ar to the state's current directory,
// expanding any environment variables within each name.
//
// The files must reside within the working directory with which the State was
// originally created.
func (s *State) ExtractFiles(ar *txtar.Archive) error {
	wd := s.workdir

	// Add trailing separator to terminate wd.
	// This prevents extracting to outside paths which prefix wd,
	// e.g. extracting to /home/foobar when wd is /home/foo
	if wd == "" {
		panic("s.workdir is unexpectedly empty")
	}
	if !os.IsPathSeparator(wd[len(wd)-1]) {
		wd += string(filepath.Separator)
	}

	for _, f := range ar.Files {
		name := s.Path(s.ExpandEnv(f.Name, false))

		if !strings.HasPrefix(name, wd) {
			return fmt.Errorf("file %#q is outside working directory", f.Name)
		}

		if err := os.MkdirAll(filepath.Dir(name), 0777); err != nil {
			return err
		}
		if err := os.WriteFile(name, f.Data, 0666); err != nil {
			return err
		}
	}

	return nil
}

// Getwd returns the directory in which to run the next script command.
func (s *State) Getwd() string { return s.pwd }

// Logf writes output to the script's log without updating its stdout or stderr
// buffers. (The output log functions as a kind of meta-stderr.)
func (s *State) Logf(format string, args ...any) {
	fmt.Fprintf(&s.log, format, args...)
}

// flushLog writes the contents of the script's log to w and clears the log.
func (s *State) flushLog(w io.Writer) error {
	_, err := w.Write(s.log.Bytes())
	s.log.Reset()
	return err
}

// LookupEnv retrieves the value of the environment variable in s named by the key.
func (s *State) LookupEnv(key string) (string, bool) {
	v, ok := s.envMap[key]
	return v, ok
}

// Path returns the absolute path in the host operating system for a
// script-based (generally slash-separated and relative) path.
func (s *State) Path(path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(s.pwd, path)
}

// Setenv sets the value of the environment variable in s named by the key.
func (s *State) Setenv(key, value string) error {
	s.env = cleanEnv(append(s.env, key+"="+value), s.pwd)
	s.envMap[key] = value
	return nil
}

// Stdout returns the stdout output of the last command run,
// or the empty string if no command has been run.
func (s *State) Stdout() string { return s.stdout }

// Stderr returns the stderr output of the last command run,
// or the empty string if no command has been run.
func (s *State) Stderr() string { return s.stderr }

// cleanEnv returns a copy of env with any duplicates removed in favor of
// later values and any required system variables defined.
//
// If env is nil, cleanEnv copies the environment from os.Environ().
func cleanEnv(env []string, pwd string) []string {
	// There are some funky edge-cases in this logic, especially on Windows (with
	// case-insensitive environment variables and variables with keys like "=C:").
	// Rather than duplicating exec.dedupEnv here, cheat and use exec.Cmd directly.
	cmd := &exec.Cmd{Env: env}
	cmd.Dir = pwd
	return cmd.Environ()
}
