// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Portable script commands. Native Windows lacks find(1), env(1), and
// extension-less executables, so the fork registers small wrappers that
// keep upstream script steps working without per-test edits.

package main_test

import (
	"cmd/internal/script"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// scriptFind implements a portable subset of find(1) for script tests:
//
//	find dir [ ( ] -name glob [ -o -name glob ... ] [ ) ]
//
// It prints each matching path, relative to the working directory, on
// its own stdout line. Native Windows has no find(1) on PATH (its find
// searches file contents), so script tests cannot exec it portably.
func scriptFind() script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "list files matching -name globs under a directory",
			Args:    "dir [ ( ] -name 'glob' [ -o -name 'glob' ... ] [ ) ]",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) < 3 {
				return nil, script.ErrUsage
			}
			root := s.Path(args[0])
			var pats []string
			for i := 1; i < len(args); i++ {
				if args[i] == "-name" && i+1 < len(args) {
					pats = append(pats, args[i+1])
					i++
				}
			}
			if len(pats) == 0 {
				return nil, script.ErrUsage
			}
			var matches []string
			err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				for _, pat := range pats {
					if ok, _ := path.Match(pat, d.Name()); ok {
						rel, err := filepath.Rel(s.Getwd(), p)
						if err != nil {
							return err
						}
						matches = append(matches, filepath.ToSlash(rel))
						break
					}
				}
				return nil
			})
			if err != nil {
				return nil, err
			}
			return func(*script.State) (stdout, stderr string, err error) {
				stdout = strings.Join(matches, "\n")
				if stdout != "" {
					stdout += "\n"
				}
				return stdout, "", nil
			}, nil
		})
}

// scriptExecEnvU wraps exec so `exec env -u NAME... [NAME=value]... [--]
// program args` works where env(1) is absent (native Windows). Every
// other exec falls through to the wrapped command unchanged.
func scriptExecEnvU(base script.Cmd, cancel func(*exec.Cmd) error, waitDelay time.Duration) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "run an executable program with arguments",
			Args:    "program [args...]",
			Detail: []string{
				"Note that 'exec' does not terminate the script (unlike Unix shells).",
			},
			Async: true,
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) < 2 || args[0] != "env" {
				return base.Run(s, args...)
			}
			var unset, sets []string
			i := 1
			for ; i < len(args); i++ {
				switch {
				case args[i] == "--":
					i++
					goto parsed
				case args[i] == "-u" && i+1 < len(args):
					unset = append(unset, args[i+1])
					i++
				case scriptEnvAssign(args[i]):
					sets = append(sets, args[i])
				default:
					goto parsed
				}
			}
		parsed:
			if len(unset) == 0 && len(sets) == 0 || i >= len(args) {
				return base.Run(s, args...)
			}
			return scriptRunEnv(s, args[i:], unset, sets, cancel, waitDelay)
		})
}

func scriptEnvAssign(arg string) bool {
	i := strings.IndexByte(arg, '=')
	if i <= 0 {
		return false
	}
	for j := 0; j < i; j++ {
		c := arg[j]
		if !(c == '_' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || j > 0 && '0' <= c && c <= '9') {
			return false
		}
	}
	return true
}

// scriptRunEnv runs prog args with the unset names removed and the
// NAME=value assignments applied to the script environment, portably.
func scriptRunEnv(s *script.State, argv []string, unset, sets []string, cancel func(*exec.Cmd) error, waitDelay time.Duration) (script.WaitFunc, error) {
	env := s.Environ()
	drop := func(env []string, name string) []string {
		out := env[:0]
		for _, e := range env {
			if !strings.HasPrefix(e, name+"=") {
				out = append(out, e)
			}
		}
		return out
	}
	for _, u := range unset {
		env = drop(env, u)
	}
	for _, set := range sets {
		env = drop(env, set[:strings.IndexByte(set, '=')])
		env = append(env, set)
	}
	name := filepath.FromSlash(argv[0])
	prog := name
	if !strings.Contains(name, string(filepath.Separator)) {
		p, err := scriptLookPath(s, name, env)
		if err != nil {
			return nil, err
		}
		prog = p
	}
	var stdoutBuf, stderrBuf strings.Builder
	cmd := exec.CommandContext(s.Context(), prog, argv[1:]...)
	if cancel != nil {
		cmd.Cancel = func() error { return cancel(cmd) }
	}
	cmd.WaitDelay = waitDelay
	cmd.Args[0] = name
	cmd.Dir = s.Getwd()
	cmd.Env = env
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return func(*script.State) (stdout, stderr string, err error) {
		err = cmd.Wait()
		return stdoutBuf.String(), stderrBuf.String(), err
	}, nil
}

// scriptLookPath finds a bare program name on the script's PATH,
// trying the platform executable extensions.
func scriptLookPath(s *script.State, name string, env []string) (string, error) {
	var pathEnv string
	for _, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			pathEnv = e[5:]
		}
	}
	exts := []string{""}
	if runtime.GOOS == "windows" {
		for _, e := range env {
			if strings.HasPrefix(e, "PATHEXT=") {
				exts = append(exts, strings.Split(strings.ToLower(e[8:]), ";")...)
			}
		}
	}
	for _, dir := range filepath.SplitList(pathEnv) {
		if dir == "" {
			dir = "."
		}
		for _, ext := range exts {
			p := filepath.Join(dir, name+ext)
			if info, err := os.Stat(p); err == nil && !info.IsDir() {
				return p, nil
			}
		}
	}
	return "", exec.ErrNotFound
}

// scriptExists wraps exists so `exists path/to/prog` also matches
// `prog.exe` on Windows, where script scripts name extension-less
// toolchains.
func scriptExists(base script.Cmd) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "check that files exist",
			Args:    "[-readonly] [-exec] file...",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if runtime.GOOS != "windows" {
				return base.Run(s, args...)
			}
			fixed := append([]string(nil), args...)
			for i, a := range fixed {
				if strings.HasPrefix(a, "-") || filepath.Ext(a) != "" {
					continue
				}
				p := s.Path(a)
				if _, err := os.Stat(p); err != nil {
					if _, err2 := os.Stat(p + ".exe"); err2 == nil {
						fixed[i] = a + ".exe"
					}
				}
			}
			return base.Run(s, fixed...)
		})
}
