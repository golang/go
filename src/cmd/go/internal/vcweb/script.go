// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"bufio"
	"bytes"
	"cmd/internal/script"
	"context"
	"errors"
	"fmt"
	"internal/txtar"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"golang.org/x/mod/module"
	"golang.org/x/mod/zip"
)

// newScriptEngine returns a script engine augmented with commands for
// reproducing version-control repositories by replaying commits.
func newScriptEngine() *script.Engine {
	conds := script.DefaultConds()

	interrupt := func(cmd *exec.Cmd) error { return cmd.Process.Signal(os.Interrupt) }
	gracePeriod := 30 * time.Second // arbitrary

	cmds := script.DefaultCmds()
	cmds["at"] = scriptAt()
	cmds["bzr"] = script.Program("bzr", interrupt, gracePeriod)
	cmds["fossil"] = script.Program("fossil", interrupt, gracePeriod)
	cmds["git"] = script.Program("git", interrupt, gracePeriod)
	cmds["hg"] = script.Program("hg", interrupt, gracePeriod)
	cmds["handle"] = scriptHandle()
	cmds["modzip"] = scriptModzip()
	cmds["svnadmin"] = script.Program("svnadmin", interrupt, gracePeriod)
	cmds["svn"] = script.Program("svn", interrupt, gracePeriod)
	cmds["unquote"] = scriptUnquote()

	return &script.Engine{
		Cmds:  cmds,
		Conds: conds,
	}
}

// loadScript interprets the given script content using the vcweb script engine.
// loadScript always returns either a non-nil handler or a non-nil error.
//
// The script content must be a txtar archive with a comment containing a script
// with exactly one "handle" command and zero or more VCS commands to prepare
// the repository to be served.
func (s *Server) loadScript(ctx context.Context, logger *log.Logger, scriptPath string, scriptContent []byte, workDir string) (http.Handler, error) {
	ar := txtar.Parse(scriptContent)

	if err := os.MkdirAll(workDir, 0755); err != nil {
		return nil, err
	}

	st, err := s.newState(ctx, workDir)
	if err != nil {
		return nil, err
	}
	if err := st.ExtractFiles(ar); err != nil {
		return nil, err
	}

	scriptName := filepath.Base(scriptPath)
	scriptLog := new(strings.Builder)
	err = s.engine.Execute(st, scriptName, bufio.NewReader(bytes.NewReader(ar.Comment)), scriptLog)
	closeErr := st.CloseAndWait(scriptLog)
	logger.Printf("%s:", scriptName)
	io.WriteString(logger.Writer(), scriptLog.String())
	io.WriteString(logger.Writer(), "\n")
	if err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, err
	}

	sc, err := getScriptCtx(st)
	if err != nil {
		return nil, err
	}
	if sc.handler == nil {
		return nil, errors.New("script completed without setting handler")
	}
	return sc.handler, nil
}

// newState returns a new script.State for executing scripts in workDir.
func (s *Server) newState(ctx context.Context, workDir string) (*script.State, error) {
	ctx = &scriptCtx{
		Context: ctx,
		server:  s,
	}

	st, err := script.NewState(ctx, workDir, s.env)
	if err != nil {
		return nil, err
	}
	return st, nil
}

// scriptEnviron returns a new environment that attempts to provide predictable
// behavior for the supported version-control tools.
func scriptEnviron(homeDir string) []string {
	env := []string{
		"USER=gopher",
		homeEnvName() + "=" + homeDir,
		"GIT_CONFIG_NOSYSTEM=1",
		"HGRCPATH=" + filepath.Join(homeDir, ".hgrc"),
		"HGENCODING=utf-8",
	}
	// Preserve additional environment variables that may be needed by VCS tools.
	for _, k := range []string{
		pathEnvName(),
		tempEnvName(),
		"SYSTEMROOT",        // must be preserved on Windows to find DLLs; golang.org/issue/25210
		"WINDIR",            // must be preserved on Windows to be able to run PowerShell command; golang.org/issue/30711
		"ComSpec",           // must be preserved on Windows to be able to run Batch files; golang.org/issue/56555
		"DYLD_LIBRARY_PATH", // must be preserved on macOS systems to find shared libraries
		"LD_LIBRARY_PATH",   // must be preserved on Unix systems to find shared libraries
		"LIBRARY_PATH",      // allow override of non-standard static library paths
		"PYTHONPATH",        // may be needed by hg to find imported modules
	} {
		if v, ok := os.LookupEnv(k); ok {
			env = append(env, k+"="+v)
		}
	}

	if os.Getenv("GO_BUILDER_NAME") != "" || os.Getenv("GIT_TRACE_CURL") == "1" {
		// To help diagnose https://go.dev/issue/52545,
		// enable tracing for Git HTTPS requests.
		env = append(env,
			"GIT_TRACE_CURL=1",
			"GIT_TRACE_CURL_NO_DATA=1",
			"GIT_REDACT_COOKIES=o,SSO,GSSO_Uberproxy")
	}

	return env
}

// homeEnvName returns the environment variable used by os.UserHomeDir
// to locate the user's home directory.
func homeEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "USERPROFILE"
	case "plan9":
		return "home"
	default:
		return "HOME"
	}
}

// tempEnvName returns the environment variable used by os.TempDir
// to locate the default directory for temporary files.
func tempEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "TMP"
	case "plan9":
		return "TMPDIR" // actually plan 9 doesn't have one at all but this is fine
	default:
		return "TMPDIR"
	}
}

// pathEnvName returns the environment variable used by exec.LookPath to
// identify directories to search for executables.
func pathEnvName() string {
	switch runtime.GOOS {
	case "plan9":
		return "path"
	default:
		return "PATH"
	}
}

// A scriptCtx is a context.Context that stores additional state for script
// commands.
type scriptCtx struct {
	context.Context
	server      *Server
	commitTime  time.Time
	handlerName string
	handler     http.Handler
}

// scriptCtxKey is the key associating the *scriptCtx in a script's Context..
type scriptCtxKey struct{}

func (sc *scriptCtx) Value(key any) any {
	if key == (scriptCtxKey{}) {
		return sc
	}
	return sc.Context.Value(key)
}

func getScriptCtx(st *script.State) (*scriptCtx, error) {
	sc, ok := st.Context().Value(scriptCtxKey{}).(*scriptCtx)
	if !ok {
		return nil, errors.New("scriptCtx not found in State.Context")
	}
	return sc, nil
}

func scriptAt() script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "set the current commit time for all version control systems",
			Args:    "time",
			Detail: []string{
				"The argument must be an absolute timestamp in RFC3339 format.",
			},
		},
		func(st *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) != 1 {
				return nil, script.ErrUsage
			}

			sc, err := getScriptCtx(st)
			if err != nil {
				return nil, err
			}

			sc.commitTime, err = time.ParseInLocation(time.RFC3339, args[0], time.UTC)
			if err == nil {
				st.Setenv("GIT_COMMITTER_DATE", args[0])
				st.Setenv("GIT_AUTHOR_DATE", args[0])
			}
			return nil, err
		})
}

func scriptHandle() script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "set the HTTP handler that will serve the script's output",
			Args:    "handler [dir]",
			Detail: []string{
				"The handler will be passed the script's current working directory and environment as arguments.",
				"Valid handlers include 'dir' (for general http.Dir serving), 'bzr', 'fossil', 'git', and 'hg'",
			},
		},
		func(st *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) == 0 || len(args) > 2 {
				return nil, script.ErrUsage
			}

			sc, err := getScriptCtx(st)
			if err != nil {
				return nil, err
			}

			if sc.handler != nil {
				return nil, fmt.Errorf("server handler already set to %s", sc.handlerName)
			}

			name := args[0]
			h, ok := sc.server.vcsHandlers[name]
			if !ok {
				return nil, fmt.Errorf("unrecognized VCS %q", name)
			}
			sc.handlerName = name
			if !h.Available() {
				return nil, ServerNotInstalledError{name}
			}

			dir := st.Getwd()
			if len(args) >= 2 {
				dir = st.Path(args[1])
			}
			sc.handler, err = h.Handler(dir, st.Environ(), sc.server.logger)
			return nil, err
		})
}

func scriptModzip() script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "create a Go module zip file from a directory",
			Args:    "zipfile path@version dir",
		},
		func(st *script.State, args ...string) (wait script.WaitFunc, err error) {
			if len(args) != 3 {
				return nil, script.ErrUsage
			}
			zipPath := st.Path(args[0])
			mPath, version, ok := strings.Cut(args[1], "@")
			if !ok {
				return nil, script.ErrUsage
			}
			dir := st.Path(args[2])

			if err := os.MkdirAll(filepath.Dir(zipPath), 0755); err != nil {
				return nil, err
			}
			f, err := os.Create(zipPath)
			if err != nil {
				return nil, err
			}
			defer func() {
				if closeErr := f.Close(); err == nil {
					err = closeErr
				}
			}()

			return nil, zip.CreateFromDir(f, module.Version{Path: mPath, Version: version}, dir)
		})
}

func scriptUnquote() script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "unquote the argument as a Go string",
			Args:    "string",
		},
		func(st *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) != 1 {
				return nil, script.ErrUsage
			}

			s, err := strconv.Unquote(`"` + args[0] + `"`)
			if err != nil {
				return nil, err
			}

			wait := func(*script.State) (stdout, stderr string, err error) {
				return s, "", nil
			}
			return wait, nil
		})
}
