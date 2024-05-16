// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/go/internal/script"
	"cmd/go/internal/script/scripttest"
	"cmd/go/internal/work"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

func scriptCommands(interrupt os.Signal, waitDelay time.Duration) map[string]script.Cmd {
	cmds := scripttest.DefaultCmds()

	// Customize the "exec" interrupt signal and grace period.
	var cancel func(cmd *exec.Cmd) error
	if interrupt != nil {
		cancel = func(cmd *exec.Cmd) error {
			return cmd.Process.Signal(interrupt)
		}
	}

	cmdExec := script.Exec(cancel, waitDelay)
	cmds["exec"] = cmdExec

	add := func(name string, cmd script.Cmd) {
		if _, ok := cmds[name]; ok {
			panic(fmt.Sprintf("command %q is already registered", name))
		}
		cmds[name] = cmd
	}

	add("cc", scriptCC(cmdExec))
	cmdGo := scriptGo(cancel, waitDelay)
	add("go", cmdGo)
	add("stale", scriptStale(cmdGo))

	return cmds
}

// scriptCC runs the C compiler along with platform specific options.
func scriptCC(cmdExec script.Cmd) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "run the platform C compiler",
			Args:    "args...",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			b := work.NewBuilder(s.Getwd())
			wait, err := cmdExec.Run(s, append(b.GccCmd(".", ""), args...)...)
			if err != nil {
				return wait, err
			}
			waitAndClean := func(s *script.State) (stdout, stderr string, err error) {
				stdout, stderr, err = wait(s)
				if closeErr := b.Close(); err == nil {
					err = closeErr
				}
				return stdout, stderr, err
			}
			return waitAndClean, nil
		})
}

var scriptGoInvoked sync.Map // testing.TB → go command was invoked

// scriptGo runs the go command.
func scriptGo(cancel func(*exec.Cmd) error, waitDelay time.Duration) script.Cmd {
	cmd := script.Program(testGo, cancel, waitDelay)
	// Inject code to update scriptGoInvoked before invoking the Go command.
	return script.Command(*cmd.Usage(), func(state *script.State, s ...string) (script.WaitFunc, error) {
		t, ok := tbFromContext(state.Context())
		if !ok {
			return nil, errors.New("script Context unexpectedly missing testing.TB key")
		}
		_, dup := scriptGoInvoked.LoadOrStore(t, true)
		if !dup {
			t.Cleanup(func() { scriptGoInvoked.Delete(t) })
		}
		return cmd.Run(state, s...)
	})
}

// scriptStale checks that the named build targets are stale.
func scriptStale(cmdGo script.Cmd) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "check that build targets are stale",
			Args:    "target...",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) == 0 {
				return nil, script.ErrUsage
			}
			tmpl := "{{if .Error}}{{.ImportPath}}: {{.Error.Err}}" +
				"{{else}}{{if not .Stale}}{{.ImportPath}} ({{.Target}}) is not stale{{end}}" +
				"{{end}}"

			wait, err := cmdGo.Run(s, append([]string{"list", "-e", "-f=" + tmpl}, args...)...)
			if err != nil {
				return nil, err
			}

			stdout, stderr, err := wait(s)
			if len(stderr) != 0 {
				s.Logf("%s", stderr)
			}
			if err != nil {
				return nil, err
			}
			if out := strings.TrimSpace(stdout); out != "" {
				return nil, errors.New(out)
			}
			return nil, nil
		})
}
