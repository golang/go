// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gocommand is a helper for calling the go command.
package gocommand

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
)

// An Runner will run go command invocations and serialize
// them if it sees a concurrency error.
type Runner struct {
	// LoadMu guards packages.Load calls and associated state.
	loadMu sync.Mutex
	// locked is true when we hold the mutex and have incremented.
	locked         bool
	serializeLoads int
}

// 1.13: go: updates to go.mod needed, but contents have changed
// 1.14: go: updating go.mod: existing contents have changed since last read
var modConcurrencyError = regexp.MustCompile(`go:.*go.mod.*contents have changed`)

// Run is a convenience wrapper around RunRaw.
// It returns only stdout and a "friendly" error.
func (runner *Runner) Run(ctx context.Context, inv Invocation) (*bytes.Buffer, error) {
	stdout, _, friendly, _ := runner.runRaw(ctx, inv)
	return stdout, friendly
}

// RunPiped runs the invocation serially, always waiting for any concurrent
// invocations to complete first.
func (runner *Runner) RunPiped(ctx context.Context, inv Invocation, stdout, stderr io.Writer) error {
	_, err := runner.runPiped(ctx, inv, stdout, stderr)
	return err
}

// RunRaw runs the invocation, serializing requests only if they fight over
// go.mod changes.
func (runner *Runner) RunRaw(ctx context.Context, inv Invocation) (*bytes.Buffer, *bytes.Buffer, error, error) {
	return runner.runRaw(ctx, inv)
}

func (runner *Runner) runPiped(ctx context.Context, inv Invocation, stdout, stderr io.Writer) (error, error) {
	runner.loadMu.Lock()
	runner.serializeLoads++
	runner.locked = true

	defer func() {
		runner.locked = false
		runner.serializeLoads--
		runner.loadMu.Unlock()
	}()

	return inv.runWithFriendlyError(ctx, stdout, stderr)
}

func (runner *Runner) runRaw(ctx context.Context, inv Invocation) (*bytes.Buffer, *bytes.Buffer, error, error) {
	// We want to run invocations concurrently as much as possible. However,
	// if go.mod updates are needed, only one can make them and the others will
	// fail. We need to retry in those cases, but we don't want to thrash so
	// badly we never recover. To avoid that, once we've seen one concurrency
	// error, start serializing everything until the backlog has cleared out.
	runner.loadMu.Lock()
	if runner.serializeLoads == 0 {
		runner.loadMu.Unlock()
	} else {
		runner.locked = true
		runner.serializeLoads++
	}
	defer func() {
		if runner.locked {
			runner.locked = false
			runner.serializeLoads--
			runner.loadMu.Unlock()
		}
	}()

	for {
		stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
		friendlyErr, err := inv.runWithFriendlyError(ctx, stdout, stderr)
		if friendlyErr == nil || !modConcurrencyError.MatchString(friendlyErr.Error()) {
			return stdout, stderr, friendlyErr, err
		}
		event.Error(ctx, "Load concurrency error, will retry serially", err)
		if !runner.locked {
			runner.loadMu.Lock()
			runner.serializeLoads++
			runner.locked = true
		}
	}
}

// An Invocation represents a call to the go command.
type Invocation struct {
	Verb       string
	Args       []string
	BuildFlags []string
	Env        []string
	WorkingDir string
	Logf       func(format string, args ...interface{})
}

func (i *Invocation) runWithFriendlyError(ctx context.Context, stdout, stderr io.Writer) (friendlyError error, rawError error) {
	rawError = i.run(ctx, stdout, stderr)
	if rawError != nil {
		friendlyError = rawError
		// Check for 'go' executable not being found.
		if ee, ok := rawError.(*exec.Error); ok && ee.Err == exec.ErrNotFound {
			friendlyError = fmt.Errorf("go command required, not found: %v", ee)
		}
		if ctx.Err() != nil {
			friendlyError = ctx.Err()
		}
		friendlyError = fmt.Errorf("err: %v: stderr: %s", friendlyError, stderr)
	}
	return
}

func (i *Invocation) run(ctx context.Context, stdout, stderr io.Writer) error {
	log := i.Logf
	if log == nil {
		log = func(string, ...interface{}) {}
	}

	goArgs := []string{i.Verb}
	switch i.Verb {
	case "mod":
		// mod needs the sub-verb before build flags.
		goArgs = append(goArgs, i.Args[0])
		goArgs = append(goArgs, i.BuildFlags...)
		goArgs = append(goArgs, i.Args[1:]...)
	case "env":
		// env doesn't take build flags.
		goArgs = append(goArgs, i.Args...)
	default:
		goArgs = append(goArgs, i.BuildFlags...)
		goArgs = append(goArgs, i.Args...)
	}
	cmd := exec.Command("go", goArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	// On darwin the cwd gets resolved to the real path, which breaks anything that
	// expects the working directory to keep the original path, including the
	// go command when dealing with modules.
	// The Go stdlib has a special feature where if the cwd and the PWD are the
	// same node then it trusts the PWD, so by setting it in the env for the child
	// process we fix up all the paths returned by the go command.
	cmd.Env = append(os.Environ(), i.Env...)
	if i.WorkingDir != "" {
		cmd.Env = append(cmd.Env, "PWD="+i.WorkingDir)
		cmd.Dir = i.WorkingDir
	}

	defer func(start time.Time) { log("%s for %v", time.Since(start), cmdDebugStr(cmd)) }(time.Now())

	return runCmdContext(ctx, cmd)
}

// runCmdContext is like exec.CommandContext except it sends os.Interrupt
// before os.Kill.
func runCmdContext(ctx context.Context, cmd *exec.Cmd) error {
	if err := cmd.Start(); err != nil {
		return err
	}
	resChan := make(chan error, 1)
	go func() {
		resChan <- cmd.Wait()
	}()

	select {
	case err := <-resChan:
		return err
	case <-ctx.Done():
	}
	// Cancelled. Interrupt and see if it ends voluntarily.
	cmd.Process.Signal(os.Interrupt)
	select {
	case err := <-resChan:
		return err
	case <-time.After(time.Second):
	}
	// Didn't shut down in response to interrupt. Kill it hard.
	cmd.Process.Kill()
	return <-resChan
}

func cmdDebugStr(cmd *exec.Cmd) string {
	env := make(map[string]string)
	for _, kv := range cmd.Env {
		split := strings.Split(kv, "=")
		k, v := split[0], split[1]
		env[k] = v
	}

	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v GOPROXY=%v PWD=%v go %v", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["GOPROXY"], env["PWD"], cmd.Args)
}
