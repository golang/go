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

	"golang.org/x/tools/internal/telemetry/event"
)

// An Runner will run go command invocations and serialize
// them if it sees a concurrency error.
type Runner struct {
	// LoadMu guards packages.Load calls and associated state.
	loadMu         sync.Mutex
	serializeLoads int
}

// 1.13: go: updates to go.mod needed, but contents have changed
// 1.14: go: updating go.mod: existing contents have changed since last read
var modConcurrencyError = regexp.MustCompile(`go:.*go.mod.*contents have changed`)

// Run calls Runner.RunRaw, serializing requests if they fight over
// go.mod changes.
func (runner *Runner) Run(ctx context.Context, inv Invocation) (*bytes.Buffer, error) {
	stdout, _, friendly, _ := runner.RunRaw(ctx, inv)
	return stdout, friendly
}

// Run calls Innvocation.RunRaw, serializing requests if they fight over
// go.mod changes.
func (runner *Runner) RunRaw(ctx context.Context, inv Invocation) (*bytes.Buffer, *bytes.Buffer, error, error) {
	// We want to run invocations concurrently as much as possible. However,
	// if go.mod updates are needed, only one can make them and the others will
	// fail. We need to retry in those cases, but we don't want to thrash so
	// badly we never recover. To avoid that, once we've seen one concurrency
	// error, start serializing everything until the backlog has cleared out.
	runner.loadMu.Lock()
	var locked bool // If true, we hold the mutex and have incremented.
	if runner.serializeLoads == 0 {
		runner.loadMu.Unlock()
	} else {
		locked = true
		runner.serializeLoads++
	}
	defer func() {
		if locked {
			runner.serializeLoads--
			runner.loadMu.Unlock()
		}
	}()

	for {
		stdout, stderr, friendlyErr, err := inv.runRaw(ctx)
		if friendlyErr == nil || !modConcurrencyError.MatchString(friendlyErr.Error()) {
			return stdout, stderr, friendlyErr, err
		}
		event.Error(ctx, "Load concurrency error, will retry serially", err)
		if !locked {
			runner.loadMu.Lock()
			runner.serializeLoads++
			locked = true
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

// RunRaw is like RunPiped, but also returns the raw stderr and error for callers
// that want to do low-level error handling/recovery.
func (i *Invocation) runRaw(ctx context.Context) (stdout *bytes.Buffer, stderr *bytes.Buffer, friendlyError error, rawError error) {
	stdout = &bytes.Buffer{}
	stderr = &bytes.Buffer{}
	rawError = i.RunPiped(ctx, stdout, stderr)
	if rawError != nil {
		// Check for 'go' executable not being found.
		if ee, ok := rawError.(*exec.Error); ok && ee.Err == exec.ErrNotFound {
			friendlyError = fmt.Errorf("go command required, not found: %v", ee)
		}
		if ctx.Err() != nil {
			friendlyError = ctx.Err()
		}
		friendlyError = fmt.Errorf("err: %v: stderr: %s", rawError, stderr)
	}
	return
}

// RunPiped is like Run, but relies on the given stdout/stderr
func (i *Invocation) RunPiped(ctx context.Context, stdout, stderr io.Writer) error {
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
