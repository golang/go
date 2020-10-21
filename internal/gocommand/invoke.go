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
	// once guards the runner initialization.
	once sync.Once

	// inFlight tracks available workers.
	inFlight chan struct{}

	// serialized guards the ability to run a go command serially,
	// to avoid deadlocks when claiming workers.
	serialized chan struct{}
}

const maxInFlight = 10

func (runner *Runner) initialize() {
	runner.once.Do(func() {
		runner.inFlight = make(chan struct{}, maxInFlight)
		runner.serialized = make(chan struct{}, 1)
	})
}

// 1.13: go: updates to go.mod needed, but contents have changed
// 1.14: go: updating go.mod: existing contents have changed since last read
var modConcurrencyError = regexp.MustCompile(`go:.*go.mod.*contents have changed`)

// Run is a convenience wrapper around RunRaw.
// It returns only stdout and a "friendly" error.
func (runner *Runner) Run(ctx context.Context, inv Invocation) (*bytes.Buffer, error) {
	stdout, _, friendly, _ := runner.RunRaw(ctx, inv)
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
	// Make sure the runner is always initialized.
	runner.initialize()

	// First, try to run the go command concurrently.
	stdout, stderr, friendlyErr, err := runner.runConcurrent(ctx, inv)

	// If we encounter a load concurrency error, we need to retry serially.
	if friendlyErr == nil || !modConcurrencyError.MatchString(friendlyErr.Error()) {
		return stdout, stderr, friendlyErr, err
	}
	event.Error(ctx, "Load concurrency error, will retry serially", err)

	// Run serially by calling runPiped.
	stdout.Reset()
	stderr.Reset()
	friendlyErr, err = runner.runPiped(ctx, inv, stdout, stderr)
	return stdout, stderr, friendlyErr, err
}

func (runner *Runner) runConcurrent(ctx context.Context, inv Invocation) (*bytes.Buffer, *bytes.Buffer, error, error) {
	// Wait for 1 worker to become available.
	select {
	case <-ctx.Done():
		return nil, nil, nil, ctx.Err()
	case runner.inFlight <- struct{}{}:
		defer func() { <-runner.inFlight }()
	}

	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	friendlyErr, err := inv.runWithFriendlyError(ctx, stdout, stderr)
	return stdout, stderr, friendlyErr, err
}

func (runner *Runner) runPiped(ctx context.Context, inv Invocation, stdout, stderr io.Writer) (error, error) {
	// Make sure the runner is always initialized.
	runner.initialize()

	// Acquire the serialization lock. This avoids deadlocks between two
	// runPiped commands.
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case runner.serialized <- struct{}{}:
		defer func() { <-runner.serialized }()
	}

	// Wait for all in-progress go commands to return before proceeding,
	// to avoid load concurrency errors.
	for i := 0; i < maxInFlight; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case runner.inFlight <- struct{}{}:
			// Make sure we always "return" any workers we took.
			defer func() { <-runner.inFlight }()
		}
	}

	return inv.runWithFriendlyError(ctx, stdout, stderr)
}

// An Invocation represents a call to the go command.
type Invocation struct {
	Verb       string
	Args       []string
	BuildFlags []string
	ModFlag    string
	ModFile    string
	Overlay    string
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

	appendModFile := func() {
		if i.ModFile != "" {
			goArgs = append(goArgs, "-modfile="+i.ModFile)
		}
	}
	appendModFlag := func() {
		if i.ModFlag != "" {
			goArgs = append(goArgs, "-mod="+i.ModFlag)
		}
	}
	appendOverlayFlag := func() {
		if i.Overlay != "" {
			goArgs = append(goArgs, "-overlay="+i.Overlay)
		}
	}

	switch i.Verb {
	case "env", "version":
		goArgs = append(goArgs, i.Args...)
	case "mod":
		// mod needs the sub-verb before flags.
		goArgs = append(goArgs, i.Args[0])
		appendModFile()
		goArgs = append(goArgs, i.Args[1:]...)
	case "get":
		goArgs = append(goArgs, i.BuildFlags...)
		appendModFile()
		goArgs = append(goArgs, i.Args...)

	default: // notably list and build.
		goArgs = append(goArgs, i.BuildFlags...)
		appendModFile()
		appendModFlag()
		appendOverlayFlag()
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
