// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"log"
	"os/exec"
	"time"
)

// run runs a command with optional arguments.
func run(cmd *exec.Cmd, opts ...runOpt) error {
	a := runArgs{cmd, *cmdTimeout}
	for _, opt := range opts {
		opt.modArgs(&a)
	}
	if *verbose {
		log.Printf("running %v in %v", a.cmd.Args, a.cmd.Dir)
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	err := timeout(a.timeout, cmd.Wait)
	if _, ok := err.(timeoutError); ok {
		cmd.Process.Kill()
	}
	return err
}

// Zero or more runOpts can be passed to run to modify the command
// before it's run.
type runOpt interface {
	modArgs(*runArgs)
}

// allOutput sends both stdout and stderr to w.
func allOutput(w io.Writer) optFunc {
	return func(a *runArgs) {
		a.cmd.Stdout = w
		a.cmd.Stderr = w
	}
}

func runTimeout(timeout time.Duration) optFunc {
	return func(a *runArgs) {
		a.timeout = timeout
	}
}

func runDir(dir string) optFunc {
	return func(a *runArgs) {
		a.cmd.Dir = dir
	}
}

func runEnv(env []string) optFunc {
	return func(a *runArgs) {
		a.cmd.Env = env
	}
}

// timeout runs f and returns its error value, or if the function does not
// complete before the provided duration it returns a timeout error.
func timeout(d time.Duration, f func() error) error {
	errc := make(chan error, 1)
	go func() {
		errc <- f()
	}()
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return timeoutError(d)
	case err := <-errc:
		return err
	}
}

type timeoutError time.Duration

func (e timeoutError) Error() string {
	return fmt.Sprintf("timed out after %v", time.Duration(e))
}

// optFunc implements runOpt with a function, like http.HandlerFunc.
type optFunc func(*runArgs)

func (f optFunc) modArgs(a *runArgs) { f(a) }

// internal detail to exec.go:
type runArgs struct {
	cmd     *exec.Cmd
	timeout time.Duration
}
