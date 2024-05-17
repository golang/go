// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package exithook provides limited support for on-exit cleanup.
//
// CAREFUL! The expectation is that Add should only be called
// from a safe context (e.g. not an error/panic path or signal
// handler, preemption enabled, allocation allowed, write barriers
// allowed, etc), and that the exit function F will be invoked under
// similar circumstances. That is the say, we are expecting that F
// uses normal / high-level Go code as opposed to one of the more
// restricted dialects used for the trickier parts of the runtime.
package exithook

// A Hook is a function to be run at program termination
// (when someone invokes os.Exit, or when main.main returns).
// Hooks are run in reverse order of registration:
// the first hook added is the last one run.
type Hook struct {
	F            func() // func to run
	RunOnFailure bool   // whether to run on non-zero exit code
}

var (
	hooks   []Hook
	running bool
)

// Add adds a new exit hook.
func Add(h Hook) {
	hooks = append(hooks, h)
}

// Run runs the exit hooks.
// It returns an error if Run is already running or
// if one of the hooks panics.
func Run(code int) (err error) {
	if running {
		return exitError("exit hook invoked exit")
	}
	running = true

	defer func() {
		if x := recover(); x != nil {
			err = exitError("exit hook invoked panic")
		}
	}()

	local := hooks
	hooks = nil
	for i := len(local) - 1; i >= 0; i-- {
		h := local[i]
		if code == 0 || h.RunOnFailure {
			h.F()
		}
	}
	running = false
	return nil
}

type exitError string

func (e exitError) Error() string { return string(e) }
