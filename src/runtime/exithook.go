// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// addExitHook registers the specified function 'f' to be run at
// program termination (e.g. when someone invokes os.Exit(), or when
// main.main returns). Hooks are run in reverse order of registration:
// first hook added is the last one run.
//
// CAREFUL: the expectation is that addExitHook should only be called
// from a safe context (e.g. not an error/panic path or signal
// handler, preemption enabled, allocation allowed, write barriers
// allowed, etc), and that the exit function 'f' will be invoked under
// similar circumstances. That is the say, we are expecting that 'f'
// uses normal / high-level Go code as opposed to one of the more
// restricted dialects used for the trickier parts of the runtime.
func addExitHook(f func(), runOnNonZeroExit bool) {
	exitHooks.hooks = append(exitHooks.hooks, exitHook{f: f, runOnNonZeroExit: runOnNonZeroExit})
}

// exitHook stores a function to be run on program exit, registered
// by the utility runtime.addExitHook.
type exitHook struct {
	f                func() // func to run
	runOnNonZeroExit bool   // whether to run on non-zero exit code
}

// exitHooks stores state related to hook functions registered to
// run when program execution terminates.
var exitHooks struct {
	hooks            []exitHook
	runningExitHooks bool
}

// runExitHooks runs any registered exit hook functions (funcs
// previously registered using runtime.addExitHook). Here 'exitCode'
// is the status code being passed to os.Exit, or zero if the program
// is terminating normally without calling os.Exit.
func runExitHooks(exitCode int) {
	if exitHooks.runningExitHooks {
		throw("internal error: exit hook invoked exit")
	}
	exitHooks.runningExitHooks = true

	runExitHook := func(f func()) (caughtPanic bool) {
		defer func() {
			if x := recover(); x != nil {
				caughtPanic = true
			}
		}()
		f()
		return
	}

	finishPageTrace()
	for i := range exitHooks.hooks {
		h := exitHooks.hooks[len(exitHooks.hooks)-i-1]
		if exitCode != 0 && !h.runOnNonZeroExit {
			continue
		}
		if caughtPanic := runExitHook(h.f); caughtPanic {
			throw("internal error: exit hook invoked panic")
		}
	}
	exitHooks.hooks = nil
	exitHooks.runningExitHooks = false
}
