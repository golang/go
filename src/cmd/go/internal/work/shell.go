// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"cmd/go/internal/par"
	"fmt"
	"os"
	"sync"
)

// A Shell runs shell commands and performs shell-like file system operations.
//
// Shell tracks context related to running commands, and form a tree much like
// context.Context.
//
// TODO: Add a RemoveAll method. "rm -rf" is pretty common.
type Shell struct {
	action       *Action // nil for the root shell
	*shellShared         // per-Builder state shared across Shells
}

// shellShared is Shell state shared across all Shells derived from a single
// root shell (generally a single Builder).
type shellShared struct {
	workDir string // $WORK, immutable

	printLock sync.Mutex
	printFunc func(args ...any) (int, error)
	scriptDir string // current directory in printed script

	mkdirCache par.Cache[string, error] // a cache of created directories
}

// NewShell returns a new Shell.
//
// Shell will internally serialize calls to the print function.
// If print is nil, it defaults to printing to stderr.
func NewShell(workDir string, print func(a ...any) (int, error)) *Shell {
	if print == nil {
		print = func(a ...any) (int, error) {
			return fmt.Fprint(os.Stderr, a...)
		}
	}
	shared := &shellShared{
		workDir:   workDir,
		printFunc: print,
	}
	return &Shell{shellShared: shared}
}

// Print emits a to this Shell's output stream, formatting it like fmt.Print.
// It is safe to call concurrently.
func (sh *Shell) Print(a ...any) {
	sh.printLock.Lock()
	defer sh.printLock.Unlock()
	sh.printFunc(a...)
}

func (sh *Shell) printLocked(a ...any) {
	sh.printFunc(a...)
}

// WithAction returns a Shell identical to sh, but bound to Action a.
func (sh *Shell) WithAction(a *Action) *Shell {
	sh2 := *sh
	sh2.action = a
	return &sh2
}

// Shell returns a shell for running commands on behalf of Action a.
func (b *Builder) Shell(a *Action) *Shell {
	if a == nil {
		// The root shell has a nil Action. The point of this method is to
		// create a Shell bound to an Action, so disallow nil Actions here.
		panic("nil Action")
	}
	if a.sh == nil {
		a.sh = b.backgroundSh.WithAction(a)
	}
	return a.sh
}

// BackgroundShell returns a Builder-wide Shell that's not bound to any Action.
// Try not to use this unless there's really no sensible Action available.
func (b *Builder) BackgroundShell() *Shell {
	return b.backgroundSh
}
