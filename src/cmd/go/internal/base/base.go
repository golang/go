// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package base defines shared basic pieces of the go command,
// in particular logging and the Command structure.
package base

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/str"
)

// A Command is an implementation of a go command
// like go build or go fix.
type Command struct {
	// Run runs the command.
	// The args are the arguments after the command name.
	Run func(ctx context.Context, cmd *Command, args []string)

	// UsageLine is the one-line usage message.
	// The words between "go" and the first flag or argument in the line are taken to be the command name.
	UsageLine string

	// Short is the short description shown in the 'go help' output.
	Short string

	// Long is the long message shown in the 'go help <this-command>' output.
	Long string

	// Flag is a set of flags specific to this command.
	Flag flag.FlagSet

	// CustomFlags indicates that the command will do its own
	// flag parsing.
	CustomFlags bool

	// Commands lists the available commands and help topics.
	// The order here is the order in which they are printed by 'go help'.
	// Note that subcommands are in general best avoided.
	Commands []*Command
}

var Go = &Command{
	UsageLine: "go",
	Long:      `Go is a tool for managing Go source code.`,
	// Commands initialized in package main
}

// Lookup returns the subcommand with the given name, if any.
// Otherwise it returns nil.
//
// Lookup ignores subcommands that have len(c.Commands) == 0 and c.Run == nil.
// Such subcommands are only for use as arguments to "help".
func (c *Command) Lookup(name string) *Command {
	for _, sub := range c.Commands {
		if sub.Name() == name && (len(c.Commands) > 0 || c.Runnable()) {
			return sub
		}
	}
	return nil
}

// hasFlag reports whether a command or any of its subcommands contain the given
// flag.
func hasFlag(c *Command, name string) bool {
	if f := c.Flag.Lookup(name); f != nil {
		return true
	}
	for _, sub := range c.Commands {
		if hasFlag(sub, name) {
			return true
		}
	}
	return false
}

// LongName returns the command's long name: all the words in the usage line between "go" and a flag or argument,
func (c *Command) LongName() string {
	name := c.UsageLine
	if i := strings.Index(name, " ["); i >= 0 {
		name = name[:i]
	}
	if name == "go" {
		return ""
	}
	return strings.TrimPrefix(name, "go ")
}

// Name returns the command's short name: the last word in the usage line before a flag or argument.
func (c *Command) Name() string {
	name := c.LongName()
	if i := strings.LastIndex(name, " "); i >= 0 {
		name = name[i+1:]
	}
	return name
}

func (c *Command) Usage() {
	fmt.Fprintf(os.Stderr, "usage: %s\n", c.UsageLine)
	fmt.Fprintf(os.Stderr, "Run 'go help %s' for details.\n", c.LongName())
	SetExitStatus(2)
	Exit()
}

// Runnable reports whether the command can be run; otherwise
// it is a documentation pseudo-command such as importpath.
func (c *Command) Runnable() bool {
	return c.Run != nil
}

var atExitFuncs []func()

func AtExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

func Exit() {
	for _, f := range atExitFuncs {
		f()
	}
	os.Exit(exitStatus)
}

func Fatalf(format string, args ...any) {
	Errorf(format, args...)
	Exit()
}

func Errorf(format string, args ...any) {
	log.Printf(format, args...)
	SetExitStatus(1)
}

func ExitIfErrors() {
	if exitStatus != 0 {
		Exit()
	}
}

func Error(err error) {
	// We use errors.Join to return multiple errors from various routines.
	// If we receive multiple errors joined with a basic errors.Join,
	// handle each one separately so that they all have the leading "go: " prefix.
	// A plain interface check is not good enough because there might be
	// other kinds of structured errors that are logically one unit and that
	// add other context: only handling the wrapped errors would lose
	// that context.
	if err != nil && reflect.TypeOf(err).String() == "*errors.joinError" {
		for _, e := range err.(interface{ Unwrap() []error }).Unwrap() {
			Error(e)
		}
		return
	}
	Errorf("go: %v", err)
}

func Fatal(err error) {
	Error(err)
	Exit()
}

var exitStatus = 0
var exitMu sync.Mutex

func SetExitStatus(n int) {
	exitMu.Lock()
	if exitStatus < n {
		exitStatus = n
	}
	exitMu.Unlock()
}

func GetExitStatus() int {
	return exitStatus
}

// Run runs the command, with stdout and stderr
// connected to the go command's own stdout and stderr.
// If the command fails, Run reports the error using Errorf.
func Run(cmdargs ...any) {
	if err := RunErr(cmdargs...); err != nil {
		Errorf("%v", err)
	}
}

// Run runs the command, with stdout and stderr
// connected to the go command's own stdout and stderr.
// If the command fails, RunErr returns the error, which
// may be an *exec.ExitError.
func RunErr(cmdargs ...any) error {
	cmdline := str.StringList(cmdargs...)
	if cfg.BuildN || cfg.BuildX {
		fmt.Printf("%s\n", strings.Join(cmdline, " "))
		if cfg.BuildN {
			return nil
		}
	}

	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// RunStdin is like run but connects Stdin. It retries if it encounters an ETXTBSY.
func RunStdin(cmdline []string) {
	env := slices.Clip(cfg.OrigEnv)
	env = AppendPATH(env)
	for try := range 3 {
		cmd := exec.Command(cmdline[0], cmdline[1:]...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Env = env
		StartSigHandlers()
		err := cmd.Run()
		if err == nil {
			break // success
		}

		if !IsETXTBSY(err) {
			Errorf("%v", err)
			break // failure
		}

		// The error was an ETXTBSY. Sleep and try again. It's possible that
		// another go command instance was racing against us to write the executable
		// to the executable cache. In that case it may still have the file open, and
		// we may get an ETXTBSY. That should resolve once that process closes the file
		// so attempt a couple more times. See the discussion in #22220 and also
		// (*runTestActor).Act in cmd/go/internal/test, which does something similar.
		time.Sleep(100 * time.Millisecond << uint(try))
	}
}

// Usage is the usage-reporting function, filled in by package main
// but here for reference by other packages.
var Usage func()
