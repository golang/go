// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gocommand is a helper for calling the go command.
package gocommand

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	exec "golang.org/x/sys/execabs"

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

	// If ModFlag is set, the go command is invoked with -mod=ModFlag.
	ModFlag string

	// If ModFile is set, the go command is invoked with -modfile=ModFile.
	ModFile string

	// If Overlay is set, the go command is invoked with -overlay=Overlay.
	Overlay string

	// If CleanEnv is set, the invocation will run only with the environment
	// in Env, not starting with os.Environ.
	CleanEnv   bool
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

	// cmd.WaitDelay was added only in go1.20 (see #50436).
	if waitDelay := reflect.ValueOf(cmd).Elem().FieldByName("WaitDelay"); waitDelay.IsValid() {
		// https://go.dev/issue/59541: don't wait forever copying stderr
		// after the command has exited.
		// After CL 484741 we copy stdout manually, so we we'll stop reading that as
		// soon as ctx is done. However, we also don't want to wait around forever
		// for stderr. Give a much-longer-than-reasonable delay and then assume that
		// something has wedged in the kernel or runtime.
		waitDelay.Set(reflect.ValueOf(30 * time.Second))
	}

	// On darwin the cwd gets resolved to the real path, which breaks anything that
	// expects the working directory to keep the original path, including the
	// go command when dealing with modules.
	// The Go stdlib has a special feature where if the cwd and the PWD are the
	// same node then it trusts the PWD, so by setting it in the env for the child
	// process we fix up all the paths returned by the go command.
	if !i.CleanEnv {
		cmd.Env = os.Environ()
	}
	cmd.Env = append(cmd.Env, i.Env...)
	if i.WorkingDir != "" {
		cmd.Env = append(cmd.Env, "PWD="+i.WorkingDir)
		cmd.Dir = i.WorkingDir
	}

	defer func(start time.Time) { log("%s for %v", time.Since(start), cmdDebugStr(cmd)) }(time.Now())

	return runCmdContext(ctx, cmd)
}

// DebugHangingGoCommands may be set by tests to enable additional
// instrumentation (including panics) for debugging hanging Go commands.
//
// See golang/go#54461 for details.
var DebugHangingGoCommands = false

// runCmdContext is like exec.CommandContext except it sends os.Interrupt
// before os.Kill.
func runCmdContext(ctx context.Context, cmd *exec.Cmd) (err error) {
	// If cmd.Stdout is not an *os.File, the exec package will create a pipe and
	// copy it to the Writer in a goroutine until the process has finished and
	// either the pipe reaches EOF or command's WaitDelay expires.
	//
	// However, the output from 'go list' can be quite large, and we don't want to
	// keep reading (and allocating buffers) if we've already decided we don't
	// care about the output. We don't want to wait for the process to finish, and
	// we don't wait to wait for the WaitDelay to expire either.
	//
	// Instead, if cmd.Stdout requires a copying goroutine we explicitly replace
	// it with a pipe (which is an *os.File), which we can close in order to stop
	// copying output as soon as we realize we don't care about it.
	var stdoutW *os.File
	if cmd.Stdout != nil {
		if _, ok := cmd.Stdout.(*os.File); !ok {
			var stdoutR *os.File
			stdoutR, stdoutW, err = os.Pipe()
			if err != nil {
				return err
			}
			prevStdout := cmd.Stdout
			cmd.Stdout = stdoutW

			stdoutErr := make(chan error, 1)
			go func() {
				_, err := io.Copy(prevStdout, stdoutR)
				if err != nil {
					err = fmt.Errorf("copying stdout: %w", err)
				}
				stdoutErr <- err
			}()
			defer func() {
				// We started a goroutine to copy a stdout pipe.
				// Wait for it to finish, or terminate it if need be.
				var err2 error
				select {
				case err2 = <-stdoutErr:
					stdoutR.Close()
				case <-ctx.Done():
					stdoutR.Close()
					// Per https://pkg.go.dev/os#File.Close, the call to stdoutR.Close
					// should cause the Read call in io.Copy to unblock and return
					// immediately, but we still need to receive from stdoutErr to confirm
					// that that has happened.
					<-stdoutErr
					err2 = ctx.Err()
				}
				if err == nil {
					err = err2
				}
			}()

			// Per https://pkg.go.dev/os/exec#Cmd, “If Stdout and Stderr are the
			// same writer, and have a type that can be compared with ==, at most
			// one goroutine at a time will call Write.”
			//
			// Since we're starting a goroutine that writes to cmd.Stdout, we must
			// also update cmd.Stderr so that that still holds.
			func() {
				defer func() { recover() }()
				if cmd.Stderr == prevStdout {
					cmd.Stderr = cmd.Stdout
				}
			}()
		}
	}

	err = cmd.Start()
	if stdoutW != nil {
		// The child process has inherited the pipe file,
		// so close the copy held in this process.
		stdoutW.Close()
		stdoutW = nil
	}
	if err != nil {
		return err
	}

	resChan := make(chan error, 1)
	go func() {
		resChan <- cmd.Wait()
	}()

	// If we're interested in debugging hanging Go commands, stop waiting after a
	// minute and panic with interesting information.
	debug := DebugHangingGoCommands
	if debug {
		timer := time.NewTimer(1 * time.Minute)
		defer timer.Stop()
		select {
		case err := <-resChan:
			return err
		case <-timer.C:
			HandleHangingGoCommand(cmd.Process)
		case <-ctx.Done():
		}
	} else {
		select {
		case err := <-resChan:
			return err
		case <-ctx.Done():
		}
	}

	// Cancelled. Interrupt and see if it ends voluntarily.
	if err := cmd.Process.Signal(os.Interrupt); err == nil {
		// (We used to wait only 1s but this proved
		// fragile on loaded builder machines.)
		timer := time.NewTimer(5 * time.Second)
		defer timer.Stop()
		select {
		case err := <-resChan:
			return err
		case <-timer.C:
		}
	}

	// Didn't shut down in response to interrupt. Kill it hard.
	// TODO(rfindley): per advice from bcmills@, it may be better to send SIGQUIT
	// on certain platforms, such as unix.
	if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) && debug {
		log.Printf("error killing the Go command: %v", err)
	}

	return <-resChan
}

func HandleHangingGoCommand(proc *os.Process) {
	switch runtime.GOOS {
	case "linux", "darwin", "freebsd", "netbsd":
		fmt.Fprintln(os.Stderr, `DETECTED A HANGING GO COMMAND

The gopls test runner has detected a hanging go command. In order to debug
this, the output of ps and lsof/fstat is printed below.

See golang/go#54461 for more details.`)

		fmt.Fprintln(os.Stderr, "\nps axo ppid,pid,command:")
		fmt.Fprintln(os.Stderr, "-------------------------")
		psCmd := exec.Command("ps", "axo", "ppid,pid,command")
		psCmd.Stdout = os.Stderr
		psCmd.Stderr = os.Stderr
		if err := psCmd.Run(); err != nil {
			panic(fmt.Sprintf("running ps: %v", err))
		}

		listFiles := "lsof"
		if runtime.GOOS == "freebsd" || runtime.GOOS == "netbsd" {
			listFiles = "fstat"
		}

		fmt.Fprintln(os.Stderr, "\n"+listFiles+":")
		fmt.Fprintln(os.Stderr, "-----")
		listFilesCmd := exec.Command(listFiles)
		listFilesCmd.Stdout = os.Stderr
		listFilesCmd.Stderr = os.Stderr
		if err := listFilesCmd.Run(); err != nil {
			panic(fmt.Sprintf("running %s: %v", listFiles, err))
		}
	}
	panic(fmt.Sprintf("detected hanging go command (pid %d): see golang/go#54461 for more details", proc.Pid))
}

func cmdDebugStr(cmd *exec.Cmd) string {
	env := make(map[string]string)
	for _, kv := range cmd.Env {
		split := strings.SplitN(kv, "=", 2)
		if len(split) == 2 {
			k, v := split[0], split[1]
			env[k] = v
		}
	}

	var args []string
	for _, arg := range cmd.Args {
		quoted := strconv.Quote(arg)
		if quoted[1:len(quoted)-1] != arg || strings.Contains(arg, " ") {
			args = append(args, quoted)
		} else {
			args = append(args, arg)
		}
	}
	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v GOPROXY=%v PWD=%v %v", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["GOPROXY"], env["PWD"], strings.Join(args, " "))
}
