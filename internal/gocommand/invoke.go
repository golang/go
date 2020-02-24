// Package gocommand is a helper for calling the go command.
package gocommand

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// An Invocation represents a call to the go command.
type Invocation struct {
	Verb       string
	Args       []string
	BuildFlags []string
	Env        []string
	WorkingDir string
	Logf       func(format string, args ...interface{})
}

// Run runs the invocation, returning its stdout and an error suitable for
// human consumption, including stderr.
func (i *Invocation) Run(ctx context.Context) (*bytes.Buffer, error) {
	stdout, _, friendly, _ := i.RunRaw(ctx)
	return stdout, friendly
}

// RunRaw is like Run, but also returns the raw stderr and error for callers
// that want to do low-level error handling/recovery.
func (i *Invocation) RunRaw(ctx context.Context) (stdout *bytes.Buffer, stderr *bytes.Buffer, friendlyError error, rawError error) {
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
	stdout = &bytes.Buffer{}
	stderr = &bytes.Buffer{}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	// On darwin the cwd gets resolved to the real path, which breaks anything that
	// expects the working directory to keep the original path, including the
	// go command when dealing with modules.
	// The Go stdlib has a special feature where if the cwd and the PWD are the
	// same node then it trusts the PWD, so by setting it in the env for the child
	// process we fix up all the paths returned by the go command.
	cmd.Env = append(append([]string{}, i.Env...), "PWD="+i.WorkingDir)
	cmd.Dir = i.WorkingDir

	defer func(start time.Time) { log("%s for %v", time.Since(start), cmdDebugStr(cmd)) }(time.Now())

	rawError = runCmdContext(ctx, cmd)
	friendlyError = rawError
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
