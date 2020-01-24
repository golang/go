package source

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"

	errors "golang.org/x/xerrors"
)

// InvokeGo returns the output of a go command invocation.
// It does not try to recover from errors.
func InvokeGo(ctx context.Context, dir string, env []string, args ...string) (*bytes.Buffer, error) {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	cmd := exec.CommandContext(ctx, "go", args...)
	// On darwin the cwd gets resolved to the real path, which breaks anything that
	// expects the working directory to keep the original path, including the
	// go command when dealing with modules.
	// The Go stdlib has a special feature where if the cwd and the PWD are the
	// same node then it trusts the PWD, so by setting it in the env for the child
	// process we fix up all the paths returned by the go command.
	cmd.Env = append(append([]string{}, env...), "PWD="+dir)
	cmd.Dir = dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	if err := cmd.Run(); err != nil {
		// Check for 'go' executable not being found.
		if ee, ok := err.(*exec.Error); ok && ee.Err == exec.ErrNotFound {
			return nil, fmt.Errorf("'gopls requires 'go', but %s", exec.ErrNotFound)
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return stdout, errors.Errorf("err: %v: stderr: %s", err, stderr)
	}
	return stdout, nil
}
