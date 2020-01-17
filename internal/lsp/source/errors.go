package source

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	errors "golang.org/x/xerrors"
)

const (
	// TODO(rstambler): We should really be able to point to a link on the website.
	modulesWiki = "https://github.com/golang/go/wiki/Modules"
)

func checkCommonErrors(ctx context.Context, v View) (string, error) {
	// Unfortunately, we probably can't have go/packages expose a function like this.
	// Since we only really understand the `go` command, check the user's GOPACKAGESDRIVER
	// and, if they are using `go list`, consider the possible error cases.
	gopackagesdriver := os.Getenv("GOPACKAGESDRIVER")
	if gopackagesdriver != "" && gopackagesdriver != "off" {
		return "", nil
	}

	// Some cases we should be able to detect:
	//
	//  1. The user is in GOPATH mode and is working outside their GOPATH
	//  2. The user is in module mode and has opened a subdirectory of their module
	//
	gopath := os.Getenv("GOPATH")
	folder := v.Folder().Filename()

	modRoot := filepath.Dir(v.ModFile())

	// Not inside of a module.
	inAModule := v.ModFile() != "" && v.ModFile() != os.DevNull

	// The user may have a multiple directories in their GOPATH.
	var inGopath bool
	for _, gp := range filepath.SplitList(gopath) {
		if strings.HasPrefix(folder, filepath.Join(gp, "src")) {
			inGopath = true
			break
		}
	}

	moduleMode := os.Getenv("GO111MODULE")

	var msg string
	// The user is in a module.
	if inAModule {
		rel, err := filepath.Rel(modRoot, folder)
		if err != nil || strings.HasPrefix(rel, "..") {
			msg = fmt.Sprintf("Your workspace root is %s, but your module root is %s. Please add %s or a subdirectory as a workspace folder.", folder, modRoot, modRoot)
		}
	} else if inGopath {
		if moduleMode == "on" {
			msg = "You are in module mode, but you are not inside of a module. Please create a module."
		}
	} else {
		msg = fmt.Sprintf("You are neither in a module nor in your GOPATH. Please see %s for information on how to set up your Go project.", modulesWiki)
	}
	return msg, nil
}

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
