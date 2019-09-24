package source

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/span"
)

const (
	// TODO(rstambler): We should really be able to point to a link on the website.
	modulesWiki = "https://github.com/golang/go/wiki/Modules"
)

func checkCommonErrors(ctx context.Context, view View, uri span.URI) (string, error) {
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
	cfg := view.Config(ctx)

	// Invoke `go env GOMOD` inside of the directory of the file.
	fdir := filepath.Dir(uri.Filename())
	b, err := invokeGo(ctx, fdir, cfg.Env, "env", "GOMOD")
	if err != nil {
		return "", err
	}
	modFile := strings.TrimSpace(b.String())
	if modFile == filepath.FromSlash("/dev/null") {
		modFile = ""
	}

	// Not inside of a module.
	inAModule := modFile != ""
	inGopath := strings.HasPrefix(uri.Filename(), filepath.Join(gopath, "src"))
	moduleMode := os.Getenv("GO111MODULE")

	var msg string
	// The user is in a module.
	if inAModule {
		// The workspace root is open to a directory different from the module root.
		if modRoot := filepath.Dir(modFile); cfg.Dir != filepath.Dir(modFile) {
			msg = fmt.Sprintf("Your workspace root is %s, but your module root is %s. Please add %s as a workspace folder.", cfg.Dir, modRoot, modRoot)
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

// invokeGo returns the stdout of a go command invocation.
// Borrowed from golang.org/x/tools/go/packages/golist.go.
func invokeGo(ctx context.Context, dir string, env []string, args ...string) (*bytes.Buffer, error) {
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
		if _, ok := err.(*exec.ExitError); !ok {
			// Catastrophic error:
			// - context cancellation
			return nil, fmt.Errorf("couldn't exec 'go %v': %s %T", args, err, err)
		}
	}
	return stdout, nil
}
