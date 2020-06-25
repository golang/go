// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

// Sandbox holds a collection of temporary resources to use for working with Go
// code in tests.
type Sandbox struct {
	gopath  string
	basedir string
	Proxy   *Proxy
	Workdir *Workdir

	// withoutWorkspaceFolders is used to simulate opening a single file in the
	// editor, without a workspace root. In that case, the client sends neither
	// workspace folders nor a root URI.
	withoutWorkspaceFolders bool
}

// NewSandbox creates a collection of named temporary resources, with a
// working directory populated by the txtar-encoded content in srctxt, and a
// file-based module proxy populated with the txtar-encoded content in
// proxytxt.
//
// If rootDir is non-empty, it will be used as the root of temporary
// directories created for the sandbox. Otherwise, a new temporary directory
// will be used as root.
func NewSandbox(rootDir, srctxt, proxytxt string, inGopath bool, withoutWorkspaceFolders bool) (_ *Sandbox, err error) {
	sb := &Sandbox{}
	defer func() {
		// Clean up if we fail at any point in this constructor.
		if err != nil {
			sb.Close()
		}
	}()

	baseDir, err := ioutil.TempDir(rootDir, "gopls-sandbox-")
	if err != nil {
		return nil, fmt.Errorf("creating temporary workdir: %v", err)
	}
	sb.basedir = baseDir
	proxydir := filepath.Join(sb.basedir, "proxy")
	sb.gopath = filepath.Join(sb.basedir, "gopath")
	// Set the working directory as $GOPATH/src if inGopath is true.
	workdir := filepath.Join(sb.gopath, "src")
	dirs := []string{sb.gopath, proxydir}
	if !inGopath {
		workdir = filepath.Join(sb.basedir, "work")
		dirs = append(dirs, workdir)
	}
	for _, subdir := range dirs {
		if err := os.Mkdir(subdir, 0755); err != nil {
			return nil, err
		}
	}
	sb.Proxy, err = NewProxy(proxydir, proxytxt)
	sb.Workdir, err = NewWorkdir(workdir, srctxt)
	sb.withoutWorkspaceFolders = withoutWorkspaceFolders

	return sb, nil
}

func unpackTxt(txt string) map[string][]byte {
	dataMap := make(map[string][]byte)
	archive := txtar.Parse([]byte(txt))
	for _, f := range archive.Files {
		dataMap[f.Name] = f.Data
	}
	return dataMap
}

// splitModuleVersionPath extracts module information from files stored in the
// directory structure modulePath@version/suffix.
// For example:
//  splitModuleVersionPath("mod.com@v1.2.3/package") = ("mod.com", "v1.2.3", "package")
func splitModuleVersionPath(path string) (modulePath, version, suffix string) {
	parts := strings.Split(path, "/")
	var modulePathParts []string
	for i, p := range parts {
		if strings.Contains(p, "@") {
			mv := strings.SplitN(p, "@", 2)
			modulePathParts = append(modulePathParts, mv[0])
			return strings.Join(modulePathParts, "/"), mv[1], strings.Join(parts[i+1:], "/")
		}
		modulePathParts = append(modulePathParts, p)
	}
	// Default behavior: this is just a module path.
	return path, "", ""
}

// GOPATH returns the value of the Sandbox GOPATH.
func (sb *Sandbox) GOPATH() string {
	return sb.gopath
}

// GoEnv returns the default environment variables that can be used for
// invoking Go commands in the sandbox.
func (sb *Sandbox) GoEnv() []string {
	vars := []string{
		"GOPATH=" + sb.GOPATH(),
		"GOPROXY=" + sb.Proxy.GOPROXY(),
		"GO111MODULE=",
		"GOSUMDB=off",
		"GOPACKAGESDRIVER=off",
	}
	if testenv.Go1Point() >= 5 {
		vars = append(vars, "GOMODCACHE=")
	}
	return vars
}

// RunGoCommand executes a go command in the sandbox.
func (sb *Sandbox) RunGoCommand(ctx context.Context, verb string, args ...string) error {
	inv := gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		WorkingDir: sb.Workdir.workdir,
		Env:        sb.GoEnv(),
	}
	gocmdRunner := &gocommand.Runner{}
	_, _, _, err := gocmdRunner.RunRaw(ctx, inv)
	if err != nil {
		return err
	}
	// Since running a go command may result in changes to workspace files,
	// check if we need to send any any "watched" file events.
	if err := sb.Workdir.CheckForFileChanges(ctx); err != nil {
		return fmt.Errorf("checking for file changes: %w", err)
	}
	return nil
}

// Close removes all state associated with the sandbox.
func (sb *Sandbox) Close() error {
	var goCleanErr error
	if sb.gopath != "" {
		if err := sb.RunGoCommand(context.Background(), "clean", "-modcache"); err != nil {
			goCleanErr = fmt.Errorf("cleaning modcache: %v", err)
		}
	}
	err := os.RemoveAll(sb.basedir)
	if err != nil || goCleanErr != nil {
		return fmt.Errorf("error(s) cleaning sandbox: cleaning modcache: %v; removing files: %v", goCleanErr, err)
	}
	return nil
}
