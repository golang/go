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
	goproxy string
	Workdir *Workdir
}

// SandboxConfig controls the behavior of a test sandbox. The zero value
// defines a reasonable default.
type SandboxConfig struct {
	// RootDir sets the base directory to use when creating temporary
	// directories. If not specified, defaults to a new temporary directory.
	RootDir string
	// Files holds a txtar-encoded archive of files to populate the initial state
	// of the working directory.
	Files string
	// InGoPath specifies that the working directory should be within the
	// temporary GOPATH.
	InGoPath bool
	// Workdir configures the working directory of the Sandbox, for running in a
	// pre-existing directory. If unset, a new working directory will be created
	// under RootDir.
	//
	// This option is incompatible with InGoPath or Files.
	Workdir string

	// ProxyFiles holds a txtar-encoded archive of files to populate a file-based
	// Go proxy.
	ProxyFiles string
	// GOPROXY is the explicit GOPROXY value that should be used for the sandbox.
	//
	// This option is incompatible with ProxyFiles.
	GOPROXY string
}

// NewSandbox creates a collection of named temporary resources, with a
// working directory populated by the txtar-encoded content in srctxt, and a
// file-based module proxy populated with the txtar-encoded content in
// proxytxt.
//
// If rootDir is non-empty, it will be used as the root of temporary
// directories created for the sandbox. Otherwise, a new temporary directory
// will be used as root.
func NewSandbox(config *SandboxConfig) (_ *Sandbox, err error) {
	if config == nil {
		config = new(SandboxConfig)
	}

	if config.Workdir != "" && (config.Files != "" || config.InGoPath) {
		return nil, fmt.Errorf("invalid SandboxConfig: Workdir cannot be used in conjunction with Files or InGoPath. Got %+v", config)
	}

	if config.GOPROXY != "" && config.ProxyFiles != "" {
		return nil, fmt.Errorf("invalid SandboxConfig: GOPROXY cannot be set in conjunction with ProxyFiles. Got %+v", config)
	}

	sb := &Sandbox{}
	defer func() {
		// Clean up if we fail at any point in this constructor.
		if err != nil {
			sb.Close()
		}
	}()

	baseDir, err := ioutil.TempDir(config.RootDir, "gopls-sandbox-")
	if err != nil {
		return nil, fmt.Errorf("creating temporary workdir: %v", err)
	}
	sb.basedir = baseDir
	sb.gopath = filepath.Join(sb.basedir, "gopath")
	if err := os.Mkdir(sb.gopath, 0755); err != nil {
		return nil, err
	}
	if config.GOPROXY != "" {
		sb.goproxy = config.GOPROXY
	} else {
		proxydir := filepath.Join(sb.basedir, "proxy")
		if err := os.Mkdir(proxydir, 0755); err != nil {
			return nil, err
		}
		sb.goproxy, err = WriteProxy(proxydir, config.ProxyFiles)
		if err != nil {
			return nil, err
		}
	}
	if config.Workdir != "" {
		sb.Workdir = NewWorkdir(config.Workdir)
	} else {
		workdir := config.Workdir
		// If we don't have a pre-existing work dir, we want to create either
		// $GOPATH/src or <RootDir/work>.
		if config.InGoPath {
			// Set the working directory as $GOPATH/src.
			workdir = filepath.Join(sb.gopath, "src")
		} else if workdir == "" {
			workdir = filepath.Join(sb.basedir, "work")
		}
		if err := os.Mkdir(workdir, 0755); err != nil {
			return nil, err
		}
		sb.Workdir = NewWorkdir(workdir)
		if err := sb.Workdir.WriteInitialFiles(config.Files); err != nil {
			return nil, err
		}
	}

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
func (sb *Sandbox) GoEnv() map[string]string {
	vars := map[string]string{
		"GOPATH":           sb.GOPATH(),
		"GOPROXY":          sb.goproxy,
		"GO111MODULE":      "",
		"GOSUMDB":          "off",
		"GOPACKAGESDRIVER": "off",
	}
	if testenv.Go1Point() >= 5 {
		vars["GOMODCACHE"] = ""
	}
	return vars
}

// RunGoCommand executes a go command in the sandbox.
func (sb *Sandbox) RunGoCommand(ctx context.Context, verb string, args ...string) error {
	var vars []string
	for k, v := range sb.GoEnv() {
		vars = append(vars, fmt.Sprintf("%s=%s", k, v))
	}
	inv := gocommand.Invocation{
		Verb: verb,
		Args: args,
		Env:  vars,
	}
	// sb.Workdir may be nil if we exited the constructor with errors (we call
	// Close to clean up any partial state from the constructor, which calls
	// RunGoCommand).
	if sb.Workdir != nil {
		inv.WorkingDir = sb.Workdir.workdir
	}
	gocmdRunner := &gocommand.Runner{}
	_, _, _, err := gocmdRunner.RunRaw(ctx, inv)
	if err != nil {
		return err
	}
	// Since running a go command may result in changes to workspace files,
	// check if we need to send any any "watched" file events.
	if sb.Workdir != nil {
		if err := sb.Workdir.CheckForFileChanges(ctx); err != nil {
			return fmt.Errorf("checking for file changes: %w", err)
		}
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
