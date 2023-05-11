// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/robustio"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

// Sandbox holds a collection of temporary resources to use for working with Go
// code in tests.
type Sandbox struct {
	gopath          string
	rootdir         string
	goproxy         string
	Workdir         *Workdir
	goCommandRunner gocommand.Runner
}

// SandboxConfig controls the behavior of a test sandbox. The zero value
// defines a reasonable default.
type SandboxConfig struct {
	// RootDir sets the base directory to use when creating temporary
	// directories. If not specified, defaults to a new temporary directory.
	RootDir string
	// Files holds a txtar-encoded archive of files to populate the initial state
	// of the working directory.
	//
	// For convenience, the special substring "$SANDBOX_WORKDIR" is replaced with
	// the sandbox's resolved working directory before writing files.
	Files map[string][]byte
	// InGoPath specifies that the working directory should be within the
	// temporary GOPATH.
	InGoPath bool
	// Workdir configures the working directory of the Sandbox. It behaves as
	// follows:
	//  - if set to an absolute path, use that path as the working directory.
	//  - if set to a relative path, create and use that path relative to the
	//    sandbox.
	//  - if unset, default to a the 'work' subdirectory of the sandbox.
	//
	// This option is incompatible with InGoPath or Files.
	Workdir string
	// ProxyFiles holds a txtar-encoded archive of files to populate a file-based
	// Go proxy.
	ProxyFiles map[string][]byte
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
//
// TODO(rfindley): the sandbox abstraction doesn't seem to carry its weight.
// Sandboxes should be composed out of their building-blocks, rather than via a
// monolithic configuration.
func NewSandbox(config *SandboxConfig) (_ *Sandbox, err error) {
	if config == nil {
		config = new(SandboxConfig)
	}
	if err := validateConfig(*config); err != nil {
		return nil, fmt.Errorf("invalid SandboxConfig: %v", err)
	}

	sb := &Sandbox{}
	defer func() {
		// Clean up if we fail at any point in this constructor.
		if err != nil {
			sb.Close()
		}
	}()

	rootDir := config.RootDir
	if rootDir == "" {
		rootDir, err = ioutil.TempDir(config.RootDir, "gopls-sandbox-")
		if err != nil {
			return nil, fmt.Errorf("creating temporary workdir: %v", err)
		}
	}
	sb.rootdir = rootDir
	sb.gopath = filepath.Join(sb.rootdir, "gopath")
	if err := os.Mkdir(sb.gopath, 0755); err != nil {
		return nil, err
	}
	if config.GOPROXY != "" {
		sb.goproxy = config.GOPROXY
	} else {
		proxydir := filepath.Join(sb.rootdir, "proxy")
		if err := os.Mkdir(proxydir, 0755); err != nil {
			return nil, err
		}
		sb.goproxy, err = WriteProxy(proxydir, config.ProxyFiles)
		if err != nil {
			return nil, err
		}
	}
	// Short-circuit writing the workdir if we're given an absolute path, since
	// this is used for running in an existing directory.
	// TODO(findleyr): refactor this to be less of a workaround.
	if filepath.IsAbs(config.Workdir) {
		sb.Workdir, err = NewWorkdir(config.Workdir, nil)
		if err != nil {
			return nil, err
		}
		return sb, nil
	}
	var workdir string
	if config.Workdir == "" {
		if config.InGoPath {
			// Set the working directory as $GOPATH/src.
			workdir = filepath.Join(sb.gopath, "src")
		} else if workdir == "" {
			workdir = filepath.Join(sb.rootdir, "work")
		}
	} else {
		// relative path
		workdir = filepath.Join(sb.rootdir, config.Workdir)
	}
	if err := os.MkdirAll(workdir, 0755); err != nil {
		return nil, err
	}
	sb.Workdir, err = NewWorkdir(workdir, config.Files)
	if err != nil {
		return nil, err
	}
	return sb, nil
}

// Tempdir creates a new temp directory with the given txtar-encoded files. It
// is the responsibility of the caller to call os.RemoveAll on the returned
// file path when it is no longer needed.
func Tempdir(files map[string][]byte) (string, error) {
	dir, err := ioutil.TempDir("", "gopls-tempdir-")
	if err != nil {
		return "", err
	}
	for name, data := range files {
		if err := writeFileData(name, data, RelativeTo(dir)); err != nil {
			return "", fmt.Errorf("writing to tempdir: %w", err)
		}
	}
	return dir, nil
}

func UnpackTxt(txt string) map[string][]byte {
	dataMap := make(map[string][]byte)
	archive := txtar.Parse([]byte(txt))
	for _, f := range archive.Files {
		if _, ok := dataMap[f.Name]; ok {
			panic(fmt.Sprintf("found file %q twice", f.Name))
		}
		dataMap[f.Name] = f.Data
	}
	return dataMap
}

func validateConfig(config SandboxConfig) error {
	if filepath.IsAbs(config.Workdir) && (len(config.Files) > 0 || config.InGoPath) {
		return errors.New("absolute Workdir cannot be set in conjunction with Files or InGoPath")
	}
	if config.Workdir != "" && config.InGoPath {
		return errors.New("Workdir cannot be set in conjunction with InGoPath")
	}
	if config.GOPROXY != "" && config.ProxyFiles != nil {
		return errors.New("GOPROXY cannot be set in conjunction with ProxyFiles")
	}
	return nil
}

// splitModuleVersionPath extracts module information from files stored in the
// directory structure modulePath@version/suffix.
// For example:
//
//	splitModuleVersionPath("mod.com@v1.2.3/package") = ("mod.com", "v1.2.3", "package")
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

func (sb *Sandbox) RootDir() string {
	return sb.rootdir
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

// goCommandInvocation returns a new gocommand.Invocation initialized with the
// sandbox environment variables and working directory.
func (sb *Sandbox) goCommandInvocation() gocommand.Invocation {
	var vars []string
	for k, v := range sb.GoEnv() {
		vars = append(vars, fmt.Sprintf("%s=%s", k, v))
	}
	inv := gocommand.Invocation{
		Env: vars,
	}
	// sb.Workdir may be nil if we exited the constructor with errors (we call
	// Close to clean up any partial state from the constructor, which calls
	// RunGoCommand).
	if sb.Workdir != nil {
		inv.WorkingDir = string(sb.Workdir.RelativeTo)
	}
	return inv
}

// RunGoCommand executes a go command in the sandbox. If checkForFileChanges is
// true, the sandbox scans the working directory and emits file change events
// for any file changes it finds.
func (sb *Sandbox) RunGoCommand(ctx context.Context, dir, verb string, args, env []string, checkForFileChanges bool) error {
	inv := sb.goCommandInvocation()
	inv.Verb = verb
	inv.Args = args
	inv.Env = append(inv.Env, env...)
	if dir != "" {
		inv.WorkingDir = sb.Workdir.AbsPath(dir)
	}
	stdout, stderr, _, err := sb.goCommandRunner.RunRaw(ctx, inv)
	if err != nil {
		return fmt.Errorf("go command failed (stdout: %s) (stderr: %s): %v", stdout.String(), stderr.String(), err)
	}
	// Since running a go command may result in changes to workspace files,
	// check if we need to send any any "watched" file events.
	//
	// TODO(rFindley): this side-effect can impact the usability of the sandbox
	//                 for benchmarks. Consider refactoring.
	if sb.Workdir != nil && checkForFileChanges {
		if err := sb.Workdir.CheckForFileChanges(ctx); err != nil {
			return fmt.Errorf("checking for file changes: %w", err)
		}
	}
	return nil
}

// GoVersion checks the version of the go command.
// It returns the X in Go 1.X.
func (sb *Sandbox) GoVersion(ctx context.Context) (int, error) {
	inv := sb.goCommandInvocation()
	return gocommand.GoVersion(ctx, inv, &sb.goCommandRunner)
}

// Close removes all state associated with the sandbox.
func (sb *Sandbox) Close() error {
	var goCleanErr error
	if sb.gopath != "" {
		goCleanErr = sb.RunGoCommand(context.Background(), "", "clean", []string{"-modcache"}, nil, false)
	}
	err := robustio.RemoveAll(sb.rootdir)
	if err != nil || goCleanErr != nil {
		return fmt.Errorf("error(s) cleaning sandbox: cleaning modcache: %v; removing files: %v", goCleanErr, err)
	}
	return nil
}
