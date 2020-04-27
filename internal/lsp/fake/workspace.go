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
	"sync"

	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/proxydir"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/txtar"
)

// FileEvent wraps the protocol.FileEvent so that it can be associated with a
// workspace-relative path.
type FileEvent struct {
	Path          string
	ProtocolEvent protocol.FileEvent
}

// The Workspace type represents a temporary workspace to use for editing Go
// files in tests.
type Workspace struct {
	name     string
	gopath   string
	workdir  string
	proxydir string
	env      []string

	watcherMu sync.Mutex
	watchers  []func(context.Context, []FileEvent)
}

// NewWorkspace creates a named workspace populated by the txtar-encoded
// content given by txt. It creates temporary directories for the workspace
// content and for GOPATH.
func NewWorkspace(name, srctxt, proxytxt string, env ...string) (_ *Workspace, err error) {
	w := &Workspace{
		name: name,
		env:  env,
	}
	defer func() {
		// Clean up if we fail at any point in this constructor.
		if err != nil {
			w.removeAll()
		}
	}()
	dir, err := ioutil.TempDir("", fmt.Sprintf("goplstest-ws-%s-", name))
	if err != nil {
		return nil, fmt.Errorf("creating temporary workdir: %v", err)
	}
	w.workdir = dir
	gopath, err := ioutil.TempDir("", fmt.Sprintf("goplstest-gopath-%s-", name))
	if err != nil {
		return nil, fmt.Errorf("creating temporary gopath: %v", err)
	}
	w.gopath = gopath
	files := unpackTxt(srctxt)
	for name, data := range files {
		if err := w.writeFileData(name, string(data)); err != nil {
			return nil, fmt.Errorf("writing to workdir: %v", err)
		}
	}
	pd, err := ioutil.TempDir("", fmt.Sprintf("goplstest-proxy-%s-", name))
	if err != nil {
		return nil, fmt.Errorf("creating temporary proxy dir: %v", err)
	}
	w.proxydir = pd
	if err := writeProxyDir(unpackTxt(proxytxt), w.proxydir); err != nil {
		return nil, fmt.Errorf("writing proxy dir: %v", err)
	}
	return w, nil
}

func unpackTxt(txt string) map[string][]byte {
	dataMap := make(map[string][]byte)
	archive := txtar.Parse([]byte(txt))
	for _, f := range archive.Files {
		dataMap[f.Name] = f.Data
	}
	return dataMap
}

func writeProxyDir(files map[string][]byte, dir string) error {
	type moduleVersion struct {
		modulePath, version string
	}
	// Transform into the format expected by the proxydir package.
	filesByModule := make(map[moduleVersion]map[string][]byte)
	for name, data := range files {
		modulePath, version, suffix := splitModuleVersionPath(name)
		mv := moduleVersion{modulePath, version}
		if _, ok := filesByModule[mv]; !ok {
			filesByModule[mv] = make(map[string][]byte)
		}
		filesByModule[mv][suffix] = data
	}
	for mv, files := range filesByModule {
		if err := proxydir.WriteModuleVersion(dir, mv.modulePath, mv.version, files); err != nil {
			return fmt.Errorf("error writing %s@%s: %v", mv.modulePath, mv.version, err)
		}
	}
	return nil
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

// RootURI returns the root URI for this workspace.
func (w *Workspace) RootURI() protocol.DocumentURI {
	return toURI(w.workdir)
}

// GOPATH returns the value that GOPATH should be set to for this workspace.
func (w *Workspace) GOPATH() string {
	return w.gopath
}

// GOPROXY returns the value that GOPROXY should be set to for this workspace.
func (w *Workspace) GOPROXY() string {
	return proxydir.ToURL(w.proxydir)
}

// AddWatcher registers the given func to be called on any file change.
func (w *Workspace) AddWatcher(watcher func(context.Context, []FileEvent)) {
	w.watcherMu.Lock()
	w.watchers = append(w.watchers, watcher)
	w.watcherMu.Unlock()
}

// filePath returns the absolute filesystem path to a the workspace-relative
// path.
func (w *Workspace) filePath(path string) string {
	fp := filepath.FromSlash(path)
	if filepath.IsAbs(fp) {
		return fp
	}
	return filepath.Join(w.workdir, filepath.FromSlash(path))
}

// URI returns the URI to a the workspace-relative path.
func (w *Workspace) URI(path string) protocol.DocumentURI {
	return toURI(w.filePath(path))
}

// URIToPath converts a uri to a workspace-relative path (or an absolute path,
// if the uri is outside of the workspace).
func (w *Workspace) URIToPath(uri protocol.DocumentURI) string {
	root := w.RootURI().SpanURI().Filename()
	path := uri.SpanURI().Filename()
	if rel, err := filepath.Rel(root, path); err == nil && !strings.HasPrefix(rel, "..") {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(path)
}

func toURI(fp string) protocol.DocumentURI {
	return protocol.DocumentURI(span.URIFromPath(fp))
}

// ReadFile reads a text file specified by a workspace-relative path.
func (w *Workspace) ReadFile(path string) (string, error) {
	b, err := ioutil.ReadFile(w.filePath(path))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// RegexpSearch searches the file corresponding to path for the first position
// matching re.
func (w *Workspace) RegexpSearch(path string, re string) (Pos, error) {
	content, err := w.ReadFile(path)
	if err != nil {
		return Pos{}, err
	}
	start, _, err := regexpRange(content, re)
	return start, err
}

// RemoveFile removes a workspace-relative file path.
func (w *Workspace) RemoveFile(ctx context.Context, path string) error {
	fp := w.filePath(path)
	if err := os.Remove(fp); err != nil {
		return fmt.Errorf("removing %q: %v", path, err)
	}
	evts := []FileEvent{{
		Path: path,
		ProtocolEvent: protocol.FileEvent{
			URI:  w.URI(path),
			Type: protocol.Deleted,
		},
	}}
	w.sendEvents(ctx, evts)
	return nil
}

// GoEnv returns the environment variables that should be used for invoking Go
// commands in the workspace.
func (w *Workspace) GoEnv() []string {
	return append([]string{
		"GOPATH=" + w.GOPATH(),
		"GOPROXY=" + w.GOPROXY(),
		"GO111MODULE=",
		"GOSUMDB=off",
	}, w.env...)
}

// RunGoCommand executes a go command in the workspace.
func (w *Workspace) RunGoCommand(ctx context.Context, verb string, args ...string) error {
	inv := gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		WorkingDir: w.workdir,
		Env:        w.GoEnv(),
	}
	gocmdRunner := &gocommand.Runner{}
	_, stderr, _, err := gocmdRunner.RunRaw(ctx, inv)
	if err != nil {
		return err
	}
	// Hardcoded "file watcher": If the command executed was "go mod init",
	// send a file creation event for a go.mod in the working directory.
	if strings.HasPrefix(stderr.String(), "go: creating new go.mod") {
		modpath := filepath.Join(w.workdir, "go.mod")
		w.sendEvents(ctx, []FileEvent{{
			Path: modpath,
			ProtocolEvent: protocol.FileEvent{
				URI:  toURI(modpath),
				Type: protocol.Created,
			},
		}})
	}
	return nil
}

func (w *Workspace) sendEvents(ctx context.Context, evts []FileEvent) {
	w.watcherMu.Lock()
	watchers := make([]func(context.Context, []FileEvent), len(w.watchers))
	copy(watchers, w.watchers)
	w.watcherMu.Unlock()
	for _, w := range watchers {
		go w(ctx, evts)
	}
}

// WriteFile writes text file content to a workspace-relative path.
func (w *Workspace) WriteFile(ctx context.Context, path, content string) error {
	fp := w.filePath(path)
	_, err := os.Stat(fp)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("checking if %q exists: %v", path, err)
	}
	var changeType protocol.FileChangeType
	if os.IsNotExist(err) {
		changeType = protocol.Created
	} else {
		changeType = protocol.Changed
	}
	if err := w.writeFileData(path, content); err != nil {
		return err
	}
	evts := []FileEvent{{
		Path: path,
		ProtocolEvent: protocol.FileEvent{
			URI:  w.URI(path),
			Type: changeType,
		},
	}}
	w.sendEvents(ctx, evts)
	return nil
}

func (w *Workspace) writeFileData(path string, content string) error {
	fp := w.filePath(path)
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return fmt.Errorf("creating nested directory: %v", err)
	}
	if err := ioutil.WriteFile(fp, []byte(content), 0644); err != nil {
		return fmt.Errorf("writing %q: %v", path, err)
	}
	return nil
}

func (w *Workspace) removeAll() error {
	var wsErr, gopathErr, proxyErr error
	if w.gopath != "" {
		if err := w.RunGoCommand(context.Background(), "clean", "-modcache"); err != nil {
			gopathErr = fmt.Errorf("cleaning modcache: %v", err)
		} else {
			gopathErr = os.RemoveAll(w.gopath)
		}
	}
	if w.workdir != "" {
		wsErr = os.RemoveAll(w.workdir)
	}
	if w.proxydir != "" {
		proxyErr = os.RemoveAll(w.proxydir)
	}
	if wsErr != nil || gopathErr != nil || proxyErr != nil {
		return fmt.Errorf("error(s) cleaning workspace: removing workdir: %v; removing gopath: %v; removing proxy: %v", wsErr, gopathErr, proxyErr)
	}
	return nil
}

// Close removes all state associated with the workspace.
func (w *Workspace) Close() error {
	return w.removeAll()
}
