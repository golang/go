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

	"golang.org/x/tools/internal/lsp/protocol"
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
	name    string
	gopath  string
	workdir string

	watcherMu sync.Mutex
	watchers  []func(context.Context, []FileEvent)
}

// NewWorkspace creates a named workspace populated by the txtar-encoded
// content given by txt. It creates temporary directories for the workspace
// content and for GOPATH.
func NewWorkspace(name string, txt []byte) (_ *Workspace, err error) {
	w := &Workspace{name: name}
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
	archive := txtar.Parse(txt)
	for _, f := range archive.Files {
		if err := w.writeFileData(f.Name, f.Data); err != nil {
			return nil, err
		}
	}
	return w, nil
}

// RootURI returns the root URI for this workspace.
func (w *Workspace) RootURI() protocol.DocumentURI {
	return toURI(w.workdir)
}

// GOPATH returns the value that GOPATH should be set to for this workspace.
func (w *Workspace) GOPATH() string {
	return w.gopath
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
	if err := w.writeFileData(path, []byte(content)); err != nil {
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

func (w *Workspace) writeFileData(path string, data []byte) error {
	fp := w.filePath(path)
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return fmt.Errorf("creating nested directory: %v", err)
	}
	if err := ioutil.WriteFile(fp, data, 0644); err != nil {
		return fmt.Errorf("writing %q: %v", path, err)
	}
	return nil
}

func (w *Workspace) removeAll() error {
	var werr, perr error
	if w.workdir != "" {
		werr = os.RemoveAll(w.workdir)
	}
	if w.gopath != "" {
		perr = os.RemoveAll(w.gopath)
	}
	if werr != nil || perr != nil {
		return fmt.Errorf("error(s) cleaning workspace: removing workdir: %v; removing gopath: %v", werr, perr)
	}
	return nil
}

// Close removes all state associated with the workspace.
func (w *Workspace) Close() error {
	return w.removeAll()
}
