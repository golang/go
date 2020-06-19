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
	"time"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

// FileEvent wraps the protocol.FileEvent so that it can be associated with a
// workdir-relative path.
type FileEvent struct {
	Path          string
	ProtocolEvent protocol.FileEvent
}

// Workdir is a temporary working directory for tests. It exposes file
// operations in terms of relative paths, and fakes file watching by triggering
// events on file operations.
type Workdir struct {
	workdir string

	watcherMu sync.Mutex
	watchers  []func(context.Context, []FileEvent)

	fileMu sync.Mutex
	files  map[string]time.Time
}

// NewWorkdir writes the txtar-encoded file data in txt to dir, and returns a
// Workir for operating on these files using
func NewWorkdir(dir, txt string) (*Workdir, error) {
	w := &Workdir{workdir: dir}
	files := unpackTxt(txt)
	for name, data := range files {
		if err := w.writeFileData(name, string(data)); err != nil {
			return nil, fmt.Errorf("writing to workdir: %w", err)
		}
	}
	// Poll to capture the current file state.
	if _, err := w.pollFiles(); err != nil {
		return nil, fmt.Errorf("polling files: %w", err)
	}
	return w, nil
}

// RootURI returns the root URI for this working directory of this scratch
// environment.
func (w *Workdir) RootURI() protocol.DocumentURI {
	return toURI(w.workdir)
}

// AddWatcher registers the given func to be called on any file change.
func (w *Workdir) AddWatcher(watcher func(context.Context, []FileEvent)) {
	w.watcherMu.Lock()
	w.watchers = append(w.watchers, watcher)
	w.watcherMu.Unlock()
}

// filePath returns an absolute filesystem path for the workdir-relative path.
func (w *Workdir) filePath(path string) string {
	fp := filepath.FromSlash(path)
	if filepath.IsAbs(fp) {
		return fp
	}
	return filepath.Join(w.workdir, filepath.FromSlash(path))
}

// URI returns the URI to a the workdir-relative path.
func (w *Workdir) URI(path string) protocol.DocumentURI {
	return toURI(w.filePath(path))
}

// URIToPath converts a uri to a workdir-relative path (or an absolute path,
// if the uri is outside of the workdir).
func (w *Workdir) URIToPath(uri protocol.DocumentURI) string {
	fp := uri.SpanURI().Filename()
	return w.relPath(fp)
}

// relPath returns a '/'-encoded path relative to the working directory (or an
// absolute path if the file is outside of workdir)
func (w *Workdir) relPath(fp string) string {
	root := w.RootURI().SpanURI().Filename()
	if rel, err := filepath.Rel(root, fp); err == nil && !strings.HasPrefix(rel, "..") {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(fp)
}

func toURI(fp string) protocol.DocumentURI {
	return protocol.DocumentURI(span.URIFromPath(fp))
}

// ReadFile reads a text file specified by a workdir-relative path.
func (w *Workdir) ReadFile(path string) (string, error) {
	b, err := ioutil.ReadFile(w.filePath(path))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// RegexpSearch searches the file corresponding to path for the first position
// matching re.
func (w *Workdir) RegexpSearch(path string, re string) (Pos, error) {
	content, err := w.ReadFile(path)
	if err != nil {
		return Pos{}, err
	}
	start, _, err := regexpRange(content, re)
	return start, err
}

// RemoveFile removes a workdir-relative file path.
func (w *Workdir) RemoveFile(ctx context.Context, path string) error {
	fp := w.filePath(path)
	if err := os.Remove(fp); err != nil {
		return fmt.Errorf("removing %q: %w", path, err)
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

func (w *Workdir) sendEvents(ctx context.Context, evts []FileEvent) {
	if len(evts) == 0 {
		return
	}
	w.watcherMu.Lock()
	watchers := make([]func(context.Context, []FileEvent), len(w.watchers))
	copy(watchers, w.watchers)
	w.watcherMu.Unlock()
	for _, w := range watchers {
		go w(ctx, evts)
	}
}

// WriteFiles writes the text file content to workdir-relative paths.
// It batches notifications rather than sending them consecutively.
func (w *Workdir) WriteFiles(ctx context.Context, files map[string]string) error {
	var evts []FileEvent
	for filename, content := range files {
		evt, err := w.writeFile(ctx, filename, content)
		if err != nil {
			return err
		}
		evts = append(evts, evt)
	}
	w.sendEvents(ctx, evts)
	return nil
}

// WriteFile writes text file content to a workdir-relative path.
func (w *Workdir) WriteFile(ctx context.Context, path, content string) error {
	evt, err := w.writeFile(ctx, path, content)
	if err != nil {
		return err
	}
	w.sendEvents(ctx, []FileEvent{evt})
	return nil
}

func (w *Workdir) writeFile(ctx context.Context, path, content string) (FileEvent, error) {
	fp := w.filePath(path)
	_, err := os.Stat(fp)
	if err != nil && !os.IsNotExist(err) {
		return FileEvent{}, fmt.Errorf("checking if %q exists: %w", path, err)
	}
	var changeType protocol.FileChangeType
	if os.IsNotExist(err) {
		changeType = protocol.Created
	} else {
		changeType = protocol.Changed
	}
	if err := w.writeFileData(path, content); err != nil {
		return FileEvent{}, err
	}
	return FileEvent{
		Path: path,
		ProtocolEvent: protocol.FileEvent{
			URI:  w.URI(path),
			Type: changeType,
		},
	}, nil
}

func (w *Workdir) writeFileData(path string, content string) error {
	fp := w.filePath(path)
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return fmt.Errorf("creating nested directory: %w", err)
	}
	if err := ioutil.WriteFile(fp, []byte(content), 0644); err != nil {
		return fmt.Errorf("writing %q: %w", path, err)
	}
	return nil
}

// ListFiles lists files in the given directory, returning a map of relative
// path to modification time.
func (w *Workdir) ListFiles(dir string) (map[string]time.Time, error) {
	files := make(map[string]time.Time)
	absDir := w.filePath(dir)
	if err := filepath.Walk(absDir, func(fp string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		path := w.relPath(fp)
		files[path] = info.ModTime()
		return nil
	}); err != nil {
		return nil, err
	}
	return files, nil
}

// CheckForFileChanges walks the working directory and checks for any files
// that have changed since the last poll.
func (w *Workdir) CheckForFileChanges(ctx context.Context) error {
	evts, err := w.pollFiles()
	if err != nil {
		return err
	}
	w.sendEvents(ctx, evts)
	return nil
}

// pollFiles updates w.files and calculates FileEvents corresponding to file
// state changes since the last poll. It does not call sendEvents.
func (w *Workdir) pollFiles() ([]FileEvent, error) {
	w.fileMu.Lock()
	defer w.fileMu.Unlock()

	files, err := w.ListFiles(".")
	if err != nil {
		return nil, err
	}
	var evts []FileEvent
	// Check which files have been added or modified.
	for path, mtime := range files {
		oldmtime, ok := w.files[path]
		delete(w.files, path)
		var typ protocol.FileChangeType
		switch {
		case !ok:
			typ = protocol.Created
		case oldmtime != mtime:
			typ = protocol.Changed
		default:
			continue
		}
		evts = append(evts, FileEvent{
			Path: path,
			ProtocolEvent: protocol.FileEvent{
				URI:  w.URI(path),
				Type: typ,
			},
		})
	}
	// Any remaining files must have been deleted.
	for path := range w.files {
		evts = append(evts, FileEvent{
			Path: path,
			ProtocolEvent: protocol.FileEvent{
				URI:  w.URI(path),
				Type: protocol.Deleted,
			},
		})
	}
	w.files = files
	return evts, nil
}
