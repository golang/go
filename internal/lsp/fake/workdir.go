// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// FileEvent wraps the protocol.FileEvent so that it can be associated with a
// workdir-relative path.
type FileEvent struct {
	Path, Content string
	ProtocolEvent protocol.FileEvent
}

// RelativeTo is a helper for operations relative to a given directory.
type RelativeTo string

// AbsPath returns an absolute filesystem path for the workdir-relative path.
func (r RelativeTo) AbsPath(path string) string {
	fp := filepath.FromSlash(path)
	if filepath.IsAbs(fp) {
		return fp
	}
	return filepath.Join(string(r), filepath.FromSlash(path))
}

// RelPath returns a '/'-encoded path relative to the working directory (or an
// absolute path if the file is outside of workdir)
func (r RelativeTo) RelPath(fp string) string {
	root := string(r)
	if rel, err := filepath.Rel(root, fp); err == nil && !strings.HasPrefix(rel, "..") {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(fp)
}

func writeTxtar(txt string, rel RelativeTo) error {
	files := UnpackTxt(txt)
	for name, data := range files {
		if err := WriteFileData(name, data, rel); err != nil {
			return errors.Errorf("writing to workdir: %w", err)
		}
	}
	return nil
}

// WriteFileData writes content to the relative path, replacing the special
// token $SANDBOX_WORKDIR with the relative root given by rel.
func WriteFileData(path string, content []byte, rel RelativeTo) error {
	content = bytes.ReplaceAll(content, []byte("$SANDBOX_WORKDIR"), []byte(rel))
	fp := rel.AbsPath(path)
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return errors.Errorf("creating nested directory: %w", err)
	}
	backoff := 1 * time.Millisecond
	for {
		err := ioutil.WriteFile(fp, []byte(content), 0644)
		if err != nil {
			if isWindowsErrLockViolation(err) {
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
			return errors.Errorf("writing %q: %w", path, err)
		}
		return nil
	}
}

// isWindowsErrLockViolation reports whether err is ERROR_LOCK_VIOLATION
// on Windows.
var isWindowsErrLockViolation = func(err error) bool { return false }

// Workdir is a temporary working directory for tests. It exposes file
// operations in terms of relative paths, and fakes file watching by triggering
// events on file operations.
type Workdir struct {
	RelativeTo

	watcherMu sync.Mutex
	watchers  []func(context.Context, []FileEvent)

	fileMu sync.Mutex
	files  map[string]string
}

// NewWorkdir writes the txtar-encoded file data in txt to dir, and returns a
// Workir for operating on these files using
func NewWorkdir(dir string) *Workdir {
	return &Workdir{RelativeTo: RelativeTo(dir)}
}

func hashFile(data []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

func (w *Workdir) writeInitialFiles(files map[string][]byte) error {
	w.files = map[string]string{}
	for name, data := range files {
		w.files[name] = hashFile(data)
		if err := WriteFileData(name, data, w.RelativeTo); err != nil {
			return errors.Errorf("writing to workdir: %w", err)
		}
	}
	return nil
}

// RootURI returns the root URI for this working directory of this scratch
// environment.
func (w *Workdir) RootURI() protocol.DocumentURI {
	return toURI(string(w.RelativeTo))
}

// AddWatcher registers the given func to be called on any file change.
func (w *Workdir) AddWatcher(watcher func(context.Context, []FileEvent)) {
	w.watcherMu.Lock()
	w.watchers = append(w.watchers, watcher)
	w.watcherMu.Unlock()
}

// URI returns the URI to a the workdir-relative path.
func (w *Workdir) URI(path string) protocol.DocumentURI {
	return toURI(w.AbsPath(path))
}

// URIToPath converts a uri to a workdir-relative path (or an absolute path,
// if the uri is outside of the workdir).
func (w *Workdir) URIToPath(uri protocol.DocumentURI) string {
	fp := uri.SpanURI().Filename()
	return w.RelPath(fp)
}

func toURI(fp string) protocol.DocumentURI {
	return protocol.DocumentURI(span.URIFromPath(fp))
}

// ReadFile reads a text file specified by a workdir-relative path.
func (w *Workdir) ReadFile(path string) (string, error) {
	backoff := 1 * time.Millisecond
	for {
		b, err := ioutil.ReadFile(w.AbsPath(path))
		if err != nil {
			if runtime.GOOS == "plan9" && strings.HasSuffix(err.Error(), " exclusive use file already open") {
				// Plan 9 enforces exclusive access to locked files.
				// Give the owner time to unlock it and retry.
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
			return "", err
		}
		return string(b), nil
	}
}

func (w *Workdir) RegexpRange(path, re string) (Pos, Pos, error) {
	content, err := w.ReadFile(path)
	if err != nil {
		return Pos{}, Pos{}, err
	}
	return regexpRange(content, re)
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

// ChangeFilesOnDisk executes the given on-disk file changes in a batch,
// simulating the action of changing branches outside of an editor.
func (w *Workdir) ChangeFilesOnDisk(ctx context.Context, events []FileEvent) error {
	for _, e := range events {
		switch e.ProtocolEvent.Type {
		case protocol.Deleted:
			fp := w.AbsPath(e.Path)
			if err := os.Remove(fp); err != nil {
				return errors.Errorf("removing %q: %w", e.Path, err)
			}
		case protocol.Changed, protocol.Created:
			if _, err := w.writeFile(ctx, e.Path, e.Content); err != nil {
				return err
			}
		}
	}
	w.sendEvents(ctx, events)
	return nil
}

// RemoveFile removes a workdir-relative file path.
func (w *Workdir) RemoveFile(ctx context.Context, path string) error {
	fp := w.AbsPath(path)
	if err := os.RemoveAll(fp); err != nil {
		return errors.Errorf("removing %q: %w", path, err)
	}
	w.fileMu.Lock()
	defer w.fileMu.Unlock()

	evts := []FileEvent{{
		Path: path,
		ProtocolEvent: protocol.FileEvent{
			URI:  w.URI(path),
			Type: protocol.Deleted,
		},
	}}
	w.sendEvents(ctx, evts)
	delete(w.files, path)
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
		w(ctx, evts)
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
	fp := w.AbsPath(path)
	_, err := os.Stat(fp)
	if err != nil && !os.IsNotExist(err) {
		return FileEvent{}, errors.Errorf("checking if %q exists: %w", path, err)
	}
	var changeType protocol.FileChangeType
	if os.IsNotExist(err) {
		changeType = protocol.Created
	} else {
		changeType = protocol.Changed
	}
	if err := WriteFileData(path, []byte(content), w.RelativeTo); err != nil {
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

// listFiles lists files in the given directory, returning a map of relative
// path to modification time.
func (w *Workdir) listFiles(dir string) (map[string]string, error) {
	files := make(map[string]string)
	absDir := w.AbsPath(dir)
	if err := filepath.Walk(absDir, func(fp string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		path := w.RelPath(fp)
		data, err := ioutil.ReadFile(fp)
		if err != nil {
			return err
		}
		files[path] = hashFile(data)
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

	files, err := w.listFiles(".")
	if err != nil {
		return nil, err
	}
	var evts []FileEvent
	// Check which files have been added or modified.
	for path, hash := range files {
		oldhash, ok := w.files[path]
		delete(w.files, path)
		var typ protocol.FileChangeType
		switch {
		case !ok:
			typ = protocol.Created
		case oldhash != hash:
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
