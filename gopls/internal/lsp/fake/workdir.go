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
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/robustio"
	"golang.org/x/tools/gopls/internal/span"
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

// WriteFileData writes content to the relative path, replacing the special
// token $SANDBOX_WORKDIR with the relative root given by rel.
func WriteFileData(path string, content []byte, rel RelativeTo) error {
	content = bytes.ReplaceAll(content, []byte("$SANDBOX_WORKDIR"), []byte(rel))
	fp := rel.AbsPath(path)
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return fmt.Errorf("creating nested directory: %w", err)
	}
	backoff := 1 * time.Millisecond
	for {
		err := ioutil.WriteFile(fp, []byte(content), 0644)
		if err != nil {
			// This lock file violation is not handled by the robustio package, as it
			// indicates a real race condition that could be avoided.
			if isWindowsErrLockViolation(err) {
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
			return fmt.Errorf("writing %q: %w", path, err)
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
	// File identities we know about, for the purpose of detecting changes.
	//
	// Since files is only used for detecting _changes_, we are tolerant of
	// fileIDs that may have hash and mtime coming from different states of the
	// file: if either are out of sync, then the next poll should detect a
	// discrepancy. It is OK if we detect too many changes, but not OK if we miss
	// changes.
	//
	// For that matter, this mechanism for detecting changes can still be flaky
	// on platforms where mtime is very coarse (such as older versions of WSL).
	// It would be much better to use a proper fs event library, but we can't
	// currently import those into x/tools.
	//
	// TODO(golang/go#52284): replace this polling mechanism with a
	// cross-platform library for filesystem notifications.
	files map[string]fileID
}

// fileID is a file identity for the purposes of detecting on-disk
// modifications.
type fileID struct {
	hash  string
	mtime time.Time
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
	w.files = map[string]fileID{}
	for name, data := range files {
		if err := WriteFileData(name, data, w.RelativeTo); err != nil {
			return fmt.Errorf("writing to workdir: %w", err)
		}
		fp := w.AbsPath(name)

		// We need the mtime of the file just written for the purposes of tracking
		// file identity. Calling Stat here could theoretically return an mtime
		// that is inconsistent with the file contents represented by the hash, but
		// since we "own" this file we assume that the mtime is correct.
		//
		// Furthermore, see the documentation for Workdir.files for why mismatches
		// between identifiers are considered to be benign.
		fi, err := os.Stat(fp)
		if err != nil {
			return fmt.Errorf("reading file info: %v", err)
		}

		w.files[name] = fileID{
			hash:  hashFile(data),
			mtime: fi.ModTime(),
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
				return fmt.Errorf("removing %q: %w", e.Path, err)
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
		return fmt.Errorf("removing %q: %w", path, err)
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
		return FileEvent{}, fmt.Errorf("checking if %q exists: %w", path, err)
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
	return w.fileEvent(path, changeType), nil
}

func (w *Workdir) fileEvent(path string, changeType protocol.FileChangeType) FileEvent {
	return FileEvent{
		Path: path,
		ProtocolEvent: protocol.FileEvent{
			URI:  w.URI(path),
			Type: changeType,
		},
	}
}

// RenameFile performs an on disk-renaming of the workdir-relative oldPath to
// workdir-relative newPath.
func (w *Workdir) RenameFile(ctx context.Context, oldPath, newPath string) error {
	oldAbs := w.AbsPath(oldPath)
	newAbs := w.AbsPath(newPath)

	if err := robustio.Rename(oldAbs, newAbs); err != nil {
		return err
	}

	// Send synthetic file events for the renaming. Renamed files are handled as
	// Delete+Create events:
	// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#fileChangeType
	events := []FileEvent{
		w.fileEvent(oldPath, protocol.Deleted),
		w.fileEvent(newPath, protocol.Created),
	}
	w.sendEvents(ctx, events)

	return nil
}

// ListFiles returns a new sorted list of the relative paths of files in dir,
// recursively.
func (w *Workdir) ListFiles(dir string) ([]string, error) {
	m, err := w.listFiles(dir)
	if err != nil {
		return nil, err
	}

	var paths []string
	for p := range m {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths, nil
}

// listFiles lists files in the given directory, returning a map of relative
// path to contents and modification time.
func (w *Workdir) listFiles(dir string) (map[string]fileID, error) {
	files := make(map[string]fileID)
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
		// The content returned by ioutil.ReadFile could be inconsistent with
		// info.ModTime(), due to a subsequent modification. See the documentation
		// for w.files for why we consider this to be benign.
		files[path] = fileID{
			hash:  hashFile(data),
			mtime: info.ModTime(),
		}
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
	for path, id := range files {
		oldID, ok := w.files[path]
		delete(w.files, path)
		var typ protocol.FileChangeType
		switch {
		case !ok:
			typ = protocol.Created
		case oldID != id:
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
