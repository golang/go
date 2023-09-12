// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/robustio"
)

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

// writeFileData writes content to the relative path, replacing the special
// token $SANDBOX_WORKDIR with the relative root given by rel. It does not
// trigger any file events.
func writeFileData(path string, content []byte, rel RelativeTo) error {
	content = bytes.ReplaceAll(content, []byte("$SANDBOX_WORKDIR"), []byte(rel))
	fp := rel.AbsPath(path)
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return fmt.Errorf("creating nested directory: %w", err)
	}
	backoff := 1 * time.Millisecond
	for {
		err := os.WriteFile(fp, []byte(content), 0644)
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
	watchers  []func(context.Context, []protocol.FileEvent)

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

// NewWorkdir writes the txtar-encoded file data in txt to dir, and returns a
// Workir for operating on these files using
func NewWorkdir(dir string, files map[string][]byte) (*Workdir, error) {
	w := &Workdir{RelativeTo: RelativeTo(dir)}
	for name, data := range files {
		if err := writeFileData(name, data, w.RelativeTo); err != nil {
			return nil, fmt.Errorf("writing to workdir: %w", err)
		}
	}
	_, err := w.pollFiles() // poll files to populate the files map.
	return w, err
}

// fileID identifies a file version on disk.
type fileID struct {
	mtime time.Time
	hash  string // empty if mtime is old enough to be reliable; otherwise a file digest
}

func hashFile(data []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

// RootURI returns the root URI for this working directory of this scratch
// environment.
func (w *Workdir) RootURI() protocol.DocumentURI {
	return toURI(string(w.RelativeTo))
}

// AddWatcher registers the given func to be called on any file change.
func (w *Workdir) AddWatcher(watcher func(context.Context, []protocol.FileEvent)) {
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
func (w *Workdir) ReadFile(path string) ([]byte, error) {
	backoff := 1 * time.Millisecond
	for {
		b, err := os.ReadFile(w.AbsPath(path))
		if err != nil {
			if runtime.GOOS == "plan9" && strings.HasSuffix(err.Error(), " exclusive use file already open") {
				// Plan 9 enforces exclusive access to locked files.
				// Give the owner time to unlock it and retry.
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
			return nil, err
		}
		return b, nil
	}
}

// RegexpSearch searches the file corresponding to path for the first position
// matching re.
func (w *Workdir) RegexpSearch(path string, re string) (protocol.Location, error) {
	content, err := w.ReadFile(path)
	if err != nil {
		return protocol.Location{}, err
	}
	mapper := protocol.NewMapper(w.URI(path).SpanURI(), content)
	return regexpLocation(mapper, re)
}

// RemoveFile removes a workdir-relative file path and notifies watchers of the
// change.
func (w *Workdir) RemoveFile(ctx context.Context, path string) error {
	fp := w.AbsPath(path)
	if err := robustio.RemoveAll(fp); err != nil {
		return fmt.Errorf("removing %q: %w", path, err)
	}

	return w.CheckForFileChanges(ctx)
}

// WriteFiles writes the text file content to workdir-relative paths and
// notifies watchers of the changes.
func (w *Workdir) WriteFiles(ctx context.Context, files map[string]string) error {
	for path, content := range files {
		fp := w.AbsPath(path)
		_, err := os.Stat(fp)
		if err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("checking if %q exists: %w", path, err)
		}
		if err := writeFileData(path, []byte(content), w.RelativeTo); err != nil {
			return err
		}
	}
	return w.CheckForFileChanges(ctx)
}

// WriteFile writes text file content to a workdir-relative path and notifies
// watchers of the change.
func (w *Workdir) WriteFile(ctx context.Context, path, content string) error {
	return w.WriteFiles(ctx, map[string]string{path: content})
}

// RenameFile performs an on disk-renaming of the workdir-relative oldPath to
// workdir-relative newPath, and notifies watchers of the changes.
//
// oldPath must either be a regular file or in the same directory as newPath.
func (w *Workdir) RenameFile(ctx context.Context, oldPath, newPath string) error {
	oldAbs := w.AbsPath(oldPath)
	newAbs := w.AbsPath(newPath)

	// For os.Rename, “OS-specific restrictions may apply when oldpath and newpath
	// are in different directories.” If that applies here, we may fall back to
	// ReadFile, WriteFile, and RemoveFile to perform the rename non-atomically.
	//
	// However, the fallback path only works for regular files: renaming a
	// directory would be much more complex and isn't needed for our tests.
	fallbackOk := false
	if filepath.Dir(oldAbs) != filepath.Dir(newAbs) {
		fi, err := os.Stat(oldAbs)
		if err == nil && !fi.Mode().IsRegular() {
			return &os.PathError{
				Op:   "RenameFile",
				Path: oldPath,
				Err:  fmt.Errorf("%w: file is not regular and not in the same directory as %s", os.ErrInvalid, newPath),
			}
		}
		fallbackOk = true
	}

	var renameErr error
	const debugFallback = false
	if fallbackOk && debugFallback {
		renameErr = fmt.Errorf("%w: debugging fallback path", os.ErrInvalid)
	} else {
		renameErr = robustio.Rename(oldAbs, newAbs)
	}
	if renameErr != nil {
		if !fallbackOk {
			return renameErr // The OS-specific Rename restrictions do not apply.
		}

		content, err := w.ReadFile(oldPath)
		if err != nil {
			// If we can't even read the file, the error from Rename may be accurate.
			return renameErr
		}
		fi, err := os.Stat(newAbs)
		if err == nil {
			if fi.IsDir() {
				// “If newpath already exists and is not a directory, Rename replaces it.”
				// But if it is a directory, maybe not?
				return renameErr
			}
			// On most platforms, Rename replaces the named file with a new file,
			// rather than overwriting the existing file it in place. Mimic that
			// behavior here.
			if err := robustio.RemoveAll(newAbs); err != nil {
				// Maybe we don't have permission to replace newPath?
				return renameErr
			}
		} else if !os.IsNotExist(err) {
			// If the destination path already exists or there is some problem with it,
			// the error from Rename may be accurate.
			return renameErr
		}
		if writeErr := writeFileData(newPath, []byte(content), w.RelativeTo); writeErr != nil {
			// At this point we have tried to actually write the file.
			// If it still doesn't exist, assume that the error from Rename was accurate:
			// for example, maybe we don't have permission to create the new path.
			// Otherwise, return the error from the write, which may indicate some
			// other problem (such as a full disk).
			if _, statErr := os.Stat(newAbs); !os.IsNotExist(statErr) {
				return writeErr
			}
			return renameErr
		}
		if err := robustio.RemoveAll(oldAbs); err != nil {
			// If we failed to remove the old file, that may explain the Rename error too.
			// Make a best effort to back out the write to the new path.
			robustio.RemoveAll(newAbs)
			return renameErr
		}
	}

	return w.CheckForFileChanges(ctx)
}

// ListFiles returns a new sorted list of the relative paths of files in dir,
// recursively.
func (w *Workdir) ListFiles(dir string) ([]string, error) {
	absDir := w.AbsPath(dir)
	var paths []string
	if err := filepath.Walk(absDir, func(fp string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.Mode()&(fs.ModeDir|fs.ModeSymlink) == 0 {
			paths = append(paths, w.RelPath(fp))
		}
		return nil
	}); err != nil {
		return nil, err
	}
	sort.Strings(paths)
	return paths, nil
}

// CheckForFileChanges walks the working directory and checks for any files
// that have changed since the last poll.
func (w *Workdir) CheckForFileChanges(ctx context.Context) error {
	evts, err := w.pollFiles()
	if err != nil {
		return err
	}
	if len(evts) == 0 {
		return nil
	}
	w.watcherMu.Lock()
	watchers := make([]func(context.Context, []protocol.FileEvent), len(w.watchers))
	copy(watchers, w.watchers)
	w.watcherMu.Unlock()
	for _, w := range watchers {
		w(ctx, evts)
	}
	return nil
}

// pollFiles updates w.files and calculates FileEvents corresponding to file
// state changes since the last poll. It does not call sendEvents.
func (w *Workdir) pollFiles() ([]protocol.FileEvent, error) {
	w.fileMu.Lock()
	defer w.fileMu.Unlock()

	newFiles := make(map[string]fileID)
	var evts []protocol.FileEvent
	if err := filepath.Walk(string(w.RelativeTo), func(fp string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		// Skip directories and symbolic links (which may be links to directories).
		//
		// The latter matters for repos like Kubernetes, which use symlinks.
		if info.Mode()&(fs.ModeDir|fs.ModeSymlink) != 0 {
			return nil
		}

		// Opt: avoid reading the file if mtime is sufficiently old to be reliable.
		//
		// If mtime is recent, it may not sufficiently identify the file contents:
		// a subsequent write could result in the same mtime. For these cases, we
		// must read the file contents.
		id := fileID{mtime: info.ModTime()}
		if time.Since(info.ModTime()) < 2*time.Second {
			data, err := os.ReadFile(fp)
			if err != nil {
				return err
			}
			id.hash = hashFile(data)
		}
		path := w.RelPath(fp)
		newFiles[path] = id

		if w.files != nil {
			oldID, ok := w.files[path]
			delete(w.files, path)
			switch {
			case !ok:
				evts = append(evts, protocol.FileEvent{
					URI:  w.URI(path),
					Type: protocol.Created,
				})
			case oldID != id:
				changed := true

				// Check whether oldID and id do not match because oldID was polled at
				// a recent enough to time such as to require hashing.
				//
				// In this case, read the content to check whether the file actually
				// changed.
				if oldID.mtime.Equal(id.mtime) && oldID.hash != "" && id.hash == "" {
					data, err := os.ReadFile(fp)
					if err != nil {
						return err
					}
					if hashFile(data) == oldID.hash {
						changed = false
					}
				}
				if changed {
					evts = append(evts, protocol.FileEvent{
						URI:  w.URI(path),
						Type: protocol.Changed,
					})
				}
			}
		}

		return nil
	}); err != nil {
		return nil, err
	}

	// Any remaining files must have been deleted.
	for path := range w.files {
		evts = append(evts, protocol.FileEvent{
			URI:  w.URI(path),
			Type: protocol.Deleted,
		})
	}
	w.files = newFiles
	return evts, nil
}
