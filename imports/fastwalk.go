// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A faster implementation of filepath.Walk.
//
// filepath.Walk's design necessarily calls os.Lstat on each file,
// even if the caller needs less info. And goimports only need to know
// the type of each file. The kernel interface provides the type in
// the Readdir call but the standard library ignored it.
// fastwalk_unix.go contains a fork of the syscall routines.
//
// See golang.org/issue/16399

package imports

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
)

// traverseLink is a sentinel error for fastWalk, similar to filepath.SkipDir.
var traverseLink = errors.New("traverse symlink, assuming target is a directory")

// fastWalk walks the file tree rooted at root, calling walkFn for
// each file or directory in the tree, including root.
//
// If fastWalk returns filepath.SkipDir, the directory is skipped.
//
// Unlike filepath.Walk:
//   * file stat calls must be done by the user.
//     The only provided metadata is the file type, which does not include
//     any permission bits.
//   * multiple goroutines stat the filesystem concurrently. The provided
//     walkFn must be safe for concurrent use.
//   * fastWalk can follow symlinks if walkFn returns the traverseLink
//     sentinel error. It is the walkFn's responsibility to prevent
//     fastWalk from going into symlink cycles.
func fastWalk(root string, walkFn func(path string, typ os.FileMode) error) error {
	// TODO(bradfitz): make numWorkers configurable? We used a
	// minimum of 4 to give the kernel more info about multiple
	// things we want, in hopes its I/O scheduling can take
	// advantage of that. Hopefully most are in cache. Maybe 4 is
	// even too low of a minimum. Profile more.
	numWorkers := 4
	if n := runtime.NumCPU(); n > numWorkers {
		numWorkers = n
	}
	w := &walker{
		fn:       walkFn,
		enqueuec: make(chan walkItem, numWorkers), // buffered for performance
		workc:    make(chan walkItem, numWorkers), // buffered for performance
		donec:    make(chan struct{}),

		// buffered for correctness & not leaking goroutines:
		resc: make(chan error, numWorkers),
	}
	defer close(w.donec)
	// TODO(bradfitz): start the workers as needed? maybe not worth it.
	for i := 0; i < numWorkers; i++ {
		go w.doWork()
	}
	todo := []walkItem{{dir: root}}
	out := 0
	for {
		workc := w.workc
		var workItem walkItem
		if len(todo) == 0 {
			workc = nil
		} else {
			workItem = todo[len(todo)-1]
		}
		select {
		case workc <- workItem:
			todo = todo[:len(todo)-1]
			out++
		case it := <-w.enqueuec:
			todo = append(todo, it)
		case err := <-w.resc:
			out--
			if err != nil {
				return err
			}
			if out == 0 && len(todo) == 0 {
				// It's safe to quit here, as long as the buffered
				// enqueue channel isn't also readable, which might
				// happen if the worker sends both another unit of
				// work and its result before the other select was
				// scheduled and both w.resc and w.enqueuec were
				// readable.
				select {
				case it := <-w.enqueuec:
					todo = append(todo, it)
				default:
					return nil
				}
			}
		}
	}
}

// doWork reads directories as instructed (via workc) and runs the
// user's callback function.
func (w *walker) doWork() {
	for {
		select {
		case <-w.donec:
			return
		case it := <-w.workc:
			w.resc <- w.walk(it.dir, !it.callbackDone)
		}
	}
}

type walker struct {
	fn func(path string, typ os.FileMode) error

	donec    chan struct{} // closed on fastWalk's return
	workc    chan walkItem // to workers
	enqueuec chan walkItem // from workers
	resc     chan error    // from workers
}

type walkItem struct {
	dir          string
	callbackDone bool // callback already called; don't do it again
}

func (w *walker) enqueue(it walkItem) {
	select {
	case w.enqueuec <- it:
	case <-w.donec:
	}
}

func (w *walker) onDirEnt(dirName, baseName string, typ os.FileMode) error {
	joined := dirName + string(os.PathSeparator) + baseName
	if typ == os.ModeDir {
		w.enqueue(walkItem{dir: joined})
		return nil
	}

	err := w.fn(joined, typ)
	if typ == os.ModeSymlink {
		if err == traverseLink {
			// Set callbackDone so we don't call it twice for both the
			// symlink-as-symlink and the symlink-as-directory later:
			w.enqueue(walkItem{dir: joined, callbackDone: true})
			return nil
		}
		if err == filepath.SkipDir {
			// Permit SkipDir on symlinks too.
			return nil
		}
	}
	return err
}
func (w *walker) walk(root string, runUserCallback bool) error {
	if runUserCallback {
		err := w.fn(root, os.ModeDir)
		if err == filepath.SkipDir {
			return nil
		}
		if err != nil {
			return err
		}
	}

	return readDir(root, w.onDirEnt)
}
