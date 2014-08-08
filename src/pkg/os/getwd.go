// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"runtime"
	"sync"
	"syscall"
)

var getwdCache struct {
	sync.Mutex
	dir string
}

// useSyscallwd determines whether to use the return value of
// syscall.Getwd based on its error.
var useSyscallwd = func(error) bool { return true }

// Getwd returns a rooted path name corresponding to the
// current directory.  If the current directory can be
// reached via multiple paths (due to symbolic links),
// Getwd may return any one of them.
func Getwd() (dir string, err error) {
	if runtime.GOOS == "windows" {
		return syscall.Getwd()
	}

	// Clumsy but widespread kludge:
	// if $PWD is set and matches ".", use it.
	dot, err := Stat(".")
	if err != nil {
		return "", err
	}
	dir = Getenv("PWD")
	if len(dir) > 0 && dir[0] == '/' {
		d, err := Stat(dir)
		if err == nil && SameFile(dot, d) {
			return dir, nil
		}
	}

	// If the operating system provides a Getwd call, use it.
	// Otherwise, we're trying to find our way back to ".".
	if syscall.ImplementsGetwd {
		s, e := syscall.Getwd()
		if useSyscallwd(e) {
			return s, NewSyscallError("getwd", e)
		}
	}

	// Apply same kludge but to cached dir instead of $PWD.
	getwdCache.Lock()
	dir = getwdCache.dir
	getwdCache.Unlock()
	if len(dir) > 0 {
		d, err := Stat(dir)
		if err == nil && SameFile(dot, d) {
			return dir, nil
		}
	}

	// Root is a special case because it has no parent
	// and ends in a slash.
	root, err := Stat("/")
	if err != nil {
		// Can't stat root - no hope.
		return "", err
	}
	if SameFile(root, dot) {
		return "/", nil
	}

	// General algorithm: find name in parent
	// and then find name of parent.  Each iteration
	// adds /name to the beginning of dir.
	dir = ""
	for parent := ".."; ; parent = "../" + parent {
		if len(parent) >= 1024 { // Sanity check
			return "", syscall.ENAMETOOLONG
		}
		fd, err := Open(parent)
		if err != nil {
			return "", err
		}

		for {
			names, err := fd.Readdirnames(100)
			if err != nil {
				fd.Close()
				return "", err
			}
			for _, name := range names {
				d, _ := Lstat(parent + "/" + name)
				if SameFile(d, dot) {
					dir = "/" + name + dir
					goto Found
				}
			}
		}

	Found:
		pd, err := fd.Stat()
		if err != nil {
			return "", err
		}
		fd.Close()
		if SameFile(pd, root) {
			break
		}
		// Set up for next round.
		dot = pd
	}

	// Save answer as hint to avoid the expensive path next time.
	getwdCache.Lock()
	getwdCache.dir = dir
	getwdCache.Unlock()

	return dir, nil
}
