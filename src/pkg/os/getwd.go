// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
)

// Getwd returns a rooted path name corresponding to the
// current directory.  If the current directory can be
// reached via multiple paths (due to symbolic links),
// Getwd may return any one of them.
func Getwd() (string, Error) {
	// If the operating system provides a Getwd call, use it.
	if syscall.ImplementsGetwd {
		s, e := syscall.Getwd()
		return s, NewSyscallError("getwd", e)
	}

	// Otherwise, we're trying to find our way back to ".".
	dot, err := Stat(".")
	if err != nil {
		return "", err
	}

	// Clumsy but widespread kludge:
	// if $PWD is set and matches ".", use it.
	pwd := Getenv("PWD")
	if len(pwd) > 0 && pwd[0] == '/' {
		d, err := Stat(pwd)
		if err == nil && d.Dev == dot.Dev && d.Ino == dot.Ino {
			return pwd, nil
		}
	}

	// Root is a special case because it has no parent
	// and ends in a slash.
	root, err := Stat("/")
	if err != nil {
		// Can't stat root - no hope.
		return "", err
	}
	if root.Dev == dot.Dev && root.Ino == dot.Ino {
		return "/", nil
	}

	// General algorithm: find name in parent
	// and then find name of parent.  Each iteration
	// adds /name to the beginning of pwd.
	pwd = ""
	for parent := ".."; ; parent = "../" + parent {
		if len(parent) >= 1024 { // Sanity check
			return "", ENAMETOOLONG
		}
		fd, err := Open(parent, O_RDONLY, 0)
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
				if d.Dev == dot.Dev && d.Ino == dot.Ino {
					pwd = "/" + name + pwd
					goto Found
				}
			}
		}
		fd.Close()
		return "", ENOENT

	Found:
		pd, err := fd.Stat()
		if err != nil {
			return "", err
		}
		fd.Close()
		if pd.Dev == root.Dev && pd.Ino == root.Ino {
			break
		}
		// Set up for next round.
		dot = pd
	}
	return pwd, nil
}
