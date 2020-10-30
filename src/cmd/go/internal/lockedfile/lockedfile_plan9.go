// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package lockedfile

import (
	"io/fs"
	"math/rand"
	"os"
	"strings"
	"time"

	"cmd/go/internal/fsys"
)

// Opening an exclusive-use file returns an error.
// The expected error strings are:
//
//  - "open/create -- file is locked" (cwfs, kfs)
//  - "exclusive lock" (fossil)
//  - "exclusive use file already open" (ramfs)
var lockedErrStrings = [...]string{
	"file is locked",
	"exclusive lock",
	"exclusive use file already open",
}

// Even though plan9 doesn't support the Lock/RLock/Unlock functions to
// manipulate already-open files, IsLocked is still meaningful: os.OpenFile
// itself may return errors that indicate that a file with the ModeExclusive bit
// set is already open.
func isLocked(err error) bool {
	s := err.Error()

	for _, frag := range lockedErrStrings {
		if strings.Contains(s, frag) {
			return true
		}
	}

	return false
}

func openFile(name string, flag int, perm fs.FileMode) (*os.File, error) {
	// Plan 9 uses a mode bit instead of explicit lock/unlock syscalls.
	//
	// Per http://man.cat-v.org/plan_9/5/stat: “Exclusive use files may be open
	// for I/O by only one fid at a time across all clients of the server. If a
	// second open is attempted, it draws an error.”
	//
	// So we can try to open a locked file, but if it fails we're on our own to
	// figure out when it becomes available. We'll use exponential backoff with
	// some jitter and an arbitrary limit of 500ms.

	// If the file was unpacked or created by some other program, it might not
	// have the ModeExclusive bit set. Set it before we call OpenFile, so that we
	// can be confident that a successful OpenFile implies exclusive use.
	if fi, err := fsys.Stat(name); err == nil {
		if fi.Mode()&fs.ModeExclusive == 0 {
			if err := os.Chmod(name, fi.Mode()|fs.ModeExclusive); err != nil {
				return nil, err
			}
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	nextSleep := 1 * time.Millisecond
	const maxSleep = 500 * time.Millisecond
	for {
		f, err := fsys.OpenFile(name, flag, perm|fs.ModeExclusive)
		if err == nil {
			return f, nil
		}

		if !isLocked(err) {
			return nil, err
		}

		time.Sleep(nextSleep)

		nextSleep += nextSleep
		if nextSleep > maxSleep {
			nextSleep = maxSleep
		}
		// Apply 10% jitter to avoid synchronizing collisions.
		nextSleep += time.Duration((0.1*rand.Float64() - 0.05) * float64(nextSleep))
	}
}

func closeFile(f *os.File) error {
	return f.Close()
}
