// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// File locking support for Plan 9. This uses fdMutex from the
// internal/poll package.

// incref adds a reference to the file. It returns an error if the file
// is already closed. This method is on File so that we can incorporate
// a nil test.
func (f *File) incref(op string) (err error) {
	if f == nil {
		return ErrInvalid
	}
	if !f.fdmu.Incref() {
		err = ErrClosed
		if op != "" {
			err = &PathError{Op: op, Path: f.name, Err: err}
		}
	}
	return err
}

// decref removes a reference to the file. If this is the last
// remaining reference, and the file has been marked to be closed,
// then actually close it.
func (file *file) decref() error {
	if file.fdmu.Decref() {
		return file.destroy()
	}
	return nil
}

// readLock adds a reference to the file and locks it for reading.
// It returns an error if the file is already closed.
func (file *file) readLock() error {
	if !file.fdmu.ReadLock() {
		return ErrClosed
	}
	return nil
}

// readUnlock removes a reference from the file and unlocks it for reading.
// It also closes the file if it marked as closed and there is no remaining
// reference.
func (file *file) readUnlock() {
	if file.fdmu.ReadUnlock() {
		file.destroy()
	}
}

// writeLock adds a reference to the file and locks it for writing.
// It returns an error if the file is already closed.
func (file *file) writeLock() error {
	if !file.fdmu.WriteLock() {
		return ErrClosed
	}
	return nil
}

// writeUnlock removes a reference from the file and unlocks it for writing.
// It also closes the file if it is marked as closed and there is no remaining
// reference.
func (file *file) writeUnlock() {
	if file.fdmu.WriteUnlock() {
		file.destroy()
	}
}
