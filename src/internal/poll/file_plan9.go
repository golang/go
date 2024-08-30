// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

// Expose fdMutex for use by the os package on Plan 9.
// On Plan 9 we don't want to use async I/O for file operations,
// but we still want the locking semantics that fdMutex provides.

// FDMutex is an exported fdMutex, only for Plan 9.
type FDMutex struct {
	fdmu fdMutex
}

func (fdmu *FDMutex) Incref() bool {
	return fdmu.fdmu.incref()
}

func (fdmu *FDMutex) Decref() bool {
	return fdmu.fdmu.decref()
}

func (fdmu *FDMutex) IncrefAndClose() bool {
	return fdmu.fdmu.increfAndClose()
}

func (fdmu *FDMutex) ReadLock() bool {
	return fdmu.fdmu.rwlock(true)
}

func (fdmu *FDMutex) ReadUnlock() bool {
	return fdmu.fdmu.rwunlock(true)
}

func (fdmu *FDMutex) WriteLock() bool {
	return fdmu.fdmu.rwlock(false)
}

func (fdmu *FDMutex) WriteUnlock() bool {
	return fdmu.fdmu.rwunlock(false)
}
