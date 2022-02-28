// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"fmt"
	"io/ioutil"
	"os"
	"unsafe"
)

// sharedMem manages access to a region of virtual memory mapped from a file,
// shared between multiple processes. The region includes space for a header and
// a value of variable length.
//
// When fuzzing, the coordinator creates a sharedMem from a temporary file for
// each worker. This buffer is used to pass values to fuzz between processes.
// Care must be taken to manage access to shared memory across processes;
// sharedMem provides no synchronization on its own. See workerComm for an
// explanation.
type sharedMem struct {
	// f is the file mapped into memory.
	f *os.File

	// region is the mapped region of virtual memory for f. The content of f may
	// be read or written through this slice.
	region []byte

	// removeOnClose is true if the file should be deleted by Close.
	removeOnClose bool

	// sys contains OS-specific information.
	sys sharedMemSys
}

// sharedMemHeader stores metadata in shared memory.
type sharedMemHeader struct {
	// count is the number of times the worker has called the fuzz function.
	// May be reset by coordinator.
	count int64

	// valueLen is the number of bytes in region which should be read.
	valueLen int

	// randState and randInc hold the state of a pseudo-random number generator.
	randState, randInc uint64

	// rawInMem is true if the region holds raw bytes, which occurs during
	// minimization. If true after the worker fails during minimization, this
	// indicates that an unrecoverable error occurred, and the region can be
	// used to retrive the raw bytes that caused the error.
	rawInMem bool
}

// sharedMemSize returns the size needed for a shared memory buffer that can
// contain values of the given size.
func sharedMemSize(valueSize int) int {
	// TODO(jayconrod): set a reasonable maximum size per platform.
	return int(unsafe.Sizeof(sharedMemHeader{})) + valueSize
}

// sharedMemTempFile creates a new temporary file of the given size, then maps
// it into memory. The file will be removed when the Close method is called.
func sharedMemTempFile(size int) (m *sharedMem, err error) {
	// Create a temporary file.
	f, err := ioutil.TempFile("", "fuzz-*")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()

	// Resize it to the correct size.
	totalSize := sharedMemSize(size)
	if err := f.Truncate(int64(totalSize)); err != nil {
		return nil, err
	}

	// Map the file into memory.
	removeOnClose := true
	return sharedMemMapFile(f, totalSize, removeOnClose)
}

// header returns a pointer to metadata within the shared memory region.
func (m *sharedMem) header() *sharedMemHeader {
	return (*sharedMemHeader)(unsafe.Pointer(&m.region[0]))
}

// valueRef returns the value currently stored in shared memory. The returned
// slice points to shared memory; it is not a copy.
func (m *sharedMem) valueRef() []byte {
	length := m.header().valueLen
	valueOffset := int(unsafe.Sizeof(sharedMemHeader{}))
	return m.region[valueOffset : valueOffset+length]
}

// valueCopy returns a copy of the value stored in shared memory.
func (m *sharedMem) valueCopy() []byte {
	ref := m.valueRef()
	b := make([]byte, len(ref))
	copy(b, ref)
	return b
}

// setValue copies the data in b into the shared memory buffer and sets
// the length. len(b) must be less than or equal to the capacity of the buffer
// (as returned by cap(m.value())).
func (m *sharedMem) setValue(b []byte) {
	v := m.valueRef()
	if len(b) > cap(v) {
		panic(fmt.Sprintf("value length %d larger than shared memory capacity %d", len(b), cap(v)))
	}
	m.header().valueLen = len(b)
	copy(v[:cap(v)], b)
}

// setValueLen sets the length of the shared memory buffer returned by valueRef
// to n, which may be at most the cap of that slice.
//
// Note that we can only store the length in the shared memory header. The full
// slice header contains a pointer, which is likely only valid for one process,
// since each process can map shared memory at a different virtual address.
func (m *sharedMem) setValueLen(n int) {
	v := m.valueRef()
	if n > cap(v) {
		panic(fmt.Sprintf("length %d larger than shared memory capacity %d", n, cap(v)))
	}
	m.header().valueLen = n
}

// TODO(jayconrod): add method to resize the buffer. We'll need that when the
// mutator can increase input length. Only the coordinator will be able to
// do it, since we'll need to send a message to the worker telling it to
// remap the file.
