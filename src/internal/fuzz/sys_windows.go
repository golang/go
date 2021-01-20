// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

type sharedMemSys struct {
	mapObj syscall.Handle
}

func sharedMemMapFile(f *os.File, size int, removeOnClose bool) (*sharedMem, error) {
	// Create a file mapping object. The object itself is not shared.
	mapObj, err := syscall.CreateFileMapping(
		syscall.Handle(f.Fd()), // fhandle
		nil,                    // sa
		syscall.PAGE_READWRITE, // prot
		0,                      // maxSizeHigh
		0,                      // maxSizeLow
		nil,                    // name
	)
	if err != nil {
		return nil, err
	}

	// Create a view from the file mapping object.
	access := uint32(syscall.FILE_MAP_READ | syscall.FILE_MAP_WRITE)
	addr, err := syscall.MapViewOfFile(
		mapObj,        // handle
		access,        // access
		0,             // offsetHigh
		0,             // offsetLow
		uintptr(size), // length
	)
	if err != nil {
		syscall.CloseHandle(mapObj)
		return nil, err
	}

	var region []byte
	header := (*reflect.SliceHeader)(unsafe.Pointer(&region))
	header.Data = addr
	header.Len = size
	header.Cap = size
	return &sharedMem{
		f:             f,
		region:        region,
		removeOnClose: removeOnClose,
		sys:           sharedMemSys{mapObj: mapObj},
	}, nil
}

// Close unmaps the shared memory and closes the temporary file. If this
// sharedMem was created with sharedMemTempFile, Close also removes the file.
func (m *sharedMem) Close() error {
	// Attempt all operations, even if we get an error for an earlier operation.
	// os.File.Close may fail due to I/O errors, but we still want to delete
	// the temporary file.
	var errs []error
	errs = append(errs,
		syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&m.region[0]))),
		syscall.CloseHandle(m.sys.mapObj),
		m.f.Close())
	if m.removeOnClose {
		errs = append(errs, os.Remove(m.f.Name()))
	}
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// setWorkerComm configures communciation channels on the cmd that will
// run a worker process.
func setWorkerComm(cmd *exec.Cmd, comm workerComm) {
	mem := <-comm.memMu
	memFD := mem.f.Fd()
	comm.memMu <- mem
	syscall.SetHandleInformation(syscall.Handle(comm.fuzzIn.Fd()), syscall.HANDLE_FLAG_INHERIT, 1)
	syscall.SetHandleInformation(syscall.Handle(comm.fuzzOut.Fd()), syscall.HANDLE_FLAG_INHERIT, 1)
	syscall.SetHandleInformation(syscall.Handle(memFD), syscall.HANDLE_FLAG_INHERIT, 1)
	cmd.Env = append(cmd.Env, fmt.Sprintf("GO_TEST_FUZZ_WORKER_HANDLES=%x,%x,%x", comm.fuzzIn.Fd(), comm.fuzzOut.Fd(), memFD))
}

// getWorkerComm returns communication channels in the worker process.
func getWorkerComm() (comm workerComm, err error) {
	v := os.Getenv("GO_TEST_FUZZ_WORKER_HANDLES")
	if v == "" {
		return workerComm{}, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES not set")
	}
	parts := strings.Split(v, ",")
	if len(parts) != 3 {
		return workerComm{}, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES has invalid value")
	}
	base := 16
	bitSize := 64
	handles := make([]syscall.Handle, len(parts))
	for i, s := range parts {
		h, err := strconv.ParseInt(s, base, bitSize)
		if err != nil {
			return workerComm{}, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES has invalid value: %v", err)
		}
		handles[i] = syscall.Handle(h)
	}

	fuzzIn := os.NewFile(uintptr(handles[0]), "fuzz_in")
	fuzzOut := os.NewFile(uintptr(handles[1]), "fuzz_out")
	tmpFile := os.NewFile(uintptr(handles[2]), "fuzz_mem")
	fi, err := tmpFile.Stat()
	if err != nil {
		return workerComm{}, err
	}
	size := int(fi.Size())
	if int64(size) != fi.Size() {
		return workerComm{}, fmt.Errorf("fuzz temp file exceeds maximum size")
	}
	removeOnClose := false
	mem, err := sharedMemMapFile(tmpFile, size, removeOnClose)
	if err != nil {
		return workerComm{}, err
	}
	memMu := make(chan *sharedMem, 1)
	memMu <- mem

	return workerComm{fuzzIn: fuzzIn, fuzzOut: fuzzOut, memMu: memMu}, nil
}

func isInterruptError(err error) bool {
	// TODO(jayconrod): implement
	return false
}
