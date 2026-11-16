// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && windows

package work

import (
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/windows"
)

const earlyExportSupported = true

var (
	exportPipePrefix = sync.OnceValue(func() string {
		var b [8]byte
		if _, err := rand.Read(b[:]); err != nil {
			panic(err)
		}
		return fmt.Sprintf(`\\.\pipe\go-export-%d-%x-`, os.Getpid(), b)
	})
	exportPipeSeq atomic.Uint64

	// Restrict named pipe access to the current user (matching mode 0600 on Unix).
	// The default named pipe DACL grants read access to Everyone and Anonymous,
	// and pipe names under \\.\pipe\ can be enumerated by other local users.
	exportPipeSecurityDescriptor = sync.OnceValues(func() (*windows.SECURITY_DESCRIPTOR, error) {
		user, err := windows.GetCurrentProcessToken().GetTokenUser()
		if err != nil {
			return nil, fmt.Errorf("get export pipe owner SID: %w", err)
		}
		sd, err := windows.SecurityDescriptorFromString("D:P(A;;GA;;;" + user.User.Sid.String() + ")")
		if err != nil {
			return nil, fmt.Errorf("create export pipe security descriptor: %w", err)
		}
		return sd, nil
	})
)

type exportPipe struct {
	pipeName  string
	h         windows.Handle
	closeEv   windows.Handle
	mu        sync.Mutex
	closed    bool
	closeOnce sync.Once
}

func newExportPipe(objdir string) (*exportPipe, error) {
	pipeName := exportPipePrefix() + strconv.FormatUint(exportPipeSeq.Add(1), 10)
	wname, err := windows.UTF16PtrFromString(pipeName)
	if err != nil {
		return nil, err
	}
	sd, err := exportPipeSecurityDescriptor()
	if err != nil {
		return nil, err
	}
	sa := &windows.SecurityAttributes{
		Length:             uint32(unsafe.Sizeof(windows.SecurityAttributes{})),
		SecurityDescriptor: sd,
	}
	openMode := uint32(windows.PIPE_ACCESS_INBOUND | windows.FILE_FLAG_FIRST_PIPE_INSTANCE | windows.FILE_FLAG_OVERLAPPED)
	pipeMode := uint32(windows.PIPE_TYPE_BYTE | windows.PIPE_READMODE_BYTE | windows.PIPE_REJECT_REMOTE_CLIENTS)
	h, err := windows.CreateNamedPipe(wname, openMode, pipeMode, 1, 0, 64, 0, sa)
	if err != nil {
		return nil, err
	}
	closeEv, err := windows.CreateEvent(nil, 1, 0, nil)
	if err != nil {
		windows.CloseHandle(h)
		return nil, err
	}
	return &exportPipe{
		pipeName: pipeName,
		h:        h,
		closeEv:  closeEv,
	}, nil
}

func (ep *exportPipe) name() string { return ep.pipeName }

func (ep *exportPipe) wait() error {
	ep.mu.Lock()
	defer ep.mu.Unlock()
	if ep.closed {
		return io.EOF
	}
	ioEv, err := windows.CreateEvent(nil, 1, 0, nil)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(ioEv)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	ov := new(windows.Overlapped)
	ov.HEvent = ioEv
	pinner.Pin(ov)

	if err := windows.ConnectNamedPipe(ep.h, ov); err != nil {
		switch err {
		case windows.ERROR_PIPE_CONNECTED, windows.ERROR_NO_DATA:
			// Client already connected (and possibly already wrote and closed).
		case windows.ERROR_IO_PENDING:
			if _, err := waitOverlapped(ep.h, ioEv, ep.closeEv, ov); err != nil {
				return err
			}
		default:
			return err
		}
	}
	return readOverlappedByte(ep.h, ioEv, ep.closeEv)
}

func (ep *exportPipe) close() {
	ep.closeOnce.Do(func() {
		windows.SetEvent(ep.closeEv)
		ep.mu.Lock()
		ep.closed = true
		windows.CloseHandle(ep.h)
		windows.CloseHandle(ep.closeEv)
		ep.mu.Unlock()
	})
}

func readOverlappedByte(h, ioEv, closeEv windows.Handle) error {
	var pinner runtime.Pinner
	defer pinner.Unpin()
	ov := new(windows.Overlapped)
	ov.HEvent = ioEv
	buf := new([1]byte)
	pinner.Pin(ov)
	pinner.Pin(buf)

	windows.ResetEvent(ioEv)
	var n uint32
	err := windows.ReadFile(h, buf[:], &n, ov)
	switch err {
	case nil:
		err = windows.GetOverlappedResult(h, ov, &n, true)
	case windows.ERROR_IO_PENDING:
		n, err = waitOverlapped(h, ioEv, closeEv, ov)
	}
	if err != nil {
		if errors.Is(err, windows.ERROR_BROKEN_PIPE) || errors.Is(err, windows.ERROR_NO_DATA) {
			return io.EOF
		}
		return err
	}
	if n == 0 {
		return io.EOF
	}
	return nil
}

func waitOverlapped(h, ioEv, closeEv windows.Handle, ov *windows.Overlapped) (uint32, error) {
	ev, err := windows.WaitForMultipleObjects([]windows.Handle{ioEv, closeEv}, false, windows.INFINITE)
	if err != nil {
		windows.CancelIoEx(h, ov)
		var n uint32
		windows.GetOverlappedResult(h, ov, &n, true)
		return 0, err
	}
	if ev == windows.WAIT_OBJECT_0+1 {
		windows.CancelIoEx(h, ov)
		var n uint32
		if gerr := windows.GetOverlappedResult(h, ov, &n, true); gerr == nil {
			return n, nil
		}
		return 0, io.EOF
	}
	var n uint32
	if err := windows.GetOverlappedResult(h, ov, &n, true); err != nil {
		return 0, err
	}
	return n, nil
}
