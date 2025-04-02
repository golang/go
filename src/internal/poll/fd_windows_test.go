// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"bytes"
	"errors"
	"internal/poll"
	"internal/syscall/windows"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

func init() {
	poll.InitWSA()
}

// checkFileIsNotPartOfNetpoll verifies that f is not managed by netpoll.
func checkFileIsNotPartOfNetpoll(t *testing.T, f *os.File) {
	t.Helper()
	sc, err := f.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	if err := sc.Control(func(fd uintptr) {
		// Only try to associate the file with an IOCP if the handle is opened for overlapped I/O,
		// else the association will always fail.
		overlapped, err := windows.IsNonblock(syscall.Handle(fd))
		if err != nil {
			t.Fatalf("%v fd=%v: %v", f.Name(), fd, err)
		}
		if overlapped {
			// If the file is part of netpoll, then associating it with another IOCP should fail.
			if _, err := windows.CreateIoCompletionPort(syscall.Handle(fd), 0, 0, 1); err != nil {
				t.Fatalf("%v fd=%v: is part of netpoll, but should not be: %v", f.Name(), fd, err)
			}
		}
	}); err != nil {
		t.Fatalf("%v fd=%v: is not initialized", f.Name(), f.Fd())
	}
}

func TestFileFdsAreInitialised(t *testing.T) {
	t.Parallel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	checkFileIsNotPartOfNetpoll(t, f)
}

func TestSerialFdsAreInitialised(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"COM1", "COM2", "COM3", "COM4"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			h, err := syscall.CreateFile(syscall.StringToUTF16Ptr(name),
				syscall.GENERIC_READ|syscall.GENERIC_WRITE,
				0,
				nil,
				syscall.OPEN_EXISTING,
				syscall.FILE_ATTRIBUTE_NORMAL|syscall.FILE_FLAG_OVERLAPPED,
				0)
			if err != nil {
				if errno, ok := err.(syscall.Errno); ok {
					switch errno {
					case syscall.ERROR_FILE_NOT_FOUND,
						syscall.ERROR_ACCESS_DENIED:
						t.Log("Skipping: ", err)
						return
					}
				}
				t.Fatal(err)
			}
			f := os.NewFile(uintptr(h), name)
			defer f.Close()

			checkFileIsNotPartOfNetpoll(t, f)
		})
	}
}

func TestWSASocketConflict(t *testing.T) {
	t.Parallel()
	s, err := windows.WSASocket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP, nil, 0, windows.WSA_FLAG_OVERLAPPED)
	if err != nil {
		t.Fatal(err)
	}
	fd := poll.FD{Sysfd: s, IsStream: true, ZeroReadIsEOF: true}
	if err = fd.Init("tcp", true); err != nil {
		syscall.CloseHandle(s)
		t.Fatal(err)
	}
	defer fd.Close()

	const SIO_TCP_INFO = syscall.IOC_INOUT | syscall.IOC_VENDOR | 39
	inbuf := uint32(0)
	var outbuf _TCP_INFO_v0
	cbbr := uint32(0)

	var ov syscall.Overlapped
	// Create an event so that we can efficiently wait for completion
	// of a requested overlapped I/O operation.
	ov.HEvent, _ = windows.CreateEvent(nil, 0, 0, nil)
	if ov.HEvent == 0 {
		t.Fatalf("could not create the event!")
	}
	defer syscall.CloseHandle(ov.HEvent)

	if err = fd.WSAIoctl(
		SIO_TCP_INFO,
		(*byte)(unsafe.Pointer(&inbuf)),
		uint32(unsafe.Sizeof(inbuf)),
		(*byte)(unsafe.Pointer(&outbuf)),
		uint32(unsafe.Sizeof(outbuf)),
		&cbbr,
		&ov,
		0,
	); err != nil && !errors.Is(err, syscall.ERROR_IO_PENDING) {
		t.Fatalf("could not perform the WSAIoctl: %v", err)
	}

	if err != nil && errors.Is(err, syscall.ERROR_IO_PENDING) {
		// It is possible that the overlapped I/O operation completed
		// immediately so there is no need to wait for it to complete.
		if res, err := syscall.WaitForSingleObject(ov.HEvent, syscall.INFINITE); res != 0 {
			t.Fatalf("waiting for the completion of the overlapped IO failed: %v", err)
		}
	}
}

type _TCP_INFO_v0 struct {
	State             uint32
	Mss               uint32
	ConnectionTimeMs  uint64
	TimestampsEnabled bool
	RttUs             uint32
	MinRttUs          uint32
	BytesInFlight     uint32
	Cwnd              uint32
	SndWnd            uint32
	RcvWnd            uint32
	RcvBuf            uint32
	BytesOut          uint64
	BytesIn           uint64
	BytesReordered    uint32
	BytesRetrans      uint32
	FastRetrans       uint32
	DupAcksIn         uint32
	TimeoutEpisodes   uint32
	SynRetrans        uint8
}

func newFD(t testing.TB, h syscall.Handle, kind string, overlapped, pollable bool) *poll.FD {
	fd := poll.FD{
		Sysfd:         h,
		IsStream:      true,
		ZeroReadIsEOF: true,
	}
	err := fd.Init(kind, pollable)
	if overlapped && err != nil {
		// Overlapped file handles should not error.
		fd.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		fd.Close()
	})
	return &fd
}

func newFile(t testing.TB, name string, overlapped, pollable bool) *poll.FD {
	namep, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		t.Fatal(err)
	}
	flags := syscall.FILE_ATTRIBUTE_NORMAL
	if overlapped {
		flags |= syscall.FILE_FLAG_OVERLAPPED
	}
	h, err := syscall.CreateFile(namep,
		syscall.GENERIC_READ|syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_READ,
		nil, syscall.OPEN_ALWAYS, uint32(flags), 0)
	if err != nil {
		t.Fatal(err)
	}
	typ, err := syscall.GetFileType(h)
	if err != nil {
		syscall.CloseHandle(h)
		t.Fatal(err)
	}
	kind := "file"
	if typ == syscall.FILE_TYPE_PIPE {
		kind = "pipe"
	}
	return newFD(t, h, kind, overlapped, pollable)
}

var currentProces = sync.OnceValue(func() string {
	// Convert the process ID to a string.
	return strconv.FormatUint(uint64(os.Getpid()), 10)
})

var pipeCounter atomic.Uint64

func newBytePipe(t testing.TB, name string, overlapped, pollable bool) *poll.FD {
	return newPipe(t, name, false, overlapped, pollable)
}

func newMessagePipe(t testing.TB, name string, overlapped, pollable bool) *poll.FD {
	return newPipe(t, name, true, overlapped, pollable)
}

func pipeName() string {
	return `\\.\pipe\go-internal-poll-test-` + currentProces() + `-` + strconv.FormatUint(pipeCounter.Add(1), 10)
}

func newPipe(t testing.TB, name string, message, overlapped, pollable bool) *poll.FD {
	wname, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		t.Fatal(err)
	}
	// Create the read handle.
	flags := windows.PIPE_ACCESS_DUPLEX
	if overlapped {
		flags |= syscall.FILE_FLAG_OVERLAPPED
	}
	typ := windows.PIPE_TYPE_BYTE | windows.PIPE_READMODE_BYTE
	if message {
		typ = windows.PIPE_TYPE_MESSAGE | windows.PIPE_READMODE_MESSAGE
	}
	h, err := windows.CreateNamedPipe(wname, uint32(flags), uint32(typ), 1, 4096, 4096, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	return newFD(t, h, "pipe", overlapped, pollable)
}

func testReadWrite(t *testing.T, fdr, fdw *poll.FD) {
	write := make(chan string, 1)
	read := make(chan struct{}, 1)
	go func() {
		for s := range write {
			n, err := fdw.Write([]byte(s))
			read <- struct{}{}
			if err != nil {
				t.Error(err)
			}
			if n != len(s) {
				t.Errorf("expected to write %d bytes, got %d", len(s), n)
			}
		}
	}()
	for i := range 10 {
		s := strconv.Itoa(i)
		write <- s
		<-read
		buf := make([]byte, len(s))
		_, err := io.ReadFull(fdr, buf)
		if err != nil {
			t.Fatalf("read failed: %v", err)
		}
		if !bytes.Equal(buf, []byte(s)) {
			t.Fatalf("expected %q, got %q", s, buf)
		}
	}
	close(read)
	close(write)
}

func testPreadPwrite(t *testing.T, fdr, fdw *poll.FD) {
	type op struct {
		s   string
		off int64
	}
	write := make(chan op, 1)
	read := make(chan struct{}, 1)
	go func() {
		for o := range write {
			n, err := fdw.Pwrite([]byte(o.s), o.off)
			read <- struct{}{}
			if err != nil {
				t.Error(err)
			}
			if n != len(o.s) {
				t.Errorf("expected to write %d bytes, got %d", len(o.s), n)
			}
		}
	}()
	for i := range 10 {
		off := int64(i % 3) // exercise some back and forth
		s := strconv.Itoa(i)
		write <- op{s, off}
		<-read
		buf := make([]byte, len(s))
		n, err := fdr.Pread(buf, off)
		if err != nil {
			t.Fatal(err)
		}
		if n != len(s) {
			t.Fatalf("expected to read %d bytes, got %d", len(s), n)
		}
		if !bytes.Equal(buf, []byte(s)) {
			t.Fatalf("expected %q, got %q", s, buf)
		}
	}
	close(read)
	close(write)
}

func testFileReadEOF(t *testing.T, f *poll.FD) {
	end, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		t.Fatal(err)
	}
	var buf [1]byte
	n, err := f.Read(buf[:])
	if err != nil && err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}

	n, err = f.Pread(buf[:], end)
	if err != nil && err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func TestFile(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name            string
		overlappedRead  bool
		overlappedWrite bool
		pollable        bool
	}{
		{"overlapped", true, true, true},
		{"overlapped-nonpollable", true, true, false},
		{"overlapped-read", true, false, true},
		{"overlapped-write", false, true, true},
		{"sync", false, false, false},
		{"sync-pollable", false, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			name := filepath.Join(t.TempDir(), "foo")
			rh := newFile(t, name, tt.overlappedRead, tt.pollable)
			wh := newFile(t, name, tt.overlappedWrite, tt.pollable)
			testReadWrite(t, rh, wh)
			testPreadPwrite(t, rh, wh)
			testFileReadEOF(t, rh)
		})
	}
}

func TestPipe(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name            string
		overlappedRead  bool
		overlappedWrite bool
		pollable        bool
	}{
		{"overlapped", true, true, true},
		{"overlapped-nonpollable", true, true, false},
		{"overlapped-write", false, true, true},
		{"overlapped-read", true, false, true},
		{"sync", false, false, false},
		{"sync-pollable", false, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			name := pipeName()
			pipe := newBytePipe(t, name, tt.overlappedWrite, tt.pollable)
			file := newFile(t, name, tt.overlappedRead, tt.pollable)
			testReadWrite(t, pipe, file)
		})
	}
	t.Run("anonymous", func(t *testing.T) {
		t.Parallel()
		var r, w syscall.Handle
		if err := syscall.CreatePipe(&r, &w, nil, 0); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := syscall.CloseHandle(r); err != nil {
				t.Fatal(err)
			}
			if err := syscall.CloseHandle(w); err != nil {
				t.Fatal(err)
			}
		}()
		// CreatePipe always returns sync handles.
		fdr := newFD(t, r, "pipe", false, false)
		fdw := newFD(t, w, "file", false, false)
		testReadWrite(t, fdr, fdw)
	})
}

func TestPipeMessageReadEOF(t *testing.T) {
	t.Parallel()
	name := pipeName()
	pipe := newMessagePipe(t, name, true, true)
	file := newFile(t, name, true, true)

	_, err := pipe.Write(nil)
	if err != nil {
		t.Error(err)
	}

	var buf [10]byte
	n, err := file.Read(buf[:])
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func TestPipeClosedEOF(t *testing.T) {
	t.Parallel()
	name := pipeName()
	pipe := newBytePipe(t, name, true, false)
	file := newFile(t, name, true, true)

	pipe.Close()

	var buf [10]byte
	n, err := file.Read(buf[:])
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func TestPipeReadTimeout(t *testing.T) {
	t.Parallel()
	name := pipeName()
	_ = newBytePipe(t, name, true, true)
	file := newFile(t, name, true, true)

	err := file.SetReadDeadline(time.Now().Add(time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}

	var buf [10]byte
	_, err = file.Read(buf[:])
	if err != poll.ErrDeadlineExceeded {
		t.Errorf("expected deadline exceeded, got %v", err)
	}
}

func TestPipeCanceled(t *testing.T) {
	t.Parallel()
	name := pipeName()
	_ = newBytePipe(t, name, true, true)
	file := newFile(t, name, true, true)
	ch := make(chan struct{}, 1)
	go func() {
		for {
			select {
			case <-ch:
				return
			default:
				syscall.CancelIo(syscall.Handle(file.Sysfd))
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
	// Try to cancel for max 1 second.
	// Canceling is normally really fast, but it can take an
	// arbitrary amount of time on busy systems.
	// If it takes too long, we skip the test.
	file.SetReadDeadline(time.Now().Add(1 * time.Second))
	var tmp [1]byte
	// Read will block until the cancel is complete.
	_, err := file.Read(tmp[:])
	ch <- struct{}{}
	if err == poll.ErrDeadlineExceeded {
		t.Skip("took too long to cancel")
	}
	if err != syscall.ERROR_OPERATION_ABORTED {
		t.Errorf("expected ERROR_OPERATION_ABORTED, got %v", err)
	}
}

func TestPipeExternalIOCP(t *testing.T) {
	// Test that a caller can associate an overlapped handle to an external IOCP
	// even when the handle is also associated to a poll.FD. Also test that
	// the FD can still perform I/O after the association.
	t.Parallel()
	name := pipeName()
	pipe := newMessagePipe(t, name, true, true)
	_ = newFile(t, name, true, true) // Just open a pipe client

	_, err := windows.CreateIoCompletionPort(syscall.Handle(pipe.Sysfd), 0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	_, err = pipe.Write([]byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
}

func BenchmarkReadOverlapped(b *testing.B) {
	benchmarkRead(b, true)
}

func BenchmarkReadSync(b *testing.B) {
	benchmarkRead(b, false)
}

func benchmarkRead(b *testing.B, overlapped bool) {
	name := filepath.Join(b.TempDir(), "foo")
	const content = "hello world"
	err := os.WriteFile(name, []byte(content), 0644)
	if err != nil {
		b.Fatal(err)
	}
	file := newFile(b, name, overlapped, true)
	var buf [len(content)]byte
	for b.Loop() {
		_, err := io.ReadFull(file, buf[:])
		if err != nil {
			b.Fatal(err)
		}
		if _, err := file.Seek(0, io.SeekStart); err != nil {
			b.Fatal(err)
		}
	}
}
