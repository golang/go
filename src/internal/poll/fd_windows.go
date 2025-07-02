// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"errors"
	"internal/race"
	"internal/syscall/windows"
	"io"
	"sync"
	"sync/atomic"
	"syscall"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

var (
	initErr error
	ioSync  uint64
)

// This package uses the SetFileCompletionNotificationModes Windows
// API to skip calling GetQueuedCompletionStatus if an IO operation
// completes synchronously. There is a known bug where
// SetFileCompletionNotificationModes crashes on some systems (see
// https://support.microsoft.com/kb/2568167 for details).

var socketCanUseSetFileCompletionNotificationModes bool // determines is SetFileCompletionNotificationModes is present and sockets can safely use it

// checkSetFileCompletionNotificationModes verifies that
// SetFileCompletionNotificationModes Windows API is present
// on the system and is safe to use.
// See https://support.microsoft.com/kb/2568167 for details.
func checkSetFileCompletionNotificationModes() {
	err := syscall.LoadSetFileCompletionNotificationModes()
	if err != nil {
		return
	}
	protos := [2]int32{syscall.IPPROTO_TCP, 0}
	var buf [32]syscall.WSAProtocolInfo
	len := uint32(unsafe.Sizeof(buf))
	n, err := syscall.WSAEnumProtocols(&protos[0], &buf[0], &len)
	if err != nil {
		return
	}
	for i := int32(0); i < n; i++ {
		if buf[i].ServiceFlags1&syscall.XP1_IFS_HANDLES == 0 {
			return
		}
	}
	socketCanUseSetFileCompletionNotificationModes = true
}

// InitWSA initiates the use of the Winsock DLL by the current process.
// It is called from the net package at init time to avoid
// loading ws2_32.dll when net is not used.
var InitWSA = sync.OnceFunc(func() {
	var d syscall.WSAData
	e := syscall.WSAStartup(uint32(0x202), &d)
	if e != nil {
		initErr = e
	}
	checkSetFileCompletionNotificationModes()
})

// operation contains superset of data necessary to perform all async IO.
type operation struct {
	// Used by IOCP interface, it must be first field
	// of the struct, as our code relies on it.
	o syscall.Overlapped

	// fields used by runtime.netpoll
	runtimeCtx uintptr
	mode       int32

	// fields used only by net package
	buf   syscall.WSABuf
	msg   windows.WSAMsg
	sa    syscall.Sockaddr
	rsa   *syscall.RawSockaddrAny
	rsan  int32
	flags uint32
	qty   uint32
	bufs  []syscall.WSABuf
}

func (o *operation) setEvent() {
	h, err := windows.CreateEvent(nil, 0, 0, nil)
	if err != nil {
		// This shouldn't happen when all CreateEvent arguments are zero.
		panic(err)
	}
	// Set the low bit so that the external IOCP doesn't receive the completion packet.
	o.o.HEvent = h | 1
}

func (o *operation) close() {
	if o.o.HEvent != 0 {
		syscall.CloseHandle(o.o.HEvent)
	}
}

func (fd *FD) overlapped(o *operation) *syscall.Overlapped {
	if fd.isBlocking {
		// Don't return the overlapped object if the file handle
		// doesn't use overlapped I/O. It could be used, but
		// that would then use the file pointer stored in the
		// overlapped object rather than the real file pointer.
		return nil
	}
	return &o.o
}

func (o *operation) InitBuf(buf []byte) {
	o.buf.Len = uint32(len(buf))
	o.buf.Buf = nil
	if len(buf) != 0 {
		o.buf.Buf = &buf[0]
	}
}

func (o *operation) InitBufs(buf *[][]byte) {
	if o.bufs == nil {
		o.bufs = make([]syscall.WSABuf, 0, len(*buf))
	} else {
		o.bufs = o.bufs[:0]
	}
	for _, b := range *buf {
		if len(b) == 0 {
			o.bufs = append(o.bufs, syscall.WSABuf{})
			continue
		}
		for len(b) > maxRW {
			o.bufs = append(o.bufs, syscall.WSABuf{Len: maxRW, Buf: &b[0]})
			b = b[maxRW:]
		}
		if len(b) > 0 {
			o.bufs = append(o.bufs, syscall.WSABuf{Len: uint32(len(b)), Buf: &b[0]})
		}
	}
}

// ClearBufs clears all pointers to Buffers parameter captured
// by InitBufs, so it can be released by garbage collector.
func (o *operation) ClearBufs() {
	for i := range o.bufs {
		o.bufs[i].Buf = nil
	}
	o.bufs = o.bufs[:0]
}

func (o *operation) InitMsg(p []byte, oob []byte) {
	o.InitBuf(p)
	o.msg.Buffers = &o.buf
	o.msg.BufferCount = 1

	o.msg.Name = nil
	o.msg.Namelen = 0

	o.msg.Flags = 0
	o.msg.Control.Len = uint32(len(oob))
	o.msg.Control.Buf = nil
	if len(oob) != 0 {
		o.msg.Control.Buf = &oob[0]
	}
}

// waitIO waits for the IO operation o to complete.
func (fd *FD) waitIO(o *operation) error {
	if fd.isBlocking {
		panic("can't wait on blocking operations")
	}
	if !fd.pollable() {
		// The overlapped handle is not added to the runtime poller,
		// the only way to wait for the IO to complete is block until
		// the overlapped event is signaled.
		_, err := syscall.WaitForSingleObject(o.o.HEvent, syscall.INFINITE)
		return err
	}
	// Wait for our request to complete.
	err := fd.pd.wait(int(o.mode), fd.isFile)
	switch err {
	case nil, ErrNetClosing, ErrFileClosing, ErrDeadlineExceeded:
		// No other error is expected.
	default:
		panic("unexpected runtime.netpoll error: " + err.Error())
	}
	return err
}

// cancelIO cancels the IO operation o and waits for it to complete.
func (fd *FD) cancelIO(o *operation) {
	if !fd.pollable() {
		return
	}
	// Cancel our request.
	err := syscall.CancelIoEx(fd.Sysfd, &o.o)
	// Assuming ERROR_NOT_FOUND is returned, if IO is completed.
	if err != nil && err != syscall.ERROR_NOT_FOUND {
		// TODO(brainman): maybe do something else, but panic.
		panic(err)
	}
	fd.pd.waitCanceled(int(o.mode))
}

// execIO executes a single IO operation o.
// It supports both synchronous and asynchronous IO.
// o.qty and o.flags are set to zero before calling submit
// to avoid reusing the values from a previous call.
func (fd *FD) execIO(o *operation, submit func(o *operation) error) (int, error) {
	// Notify runtime netpoll about starting IO.
	err := fd.pd.prepare(int(o.mode), fd.isFile)
	if err != nil {
		return 0, err
	}
	// Start IO.
	if !fd.isBlocking && o.o.HEvent == 0 && !fd.pollable() {
		// If the handle is opened for overlapped IO but we can't
		// use the runtime poller, then we need to use an
		// event to wait for the IO to complete.
		o.setEvent()
	}
	o.qty = 0
	o.flags = 0
	err = submit(o)
	var waitErr error
	// Blocking operations shouldn't return ERROR_IO_PENDING.
	// Continue without waiting if that happens.
	if !fd.isBlocking && (err == syscall.ERROR_IO_PENDING || (err == nil && !fd.skipSyncNotif)) {
		// IO started asynchronously or completed synchronously but
		// a sync notification is required. Wait for it to complete.
		waitErr = fd.waitIO(o)
		if waitErr != nil {
			// IO interrupted by "close" or "timeout".
			fd.cancelIO(o)
			// We issued a cancellation request, but the IO operation may still succeeded
			// before the cancellation request runs.
		}
		if fd.isFile {
			err = windows.GetOverlappedResult(fd.Sysfd, &o.o, &o.qty, false)
		} else {
			err = windows.WSAGetOverlappedResult(fd.Sysfd, &o.o, &o.qty, false, &o.flags)
		}
	}
	switch err {
	case syscall.ERROR_OPERATION_ABORTED:
		// ERROR_OPERATION_ABORTED may have been caused by us. In that case,
		// map it to our own error. Don't do more than that, each submitted
		// function may have its own meaning for each error.
		if waitErr != nil {
			// IO canceled by the poller while waiting for completion.
			err = waitErr
		} else if fd.kind == kindPipe && fd.closing() {
			// Close uses CancelIoEx to interrupt concurrent I/O for pipes.
			// If the fd is a pipe and the Write was interrupted by CancelIoEx,
			// we assume it is interrupted by Close.
			err = errClosing(fd.isFile)
		}
	case windows.ERROR_IO_INCOMPLETE:
		// waitIO couldn't wait for the IO to complete.
		if waitErr != nil {
			// The wait error will be more informative.
			err = waitErr
		}
	}
	return int(o.qty), err
}

// FD is a file descriptor. The net and os packages embed this type in
// a larger type representing a network connection or OS file.
type FD struct {
	// Lock sysfd and serialize access to Read and Write methods.
	fdmu fdMutex

	// System file descriptor. Immutable until Close.
	Sysfd syscall.Handle

	// Read operation.
	rop operation
	// Write operation.
	wop operation

	// I/O poller.
	pd pollDesc

	// Used to implement pread/pwrite.
	l sync.Mutex

	// The file offset for the next read or write.
	// Overlapped IO operations don't use the real file pointer,
	// so we need to keep track of the offset ourselves.
	offset int64

	// For console I/O.
	lastbits       []byte   // first few bytes of the last incomplete rune in last write
	readuint16     []uint16 // buffer to hold uint16s obtained with ReadConsole
	readbyte       []byte   // buffer to hold decoding of readuint16 from utf16 to utf8
	readbyteOffset int      // readbyte[readOffset:] is yet to be consumed with file.Read

	// Semaphore signaled when file is closed.
	csema uint32

	skipSyncNotif bool

	// Whether this is a streaming descriptor, as opposed to a
	// packet-based descriptor like a UDP socket.
	IsStream bool

	// Whether a zero byte read indicates EOF. This is false for a
	// message based socket connection.
	ZeroReadIsEOF bool

	// Whether the handle is owned by os.File.
	isFile bool

	// The kind of this file.
	kind fileKind

	// Whether FILE_FLAG_OVERLAPPED was not set when opening the file.
	isBlocking bool

	disassociated atomic.Bool
}

// setOffset sets the offset fields of the overlapped object
// to the given offset. The fd.l lock must be held.
//
// Overlapped IO operations don't update the offset fields
// of the overlapped object nor the file pointer automatically,
// so we do that manually here.
// Note that this is a best effort that only works if the file
// pointer is completely owned by this operation. We could
// call seek to allow other processes or other operations on the
// same file to see the updated offset. That would be inefficient
// and won't work for concurrent operations anyway. If concurrent
// operations are needed, then the caller should serialize them
// using an external mechanism.
func (fd *FD) setOffset(off int64) {
	fd.offset = off
	fd.rop.o.OffsetHigh, fd.rop.o.Offset = uint32(off>>32), uint32(off)
	fd.wop.o.OffsetHigh, fd.wop.o.Offset = uint32(off>>32), uint32(off)
}

// addOffset adds the given offset to the current offset.
func (fd *FD) addOffset(off int) {
	fd.setOffset(fd.offset + int64(off))
}

// pollable should be used instead of fd.pd.pollable(),
// as it is aware of the disassociated state.
func (fd *FD) pollable() bool {
	return fd.pd.pollable() && !fd.disassociated.Load()
}

// fileKind describes the kind of file.
type fileKind byte

const (
	kindNet fileKind = iota
	kindFile
	kindConsole
	kindPipe
	kindFileNet
)

// Init initializes the FD. The Sysfd field should already be set.
// This can be called multiple times on a single FD.
// The net argument is a network name from the net package (e.g., "tcp"),
// or "file" or "console" or "dir".
// Set pollable to true if fd should be managed by runtime netpoll.
// Pollable must be set to true for overlapped fds.
func (fd *FD) Init(net string, pollable bool) error {
	if initErr != nil {
		return initErr
	}

	switch net {
	case "file":
		fd.kind = kindFile
	case "console":
		fd.kind = kindConsole
	case "pipe":
		fd.kind = kindPipe
	case "file+net":
		fd.kind = kindFileNet
	default:
		// We don't actually care about the various network types.
		fd.kind = kindNet
	}
	fd.isFile = fd.kind != kindNet
	fd.isBlocking = !pollable
	fd.rop.mode = 'r'
	fd.wop.mode = 'w'

	// It is safe to add overlapped handles that also perform I/O
	// outside of the runtime poller. The runtime poller will ignore
	// I/O completion notifications not initiated by us.
	err := fd.pd.init(fd)
	if err != nil {
		return err
	}
	fd.rop.runtimeCtx = fd.pd.runtimeCtx
	fd.wop.runtimeCtx = fd.pd.runtimeCtx
	if fd.kind != kindNet || socketCanUseSetFileCompletionNotificationModes {
		// Non-socket handles can use SetFileCompletionNotificationModes without problems.
		err := syscall.SetFileCompletionNotificationModes(fd.Sysfd,
			syscall.FILE_SKIP_SET_EVENT_ON_HANDLE|syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS,
		)
		fd.skipSyncNotif = err == nil
	}
	return nil
}

// DisassociateIOCP disassociates the file handle from the IOCP.
// The disassociate operation will not succeed if there is any
// in-progress IO operation on the file handle.
func (fd *FD) DisassociateIOCP() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()

	if fd.isBlocking || !fd.pollable() {
		// Nothing to disassociate.
		return nil
	}

	info := windows.FILE_COMPLETION_INFORMATION{}
	if err := windows.NtSetInformationFile(fd.Sysfd, &windows.IO_STATUS_BLOCK{}, unsafe.Pointer(&info), uint32(unsafe.Sizeof(info)), windows.FileReplaceCompletionInformation); err != nil {
		return err
	}
	fd.disassociated.Store(true)
	// Don't call fd.pd.close(), it would be too racy.
	// There is no harm on leaving fd.pd open until Close is called.
	return nil
}

func (fd *FD) destroy() error {
	if fd.Sysfd == syscall.InvalidHandle {
		return syscall.EINVAL
	}
	fd.rop.close()
	fd.wop.close()
	// Poller may want to unregister fd in readiness notification mechanism,
	// so this must be executed before fd.CloseFunc.
	fd.pd.close()
	var err error
	switch fd.kind {
	case kindNet, kindFileNet:
		// The net package uses the CloseFunc variable for testing.
		err = CloseFunc(fd.Sysfd)
	default:
		err = syscall.CloseHandle(fd.Sysfd)
	}
	fd.Sysfd = syscall.InvalidHandle
	runtime_Semrelease(&fd.csema)
	return err
}

// Close closes the FD. The underlying file descriptor is closed by
// the destroy method when there are no remaining references.
func (fd *FD) Close() error {
	if !fd.fdmu.increfAndClose() {
		return errClosing(fd.isFile)
	}

	if fd.kind == kindPipe {
		syscall.CancelIoEx(fd.Sysfd, nil)
	}
	// unblock pending reader and writer
	fd.pd.evict()
	err := fd.decref()
	// Wait until the descriptor is closed. If this was the only
	// reference, it is already closed.
	runtime_Semacquire(&fd.csema)
	return err
}

// Windows ReadFile and WSARecv use DWORD (uint32) parameter to pass buffer length.
// This prevents us reading blocks larger than 4GB.
// See golang.org/issue/26923.
const maxRW = 1 << 30 // 1GB is large enough and keeps subsequent reads aligned

// Read implements io.Reader.
func (fd *FD) Read(buf []byte) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if fd.kind == kindFile {
		fd.l.Lock()
		defer fd.l.Unlock()
	}

	if len(buf) > maxRW {
		buf = buf[:maxRW]
	}

	var n int
	var err error
	switch fd.kind {
	case kindConsole:
		n, err = fd.readConsole(buf)
	case kindFile, kindPipe:
		o := &fd.rop
		o.InitBuf(buf)
		n, err = fd.execIO(o, func(o *operation) error {
			return syscall.ReadFile(fd.Sysfd, unsafe.Slice(o.buf.Buf, o.buf.Len), &o.qty, fd.overlapped(o))
		})
		fd.addOffset(n)
		switch err {
		case syscall.ERROR_HANDLE_EOF:
			err = io.EOF
		case syscall.ERROR_BROKEN_PIPE:
			// ReadFile only documents ERROR_BROKEN_PIPE for pipes.
			if fd.kind == kindPipe {
				err = io.EOF
			}
		}
	case kindNet:
		o := &fd.rop
		o.InitBuf(buf)
		n, err = fd.execIO(o, func(o *operation) error {
			return syscall.WSARecv(fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, &o.o, nil)
		})
		if race.Enabled {
			race.Acquire(unsafe.Pointer(&ioSync))
		}
	}
	if len(buf) != 0 {
		err = fd.eofError(n, err)
	}
	return n, err
}

var ReadConsole = syscall.ReadConsole // changed for testing

// readConsole reads utf16 characters from console File,
// encodes them into utf8 and stores them in buffer b.
// It returns the number of utf8 bytes read and an error, if any.
func (fd *FD) readConsole(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}

	if fd.readuint16 == nil {
		// Note: syscall.ReadConsole fails for very large buffers.
		// The limit is somewhere around (but not exactly) 16384.
		// Stay well below.
		fd.readuint16 = make([]uint16, 0, 10000)
		fd.readbyte = make([]byte, 0, 4*cap(fd.readuint16))
	}

	for fd.readbyteOffset >= len(fd.readbyte) {
		n := cap(fd.readuint16) - len(fd.readuint16)
		if n > len(b) {
			n = len(b)
		}
		var nw uint32
		err := ReadConsole(fd.Sysfd, &fd.readuint16[:len(fd.readuint16)+1][len(fd.readuint16)], uint32(n), &nw, nil)
		if err != nil {
			return 0, err
		}
		uint16s := fd.readuint16[:len(fd.readuint16)+int(nw)]
		fd.readuint16 = fd.readuint16[:0]
		buf := fd.readbyte[:0]
		for i := 0; i < len(uint16s); i++ {
			r := rune(uint16s[i])
			if utf16.IsSurrogate(r) {
				if i+1 == len(uint16s) {
					if nw > 0 {
						// Save half surrogate pair for next time.
						fd.readuint16 = fd.readuint16[:1]
						fd.readuint16[0] = uint16(r)
						break
					}
					r = utf8.RuneError
				} else {
					r = utf16.DecodeRune(r, rune(uint16s[i+1]))
					if r != utf8.RuneError {
						i++
					}
				}
			}
			buf = utf8.AppendRune(buf, r)
		}
		fd.readbyte = buf
		fd.readbyteOffset = 0
		if nw == 0 {
			break
		}
	}

	src := fd.readbyte[fd.readbyteOffset:]
	var i int
	for i = 0; i < len(src) && i < len(b); i++ {
		x := src[i]
		if x == 0x1A { // Ctrl-Z
			if i == 0 {
				fd.readbyteOffset++
			}
			break
		}
		b[i] = x
	}
	fd.readbyteOffset += i
	return i, nil
}

// Pread emulates the Unix pread system call.
func (fd *FD) Pread(b []byte, off int64) (int, error) {
	if fd.kind == kindPipe {
		// Pread does not work with pipes
		return 0, syscall.ESPIPE
	}
	// Call incref, not readLock, because since pread specifies the
	// offset it is independent from other reads.
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()

	if len(b) > maxRW {
		b = b[:maxRW]
	}

	fd.l.Lock()
	defer fd.l.Unlock()
	curoffset, err := syscall.Seek(fd.Sysfd, 0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	defer syscall.Seek(fd.Sysfd, curoffset, io.SeekStart)
	defer fd.setOffset(curoffset)
	o := &fd.rop
	o.InitBuf(b)
	fd.setOffset(off)
	n, err := fd.execIO(o, func(o *operation) error {
		return syscall.ReadFile(fd.Sysfd, unsafe.Slice(o.buf.Buf, o.buf.Len), &o.qty, &o.o)
	})
	if err == syscall.ERROR_HANDLE_EOF {
		err = io.EOF
	}
	if len(b) != 0 {
		err = fd.eofError(n, err)
	}
	return n, err
}

// ReadFrom wraps the recvfrom network call.
func (fd *FD) ReadFrom(buf []byte) (int, syscall.Sockaddr, error) {
	if len(buf) == 0 {
		return 0, nil, nil
	}
	if len(buf) > maxRW {
		buf = buf[:maxRW]
	}
	if err := fd.readLock(); err != nil {
		return 0, nil, err
	}
	defer fd.readUnlock()
	o := &fd.rop
	o.InitBuf(buf)
	n, err := fd.execIO(o, func(o *operation) error {
		if o.rsa == nil {
			o.rsa = new(syscall.RawSockaddrAny)
		}
		o.rsan = int32(unsafe.Sizeof(*o.rsa))
		return syscall.WSARecvFrom(fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, o.rsa, &o.rsan, &o.o, nil)
	})
	err = fd.eofError(n, err)
	if err != nil {
		return n, nil, err
	}
	sa, _ := o.rsa.Sockaddr()
	return n, sa, nil
}

// ReadFromInet4 wraps the recvfrom network call for IPv4.
func (fd *FD) ReadFromInet4(buf []byte, sa4 *syscall.SockaddrInet4) (int, error) {
	if len(buf) == 0 {
		return 0, nil
	}
	if len(buf) > maxRW {
		buf = buf[:maxRW]
	}
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	o := &fd.rop
	o.InitBuf(buf)
	n, err := fd.execIO(o, func(o *operation) error {
		if o.rsa == nil {
			o.rsa = new(syscall.RawSockaddrAny)
		}
		o.rsan = int32(unsafe.Sizeof(*o.rsa))
		return syscall.WSARecvFrom(fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, o.rsa, &o.rsan, &o.o, nil)
	})
	err = fd.eofError(n, err)
	if err != nil {
		return n, err
	}
	rawToSockaddrInet4(o.rsa, sa4)
	return n, err
}

// ReadFromInet6 wraps the recvfrom network call for IPv6.
func (fd *FD) ReadFromInet6(buf []byte, sa6 *syscall.SockaddrInet6) (int, error) {
	if len(buf) == 0 {
		return 0, nil
	}
	if len(buf) > maxRW {
		buf = buf[:maxRW]
	}
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	o := &fd.rop
	o.InitBuf(buf)
	n, err := fd.execIO(o, func(o *operation) error {
		if o.rsa == nil {
			o.rsa = new(syscall.RawSockaddrAny)
		}
		o.rsan = int32(unsafe.Sizeof(*o.rsa))
		return syscall.WSARecvFrom(fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, o.rsa, &o.rsan, &o.o, nil)
	})
	err = fd.eofError(n, err)
	if err != nil {
		return n, err
	}
	rawToSockaddrInet6(o.rsa, sa6)
	return n, err
}

// Write implements io.Writer.
func (fd *FD) Write(buf []byte) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if fd.kind == kindFile {
		fd.l.Lock()
		defer fd.l.Unlock()
	}

	var ntotal int
	for {
		max := len(buf)
		if max-ntotal > maxRW {
			max = ntotal + maxRW
		}
		b := buf[ntotal:max]
		var n int
		var err error
		switch fd.kind {
		case kindConsole:
			n, err = fd.writeConsole(b)
		case kindPipe, kindFile:
			o := &fd.wop
			o.InitBuf(b)
			n, err = fd.execIO(o, func(o *operation) error {
				return syscall.WriteFile(fd.Sysfd, unsafe.Slice(o.buf.Buf, o.buf.Len), &o.qty, fd.overlapped(o))
			})
			fd.addOffset(n)
		case kindNet:
			if race.Enabled {
				race.ReleaseMerge(unsafe.Pointer(&ioSync))
			}
			o := &fd.wop
			o.InitBuf(b)
			n, err = fd.execIO(o, func(o *operation) error {
				return syscall.WSASend(fd.Sysfd, &o.buf, 1, &o.qty, 0, &o.o, nil)
			})
		}
		ntotal += n
		if ntotal == len(buf) || err != nil {
			return ntotal, err
		}
		if n == 0 {
			return ntotal, io.ErrUnexpectedEOF
		}
	}
}

// writeConsole writes len(b) bytes to the console File.
// It returns the number of bytes written and an error, if any.
func (fd *FD) writeConsole(b []byte) (int, error) {
	n := len(b)
	runes := make([]rune, 0, 256)
	if len(fd.lastbits) > 0 {
		b = append(fd.lastbits, b...)
		fd.lastbits = nil

	}
	for len(b) >= utf8.UTFMax || utf8.FullRune(b) {
		r, l := utf8.DecodeRune(b)
		runes = append(runes, r)
		b = b[l:]
	}
	if len(b) > 0 {
		fd.lastbits = make([]byte, len(b))
		copy(fd.lastbits, b)
	}
	// syscall.WriteConsole seems to fail, if given large buffer.
	// So limit the buffer to 16000 characters. This number was
	// discovered by experimenting with syscall.WriteConsole.
	const maxWrite = 16000
	for len(runes) > 0 {
		m := len(runes)
		if m > maxWrite {
			m = maxWrite
		}
		chunk := runes[:m]
		runes = runes[m:]
		uint16s := utf16.Encode(chunk)
		for len(uint16s) > 0 {
			var written uint32
			err := syscall.WriteConsole(fd.Sysfd, &uint16s[0], uint32(len(uint16s)), &written, nil)
			if err != nil {
				return 0, err
			}
			uint16s = uint16s[written:]
		}
	}
	return n, nil
}

// Pwrite emulates the Unix pwrite system call.
func (fd *FD) Pwrite(buf []byte, off int64) (int, error) {
	if fd.kind == kindPipe {
		// Pwrite does not work with pipes
		return 0, syscall.ESPIPE
	}
	// Call incref, not writeLock, because since pwrite specifies the
	// offset it is independent from other writes.
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()

	fd.l.Lock()
	defer fd.l.Unlock()
	curoffset, err := syscall.Seek(fd.Sysfd, 0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	defer syscall.Seek(fd.Sysfd, curoffset, io.SeekStart)
	defer fd.setOffset(curoffset)

	var ntotal int
	for {
		max := len(buf)
		if max-ntotal > maxRW {
			max = ntotal + maxRW
		}
		b := buf[ntotal:max]
		o := &fd.wop
		o.InitBuf(b)
		fd.setOffset(off + int64(ntotal))
		n, err := fd.execIO(o, func(o *operation) error {
			return syscall.WriteFile(fd.Sysfd, unsafe.Slice(o.buf.Buf, o.buf.Len), &o.qty, &o.o)
		})
		if n > 0 {
			ntotal += n
		}
		if ntotal == len(buf) || err != nil {
			return ntotal, err
		}
		if n == 0 {
			return ntotal, io.ErrUnexpectedEOF
		}
	}
}

// Writev emulates the Unix writev system call.
func (fd *FD) Writev(buf *[][]byte) (int64, error) {
	if len(*buf) == 0 {
		return 0, nil
	}
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if race.Enabled {
		race.ReleaseMerge(unsafe.Pointer(&ioSync))
	}
	o := &fd.wop
	o.InitBufs(buf)
	n, err := fd.execIO(o, func(o *operation) error {
		return syscall.WSASend(fd.Sysfd, &o.bufs[0], uint32(len(o.bufs)), &o.qty, 0, &o.o, nil)
	})
	o.ClearBufs()
	TestHookDidWritev(n)
	consume(buf, int64(n))
	return int64(n), err
}

// WriteTo wraps the sendto network call.
func (fd *FD) WriteTo(buf []byte, sa syscall.Sockaddr) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()

	if len(buf) == 0 {
		// handle zero-byte payload
		o := &fd.wop
		o.InitBuf(buf)
		o.sa = sa
		n, err := fd.execIO(o, func(o *operation) error {
			return syscall.WSASendto(fd.Sysfd, &o.buf, 1, &o.qty, 0, o.sa, &o.o, nil)
		})
		return n, err
	}

	ntotal := 0
	for len(buf) > 0 {
		b := buf
		if len(b) > maxRW {
			b = b[:maxRW]
		}
		o := &fd.wop
		o.InitBuf(b)
		o.sa = sa
		n, err := fd.execIO(o, func(o *operation) error {
			return syscall.WSASendto(fd.Sysfd, &o.buf, 1, &o.qty, 0, o.sa, &o.o, nil)
		})
		ntotal += int(n)
		if err != nil {
			return ntotal, err
		}
		buf = buf[n:]
	}
	return ntotal, nil
}

// WriteToInet4 is WriteTo, specialized for syscall.SockaddrInet4.
func (fd *FD) WriteToInet4(buf []byte, sa4 *syscall.SockaddrInet4) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()

	if len(buf) == 0 {
		// handle zero-byte payload
		o := &fd.wop
		o.InitBuf(buf)
		n, err := fd.execIO(o, func(o *operation) error {
			return windows.WSASendtoInet4(fd.Sysfd, &o.buf, 1, &o.qty, 0, sa4, &o.o, nil)
		})
		return n, err
	}

	ntotal := 0
	for len(buf) > 0 {
		b := buf
		if len(b) > maxRW {
			b = b[:maxRW]
		}
		o := &fd.wop
		o.InitBuf(b)
		n, err := fd.execIO(o, func(o *operation) error {
			return windows.WSASendtoInet4(fd.Sysfd, &o.buf, 1, &o.qty, 0, sa4, &o.o, nil)
		})
		ntotal += int(n)
		if err != nil {
			return ntotal, err
		}
		buf = buf[n:]
	}
	return ntotal, nil
}

// WriteToInet6 is WriteTo, specialized for syscall.SockaddrInet6.
func (fd *FD) WriteToInet6(buf []byte, sa6 *syscall.SockaddrInet6) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()

	if len(buf) == 0 {
		// handle zero-byte payload
		o := &fd.wop
		o.InitBuf(buf)
		n, err := fd.execIO(o, func(o *operation) error {
			return windows.WSASendtoInet6(fd.Sysfd, &o.buf, 1, &o.qty, 0, sa6, &o.o, nil)
		})
		return n, err
	}

	ntotal := 0
	for len(buf) > 0 {
		b := buf
		if len(b) > maxRW {
			b = b[:maxRW]
		}
		o := &fd.wop
		o.InitBuf(b)
		n, err := fd.execIO(o, func(o *operation) error {
			return windows.WSASendtoInet6(fd.Sysfd, &o.buf, 1, &o.qty, 0, sa6, &o.o, nil)
		})
		ntotal += int(n)
		if err != nil {
			return ntotal, err
		}
		buf = buf[n:]
	}
	return ntotal, nil
}

// Call ConnectEx. This doesn't need any locking, since it is only
// called when the descriptor is first created. This is here rather
// than in the net package so that it can use fd.wop.
func (fd *FD) ConnectEx(ra syscall.Sockaddr) error {
	o := &fd.wop
	o.sa = ra
	_, err := fd.execIO(o, func(o *operation) error {
		return ConnectExFunc(fd.Sysfd, o.sa, nil, 0, nil, &o.o)
	})
	return err
}

func (fd *FD) acceptOne(s syscall.Handle, rawsa []syscall.RawSockaddrAny, o *operation) (string, error) {
	// Submit accept request.
	o.rsan = int32(unsafe.Sizeof(rawsa[0]))
	_, err := fd.execIO(o, func(o *operation) error {
		return AcceptFunc(fd.Sysfd, s, (*byte)(unsafe.Pointer(&rawsa[0])), 0, uint32(o.rsan), uint32(o.rsan), &o.qty, &o.o)
	})
	if err != nil {
		CloseFunc(s)
		return "acceptex", err
	}

	// Inherit properties of the listening socket.
	err = syscall.Setsockopt(s, syscall.SOL_SOCKET, syscall.SO_UPDATE_ACCEPT_CONTEXT, (*byte)(unsafe.Pointer(&fd.Sysfd)), int32(unsafe.Sizeof(fd.Sysfd)))
	if err != nil {
		CloseFunc(s)
		return "setsockopt", err
	}

	return "", nil
}

// Accept handles accepting a socket. The sysSocket parameter is used
// to allocate the net socket.
func (fd *FD) Accept(sysSocket func() (syscall.Handle, error)) (syscall.Handle, []syscall.RawSockaddrAny, uint32, string, error) {
	if err := fd.readLock(); err != nil {
		return syscall.InvalidHandle, nil, 0, "", err
	}
	defer fd.readUnlock()

	o := &fd.rop
	var rawsa [2]syscall.RawSockaddrAny
	for {
		s, err := sysSocket()
		if err != nil {
			return syscall.InvalidHandle, nil, 0, "", err
		}

		errcall, err := fd.acceptOne(s, rawsa[:], o)
		if err == nil {
			return s, rawsa[:], uint32(o.rsan), "", nil
		}

		// Sometimes we see WSAECONNRESET and ERROR_NETNAME_DELETED is
		// returned here. These happen if connection reset is received
		// before AcceptEx could complete. These errors relate to new
		// connection, not to AcceptEx, so ignore broken connection and
		// try AcceptEx again for more connections.
		errno, ok := err.(syscall.Errno)
		if !ok {
			return syscall.InvalidHandle, nil, 0, errcall, err
		}
		switch errno {
		case syscall.ERROR_NETNAME_DELETED, syscall.WSAECONNRESET:
			// ignore these and try again
		default:
			return syscall.InvalidHandle, nil, 0, errcall, err
		}
	}
}

// Seek wraps syscall.Seek.
func (fd *FD) Seek(offset int64, whence int) (int64, error) {
	if fd.kind == kindPipe {
		return 0, syscall.ESPIPE
	}
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()

	fd.l.Lock()
	defer fd.l.Unlock()

	n, err := syscall.Seek(fd.Sysfd, offset, whence)
	fd.setOffset(n)
	return n, err
}

// Fchmod updates syscall.ByHandleFileInformation.Fileattributes when needed.
func (fd *FD) Fchmod(mode uint32) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()

	var d syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(fd.Sysfd, &d); err != nil {
		return err
	}
	attrs := d.FileAttributes
	if mode&syscall.S_IWRITE != 0 {
		attrs &^= syscall.FILE_ATTRIBUTE_READONLY
	} else {
		attrs |= syscall.FILE_ATTRIBUTE_READONLY
	}
	if attrs == d.FileAttributes {
		return nil
	}

	var du windows.FILE_BASIC_INFO
	du.FileAttributes = attrs
	return windows.SetFileInformationByHandle(fd.Sysfd, windows.FileBasicInfo, unsafe.Pointer(&du), uint32(unsafe.Sizeof(du)))
}

// Fchdir wraps syscall.Fchdir.
func (fd *FD) Fchdir() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Fchdir(fd.Sysfd)
}

// GetFileType wraps syscall.GetFileType.
func (fd *FD) GetFileType() (uint32, error) {
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()
	return syscall.GetFileType(fd.Sysfd)
}

// GetFileInformationByHandle wraps GetFileInformationByHandle.
func (fd *FD) GetFileInformationByHandle(data *syscall.ByHandleFileInformation) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.GetFileInformationByHandle(fd.Sysfd, data)
}

// RawRead invokes the user-defined function f for a read operation.
func (fd *FD) RawRead(f func(uintptr) bool) error {
	if err := fd.readLock(); err != nil {
		return err
	}
	defer fd.readUnlock()
	for {
		if f(uintptr(fd.Sysfd)) {
			return nil
		}

		// Use a zero-byte read as a way to get notified when this
		// socket is readable. h/t https://stackoverflow.com/a/42019668/332798
		o := &fd.rop
		o.InitBuf(nil)
		_, err := fd.execIO(o, func(o *operation) error {
			if !fd.IsStream {
				o.flags |= windows.MSG_PEEK
			}
			return syscall.WSARecv(fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, &o.o, nil)
		})
		if err == windows.WSAEMSGSIZE {
			// expected with a 0-byte peek, ignore.
		} else if err != nil {
			return err
		}
	}
}

// RawWrite invokes the user-defined function f for a write operation.
func (fd *FD) RawWrite(f func(uintptr) bool) error {
	if err := fd.writeLock(); err != nil {
		return err
	}
	defer fd.writeUnlock()

	if f(uintptr(fd.Sysfd)) {
		return nil
	}

	// TODO(tmm1): find a way to detect socket writability
	return syscall.EWINDOWS
}

func sockaddrInet4ToRaw(rsa *syscall.RawSockaddrAny, sa *syscall.SockaddrInet4) int32 {
	*rsa = syscall.RawSockaddrAny{}
	raw := (*syscall.RawSockaddrInet4)(unsafe.Pointer(rsa))
	raw.Family = syscall.AF_INET
	p := (*[2]byte)(unsafe.Pointer(&raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	raw.Addr = sa.Addr
	return int32(unsafe.Sizeof(*raw))
}

func sockaddrInet6ToRaw(rsa *syscall.RawSockaddrAny, sa *syscall.SockaddrInet6) int32 {
	*rsa = syscall.RawSockaddrAny{}
	raw := (*syscall.RawSockaddrInet6)(unsafe.Pointer(rsa))
	raw.Family = syscall.AF_INET6
	p := (*[2]byte)(unsafe.Pointer(&raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	raw.Scope_id = sa.ZoneId
	raw.Addr = sa.Addr
	return int32(unsafe.Sizeof(*raw))
}

func rawToSockaddrInet4(rsa *syscall.RawSockaddrAny, sa *syscall.SockaddrInet4) {
	pp := (*syscall.RawSockaddrInet4)(unsafe.Pointer(rsa))
	p := (*[2]byte)(unsafe.Pointer(&pp.Port))
	sa.Port = int(p[0])<<8 + int(p[1])
	sa.Addr = pp.Addr
}

func rawToSockaddrInet6(rsa *syscall.RawSockaddrAny, sa *syscall.SockaddrInet6) {
	pp := (*syscall.RawSockaddrInet6)(unsafe.Pointer(rsa))
	p := (*[2]byte)(unsafe.Pointer(&pp.Port))
	sa.Port = int(p[0])<<8 + int(p[1])
	sa.ZoneId = pp.Scope_id
	sa.Addr = pp.Addr
}

func sockaddrToRaw(rsa *syscall.RawSockaddrAny, sa syscall.Sockaddr) (int32, error) {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		sz := sockaddrInet4ToRaw(rsa, sa)
		return sz, nil
	case *syscall.SockaddrInet6:
		sz := sockaddrInet6ToRaw(rsa, sa)
		return sz, nil
	default:
		return 0, syscall.EWINDOWS
	}
}

// ReadMsg wraps the WSARecvMsg network call.
func (fd *FD) ReadMsg(p []byte, oob []byte, flags int) (int, int, int, syscall.Sockaddr, error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, nil, err
	}
	defer fd.readUnlock()

	if len(p) > maxRW {
		p = p[:maxRW]
	}

	o := &fd.rop
	o.InitMsg(p, oob)
	if o.rsa == nil {
		o.rsa = new(syscall.RawSockaddrAny)
	}
	o.msg.Name = (syscall.Pointer)(unsafe.Pointer(o.rsa))
	o.msg.Namelen = int32(unsafe.Sizeof(*o.rsa))
	o.msg.Flags = uint32(flags)
	n, err := fd.execIO(o, func(o *operation) error {
		return windows.WSARecvMsg(fd.Sysfd, &o.msg, &o.qty, &o.o, nil)
	})
	err = fd.eofError(n, err)
	var sa syscall.Sockaddr
	if err == nil {
		sa, err = o.rsa.Sockaddr()
	}
	return n, int(o.msg.Control.Len), int(o.msg.Flags), sa, err
}

// ReadMsgInet4 is ReadMsg, but specialized to return a syscall.SockaddrInet4.
func (fd *FD) ReadMsgInet4(p []byte, oob []byte, flags int, sa4 *syscall.SockaddrInet4) (int, int, int, error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, err
	}
	defer fd.readUnlock()

	if len(p) > maxRW {
		p = p[:maxRW]
	}

	o := &fd.rop
	o.InitMsg(p, oob)
	if o.rsa == nil {
		o.rsa = new(syscall.RawSockaddrAny)
	}
	o.msg.Name = (syscall.Pointer)(unsafe.Pointer(o.rsa))
	o.msg.Namelen = int32(unsafe.Sizeof(*o.rsa))
	o.msg.Flags = uint32(flags)
	n, err := fd.execIO(o, func(o *operation) error {
		return windows.WSARecvMsg(fd.Sysfd, &o.msg, &o.qty, &o.o, nil)
	})
	err = fd.eofError(n, err)
	if err == nil {
		rawToSockaddrInet4(o.rsa, sa4)
	}
	return n, int(o.msg.Control.Len), int(o.msg.Flags), err
}

// ReadMsgInet6 is ReadMsg, but specialized to return a syscall.SockaddrInet6.
func (fd *FD) ReadMsgInet6(p []byte, oob []byte, flags int, sa6 *syscall.SockaddrInet6) (int, int, int, error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, err
	}
	defer fd.readUnlock()

	if len(p) > maxRW {
		p = p[:maxRW]
	}

	o := &fd.rop
	o.InitMsg(p, oob)
	if o.rsa == nil {
		o.rsa = new(syscall.RawSockaddrAny)
	}
	o.msg.Name = (syscall.Pointer)(unsafe.Pointer(o.rsa))
	o.msg.Namelen = int32(unsafe.Sizeof(*o.rsa))
	o.msg.Flags = uint32(flags)
	n, err := fd.execIO(o, func(o *operation) error {
		return windows.WSARecvMsg(fd.Sysfd, &o.msg, &o.qty, &o.o, nil)
	})
	err = fd.eofError(n, err)
	if err == nil {
		rawToSockaddrInet6(o.rsa, sa6)
	}
	return n, int(o.msg.Control.Len), int(o.msg.Flags), err
}

// WriteMsg wraps the WSASendMsg network call.
func (fd *FD) WriteMsg(p []byte, oob []byte, sa syscall.Sockaddr) (int, int, error) {
	if len(p) > maxRW {
		return 0, 0, errors.New("packet is too large (only 1GB is allowed)")
	}

	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()

	o := &fd.wop
	o.InitMsg(p, oob)
	if sa != nil {
		if o.rsa == nil {
			o.rsa = new(syscall.RawSockaddrAny)
		}
		len, err := sockaddrToRaw(o.rsa, sa)
		if err != nil {
			return 0, 0, err
		}
		o.msg.Name = (syscall.Pointer)(unsafe.Pointer(o.rsa))
		o.msg.Namelen = len
	}
	n, err := fd.execIO(o, func(o *operation) error {
		return windows.WSASendMsg(fd.Sysfd, &o.msg, 0, &o.qty, &o.o, nil)
	})
	return n, int(o.msg.Control.Len), err
}

// WriteMsgInet4 is WriteMsg specialized for syscall.SockaddrInet4.
func (fd *FD) WriteMsgInet4(p []byte, oob []byte, sa *syscall.SockaddrInet4) (int, int, error) {
	if len(p) > maxRW {
		return 0, 0, errors.New("packet is too large (only 1GB is allowed)")
	}

	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()

	o := &fd.wop
	o.InitMsg(p, oob)
	if o.rsa == nil {
		o.rsa = new(syscall.RawSockaddrAny)
	}
	len := sockaddrInet4ToRaw(o.rsa, sa)
	o.msg.Name = (syscall.Pointer)(unsafe.Pointer(o.rsa))
	o.msg.Namelen = len
	n, err := fd.execIO(o, func(o *operation) error {
		return windows.WSASendMsg(fd.Sysfd, &o.msg, 0, &o.qty, &o.o, nil)
	})
	return n, int(o.msg.Control.Len), err
}

// WriteMsgInet6 is WriteMsg specialized for syscall.SockaddrInet6.
func (fd *FD) WriteMsgInet6(p []byte, oob []byte, sa *syscall.SockaddrInet6) (int, int, error) {
	if len(p) > maxRW {
		return 0, 0, errors.New("packet is too large (only 1GB is allowed)")
	}

	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()

	o := &fd.wop
	o.InitMsg(p, oob)
	if o.rsa == nil {
		o.rsa = new(syscall.RawSockaddrAny)
	}
	len := sockaddrInet6ToRaw(o.rsa, sa)
	o.msg.Name = (syscall.Pointer)(unsafe.Pointer(o.rsa))
	o.msg.Namelen = len
	n, err := fd.execIO(o, func(o *operation) error {
		return windows.WSASendMsg(fd.Sysfd, &o.msg, 0, &o.qty, &o.o, nil)
	})
	return n, int(o.msg.Control.Len), err
}

func DupCloseOnExec(fd int) (int, string, error) {
	proc, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, "GetCurrentProcess", err
	}

	var nfd syscall.Handle
	const inherit = false // analogous to CLOEXEC
	if err := syscall.DuplicateHandle(proc, syscall.Handle(fd), proc, &nfd, 0, inherit, syscall.DUPLICATE_SAME_ACCESS); err != nil {
		return 0, "DuplicateHandle", err
	}
	return int(nfd), "", nil
}
