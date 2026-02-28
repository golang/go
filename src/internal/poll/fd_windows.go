// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"errors"
	"internal/race"
	"internal/syscall/windows"
	"io"
	"runtime"
	"sync"
	"syscall"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

var (
	initErr error
	ioSync  uint64
)

// ifsHandlesOnly returns true if the system only has IFS handles for TCP sockets.
// See https://support.microsoft.com/kb/2568167 for details.
var ifsHandlesOnly = sync.OnceValue(func() bool {
	protos := [2]int32{syscall.IPPROTO_TCP, 0}
	var buf [32]syscall.WSAProtocolInfo
	len := uint32(unsafe.Sizeof(buf))
	n, err := syscall.WSAEnumProtocols(&protos[0], &buf[0], &len)
	if err != nil {
		return false
	}
	for i := range n {
		if buf[i].ServiceFlags1&syscall.XP1_IFS_HANDLES == 0 {
			return false
		}
	}
	return true
})

// canSkipCompletionPortOnSuccess returns true if we use FILE_SKIP_COMPLETION_PORT_ON_SUCCESS for the given handle.
// See https://support.microsoft.com/kb/2568167 for details.
func canSkipCompletionPortOnSuccess(h syscall.Handle, isSocket bool) bool {
	if !isSocket {
		// Non-socket handles can use SetFileCompletionNotificationModes without problems.
		return true
	}
	if ifsHandlesOnly() {
		// If the system only has IFS handles for TCP sockets, then there is nothing else to check.
		return true
	}
	var info syscall.WSAProtocolInfo
	size := int32(unsafe.Sizeof(info))
	if syscall.Getsockopt(h, syscall.SOL_SOCKET, windows.SO_PROTOCOL_INFOW, (*byte)(unsafe.Pointer(&info)), &size) != nil {
		return false
	}
	return info.ServiceFlags1&syscall.XP1_IFS_HANDLES != 0
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
})

// operation contains superset of data necessary to perform all async IO.
type operation struct {
	// Used by IOCP interface, it must be first field
	// of the struct, as our code relies on it.
	o syscall.Overlapped

	// fields used by runtime.netpoll
	runtimeCtx uintptr
	mode       int32
}

func (o *operation) setOffset(off int64) {
	o.o.OffsetHigh = uint32(off >> 32)
	o.o.Offset = uint32(off)
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

func newWsaBuf(b []byte) *syscall.WSABuf {
	return &syscall.WSABuf{Buf: unsafe.SliceData(b), Len: uint32(len(b))}
}

var wsaBufsPool = sync.Pool{
	New: func() any {
		buf := make([]syscall.WSABuf, 0, 16)
		return &buf
	},
}

func newWSABufs(buf *[][]byte) *[]syscall.WSABuf {
	bufsPtr := wsaBufsPool.Get().(*[]syscall.WSABuf)
	*bufsPtr = (*bufsPtr)[:0]
	for _, b := range *buf {
		if len(b) == 0 {
			*bufsPtr = append(*bufsPtr, syscall.WSABuf{})
			continue
		}
		for len(b) > maxRW {
			*bufsPtr = append(*bufsPtr, syscall.WSABuf{Len: maxRW, Buf: &b[0]})
			b = b[maxRW:]
		}
		if len(b) > 0 {
			*bufsPtr = append(*bufsPtr, syscall.WSABuf{Len: uint32(len(b)), Buf: &b[0]})
		}
	}
	return bufsPtr
}

func freeWSABufs(bufsPtr *[]syscall.WSABuf) {
	// Clear pointers to buffers so they can be released by garbage collector.
	bufs := *bufsPtr
	for i := range bufs {
		bufs[i].Buf = nil
	}
	// Proper usage of a sync.Pool requires each entry to have approximately
	// the same memory cost. To obtain this property when the stored type
	// contains a variably-sized buffer, we add a hard limit on the maximum buffer
	// to place back in the pool.
	//
	// See https://go.dev/issue/23199
	if cap(*bufsPtr) > 128 {
		*bufsPtr = nil
	}
	wsaBufsPool.Put(bufsPtr)
}

// wsaMsgPool is a pool of WSAMsg structures that can only hold a single WSABuf.
var wsaMsgPool = sync.Pool{
	New: func() any {
		return &windows.WSAMsg{
			Buffers:     &syscall.WSABuf{},
			BufferCount: 1,
		}
	},
}

// newWSAMsg creates a new WSAMsg with the provided parameters.
// Use [freeWSAMsg] to free it.
func newWSAMsg(p []byte, oob []byte, flags int, unconnected bool) *windows.WSAMsg {
	// The returned object can't be allocated in the stack because it is accessed asynchronously
	// by Windows in between several system calls. If the stack frame is moved while that happens,
	// then Windows may access invalid memory.
	// TODO(qmuntal): investigate using runtime.Pinner keeping this path allocation-free.

	// Use a pool to reuse allocations.
	msg := wsaMsgPool.Get().(*windows.WSAMsg)
	msg.Buffers.Len = uint32(len(p))
	msg.Buffers.Buf = unsafe.SliceData(p)
	msg.Control = syscall.WSABuf{
		Len: uint32(len(oob)),
		Buf: unsafe.SliceData(oob),
	}
	msg.Flags = uint32(flags)
	if unconnected {
		msg.Name = wsaRsaPool.Get().(*syscall.RawSockaddrAny)
		msg.Namelen = int32(unsafe.Sizeof(syscall.RawSockaddrAny{}))
	}
	return msg
}

func freeWSAMsg(msg *windows.WSAMsg) {
	// Clear pointers to buffers so they can be released by garbage collector.
	msg.Buffers.Len = 0
	msg.Buffers.Buf = nil
	msg.Control.Len = 0
	msg.Control.Buf = nil
	if msg.Name != nil {
		*msg.Name = syscall.RawSockaddrAny{}
		wsaRsaPool.Put(msg.Name)
		msg.Name = nil
		msg.Namelen = 0
	}
	wsaMsgPool.Put(msg)
}

var wsaRsaPool = sync.Pool{
	New: func() any {
		return new(syscall.RawSockaddrAny)
	},
}

var operationPool = sync.Pool{
	New: func() any {
		return new(operation)
	},
}

// waitIO waits for the IO operation to complete,
// handling cancellation if necessary.
func (fd *FD) waitIO(o *operation) error {
	if o.o.HEvent != 0 {
		// The overlapped handle is not added to the runtime poller,
		// the only way to wait for the IO to complete is block until
		// the overlapped event is signaled.
		_, err := syscall.WaitForSingleObject(o.o.HEvent, syscall.INFINITE)
		return err
	}
	// Wait for our request to complete.
	err := fd.pd.wait(int(o.mode), fd.isFile)
	switch err {
	case nil:
		// IO completed successfully.
	case ErrNetClosing, ErrFileClosing, ErrDeadlineExceeded:
		// IO interrupted by "close" or "timeout", cancel our request.
		// ERROR_NOT_FOUND can be returned when the request succeded
		// between the time wait returned and CancelIoEx was executed.
		if err := syscall.CancelIoEx(fd.Sysfd, &o.o); err != nil && err != syscall.ERROR_NOT_FOUND {
			// TODO(brainman): maybe do something else, but panic.
			panic(err)
		}
		fd.pd.waitCanceled(int(o.mode))
	default:
		// No other error is expected.
		panic("unexpected runtime.netpoll error: " + err.Error())
	}
	return err
}

// execIO executes a single IO operation o.
// It supports both synchronous and asynchronous IO.
// buf, if not nil, will be pinned during the lifetime of the operation.
func (fd *FD) execIO(mode int, submit func(o *operation) (uint32, error), buf []byte) (int, error) {
	// Notify runtime netpoll about starting IO.
	err := fd.pd.prepare(mode, fd.isFile)
	if err != nil {
		return 0, err
	}
	o := operationPool.Get().(*operation)
	defer operationPool.Put(o)
	*o = operation{
		runtimeCtx: fd.pd.runtimeCtx,
		mode:       int32(mode),
	}
	o.setOffset(fd.offset)
	if !fd.isBlocking {
		if len(buf) > 0 {
			ptr := unsafe.SliceData(buf)
			if mode == 'r' {
				fd.readPinner.Pin(ptr)
			} else {
				fd.writePinner.Pin(ptr)
			}
			defer func() {
				if mode == 'r' {
					fd.readPinner.Unpin()
				} else {
					fd.writePinner.Unpin()
				}
			}()
		}
		if !fd.associated {
			// If the handle is opened for overlapped IO but we can't
			// use the runtime poller, then we need to use an
			// event to wait for the IO to complete.
			h, err := windows.CreateEvent(nil, 0, 0, nil)
			if err != nil {
				// This shouldn't happen when all CreateEvent arguments are zero.
				panic(err)
			}
			// Set the low bit so that the external IOCP doesn't receive the completion packet.
			o.o.HEvent = h | 1
			defer syscall.CloseHandle(h)
		}
	}
	// Start IO.
	qty, err := submit(o)
	var waitErr error
	// Blocking operations shouldn't return ERROR_IO_PENDING.
	// Continue without waiting if that happens.
	if !fd.isBlocking && (err == syscall.ERROR_IO_PENDING || (err == nil && fd.waitOnSuccess)) {
		// IO started asynchronously or completed synchronously but
		// a sync notification is required. Wait for it to complete.
		waitErr = fd.waitIO(o)
		if fd.isFile {
			err = windows.GetOverlappedResult(fd.Sysfd, &o.o, &qty, false)
		} else {
			var flags uint32
			err = windows.WSAGetOverlappedResult(fd.Sysfd, &o.o, &qty, false, &flags)
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
	return int(qty), err
}

// FD is a file descriptor. The net and os packages embed this type in
// a larger type representing a network connection or OS file.
type FD struct {
	// Lock sysfd and serialize access to Read and Write methods.
	fdmu fdMutex

	// System file descriptor. Immutable until Close.
	Sysfd syscall.Handle

	// I/O poller.
	pd pollDesc

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

	// Don't wait from completion port notifications for successful
	// operations that complete synchronously.
	waitOnSuccess bool

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

	// Whether the handle is currently associated with the IOCP.
	associated bool

	// readPinner and writePinner are automatically unpinned
	// before execIO returns.
	readPinner  runtime.Pinner
	writePinner runtime.Pinner
}

// setOffset sets the offset fields of the overlapped object
// to the given offset. The fd read/write lock must be held.
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
}

// addOffset adds the given offset to the current offset.
func (fd *FD) addOffset(off int) {
	fd.offset += int64(off)
}

// fileKind describes the kind of file.
type fileKind byte

const (
	kindNet fileKind = iota
	kindFile
	kindConsole
	kindPipe
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
	default:
		// We don't actually care about the various network types.
		fd.kind = kindNet
	}
	fd.isFile = fd.kind != kindNet
	fd.isBlocking = !pollable

	if !pollable {
		return nil
	}

	// The default behavior of the Windows I/O manager is to queue a completion
	// port entry for successful operations that complete synchronously when
	// the handle is opened for overlapped I/O. We will try to disable that
	// behavior below, as it requires an extra syscall.
	fd.waitOnSuccess = true

	// It is safe to add overlapped handles that also perform I/O
	// outside of the runtime poller. The runtime poller will ignore
	// I/O completion notifications not initiated by us.
	err := fd.pd.init(fd)
	if err != nil {
		return err
	}
	fd.associated = true

	// FILE_SKIP_SET_EVENT_ON_HANDLE is always safe to use. We don't use that feature
	// and it adds some overhead to the Windows I/O manager.
	// See https://devblogs.microsoft.com/oldnewthing/20200221-00/?p=103466.
	modes := uint8(syscall.FILE_SKIP_SET_EVENT_ON_HANDLE)
	if canSkipCompletionPortOnSuccess(fd.Sysfd, fd.kind == kindNet) {
		modes |= syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS
	}
	if syscall.SetFileCompletionNotificationModes(fd.Sysfd, modes) == nil {
		if modes&syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS != 0 {
			fd.waitOnSuccess = false
		}
	}
	return nil
}

// DisassociateIOCP disassociates the file handle from the IOCP.
// The disassociate operation will not succeed if there is any
// in-progress I/O operation on the file handle.
func (fd *FD) DisassociateIOCP() error {
	// There is a small race window between execIO checking fd.disassociated and
	// DisassociateIOCP setting it. NtSetInformationFile will fail anyway if
	// there is any in-progress I/O operation, so just take a read-write lock
	// to ensure there is no in-progress I/O and fail early if we can't get the lock.
	if ok, err := fd.tryReadWriteLock(); err != nil || !ok {
		if err == nil {
			err = errors.New("can't disassociate the handle while there is in-progress I/O")
		}
		return err
	}
	defer fd.readWriteUnlock()

	if !fd.associated {
		// Nothing to disassociate.
		return nil
	}

	info := windows.FILE_COMPLETION_INFORMATION{}
	if err := windows.NtSetInformationFile(fd.Sysfd, &windows.IO_STATUS_BLOCK{}, unsafe.Pointer(&info), uint32(unsafe.Sizeof(info)), windows.FileReplaceCompletionInformation); err != nil {
		return err
	}
	// tryReadWriteLock means we have exclusive access to fd.
	fd.associated = false
	// Don't call fd.pd.close(), it would be too racy.
	// There is no harm on leaving fd.pd open until Close is called.
	return nil
}

func (fd *FD) destroy() error {
	if fd.Sysfd == syscall.InvalidHandle {
		return syscall.EINVAL
	}
	// Poller may want to unregister fd in readiness notification mechanism,
	// so this must be executed before fd.CloseFunc.
	fd.pd.close()
	var err error
	switch fd.kind {
	case kindNet:
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
	if fd.kind == kindFile {
		if err := fd.readWriteLock(); err != nil {
			return 0, err
		}
		defer fd.readWriteUnlock()
	} else {
		if err := fd.readLock(); err != nil {
			return 0, err
		}
		defer fd.readUnlock()
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
		n, err = fd.execIO('r', func(o *operation) (qty uint32, err error) {
			err = syscall.ReadFile(fd.Sysfd, buf, &qty, fd.overlapped(o))
			return qty, err
		}, buf)
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
		n, err = fd.execIO('r', func(o *operation) (qty uint32, err error) {
			var flags uint32
			err = syscall.WSARecv(fd.Sysfd, newWsaBuf(buf), 1, &qty, &flags, &o.o, nil)
			return qty, err
		}, buf)
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
func (fd *FD) Pread(buf []byte, off int64) (int, error) {
	if fd.kind == kindPipe {
		// Pread does not work with pipes
		return 0, syscall.ESPIPE
	}

	if err := fd.readWriteLock(); err != nil {
		return 0, err
	}
	defer fd.readWriteUnlock()

	if len(buf) > maxRW {
		buf = buf[:maxRW]
	}

	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		// Overlapped handles don't have the file pointer updated
		// when performing I/O operations, so there is no need to
		// call Seek to reset the file pointer.
		// Also, some overlapped file handles don't support seeking.
		// See https://go.dev/issues/74951.
		if fd.isBlocking {
			curoffset, err := syscall.Seek(fd.Sysfd, 0, io.SeekCurrent)
			if err != nil {
				return 0, err
			}
			defer syscall.Seek(fd.Sysfd, curoffset, io.SeekStart)
		}
		o.setOffset(off)

		err = syscall.ReadFile(fd.Sysfd, buf, &qty, &o.o)
		return qty, err
	}, buf)
	if err == syscall.ERROR_HANDLE_EOF {
		err = io.EOF
	}
	if len(buf) != 0 {
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

	rsa := wsaRsaPool.Get().(*syscall.RawSockaddrAny)
	defer wsaRsaPool.Put(rsa)
	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		rsan := int32(unsafe.Sizeof(*rsa))
		var flags uint32
		err = syscall.WSARecvFrom(fd.Sysfd, newWsaBuf(buf), 1, &qty, &flags, rsa, &rsan, &o.o, nil)
		return qty, err
	}, buf)
	err = fd.eofError(n, err)
	if err != nil {
		return n, nil, err
	}
	sa, _ := rsa.Sockaddr()
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

	rsa := wsaRsaPool.Get().(*syscall.RawSockaddrAny)
	defer wsaRsaPool.Put(rsa)
	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		rsan := int32(unsafe.Sizeof(*rsa))
		var flags uint32
		err = syscall.WSARecvFrom(fd.Sysfd, newWsaBuf(buf), 1, &qty, &flags, rsa, &rsan, &o.o, nil)
		return qty, err
	}, buf)
	err = fd.eofError(n, err)
	if err != nil {
		return n, err
	}
	rawToSockaddrInet4(rsa, sa4)
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

	rsa := wsaRsaPool.Get().(*syscall.RawSockaddrAny)
	defer wsaRsaPool.Put(rsa)
	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		rsan := int32(unsafe.Sizeof(*rsa))
		var flags uint32
		err = syscall.WSARecvFrom(fd.Sysfd, newWsaBuf(buf), 1, &qty, &flags, rsa, &rsan, &o.o, nil)
		return qty, err
	}, buf)
	err = fd.eofError(n, err)
	if err != nil {
		return n, err
	}
	rawToSockaddrInet6(rsa, sa6)
	return n, err
}

// Write implements io.Writer.
func (fd *FD) Write(buf []byte) (int, error) {
	if fd.kind == kindFile {
		if err := fd.readWriteLock(); err != nil {
			return 0, err
		}
		defer fd.readWriteUnlock()
	} else {
		if err := fd.writeLock(); err != nil {
			return 0, err
		}
		defer fd.writeUnlock()
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
			n, err = fd.execIO('w', func(o *operation) (qty uint32, err error) {
				err = syscall.WriteFile(fd.Sysfd, b, &qty, fd.overlapped(o))
				return qty, err
			}, b)
			fd.addOffset(n)
		case kindNet:
			if race.Enabled {
				race.ReleaseMerge(unsafe.Pointer(&ioSync))
			}
			n, err = fd.execIO('w', func(o *operation) (qty uint32, err error) {
				err = syscall.WSASend(fd.Sysfd, newWsaBuf(b), 1, &qty, 0, &o.o, nil)
				return qty, err
			}, b)
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

	if err := fd.readWriteLock(); err != nil {
		return 0, err
	}
	defer fd.readWriteUnlock()

	var ntotal int
	for {
		max := len(buf)
		if max-ntotal > maxRW {
			max = ntotal + maxRW
		}
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			// Overlapped handles don't have the file pointer updated
			// when performing I/O operations, so there is no need to
			// call Seek to reset the file pointer.
			// Also, some overlapped file handles don't support seeking.
			// See https://go.dev/issues/74951.
			if fd.isBlocking {
				curoffset, err := syscall.Seek(fd.Sysfd, 0, io.SeekCurrent)
				if err != nil {
					return 0, err
				}
				defer syscall.Seek(fd.Sysfd, curoffset, io.SeekStart)
			}
			o.setOffset(off + int64(ntotal))

			err = syscall.WriteFile(fd.Sysfd, buf[ntotal:max], &qty, &o.o)
			return qty, err
		}, buf[ntotal:max])
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
	bufs := newWSABufs(buf)
	defer freeWSABufs(bufs)
	n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
		err = syscall.WSASend(fd.Sysfd, &(*bufs)[0], uint32(len(*bufs)), &qty, 0, &o.o, nil)
		return qty, err
	}, nil)
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
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			err = syscall.WSASendto(fd.Sysfd, &syscall.WSABuf{}, 1, &qty, 0, sa, &o.o, nil)
			return qty, err
		}, nil)
		return n, err
	}

	ntotal := 0
	for len(buf) > 0 {
		b := buf
		if len(b) > maxRW {
			b = b[:maxRW]
		}
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			err = syscall.WSASendto(fd.Sysfd, newWsaBuf(b), 1, &qty, 0, sa, &o.o, nil)
			return qty, err
		}, b)
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
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			err = windows.WSASendtoInet4(fd.Sysfd, &syscall.WSABuf{}, 1, &qty, 0, sa4, &o.o, nil)
			return qty, err
		}, nil)
		return n, err
	}

	ntotal := 0
	for len(buf) > 0 {
		b := buf
		if len(b) > maxRW {
			b = b[:maxRW]
		}
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			err = windows.WSASendtoInet4(fd.Sysfd, newWsaBuf(b), 1, &qty, 0, sa4, &o.o, nil)
			return qty, err
		}, b)
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
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			err = windows.WSASendtoInet6(fd.Sysfd, &syscall.WSABuf{}, 1, &qty, 0, sa6, &o.o, nil)
			return qty, err
		}, nil)
		return n, err
	}

	ntotal := 0
	for len(buf) > 0 {
		b := buf
		if len(b) > maxRW {
			b = b[:maxRW]
		}
		n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
			err = windows.WSASendtoInet6(fd.Sysfd, newWsaBuf(b), 1, &qty, 0, sa6, &o.o, nil)
			return qty, err
		}, b)
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
	_, err := fd.execIO('w', func(o *operation) (uint32, error) {
		return 0, ConnectExFunc(fd.Sysfd, ra, nil, 0, nil, &o.o)
	}, nil)
	return err
}

func (fd *FD) acceptOne(s syscall.Handle, rawsa []syscall.RawSockaddrAny) (string, error) {
	// Submit accept request.
	rsan := uint32(unsafe.Sizeof(rawsa[0]))
	_, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		err = AcceptFunc(fd.Sysfd, s, (*byte)(unsafe.Pointer(&rawsa[0])), 0, rsan, rsan, &qty, &o.o)
		return qty, err

	}, nil)
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

	var rawsa [2]syscall.RawSockaddrAny
	for {
		s, err := sysSocket()
		if err != nil {
			return syscall.InvalidHandle, nil, 0, "", err
		}

		errcall, err := fd.acceptOne(s, rawsa[:])
		if err == nil {
			return s, rawsa[:], uint32(unsafe.Sizeof(rawsa[0])), "", nil
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
	if err := fd.readWriteLock(); err != nil {
		return 0, err
	}
	defer fd.readWriteUnlock()

	if !fd.isBlocking {
		// Windows doesn't use the file pointer for overlapped file handles,
		// there is no point on calling syscall.Seek.
		var newOffset int64
		switch whence {
		case io.SeekStart:
			newOffset = offset
		case io.SeekCurrent:
			newOffset = fd.offset + offset
		case io.SeekEnd:
			var size int64
			if err := windows.GetFileSizeEx(fd.Sysfd, &size); err != nil {
				return 0, err
			}
			newOffset = size + offset
		default:
			return 0, windows.ERROR_INVALID_PARAMETER
		}
		if newOffset < 0 {
			return 0, windows.ERROR_NEGATIVE_SEEK
		}
		fd.setOffset(newOffset)
		return newOffset, nil
	}
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
		_, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
			var flags uint32
			if !fd.IsStream {
				flags |= windows.MSG_PEEK
			}
			err = syscall.WSARecv(fd.Sysfd, &syscall.WSABuf{}, 1, &qty, &flags, &o.o, nil)
			return qty, err
		}, nil)
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

	msg := newWSAMsg(p, oob, flags, true)
	defer freeWSAMsg(msg)
	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		err = windows.WSARecvMsg(fd.Sysfd, msg, &qty, &o.o, nil)
		return qty, err
	}, nil)
	err = fd.eofError(n, err)
	var sa syscall.Sockaddr
	if err == nil {
		sa, err = msg.Name.Sockaddr()
	}
	return n, int(msg.Control.Len), int(msg.Flags), sa, err
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

	msg := newWSAMsg(p, oob, flags, true)
	defer freeWSAMsg(msg)
	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		err = windows.WSARecvMsg(fd.Sysfd, msg, &qty, &o.o, nil)
		return qty, err
	}, nil)
	err = fd.eofError(n, err)
	if err == nil {
		rawToSockaddrInet4(msg.Name, sa4)
	}
	return n, int(msg.Control.Len), int(msg.Flags), err
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

	msg := newWSAMsg(p, oob, flags, true)
	defer freeWSAMsg(msg)
	n, err := fd.execIO('r', func(o *operation) (qty uint32, err error) {
		err = windows.WSARecvMsg(fd.Sysfd, msg, &qty, &o.o, nil)
		return qty, err
	}, nil)
	err = fd.eofError(n, err)
	if err == nil {
		rawToSockaddrInet6(msg.Name, sa6)
	}
	return n, int(msg.Control.Len), int(msg.Flags), err
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

	msg := newWSAMsg(p, oob, 0, sa != nil)
	defer freeWSAMsg(msg)
	if sa != nil {
		var err error
		msg.Namelen, err = sockaddrToRaw(msg.Name, sa)
		if err != nil {
			return 0, 0, err
		}
	}
	n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
		err = windows.WSASendMsg(fd.Sysfd, msg, 0, nil, &o.o, nil)
		return qty, err
	}, nil)
	return n, int(msg.Control.Len), err
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

	msg := newWSAMsg(p, oob, 0, sa != nil)
	defer freeWSAMsg(msg)
	if sa != nil {
		msg.Namelen = sockaddrInet4ToRaw(msg.Name, sa)
	}
	n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
		err = windows.WSASendMsg(fd.Sysfd, msg, 0, nil, &o.o, nil)
		return qty, err
	}, nil)
	return n, int(msg.Control.Len), err
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

	msg := newWSAMsg(p, oob, 0, sa != nil)
	defer freeWSAMsg(msg)
	if sa != nil {
		msg.Namelen = sockaddrInet6ToRaw(msg.Name, sa)
	}
	n, err := fd.execIO('w', func(o *operation) (qty uint32, err error) {
		err = windows.WSASendMsg(fd.Sysfd, msg, 0, nil, &o.o, nil)
		return qty, err
	}, nil)
	return n, int(msg.Control.Len), err
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
