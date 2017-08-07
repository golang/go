// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"errors"
	"internal/race"
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

// CancelIo Windows API cancels all outstanding IO for a particular
// socket on current thread. To overcome that limitation, we run
// special goroutine, locked to OS single thread, that both starts
// and cancels IO. It means, there are 2 unavoidable thread switches
// for every IO.
// Some newer versions of Windows has new CancelIoEx API, that does
// not have that limitation and can be used from any thread. This
// package uses CancelIoEx API, if present, otherwise it fallback
// to CancelIo.

var (
	canCancelIO                               bool // determines if CancelIoEx API is present
	skipSyncNotif                             bool
	hasLoadSetFileCompletionNotificationModes bool
)

func init() {
	var d syscall.WSAData
	e := syscall.WSAStartup(uint32(0x202), &d)
	if e != nil {
		initErr = e
	}
	canCancelIO = syscall.LoadCancelIoEx() == nil
	hasLoadSetFileCompletionNotificationModes = syscall.LoadSetFileCompletionNotificationModes() == nil
	if hasLoadSetFileCompletionNotificationModes {
		// It's not safe to use FILE_SKIP_COMPLETION_PORT_ON_SUCCESS if non IFS providers are installed:
		// http://support.microsoft.com/kb/2568167
		skipSyncNotif = true
		protos := [2]int32{syscall.IPPROTO_TCP, 0}
		var buf [32]syscall.WSAProtocolInfo
		len := uint32(unsafe.Sizeof(buf))
		n, err := syscall.WSAEnumProtocols(&protos[0], &buf[0], &len)
		if err != nil {
			skipSyncNotif = false
		} else {
			for i := int32(0); i < n; i++ {
				if buf[i].ServiceFlags1&syscall.XP1_IFS_HANDLES == 0 {
					skipSyncNotif = false
					break
				}
			}
		}
	}
}

// operation contains superset of data necessary to perform all async IO.
type operation struct {
	// Used by IOCP interface, it must be first field
	// of the struct, as our code rely on it.
	o syscall.Overlapped

	// fields used by runtime.netpoll
	runtimeCtx uintptr
	mode       int32
	errno      int32
	qty        uint32

	// fields used only by net package
	fd     *FD
	errc   chan error
	buf    syscall.WSABuf
	sa     syscall.Sockaddr
	rsa    *syscall.RawSockaddrAny
	rsan   int32
	handle syscall.Handle
	flags  uint32
	bufs   []syscall.WSABuf
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
		var p *byte
		if len(b) > 0 {
			p = &b[0]
		}
		o.bufs = append(o.bufs, syscall.WSABuf{Len: uint32(len(b)), Buf: p})
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

// ioSrv executes net IO requests.
type ioSrv struct {
	req chan ioSrvReq
}

type ioSrvReq struct {
	o      *operation
	submit func(o *operation) error // if nil, cancel the operation
}

// ProcessRemoteIO will execute submit IO requests on behalf
// of other goroutines, all on a single os thread, so it can
// cancel them later. Results of all operations will be sent
// back to their requesters via channel supplied in request.
// It is used only when the CancelIoEx API is unavailable.
func (s *ioSrv) ProcessRemoteIO() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for r := range s.req {
		if r.submit != nil {
			r.o.errc <- r.submit(r.o)
		} else {
			r.o.errc <- syscall.CancelIo(r.o.fd.Sysfd)
		}
	}
}

// ExecIO executes a single IO operation o. It submits and cancels
// IO in the current thread for systems where Windows CancelIoEx API
// is available. Alternatively, it passes the request onto
// runtime netpoll and waits for completion or cancels request.
func (s *ioSrv) ExecIO(o *operation, submit func(o *operation) error) (int, error) {
	if o.fd.pd.runtimeCtx == 0 {
		return 0, errors.New("internal error: polling on unsupported descriptor type")
	}

	if !canCancelIO {
		onceStartServer.Do(startServer)
	}

	fd := o.fd
	// Notify runtime netpoll about starting IO.
	err := fd.pd.prepare(int(o.mode), fd.isFile)
	if err != nil {
		return 0, err
	}
	// Start IO.
	if canCancelIO {
		err = submit(o)
	} else {
		// Send request to a special dedicated thread,
		// so it can stop the IO with CancelIO later.
		s.req <- ioSrvReq{o, submit}
		err = <-o.errc
	}
	switch err {
	case nil:
		// IO completed immediately
		if o.fd.skipSyncNotif {
			// No completion message will follow, so return immediately.
			return int(o.qty), nil
		}
		// Need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for its completion.
		err = nil
	default:
		return 0, err
	}
	// Wait for our request to complete.
	err = fd.pd.wait(int(o.mode), fd.isFile)
	if err == nil {
		// All is good. Extract our IO results and return.
		if o.errno != 0 {
			err = syscall.Errno(o.errno)
			return 0, err
		}
		return int(o.qty), nil
	}
	// IO is interrupted by "close" or "timeout"
	netpollErr := err
	switch netpollErr {
	case ErrNetClosing, ErrFileClosing, ErrTimeout:
		// will deal with those.
	default:
		panic("unexpected runtime.netpoll error: " + netpollErr.Error())
	}
	// Cancel our request.
	if canCancelIO {
		err := syscall.CancelIoEx(fd.Sysfd, &o.o)
		// Assuming ERROR_NOT_FOUND is returned, if IO is completed.
		if err != nil && err != syscall.ERROR_NOT_FOUND {
			// TODO(brainman): maybe do something else, but panic.
			panic(err)
		}
	} else {
		s.req <- ioSrvReq{o, nil}
		<-o.errc
	}
	// Wait for cancelation to complete.
	fd.pd.waitCanceled(int(o.mode))
	if o.errno != 0 {
		err = syscall.Errno(o.errno)
		if err == syscall.ERROR_OPERATION_ABORTED { // IO Canceled
			err = netpollErr
		}
		return 0, err
	}
	// We issued a cancelation request. But, it seems, IO operation succeeded
	// before the cancelation request run. We need to treat the IO operation as
	// succeeded (the bytes are actually sent/recv from network).
	return int(o.qty), nil
}

// Start helper goroutines.
var rsrv, wsrv ioSrv
var onceStartServer sync.Once

func startServer() {
	// This is called, once, when only the CancelIo API is available.
	// Start two special goroutines, both locked to an OS thread,
	// that start and cancel IO requests.
	// One will process read requests, while the other will do writes.
	rsrv.req = make(chan ioSrvReq)
	go rsrv.ProcessRemoteIO()
	wsrv.req = make(chan ioSrvReq)
	go wsrv.ProcessRemoteIO()
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

	// For console I/O.
	isConsole      bool
	lastbits       []byte   // first few bytes of the last incomplete rune in last write
	readuint16     []uint16 // buffer to hold uint16s obtained with ReadConsole
	readbyte       []byte   // buffer to hold decoding of readuint16 from utf16 to utf8
	readbyteOffset int      // readbyte[readOffset:] is yet to be consumed with file.Read

	skipSyncNotif bool

	// Whether this is a streaming descriptor, as opposed to a
	// packet-based descriptor like a UDP socket.
	IsStream bool

	// Whether a zero byte read indicates EOF. This is false for a
	// message based socket connection.
	ZeroReadIsEOF bool

	// Whether this is a normal file.
	isFile bool

	// Whether this is a directory.
	isDir bool
}

// Init initializes the FD. The Sysfd field should already be set.
// This can be called multiple times on a single FD.
// The net argument is a network name from the net package (e.g., "tcp"),
// or "file" or "console" or "dir".
func (fd *FD) Init(net string) (string, error) {
	if initErr != nil {
		return "", initErr
	}

	switch net {
	case "file":
		fd.isFile = true
	case "console":
		fd.isConsole = true
	case "dir":
		fd.isDir = true
	case "tcp", "tcp4", "tcp6":
	case "udp", "udp4", "udp6":
	case "ip", "ip4", "ip6":
	case "unix", "unixgram", "unixpacket":
	default:
		return "", errors.New("internal error: unknown network type " + net)
	}

	if !fd.isFile && !fd.isConsole && !fd.isDir {
		// Only call init for a network socket.
		// This means that we don't add files to the runtime poller.
		// Adding files to the runtime poller can confuse matters
		// if the user is doing their own overlapped I/O.
		// See issue #21172.
		//
		// In general the code below avoids calling the ExecIO
		// method for non-network sockets. If some method does
		// somehow call ExecIO, then ExecIO, and therefore the
		// calling method, will return an error, because
		// fd.pd.runtimeCtx will be 0.
		if err := fd.pd.init(fd); err != nil {
			return "", err
		}
	}
	if hasLoadSetFileCompletionNotificationModes {
		// We do not use events, so we can skip them always.
		flags := uint8(syscall.FILE_SKIP_SET_EVENT_ON_HANDLE)
		// It's not safe to skip completion notifications for UDP:
		// http://blogs.technet.com/b/winserverperformance/archive/2008/06/26/designing-applications-for-high-performance-part-iii.aspx
		if skipSyncNotif && (net == "tcp" || net == "file") {
			flags |= syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS
		}
		err := syscall.SetFileCompletionNotificationModes(fd.Sysfd, flags)
		if err == nil && flags&syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS != 0 {
			fd.skipSyncNotif = true
		}
	}
	// Disable SIO_UDP_CONNRESET behavior.
	// http://support.microsoft.com/kb/263823
	switch net {
	case "udp", "udp4", "udp6":
		ret := uint32(0)
		flag := uint32(0)
		size := uint32(unsafe.Sizeof(flag))
		err := syscall.WSAIoctl(fd.Sysfd, syscall.SIO_UDP_CONNRESET, (*byte)(unsafe.Pointer(&flag)), size, nil, 0, &ret, nil, 0)
		if err != nil {
			return "wsaioctl", err
		}
	}
	fd.rop.mode = 'r'
	fd.wop.mode = 'w'
	fd.rop.fd = fd
	fd.wop.fd = fd
	fd.rop.runtimeCtx = fd.pd.runtimeCtx
	fd.wop.runtimeCtx = fd.pd.runtimeCtx
	if !canCancelIO {
		fd.rop.errc = make(chan error)
		fd.wop.errc = make(chan error)
	}
	return "", nil
}

func (fd *FD) destroy() error {
	if fd.Sysfd == syscall.InvalidHandle {
		return syscall.EINVAL
	}
	// Poller may want to unregister fd in readiness notification mechanism,
	// so this must be executed before fd.CloseFunc.
	fd.pd.close()
	var err error
	if fd.isFile || fd.isConsole {
		err = syscall.CloseHandle(fd.Sysfd)
	} else if fd.isDir {
		err = syscall.FindClose(fd.Sysfd)
	} else {
		// The net package uses the CloseFunc variable for testing.
		err = CloseFunc(fd.Sysfd)
	}
	fd.Sysfd = syscall.InvalidHandle
	return err
}

// Close closes the FD. The underlying file descriptor is closed by
// the destroy method when there are no remaining references.
func (fd *FD) Close() error {
	if !fd.fdmu.increfAndClose() {
		return errClosing(fd.isFile)
	}
	// unblock pending reader and writer
	fd.pd.evict()
	return fd.decref()
}

// Shutdown wraps the shutdown network call.
func (fd *FD) Shutdown(how int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Shutdown(fd.Sysfd, how)
}

// Read implements io.Reader.
func (fd *FD) Read(buf []byte) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()

	var n int
	var err error
	if fd.isFile || fd.isDir || fd.isConsole {
		fd.l.Lock()
		defer fd.l.Unlock()
		if fd.isConsole {
			n, err = fd.readConsole(buf)
		} else {
			n, err = syscall.Read(fd.Sysfd, buf)
		}
		if err != nil {
			n = 0
		}
	} else {
		o := &fd.rop
		o.InitBuf(buf)
		n, err = rsrv.ExecIO(o, func(o *operation) error {
			return syscall.WSARecv(o.fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, &o.o, nil)
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
			n := utf8.EncodeRune(buf[len(buf):cap(buf)], r)
			buf = buf[:len(buf)+n]
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
	// Call incref, not readLock, because since pread specifies the
	// offset it is independent from other reads.
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()

	fd.l.Lock()
	defer fd.l.Unlock()
	curoffset, e := syscall.Seek(fd.Sysfd, 0, io.SeekCurrent)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(fd.Sysfd, curoffset, io.SeekStart)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.ReadFile(fd.Sysfd, b, &done, &o)
	if e != nil {
		done = 0
		if e == syscall.ERROR_HANDLE_EOF {
			e = io.EOF
		}
	}
	if len(b) != 0 {
		e = fd.eofError(int(done), e)
	}
	return int(done), e
}

// ReadFrom wraps the recvfrom network call.
func (fd *FD) ReadFrom(buf []byte) (int, syscall.Sockaddr, error) {
	if len(buf) == 0 {
		return 0, nil, nil
	}
	if err := fd.readLock(); err != nil {
		return 0, nil, err
	}
	defer fd.readUnlock()
	o := &fd.rop
	o.InitBuf(buf)
	n, err := rsrv.ExecIO(o, func(o *operation) error {
		if o.rsa == nil {
			o.rsa = new(syscall.RawSockaddrAny)
		}
		o.rsan = int32(unsafe.Sizeof(*o.rsa))
		return syscall.WSARecvFrom(o.fd.Sysfd, &o.buf, 1, &o.qty, &o.flags, o.rsa, &o.rsan, &o.o, nil)
	})
	err = fd.eofError(n, err)
	if err != nil {
		return n, nil, err
	}
	sa, _ := o.rsa.Sockaddr()
	return n, sa, nil
}

// Write implements io.Writer.
func (fd *FD) Write(buf []byte) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()

	var n int
	var err error
	if fd.isFile || fd.isDir || fd.isConsole {
		fd.l.Lock()
		defer fd.l.Unlock()
		if fd.isConsole {
			n, err = fd.writeConsole(buf)
		} else {
			n, err = syscall.Write(fd.Sysfd, buf)
		}
		if err != nil {
			n = 0
		}
	} else {
		if race.Enabled {
			race.ReleaseMerge(unsafe.Pointer(&ioSync))
		}
		o := &fd.wop
		o.InitBuf(buf)
		n, err = wsrv.ExecIO(o, func(o *operation) error {
			return syscall.WSASend(o.fd.Sysfd, &o.buf, 1, &o.qty, 0, &o.o, nil)
		})
	}
	return n, err
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
func (fd *FD) Pwrite(b []byte, off int64) (int, error) {
	// Call incref, not writeLock, because since pwrite specifies the
	// offset it is independent from other writes.
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()

	fd.l.Lock()
	defer fd.l.Unlock()
	curoffset, e := syscall.Seek(fd.Sysfd, 0, io.SeekCurrent)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(fd.Sysfd, curoffset, io.SeekStart)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.WriteFile(fd.Sysfd, b, &done, &o)
	if e != nil {
		return 0, e
	}
	return int(done), nil
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
	n, err := wsrv.ExecIO(o, func(o *operation) error {
		return syscall.WSASend(o.fd.Sysfd, &o.bufs[0], uint32(len(o.bufs)), &o.qty, 0, &o.o, nil)
	})
	o.ClearBufs()
	TestHookDidWritev(n)
	consume(buf, int64(n))
	return int64(n), err
}

// WriteTo wraps the sendto network call.
func (fd *FD) WriteTo(buf []byte, sa syscall.Sockaddr) (int, error) {
	if len(buf) == 0 {
		return 0, nil
	}
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	o := &fd.wop
	o.InitBuf(buf)
	o.sa = sa
	n, err := wsrv.ExecIO(o, func(o *operation) error {
		return syscall.WSASendto(o.fd.Sysfd, &o.buf, 1, &o.qty, 0, o.sa, &o.o, nil)
	})
	return n, err
}

// Call ConnectEx. This doesn't need any locking, since it is only
// called when the descriptor is first created. This is here rather
// than in the net package so that it can use fd.wop.
func (fd *FD) ConnectEx(ra syscall.Sockaddr) error {
	o := &fd.wop
	o.sa = ra
	_, err := wsrv.ExecIO(o, func(o *operation) error {
		return ConnectExFunc(o.fd.Sysfd, o.sa, nil, 0, nil, &o.o)
	})
	return err
}

func (fd *FD) acceptOne(s syscall.Handle, rawsa []syscall.RawSockaddrAny, o *operation) (string, error) {
	// Submit accept request.
	o.handle = s
	o.rsan = int32(unsafe.Sizeof(rawsa[0]))
	_, err := rsrv.ExecIO(o, func(o *operation) error {
		return AcceptFunc(o.fd.Sysfd, o.handle, (*byte)(unsafe.Pointer(&rawsa[0])), 0, uint32(o.rsan), uint32(o.rsan), &o.qty, &o.o)
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
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()

	fd.l.Lock()
	defer fd.l.Unlock()

	return syscall.Seek(fd.Sysfd, offset, whence)
}

// FindNextFile wraps syscall.FindNextFile.
func (fd *FD) FindNextFile(data *syscall.Win32finddata) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.FindNextFile(fd.Sysfd, data)
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

// RawControl invokes the user-defined function f for a non-IO
// operation.
func (fd *FD) RawControl(f func(uintptr)) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	f(uintptr(fd.Sysfd))
	return nil
}

// RawRead invokes the user-defined function f for a read operation.
func (fd *FD) RawRead(f func(uintptr) bool) error {
	return errors.New("not implemented")
}

// RawWrite invokes the user-defined function f for a write operation.
func (fd *FD) RawWrite(f func(uintptr) bool) error {
	return errors.New("not implemented")
}
