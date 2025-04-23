// Copyright 2009,2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and wrap
// it in our own nicer implementation, either here or in
// syscall_bsd.go or syscall_unix.go.

package unix

import (
	"fmt"
	"syscall"
	"unsafe"
)

//sys	closedir(dir uintptr) (err error)
//sys	readdir_r(dir uintptr, entry *Dirent, result **Dirent) (res Errno)

func fdopendir(fd int) (dir uintptr, err error) {
	r0, _, e1 := syscall_syscallPtr(libc_fdopendir_trampoline_addr, uintptr(fd), 0, 0)
	dir = uintptr(r0)
	if e1 != 0 {
		err = errnoErr(e1)
	}
	return
}

var libc_fdopendir_trampoline_addr uintptr

//go:cgo_import_dynamic libc_fdopendir fdopendir "/usr/lib/libSystem.B.dylib"

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	// Simulate Getdirentries using fdopendir/readdir_r/closedir.
	// We store the number of entries to skip in the seek
	// offset of fd. See issue #31368.
	// It's not the full required semantics, but should handle the case
	// of calling Getdirentries or ReadDirent repeatedly.
	// It won't handle assigning the results of lseek to *basep, or handle
	// the directory being edited underfoot.
	skip, err := Seek(fd, 0, 1 /* SEEK_CUR */)
	if err != nil {
		return 0, err
	}

	// We need to duplicate the incoming file descriptor
	// because the caller expects to retain control of it, but
	// fdopendir expects to take control of its argument.
	// Just Dup'ing the file descriptor is not enough, as the
	// result shares underlying state. Use Openat to make a really
	// new file descriptor referring to the same directory.
	fd2, err := Openat(fd, ".", O_RDONLY, 0)
	if err != nil {
		return 0, err
	}
	d, err := fdopendir(fd2)
	if err != nil {
		Close(fd2)
		return 0, err
	}
	defer closedir(d)

	var cnt int64
	for {
		var entry Dirent
		var entryp *Dirent
		e := readdir_r(d, &entry, &entryp)
		if e != 0 {
			return n, errnoErr(e)
		}
		if entryp == nil {
			break
		}
		if skip > 0 {
			skip--
			cnt++
			continue
		}

		reclen := int(entry.Reclen)
		if reclen > len(buf) {
			// Not enough room. Return for now.
			// The counter will let us know where we should start up again.
			// Note: this strategy for suspending in the middle and
			// restarting is O(n^2) in the length of the directory. Oh well.
			break
		}

		// Copy entry into return buffer.
		s := unsafe.Slice((*byte)(unsafe.Pointer(&entry)), reclen)
		copy(buf, s)

		buf = buf[reclen:]
		n += reclen
		cnt++
	}
	// Set the seek offset of the input fd to record
	// how many files we've already returned.
	_, err = Seek(fd, cnt, 0 /* SEEK_SET */)
	if err != nil {
		return n, err
	}

	return n, nil
}

// SockaddrDatalink implements the Sockaddr interface for AF_LINK type sockets.
type SockaddrDatalink struct {
	Len    uint8
	Family uint8
	Index  uint16
	Type   uint8
	Nlen   uint8
	Alen   uint8
	Slen   uint8
	Data   [12]int8
	raw    RawSockaddrDatalink
}

// SockaddrCtl implements the Sockaddr interface for AF_SYSTEM type sockets.
type SockaddrCtl struct {
	ID   uint32
	Unit uint32
	raw  RawSockaddrCtl
}

func (sa *SockaddrCtl) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Sc_len = SizeofSockaddrCtl
	sa.raw.Sc_family = AF_SYSTEM
	sa.raw.Ss_sysaddr = AF_SYS_CONTROL
	sa.raw.Sc_id = sa.ID
	sa.raw.Sc_unit = sa.Unit
	return unsafe.Pointer(&sa.raw), SizeofSockaddrCtl, nil
}

// SockaddrVM implements the Sockaddr interface for AF_VSOCK type sockets.
// SockaddrVM provides access to Darwin VM sockets: a mechanism that enables
// bidirectional communication between a hypervisor and its guest virtual
// machines.
type SockaddrVM struct {
	// CID and Port specify a context ID and port address for a VM socket.
	// Guests have a unique CID, and hosts may have a well-known CID of:
	//  - VMADDR_CID_HYPERVISOR: refers to the hypervisor process.
	//  - VMADDR_CID_LOCAL: refers to local communication (loopback).
	//  - VMADDR_CID_HOST: refers to other processes on the host.
	CID  uint32
	Port uint32
	raw  RawSockaddrVM
}

func (sa *SockaddrVM) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Len = SizeofSockaddrVM
	sa.raw.Family = AF_VSOCK
	sa.raw.Port = sa.Port
	sa.raw.Cid = sa.CID

	return unsafe.Pointer(&sa.raw), SizeofSockaddrVM, nil
}

func anyToSockaddrGOOS(fd int, rsa *RawSockaddrAny) (Sockaddr, error) {
	switch rsa.Addr.Family {
	case AF_SYSTEM:
		pp := (*RawSockaddrCtl)(unsafe.Pointer(rsa))
		if pp.Ss_sysaddr == AF_SYS_CONTROL {
			sa := new(SockaddrCtl)
			sa.ID = pp.Sc_id
			sa.Unit = pp.Sc_unit
			return sa, nil
		}
	case AF_VSOCK:
		pp := (*RawSockaddrVM)(unsafe.Pointer(rsa))
		sa := &SockaddrVM{
			CID:  pp.Cid,
			Port: pp.Port,
		}
		return sa, nil
	}
	return nil, EAFNOSUPPORT
}

// Some external packages rely on SYS___SYSCTL being defined to implement their
// own sysctl wrappers. Provide it here, even though direct syscalls are no
// longer supported on darwin.
const SYS___SYSCTL = SYS_SYSCTL

// Translate "kern.hostname" to []_C_int{0,1,2,3}.
func nametomib(name string) (mib []_C_int, err error) {
	const siz = unsafe.Sizeof(mib[0])

	// NOTE(rsc): It seems strange to set the buffer to have
	// size CTL_MAXNAME+2 but use only CTL_MAXNAME
	// as the size. I don't know why the +2 is here, but the
	// kernel uses +2 for its own implementation of this function.
	// I am scared that if we don't include the +2 here, the kernel
	// will silently write 2 words farther than we specify
	// and we'll get memory corruption.
	var buf [CTL_MAXNAME + 2]_C_int
	n := uintptr(CTL_MAXNAME) * siz

	p := (*byte)(unsafe.Pointer(&buf[0]))
	bytes, err := ByteSliceFromString(name)
	if err != nil {
		return nil, err
	}

	// Magic sysctl: "setting" 0.3 to a string name
	// lets you read back the array of integers form.
	if err = sysctl([]_C_int{0, 3}, p, &n, &bytes[0], uintptr(len(name))); err != nil {
		return nil, err
	}
	return buf[0 : n/siz], nil
}

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Ino), unsafe.Sizeof(Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Reclen), unsafe.Sizeof(Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Namlen), unsafe.Sizeof(Dirent{}.Namlen))
}

func PtraceAttach(pid int) (err error) { return ptrace(PT_ATTACH, pid, 0, 0) }
func PtraceDetach(pid int) (err error) { return ptrace(PT_DETACH, pid, 0, 0) }
func PtraceDenyAttach() (err error)    { return ptrace(PT_DENY_ATTACH, 0, 0, 0) }

//sysnb	pipe(p *[2]int32) (err error)

func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var x [2]int32
	err = pipe(&x)
	if err == nil {
		p[0] = int(x[0])
		p[1] = int(x[1])
	}
	return
}

func Getfsstat(buf []Statfs_t, flags int) (n int, err error) {
	var _p0 unsafe.Pointer
	var bufsize uintptr
	if len(buf) > 0 {
		_p0 = unsafe.Pointer(&buf[0])
		bufsize = unsafe.Sizeof(Statfs_t{}) * uintptr(len(buf))
	}
	return getfsstat(_p0, bufsize, flags)
}

func xattrPointer(dest []byte) *byte {
	// It's only when dest is set to NULL that the OS X implementations of
	// getxattr() and listxattr() return the current sizes of the named attributes.
	// An empty byte array is not sufficient. To maintain the same behaviour as the
	// linux implementation, we wrap around the system calls and pass in NULL when
	// dest is empty.
	var destp *byte
	if len(dest) > 0 {
		destp = &dest[0]
	}
	return destp
}

//sys	getxattr(path string, attr string, dest *byte, size int, position uint32, options int) (sz int, err error)

func Getxattr(path string, attr string, dest []byte) (sz int, err error) {
	return getxattr(path, attr, xattrPointer(dest), len(dest), 0, 0)
}

func Lgetxattr(link string, attr string, dest []byte) (sz int, err error) {
	return getxattr(link, attr, xattrPointer(dest), len(dest), 0, XATTR_NOFOLLOW)
}

//sys	fgetxattr(fd int, attr string, dest *byte, size int, position uint32, options int) (sz int, err error)

func Fgetxattr(fd int, attr string, dest []byte) (sz int, err error) {
	return fgetxattr(fd, attr, xattrPointer(dest), len(dest), 0, 0)
}

//sys	setxattr(path string, attr string, data *byte, size int, position uint32, options int) (err error)

func Setxattr(path string, attr string, data []byte, flags int) (err error) {
	// The parameters for the OS X implementation vary slightly compared to the
	// linux system call, specifically the position parameter:
	//
	//  linux:
	//      int setxattr(
	//          const char *path,
	//          const char *name,
	//          const void *value,
	//          size_t size,
	//          int flags
	//      );
	//
	//  darwin:
	//      int setxattr(
	//          const char *path,
	//          const char *name,
	//          void *value,
	//          size_t size,
	//          u_int32_t position,
	//          int options
	//      );
	//
	// position specifies the offset within the extended attribute. In the
	// current implementation, only the resource fork extended attribute makes
	// use of this argument. For all others, position is reserved. We simply
	// default to setting it to zero.
	return setxattr(path, attr, xattrPointer(data), len(data), 0, flags)
}

func Lsetxattr(link string, attr string, data []byte, flags int) (err error) {
	return setxattr(link, attr, xattrPointer(data), len(data), 0, flags|XATTR_NOFOLLOW)
}

//sys	fsetxattr(fd int, attr string, data *byte, size int, position uint32, options int) (err error)

func Fsetxattr(fd int, attr string, data []byte, flags int) (err error) {
	return fsetxattr(fd, attr, xattrPointer(data), len(data), 0, 0)
}

//sys	removexattr(path string, attr string, options int) (err error)

func Removexattr(path string, attr string) (err error) {
	// We wrap around and explicitly zero out the options provided to the OS X
	// implementation of removexattr, we do so for interoperability with the
	// linux variant.
	return removexattr(path, attr, 0)
}

func Lremovexattr(link string, attr string) (err error) {
	return removexattr(link, attr, XATTR_NOFOLLOW)
}

//sys	fremovexattr(fd int, attr string, options int) (err error)

func Fremovexattr(fd int, attr string) (err error) {
	return fremovexattr(fd, attr, 0)
}

//sys	listxattr(path string, dest *byte, size int, options int) (sz int, err error)

func Listxattr(path string, dest []byte) (sz int, err error) {
	return listxattr(path, xattrPointer(dest), len(dest), 0)
}

func Llistxattr(link string, dest []byte) (sz int, err error) {
	return listxattr(link, xattrPointer(dest), len(dest), XATTR_NOFOLLOW)
}

//sys	flistxattr(fd int, dest *byte, size int, options int) (sz int, err error)

func Flistxattr(fd int, dest []byte) (sz int, err error) {
	return flistxattr(fd, xattrPointer(dest), len(dest), 0)
}

//sys	utimensat(dirfd int, path string, times *[2]Timespec, flags int) (err error)

/*
 * Wrapped
 */

//sys	fcntl(fd int, cmd int, arg int) (val int, err error)

//sys	kill(pid int, signum int, posix int) (err error)

func Kill(pid int, signum syscall.Signal) (err error) { return kill(pid, int(signum), 1) }

//sys	ioctl(fd int, req uint, arg uintptr) (err error)
//sys	ioctlPtr(fd int, req uint, arg unsafe.Pointer) (err error) = SYS_IOCTL

func IoctlCtlInfo(fd int, ctlInfo *CtlInfo) error {
	return ioctlPtr(fd, CTLIOCGINFO, unsafe.Pointer(ctlInfo))
}

// IfreqMTU is struct ifreq used to get or set a network device's MTU.
type IfreqMTU struct {
	Name [IFNAMSIZ]byte
	MTU  int32
}

// IoctlGetIfreqMTU performs the SIOCGIFMTU ioctl operation on fd to get the MTU
// of the network device specified by ifname.
func IoctlGetIfreqMTU(fd int, ifname string) (*IfreqMTU, error) {
	var ifreq IfreqMTU
	copy(ifreq.Name[:], ifname)
	err := ioctlPtr(fd, SIOCGIFMTU, unsafe.Pointer(&ifreq))
	return &ifreq, err
}

// IoctlSetIfreqMTU performs the SIOCSIFMTU ioctl operation on fd to set the MTU
// of the network device specified by ifreq.Name.
func IoctlSetIfreqMTU(fd int, ifreq *IfreqMTU) error {
	return ioctlPtr(fd, SIOCSIFMTU, unsafe.Pointer(ifreq))
}

//sys	renamexNp(from string, to string, flag uint32) (err error)

func RenamexNp(from string, to string, flag uint32) (err error) {
	return renamexNp(from, to, flag)
}

//sys	renameatxNp(fromfd int, from string, tofd int, to string, flag uint32) (err error)

func RenameatxNp(fromfd int, from string, tofd int, to string, flag uint32) (err error) {
	return renameatxNp(fromfd, from, tofd, to, flag)
}

//sys	sysctl(mib []_C_int, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (err error) = SYS_SYSCTL

func Uname(uname *Utsname) error {
	mib := []_C_int{CTL_KERN, KERN_OSTYPE}
	n := unsafe.Sizeof(uname.Sysname)
	if err := sysctl(mib, &uname.Sysname[0], &n, nil, 0); err != nil {
		return err
	}

	mib = []_C_int{CTL_KERN, KERN_HOSTNAME}
	n = unsafe.Sizeof(uname.Nodename)
	if err := sysctl(mib, &uname.Nodename[0], &n, nil, 0); err != nil {
		return err
	}

	mib = []_C_int{CTL_KERN, KERN_OSRELEASE}
	n = unsafe.Sizeof(uname.Release)
	if err := sysctl(mib, &uname.Release[0], &n, nil, 0); err != nil {
		return err
	}

	mib = []_C_int{CTL_KERN, KERN_VERSION}
	n = unsafe.Sizeof(uname.Version)
	if err := sysctl(mib, &uname.Version[0], &n, nil, 0); err != nil {
		return err
	}

	// The version might have newlines or tabs in it, convert them to
	// spaces.
	for i, b := range uname.Version {
		if b == '\n' || b == '\t' {
			if i == len(uname.Version)-1 {
				uname.Version[i] = 0
			} else {
				uname.Version[i] = ' '
			}
		}
	}

	mib = []_C_int{CTL_HW, HW_MACHINE}
	n = unsafe.Sizeof(uname.Machine)
	if err := sysctl(mib, &uname.Machine[0], &n, nil, 0); err != nil {
		return err
	}

	return nil
}

func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	var length = int64(count)
	err = sendfile(infd, outfd, *offset, &length, nil, 0)
	written = int(length)
	return
}

func GetsockoptIPMreqn(fd, level, opt int) (*IPMreqn, error) {
	var value IPMreqn
	vallen := _Socklen(SizeofIPMreqn)
	errno := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, errno
}

func SetsockoptIPMreqn(fd, level, opt int, mreq *IPMreqn) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), unsafe.Sizeof(*mreq))
}

// GetsockoptXucred is a getsockopt wrapper that returns an Xucred struct.
// The usual level and opt are SOL_LOCAL and LOCAL_PEERCRED, respectively.
func GetsockoptXucred(fd, level, opt int) (*Xucred, error) {
	x := new(Xucred)
	vallen := _Socklen(SizeofXucred)
	err := getsockopt(fd, level, opt, unsafe.Pointer(x), &vallen)
	return x, err
}

func GetsockoptTCPConnectionInfo(fd, level, opt int) (*TCPConnectionInfo, error) {
	var value TCPConnectionInfo
	vallen := _Socklen(SizeofTCPConnectionInfo)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func SysctlKinfoProc(name string, args ...int) (*KinfoProc, error) {
	mib, err := sysctlmib(name, args...)
	if err != nil {
		return nil, err
	}

	var kinfo KinfoProc
	n := uintptr(SizeofKinfoProc)
	if err := sysctl(mib, (*byte)(unsafe.Pointer(&kinfo)), &n, nil, 0); err != nil {
		return nil, err
	}
	if n != SizeofKinfoProc {
		return nil, EIO
	}
	return &kinfo, nil
}

func SysctlKinfoProcSlice(name string, args ...int) ([]KinfoProc, error) {
	mib, err := sysctlmib(name, args...)
	if err != nil {
		return nil, err
	}

	for {
		// Find size.
		n := uintptr(0)
		if err := sysctl(mib, nil, &n, nil, 0); err != nil {
			return nil, err
		}
		if n == 0 {
			return nil, nil
		}
		if n%SizeofKinfoProc != 0 {
			return nil, fmt.Errorf("sysctl() returned a size of %d, which is not a multiple of %d", n, SizeofKinfoProc)
		}

		// Read into buffer of that size.
		buf := make([]KinfoProc, n/SizeofKinfoProc)
		if err := sysctl(mib, (*byte)(unsafe.Pointer(&buf[0])), &n, nil, 0); err != nil {
			if err == ENOMEM {
				// Process table grew. Try again.
				continue
			}
			return nil, err
		}
		if n%SizeofKinfoProc != 0 {
			return nil, fmt.Errorf("sysctl() returned a size of %d, which is not a multiple of %d", n, SizeofKinfoProc)
		}

		// The actual call may return less than the original reported required
		// size so ensure we deal with that.
		return buf[:n/SizeofKinfoProc], nil
	}
}

//sys	pthread_chdir_np(path string) (err error)

func PthreadChdir(path string) (err error) {
	return pthread_chdir_np(path)
}

//sys	pthread_fchdir_np(fd int) (err error)

func PthreadFchdir(fd int) (err error) {
	return pthread_fchdir_np(fd)
}

// Connectx calls connectx(2) to initiate a connection on a socket.
//
// srcIf, srcAddr, and dstAddr are filled into a [SaEndpoints] struct and passed as the endpoints argument.
//
//   - srcIf is the optional source interface index. 0 means unspecified.
//   - srcAddr is the optional source address. nil means unspecified.
//   - dstAddr is the destination address.
//
// On success, Connectx returns the number of bytes enqueued for transmission.
func Connectx(fd int, srcIf uint32, srcAddr, dstAddr Sockaddr, associd SaeAssocID, flags uint32, iov []Iovec, connid *SaeConnID) (n uintptr, err error) {
	endpoints := SaEndpoints{
		Srcif: srcIf,
	}

	if srcAddr != nil {
		addrp, addrlen, err := srcAddr.sockaddr()
		if err != nil {
			return 0, err
		}
		endpoints.Srcaddr = (*RawSockaddr)(addrp)
		endpoints.Srcaddrlen = uint32(addrlen)
	}

	if dstAddr != nil {
		addrp, addrlen, err := dstAddr.sockaddr()
		if err != nil {
			return 0, err
		}
		endpoints.Dstaddr = (*RawSockaddr)(addrp)
		endpoints.Dstaddrlen = uint32(addrlen)
	}

	err = connectx(fd, &endpoints, associd, flags, iov, &n, connid)
	return
}

// sys	connectx(fd int, endpoints *SaEndpoints, associd SaeAssocID, flags uint32, iov []Iovec, n *uintptr, connid *SaeConnID) (err error)
const minIovec = 8

func Readv(fd int, iovs [][]byte) (n int, err error) {
	if !darwinKernelVersionMin(11, 0, 0) {
		return 0, ENOSYS
	}

	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	n, err = readv(fd, iovecs)
	readvRacedetect(iovecs, n, err)
	return n, err
}

func Preadv(fd int, iovs [][]byte, offset int64) (n int, err error) {
	if !darwinKernelVersionMin(11, 0, 0) {
		return 0, ENOSYS
	}
	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	n, err = preadv(fd, iovecs, offset)
	readvRacedetect(iovecs, n, err)
	return n, err
}

func Writev(fd int, iovs [][]byte) (n int, err error) {
	if !darwinKernelVersionMin(11, 0, 0) {
		return 0, ENOSYS
	}

	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = writev(fd, iovecs)
	writevRacedetect(iovecs, n)
	return n, err
}

func Pwritev(fd int, iovs [][]byte, offset int64) (n int, err error) {
	if !darwinKernelVersionMin(11, 0, 0) {
		return 0, ENOSYS
	}

	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = pwritev(fd, iovecs, offset)
	writevRacedetect(iovecs, n)
	return n, err
}

func appendBytes(vecs []Iovec, bs [][]byte) []Iovec {
	for _, b := range bs {
		var v Iovec
		v.SetLen(len(b))
		if len(b) > 0 {
			v.Base = &b[0]
		} else {
			v.Base = (*byte)(unsafe.Pointer(&_zero))
		}
		vecs = append(vecs, v)
	}
	return vecs
}

func writevRacedetect(iovecs []Iovec, n int) {
	if !raceenabled {
		return
	}
	for i := 0; n > 0 && i < len(iovecs); i++ {
		m := int(iovecs[i].Len)
		if m > n {
			m = n
		}
		n -= m
		if m > 0 {
			raceReadRange(unsafe.Pointer(iovecs[i].Base), m)
		}
	}
}

func readvRacedetect(iovecs []Iovec, n int, err error) {
	if !raceenabled {
		return
	}
	for i := 0; n > 0 && i < len(iovecs); i++ {
		m := int(iovecs[i].Len)
		if m > n {
			m = n
		}
		n -= m
		if m > 0 {
			raceWriteRange(unsafe.Pointer(iovecs[i].Base), m)
		}
	}
	if err == nil {
		raceAcquire(unsafe.Pointer(&ioSync))
	}
}

func darwinMajorMinPatch() (maj, min, patch int, err error) {
	var un Utsname
	err = Uname(&un)
	if err != nil {
		return
	}

	var mmp [3]int
	c := 0
Loop:
	for _, b := range un.Release[:] {
		switch {
		case b >= '0' && b <= '9':
			mmp[c] = 10*mmp[c] + int(b-'0')
		case b == '.':
			c++
			if c > 2 {
				return 0, 0, 0, ENOTSUP
			}
		case b == 0:
			break Loop
		default:
			return 0, 0, 0, ENOTSUP
		}
	}
	if c != 2 {
		return 0, 0, 0, ENOTSUP
	}
	return mmp[0], mmp[1], mmp[2], nil
}

func darwinKernelVersionMin(maj, min, patch int) bool {
	actualMaj, actualMin, actualPatch, err := darwinMajorMinPatch()
	if err != nil {
		return false
	}
	return actualMaj > maj || actualMaj == maj && (actualMin > min || actualMin == min && actualPatch >= patch)
}

//sys	sendfile(infd int, outfd int, offset int64, len *int64, hdtr unsafe.Pointer, flags int) (err error)

//sys	shmat(id int, addr uintptr, flag int) (ret uintptr, err error)
//sys	shmctl(id int, cmd int, buf *SysvShmDesc) (result int, err error)
//sys	shmdt(addr uintptr) (err error)
//sys	shmget(key int, size int, flag int) (id int, err error)

/*
 * Exposed directly
 */
//sys	Access(path string, mode uint32) (err error)
//sys	Adjtime(delta *Timeval, olddelta *Timeval) (err error)
//sys	Chdir(path string) (err error)
//sys	Chflags(path string, flags int) (err error)
//sys	Chmod(path string, mode uint32) (err error)
//sys	Chown(path string, uid int, gid int) (err error)
//sys	Chroot(path string) (err error)
//sys	ClockGettime(clockid int32, time *Timespec) (err error)
//sys	Close(fd int) (err error)
//sys	Clonefile(src string, dst string, flags int) (err error)
//sys	Clonefileat(srcDirfd int, src string, dstDirfd int, dst string, flags int) (err error)
//sys	Dup(fd int) (nfd int, err error)
//sys	Dup2(from int, to int) (err error)
//sys	Exchangedata(path1 string, path2 string, options int) (err error)
//sys	Exit(code int)
//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchflags(fd int, flags int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error)
//sys	Fchown(fd int, uid int, gid int) (err error)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//sys	Fclonefileat(srcDirfd int, dstDirfd int, dst string, flags int) (err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fpathconf(fd int, name int) (val int, err error)
//sys	Fsync(fd int) (err error)
//sys	Ftruncate(fd int, length int64) (err error)
//sys	Getcwd(buf []byte) (n int, err error)
//sys	Getdtablesize() (size int)
//sysnb	Getegid() (egid int)
//sysnb	Geteuid() (uid int)
//sysnb	Getgid() (gid int)
//sysnb	Getpgid(pid int) (pgid int, err error)
//sysnb	Getpgrp() (pgrp int)
//sysnb	Getpid() (pid int)
//sysnb	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (prio int, err error)
//sysnb	Getrlimit(which int, lim *Rlimit) (err error)
//sysnb	Getrusage(who int, rusage *Rusage) (err error)
//sysnb	Getsid(pid int) (sid int, err error)
//sysnb	Gettimeofday(tp *Timeval) (err error)
//sysnb	Getuid() (uid int)
//sysnb	Issetugid() (tainted bool)
//sys	Kqueue() (fd int, err error)
//sys	Lchown(path string, uid int, gid int) (err error)
//sys	Link(path string, link string) (err error)
//sys	Linkat(pathfd int, path string, linkfd int, link string, flags int) (err error)
//sys	Listen(s int, backlog int) (err error)
//sys	Mkdir(path string, mode uint32) (err error)
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//sys	Mkfifo(path string, mode uint32) (err error)
//sys	Mknod(path string, mode uint32, dev int) (err error)
//sys	Mount(fsType string, dir string, flags int, data unsafe.Pointer) (err error)
//sys	Open(path string, mode int, perm uint32) (fd int, err error)
//sys	Openat(dirfd int, path string, mode int, perm uint32) (fd int, err error)
//sys	Pathconf(path string, name int) (val int, err error)
//sys	pread(fd int, p []byte, offset int64) (n int, err error)
//sys	pwrite(fd int, p []byte, offset int64) (n int, err error)
//sys	read(fd int, p []byte) (n int, err error)
//sys	Readlink(path string, buf []byte) (n int, err error)
//sys	Readlinkat(dirfd int, path string, buf []byte) (n int, err error)
//sys	Rename(from string, to string) (err error)
//sys	Renameat(fromfd int, from string, tofd int, to string) (err error)
//sys	Revoke(path string) (err error)
//sys	Rmdir(path string) (err error)
//sys	Seek(fd int, offset int64, whence int) (newoffset int64, err error) = SYS_LSEEK
//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error)
//sys	Setattrlist(path string, attrlist *Attrlist, attrBuf []byte, options int) (err error)
//sys	Setegid(egid int) (err error)
//sysnb	Seteuid(euid int) (err error)
//sysnb	Setgid(gid int) (err error)
//sys	Setlogin(name string) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sys	Setpriority(which int, who int, prio int) (err error)
//sys	Setprivexec(flag int) (err error)
//sysnb	Setregid(rgid int, egid int) (err error)
//sysnb	Setreuid(ruid int, euid int) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Settimeofday(tp *Timeval) (err error)
//sysnb	Setuid(uid int) (err error)
//sys	Symlink(path string, link string) (err error)
//sys	Symlinkat(oldpath string, newdirfd int, newpath string) (err error)
//sys	Sync() (err error)
//sys	Truncate(path string, length int64) (err error)
//sys	Umask(newmask int) (oldmask int)
//sys	Undelete(path string) (err error)
//sys	Unlink(path string) (err error)
//sys	Unlinkat(dirfd int, path string, flags int) (err error)
//sys	Unmount(path string, flags int) (err error)
//sys	write(fd int, p []byte) (n int, err error)
//sys	mmap(addr uintptr, length uintptr, prot int, flag int, fd int, pos int64) (ret uintptr, err error)
//sys	munmap(addr uintptr, length uintptr) (err error)
//sys	readv(fd int, iovecs []Iovec) (n int, err error)
//sys	preadv(fd int, iovecs []Iovec, offset int64) (n int, err error)
//sys	writev(fd int, iovecs []Iovec) (n int, err error)
//sys	pwritev(fd int, iovecs []Iovec, offset int64) (n int, err error)
