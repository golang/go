// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linux system calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates system call stubs.
// Note that sometimes we use a lowercase //sys name and
// wrap it in our own nicer implementation.

package unix

import (
	"encoding/binary"
	"runtime"
	"syscall"
	"unsafe"
)

/*
 * Wrapped
 */

func Access(path string, mode uint32) (err error) {
	return Faccessat(AT_FDCWD, path, mode, 0)
}

func Chmod(path string, mode uint32) (err error) {
	return Fchmodat(AT_FDCWD, path, mode, 0)
}

func Chown(path string, uid int, gid int) (err error) {
	return Fchownat(AT_FDCWD, path, uid, gid, 0)
}

func Creat(path string, mode uint32) (fd int, err error) {
	return Open(path, O_CREAT|O_WRONLY|O_TRUNC, mode)
}

//sys	FanotifyInit(flags uint, event_f_flags uint) (fd int, err error)
//sys	fanotifyMark(fd int, flags uint, mask uint64, dirFd int, pathname *byte) (err error)

func FanotifyMark(fd int, flags uint, mask uint64, dirFd int, pathname string) (err error) {
	if pathname == "" {
		return fanotifyMark(fd, flags, mask, dirFd, nil)
	}
	p, err := BytePtrFromString(pathname)
	if err != nil {
		return err
	}
	return fanotifyMark(fd, flags, mask, dirFd, p)
}

//sys	fchmodat(dirfd int, path string, mode uint32) (err error)

func Fchmodat(dirfd int, path string, mode uint32, flags int) (err error) {
	// Linux fchmodat doesn't support the flags parameter. Mimick glibc's behavior
	// and check the flags. Otherwise the mode would be applied to the symlink
	// destination which is not what the user expects.
	if flags&^AT_SYMLINK_NOFOLLOW != 0 {
		return EINVAL
	} else if flags&AT_SYMLINK_NOFOLLOW != 0 {
		return EOPNOTSUPP
	}
	return fchmodat(dirfd, path, mode)
}

//sys	ioctl(fd int, req uint, arg uintptr) (err error)

// ioctl itself should not be exposed directly, but additional get/set
// functions for specific types are permissible.

// IoctlRetInt performs an ioctl operation specified by req on a device
// associated with opened file descriptor fd, and returns a non-negative
// integer that is returned by the ioctl syscall.
func IoctlRetInt(fd int, req uint) (int, error) {
	ret, _, err := Syscall(SYS_IOCTL, uintptr(fd), uintptr(req), 0)
	if err != 0 {
		return 0, err
	}
	return int(ret), nil
}

func IoctlSetRTCTime(fd int, value *RTCTime) error {
	err := ioctl(fd, RTC_SET_TIME, uintptr(unsafe.Pointer(value)))
	runtime.KeepAlive(value)
	return err
}

func IoctlSetRTCWkAlrm(fd int, value *RTCWkAlrm) error {
	err := ioctl(fd, RTC_WKALM_SET, uintptr(unsafe.Pointer(value)))
	runtime.KeepAlive(value)
	return err
}

func IoctlGetUint32(fd int, req uint) (uint32, error) {
	var value uint32
	err := ioctl(fd, req, uintptr(unsafe.Pointer(&value)))
	return value, err
}

func IoctlGetRTCTime(fd int) (*RTCTime, error) {
	var value RTCTime
	err := ioctl(fd, RTC_RD_TIME, uintptr(unsafe.Pointer(&value)))
	return &value, err
}

func IoctlGetRTCWkAlrm(fd int) (*RTCWkAlrm, error) {
	var value RTCWkAlrm
	err := ioctl(fd, RTC_WKALM_RD, uintptr(unsafe.Pointer(&value)))
	return &value, err
}

// IoctlFileClone performs an FICLONERANGE ioctl operation to clone the range of
// data conveyed in value to the file associated with the file descriptor
// destFd. See the ioctl_ficlonerange(2) man page for details.
func IoctlFileCloneRange(destFd int, value *FileCloneRange) error {
	err := ioctl(destFd, FICLONERANGE, uintptr(unsafe.Pointer(value)))
	runtime.KeepAlive(value)
	return err
}

// IoctlFileClone performs an FICLONE ioctl operation to clone the entire file
// associated with the file description srcFd to the file associated with the
// file descriptor destFd. See the ioctl_ficlone(2) man page for details.
func IoctlFileClone(destFd, srcFd int) error {
	return ioctl(destFd, FICLONE, uintptr(srcFd))
}

// IoctlFileClone performs an FIDEDUPERANGE ioctl operation to share the range of
// data conveyed in value with the file associated with the file descriptor
// destFd. See the ioctl_fideduperange(2) man page for details.
func IoctlFileDedupeRange(destFd int, value *FileDedupeRange) error {
	err := ioctl(destFd, FIDEDUPERANGE, uintptr(unsafe.Pointer(value)))
	runtime.KeepAlive(value)
	return err
}

//sys	Linkat(olddirfd int, oldpath string, newdirfd int, newpath string, flags int) (err error)

func Link(oldpath string, newpath string) (err error) {
	return Linkat(AT_FDCWD, oldpath, AT_FDCWD, newpath, 0)
}

func Mkdir(path string, mode uint32) (err error) {
	return Mkdirat(AT_FDCWD, path, mode)
}

func Mknod(path string, mode uint32, dev int) (err error) {
	return Mknodat(AT_FDCWD, path, mode, dev)
}

func Open(path string, mode int, perm uint32) (fd int, err error) {
	return openat(AT_FDCWD, path, mode|O_LARGEFILE, perm)
}

//sys	openat(dirfd int, path string, flags int, mode uint32) (fd int, err error)

func Openat(dirfd int, path string, flags int, mode uint32) (fd int, err error) {
	return openat(dirfd, path, flags|O_LARGEFILE, mode)
}

//sys	openat2(dirfd int, path string, open_how *OpenHow, size int) (fd int, err error)

func Openat2(dirfd int, path string, how *OpenHow) (fd int, err error) {
	return openat2(dirfd, path, how, SizeofOpenHow)
}

//sys	ppoll(fds *PollFd, nfds int, timeout *Timespec, sigmask *Sigset_t) (n int, err error)

func Ppoll(fds []PollFd, timeout *Timespec, sigmask *Sigset_t) (n int, err error) {
	if len(fds) == 0 {
		return ppoll(nil, 0, timeout, sigmask)
	}
	return ppoll(&fds[0], len(fds), timeout, sigmask)
}

//sys	Readlinkat(dirfd int, path string, buf []byte) (n int, err error)

func Readlink(path string, buf []byte) (n int, err error) {
	return Readlinkat(AT_FDCWD, path, buf)
}

func Rename(oldpath string, newpath string) (err error) {
	return Renameat(AT_FDCWD, oldpath, AT_FDCWD, newpath)
}

func Rmdir(path string) error {
	return Unlinkat(AT_FDCWD, path, AT_REMOVEDIR)
}

//sys	Symlinkat(oldpath string, newdirfd int, newpath string) (err error)

func Symlink(oldpath string, newpath string) (err error) {
	return Symlinkat(oldpath, AT_FDCWD, newpath)
}

func Unlink(path string) error {
	return Unlinkat(AT_FDCWD, path, 0)
}

//sys	Unlinkat(dirfd int, path string, flags int) (err error)

func Utimes(path string, tv []Timeval) error {
	if tv == nil {
		err := utimensat(AT_FDCWD, path, nil, 0)
		if err != ENOSYS {
			return err
		}
		return utimes(path, nil)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	var ts [2]Timespec
	ts[0] = NsecToTimespec(TimevalToNsec(tv[0]))
	ts[1] = NsecToTimespec(TimevalToNsec(tv[1]))
	err := utimensat(AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
	if err != ENOSYS {
		return err
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	utimensat(dirfd int, path string, times *[2]Timespec, flags int) (err error)

func UtimesNano(path string, ts []Timespec) error {
	if ts == nil {
		err := utimensat(AT_FDCWD, path, nil, 0)
		if err != ENOSYS {
			return err
		}
		return utimes(path, nil)
	}
	if len(ts) != 2 {
		return EINVAL
	}
	err := utimensat(AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
	if err != ENOSYS {
		return err
	}
	// If the utimensat syscall isn't available (utimensat was added to Linux
	// in 2.6.22, Released, 8 July 2007) then fall back to utimes
	var tv [2]Timeval
	for i := 0; i < 2; i++ {
		tv[i] = NsecToTimeval(TimespecToNsec(ts[i]))
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

func UtimesNanoAt(dirfd int, path string, ts []Timespec, flags int) error {
	if ts == nil {
		return utimensat(dirfd, path, nil, flags)
	}
	if len(ts) != 2 {
		return EINVAL
	}
	return utimensat(dirfd, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), flags)
}

func Futimesat(dirfd int, path string, tv []Timeval) error {
	if tv == nil {
		return futimesat(dirfd, path, nil)
	}
	if len(tv) != 2 {
		return EINVAL
	}
	return futimesat(dirfd, path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

func Futimes(fd int, tv []Timeval) (err error) {
	// Believe it or not, this is the best we can do on Linux
	// (and is what glibc does).
	return Utimes("/proc/self/fd/"+itoa(fd), tv)
}

const ImplementsGetwd = true

//sys	Getcwd(buf []byte) (n int, err error)

func Getwd() (wd string, err error) {
	var buf [PathMax]byte
	n, err := Getcwd(buf[0:])
	if err != nil {
		return "", err
	}
	// Getcwd returns the number of bytes written to buf, including the NUL.
	if n < 1 || n > len(buf) || buf[n-1] != 0 {
		return "", EINVAL
	}
	return string(buf[0 : n-1]), nil
}

func Getgroups() (gids []int, err error) {
	n, err := getgroups(0, nil)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}

	// Sanity check group count. Max is 1<<16 on Linux.
	if n < 0 || n > 1<<20 {
		return nil, EINVAL
	}

	a := make([]_Gid_t, n)
	n, err = getgroups(n, &a[0])
	if err != nil {
		return nil, err
	}
	gids = make([]int, n)
	for i, v := range a[0:n] {
		gids[i] = int(v)
	}
	return
}

func Setgroups(gids []int) (err error) {
	if len(gids) == 0 {
		return setgroups(0, nil)
	}

	a := make([]_Gid_t, len(gids))
	for i, v := range gids {
		a[i] = _Gid_t(v)
	}
	return setgroups(len(a), &a[0])
}

type WaitStatus uint32

// Wait status is 7 bits at bottom, either 0 (exited),
// 0x7F (stopped), or a signal number that caused an exit.
// The 0x80 bit is whether there was a core dump.
// An extra number (exit code, signal causing a stop)
// is in the high bits. At least that's the idea.
// There are various irregularities. For example, the
// "continued" status is 0xFFFF, distinguishing itself
// from stopped via the core dump bit.

const (
	mask    = 0x7F
	core    = 0x80
	exited  = 0x00
	stopped = 0x7F
	shift   = 8
)

func (w WaitStatus) Exited() bool { return w&mask == exited }

func (w WaitStatus) Signaled() bool { return w&mask != stopped && w&mask != exited }

func (w WaitStatus) Stopped() bool { return w&0xFF == stopped }

func (w WaitStatus) Continued() bool { return w == 0xFFFF }

func (w WaitStatus) CoreDump() bool { return w.Signaled() && w&core != 0 }

func (w WaitStatus) ExitStatus() int {
	if !w.Exited() {
		return -1
	}
	return int(w>>shift) & 0xFF
}

func (w WaitStatus) Signal() syscall.Signal {
	if !w.Signaled() {
		return -1
	}
	return syscall.Signal(w & mask)
}

func (w WaitStatus) StopSignal() syscall.Signal {
	if !w.Stopped() {
		return -1
	}
	return syscall.Signal(w>>shift) & 0xFF
}

func (w WaitStatus) TrapCause() int {
	if w.StopSignal() != SIGTRAP {
		return -1
	}
	return int(w>>shift) >> 8
}

//sys	wait4(pid int, wstatus *_C_int, options int, rusage *Rusage) (wpid int, err error)

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	var status _C_int
	wpid, err = wait4(pid, &status, options, rusage)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}

func Mkfifo(path string, mode uint32) error {
	return Mknod(path, mode|S_IFIFO, 0)
}

func Mkfifoat(dirfd int, path string, mode uint32) error {
	return Mknodat(dirfd, path, mode|S_IFIFO, 0)
}

func (sa *SockaddrInet4) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_INET
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrInet4, nil
}

func (sa *SockaddrInet6) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_INET6
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	sa.raw.Scope_id = sa.ZoneId
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrInet6, nil
}

func (sa *SockaddrUnix) sockaddr() (unsafe.Pointer, _Socklen, error) {
	name := sa.Name
	n := len(name)
	if n >= len(sa.raw.Path) {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_UNIX
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = int8(name[i])
	}
	// length is family (uint16), name, NUL.
	sl := _Socklen(2)
	if n > 0 {
		sl += _Socklen(n) + 1
	}
	if sa.raw.Path[0] == '@' {
		sa.raw.Path[0] = 0
		// Don't count trailing NUL for abstract address.
		sl--
	}

	return unsafe.Pointer(&sa.raw), sl, nil
}

// SockaddrLinklayer implements the Sockaddr interface for AF_PACKET type sockets.
type SockaddrLinklayer struct {
	Protocol uint16
	Ifindex  int
	Hatype   uint16
	Pkttype  uint8
	Halen    uint8
	Addr     [8]byte
	raw      RawSockaddrLinklayer
}

func (sa *SockaddrLinklayer) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Ifindex < 0 || sa.Ifindex > 0x7fffffff {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_PACKET
	sa.raw.Protocol = sa.Protocol
	sa.raw.Ifindex = int32(sa.Ifindex)
	sa.raw.Hatype = sa.Hatype
	sa.raw.Pkttype = sa.Pkttype
	sa.raw.Halen = sa.Halen
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrLinklayer, nil
}

// SockaddrNetlink implements the Sockaddr interface for AF_NETLINK type sockets.
type SockaddrNetlink struct {
	Family uint16
	Pad    uint16
	Pid    uint32
	Groups uint32
	raw    RawSockaddrNetlink
}

func (sa *SockaddrNetlink) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_NETLINK
	sa.raw.Pad = sa.Pad
	sa.raw.Pid = sa.Pid
	sa.raw.Groups = sa.Groups
	return unsafe.Pointer(&sa.raw), SizeofSockaddrNetlink, nil
}

// SockaddrHCI implements the Sockaddr interface for AF_BLUETOOTH type sockets
// using the HCI protocol.
type SockaddrHCI struct {
	Dev     uint16
	Channel uint16
	raw     RawSockaddrHCI
}

func (sa *SockaddrHCI) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_BLUETOOTH
	sa.raw.Dev = sa.Dev
	sa.raw.Channel = sa.Channel
	return unsafe.Pointer(&sa.raw), SizeofSockaddrHCI, nil
}

// SockaddrL2 implements the Sockaddr interface for AF_BLUETOOTH type sockets
// using the L2CAP protocol.
type SockaddrL2 struct {
	PSM      uint16
	CID      uint16
	Addr     [6]uint8
	AddrType uint8
	raw      RawSockaddrL2
}

func (sa *SockaddrL2) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_BLUETOOTH
	psm := (*[2]byte)(unsafe.Pointer(&sa.raw.Psm))
	psm[0] = byte(sa.PSM)
	psm[1] = byte(sa.PSM >> 8)
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Bdaddr[i] = sa.Addr[len(sa.Addr)-1-i]
	}
	cid := (*[2]byte)(unsafe.Pointer(&sa.raw.Cid))
	cid[0] = byte(sa.CID)
	cid[1] = byte(sa.CID >> 8)
	sa.raw.Bdaddr_type = sa.AddrType
	return unsafe.Pointer(&sa.raw), SizeofSockaddrL2, nil
}

// SockaddrRFCOMM implements the Sockaddr interface for AF_BLUETOOTH type sockets
// using the RFCOMM protocol.
//
// Server example:
//
//      fd, _ := Socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM)
//      _ = unix.Bind(fd, &unix.SockaddrRFCOMM{
//      	Channel: 1,
//      	Addr:    [6]uint8{0, 0, 0, 0, 0, 0}, // BDADDR_ANY or 00:00:00:00:00:00
//      })
//      _ = Listen(fd, 1)
//      nfd, sa, _ := Accept(fd)
//      fmt.Printf("conn addr=%v fd=%d", sa.(*unix.SockaddrRFCOMM).Addr, nfd)
//      Read(nfd, buf)
//
// Client example:
//
//      fd, _ := Socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM)
//      _ = Connect(fd, &SockaddrRFCOMM{
//      	Channel: 1,
//      	Addr:    [6]byte{0x11, 0x22, 0x33, 0xaa, 0xbb, 0xcc}, // CC:BB:AA:33:22:11
//      })
//      Write(fd, []byte(`hello`))
type SockaddrRFCOMM struct {
	// Addr represents a bluetooth address, byte ordering is little-endian.
	Addr [6]uint8

	// Channel is a designated bluetooth channel, only 1-30 are available for use.
	// Since Linux 2.6.7 and further zero value is the first available channel.
	Channel uint8

	raw RawSockaddrRFCOMM
}

func (sa *SockaddrRFCOMM) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_BLUETOOTH
	sa.raw.Channel = sa.Channel
	sa.raw.Bdaddr = sa.Addr
	return unsafe.Pointer(&sa.raw), SizeofSockaddrRFCOMM, nil
}

// SockaddrCAN implements the Sockaddr interface for AF_CAN type sockets.
// The RxID and TxID fields are used for transport protocol addressing in
// (CAN_TP16, CAN_TP20, CAN_MCNET, and CAN_ISOTP), they can be left with
// zero values for CAN_RAW and CAN_BCM sockets as they have no meaning.
//
// The SockaddrCAN struct must be bound to the socket file descriptor
// using Bind before the CAN socket can be used.
//
//      // Read one raw CAN frame
//      fd, _ := Socket(AF_CAN, SOCK_RAW, CAN_RAW)
//      addr := &SockaddrCAN{Ifindex: index}
//      Bind(fd, addr)
//      frame := make([]byte, 16)
//      Read(fd, frame)
//
// The full SocketCAN documentation can be found in the linux kernel
// archives at: https://www.kernel.org/doc/Documentation/networking/can.txt
type SockaddrCAN struct {
	Ifindex int
	RxID    uint32
	TxID    uint32
	raw     RawSockaddrCAN
}

func (sa *SockaddrCAN) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Ifindex < 0 || sa.Ifindex > 0x7fffffff {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_CAN
	sa.raw.Ifindex = int32(sa.Ifindex)
	rx := (*[4]byte)(unsafe.Pointer(&sa.RxID))
	for i := 0; i < 4; i++ {
		sa.raw.Addr[i] = rx[i]
	}
	tx := (*[4]byte)(unsafe.Pointer(&sa.TxID))
	for i := 0; i < 4; i++ {
		sa.raw.Addr[i+4] = tx[i]
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrCAN, nil
}

// SockaddrALG implements the Sockaddr interface for AF_ALG type sockets.
// SockaddrALG enables userspace access to the Linux kernel's cryptography
// subsystem. The Type and Name fields specify which type of hash or cipher
// should be used with a given socket.
//
// To create a file descriptor that provides access to a hash or cipher, both
// Bind and Accept must be used. Once the setup process is complete, input
// data can be written to the socket, processed by the kernel, and then read
// back as hash output or ciphertext.
//
// Here is an example of using an AF_ALG socket with SHA1 hashing.
// The initial socket setup process is as follows:
//
//      // Open a socket to perform SHA1 hashing.
//      fd, _ := unix.Socket(unix.AF_ALG, unix.SOCK_SEQPACKET, 0)
//      addr := &unix.SockaddrALG{Type: "hash", Name: "sha1"}
//      unix.Bind(fd, addr)
//      // Note: unix.Accept does not work at this time; must invoke accept()
//      // manually using unix.Syscall.
//      hashfd, _, _ := unix.Syscall(unix.SYS_ACCEPT, uintptr(fd), 0, 0)
//
// Once a file descriptor has been returned from Accept, it may be used to
// perform SHA1 hashing. The descriptor is not safe for concurrent use, but
// may be re-used repeatedly with subsequent Write and Read operations.
//
// When hashing a small byte slice or string, a single Write and Read may
// be used:
//
//      // Assume hashfd is already configured using the setup process.
//      hash := os.NewFile(hashfd, "sha1")
//      // Hash an input string and read the results. Each Write discards
//      // previous hash state. Read always reads the current state.
//      b := make([]byte, 20)
//      for i := 0; i < 2; i++ {
//          io.WriteString(hash, "Hello, world.")
//          hash.Read(b)
//          fmt.Println(hex.EncodeToString(b))
//      }
//      // Output:
//      // 2ae01472317d1935a84797ec1983ae243fc6aa28
//      // 2ae01472317d1935a84797ec1983ae243fc6aa28
//
// For hashing larger byte slices, or byte streams such as those read from
// a file or socket, use Sendto with MSG_MORE to instruct the kernel to update
// the hash digest instead of creating a new one for a given chunk and finalizing it.
//
//      // Assume hashfd and addr are already configured using the setup process.
//      hash := os.NewFile(hashfd, "sha1")
//      // Hash the contents of a file.
//      f, _ := os.Open("/tmp/linux-4.10-rc7.tar.xz")
//      b := make([]byte, 4096)
//      for {
//          n, err := f.Read(b)
//          if err == io.EOF {
//              break
//          }
//          unix.Sendto(hashfd, b[:n], unix.MSG_MORE, addr)
//      }
//      hash.Read(b)
//      fmt.Println(hex.EncodeToString(b))
//      // Output: 85cdcad0c06eef66f805ecce353bec9accbeecc5
//
// For more information, see: http://www.chronox.de/crypto-API/crypto/userspace-if.html.
type SockaddrALG struct {
	Type    string
	Name    string
	Feature uint32
	Mask    uint32
	raw     RawSockaddrALG
}

func (sa *SockaddrALG) sockaddr() (unsafe.Pointer, _Socklen, error) {
	// Leave room for NUL byte terminator.
	if len(sa.Type) > 13 {
		return nil, 0, EINVAL
	}
	if len(sa.Name) > 63 {
		return nil, 0, EINVAL
	}

	sa.raw.Family = AF_ALG
	sa.raw.Feat = sa.Feature
	sa.raw.Mask = sa.Mask

	typ, err := ByteSliceFromString(sa.Type)
	if err != nil {
		return nil, 0, err
	}
	name, err := ByteSliceFromString(sa.Name)
	if err != nil {
		return nil, 0, err
	}

	copy(sa.raw.Type[:], typ)
	copy(sa.raw.Name[:], name)

	return unsafe.Pointer(&sa.raw), SizeofSockaddrALG, nil
}

// SockaddrVM implements the Sockaddr interface for AF_VSOCK type sockets.
// SockaddrVM provides access to Linux VM sockets: a mechanism that enables
// bidirectional communication between a hypervisor and its guest virtual
// machines.
type SockaddrVM struct {
	// CID and Port specify a context ID and port address for a VM socket.
	// Guests have a unique CID, and hosts may have a well-known CID of:
	//  - VMADDR_CID_HYPERVISOR: refers to the hypervisor process.
	//  - VMADDR_CID_HOST: refers to other processes on the host.
	CID  uint32
	Port uint32
	raw  RawSockaddrVM
}

func (sa *SockaddrVM) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_VSOCK
	sa.raw.Port = sa.Port
	sa.raw.Cid = sa.CID

	return unsafe.Pointer(&sa.raw), SizeofSockaddrVM, nil
}

type SockaddrXDP struct {
	Flags        uint16
	Ifindex      uint32
	QueueID      uint32
	SharedUmemFD uint32
	raw          RawSockaddrXDP
}

func (sa *SockaddrXDP) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_XDP
	sa.raw.Flags = sa.Flags
	sa.raw.Ifindex = sa.Ifindex
	sa.raw.Queue_id = sa.QueueID
	sa.raw.Shared_umem_fd = sa.SharedUmemFD

	return unsafe.Pointer(&sa.raw), SizeofSockaddrXDP, nil
}

// This constant mirrors the #define of PX_PROTO_OE in
// linux/if_pppox.h. We're defining this by hand here instead of
// autogenerating through mkerrors.sh because including
// linux/if_pppox.h causes some declaration conflicts with other
// includes (linux/if_pppox.h includes linux/in.h, which conflicts
// with netinet/in.h). Given that we only need a single zero constant
// out of that file, it's cleaner to just define it by hand here.
const px_proto_oe = 0

type SockaddrPPPoE struct {
	SID    uint16
	Remote []byte
	Dev    string
	raw    RawSockaddrPPPoX
}

func (sa *SockaddrPPPoE) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if len(sa.Remote) != 6 {
		return nil, 0, EINVAL
	}
	if len(sa.Dev) > IFNAMSIZ-1 {
		return nil, 0, EINVAL
	}

	*(*uint16)(unsafe.Pointer(&sa.raw[0])) = AF_PPPOX
	// This next field is in host-endian byte order. We can't use the
	// same unsafe pointer cast as above, because this value is not
	// 32-bit aligned and some architectures don't allow unaligned
	// access.
	//
	// However, the value of px_proto_oe is 0, so we can use
	// encoding/binary helpers to write the bytes without worrying
	// about the ordering.
	binary.BigEndian.PutUint32(sa.raw[2:6], px_proto_oe)
	// This field is deliberately big-endian, unlike the previous
	// one. The kernel expects SID to be in network byte order.
	binary.BigEndian.PutUint16(sa.raw[6:8], sa.SID)
	copy(sa.raw[8:14], sa.Remote)
	for i := 14; i < 14+IFNAMSIZ; i++ {
		sa.raw[i] = 0
	}
	copy(sa.raw[14:], sa.Dev)
	return unsafe.Pointer(&sa.raw), SizeofSockaddrPPPoX, nil
}

// SockaddrTIPC implements the Sockaddr interface for AF_TIPC type sockets.
// For more information on TIPC, see: http://tipc.sourceforge.net/.
type SockaddrTIPC struct {
	// Scope is the publication scopes when binding service/service range.
	// Should be set to TIPC_CLUSTER_SCOPE or TIPC_NODE_SCOPE.
	Scope int

	// Addr is the type of address used to manipulate a socket. Addr must be
	// one of:
	//  - *TIPCSocketAddr: "id" variant in the C addr union
	//  - *TIPCServiceRange: "nameseq" variant in the C addr union
	//  - *TIPCServiceName: "name" variant in the C addr union
	//
	// If nil, EINVAL will be returned when the structure is used.
	Addr TIPCAddr

	raw RawSockaddrTIPC
}

// TIPCAddr is implemented by types that can be used as an address for
// SockaddrTIPC. It is only implemented by *TIPCSocketAddr, *TIPCServiceRange,
// and *TIPCServiceName.
type TIPCAddr interface {
	tipcAddrtype() uint8
	tipcAddr() [12]byte
}

func (sa *TIPCSocketAddr) tipcAddr() [12]byte {
	var out [12]byte
	copy(out[:], (*(*[unsafe.Sizeof(TIPCSocketAddr{})]byte)(unsafe.Pointer(sa)))[:])
	return out
}

func (sa *TIPCSocketAddr) tipcAddrtype() uint8 { return TIPC_SOCKET_ADDR }

func (sa *TIPCServiceRange) tipcAddr() [12]byte {
	var out [12]byte
	copy(out[:], (*(*[unsafe.Sizeof(TIPCServiceRange{})]byte)(unsafe.Pointer(sa)))[:])
	return out
}

func (sa *TIPCServiceRange) tipcAddrtype() uint8 { return TIPC_SERVICE_RANGE }

func (sa *TIPCServiceName) tipcAddr() [12]byte {
	var out [12]byte
	copy(out[:], (*(*[unsafe.Sizeof(TIPCServiceName{})]byte)(unsafe.Pointer(sa)))[:])
	return out
}

func (sa *TIPCServiceName) tipcAddrtype() uint8 { return TIPC_SERVICE_ADDR }

func (sa *SockaddrTIPC) sockaddr() (unsafe.Pointer, _Socklen, error) {
	if sa.Addr == nil {
		return nil, 0, EINVAL
	}

	sa.raw.Family = AF_TIPC
	sa.raw.Scope = int8(sa.Scope)
	sa.raw.Addrtype = sa.Addr.tipcAddrtype()
	sa.raw.Addr = sa.Addr.tipcAddr()

	return unsafe.Pointer(&sa.raw), SizeofSockaddrTIPC, nil
}

// SockaddrL2TPIP implements the Sockaddr interface for IPPROTO_L2TP/AF_INET sockets.
type SockaddrL2TPIP struct {
	Addr   [4]byte
	ConnId uint32
	raw    RawSockaddrL2TPIP
}

func (sa *SockaddrL2TPIP) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_INET
	sa.raw.Conn_id = sa.ConnId
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrL2TPIP, nil
}

// SockaddrL2TPIP6 implements the Sockaddr interface for IPPROTO_L2TP/AF_INET6 sockets.
type SockaddrL2TPIP6 struct {
	Addr   [16]byte
	ZoneId uint32
	ConnId uint32
	raw    RawSockaddrL2TPIP6
}

func (sa *SockaddrL2TPIP6) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_INET6
	sa.raw.Conn_id = sa.ConnId
	sa.raw.Scope_id = sa.ZoneId
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrL2TPIP6, nil
}

// SockaddrIUCV implements the Sockaddr interface for AF_IUCV sockets.
type SockaddrIUCV struct {
	UserID string
	Name   string
	raw    RawSockaddrIUCV
}

func (sa *SockaddrIUCV) sockaddr() (unsafe.Pointer, _Socklen, error) {
	sa.raw.Family = AF_IUCV
	// These are EBCDIC encoded by the kernel, but we still need to pad them
	// with blanks. Initializing with blanks allows the caller to feed in either
	// a padded or an unpadded string.
	for i := 0; i < 8; i++ {
		sa.raw.Nodeid[i] = ' '
		sa.raw.User_id[i] = ' '
		sa.raw.Name[i] = ' '
	}
	if len(sa.UserID) > 8 || len(sa.Name) > 8 {
		return nil, 0, EINVAL
	}
	for i, b := range []byte(sa.UserID[:]) {
		sa.raw.User_id[i] = int8(b)
	}
	for i, b := range []byte(sa.Name[:]) {
		sa.raw.Name[i] = int8(b)
	}
	return unsafe.Pointer(&sa.raw), SizeofSockaddrIUCV, nil
}

func anyToSockaddr(fd int, rsa *RawSockaddrAny) (Sockaddr, error) {
	switch rsa.Addr.Family {
	case AF_NETLINK:
		pp := (*RawSockaddrNetlink)(unsafe.Pointer(rsa))
		sa := new(SockaddrNetlink)
		sa.Family = pp.Family
		sa.Pad = pp.Pad
		sa.Pid = pp.Pid
		sa.Groups = pp.Groups
		return sa, nil

	case AF_PACKET:
		pp := (*RawSockaddrLinklayer)(unsafe.Pointer(rsa))
		sa := new(SockaddrLinklayer)
		sa.Protocol = pp.Protocol
		sa.Ifindex = int(pp.Ifindex)
		sa.Hatype = pp.Hatype
		sa.Pkttype = pp.Pkttype
		sa.Halen = pp.Halen
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, nil

	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		sa := new(SockaddrUnix)
		if pp.Path[0] == 0 {
			// "Abstract" Unix domain socket.
			// Rewrite leading NUL as @ for textual display.
			// (This is the standard convention.)
			// Not friendly to overwrite in place,
			// but the callers below don't care.
			pp.Path[0] = '@'
		}

		// Assume path ends at NUL.
		// This is not technically the Linux semantics for
		// abstract Unix domain sockets--they are supposed
		// to be uninterpreted fixed-size binary blobs--but
		// everyone uses this convention.
		n := 0
		for n < len(pp.Path) && pp.Path[n] != 0 {
			n++
		}
		bytes := (*[len(pp.Path)]byte)(unsafe.Pointer(&pp.Path[0]))[0:n]
		sa.Name = string(bytes)
		return sa, nil

	case AF_INET:
		proto, err := GetsockoptInt(fd, SOL_SOCKET, SO_PROTOCOL)
		if err != nil {
			return nil, err
		}

		switch proto {
		case IPPROTO_L2TP:
			pp := (*RawSockaddrL2TPIP)(unsafe.Pointer(rsa))
			sa := new(SockaddrL2TPIP)
			sa.ConnId = pp.Conn_id
			for i := 0; i < len(sa.Addr); i++ {
				sa.Addr[i] = pp.Addr[i]
			}
			return sa, nil
		default:
			pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa))
			sa := new(SockaddrInet4)
			p := (*[2]byte)(unsafe.Pointer(&pp.Port))
			sa.Port = int(p[0])<<8 + int(p[1])
			for i := 0; i < len(sa.Addr); i++ {
				sa.Addr[i] = pp.Addr[i]
			}
			return sa, nil
		}

	case AF_INET6:
		proto, err := GetsockoptInt(fd, SOL_SOCKET, SO_PROTOCOL)
		if err != nil {
			return nil, err
		}

		switch proto {
		case IPPROTO_L2TP:
			pp := (*RawSockaddrL2TPIP6)(unsafe.Pointer(rsa))
			sa := new(SockaddrL2TPIP6)
			sa.ConnId = pp.Conn_id
			sa.ZoneId = pp.Scope_id
			for i := 0; i < len(sa.Addr); i++ {
				sa.Addr[i] = pp.Addr[i]
			}
			return sa, nil
		default:
			pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa))
			sa := new(SockaddrInet6)
			p := (*[2]byte)(unsafe.Pointer(&pp.Port))
			sa.Port = int(p[0])<<8 + int(p[1])
			sa.ZoneId = pp.Scope_id
			for i := 0; i < len(sa.Addr); i++ {
				sa.Addr[i] = pp.Addr[i]
			}
			return sa, nil
		}

	case AF_VSOCK:
		pp := (*RawSockaddrVM)(unsafe.Pointer(rsa))
		sa := &SockaddrVM{
			CID:  pp.Cid,
			Port: pp.Port,
		}
		return sa, nil
	case AF_BLUETOOTH:
		proto, err := GetsockoptInt(fd, SOL_SOCKET, SO_PROTOCOL)
		if err != nil {
			return nil, err
		}
		// only BTPROTO_L2CAP and BTPROTO_RFCOMM can accept connections
		switch proto {
		case BTPROTO_L2CAP:
			pp := (*RawSockaddrL2)(unsafe.Pointer(rsa))
			sa := &SockaddrL2{
				PSM:      pp.Psm,
				CID:      pp.Cid,
				Addr:     pp.Bdaddr,
				AddrType: pp.Bdaddr_type,
			}
			return sa, nil
		case BTPROTO_RFCOMM:
			pp := (*RawSockaddrRFCOMM)(unsafe.Pointer(rsa))
			sa := &SockaddrRFCOMM{
				Channel: pp.Channel,
				Addr:    pp.Bdaddr,
			}
			return sa, nil
		}
	case AF_XDP:
		pp := (*RawSockaddrXDP)(unsafe.Pointer(rsa))
		sa := &SockaddrXDP{
			Flags:        pp.Flags,
			Ifindex:      pp.Ifindex,
			QueueID:      pp.Queue_id,
			SharedUmemFD: pp.Shared_umem_fd,
		}
		return sa, nil
	case AF_PPPOX:
		pp := (*RawSockaddrPPPoX)(unsafe.Pointer(rsa))
		if binary.BigEndian.Uint32(pp[2:6]) != px_proto_oe {
			return nil, EINVAL
		}
		sa := &SockaddrPPPoE{
			SID:    binary.BigEndian.Uint16(pp[6:8]),
			Remote: pp[8:14],
		}
		for i := 14; i < 14+IFNAMSIZ; i++ {
			if pp[i] == 0 {
				sa.Dev = string(pp[14:i])
				break
			}
		}
		return sa, nil
	case AF_TIPC:
		pp := (*RawSockaddrTIPC)(unsafe.Pointer(rsa))

		sa := &SockaddrTIPC{
			Scope: int(pp.Scope),
		}

		// Determine which union variant is present in pp.Addr by checking
		// pp.Addrtype.
		switch pp.Addrtype {
		case TIPC_SERVICE_RANGE:
			sa.Addr = (*TIPCServiceRange)(unsafe.Pointer(&pp.Addr))
		case TIPC_SERVICE_ADDR:
			sa.Addr = (*TIPCServiceName)(unsafe.Pointer(&pp.Addr))
		case TIPC_SOCKET_ADDR:
			sa.Addr = (*TIPCSocketAddr)(unsafe.Pointer(&pp.Addr))
		default:
			return nil, EINVAL
		}

		return sa, nil
	case AF_IUCV:
		pp := (*RawSockaddrIUCV)(unsafe.Pointer(rsa))

		var user [8]byte
		var name [8]byte

		for i := 0; i < 8; i++ {
			user[i] = byte(pp.User_id[i])
			name[i] = byte(pp.Name[i])
		}

		sa := &SockaddrIUCV{
			UserID: string(user[:]),
			Name:   string(name[:]),
		}
		return sa, nil

	case AF_CAN:
		pp := (*RawSockaddrCAN)(unsafe.Pointer(rsa))
		sa := &SockaddrCAN{
			Ifindex: int(pp.Ifindex),
		}
		rx := (*[4]byte)(unsafe.Pointer(&sa.RxID))
		for i := 0; i < 4; i++ {
			rx[i] = pp.Addr[i]
		}
		tx := (*[4]byte)(unsafe.Pointer(&sa.TxID))
		for i := 0; i < 4; i++ {
			tx[i] = pp.Addr[i+4]
		}
		return sa, nil

	}
	return nil, EAFNOSUPPORT
}

func Accept(fd int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept(fd, &rsa, &len)
	if err != nil {
		return
	}
	sa, err = anyToSockaddr(fd, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func Accept4(fd int, flags int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept4(fd, &rsa, &len, flags)
	if err != nil {
		return
	}
	if len > SizeofSockaddrAny {
		panic("RawSockaddrAny too small")
	}
	sa, err = anyToSockaddr(fd, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

func Getsockname(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	if err = getsockname(fd, &rsa, &len); err != nil {
		return
	}
	return anyToSockaddr(fd, &rsa)
}

func GetsockoptIPMreqn(fd, level, opt int) (*IPMreqn, error) {
	var value IPMreqn
	vallen := _Socklen(SizeofIPMreqn)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptUcred(fd, level, opt int) (*Ucred, error) {
	var value Ucred
	vallen := _Socklen(SizeofUcred)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptTCPInfo(fd, level, opt int) (*TCPInfo, error) {
	var value TCPInfo
	vallen := _Socklen(SizeofTCPInfo)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

// GetsockoptString returns the string value of the socket option opt for the
// socket associated with fd at the given socket level.
func GetsockoptString(fd, level, opt int) (string, error) {
	buf := make([]byte, 256)
	vallen := _Socklen(len(buf))
	err := getsockopt(fd, level, opt, unsafe.Pointer(&buf[0]), &vallen)
	if err != nil {
		if err == ERANGE {
			buf = make([]byte, vallen)
			err = getsockopt(fd, level, opt, unsafe.Pointer(&buf[0]), &vallen)
		}
		if err != nil {
			return "", err
		}
	}
	return string(buf[:vallen-1]), nil
}

func GetsockoptTpacketStats(fd, level, opt int) (*TpacketStats, error) {
	var value TpacketStats
	vallen := _Socklen(SizeofTpacketStats)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptTpacketStatsV3(fd, level, opt int) (*TpacketStatsV3, error) {
	var value TpacketStatsV3
	vallen := _Socklen(SizeofTpacketStatsV3)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func SetsockoptIPMreqn(fd, level, opt int, mreq *IPMreqn) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), unsafe.Sizeof(*mreq))
}

func SetsockoptPacketMreq(fd, level, opt int, mreq *PacketMreq) error {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), unsafe.Sizeof(*mreq))
}

// SetsockoptSockFprog attaches a classic BPF or an extended BPF program to a
// socket to filter incoming packets.  See 'man 7 socket' for usage information.
func SetsockoptSockFprog(fd, level, opt int, fprog *SockFprog) error {
	return setsockopt(fd, level, opt, unsafe.Pointer(fprog), unsafe.Sizeof(*fprog))
}

func SetsockoptCanRawFilter(fd, level, opt int, filter []CanFilter) error {
	var p unsafe.Pointer
	if len(filter) > 0 {
		p = unsafe.Pointer(&filter[0])
	}
	return setsockopt(fd, level, opt, p, uintptr(len(filter)*SizeofCanFilter))
}

func SetsockoptTpacketReq(fd, level, opt int, tp *TpacketReq) error {
	return setsockopt(fd, level, opt, unsafe.Pointer(tp), unsafe.Sizeof(*tp))
}

func SetsockoptTpacketReq3(fd, level, opt int, tp *TpacketReq3) error {
	return setsockopt(fd, level, opt, unsafe.Pointer(tp), unsafe.Sizeof(*tp))
}

// Keyctl Commands (http://man7.org/linux/man-pages/man2/keyctl.2.html)

// KeyctlInt calls keyctl commands in which each argument is an int.
// These commands are KEYCTL_REVOKE, KEYCTL_CHOWN, KEYCTL_CLEAR, KEYCTL_LINK,
// KEYCTL_UNLINK, KEYCTL_NEGATE, KEYCTL_SET_REQKEY_KEYRING, KEYCTL_SET_TIMEOUT,
// KEYCTL_ASSUME_AUTHORITY, KEYCTL_SESSION_TO_PARENT, KEYCTL_REJECT,
// KEYCTL_INVALIDATE, and KEYCTL_GET_PERSISTENT.
//sys	KeyctlInt(cmd int, arg2 int, arg3 int, arg4 int, arg5 int) (ret int, err error) = SYS_KEYCTL

// KeyctlBuffer calls keyctl commands in which the third and fourth
// arguments are a buffer and its length, respectively.
// These commands are KEYCTL_UPDATE, KEYCTL_READ, and KEYCTL_INSTANTIATE.
//sys	KeyctlBuffer(cmd int, arg2 int, buf []byte, arg5 int) (ret int, err error) = SYS_KEYCTL

// KeyctlString calls keyctl commands which return a string.
// These commands are KEYCTL_DESCRIBE and KEYCTL_GET_SECURITY.
func KeyctlString(cmd int, id int) (string, error) {
	// We must loop as the string data may change in between the syscalls.
	// We could allocate a large buffer here to reduce the chance that the
	// syscall needs to be called twice; however, this is unnecessary as
	// the performance loss is negligible.
	var buffer []byte
	for {
		// Try to fill the buffer with data
		length, err := KeyctlBuffer(cmd, id, buffer, 0)
		if err != nil {
			return "", err
		}

		// Check if the data was written
		if length <= len(buffer) {
			// Exclude the null terminator
			return string(buffer[:length-1]), nil
		}

		// Make a bigger buffer if needed
		buffer = make([]byte, length)
	}
}

// Keyctl commands with special signatures.

// KeyctlGetKeyringID implements the KEYCTL_GET_KEYRING_ID command.
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_get_keyring_ID.3.html
func KeyctlGetKeyringID(id int, create bool) (ringid int, err error) {
	createInt := 0
	if create {
		createInt = 1
	}
	return KeyctlInt(KEYCTL_GET_KEYRING_ID, id, createInt, 0, 0)
}

// KeyctlSetperm implements the KEYCTL_SETPERM command. The perm value is the
// key handle permission mask as described in the "keyctl setperm" section of
// http://man7.org/linux/man-pages/man1/keyctl.1.html.
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_setperm.3.html
func KeyctlSetperm(id int, perm uint32) error {
	_, err := KeyctlInt(KEYCTL_SETPERM, id, int(perm), 0, 0)
	return err
}

//sys	keyctlJoin(cmd int, arg2 string) (ret int, err error) = SYS_KEYCTL

// KeyctlJoinSessionKeyring implements the KEYCTL_JOIN_SESSION_KEYRING command.
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_join_session_keyring.3.html
func KeyctlJoinSessionKeyring(name string) (ringid int, err error) {
	return keyctlJoin(KEYCTL_JOIN_SESSION_KEYRING, name)
}

//sys	keyctlSearch(cmd int, arg2 int, arg3 string, arg4 string, arg5 int) (ret int, err error) = SYS_KEYCTL

// KeyctlSearch implements the KEYCTL_SEARCH command.
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_search.3.html
func KeyctlSearch(ringid int, keyType, description string, destRingid int) (id int, err error) {
	return keyctlSearch(KEYCTL_SEARCH, ringid, keyType, description, destRingid)
}

//sys	keyctlIOV(cmd int, arg2 int, payload []Iovec, arg5 int) (err error) = SYS_KEYCTL

// KeyctlInstantiateIOV implements the KEYCTL_INSTANTIATE_IOV command. This
// command is similar to KEYCTL_INSTANTIATE, except that the payload is a slice
// of Iovec (each of which represents a buffer) instead of a single buffer.
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_instantiate_iov.3.html
func KeyctlInstantiateIOV(id int, payload []Iovec, ringid int) error {
	return keyctlIOV(KEYCTL_INSTANTIATE_IOV, id, payload, ringid)
}

//sys	keyctlDH(cmd int, arg2 *KeyctlDHParams, buf []byte) (ret int, err error) = SYS_KEYCTL

// KeyctlDHCompute implements the KEYCTL_DH_COMPUTE command. This command
// computes a Diffie-Hellman shared secret based on the provide params. The
// secret is written to the provided buffer and the returned size is the number
// of bytes written (returning an error if there is insufficient space in the
// buffer). If a nil buffer is passed in, this function returns the minimum
// buffer length needed to store the appropriate data. Note that this differs
// from KEYCTL_READ's behavior which always returns the requested payload size.
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_dh_compute.3.html
func KeyctlDHCompute(params *KeyctlDHParams, buffer []byte) (size int, err error) {
	return keyctlDH(KEYCTL_DH_COMPUTE, params, buffer)
}

// KeyctlRestrictKeyring implements the KEYCTL_RESTRICT_KEYRING command. This
// command limits the set of keys that can be linked to the keyring, regardless
// of keyring permissions. The command requires the "setattr" permission.
//
// When called with an empty keyType the command locks the keyring, preventing
// any further keys from being linked to the keyring.
//
// The "asymmetric" keyType defines restrictions requiring key payloads to be
// DER encoded X.509 certificates signed by keys in another keyring. Restrictions
// for "asymmetric" include "builtin_trusted", "builtin_and_secondary_trusted",
// "key_or_keyring:<key>", and "key_or_keyring:<key>:chain".
//
// As of Linux 4.12, only the "asymmetric" keyType defines type-specific
// restrictions.
//
// See the full documentation at:
// http://man7.org/linux/man-pages/man3/keyctl_restrict_keyring.3.html
// http://man7.org/linux/man-pages/man2/keyctl.2.html
func KeyctlRestrictKeyring(ringid int, keyType string, restriction string) error {
	if keyType == "" {
		return keyctlRestrictKeyring(KEYCTL_RESTRICT_KEYRING, ringid)
	}
	return keyctlRestrictKeyringByType(KEYCTL_RESTRICT_KEYRING, ringid, keyType, restriction)
}

//sys keyctlRestrictKeyringByType(cmd int, arg2 int, keyType string, restriction string) (err error) = SYS_KEYCTL
//sys keyctlRestrictKeyring(cmd int, arg2 int) (err error) = SYS_KEYCTL

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, err error) {
	var msg Msghdr
	var rsa RawSockaddrAny
	msg.Name = (*byte)(unsafe.Pointer(&rsa))
	msg.Namelen = uint32(SizeofSockaddrAny)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = &p[0]
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		if len(p) == 0 {
			var sockType int
			sockType, err = GetsockoptInt(fd, SOL_SOCKET, SO_TYPE)
			if err != nil {
				return
			}
			// receive at least one normal byte
			if sockType != SOCK_DGRAM {
				iov.Base = &dummy
				iov.SetLen(1)
			}
		}
		msg.Control = &oob[0]
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = recvmsg(fd, &msg, flags); err != nil {
		return
	}
	oobn = int(msg.Controllen)
	recvflags = int(msg.Flags)
	// source address is only specified if the socket is unconnected
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(fd, &rsa)
	}
	return
}

func Sendmsg(fd int, p, oob []byte, to Sockaddr, flags int) (err error) {
	_, err = SendmsgN(fd, p, oob, to, flags)
	return
}

func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	var ptr unsafe.Pointer
	var salen _Socklen
	if to != nil {
		var err error
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return 0, err
		}
	}
	var msg Msghdr
	msg.Name = (*byte)(ptr)
	msg.Namelen = uint32(salen)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = &p[0]
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		if len(p) == 0 {
			var sockType int
			sockType, err = GetsockoptInt(fd, SOL_SOCKET, SO_TYPE)
			if err != nil {
				return 0, err
			}
			// send at least one normal byte
			if sockType != SOCK_DGRAM {
				iov.Base = &dummy
				iov.SetLen(1)
			}
		}
		msg.Control = &oob[0]
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = sendmsg(fd, &msg, flags); err != nil {
		return 0, err
	}
	if len(oob) > 0 && len(p) == 0 {
		n = 0
	}
	return n, nil
}

// BindToDevice binds the socket associated with fd to device.
func BindToDevice(fd int, device string) (err error) {
	return SetsockoptString(fd, SOL_SOCKET, SO_BINDTODEVICE, device)
}

//sys	ptrace(request int, pid int, addr uintptr, data uintptr) (err error)

func ptracePeek(req int, pid int, addr uintptr, out []byte) (count int, err error) {
	// The peek requests are machine-size oriented, so we wrap it
	// to retrieve arbitrary-length data.

	// The ptrace syscall differs from glibc's ptrace.
	// Peeks returns the word in *data, not as the return value.

	var buf [SizeofPtr]byte

	// Leading edge. PEEKTEXT/PEEKDATA don't require aligned
	// access (PEEKUSER warns that it might), but if we don't
	// align our reads, we might straddle an unmapped page
	// boundary and not get the bytes leading up to the page
	// boundary.
	n := 0
	if addr%SizeofPtr != 0 {
		err = ptrace(req, pid, addr-addr%SizeofPtr, uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return 0, err
		}
		n += copy(out, buf[addr%SizeofPtr:])
		out = out[n:]
	}

	// Remainder.
	for len(out) > 0 {
		// We use an internal buffer to guarantee alignment.
		// It's not documented if this is necessary, but we're paranoid.
		err = ptrace(req, pid, addr+uintptr(n), uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return n, err
		}
		copied := copy(out, buf[0:])
		n += copied
		out = out[copied:]
	}

	return n, nil
}

func PtracePeekText(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PTRACE_PEEKTEXT, pid, addr, out)
}

func PtracePeekData(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PTRACE_PEEKDATA, pid, addr, out)
}

func PtracePeekUser(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PTRACE_PEEKUSR, pid, addr, out)
}

func ptracePoke(pokeReq int, peekReq int, pid int, addr uintptr, data []byte) (count int, err error) {
	// As for ptracePeek, we need to align our accesses to deal
	// with the possibility of straddling an invalid page.

	// Leading edge.
	n := 0
	if addr%SizeofPtr != 0 {
		var buf [SizeofPtr]byte
		err = ptrace(peekReq, pid, addr-addr%SizeofPtr, uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return 0, err
		}
		n += copy(buf[addr%SizeofPtr:], data)
		word := *((*uintptr)(unsafe.Pointer(&buf[0])))
		err = ptrace(pokeReq, pid, addr-addr%SizeofPtr, word)
		if err != nil {
			return 0, err
		}
		data = data[n:]
	}

	// Interior.
	for len(data) > SizeofPtr {
		word := *((*uintptr)(unsafe.Pointer(&data[0])))
		err = ptrace(pokeReq, pid, addr+uintptr(n), word)
		if err != nil {
			return n, err
		}
		n += SizeofPtr
		data = data[SizeofPtr:]
	}

	// Trailing edge.
	if len(data) > 0 {
		var buf [SizeofPtr]byte
		err = ptrace(peekReq, pid, addr+uintptr(n), uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return n, err
		}
		copy(buf[0:], data)
		word := *((*uintptr)(unsafe.Pointer(&buf[0])))
		err = ptrace(pokeReq, pid, addr+uintptr(n), word)
		if err != nil {
			return n, err
		}
		n += len(data)
	}

	return n, nil
}

func PtracePokeText(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PTRACE_POKETEXT, PTRACE_PEEKTEXT, pid, addr, data)
}

func PtracePokeData(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PTRACE_POKEDATA, PTRACE_PEEKDATA, pid, addr, data)
}

func PtracePokeUser(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PTRACE_POKEUSR, PTRACE_PEEKUSR, pid, addr, data)
}

func PtraceGetRegs(pid int, regsout *PtraceRegs) (err error) {
	return ptrace(PTRACE_GETREGS, pid, 0, uintptr(unsafe.Pointer(regsout)))
}

func PtraceSetRegs(pid int, regs *PtraceRegs) (err error) {
	return ptrace(PTRACE_SETREGS, pid, 0, uintptr(unsafe.Pointer(regs)))
}

func PtraceSetOptions(pid int, options int) (err error) {
	return ptrace(PTRACE_SETOPTIONS, pid, 0, uintptr(options))
}

func PtraceGetEventMsg(pid int) (msg uint, err error) {
	var data _C_long
	err = ptrace(PTRACE_GETEVENTMSG, pid, 0, uintptr(unsafe.Pointer(&data)))
	msg = uint(data)
	return
}

func PtraceCont(pid int, signal int) (err error) {
	return ptrace(PTRACE_CONT, pid, 0, uintptr(signal))
}

func PtraceSyscall(pid int, signal int) (err error) {
	return ptrace(PTRACE_SYSCALL, pid, 0, uintptr(signal))
}

func PtraceSingleStep(pid int) (err error) { return ptrace(PTRACE_SINGLESTEP, pid, 0, 0) }

func PtraceInterrupt(pid int) (err error) { return ptrace(PTRACE_INTERRUPT, pid, 0, 0) }

func PtraceAttach(pid int) (err error) { return ptrace(PTRACE_ATTACH, pid, 0, 0) }

func PtraceSeize(pid int) (err error) { return ptrace(PTRACE_SEIZE, pid, 0, 0) }

func PtraceDetach(pid int) (err error) { return ptrace(PTRACE_DETACH, pid, 0, 0) }

//sys	reboot(magic1 uint, magic2 uint, cmd int, arg string) (err error)

func Reboot(cmd int) (err error) {
	return reboot(LINUX_REBOOT_MAGIC1, LINUX_REBOOT_MAGIC2, cmd, "")
}

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Ino), unsafe.Sizeof(Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(Dirent{}.Reclen), unsafe.Sizeof(Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	reclen, ok := direntReclen(buf)
	if !ok {
		return 0, false
	}
	return reclen - uint64(unsafe.Offsetof(Dirent{}.Name)), true
}

//sys	mount(source string, target string, fstype string, flags uintptr, data *byte) (err error)

func Mount(source string, target string, fstype string, flags uintptr, data string) (err error) {
	// Certain file systems get rather angry and EINVAL if you give
	// them an empty string of data, rather than NULL.
	if data == "" {
		return mount(source, target, fstype, flags, nil)
	}
	datap, err := BytePtrFromString(data)
	if err != nil {
		return err
	}
	return mount(source, target, fstype, flags, datap)
}

func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	return sendfile(outfd, infd, offset, count)
}

// Sendto
// Recvfrom
// Socketpair

/*
 * Direct access
 */
//sys	Acct(path string) (err error)
//sys	AddKey(keyType string, description string, payload []byte, ringid int) (id int, err error)
//sys	Adjtimex(buf *Timex) (state int, err error)
//sysnb	Capget(hdr *CapUserHeader, data *CapUserData) (err error)
//sysnb	Capset(hdr *CapUserHeader, data *CapUserData) (err error)
//sys	Chdir(path string) (err error)
//sys	Chroot(path string) (err error)
//sys	ClockGetres(clockid int32, res *Timespec) (err error)
//sys	ClockGettime(clockid int32, time *Timespec) (err error)
//sys	ClockNanosleep(clockid int32, flags int, request *Timespec, remain *Timespec) (err error)
//sys	Close(fd int) (err error)
//sys	CopyFileRange(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int, err error)
//sys	DeleteModule(name string, flags int) (err error)
//sys	Dup(oldfd int) (fd int, err error)

func Dup2(oldfd, newfd int) error {
	// Android O and newer blocks dup2; riscv and arm64 don't implement dup2.
	if runtime.GOOS == "android" || runtime.GOARCH == "riscv64" || runtime.GOARCH == "arm64" {
		return Dup3(oldfd, newfd, 0)
	}
	return dup2(oldfd, newfd)
}

//sys	Dup3(oldfd int, newfd int, flags int) (err error)
//sysnb	EpollCreate1(flag int) (fd int, err error)
//sysnb	EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error)
//sys	Eventfd(initval uint, flags int) (fd int, err error) = SYS_EVENTFD2
//sys	Exit(code int) = SYS_EXIT_GROUP
//sys	Fallocate(fd int, mode uint32, off int64, len int64) (err error)
//sys	Fchdir(fd int) (err error)
//sys	Fchmod(fd int, mode uint32) (err error)
//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//sys	Fdatasync(fd int) (err error)
//sys	Fgetxattr(fd int, attr string, dest []byte) (sz int, err error)
//sys	FinitModule(fd int, params string, flags int) (err error)
//sys	Flistxattr(fd int, dest []byte) (sz int, err error)
//sys	Flock(fd int, how int) (err error)
//sys	Fremovexattr(fd int, attr string) (err error)
//sys	Fsetxattr(fd int, attr string, dest []byte, flags int) (err error)
//sys	Fsync(fd int) (err error)
//sys	Getdents(fd int, buf []byte) (n int, err error) = SYS_GETDENTS64
//sysnb	Getpgid(pid int) (pgid int, err error)

func Getpgrp() (pid int) {
	pid, _ = Getpgid(0)
	return
}

//sysnb	Getpid() (pid int)
//sysnb	Getppid() (ppid int)
//sys	Getpriority(which int, who int) (prio int, err error)
//sys	Getrandom(buf []byte, flags int) (n int, err error)
//sysnb	Getrusage(who int, rusage *Rusage) (err error)
//sysnb	Getsid(pid int) (sid int, err error)
//sysnb	Gettid() (tid int)
//sys	Getxattr(path string, attr string, dest []byte) (sz int, err error)
//sys	InitModule(moduleImage []byte, params string) (err error)
//sys	InotifyAddWatch(fd int, pathname string, mask uint32) (watchdesc int, err error)
//sysnb	InotifyInit1(flags int) (fd int, err error)
//sysnb	InotifyRmWatch(fd int, watchdesc uint32) (success int, err error)
//sysnb	Kill(pid int, sig syscall.Signal) (err error)
//sys	Klogctl(typ int, buf []byte) (n int, err error) = SYS_SYSLOG
//sys	Lgetxattr(path string, attr string, dest []byte) (sz int, err error)
//sys	Listxattr(path string, dest []byte) (sz int, err error)
//sys	Llistxattr(path string, dest []byte) (sz int, err error)
//sys	Lremovexattr(path string, attr string) (err error)
//sys	Lsetxattr(path string, attr string, data []byte, flags int) (err error)
//sys	MemfdCreate(name string, flags int) (fd int, err error)
//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//sys	Mknodat(dirfd int, path string, mode uint32, dev int) (err error)
//sys	Nanosleep(time *Timespec, leftover *Timespec) (err error)
//sys	PerfEventOpen(attr *PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error)
//sys	PivotRoot(newroot string, putold string) (err error) = SYS_PIVOT_ROOT
//sysnb prlimit(pid int, resource int, newlimit *Rlimit, old *Rlimit) (err error) = SYS_PRLIMIT64
//sys   Prctl(option int, arg2 uintptr, arg3 uintptr, arg4 uintptr, arg5 uintptr) (err error)
//sys	Pselect(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timespec, sigmask *Sigset_t) (n int, err error) = SYS_PSELECT6
//sys	read(fd int, p []byte) (n int, err error)
//sys	Removexattr(path string, attr string) (err error)
//sys	Renameat2(olddirfd int, oldpath string, newdirfd int, newpath string, flags uint) (err error)
//sys	RequestKey(keyType string, description string, callback string, destRingid int) (id int, err error)
//sys	Setdomainname(p []byte) (err error)
//sys	Sethostname(p []byte) (err error)
//sysnb	Setpgid(pid int, pgid int) (err error)
//sysnb	Setsid() (pid int, err error)
//sysnb	Settimeofday(tv *Timeval) (err error)
//sys	Setns(fd int, nstype int) (err error)

// PrctlRetInt performs a prctl operation specified by option and further
// optional arguments arg2 through arg5 depending on option. It returns a
// non-negative integer that is returned by the prctl syscall.
func PrctlRetInt(option int, arg2 uintptr, arg3 uintptr, arg4 uintptr, arg5 uintptr) (int, error) {
	ret, _, err := Syscall6(SYS_PRCTL, uintptr(option), uintptr(arg2), uintptr(arg3), uintptr(arg4), uintptr(arg5), 0)
	if err != 0 {
		return 0, err
	}
	return int(ret), nil
}

// issue 1435.
// On linux Setuid and Setgid only affects the current thread, not the process.
// This does not match what most callers expect so we must return an error
// here rather than letting the caller think that the call succeeded.

func Setuid(uid int) (err error) {
	return EOPNOTSUPP
}

func Setgid(uid int) (err error) {
	return EOPNOTSUPP
}

// SetfsgidRetGid sets fsgid for current thread and returns previous fsgid set.
// setfsgid(2) will return a non-nil error only if its caller lacks CAP_SETUID capability.
// If the call fails due to other reasons, current fsgid will be returned.
func SetfsgidRetGid(gid int) (int, error) {
	return setfsgid(gid)
}

// SetfsuidRetUid sets fsuid for current thread and returns previous fsuid set.
// setfsgid(2) will return a non-nil error only if its caller lacks CAP_SETUID capability
// If the call fails due to other reasons, current fsuid will be returned.
func SetfsuidRetUid(uid int) (int, error) {
	return setfsuid(uid)
}

func Setfsgid(gid int) error {
	_, err := setfsgid(gid)
	return err
}

func Setfsuid(uid int) error {
	_, err := setfsuid(uid)
	return err
}

func Signalfd(fd int, sigmask *Sigset_t, flags int) (newfd int, err error) {
	return signalfd(fd, sigmask, _C__NSIG/8, flags)
}

//sys	Setpriority(which int, who int, prio int) (err error)
//sys	Setxattr(path string, attr string, data []byte, flags int) (err error)
//sys	signalfd(fd int, sigmask *Sigset_t, maskSize uintptr, flags int) (newfd int, err error) = SYS_SIGNALFD4
//sys	Statx(dirfd int, path string, flags int, mask int, stat *Statx_t) (err error)
//sys	Sync()
//sys	Syncfs(fd int) (err error)
//sysnb	Sysinfo(info *Sysinfo_t) (err error)
//sys	Tee(rfd int, wfd int, len int, flags int) (n int64, err error)
//sysnb TimerfdCreate(clockid int, flags int) (fd int, err error)
//sysnb TimerfdGettime(fd int, currValue *ItimerSpec) (err error)
//sysnb TimerfdSettime(fd int, flags int, newValue *ItimerSpec, oldValue *ItimerSpec) (err error)
//sysnb	Tgkill(tgid int, tid int, sig syscall.Signal) (err error)
//sysnb	Times(tms *Tms) (ticks uintptr, err error)
//sysnb	Umask(mask int) (oldmask int)
//sysnb	Uname(buf *Utsname) (err error)
//sys	Unmount(target string, flags int) (err error) = SYS_UMOUNT2
//sys	Unshare(flags int) (err error)
//sys	write(fd int, p []byte) (n int, err error)
//sys	exitThread(code int) (err error) = SYS_EXIT
//sys	readlen(fd int, p *byte, np int) (n int, err error) = SYS_READ
//sys	writelen(fd int, p *byte, np int) (n int, err error) = SYS_WRITE
//sys	readv(fd int, iovs []Iovec) (n int, err error) = SYS_READV
//sys	writev(fd int, iovs []Iovec) (n int, err error) = SYS_WRITEV
//sys	preadv(fd int, iovs []Iovec, offs_l uintptr, offs_h uintptr) (n int, err error) = SYS_PREADV
//sys	pwritev(fd int, iovs []Iovec, offs_l uintptr, offs_h uintptr) (n int, err error) = SYS_PWRITEV
//sys	preadv2(fd int, iovs []Iovec, offs_l uintptr, offs_h uintptr, flags int) (n int, err error) = SYS_PREADV2
//sys	pwritev2(fd int, iovs []Iovec, offs_l uintptr, offs_h uintptr, flags int) (n int, err error) = SYS_PWRITEV2

func bytes2iovec(bs [][]byte) []Iovec {
	iovecs := make([]Iovec, len(bs))
	for i, b := range bs {
		iovecs[i].SetLen(len(b))
		if len(b) > 0 {
			iovecs[i].Base = &b[0]
		} else {
			iovecs[i].Base = (*byte)(unsafe.Pointer(&_zero))
		}
	}
	return iovecs
}

// offs2lohi splits offs into its lower and upper unsigned long. On 64-bit
// systems, hi will always be 0. On 32-bit systems, offs will be split in half.
// preadv/pwritev chose this calling convention so they don't need to add a
// padding-register for alignment on ARM.
func offs2lohi(offs int64) (lo, hi uintptr) {
	return uintptr(offs), uintptr(uint64(offs) >> SizeofLong)
}

func Readv(fd int, iovs [][]byte) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	n, err = readv(fd, iovecs)
	readvRacedetect(iovecs, n, err)
	return n, err
}

func Preadv(fd int, iovs [][]byte, offset int64) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	lo, hi := offs2lohi(offset)
	n, err = preadv(fd, iovecs, lo, hi)
	readvRacedetect(iovecs, n, err)
	return n, err
}

func Preadv2(fd int, iovs [][]byte, offset int64, flags int) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	lo, hi := offs2lohi(offset)
	n, err = preadv2(fd, iovecs, lo, hi, flags)
	readvRacedetect(iovecs, n, err)
	return n, err
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

func Writev(fd int, iovs [][]byte) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = writev(fd, iovecs)
	writevRacedetect(iovecs, n)
	return n, err
}

func Pwritev(fd int, iovs [][]byte, offset int64) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	lo, hi := offs2lohi(offset)
	n, err = pwritev(fd, iovecs, lo, hi)
	writevRacedetect(iovecs, n)
	return n, err
}

func Pwritev2(fd int, iovs [][]byte, offset int64, flags int) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	lo, hi := offs2lohi(offset)
	n, err = pwritev2(fd, iovecs, lo, hi, flags)
	writevRacedetect(iovecs, n)
	return n, err
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

// mmap varies by architecture; see syscall_linux_*.go.
//sys	munmap(addr uintptr, length uintptr) (err error)

var mapper = &mmapper{
	active: make(map[*byte][]byte),
	mmap:   mmap,
	munmap: munmap,
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
}

//sys	Madvise(b []byte, advice int) (err error)
//sys	Mprotect(b []byte, prot int) (err error)
//sys	Mlock(b []byte) (err error)
//sys	Mlockall(flags int) (err error)
//sys	Msync(b []byte, flags int) (err error)
//sys	Munlock(b []byte) (err error)
//sys	Munlockall() (err error)

// Vmsplice splices user pages from a slice of Iovecs into a pipe specified by fd,
// using the specified flags.
func Vmsplice(fd int, iovs []Iovec, flags int) (int, error) {
	var p unsafe.Pointer
	if len(iovs) > 0 {
		p = unsafe.Pointer(&iovs[0])
	}

	n, _, errno := Syscall6(SYS_VMSPLICE, uintptr(fd), uintptr(p), uintptr(len(iovs)), uintptr(flags), 0, 0)
	if errno != 0 {
		return 0, syscall.Errno(errno)
	}

	return int(n), nil
}

func isGroupMember(gid int) bool {
	groups, err := Getgroups()
	if err != nil {
		return false
	}

	for _, g := range groups {
		if g == gid {
			return true
		}
	}
	return false
}

//sys	faccessat(dirfd int, path string, mode uint32) (err error)
//sys	Faccessat2(dirfd int, path string, mode uint32, flags int) (err error)

func Faccessat(dirfd int, path string, mode uint32, flags int) (err error) {
	if flags == 0 {
		return faccessat(dirfd, path, mode)
	}

	if err := Faccessat2(dirfd, path, mode, flags); err != ENOSYS && err != EPERM {
		return err
	}

	// The Linux kernel faccessat system call does not take any flags.
	// The glibc faccessat implements the flags itself; see
	// https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/unix/sysv/linux/faccessat.c;hb=HEAD
	// Because people naturally expect syscall.Faccessat to act
	// like C faccessat, we do the same.

	if flags & ^(AT_SYMLINK_NOFOLLOW|AT_EACCESS) != 0 {
		return EINVAL
	}

	var st Stat_t
	if err := Fstatat(dirfd, path, &st, flags&AT_SYMLINK_NOFOLLOW); err != nil {
		return err
	}

	mode &= 7
	if mode == 0 {
		return nil
	}

	var uid int
	if flags&AT_EACCESS != 0 {
		uid = Geteuid()
	} else {
		uid = Getuid()
	}

	if uid == 0 {
		if mode&1 == 0 {
			// Root can read and write any file.
			return nil
		}
		if st.Mode&0111 != 0 {
			// Root can execute any file that anybody can execute.
			return nil
		}
		return EACCES
	}

	var fmode uint32
	if uint32(uid) == st.Uid {
		fmode = (st.Mode >> 6) & 7
	} else {
		var gid int
		if flags&AT_EACCESS != 0 {
			gid = Getegid()
		} else {
			gid = Getgid()
		}

		if uint32(gid) == st.Gid || isGroupMember(gid) {
			fmode = (st.Mode >> 3) & 7
		} else {
			fmode = st.Mode & 7
		}
	}

	if fmode&mode == mode {
		return nil
	}

	return EACCES
}

//sys nameToHandleAt(dirFD int, pathname string, fh *fileHandle, mountID *_C_int, flags int) (err error) = SYS_NAME_TO_HANDLE_AT
//sys openByHandleAt(mountFD int, fh *fileHandle, flags int) (fd int, err error) = SYS_OPEN_BY_HANDLE_AT

// fileHandle is the argument to nameToHandleAt and openByHandleAt. We
// originally tried to generate it via unix/linux/types.go with "type
// fileHandle C.struct_file_handle" but that generated empty structs
// for mips64 and mips64le. Instead, hard code it for now (it's the
// same everywhere else) until the mips64 generator issue is fixed.
type fileHandle struct {
	Bytes uint32
	Type  int32
}

// FileHandle represents the C struct file_handle used by
// name_to_handle_at (see NameToHandleAt) and open_by_handle_at (see
// OpenByHandleAt).
type FileHandle struct {
	*fileHandle
}

// NewFileHandle constructs a FileHandle.
func NewFileHandle(handleType int32, handle []byte) FileHandle {
	const hdrSize = unsafe.Sizeof(fileHandle{})
	buf := make([]byte, hdrSize+uintptr(len(handle)))
	copy(buf[hdrSize:], handle)
	fh := (*fileHandle)(unsafe.Pointer(&buf[0]))
	fh.Type = handleType
	fh.Bytes = uint32(len(handle))
	return FileHandle{fh}
}

func (fh *FileHandle) Size() int   { return int(fh.fileHandle.Bytes) }
func (fh *FileHandle) Type() int32 { return fh.fileHandle.Type }
func (fh *FileHandle) Bytes() []byte {
	n := fh.Size()
	if n == 0 {
		return nil
	}
	return (*[1 << 30]byte)(unsafe.Pointer(uintptr(unsafe.Pointer(&fh.fileHandle.Type)) + 4))[:n:n]
}

// NameToHandleAt wraps the name_to_handle_at system call; it obtains
// a handle for a path name.
func NameToHandleAt(dirfd int, path string, flags int) (handle FileHandle, mountID int, err error) {
	var mid _C_int
	// Try first with a small buffer, assuming the handle will
	// only be 32 bytes.
	size := uint32(32 + unsafe.Sizeof(fileHandle{}))
	didResize := false
	for {
		buf := make([]byte, size)
		fh := (*fileHandle)(unsafe.Pointer(&buf[0]))
		fh.Bytes = size - uint32(unsafe.Sizeof(fileHandle{}))
		err = nameToHandleAt(dirfd, path, fh, &mid, flags)
		if err == EOVERFLOW {
			if didResize {
				// We shouldn't need to resize more than once
				return
			}
			didResize = true
			size = fh.Bytes + uint32(unsafe.Sizeof(fileHandle{}))
			continue
		}
		if err != nil {
			return
		}
		return FileHandle{fh}, int(mid), nil
	}
}

// OpenByHandleAt wraps the open_by_handle_at system call; it opens a
// file via a handle as previously returned by NameToHandleAt.
func OpenByHandleAt(mountFD int, handle FileHandle, flags int) (fd int, err error) {
	return openByHandleAt(mountFD, handle.fileHandle, flags)
}

// Klogset wraps the sys_syslog system call; it sets console_loglevel to
// the value specified by arg and passes a dummy pointer to bufp.
func Klogset(typ int, arg int) (err error) {
	var p unsafe.Pointer
	_, _, errno := Syscall(SYS_SYSLOG, uintptr(typ), uintptr(p), uintptr(arg))
	if errno != 0 {
		return errnoErr(errno)
	}
	return nil
}

// RemoteIovec is Iovec with the pointer replaced with an integer.
// It is used for ProcessVMReadv and ProcessVMWritev, where the pointer
// refers to a location in a different process' address space, which
// would confuse the Go garbage collector.
type RemoteIovec struct {
	Base uintptr
	Len  int
}

//sys	ProcessVMReadv(pid int, localIov []Iovec, remoteIov []RemoteIovec, flags uint) (n int, err error) = SYS_PROCESS_VM_READV
//sys	ProcessVMWritev(pid int, localIov []Iovec, remoteIov []RemoteIovec, flags uint) (n int, err error) = SYS_PROCESS_VM_WRITEV

/*
 * Unimplemented
 */
// AfsSyscall
// Alarm
// ArchPrctl
// Brk
// ClockNanosleep
// ClockSettime
// Clone
// EpollCtlOld
// EpollPwait
// EpollWaitOld
// Execve
// Fork
// Futex
// GetKernelSyms
// GetMempolicy
// GetRobustList
// GetThreadArea
// Getitimer
// Getpmsg
// IoCancel
// IoDestroy
// IoGetevents
// IoSetup
// IoSubmit
// IoprioGet
// IoprioSet
// KexecLoad
// LookupDcookie
// Mbind
// MigratePages
// Mincore
// ModifyLdt
// Mount
// MovePages
// MqGetsetattr
// MqNotify
// MqOpen
// MqTimedreceive
// MqTimedsend
// MqUnlink
// Mremap
// Msgctl
// Msgget
// Msgrcv
// Msgsnd
// Nfsservctl
// Personality
// Pselect6
// Ptrace
// Putpmsg
// Quotactl
// Readahead
// Readv
// RemapFilePages
// RestartSyscall
// RtSigaction
// RtSigpending
// RtSigprocmask
// RtSigqueueinfo
// RtSigreturn
// RtSigsuspend
// RtSigtimedwait
// SchedGetPriorityMax
// SchedGetPriorityMin
// SchedGetparam
// SchedGetscheduler
// SchedRrGetInterval
// SchedSetparam
// SchedYield
// Security
// Semctl
// Semget
// Semop
// Semtimedop
// SetMempolicy
// SetRobustList
// SetThreadArea
// SetTidAddress
// Shmat
// Shmctl
// Shmdt
// Shmget
// Sigaltstack
// Swapoff
// Swapon
// Sysfs
// TimerCreate
// TimerDelete
// TimerGetoverrun
// TimerGettime
// TimerSettime
// Tkill (obsolete)
// Tuxcall
// Umount2
// Uselib
// Utimensat
// Vfork
// Vhangup
// Vserver
// Waitid
// _Sysctl
