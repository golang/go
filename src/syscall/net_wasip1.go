// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package syscall

const (
	SHUT_RD   = 0x1
	SHUT_WR   = 0x2
	SHUT_RDWR = SHUT_RD | SHUT_WR
)

type sdflags = uint32

//go:wasmimport wasi_snapshot_preview1 sock_accept
//go:noescape
func sock_accept(fd int32, flags fdflags, newfd *int32) Errno

//go:wasmimport wasi_snapshot_preview1 sock_shutdown
//go:noescape
func sock_shutdown(fd int32, flags sdflags) Errno

func Socket(proto, sotype, unused int) (fd int, err error) {
	return 0, ENOSYS
}

func Bind(fd int, sa Sockaddr) error {
	return ENOSYS
}

func StopIO(fd int) error {
	return ENOSYS
}

func Listen(fd int, backlog int) error {
	return ENOSYS
}

func Accept(fd int) (int, Sockaddr, error) {
	var newfd int32
	errno := sock_accept(int32(fd), 0, &newfd)
	return int(newfd), nil, errnoErr(errno)
}

func Connect(fd int, sa Sockaddr) error {
	return ENOSYS
}

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, err error) {
	return 0, nil, ENOSYS
}

func Sendto(fd int, p []byte, flags int, to Sockaddr) error {
	return ENOSYS
}

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn, recvflags int, from Sockaddr, err error) {
	return 0, 0, 0, nil, ENOSYS
}

func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	return 0, ENOSYS
}

func GetsockoptInt(fd, level, opt int) (value int, err error) {
	return 0, ENOSYS
}

func SetsockoptInt(fd, level, opt int, value int) error {
	return ENOSYS
}

func SetReadDeadline(fd int, t int64) error {
	return ENOSYS
}

func SetWriteDeadline(fd int, t int64) error {
	return ENOSYS
}

func Shutdown(fd int, how int) error {
	errno := sock_shutdown(int32(fd), sdflags(how))
	return errnoErr(errno)
}
