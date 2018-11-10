package runtime

import "unsafe"

// Return values of access/connect/socket are the return values of the syscall
// (may encode error numbers).

// int access(const char *, int)
//go:noescape
func access(name *byte, mode int32) int32

// int connect(int, const struct sockaddr*, socklen_t)
func connect(fd int32, addr unsafe.Pointer, len int32) int32

// int socket(int, int, int)
func socket(domain int32, typ int32, prot int32) int32
