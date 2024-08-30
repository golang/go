// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package syscall

import "runtime"

// TODO: Auto-generate some day. (Hard-coded in binaries so not likely to change.)
const (
	E2BIG           Errno = 1
	EACCES          Errno = 2
	EADDRINUSE      Errno = 3
	EADDRNOTAVAIL   Errno = 4
	EAFNOSUPPORT    Errno = 5
	EAGAIN          Errno = 6
	EALREADY        Errno = 7
	EBADF           Errno = 8
	EBADMSG         Errno = 9
	EBUSY           Errno = 10
	ECANCELED       Errno = 11
	ECHILD          Errno = 12
	ECONNABORTED    Errno = 13
	ECONNREFUSED    Errno = 14
	ECONNRESET      Errno = 15
	EDEADLK         Errno = 16
	EDESTADDRREQ    Errno = 17
	EDOM            Errno = 18
	EDQUOT          Errno = 19
	EEXIST          Errno = 20
	EFAULT          Errno = 21
	EFBIG           Errno = 22
	EHOSTUNREACH    Errno = 23
	EIDRM           Errno = 24
	EILSEQ          Errno = 25
	EINPROGRESS     Errno = 26
	EINTR           Errno = 27
	EINVAL          Errno = 28
	EIO             Errno = 29
	EISCONN         Errno = 30
	EISDIR          Errno = 31
	ELOOP           Errno = 32
	EMFILE          Errno = 33
	EMLINK          Errno = 34
	EMSGSIZE        Errno = 35
	EMULTIHOP       Errno = 36
	ENAMETOOLONG    Errno = 37
	ENETDOWN        Errno = 38
	ENETRESET       Errno = 39
	ENETUNREACH     Errno = 40
	ENFILE          Errno = 41
	ENOBUFS         Errno = 42
	ENODEV          Errno = 43
	ENOENT          Errno = 44
	ENOEXEC         Errno = 45
	ENOLCK          Errno = 46
	ENOLINK         Errno = 47
	ENOMEM          Errno = 48
	ENOMSG          Errno = 49
	ENOPROTOOPT     Errno = 50
	ENOSPC          Errno = 51
	ENOSYS          Errno = 52
	ENOTCONN        Errno = 53
	ENOTDIR         Errno = 54
	ENOTEMPTY       Errno = 55
	ENOTRECOVERABLE Errno = 56
	ENOTSOCK        Errno = 57
	ENOTSUP         Errno = 58
	ENOTTY          Errno = 59
	ENXIO           Errno = 60
	EOVERFLOW       Errno = 61
	EOWNERDEAD      Errno = 62
	EPERM           Errno = 63
	EPIPE           Errno = 64
	EPROTO          Errno = 65
	EPROTONOSUPPORT Errno = 66
	EPROTOTYPE      Errno = 67
	ERANGE          Errno = 68
	EROFS           Errno = 69
	ESPIPE          Errno = 70
	ESRCH           Errno = 71
	ESTALE          Errno = 72
	ETIMEDOUT       Errno = 73
	ETXTBSY         Errno = 74
	EXDEV           Errno = 75
	ENOTCAPABLE     Errno = 76
	// needed by src/net/error_unix_test.go
	EOPNOTSUPP = ENOTSUP
)

// TODO: Auto-generate some day. (Hard-coded in binaries so not likely to change.)
var errorstr = [...]string{
	E2BIG:           "Argument list too long",
	EACCES:          "Permission denied",
	EADDRINUSE:      "Address already in use",
	EADDRNOTAVAIL:   "Address not available",
	EAFNOSUPPORT:    "Address family not supported by protocol family",
	EAGAIN:          "Try again",
	EALREADY:        "Socket already connected",
	EBADF:           "Bad file number",
	EBADMSG:         "Trying to read unreadable message",
	EBUSY:           "Device or resource busy",
	ECANCELED:       "Operation canceled.",
	ECHILD:          "No child processes",
	ECONNABORTED:    "Connection aborted",
	ECONNREFUSED:    "Connection refused",
	ECONNRESET:      "Connection reset by peer",
	EDEADLK:         "Deadlock condition",
	EDESTADDRREQ:    "Destination address required",
	EDOM:            "Math arg out of domain of func",
	EDQUOT:          "Quota exceeded",
	EEXIST:          "File exists",
	EFAULT:          "Bad address",
	EFBIG:           "File too large",
	EHOSTUNREACH:    "Host is unreachable",
	EIDRM:           "Identifier removed",
	EILSEQ:          "EILSEQ",
	EINPROGRESS:     "Connection already in progress",
	EINTR:           "Interrupted system call",
	EINVAL:          "Invalid argument",
	EIO:             "I/O error",
	EISCONN:         "Socket is already connected",
	EISDIR:          "Is a directory",
	ELOOP:           "Too many symbolic links",
	EMFILE:          "Too many open files",
	EMLINK:          "Too many links",
	EMSGSIZE:        "Message too long",
	EMULTIHOP:       "Multihop attempted",
	ENAMETOOLONG:    "File name too long",
	ENETDOWN:        "Network interface is not configured",
	ENETRESET:       "Network dropped connection on reset",
	ENETUNREACH:     "Network is unreachable",
	ENFILE:          "File table overflow",
	ENOBUFS:         "No buffer space available",
	ENODEV:          "No such device",
	ENOENT:          "No such file or directory",
	ENOEXEC:         "Exec format error",
	ENOLCK:          "No record locks available",
	ENOLINK:         "The link has been severed",
	ENOMEM:          "Out of memory",
	ENOMSG:          "No message of desired type",
	ENOPROTOOPT:     "Protocol not available",
	ENOSPC:          "No space left on device",
	ENOSYS:          "Not implemented on " + runtime.GOOS,
	ENOTCONN:        "Socket is not connected",
	ENOTDIR:         "Not a directory",
	ENOTEMPTY:       "Directory not empty",
	ENOTRECOVERABLE: "State not recoverable",
	ENOTSOCK:        "Socket operation on non-socket",
	ENOTSUP:         "Not supported",
	ENOTTY:          "Not a typewriter",
	ENXIO:           "No such device or address",
	EOVERFLOW:       "Value too large for defined data type",
	EOWNERDEAD:      "Owner died",
	EPERM:           "Operation not permitted",
	EPIPE:           "Broken pipe",
	EPROTO:          "Protocol error",
	EPROTONOSUPPORT: "Unknown protocol",
	EPROTOTYPE:      "Protocol wrong type for socket",
	ERANGE:          "Math result not representable",
	EROFS:           "Read-only file system",
	ESPIPE:          "Illegal seek",
	ESRCH:           "No such process",
	ESTALE:          "Stale file handle",
	ETIMEDOUT:       "Connection timed out",
	ETXTBSY:         "Text file busy",
	EXDEV:           "Cross-device link",
	ENOTCAPABLE:     "Capabilities insufficient",
}

// Do the interface allocations only once for common
// Errno values.
var (
	errEAGAIN error = EAGAIN
	errEINVAL error = EINVAL
	errENOENT error = ENOENT
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
//
// We set both noinline and nosplit to reduce code size, this function has many
// call sites in the syscall package, inlining it causes a significant increase
// of the compiled code; the function call ultimately does not make a difference
// in the performance of syscall functions since the time is dominated by calls
// to the imports and path resolution.
//
//go:noinline
//go:nosplit
func errnoErr(e Errno) error {
	switch e {
	case 0:
		return nil
	case EAGAIN:
		return errEAGAIN
	case EINVAL:
		return errEINVAL
	case ENOENT:
		return errENOENT
	}
	return e
}
