// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_NSIG    = 32
	_SI_USER = 1

	// native_client/src/trusted/service_runtime/include/sys/errno.h
	// The errors are mainly copied from Linux.
	_EPERM   = 1  /* Operation not permitted */
	_ENOENT  = 2  /* No such file or directory */
	_ESRCH   = 3  /* No such process */
	_EINTR   = 4  /* Interrupted system call */
	_EIO     = 5  /* I/O error */
	_ENXIO   = 6  /* No such device or address */
	_E2BIG   = 7  /* Argument list too long */
	_ENOEXEC = 8  /* Exec format error */
	_EBADF   = 9  /* Bad file number */
	_ECHILD  = 10 /* No child processes */
	_EAGAIN  = 11 /* Try again */
	// _ENOMEM is defined in mem_bsd.go for nacl.
	// _ENOMEM          = 12       /* Out of memory */
	_EACCES          = 13       /* Permission denied */
	_EFAULT          = 14       /* Bad address */
	_EBUSY           = 16       /* Device or resource busy */
	_EEXIST          = 17       /* File exists */
	_EXDEV           = 18       /* Cross-device link */
	_ENODEV          = 19       /* No such device */
	_ENOTDIR         = 20       /* Not a directory */
	_EISDIR          = 21       /* Is a directory */
	_EINVAL          = 22       /* Invalid argument */
	_ENFILE          = 23       /* File table overflow */
	_EMFILE          = 24       /* Too many open files */
	_ENOTTY          = 25       /* Not a typewriter */
	_EFBIG           = 27       /* File too large */
	_ENOSPC          = 28       /* No space left on device */
	_ESPIPE          = 29       /* Illegal seek */
	_EROFS           = 30       /* Read-only file system */
	_EMLINK          = 31       /* Too many links */
	_EPIPE           = 32       /* Broken pipe */
	_ENAMETOOLONG    = 36       /* File name too long */
	_ENOSYS          = 38       /* Function not implemented */
	_EDQUOT          = 122      /* Quota exceeded */
	_EDOM            = 33       /* Math arg out of domain of func */
	_ERANGE          = 34       /* Math result not representable */
	_EDEADLK         = 35       /* Deadlock condition */
	_ENOLCK          = 37       /* No record locks available */
	_ENOTEMPTY       = 39       /* Directory not empty */
	_ELOOP           = 40       /* Too many symbolic links */
	_ENOMSG          = 42       /* No message of desired type */
	_EIDRM           = 43       /* Identifier removed */
	_ECHRNG          = 44       /* Channel number out of range */
	_EL2NSYNC        = 45       /* Level 2 not synchronized */
	_EL3HLT          = 46       /* Level 3 halted */
	_EL3RST          = 47       /* Level 3 reset */
	_ELNRNG          = 48       /* Link number out of range */
	_EUNATCH         = 49       /* Protocol driver not attached */
	_ENOCSI          = 50       /* No CSI structure available */
	_EL2HLT          = 51       /* Level 2 halted */
	_EBADE           = 52       /* Invalid exchange */
	_EBADR           = 53       /* Invalid request descriptor */
	_EXFULL          = 54       /* Exchange full */
	_ENOANO          = 55       /* No anode */
	_EBADRQC         = 56       /* Invalid request code */
	_EBADSLT         = 57       /* Invalid slot */
	_EDEADLOCK       = _EDEADLK /* File locking deadlock error */
	_EBFONT          = 59       /* Bad font file fmt */
	_ENOSTR          = 60       /* Device not a stream */
	_ENODATA         = 61       /* No data (for no delay io) */
	_ETIME           = 62       /* Timer expired */
	_ENOSR           = 63       /* Out of streams resources */
	_ENONET          = 64       /* Machine is not on the network */
	_ENOPKG          = 65       /* Package not installed */
	_EREMOTE         = 66       /* The object is remote */
	_ENOLINK         = 67       /* The link has been severed */
	_EADV            = 68       /* Advertise error */
	_ESRMNT          = 69       /* Srmount error */
	_ECOMM           = 70       /* Communication error on send */
	_EPROTO          = 71       /* Protocol error */
	_EMULTIHOP       = 72       /* Multihop attempted */
	_EDOTDOT         = 73       /* Cross mount point (not really error) */
	_EBADMSG         = 74       /* Trying to read unreadable message */
	_EOVERFLOW       = 75       /* Value too large for defined data type */
	_ENOTUNIQ        = 76       /* Given log. name not unique */
	_EBADFD          = 77       /* f.d. invalid for this operation */
	_EREMCHG         = 78       /* Remote address changed */
	_ELIBACC         = 79       /* Can't access a needed shared lib */
	_ELIBBAD         = 80       /* Accessing a corrupted shared lib */
	_ELIBSCN         = 81       /* .lib section in a.out corrupted */
	_ELIBMAX         = 82       /* Attempting to link in too many libs */
	_ELIBEXEC        = 83       /* Attempting to exec a shared library */
	_EILSEQ          = 84
	_EUSERS          = 87
	_ENOTSOCK        = 88  /* Socket operation on non-socket */
	_EDESTADDRREQ    = 89  /* Destination address required */
	_EMSGSIZE        = 90  /* Message too long */
	_EPROTOTYPE      = 91  /* Protocol wrong type for socket */
	_ENOPROTOOPT     = 92  /* Protocol not available */
	_EPROTONOSUPPORT = 93  /* Unknown protocol */
	_ESOCKTNOSUPPORT = 94  /* Socket type not supported */
	_EOPNOTSUPP      = 95  /* Operation not supported on transport endpoint */
	_EPFNOSUPPORT    = 96  /* Protocol family not supported */
	_EAFNOSUPPORT    = 97  /* Address family not supported by protocol family */
	_EADDRINUSE      = 98  /* Address already in use */
	_EADDRNOTAVAIL   = 99  /* Address not available */
	_ENETDOWN        = 100 /* Network interface is not configured */
	_ENETUNREACH     = 101 /* Network is unreachable */
	_ENETRESET       = 102
	_ECONNABORTED    = 103 /* Connection aborted */
	_ECONNRESET      = 104 /* Connection reset by peer */
	_ENOBUFS         = 105 /* No buffer space available */
	_EISCONN         = 106 /* Socket is already connected */
	_ENOTCONN        = 107 /* Socket is not connected */
	_ESHUTDOWN       = 108 /* Can't send after socket shutdown */
	_ETOOMANYREFS    = 109
	_ETIMEDOUT       = 110 /* Connection timed out */
	_ECONNREFUSED    = 111 /* Connection refused */
	_EHOSTDOWN       = 112 /* Host is down */
	_EHOSTUNREACH    = 113 /* Host is unreachable */
	_EALREADY        = 114 /* Socket already connected */
	_EINPROGRESS     = 115 /* Connection already in progress */
	_ESTALE          = 116
	_ENOTSUP         = _EOPNOTSUPP /* Not supported */
	_ENOMEDIUM       = 123         /* No medium (in tape drive) */
	_ECANCELED       = 125         /* Operation canceled. */
	_ELBIN           = 2048        /* Inode is remote (not really error) */
	_EFTYPE          = 2049        /* Inappropriate file type or format */
	_ENMFILE         = 2050        /* No more files */
	_EPROCLIM        = 2051
	_ENOSHARE        = 2052    /* No such host or network path */
	_ECASECLASH      = 2053    /* Filename exists with different case */
	_EWOULDBLOCK     = _EAGAIN /* Operation would block */

	// native_client/src/trusted/service_runtime/include/bits/mman.h.
	// NOTE: DO NOT USE native_client/src/shared/imc/nacl_imc_c.h.
	// Those MAP_*values are different from these.
	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_SHARED  = 0x1
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10
	_MAP_ANON    = 0x20

	_MADV_FREE  = 0
	_SIGFPE     = 8
	_FPE_INTDIV = 0
)

type siginfo struct{}
