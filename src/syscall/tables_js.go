// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm
// +build js,wasm

package syscall

import "runtime"

// These were originally used by Nacl, then later also used by
// js/wasm. Now that they're only used by js/wasm, these numbers are
// just arbitrary.
//
// TODO: delete? replace with something meaningful?
const (
	sys_null                 = 1
	sys_nameservice          = 2
	sys_dup                  = 8
	sys_dup2                 = 9
	sys_open                 = 10
	sys_close                = 11
	sys_read                 = 12
	sys_write                = 13
	sys_lseek                = 14
	sys_stat                 = 16
	sys_fstat                = 17
	sys_chmod                = 18
	sys_isatty               = 19
	sys_brk                  = 20
	sys_mmap                 = 21
	sys_munmap               = 22
	sys_getdents             = 23
	sys_mprotect             = 24
	sys_list_mappings        = 25
	sys_exit                 = 30
	sys_getpid               = 31
	sys_sched_yield          = 32
	sys_sysconf              = 33
	sys_gettimeofday         = 40
	sys_clock                = 41
	sys_nanosleep            = 42
	sys_clock_getres         = 43
	sys_clock_gettime        = 44
	sys_mkdir                = 45
	sys_rmdir                = 46
	sys_chdir                = 47
	sys_getcwd               = 48
	sys_unlink               = 49
	sys_imc_makeboundsock    = 60
	sys_imc_accept           = 61
	sys_imc_connect          = 62
	sys_imc_sendmsg          = 63
	sys_imc_recvmsg          = 64
	sys_imc_mem_obj_create   = 65
	sys_imc_socketpair       = 66
	sys_mutex_create         = 70
	sys_mutex_lock           = 71
	sys_mutex_trylock        = 72
	sys_mutex_unlock         = 73
	sys_cond_create          = 74
	sys_cond_wait            = 75
	sys_cond_signal          = 76
	sys_cond_broadcast       = 77
	sys_cond_timed_wait_abs  = 79
	sys_thread_create        = 80
	sys_thread_exit          = 81
	sys_tls_init             = 82
	sys_thread_nice          = 83
	sys_tls_get              = 84
	sys_second_tls_set       = 85
	sys_second_tls_get       = 86
	sys_exception_handler    = 87
	sys_exception_stack      = 88
	sys_exception_clear_flag = 89
	sys_sem_create           = 100
	sys_sem_wait             = 101
	sys_sem_post             = 102
	sys_sem_get_value        = 103
	sys_dyncode_create       = 104
	sys_dyncode_modify       = 105
	sys_dyncode_delete       = 106
	sys_test_infoleak        = 109
	sys_test_crash           = 110
	sys_test_syscall_1       = 111
	sys_test_syscall_2       = 112
	sys_futex_wait_abs       = 120
	sys_futex_wake           = 121
	sys_pread                = 130
	sys_pwrite               = 131
	sys_truncate             = 140
	sys_lstat                = 141
	sys_link                 = 142
	sys_rename               = 143
	sys_symlink              = 144
	sys_access               = 145
	sys_readlink             = 146
	sys_utimes               = 147
	sys_get_random_bytes     = 150
)

// TODO: Auto-generate some day. (Hard-coded in binaries so not likely to change.)
const (
	// native_client/src/trusted/service_runtime/include/sys/errno.h
	// The errors are mainly copied from Linux.
	EPERM           Errno = 1       /* Operation not permitted */
	ENOENT          Errno = 2       /* No such file or directory */
	ESRCH           Errno = 3       /* No such process */
	EINTR           Errno = 4       /* Interrupted system call */
	EIO             Errno = 5       /* I/O error */
	ENXIO           Errno = 6       /* No such device or address */
	E2BIG           Errno = 7       /* Argument list too long */
	ENOEXEC         Errno = 8       /* Exec format error */
	EBADF           Errno = 9       /* Bad file number */
	ECHILD          Errno = 10      /* No child processes */
	EAGAIN          Errno = 11      /* Try again */
	ENOMEM          Errno = 12      /* Out of memory */
	EACCES          Errno = 13      /* Permission denied */
	EFAULT          Errno = 14      /* Bad address */
	EBUSY           Errno = 16      /* Device or resource busy */
	EEXIST          Errno = 17      /* File exists */
	EXDEV           Errno = 18      /* Cross-device link */
	ENODEV          Errno = 19      /* No such device */
	ENOTDIR         Errno = 20      /* Not a directory */
	EISDIR          Errno = 21      /* Is a directory */
	EINVAL          Errno = 22      /* Invalid argument */
	ENFILE          Errno = 23      /* File table overflow */
	EMFILE          Errno = 24      /* Too many open files */
	ENOTTY          Errno = 25      /* Not a typewriter */
	EFBIG           Errno = 27      /* File too large */
	ENOSPC          Errno = 28      /* No space left on device */
	ESPIPE          Errno = 29      /* Illegal seek */
	EROFS           Errno = 30      /* Read-only file system */
	EMLINK          Errno = 31      /* Too many links */
	EPIPE           Errno = 32      /* Broken pipe */
	ENAMETOOLONG    Errno = 36      /* File name too long */
	ENOSYS          Errno = 38      /* Function not implemented */
	EDQUOT          Errno = 122     /* Quota exceeded */
	EDOM            Errno = 33      /* Math arg out of domain of func */
	ERANGE          Errno = 34      /* Math result not representable */
	EDEADLK         Errno = 35      /* Deadlock condition */
	ENOLCK          Errno = 37      /* No record locks available */
	ENOTEMPTY       Errno = 39      /* Directory not empty */
	ELOOP           Errno = 40      /* Too many symbolic links */
	ENOMSG          Errno = 42      /* No message of desired type */
	EIDRM           Errno = 43      /* Identifier removed */
	ECHRNG          Errno = 44      /* Channel number out of range */
	EL2NSYNC        Errno = 45      /* Level 2 not synchronized */
	EL3HLT          Errno = 46      /* Level 3 halted */
	EL3RST          Errno = 47      /* Level 3 reset */
	ELNRNG          Errno = 48      /* Link number out of range */
	EUNATCH         Errno = 49      /* Protocol driver not attached */
	ENOCSI          Errno = 50      /* No CSI structure available */
	EL2HLT          Errno = 51      /* Level 2 halted */
	EBADE           Errno = 52      /* Invalid exchange */
	EBADR           Errno = 53      /* Invalid request descriptor */
	EXFULL          Errno = 54      /* Exchange full */
	ENOANO          Errno = 55      /* No anode */
	EBADRQC         Errno = 56      /* Invalid request code */
	EBADSLT         Errno = 57      /* Invalid slot */
	EDEADLOCK       Errno = EDEADLK /* File locking deadlock error */
	EBFONT          Errno = 59      /* Bad font file fmt */
	ENOSTR          Errno = 60      /* Device not a stream */
	ENODATA         Errno = 61      /* No data (for no delay io) */
	ETIME           Errno = 62      /* Timer expired */
	ENOSR           Errno = 63      /* Out of streams resources */
	ENONET          Errno = 64      /* Machine is not on the network */
	ENOPKG          Errno = 65      /* Package not installed */
	EREMOTE         Errno = 66      /* The object is remote */
	ENOLINK         Errno = 67      /* The link has been severed */
	EADV            Errno = 68      /* Advertise error */
	ESRMNT          Errno = 69      /* Srmount error */
	ECOMM           Errno = 70      /* Communication error on send */
	EPROTO          Errno = 71      /* Protocol error */
	EMULTIHOP       Errno = 72      /* Multihop attempted */
	EDOTDOT         Errno = 73      /* Cross mount point (not really error) */
	EBADMSG         Errno = 74      /* Trying to read unreadable message */
	EOVERFLOW       Errno = 75      /* Value too large for defined data type */
	ENOTUNIQ        Errno = 76      /* Given log. name not unique */
	EBADFD          Errno = 77      /* f.d. invalid for this operation */
	EREMCHG         Errno = 78      /* Remote address changed */
	ELIBACC         Errno = 79      /* Can't access a needed shared lib */
	ELIBBAD         Errno = 80      /* Accessing a corrupted shared lib */
	ELIBSCN         Errno = 81      /* .lib section in a.out corrupted */
	ELIBMAX         Errno = 82      /* Attempting to link in too many libs */
	ELIBEXEC        Errno = 83      /* Attempting to exec a shared library */
	EILSEQ          Errno = 84
	EUSERS          Errno = 87
	ENOTSOCK        Errno = 88  /* Socket operation on non-socket */
	EDESTADDRREQ    Errno = 89  /* Destination address required */
	EMSGSIZE        Errno = 90  /* Message too long */
	EPROTOTYPE      Errno = 91  /* Protocol wrong type for socket */
	ENOPROTOOPT     Errno = 92  /* Protocol not available */
	EPROTONOSUPPORT Errno = 93  /* Unknown protocol */
	ESOCKTNOSUPPORT Errno = 94  /* Socket type not supported */
	EOPNOTSUPP      Errno = 95  /* Operation not supported on transport endpoint */
	EPFNOSUPPORT    Errno = 96  /* Protocol family not supported */
	EAFNOSUPPORT    Errno = 97  /* Address family not supported by protocol family */
	EADDRINUSE      Errno = 98  /* Address already in use */
	EADDRNOTAVAIL   Errno = 99  /* Address not available */
	ENETDOWN        Errno = 100 /* Network interface is not configured */
	ENETUNREACH     Errno = 101 /* Network is unreachable */
	ENETRESET       Errno = 102
	ECONNABORTED    Errno = 103 /* Connection aborted */
	ECONNRESET      Errno = 104 /* Connection reset by peer */
	ENOBUFS         Errno = 105 /* No buffer space available */
	EISCONN         Errno = 106 /* Socket is already connected */
	ENOTCONN        Errno = 107 /* Socket is not connected */
	ESHUTDOWN       Errno = 108 /* Can't send after socket shutdown */
	ETOOMANYREFS    Errno = 109
	ETIMEDOUT       Errno = 110 /* Connection timed out */
	ECONNREFUSED    Errno = 111 /* Connection refused */
	EHOSTDOWN       Errno = 112 /* Host is down */
	EHOSTUNREACH    Errno = 113 /* Host is unreachable */
	EALREADY        Errno = 114 /* Socket already connected */
	EINPROGRESS     Errno = 115 /* Connection already in progress */
	ESTALE          Errno = 116
	ENOTSUP         Errno = EOPNOTSUPP /* Not supported */
	ENOMEDIUM       Errno = 123        /* No medium (in tape drive) */
	ECANCELED       Errno = 125        /* Operation canceled. */
	ELBIN           Errno = 2048       /* Inode is remote (not really error) */
	EFTYPE          Errno = 2049       /* Inappropriate file type or format */
	ENMFILE         Errno = 2050       /* No more files */
	EPROCLIM        Errno = 2051
	ENOSHARE        Errno = 2052   /* No such host or network path */
	ECASECLASH      Errno = 2053   /* Filename exists with different case */
	EWOULDBLOCK     Errno = EAGAIN /* Operation would block */
)

// TODO: Auto-generate some day. (Hard-coded in binaries so not likely to change.)
var errorstr = [...]string{
	EPERM:           "Operation not permitted",
	ENOENT:          "No such file or directory",
	ESRCH:           "No such process",
	EINTR:           "Interrupted system call",
	EIO:             "I/O error",
	ENXIO:           "No such device or address",
	E2BIG:           "Argument list too long",
	ENOEXEC:         "Exec format error",
	EBADF:           "Bad file number",
	ECHILD:          "No child processes",
	EAGAIN:          "Try again",
	ENOMEM:          "Out of memory",
	EACCES:          "Permission denied",
	EFAULT:          "Bad address",
	EBUSY:           "Device or resource busy",
	EEXIST:          "File exists",
	EXDEV:           "Cross-device link",
	ENODEV:          "No such device",
	ENOTDIR:         "Not a directory",
	EISDIR:          "Is a directory",
	EINVAL:          "Invalid argument",
	ENFILE:          "File table overflow",
	EMFILE:          "Too many open files",
	ENOTTY:          "Not a typewriter",
	EFBIG:           "File too large",
	ENOSPC:          "No space left on device",
	ESPIPE:          "Illegal seek",
	EROFS:           "Read-only file system",
	EMLINK:          "Too many links",
	EPIPE:           "Broken pipe",
	ENAMETOOLONG:    "File name too long",
	ENOSYS:          "not implemented on " + runtime.GOOS,
	EDQUOT:          "Quota exceeded",
	EDOM:            "Math arg out of domain of func",
	ERANGE:          "Math result not representable",
	EDEADLK:         "Deadlock condition",
	ENOLCK:          "No record locks available",
	ENOTEMPTY:       "Directory not empty",
	ELOOP:           "Too many symbolic links",
	ENOMSG:          "No message of desired type",
	EIDRM:           "Identifier removed",
	ECHRNG:          "Channel number out of range",
	EL2NSYNC:        "Level 2 not synchronized",
	EL3HLT:          "Level 3 halted",
	EL3RST:          "Level 3 reset",
	ELNRNG:          "Link number out of range",
	EUNATCH:         "Protocol driver not attached",
	ENOCSI:          "No CSI structure available",
	EL2HLT:          "Level 2 halted",
	EBADE:           "Invalid exchange",
	EBADR:           "Invalid request descriptor",
	EXFULL:          "Exchange full",
	ENOANO:          "No anode",
	EBADRQC:         "Invalid request code",
	EBADSLT:         "Invalid slot",
	EBFONT:          "Bad font file fmt",
	ENOSTR:          "Device not a stream",
	ENODATA:         "No data (for no delay io)",
	ETIME:           "Timer expired",
	ENOSR:           "Out of streams resources",
	ENONET:          "Machine is not on the network",
	ENOPKG:          "Package not installed",
	EREMOTE:         "The object is remote",
	ENOLINK:         "The link has been severed",
	EADV:            "Advertise error",
	ESRMNT:          "Srmount error",
	ECOMM:           "Communication error on send",
	EPROTO:          "Protocol error",
	EMULTIHOP:       "Multihop attempted",
	EDOTDOT:         "Cross mount point (not really error)",
	EBADMSG:         "Trying to read unreadable message",
	EOVERFLOW:       "Value too large for defined data type",
	ENOTUNIQ:        "Given log. name not unique",
	EBADFD:          "f.d. invalid for this operation",
	EREMCHG:         "Remote address changed",
	ELIBACC:         "Can't access a needed shared lib",
	ELIBBAD:         "Accessing a corrupted shared lib",
	ELIBSCN:         ".lib section in a.out corrupted",
	ELIBMAX:         "Attempting to link in too many libs",
	ELIBEXEC:        "Attempting to exec a shared library",
	ENOTSOCK:        "Socket operation on non-socket",
	EDESTADDRREQ:    "Destination address required",
	EMSGSIZE:        "Message too long",
	EPROTOTYPE:      "Protocol wrong type for socket",
	ENOPROTOOPT:     "Protocol not available",
	EPROTONOSUPPORT: "Unknown protocol",
	ESOCKTNOSUPPORT: "Socket type not supported",
	EOPNOTSUPP:      "Operation not supported on transport endpoint",
	EPFNOSUPPORT:    "Protocol family not supported",
	EAFNOSUPPORT:    "Address family not supported by protocol family",
	EADDRINUSE:      "Address already in use",
	EADDRNOTAVAIL:   "Address not available",
	ENETDOWN:        "Network interface is not configured",
	ENETUNREACH:     "Network is unreachable",
	ECONNABORTED:    "Connection aborted",
	ECONNRESET:      "Connection reset by peer",
	ENOBUFS:         "No buffer space available",
	EISCONN:         "Socket is already connected",
	ENOTCONN:        "Socket is not connected",
	ESHUTDOWN:       "Can't send after socket shutdown",
	ETIMEDOUT:       "Connection timed out",
	ECONNREFUSED:    "Connection refused",
	EHOSTDOWN:       "Host is down",
	EHOSTUNREACH:    "Host is unreachable",
	EALREADY:        "Socket already connected",
	EINPROGRESS:     "Connection already in progress",
	ENOMEDIUM:       "No medium (in tape drive)",
	ECANCELED:       "Operation canceled.",
	ELBIN:           "Inode is remote (not really error)",
	EFTYPE:          "Inappropriate file type or format",
	ENMFILE:         "No more files",
	ENOSHARE:        "No such host or network path",
	ECASECLASH:      "Filename exists with different case",
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

var errnoByCode = map[string]Errno{
	"EPERM":           EPERM,
	"ENOENT":          ENOENT,
	"ESRCH":           ESRCH,
	"EINTR":           EINTR,
	"EIO":             EIO,
	"ENXIO":           ENXIO,
	"E2BIG":           E2BIG,
	"ENOEXEC":         ENOEXEC,
	"EBADF":           EBADF,
	"ECHILD":          ECHILD,
	"EAGAIN":          EAGAIN,
	"ENOMEM":          ENOMEM,
	"EACCES":          EACCES,
	"EFAULT":          EFAULT,
	"EBUSY":           EBUSY,
	"EEXIST":          EEXIST,
	"EXDEV":           EXDEV,
	"ENODEV":          ENODEV,
	"ENOTDIR":         ENOTDIR,
	"EISDIR":          EISDIR,
	"EINVAL":          EINVAL,
	"ENFILE":          ENFILE,
	"EMFILE":          EMFILE,
	"ENOTTY":          ENOTTY,
	"EFBIG":           EFBIG,
	"ENOSPC":          ENOSPC,
	"ESPIPE":          ESPIPE,
	"EROFS":           EROFS,
	"EMLINK":          EMLINK,
	"EPIPE":           EPIPE,
	"ENAMETOOLONG":    ENAMETOOLONG,
	"ENOSYS":          ENOSYS,
	"EDQUOT":          EDQUOT,
	"EDOM":            EDOM,
	"ERANGE":          ERANGE,
	"EDEADLK":         EDEADLK,
	"ENOLCK":          ENOLCK,
	"ENOTEMPTY":       ENOTEMPTY,
	"ELOOP":           ELOOP,
	"ENOMSG":          ENOMSG,
	"EIDRM":           EIDRM,
	"ECHRNG":          ECHRNG,
	"EL2NSYNC":        EL2NSYNC,
	"EL3HLT":          EL3HLT,
	"EL3RST":          EL3RST,
	"ELNRNG":          ELNRNG,
	"EUNATCH":         EUNATCH,
	"ENOCSI":          ENOCSI,
	"EL2HLT":          EL2HLT,
	"EBADE":           EBADE,
	"EBADR":           EBADR,
	"EXFULL":          EXFULL,
	"ENOANO":          ENOANO,
	"EBADRQC":         EBADRQC,
	"EBADSLT":         EBADSLT,
	"EDEADLOCK":       EDEADLOCK,
	"EBFONT":          EBFONT,
	"ENOSTR":          ENOSTR,
	"ENODATA":         ENODATA,
	"ETIME":           ETIME,
	"ENOSR":           ENOSR,
	"ENONET":          ENONET,
	"ENOPKG":          ENOPKG,
	"EREMOTE":         EREMOTE,
	"ENOLINK":         ENOLINK,
	"EADV":            EADV,
	"ESRMNT":          ESRMNT,
	"ECOMM":           ECOMM,
	"EPROTO":          EPROTO,
	"EMULTIHOP":       EMULTIHOP,
	"EDOTDOT":         EDOTDOT,
	"EBADMSG":         EBADMSG,
	"EOVERFLOW":       EOVERFLOW,
	"ENOTUNIQ":        ENOTUNIQ,
	"EBADFD":          EBADFD,
	"EREMCHG":         EREMCHG,
	"ELIBACC":         ELIBACC,
	"ELIBBAD":         ELIBBAD,
	"ELIBSCN":         ELIBSCN,
	"ELIBMAX":         ELIBMAX,
	"ELIBEXEC":        ELIBEXEC,
	"EILSEQ":          EILSEQ,
	"EUSERS":          EUSERS,
	"ENOTSOCK":        ENOTSOCK,
	"EDESTADDRREQ":    EDESTADDRREQ,
	"EMSGSIZE":        EMSGSIZE,
	"EPROTOTYPE":      EPROTOTYPE,
	"ENOPROTOOPT":     ENOPROTOOPT,
	"EPROTONOSUPPORT": EPROTONOSUPPORT,
	"ESOCKTNOSUPPORT": ESOCKTNOSUPPORT,
	"EOPNOTSUPP":      EOPNOTSUPP,
	"EPFNOSUPPORT":    EPFNOSUPPORT,
	"EAFNOSUPPORT":    EAFNOSUPPORT,
	"EADDRINUSE":      EADDRINUSE,
	"EADDRNOTAVAIL":   EADDRNOTAVAIL,
	"ENETDOWN":        ENETDOWN,
	"ENETUNREACH":     ENETUNREACH,
	"ENETRESET":       ENETRESET,
	"ECONNABORTED":    ECONNABORTED,
	"ECONNRESET":      ECONNRESET,
	"ENOBUFS":         ENOBUFS,
	"EISCONN":         EISCONN,
	"ENOTCONN":        ENOTCONN,
	"ESHUTDOWN":       ESHUTDOWN,
	"ETOOMANYREFS":    ETOOMANYREFS,
	"ETIMEDOUT":       ETIMEDOUT,
	"ECONNREFUSED":    ECONNREFUSED,
	"EHOSTDOWN":       EHOSTDOWN,
	"EHOSTUNREACH":    EHOSTUNREACH,
	"EALREADY":        EALREADY,
	"EINPROGRESS":     EINPROGRESS,
	"ESTALE":          ESTALE,
	"ENOTSUP":         ENOTSUP,
	"ENOMEDIUM":       ENOMEDIUM,
	"ECANCELED":       ECANCELED,
	"ELBIN":           ELBIN,
	"EFTYPE":          EFTYPE,
	"ENMFILE":         ENMFILE,
	"EPROCLIM":        EPROCLIM,
	"ENOSHARE":        ENOSHARE,
	"ECASECLASH":      ECASECLASH,
	"EWOULDBLOCK":     EWOULDBLOCK,
}
