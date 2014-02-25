enum {
	NSIG = 32,
	SI_USER = 1,

	// native_client/src/trusted/service_runtime/include/sys/errno.h
	// The errors are mainly copied from Linux.
	EPERM = 1,  /* Operation not permitted */
	ENOENT = 2,  /* No such file or directory */
	ESRCH = 3,  /* No such process */
	EINTR = 4,  /* Interrupted system call */
	EIO = 5,  /* I/O error */
	ENXIO = 6,  /* No such device or address */
	E2BIG = 7,  /* Argument list too long */
	ENOEXEC = 8,  /* Exec format error */
	EBADF = 9,  /* Bad file number */
	ECHILD = 10,  /* No child processes */
	EAGAIN = 11,  /* Try again */
	ENOMEM = 12,  /* Out of memory */
	EACCES = 13,  /* Permission denied */
	EFAULT = 14,  /* Bad address */
	EBUSY = 16,  /* Device or resource busy */
	EEXIST = 17,  /* File exists */
	EXDEV = 18,  /* Cross-device link */
	ENODEV = 19,  /* No such device */
	ENOTDIR = 20,  /* Not a directory */
	EISDIR = 21,  /* Is a directory */
	EINVAL = 22,  /* Invalid argument */
	ENFILE = 23,  /* File table overflow */
	EMFILE = 24,  /* Too many open files */
	ENOTTY = 25,  /* Not a typewriter */
	EFBIG = 27,  /* File too large */
	ENOSPC = 28,  /* No space left on device */
	ESPIPE = 29,  /* Illegal seek */
	EROFS = 30,  /* Read-only file system */
	EMLINK = 31,  /* Too many links */
	EPIPE = 32,  /* Broken pipe */
	ENAMETOOLONG = 36,  /* File name too long */
	ENOSYS = 38,  /* Function not implemented */
	EDQUOT = 122, /* Quota exceeded */
	EDOM = 33,   /* Math arg out of domain of func */
	ERANGE = 34, /* Math result not representable */
	EDEADLK = 35,  /* Deadlock condition */
	ENOLCK = 37, /* No record locks available */
	ENOTEMPTY = 39,  /* Directory not empty */
	ELOOP = 40,  /* Too many symbolic links */
	ENOMSG = 42, /* No message of desired type */
	EIDRM = 43,  /* Identifier removed */
	ECHRNG = 44, /* Channel number out of range */
	EL2NSYNC = 45, /* Level 2 not synchronized */
	EL3HLT = 46, /* Level 3 halted */
	EL3RST = 47, /* Level 3 reset */
	ELNRNG = 48, /* Link number out of range */
	EUNATCH = 49,  /* Protocol driver not attached */
	ENOCSI = 50, /* No CSI structure available */
	EL2HLT = 51, /* Level 2 halted */
	EBADE = 52,  /* Invalid exchange */
	EBADR = 53,  /* Invalid request descriptor */
	EXFULL = 54, /* Exchange full */
	ENOANO = 55, /* No anode */
	EBADRQC = 56,  /* Invalid request code */
	EBADSLT = 57,  /* Invalid slot */
	EDEADLOCK = EDEADLK,  /* File locking deadlock error */
	EBFONT = 59, /* Bad font file fmt */
	ENOSTR = 60, /* Device not a stream */
	ENODATA = 61,  /* No data (for no delay io) */
	ETIME = 62,  /* Timer expired */
	ENOSR = 63,  /* Out of streams resources */
	ENONET = 64, /* Machine is not on the network */
	ENOPKG = 65, /* Package not installed */
	EREMOTE = 66,  /* The object is remote */
	ENOLINK = 67,  /* The link has been severed */
	EADV = 68,   /* Advertise error */
	ESRMNT = 69, /* Srmount error */
	ECOMM = 70,  /* Communication error on send */
	EPROTO = 71, /* Protocol error */
	EMULTIHOP = 72,  /* Multihop attempted */
	EDOTDOT = 73,  /* Cross mount point (not really error) */
	EBADMSG = 74,  /* Trying to read unreadable message */
	EOVERFLOW = 75, /* Value too large for defined data type */
	ENOTUNIQ = 76, /* Given log. name not unique */
	EBADFD = 77, /* f.d. invalid for this operation */
	EREMCHG = 78,  /* Remote address changed */
	ELIBACC = 79,  /* Can't access a needed shared lib */
	ELIBBAD = 80,  /* Accessing a corrupted shared lib */
	ELIBSCN = 81,  /* .lib section in a.out corrupted */
	ELIBMAX = 82,  /* Attempting to link in too many libs */
	ELIBEXEC = 83, /* Attempting to exec a shared library */
	EILSEQ = 84,
	EUSERS = 87,
	ENOTSOCK = 88,  /* Socket operation on non-socket */
	EDESTADDRREQ = 89,  /* Destination address required */
	EMSGSIZE = 90,    /* Message too long */
	EPROTOTYPE = 91,  /* Protocol wrong type for socket */
	ENOPROTOOPT = 92, /* Protocol not available */
	EPROTONOSUPPORT = 93, /* Unknown protocol */
	ESOCKTNOSUPPORT = 94, /* Socket type not supported */
	EOPNOTSUPP = 95, /* Operation not supported on transport endpoint */
	EPFNOSUPPORT = 96, /* Protocol family not supported */
	EAFNOSUPPORT = 97, /* Address family not supported by protocol family */
	EADDRINUSE = 98,    /* Address already in use */
	EADDRNOTAVAIL = 99, /* Address not available */
	ENETDOWN = 100,    /* Network interface is not configured */
	ENETUNREACH = 101,   /* Network is unreachable */
	ENETRESET = 102,
	ECONNABORTED = 103,  /* Connection aborted */
	ECONNRESET = 104,  /* Connection reset by peer */
	ENOBUFS = 105, /* No buffer space available */
	EISCONN = 106,   /* Socket is already connected */
	ENOTCONN = 107,    /* Socket is not connected */
	ESHUTDOWN = 108, /* Can't send after socket shutdown */
	ETOOMANYREFS = 109,
	ETIMEDOUT = 110,   /* Connection timed out */
	ECONNREFUSED = 111,  /* Connection refused */
	EHOSTDOWN = 112,   /* Host is down */
	EHOSTUNREACH = 113,  /* Host is unreachable */
	EALREADY = 114,    /* Socket already connected */
	EINPROGRESS = 115,   /* Connection already in progress */
	ESTALE = 116,
	ENOTSUP = EOPNOTSUPP,   /* Not supported */
	ENOMEDIUM = 123,   /* No medium (in tape drive) */
	ECANCELED = 125, /* Operation canceled. */
	ELBIN = 2048,  /* Inode is remote (not really error) */
	EFTYPE = 2049,  /* Inappropriate file type or format */
	ENMFILE = 2050,  /* No more files */
	EPROCLIM = 2051,
	ENOSHARE = 2052,  /* No such host or network path */
	ECASECLASH = 2053,  /* Filename exists with different case */
	EWOULDBLOCK = EAGAIN,      /* Operation would block */

	// native_client/src/trusted/service_runtime/include/bits/mman.h.
	// NOTE: DO NOT USE native_client/src/shared/imc/nacl_imc_c.h.
	// Those MAP_*values are different from these.
	PROT_NONE	= 0x0,
	PROT_READ	= 0x1,
	PROT_WRITE	= 0x2,
	PROT_EXEC	= 0x4,

	MAP_SHARED	= 0x1,
	MAP_PRIVATE	= 0x2,
	MAP_FIXED	= 0x10,
	MAP_ANON	= 0x20,
};
typedef byte* kevent_udata;

int32	runtime·nacl_exception_stack(byte*, int32);
int32	runtime·nacl_exception_handler(void*, void*);
int32	runtime·nacl_sem_create(int32);
int32	runtime·nacl_sem_wait(int32);
int32	runtime·nacl_sem_post(int32);
int32	runtime·nacl_mutex_create(int32);
int32	runtime·nacl_mutex_lock(int32);
int32	runtime·nacl_mutex_trylock(int32);
int32	runtime·nacl_mutex_unlock(int32);
int32	runtime·nacl_cond_create(int32);
int32	runtime·nacl_cond_wait(int32, int32);
int32	runtime·nacl_cond_signal(int32);
int32	runtime·nacl_cond_broadcast(int32);
int32	runtime·nacl_cond_timed_wait_abs(int32, int32, Timespec*);
int32	runtime·nacl_thread_create(void*, void*, void*, void*);
int32	runtime·nacl_nanosleep(Timespec*, Timespec*);

void	runtime·sigpanic(void);
