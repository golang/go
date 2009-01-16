// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

export const (
	ENONE=0;
	EPERM=1;
	ENOENT=2;
	ESRCH=3;
	EINTR=4;
	EIO=5;
	ENXIO=6;
	E2BIG=7;
	ENOEXEC=8;
	EBADF=9;
	ECHILD=10;
	EDEADLK=11;
	ENOMEM=12;
	EACCES=13;
	EFAULT=14;
	ENOTBLK=15;
	EBUSY=16;
	EEXIST=17;
	EXDEV=18;
	ENODEV=19;
	ENOTDIR=20;
	EISDIR=21;
	EINVAL=22;
	ENFILE=23;
	EMFILE=24;
	ENOTTY=25;
	ETXTBSY=26;
	EFBIG=27;
	ENOSPC=28;
	ESPIPE=29;
	EROFS=30;
	EMLINK=31;
	EPIPE=32;
	EDOM=33;
	ERANGE=34;
	EAGAIN=35;
	EINPROGRESS=36;
	EALREADY=37;
	ENOTSOCK=38;
	EDESTADDRREQ=39;
	EMSGSIZE=40;
	EPROTOTYPE=41;
	ENOPROTOOPT=42;
	EPROTONOSUPPORT=43;
	ESOCKTNOSUPPORT=44;
	ENOTSUP=45;
	EPFNOSUPPORT=46;
	EAFNOSUPPORT=47;
	EADDRINUSE=48;
	EADDRNOTAVAIL=49;
	ENETDOWN=50;
	ENETUNREACH=51;
	ENETRESET=52;
	ECONNABORTED=53;
	ECONNRESET=54;
	ENOBUFS=55;
	EISCONN=56;
	ENOTCONN=57;
	ESHUTDOWN=58;
	ETOOMANYREFS=59;
	ETIMEDOUT=60;
	ECONNREFUSED=61;
	ELOOP=62;
	ENAMETOOLONG=63;
	EHOSTDOWN=64;
	EHOSTUNREACH=65;
	ENOTEMPTY=66;
	EPROCLIM=67;
	EUSERS=68;
	EDQUOT=69;
	ESTALE=70;
	EREMOTE=71;
	EBADRPC=72;
	ERPCMISMATCH=73;
	EPROGUNAVAIL=74;
	EPROGMISMATCH=75;
	EPROCUNAVAIL=76;
	ENOLCK=77;
	ENOSYS=78;
	EFTYPE=79;
	EAUTH=80;
	ENEEDAUTH=81;
	EPWROFF=82;
	EDEVERR=83;
	EOVERFLOW=84;
	EBADEXEC=85;
	EBADARCH=86;
	ESHLIBVERS=87;
	EBADMACHO=88;
	ECANCELED=89;
	EIDRM=90;
	ENOMSG=91;
	EILSEQ=92;
	ENOATTR=93;
	EBADMSG=94;
	EMULTIHOP=95;
	ENODATA=96;
	ENOLINK=97;
	ENOSR=98;
	ENOSTR=99;
	EPROTO=100;
	ETIME=101;
	EOPNOTSUPP=102;
	ELAST=103;
)

var error [ELAST]string;

func init(){
	error[ENONE] = "No error";
	error[EPERM] = "Operation not permitted";
	error[ENOENT] = "No such file or directory";
	error[ESRCH] = "No such process";
	error[EINTR] = "Interrupted system call";
	error[EIO] = "Input/output error";
	error[ENXIO] = "Device not configured";
	error[E2BIG] = "Argument list too long";
	error[ENOEXEC] = "Exec format error";
	error[EBADF] = "Bad file descriptor";
	error[ECHILD] = "No child processes";
	error[EDEADLK] = "Resource deadlock avoided";
	error[ENOMEM] = "Cannot allocate memory";
	error[EACCES] = "Permission denied";
	error[EFAULT] = "Bad address";
	error[ENOTBLK] = "Block device required";
	error[EBUSY] = "Device / Resource busy";
	error[EEXIST] = "File exists";
	error[EXDEV] = "Cross-device link";
	error[ENODEV] = "Operation not supported by device";
	error[ENOTDIR] = "Not a directory";
	error[EISDIR] = "Is a directory";
	error[EINVAL] = "Invalid argument";
	error[ENFILE] = "Too many open files in system";
	error[EMFILE] = "Too many open files";
	error[ENOTTY] = "Inappropriate ioctl for device";
	error[ETXTBSY] = "Text file busy";
	error[EFBIG] = "File too large";
	error[ENOSPC] = "No space left on device";
	error[ESPIPE] = "Illegal seek";
	error[EROFS] = "Read-only file system";
	error[EMLINK] = "Too many links";
	error[EPIPE] = "Broken pipe";
	error[EDOM] = "Numerical argument out of domain";
	error[ERANGE] = "Result too large";
	error[EAGAIN] = "Resource temporarily unavailable";
	error[EINPROGRESS] = "Operation now in progress";
	error[EALREADY] = "Operation already in progress";
	error[ENOTSOCK] = "Socket operation on non-socket";
	error[EDESTADDRREQ] = "Destination address required";
	error[EMSGSIZE] = "Message too long";
	error[EPROTOTYPE] = "Protocol wrong type for socket";
	error[ENOPROTOOPT] = "Protocol not available";
	error[EPROTONOSUPPORT] = "Protocol not supported";
	error[ESOCKTNOSUPPORT] = "Socket type not supported";
	error[ENOTSUP] = "Operation not supported";
	error[EPFNOSUPPORT] = "Protocol family not supported";
	error[EAFNOSUPPORT] = "Address family not supported by protocol family";
	error[EADDRINUSE] = "Address already in use";
	error[EADDRNOTAVAIL] = "Can't assign requested address";
	error[ENETDOWN] = "Network is down";
	error[ENETUNREACH] = "Network is unreachable";
	error[ENETRESET] = "Network dropped connection on reset";
	error[ECONNABORTED] = "Software caused connection abort";
	error[ECONNRESET] = "Connection reset by peer";
	error[ENOBUFS] = "No buffer space available";
	error[EISCONN] = "Socket is already connected";
	error[ENOTCONN] = "Socket is not connected";
	error[ESHUTDOWN] = "Can't send after socket shutdown";
	error[ETOOMANYREFS] = "Too many references: can't splice";
	error[ETIMEDOUT] = "Operation timed out";
	error[ECONNREFUSED] = "Connection refused";
	error[ELOOP] = "Too many levels of symbolic links";
	error[ENAMETOOLONG] = "File name too long";
	error[EHOSTDOWN] = "Host is down";
	error[EHOSTUNREACH] = "No route to host";
	error[ENOTEMPTY] = "Directory not empty";
	error[EPROCLIM] = "Too many processes";
	error[EUSERS] = "Too many users";
	error[EDQUOT] = "Disc quota exceeded";
	error[ESTALE] = "Stale NFS file handle";
	error[EREMOTE] = "Too many levels of remote in path";
	error[EBADRPC] = "RPC struct is bad";
	error[ERPCMISMATCH] = "RPC version wrong";
	error[EPROGUNAVAIL] = "RPC prog. not avail";
	error[EPROGMISMATCH] = "Program version wrong";
	error[EPROCUNAVAIL] = "Bad procedure for program";
	error[ENOLCK] = "No locks available";
	error[ENOSYS] = "Function not implemented";
	error[EFTYPE] = "Inappropriate file type or format";
	error[EAUTH] = "Authentication error";
	error[ENEEDAUTH] = "Need authenticator";
	error[EPWROFF] = "Device power is off";
	error[EDEVERR] = "Device error, e.g. paper out";
	error[EOVERFLOW] = "Value too large to be stored in data type";
	error[EBADEXEC] = "Bad executable";
	error[EBADARCH] = "Bad CPU type in executable";
	error[ESHLIBVERS] = "Shared library version mismatch";
	error[EBADMACHO] = "Malformed Macho file";
	error[ECANCELED] = "Operation canceled";
	error[EIDRM] = "Identifier removed";
	error[ENOMSG] = "No message of desired type";
	error[EILSEQ] = "Illegal byte sequence";
	error[ENOATTR] = "Attribute not found";
	error[EBADMSG] = "Bad message";
	error[EMULTIHOP] = "Reserved";
	error[ENODATA] = "No message available on STREAM";
	error[ENOLINK] = "Reserved";
	error[ENOSR] = "No STREAM resources";
	error[ENOSTR] = "Not a STREAM";
	error[EPROTO] = "Protocol error";
	error[ETIME] = "STREAM ioctl timeout";
	error[EOPNOTSUPP] = "Operation not supported on socket";
}

func str(val int64) string {  // do it here rather than with fmt to avoid dependency
	if val < 0 {
		return "-" + str(-val);
	}
	var buf [32]byte;  // big enough for int64
	i := len(buf)-1;
	for val >= 10 {
		buf[i] = byte(val%10 + '0');
		i--;
		val /= 10;
	}
	buf[i] = byte(val + '0');
	return string(buf)[i:len(buf)];
}

export func Errstr(errno int64) string {
	if errno < 0 || errno >= len(error) {
		return "Error " + str(errno)
	}
	return error[errno]
}
