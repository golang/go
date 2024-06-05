// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x

// Hand edited based on zerrors_linux_s390x.go
// TODO: auto-generate.

package unix

const (
	BRKINT                   = 0x0001
	CLOCAL                   = 0x1
	CLOCK_MONOTONIC          = 0x1
	CLOCK_PROCESS_CPUTIME_ID = 0x2
	CLOCK_REALTIME           = 0x0
	CLOCK_THREAD_CPUTIME_ID  = 0x3
	CLONE_NEWIPC             = 0x08000000
	CLONE_NEWNET             = 0x40000000
	CLONE_NEWNS              = 0x00020000
	CLONE_NEWPID             = 0x20000000
	CLONE_NEWUTS             = 0x04000000
	CLONE_PARENT             = 0x00008000
	CS8                      = 0x0030
	CSIZE                    = 0x0030
	ECHO                     = 0x00000008
	ECHONL                   = 0x00000001
	EFD_SEMAPHORE            = 0x00002000
	EFD_CLOEXEC              = 0x00001000
	EFD_NONBLOCK             = 0x00000004
	EPOLL_CLOEXEC            = 0x00001000
	EPOLL_CTL_ADD            = 0
	EPOLL_CTL_MOD            = 1
	EPOLL_CTL_DEL            = 2
	EPOLLRDNORM              = 0x0001
	EPOLLRDBAND              = 0x0002
	EPOLLIN                  = 0x0003
	EPOLLOUT                 = 0x0004
	EPOLLWRBAND              = 0x0008
	EPOLLPRI                 = 0x0010
	EPOLLERR                 = 0x0020
	EPOLLHUP                 = 0x0040
	EPOLLEXCLUSIVE           = 0x20000000
	EPOLLONESHOT             = 0x40000000
	FD_CLOEXEC               = 0x01
	FD_CLOFORK               = 0x02
	FD_SETSIZE               = 0x800
	FNDELAY                  = 0x04
	F_CLOSFD                 = 9
	F_CONTROL_CVT            = 13
	F_DUPFD                  = 0
	F_DUPFD2                 = 8
	F_GETFD                  = 1
	F_GETFL                  = 259
	F_GETLK                  = 5
	F_GETOWN                 = 10
	F_OK                     = 0x0
	F_RDLCK                  = 1
	F_SETFD                  = 2
	F_SETFL                  = 4
	F_SETLK                  = 6
	F_SETLKW                 = 7
	F_SETOWN                 = 11
	F_SETTAG                 = 12
	F_UNLCK                  = 3
	F_WRLCK                  = 2
	FSTYPE_ZFS               = 0xe9 //"Z"
	FSTYPE_HFS               = 0xc8 //"H"
	FSTYPE_NFS               = 0xd5 //"N"
	FSTYPE_TFS               = 0xe3 //"T"
	FSTYPE_AUTOMOUNT         = 0xc1 //"A"
	GRND_NONBLOCK            = 1
	GRND_RANDOM              = 2
	HUPCL                    = 0x0100 // Hang up on last close
	IN_CLOEXEC               = 0x00001000
	IN_NONBLOCK              = 0x00000004
	IN_ACCESS                = 0x00000001
	IN_MODIFY                = 0x00000002
	IN_ATTRIB                = 0x00000004
	IN_CLOSE_WRITE           = 0x00000008
	IN_CLOSE_NOWRITE         = 0x00000010
	IN_OPEN                  = 0x00000020
	IN_MOVED_FROM            = 0x00000040
	IN_MOVED_TO              = 0x00000080
	IN_CREATE                = 0x00000100
	IN_DELETE                = 0x00000200
	IN_DELETE_SELF           = 0x00000400
	IN_MOVE_SELF             = 0x00000800
	IN_UNMOUNT               = 0x00002000
	IN_Q_OVERFLOW            = 0x00004000
	IN_IGNORED               = 0x00008000
	IN_CLOSE                 = (IN_CLOSE_WRITE | IN_CLOSE_NOWRITE)
	IN_MOVE                  = (IN_MOVED_FROM | IN_MOVED_TO)
	IN_ALL_EVENTS            = (IN_ACCESS | IN_MODIFY | IN_ATTRIB |
		IN_CLOSE | IN_OPEN | IN_MOVE |
		IN_CREATE | IN_DELETE | IN_DELETE_SELF |
		IN_MOVE_SELF)
	IN_ONLYDIR                      = 0x01000000
	IN_DONT_FOLLOW                  = 0x02000000
	IN_EXCL_UNLINK                  = 0x04000000
	IN_MASK_CREATE                  = 0x10000000
	IN_MASK_ADD                     = 0x20000000
	IN_ISDIR                        = 0x40000000
	IN_ONESHOT                      = 0x80000000
	IP6F_MORE_FRAG                  = 0x0001
	IP6F_OFF_MASK                   = 0xfff8
	IP6F_RESERVED_MASK              = 0x0006
	IP6OPT_JUMBO                    = 0xc2
	IP6OPT_JUMBO_LEN                = 6
	IP6OPT_MUTABLE                  = 0x20
	IP6OPT_NSAP_ADDR                = 0xc3
	IP6OPT_PAD1                     = 0x00
	IP6OPT_PADN                     = 0x01
	IP6OPT_ROUTER_ALERT             = 0x05
	IP6OPT_TUNNEL_LIMIT             = 0x04
	IP6OPT_TYPE_DISCARD             = 0x40
	IP6OPT_TYPE_FORCEICMP           = 0x80
	IP6OPT_TYPE_ICMP                = 0xc0
	IP6OPT_TYPE_SKIP                = 0x00
	IP6_ALERT_AN                    = 0x0002
	IP6_ALERT_MLD                   = 0x0000
	IP6_ALERT_RSVP                  = 0x0001
	IPPORT_RESERVED                 = 1024
	IPPORT_USERRESERVED             = 5000
	IPPROTO_AH                      = 51
	SOL_AH                          = 51
	IPPROTO_DSTOPTS                 = 60
	SOL_DSTOPTS                     = 60
	IPPROTO_EGP                     = 8
	SOL_EGP                         = 8
	IPPROTO_ESP                     = 50
	SOL_ESP                         = 50
	IPPROTO_FRAGMENT                = 44
	SOL_FRAGMENT                    = 44
	IPPROTO_GGP                     = 2
	SOL_GGP                         = 2
	IPPROTO_HOPOPTS                 = 0
	SOL_HOPOPTS                     = 0
	IPPROTO_ICMP                    = 1
	SOL_ICMP                        = 1
	IPPROTO_ICMPV6                  = 58
	SOL_ICMPV6                      = 58
	IPPROTO_IDP                     = 22
	SOL_IDP                         = 22
	IPPROTO_IP                      = 0
	SOL_IP                          = 0
	IPPROTO_IPV6                    = 41
	SOL_IPV6                        = 41
	IPPROTO_MAX                     = 256
	SOL_MAX                         = 256
	IPPROTO_NONE                    = 59
	SOL_NONE                        = 59
	IPPROTO_PUP                     = 12
	SOL_PUP                         = 12
	IPPROTO_RAW                     = 255
	SOL_RAW                         = 255
	IPPROTO_ROUTING                 = 43
	SOL_ROUTING                     = 43
	IPPROTO_TCP                     = 6
	SOL_TCP                         = 6
	IPPROTO_UDP                     = 17
	SOL_UDP                         = 17
	IPV6_ADDR_PREFERENCES           = 32
	IPV6_CHECKSUM                   = 19
	IPV6_DONTFRAG                   = 29
	IPV6_DSTOPTS                    = 23
	IPV6_HOPLIMIT                   = 11
	IPV6_HOPOPTS                    = 22
	IPV6_JOIN_GROUP                 = 5
	IPV6_LEAVE_GROUP                = 6
	IPV6_MULTICAST_HOPS             = 9
	IPV6_MULTICAST_IF               = 7
	IPV6_MULTICAST_LOOP             = 4
	IPV6_NEXTHOP                    = 20
	IPV6_PATHMTU                    = 12
	IPV6_PKTINFO                    = 13
	IPV6_PREFER_SRC_CGA             = 0x10
	IPV6_PREFER_SRC_COA             = 0x02
	IPV6_PREFER_SRC_HOME            = 0x01
	IPV6_PREFER_SRC_NONCGA          = 0x20
	IPV6_PREFER_SRC_PUBLIC          = 0x08
	IPV6_PREFER_SRC_TMP             = 0x04
	IPV6_RECVDSTOPTS                = 28
	IPV6_RECVHOPLIMIT               = 14
	IPV6_RECVHOPOPTS                = 26
	IPV6_RECVPATHMTU                = 16
	IPV6_RECVPKTINFO                = 15
	IPV6_RECVRTHDR                  = 25
	IPV6_RECVTCLASS                 = 31
	IPV6_RTHDR                      = 21
	IPV6_RTHDRDSTOPTS               = 24
	IPV6_RTHDR_TYPE_0               = 0
	IPV6_TCLASS                     = 30
	IPV6_UNICAST_HOPS               = 3
	IPV6_USE_MIN_MTU                = 18
	IPV6_V6ONLY                     = 10
	IP_ADD_MEMBERSHIP               = 5
	IP_ADD_SOURCE_MEMBERSHIP        = 12
	IP_BLOCK_SOURCE                 = 10
	IP_DEFAULT_MULTICAST_LOOP       = 1
	IP_DEFAULT_MULTICAST_TTL        = 1
	IP_DROP_MEMBERSHIP              = 6
	IP_DROP_SOURCE_MEMBERSHIP       = 13
	IP_MAX_MEMBERSHIPS              = 20
	IP_MULTICAST_IF                 = 7
	IP_MULTICAST_LOOP               = 4
	IP_MULTICAST_TTL                = 3
	IP_OPTIONS                      = 1
	IP_PKTINFO                      = 101
	IP_RECVPKTINFO                  = 102
	IP_TOS                          = 2
	IP_TTL                          = 14
	IP_UNBLOCK_SOURCE               = 11
	ICMP6_FILTER                    = 1
	MCAST_INCLUDE                   = 0
	MCAST_EXCLUDE                   = 1
	MCAST_JOIN_GROUP                = 40
	MCAST_LEAVE_GROUP               = 41
	MCAST_JOIN_SOURCE_GROUP         = 42
	MCAST_LEAVE_SOURCE_GROUP        = 43
	MCAST_BLOCK_SOURCE              = 44
	MCAST_UNBLOCK_SOURCE            = 46
	ICANON                          = 0x0010
	ICRNL                           = 0x0002
	IEXTEN                          = 0x0020
	IGNBRK                          = 0x0004
	IGNCR                           = 0x0008
	INLCR                           = 0x0020
	ISIG                            = 0x0040
	ISTRIP                          = 0x0080
	IXON                            = 0x0200
	IXOFF                           = 0x0100
	LOCK_SH                         = 0x1
	LOCK_EX                         = 0x2
	LOCK_NB                         = 0x4
	LOCK_UN                         = 0x8
	POLLIN                          = 0x0003
	POLLOUT                         = 0x0004
	POLLPRI                         = 0x0010
	POLLERR                         = 0x0020
	POLLHUP                         = 0x0040
	POLLNVAL                        = 0x0080
	PROT_READ                       = 0x1 // mmap - page can be read
	PROT_WRITE                      = 0x2 // page can be written
	PROT_NONE                       = 0x4 // can't be accessed
	PROT_EXEC                       = 0x8 // can be executed
	MAP_PRIVATE                     = 0x1 // changes are private
	MAP_SHARED                      = 0x2 // changes are shared
	MAP_FIXED                       = 0x4 // place exactly
	__MAP_MEGA                      = 0x8
	__MAP_64                        = 0x10
	MAP_ANON                        = 0x20
	MAP_ANONYMOUS                   = 0x20
	MS_SYNC                         = 0x1 // msync - synchronous writes
	MS_ASYNC                        = 0x2 // asynchronous writes
	MS_INVALIDATE                   = 0x4 // invalidate mappings
	MS_BIND                         = 0x00001000
	MS_MOVE                         = 0x00002000
	MS_NOSUID                       = 0x00000002
	MS_PRIVATE                      = 0x00040000
	MS_REC                          = 0x00004000
	MS_REMOUNT                      = 0x00008000
	MS_RDONLY                       = 0x00000001
	MS_UNBINDABLE                   = 0x00020000
	MNT_DETACH                      = 0x00000004
	ZOSDSFS_SUPER_MAGIC             = 0x44534653 // zOS DSFS
	NFS_SUPER_MAGIC                 = 0x6969     // NFS
	NSFS_MAGIC                      = 0x6e736673 // PROCNS
	PROC_SUPER_MAGIC                = 0x9fa0     // proc FS
	ZOSTFS_SUPER_MAGIC              = 0x544653   // zOS TFS
	ZOSUFS_SUPER_MAGIC              = 0x554653   // zOS UFS
	ZOSZFS_SUPER_MAGIC              = 0x5A4653   // zOS ZFS
	MTM_RDONLY                      = 0x80000000
	MTM_RDWR                        = 0x40000000
	MTM_UMOUNT                      = 0x10000000
	MTM_IMMED                       = 0x08000000
	MTM_FORCE                       = 0x04000000
	MTM_DRAIN                       = 0x02000000
	MTM_RESET                       = 0x01000000
	MTM_SAMEMODE                    = 0x00100000
	MTM_UNQSEFORCE                  = 0x00040000
	MTM_NOSUID                      = 0x00000400
	MTM_SYNCHONLY                   = 0x00000200
	MTM_REMOUNT                     = 0x00000100
	MTM_NOSECURITY                  = 0x00000080
	NFDBITS                         = 0x20
	ONLRET                          = 0x0020 // NL performs CR function
	O_ACCMODE                       = 0x03
	O_APPEND                        = 0x08
	O_ASYNCSIG                      = 0x0200
	O_CREAT                         = 0x80
	O_DIRECT                        = 0x00002000
	O_NOFOLLOW                      = 0x00004000
	O_DIRECTORY                     = 0x00008000
	O_PATH                          = 0x00080000
	O_CLOEXEC                       = 0x00001000
	O_EXCL                          = 0x40
	O_GETFL                         = 0x0F
	O_LARGEFILE                     = 0x0400
	O_NDELAY                        = 0x4
	O_NONBLOCK                      = 0x04
	O_RDONLY                        = 0x02
	O_RDWR                          = 0x03
	O_SYNC                          = 0x0100
	O_TRUNC                         = 0x10
	O_WRONLY                        = 0x01
	O_NOCTTY                        = 0x20
	OPOST                           = 0x0001
	ONLCR                           = 0x0004
	PARENB                          = 0x0200
	PARMRK                          = 0x0400
	QUERYCVT                        = 3
	RUSAGE_CHILDREN                 = -0x1
	RUSAGE_SELF                     = 0x0 // RUSAGE_THREAD unsupported on z/OS
	SEEK_CUR                        = 1
	SEEK_END                        = 2
	SEEK_SET                        = 0
	SETAUTOCVTALL                   = 5
	SETAUTOCVTON                    = 2
	SETCVTALL                       = 4
	SETCVTOFF                       = 0
	SETCVTON                        = 1
	AF_APPLETALK                    = 16
	AF_CCITT                        = 10
	AF_CHAOS                        = 5
	AF_DATAKIT                      = 9
	AF_DLI                          = 13
	AF_ECMA                         = 8
	AF_HYLINK                       = 15
	AF_IMPLINK                      = 3
	AF_INET                         = 2
	AF_INET6                        = 19
	AF_INTF                         = 20
	AF_IUCV                         = 17
	AF_LAT                          = 14
	AF_LINK                         = 18
	AF_LOCAL                        = AF_UNIX // AF_LOCAL is an alias for AF_UNIX
	AF_MAX                          = 30
	AF_NBS                          = 7
	AF_NDD                          = 23
	AF_NETWARE                      = 22
	AF_NS                           = 6
	AF_PUP                          = 4
	AF_RIF                          = 21
	AF_ROUTE                        = 20
	AF_SNA                          = 11
	AF_UNIX                         = 1
	AF_UNSPEC                       = 0
	IBMTCP_IMAGE                    = 1
	MSG_ACK_EXPECTED                = 0x10
	MSG_ACK_GEN                     = 0x40
	MSG_ACK_TIMEOUT                 = 0x20
	MSG_CONNTERM                    = 0x80
	MSG_CTRUNC                      = 0x20
	MSG_DONTROUTE                   = 0x4
	MSG_EOF                         = 0x8000
	MSG_EOR                         = 0x8
	MSG_MAXIOVLEN                   = 16
	MSG_NONBLOCK                    = 0x4000
	MSG_OOB                         = 0x1
	MSG_PEEK                        = 0x2
	MSG_TRUNC                       = 0x10
	MSG_WAITALL                     = 0x40
	PRIO_PROCESS                    = 1
	PRIO_PGRP                       = 2
	PRIO_USER                       = 3
	RLIMIT_CPU                      = 0
	RLIMIT_FSIZE                    = 1
	RLIMIT_DATA                     = 2
	RLIMIT_STACK                    = 3
	RLIMIT_CORE                     = 4
	RLIMIT_AS                       = 5
	RLIMIT_NOFILE                   = 6
	RLIMIT_MEMLIMIT                 = 7
	RLIMIT_MEMLOCK                  = 0x8
	RLIM_INFINITY                   = 2147483647
	SCHED_FIFO                      = 0x2
	SCM_CREDENTIALS                 = 0x2
	SCM_RIGHTS                      = 0x01
	SF_CLOSE                        = 0x00000002
	SF_REUSE                        = 0x00000001
	SHM_RND                         = 0x2
	SHM_RDONLY                      = 0x1
	SHMLBA                          = 0x1000
	IPC_STAT                        = 0x3
	IPC_SET                         = 0x2
	IPC_RMID                        = 0x1
	IPC_PRIVATE                     = 0x0
	IPC_CREAT                       = 0x1000000
	__IPC_MEGA                      = 0x4000000
	__IPC_SHAREAS                   = 0x20000000
	__IPC_BELOWBAR                  = 0x10000000
	IPC_EXCL                        = 0x2000000
	__IPC_GIGA                      = 0x8000000
	SHUT_RD                         = 0
	SHUT_RDWR                       = 2
	SHUT_WR                         = 1
	SOCK_CLOEXEC                    = 0x00001000
	SOCK_CONN_DGRAM                 = 6
	SOCK_DGRAM                      = 2
	SOCK_NONBLOCK                   = 0x800
	SOCK_RAW                        = 3
	SOCK_RDM                        = 4
	SOCK_SEQPACKET                  = 5
	SOCK_STREAM                     = 1
	SOL_SOCKET                      = 0xffff
	SOMAXCONN                       = 10
	SO_ACCEPTCONN                   = 0x0002
	SO_ACCEPTECONNABORTED           = 0x0006
	SO_ACKNOW                       = 0x7700
	SO_BROADCAST                    = 0x0020
	SO_BULKMODE                     = 0x8000
	SO_CKSUMRECV                    = 0x0800
	SO_CLOSE                        = 0x01
	SO_CLUSTERCONNTYPE              = 0x00004001
	SO_CLUSTERCONNTYPE_INTERNAL     = 8
	SO_CLUSTERCONNTYPE_NOCONN       = 0
	SO_CLUSTERCONNTYPE_NONE         = 1
	SO_CLUSTERCONNTYPE_SAME_CLUSTER = 2
	SO_CLUSTERCONNTYPE_SAME_IMAGE   = 4
	SO_DEBUG                        = 0x0001
	SO_DONTROUTE                    = 0x0010
	SO_ERROR                        = 0x1007
	SO_IGNOREINCOMINGPUSH           = 0x1
	SO_IGNORESOURCEVIPA             = 0x0002
	SO_KEEPALIVE                    = 0x0008
	SO_LINGER                       = 0x0080
	SO_NONBLOCKLOCAL                = 0x8001
	SO_NOREUSEADDR                  = 0x1000
	SO_OOBINLINE                    = 0x0100
	SO_OPTACK                       = 0x8004
	SO_OPTMSS                       = 0x8003
	SO_RCVBUF                       = 0x1002
	SO_RCVLOWAT                     = 0x1004
	SO_RCVTIMEO                     = 0x1006
	SO_REUSEADDR                    = 0x0004
	SO_REUSEPORT                    = 0x0200
	SO_SECINFO                      = 0x00004002
	SO_SET                          = 0x0200
	SO_SNDBUF                       = 0x1001
	SO_SNDLOWAT                     = 0x1003
	SO_SNDTIMEO                     = 0x1005
	SO_TYPE                         = 0x1008
	SO_UNSET                        = 0x0400
	SO_USELOOPBACK                  = 0x0040
	SO_USE_IFBUFS                   = 0x0400
	S_ISUID                         = 0x0800
	S_ISGID                         = 0x0400
	S_ISVTX                         = 0x0200
	S_IRUSR                         = 0x0100
	S_IWUSR                         = 0x0080
	S_IXUSR                         = 0x0040
	S_IRWXU                         = 0x01C0
	S_IRGRP                         = 0x0020
	S_IWGRP                         = 0x0010
	S_IXGRP                         = 0x0008
	S_IRWXG                         = 0x0038
	S_IROTH                         = 0x0004
	S_IWOTH                         = 0x0002
	S_IXOTH                         = 0x0001
	S_IRWXO                         = 0x0007
	S_IREAD                         = S_IRUSR
	S_IWRITE                        = S_IWUSR
	S_IEXEC                         = S_IXUSR
	S_IFDIR                         = 0x01000000
	S_IFCHR                         = 0x02000000
	S_IFREG                         = 0x03000000
	S_IFFIFO                        = 0x04000000
	S_IFIFO                         = 0x04000000
	S_IFLNK                         = 0x05000000
	S_IFBLK                         = 0x06000000
	S_IFSOCK                        = 0x07000000
	S_IFVMEXTL                      = 0xFE000000
	S_IFVMEXTL_EXEC                 = 0x00010000
	S_IFVMEXTL_DATA                 = 0x00020000
	S_IFVMEXTL_MEL                  = 0x00030000
	S_IFEXTL                        = 0x00000001
	S_IFPROGCTL                     = 0x00000002
	S_IFAPFCTL                      = 0x00000004
	S_IFNOSHARE                     = 0x00000008
	S_IFSHARELIB                    = 0x00000010
	S_IFMT                          = 0xFF000000
	S_IFMST                         = 0x00FF0000
	TCP_KEEPALIVE                   = 0x8
	TCP_NODELAY                     = 0x1
	TIOCGWINSZ                      = 0x4008a368
	TIOCSWINSZ                      = 0x8008a367
	TIOCSBRK                        = 0x2000a77b
	TIOCCBRK                        = 0x2000a77a
	TIOCSTI                         = 0x8001a772
	TIOCGPGRP                       = 0x4004a777 // _IOR(167, 119, int)
	TCSANOW                         = 0
	TCSETS                          = 0 // equivalent to TCSANOW for tcsetattr
	TCSADRAIN                       = 1
	TCSETSW                         = 1 // equivalent to TCSADRAIN for tcsetattr
	TCSAFLUSH                       = 2
	TCSETSF                         = 2 // equivalent to TCSAFLUSH for tcsetattr
	TCGETS                          = 3 // not defined in ioctl.h -- zos golang only
	TCIFLUSH                        = 0
	TCOFLUSH                        = 1
	TCIOFLUSH                       = 2
	TCOOFF                          = 0
	TCOON                           = 1
	TCIOFF                          = 2
	TCION                           = 3
	TIOCSPGRP                       = 0x8004a776
	TIOCNOTTY                       = 0x2000a771
	TIOCEXCL                        = 0x2000a70d
	TIOCNXCL                        = 0x2000a70e
	TIOCGETD                        = 0x4004a700
	TIOCSETD                        = 0x8004a701
	TIOCPKT                         = 0x8004a770
	TIOCSTOP                        = 0x2000a76f
	TIOCSTART                       = 0x2000a76e
	TIOCUCNTL                       = 0x8004a766
	TIOCREMOTE                      = 0x8004a769
	TIOCMGET                        = 0x4004a76a
	TIOCMSET                        = 0x8004a76d
	TIOCMBIC                        = 0x8004a76b
	TIOCMBIS                        = 0x8004a76c
	VINTR                           = 0
	VQUIT                           = 1
	VERASE                          = 2
	VKILL                           = 3
	VEOF                            = 4
	VEOL                            = 5
	VMIN                            = 6
	VSTART                          = 7
	VSTOP                           = 8
	VSUSP                           = 9
	VTIME                           = 10
	WCONTINUED                      = 0x4
	WEXITED                         = 0x8
	WNOHANG                         = 0x1
	WNOWAIT                         = 0x20
	WSTOPPED                        = 0x10
	WUNTRACED                       = 0x2
	_BPX_SWAP                       = 1
	_BPX_NONSWAP                    = 2
	MCL_CURRENT                     = 1  // for Linux compatibility -- no zos semantics
	MCL_FUTURE                      = 2  // for Linux compatibility -- no zos semantics
	MCL_ONFAULT                     = 3  // for Linux compatibility -- no zos semantics
	MADV_NORMAL                     = 0  // for Linux compatibility -- no zos semantics
	MADV_RANDOM                     = 1  // for Linux compatibility -- no zos semantics
	MADV_SEQUENTIAL                 = 2  // for Linux compatibility -- no zos semantics
	MADV_WILLNEED                   = 3  // for Linux compatibility -- no zos semantics
	MADV_REMOVE                     = 4  // for Linux compatibility -- no zos semantics
	MADV_DONTFORK                   = 5  // for Linux compatibility -- no zos semantics
	MADV_DOFORK                     = 6  // for Linux compatibility -- no zos semantics
	MADV_HWPOISON                   = 7  // for Linux compatibility -- no zos semantics
	MADV_MERGEABLE                  = 8  // for Linux compatibility -- no zos semantics
	MADV_UNMERGEABLE                = 9  // for Linux compatibility -- no zos semantics
	MADV_SOFT_OFFLINE               = 10 // for Linux compatibility -- no zos semantics
	MADV_HUGEPAGE                   = 11 // for Linux compatibility -- no zos semantics
	MADV_NOHUGEPAGE                 = 12 // for Linux compatibility -- no zos semantics
	MADV_DONTDUMP                   = 13 // for Linux compatibility -- no zos semantics
	MADV_DODUMP                     = 14 // for Linux compatibility -- no zos semantics
	MADV_FREE                       = 15 // for Linux compatibility -- no zos semantics
	MADV_WIPEONFORK                 = 16 // for Linux compatibility -- no zos semantics
	MADV_KEEPONFORK                 = 17 // for Linux compatibility -- no zos semantics
	AT_SYMLINK_FOLLOW               = 0x400
	AT_SYMLINK_NOFOLLOW             = 0x100
	XATTR_CREATE                    = 0x1
	XATTR_REPLACE                   = 0x2
	P_PID                           = 0
	P_PGID                          = 1
	P_ALL                           = 2
	PR_SET_NAME                     = 15
	PR_GET_NAME                     = 16
	PR_SET_NO_NEW_PRIVS             = 38
	PR_GET_NO_NEW_PRIVS             = 39
	PR_SET_DUMPABLE                 = 4
	PR_GET_DUMPABLE                 = 3
	PR_SET_PDEATHSIG                = 1
	PR_GET_PDEATHSIG                = 2
	PR_SET_CHILD_SUBREAPER          = 36
	PR_GET_CHILD_SUBREAPER          = 37
	AT_FDCWD                        = -100
	AT_EACCESS                      = 0x200
	AT_EMPTY_PATH                   = 0x1000
	AT_REMOVEDIR                    = 0x200
	RENAME_NOREPLACE                = 1 << 0
)

const (
	EDOM               = Errno(1)
	ERANGE             = Errno(2)
	EACCES             = Errno(111)
	EAGAIN             = Errno(112)
	EBADF              = Errno(113)
	EBUSY              = Errno(114)
	ECHILD             = Errno(115)
	EDEADLK            = Errno(116)
	EEXIST             = Errno(117)
	EFAULT             = Errno(118)
	EFBIG              = Errno(119)
	EINTR              = Errno(120)
	EINVAL             = Errno(121)
	EIO                = Errno(122)
	EISDIR             = Errno(123)
	EMFILE             = Errno(124)
	EMLINK             = Errno(125)
	ENAMETOOLONG       = Errno(126)
	ENFILE             = Errno(127)
	ENOATTR            = Errno(265)
	ENODEV             = Errno(128)
	ENOENT             = Errno(129)
	ENOEXEC            = Errno(130)
	ENOLCK             = Errno(131)
	ENOMEM             = Errno(132)
	ENOSPC             = Errno(133)
	ENOSYS             = Errno(134)
	ENOTDIR            = Errno(135)
	ENOTEMPTY          = Errno(136)
	ENOTTY             = Errno(137)
	ENXIO              = Errno(138)
	EPERM              = Errno(139)
	EPIPE              = Errno(140)
	EROFS              = Errno(141)
	ESPIPE             = Errno(142)
	ESRCH              = Errno(143)
	EXDEV              = Errno(144)
	E2BIG              = Errno(145)
	ELOOP              = Errno(146)
	EILSEQ             = Errno(147)
	ENODATA            = Errno(148)
	EOVERFLOW          = Errno(149)
	EMVSNOTUP          = Errno(150)
	ECMSSTORAGE        = Errno(151)
	EMVSDYNALC         = Errno(151)
	EMVSCVAF           = Errno(152)
	EMVSCATLG          = Errno(153)
	ECMSINITIAL        = Errno(156)
	EMVSINITIAL        = Errno(156)
	ECMSERR            = Errno(157)
	EMVSERR            = Errno(157)
	EMVSPARM           = Errno(158)
	ECMSPFSFILE        = Errno(159)
	EMVSPFSFILE        = Errno(159)
	EMVSBADCHAR        = Errno(160)
	ECMSPFSPERM        = Errno(162)
	EMVSPFSPERM        = Errno(162)
	EMVSSAFEXTRERR     = Errno(163)
	EMVSSAF2ERR        = Errno(164)
	EMVSTODNOTSET      = Errno(165)
	EMVSPATHOPTS       = Errno(166)
	EMVSNORTL          = Errno(167)
	EMVSEXPIRE         = Errno(168)
	EMVSPASSWORD       = Errno(169)
	EMVSWLMERROR       = Errno(170)
	EMVSCPLERROR       = Errno(171)
	EMVSARMERROR       = Errno(172)
	ELENOFORK          = Errno(200)
	ELEMSGERR          = Errno(201)
	EFPMASKINV         = Errno(202)
	EFPMODEINV         = Errno(203)
	EBUFLEN            = Errno(227)
	EEXTLINK           = Errno(228)
	ENODD              = Errno(229)
	ECMSESMERR         = Errno(230)
	ECPERR             = Errno(231)
	ELEMULTITHREAD     = Errno(232)
	ELEFENCE           = Errno(244)
	EBADDATA           = Errno(245)
	EUNKNOWN           = Errno(246)
	ENOTSUP            = Errno(247)
	EBADNAME           = Errno(248)
	ENOTSAFE           = Errno(249)
	ELEMULTITHREADFORK = Errno(257)
	ECUNNOENV          = Errno(258)
	ECUNNOCONV         = Errno(259)
	ECUNNOTALIGNED     = Errno(260)
	ECUNERR            = Errno(262)
	EIBMBADCALL        = Errno(1000)
	EIBMBADPARM        = Errno(1001)
	EIBMSOCKOUTOFRANGE = Errno(1002)
	EIBMSOCKINUSE      = Errno(1003)
	EIBMIUCVERR        = Errno(1004)
	EOFFLOADboxERROR   = Errno(1005)
	EOFFLOADboxRESTART = Errno(1006)
	EOFFLOADboxDOWN    = Errno(1007)
	EIBMCONFLICT       = Errno(1008)
	EIBMCANCELLED      = Errno(1009)
	EIBMBADTCPNAME     = Errno(1011)
	ENOTBLK            = Errno(1100)
	ETXTBSY            = Errno(1101)
	EWOULDBLOCK        = Errno(1102)
	EINPROGRESS        = Errno(1103)
	EALREADY           = Errno(1104)
	ENOTSOCK           = Errno(1105)
	EDESTADDRREQ       = Errno(1106)
	EMSGSIZE           = Errno(1107)
	EPROTOTYPE         = Errno(1108)
	ENOPROTOOPT        = Errno(1109)
	EPROTONOSUPPORT    = Errno(1110)
	ESOCKTNOSUPPORT    = Errno(1111)
	EOPNOTSUPP         = Errno(1112)
	EPFNOSUPPORT       = Errno(1113)
	EAFNOSUPPORT       = Errno(1114)
	EADDRINUSE         = Errno(1115)
	EADDRNOTAVAIL      = Errno(1116)
	ENETDOWN           = Errno(1117)
	ENETUNREACH        = Errno(1118)
	ENETRESET          = Errno(1119)
	ECONNABORTED       = Errno(1120)
	ECONNRESET         = Errno(1121)
	ENOBUFS            = Errno(1122)
	EISCONN            = Errno(1123)
	ENOTCONN           = Errno(1124)
	ESHUTDOWN          = Errno(1125)
	ETOOMANYREFS       = Errno(1126)
	ETIMEDOUT          = Errno(1127)
	ECONNREFUSED       = Errno(1128)
	EHOSTDOWN          = Errno(1129)
	EHOSTUNREACH       = Errno(1130)
	EPROCLIM           = Errno(1131)
	EUSERS             = Errno(1132)
	EDQUOT             = Errno(1133)
	ESTALE             = Errno(1134)
	EREMOTE            = Errno(1135)
	ENOSTR             = Errno(1136)
	ETIME              = Errno(1137)
	ENOSR              = Errno(1138)
	ENOMSG             = Errno(1139)
	EBADMSG            = Errno(1140)
	EIDRM              = Errno(1141)
	ENONET             = Errno(1142)
	ERREMOTE           = Errno(1143)
	ENOLINK            = Errno(1144)
	EADV               = Errno(1145)
	ESRMNT             = Errno(1146)
	ECOMM              = Errno(1147)
	EPROTO             = Errno(1148)
	EMULTIHOP          = Errno(1149)
	EDOTDOT            = Errno(1150)
	EREMCHG            = Errno(1151)
	ECANCELED          = Errno(1152)
	EINTRNODATA        = Errno(1159)
	ENOREUSE           = Errno(1160)
	ENOMOVE            = Errno(1161)
)

// Signals
const (
	SIGHUP    = Signal(1)
	SIGINT    = Signal(2)
	SIGABRT   = Signal(3)
	SIGILL    = Signal(4)
	SIGPOLL   = Signal(5)
	SIGURG    = Signal(6)
	SIGSTOP   = Signal(7)
	SIGFPE    = Signal(8)
	SIGKILL   = Signal(9)
	SIGBUS    = Signal(10)
	SIGSEGV   = Signal(11)
	SIGSYS    = Signal(12)
	SIGPIPE   = Signal(13)
	SIGALRM   = Signal(14)
	SIGTERM   = Signal(15)
	SIGUSR1   = Signal(16)
	SIGUSR2   = Signal(17)
	SIGABND   = Signal(18)
	SIGCONT   = Signal(19)
	SIGCHLD   = Signal(20)
	SIGTTIN   = Signal(21)
	SIGTTOU   = Signal(22)
	SIGIO     = Signal(23)
	SIGQUIT   = Signal(24)
	SIGTSTP   = Signal(25)
	SIGTRAP   = Signal(26)
	SIGIOERR  = Signal(27)
	SIGWINCH  = Signal(28)
	SIGXCPU   = Signal(29)
	SIGXFSZ   = Signal(30)
	SIGVTALRM = Signal(31)
	SIGPROF   = Signal(32)
	SIGDANGER = Signal(33)
	SIGTHSTOP = Signal(34)
	SIGTHCONT = Signal(35)
	SIGTRACE  = Signal(37)
	SIGDCE    = Signal(38)
	SIGDUMP   = Signal(39)
)

// Error table
var errorList = [...]struct {
	num  Errno
	name string
	desc string
}{
	{1, "EDC5001I", "A domain error occurred."},
	{2, "EDC5002I", "A range error occurred."},
	{111, "EDC5111I", "Permission denied."},
	{112, "EDC5112I", "Resource temporarily unavailable."},
	{113, "EDC5113I", "Bad file descriptor."},
	{114, "EDC5114I", "Resource busy."},
	{115, "EDC5115I", "No child processes."},
	{116, "EDC5116I", "Resource deadlock avoided."},
	{117, "EDC5117I", "File exists."},
	{118, "EDC5118I", "Incorrect address."},
	{119, "EDC5119I", "File too large."},
	{120, "EDC5120I", "Interrupted function call."},
	{121, "EDC5121I", "Invalid argument."},
	{122, "EDC5122I", "Input/output error."},
	{123, "EDC5123I", "Is a directory."},
	{124, "EDC5124I", "Too many open files."},
	{125, "EDC5125I", "Too many links."},
	{126, "EDC5126I", "Filename too long."},
	{127, "EDC5127I", "Too many open files in system."},
	{128, "EDC5128I", "No such device."},
	{129, "EDC5129I", "No such file or directory."},
	{130, "EDC5130I", "Exec format error."},
	{131, "EDC5131I", "No locks available."},
	{132, "EDC5132I", "Not enough memory."},
	{133, "EDC5133I", "No space left on device."},
	{134, "EDC5134I", "Function not implemented."},
	{135, "EDC5135I", "Not a directory."},
	{136, "EDC5136I", "Directory not empty."},
	{137, "EDC5137I", "Inappropriate I/O control operation."},
	{138, "EDC5138I", "No such device or address."},
	{139, "EDC5139I", "Operation not permitted."},
	{140, "EDC5140I", "Broken pipe."},
	{141, "EDC5141I", "Read-only file system."},
	{142, "EDC5142I", "Invalid seek."},
	{143, "EDC5143I", "No such process."},
	{144, "EDC5144I", "Improper link."},
	{145, "EDC5145I", "The parameter list is too long, or the message to receive was too large for the buffer."},
	{146, "EDC5146I", "Too many levels of symbolic links."},
	{147, "EDC5147I", "Illegal byte sequence."},
	{148, "EDC5148I", "The named attribute or data not available."},
	{149, "EDC5149I", "Value Overflow Error."},
	{150, "EDC5150I", "UNIX System Services is not active."},
	{151, "EDC5151I", "Dynamic allocation error."},
	{152, "EDC5152I", "Common VTOC access facility (CVAF) error."},
	{153, "EDC5153I", "Catalog obtain error."},
	{156, "EDC5156I", "Process initialization error."},
	{157, "EDC5157I", "An internal error has occurred."},
	{158, "EDC5158I", "Bad parameters were passed to the service."},
	{159, "EDC5159I", "The Physical File System encountered a permanent file error."},
	{160, "EDC5160I", "Bad character in environment variable name."},
	{162, "EDC5162I", "The Physical File System encountered a system error."},
	{163, "EDC5163I", "SAF/RACF extract error."},
	{164, "EDC5164I", "SAF/RACF error."},
	{165, "EDC5165I", "System TOD clock not set."},
	{166, "EDC5166I", "Access mode argument on function call conflicts with PATHOPTS parameter on JCL DD statement."},
	{167, "EDC5167I", "Access to the UNIX System Services version of the C RTL is denied."},
	{168, "EDC5168I", "Password has expired."},
	{169, "EDC5169I", "Password is invalid."},
	{170, "EDC5170I", "An error was encountered with WLM."},
	{171, "EDC5171I", "An error was encountered with CPL."},
	{172, "EDC5172I", "An error was encountered with Application Response Measurement (ARM) component."},
	{200, "EDC5200I", "The application contains a Language Environment member language that cannot tolerate a fork()."},
	{201, "EDC5201I", "The Language Environment message file was not found in the hierarchical file system."},
	{202, "EDC5202E", "DLL facilities are not supported under SPC environment."},
	{203, "EDC5203E", "DLL facilities are not supported under POSIX environment."},
	{227, "EDC5227I", "Buffer is not long enough to contain a path definition"},
	{228, "EDC5228I", "The file referred to is an external link"},
	{229, "EDC5229I", "No path definition for ddname in effect"},
	{230, "EDC5230I", "ESM error."},
	{231, "EDC5231I", "CP or the external security manager had an error"},
	{232, "EDC5232I", "The function failed because it was invoked from a multithread environment."},
	{244, "EDC5244I", "The program, module or DLL is not supported in this environment."},
	{245, "EDC5245I", "Data is not valid."},
	{246, "EDC5246I", "Unknown system state."},
	{247, "EDC5247I", "Operation not supported."},
	{248, "EDC5248I", "The object name specified is not correct."},
	{249, "EDC5249I", "The function is not allowed."},
	{257, "EDC5257I", "Function cannot be called in the child process of a fork() from a multithreaded process until exec() is called."},
	{258, "EDC5258I", "A CUN_RS_NO_UNI_ENV error was issued by Unicode Services."},
	{259, "EDC5259I", "A CUN_RS_NO_CONVERSION error was issued by Unicode Services."},
	{260, "EDC5260I", "A CUN_RS_TABLE_NOT_ALIGNED error was issued by Unicode Services."},
	{262, "EDC5262I", "An iconv() function encountered an unexpected error while using Unicode Services."},
	{265, "EDC5265I", "The named attribute not available."},
	{1000, "EDC8000I", "A bad socket-call constant was found in the IUCV header."},
	{1001, "EDC8001I", "An error was found in the IUCV header."},
	{1002, "EDC8002I", "A socket descriptor is out of range."},
	{1003, "EDC8003I", "A socket descriptor is in use."},
	{1004, "EDC8004I", "Request failed because of an IUCV error."},
	{1005, "EDC8005I", "Offload box error."},
	{1006, "EDC8006I", "Offload box restarted."},
	{1007, "EDC8007I", "Offload box down."},
	{1008, "EDC8008I", "Already a conflicting call outstanding on socket."},
	{1009, "EDC8009I", "Request cancelled using a SOCKcallCANCEL request."},
	{1011, "EDC8011I", "A name of a PFS was specified that either is not configured or is not a Sockets PFS."},
	{1100, "EDC8100I", "Block device required."},
	{1101, "EDC8101I", "Text file busy."},
	{1102, "EDC8102I", "Operation would block."},
	{1103, "EDC8103I", "Operation now in progress."},
	{1104, "EDC8104I", "Connection already in progress."},
	{1105, "EDC8105I", "Socket operation on non-socket."},
	{1106, "EDC8106I", "Destination address required."},
	{1107, "EDC8107I", "Message too long."},
	{1108, "EDC8108I", "Protocol wrong type for socket."},
	{1109, "EDC8109I", "Protocol not available."},
	{1110, "EDC8110I", "Protocol not supported."},
	{1111, "EDC8111I", "Socket type not supported."},
	{1112, "EDC8112I", "Operation not supported on socket."},
	{1113, "EDC8113I", "Protocol family not supported."},
	{1114, "EDC8114I", "Address family not supported."},
	{1115, "EDC8115I", "Address already in use."},
	{1116, "EDC8116I", "Address not available."},
	{1117, "EDC8117I", "Network is down."},
	{1118, "EDC8118I", "Network is unreachable."},
	{1119, "EDC8119I", "Network dropped connection on reset."},
	{1120, "EDC8120I", "Connection ended abnormally."},
	{1121, "EDC8121I", "Connection reset."},
	{1122, "EDC8122I", "No buffer space available."},
	{1123, "EDC8123I", "Socket already connected."},
	{1124, "EDC8124I", "Socket not connected."},
	{1125, "EDC8125I", "Can't send after socket shutdown."},
	{1126, "EDC8126I", "Too many references; can't splice."},
	{1127, "EDC8127I", "Connection timed out."},
	{1128, "EDC8128I", "Connection refused."},
	{1129, "EDC8129I", "Host is not available."},
	{1130, "EDC8130I", "Host cannot be reached."},
	{1131, "EDC8131I", "Too many processes."},
	{1132, "EDC8132I", "Too many users."},
	{1133, "EDC8133I", "Disk quota exceeded."},
	{1134, "EDC8134I", "Stale file handle."},
	{1135, "", ""},
	{1136, "EDC8136I", "File is not a STREAM."},
	{1137, "EDC8137I", "STREAMS ioctl() timeout."},
	{1138, "EDC8138I", "No STREAMS resources."},
	{1139, "EDC8139I", "The message identified by set_id and msg_id is not in the message catalog."},
	{1140, "EDC8140I", "Bad message."},
	{1141, "EDC8141I", "Identifier removed."},
	{1142, "", ""},
	{1143, "", ""},
	{1144, "EDC8144I", "The link has been severed."},
	{1145, "", ""},
	{1146, "", ""},
	{1147, "", ""},
	{1148, "EDC8148I", "Protocol error."},
	{1149, "EDC8149I", "Multihop not allowed."},
	{1150, "", ""},
	{1151, "", ""},
	{1152, "EDC8152I", "The asynchronous I/O request has been canceled."},
	{1159, "EDC8159I", "Function call was interrupted before any data was received."},
	{1160, "EDC8160I", "Socket reuse is not supported."},
	{1161, "EDC8161I", "The file system cannot currently be moved."},
}

// Signal table
var signalList = [...]struct {
	num  Signal
	name string
	desc string
}{
	{1, "SIGHUP", "hangup"},
	{2, "SIGINT", "interrupt"},
	{3, "SIGABT", "aborted"},
	{4, "SIGILL", "illegal instruction"},
	{5, "SIGPOLL", "pollable event"},
	{6, "SIGURG", "urgent I/O condition"},
	{7, "SIGSTOP", "stop process"},
	{8, "SIGFPE", "floating point exception"},
	{9, "SIGKILL", "killed"},
	{10, "SIGBUS", "bus error"},
	{11, "SIGSEGV", "segmentation fault"},
	{12, "SIGSYS", "bad argument to routine"},
	{13, "SIGPIPE", "broken pipe"},
	{14, "SIGALRM", "alarm clock"},
	{15, "SIGTERM", "terminated"},
	{16, "SIGUSR1", "user defined signal 1"},
	{17, "SIGUSR2", "user defined signal 2"},
	{18, "SIGABND", "abend"},
	{19, "SIGCONT", "continued"},
	{20, "SIGCHLD", "child exited"},
	{21, "SIGTTIN", "stopped (tty input)"},
	{22, "SIGTTOU", "stopped (tty output)"},
	{23, "SIGIO", "I/O possible"},
	{24, "SIGQUIT", "quit"},
	{25, "SIGTSTP", "stopped"},
	{26, "SIGTRAP", "trace/breakpoint trap"},
	{27, "SIGIOER", "I/O error"},
	{28, "SIGWINCH", "window changed"},
	{29, "SIGXCPU", "CPU time limit exceeded"},
	{30, "SIGXFSZ", "file size limit exceeded"},
	{31, "SIGVTALRM", "virtual timer expired"},
	{32, "SIGPROF", "profiling timer expired"},
	{33, "SIGDANGER", "danger"},
	{34, "SIGTHSTOP", "stop thread"},
	{35, "SIGTHCONT", "continue thread"},
	{37, "SIGTRACE", "trace"},
	{38, "", "DCE"},
	{39, "SIGDUMP", "dump"},
}
