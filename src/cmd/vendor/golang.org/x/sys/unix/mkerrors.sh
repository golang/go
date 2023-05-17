#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Generate Go code listing errors and other #defined constant
# values (ENAMETOOLONG etc.), by asking the preprocessor
# about the definitions.

unset LANG
export LC_ALL=C
export LC_CTYPE=C

if test -z "$GOARCH" -o -z "$GOOS"; then
	echo 1>&2 "GOARCH or GOOS not defined in environment"
	exit 1
fi

# Check that we are using the new build system if we should
if [[ "$GOOS" = "linux" ]] && [[ "$GOLANG_SYS_BUILD" != "docker" ]]; then
	echo 1>&2 "In the Docker based build system, mkerrors should not be called directly."
	echo 1>&2 "See README.md"
	exit 1
fi

if [[ "$GOOS" = "aix" ]]; then
	CC=${CC:-gcc}
else
	CC=${CC:-cc}
fi

if [[ "$GOOS" = "solaris" ]]; then
	# Assumes GNU versions of utilities in PATH.
	export PATH=/usr/gnu/bin:$PATH
fi

uname=$(uname)

includes_AIX='
#include <net/if.h>
#include <net/netopt.h>
#include <netinet/ip_mroute.h>
#include <sys/protosw.h>
#include <sys/stropts.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/select.h>
#include <sys/termio.h>
#include <termios.h>
#include <fcntl.h>

#define AF_LOCAL AF_UNIX
'

includes_Darwin='
#define _DARWIN_C_SOURCE
#define KERNEL 1
#define _DARWIN_USE_64_BIT_INODE
#define __APPLE_USE_RFC_3542
#include <stdint.h>
#include <sys/attr.h>
#include <sys/clonefile.h>
#include <sys/kern_control.h>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/ptrace.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/sockio.h>
#include <sys/sys_domain.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <sys/vsock.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <termios.h>

// for backwards compatibility because moved TIOCREMOTE to Kernel.framework after MacOSX12.0.sdk.
#define TIOCREMOTE 0x80047469
'

includes_DragonFly='
#include <sys/types.h>
#include <sys/event.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_clone.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <termios.h>
#include <netinet/ip.h>
#include <net/ip_mroute/ip_mroute.h>
'

includes_FreeBSD='
#include <sys/capsicum.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/disk.h>
#include <sys/event.h>
#include <sys/sched.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/sockio.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/ptrace.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <termios.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>
#include <sys/extattr.h>

#if __FreeBSD__ >= 10
#define IFT_CARP	0xf8	// IFT_CARP is deprecated in FreeBSD 10
#undef SIOCAIFADDR
#define SIOCAIFADDR	_IOW(105, 26, struct oifaliasreq)	// ifaliasreq contains if_data
#undef SIOCSIFPHYADDR
#define SIOCSIFPHYADDR	_IOW(105, 70, struct oifaliasreq)	// ifaliasreq contains if_data
#endif
'

includes_Linux='
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#ifndef __LP64__
#define _FILE_OFFSET_BITS 64
#endif
#define _GNU_SOURCE

// <sys/ioctl.h> is broken on powerpc64, as it fails to include definitions of
// these structures. We just include them copied from <bits/termios.h>.
#if defined(__powerpc__)
struct sgttyb {
        char    sg_ispeed;
        char    sg_ospeed;
        char    sg_erase;
        char    sg_kill;
        short   sg_flags;
};

struct tchars {
        char    t_intrc;
        char    t_quitc;
        char    t_startc;
        char    t_stopc;
        char    t_eofc;
        char    t_brkc;
};

struct ltchars {
        char    t_suspc;
        char    t_dsuspc;
        char    t_rprntc;
        char    t_flushc;
        char    t_werasc;
        char    t_lnextc;
};
#endif

#include <bits/sockaddr.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/inotify.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/timerfd.h>
#include <sys/uio.h>
#include <sys/xattr.h>
#include <netinet/udp.h>
#include <linux/audit.h>
#include <linux/bpf.h>
#include <linux/can.h>
#include <linux/can/error.h>
#include <linux/can/netlink.h>
#include <linux/can/raw.h>
#include <linux/capability.h>
#include <linux/cryptouser.h>
#include <linux/devlink.h>
#include <linux/dm-ioctl.h>
#include <linux/errqueue.h>
#include <linux/ethtool_netlink.h>
#include <linux/falloc.h>
#include <linux/fanotify.h>
#include <linux/fib_rules.h>
#include <linux/filter.h>
#include <linux/fs.h>
#include <linux/fscrypt.h>
#include <linux/fsverity.h>
#include <linux/genetlink.h>
#include <linux/hdreg.h>
#include <linux/hidraw.h>
#include <linux/if.h>
#include <linux/if_addr.h>
#include <linux/if_alg.h>
#include <linux/if_arp.h>
#include <linux/if_ether.h>
#include <linux/if_ppp.h>
#include <linux/if_tun.h>
#include <linux/if_packet.h>
#include <linux/if_xdp.h>
#include <linux/input.h>
#include <linux/kcm.h>
#include <linux/kexec.h>
#include <linux/keyctl.h>
#include <linux/landlock.h>
#include <linux/loop.h>
#include <linux/lwtunnel.h>
#include <linux/magic.h>
#include <linux/memfd.h>
#include <linux/module.h>
#include <linux/mount.h>
#include <linux/netfilter/nfnetlink.h>
#include <linux/netlink.h>
#include <linux/net_namespace.h>
#include <linux/nfc.h>
#include <linux/nsfs.h>
#include <linux/perf_event.h>
#include <linux/pps.h>
#include <linux/ptrace.h>
#include <linux/random.h>
#include <linux/reboot.h>
#include <linux/rtc.h>
#include <linux/rtnetlink.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <linux/serial.h>
#include <linux/sockios.h>
#include <linux/taskstats.h>
#include <linux/tipc.h>
#include <linux/vm_sockets.h>
#include <linux/wait.h>
#include <linux/watchdog.h>
#include <linux/wireguard.h>

#include <mtd/ubi-user.h>
#include <mtd/mtd-user.h>
#include <net/route.h>

#if defined(__sparc__)
// On sparc{,64}, the kernel defines struct termios2 itself which clashes with the
// definition in glibc. As only the error constants are needed here, include the
// generic termibits.h (which is included by termbits.h on sparc).
#include <asm-generic/termbits.h>
#else
#include <asm/termbits.h>
#endif

#ifndef MSG_FASTOPEN
#define MSG_FASTOPEN    0x20000000
#endif

#ifndef PTRACE_GETREGS
#define PTRACE_GETREGS	0xc
#endif

#ifndef PTRACE_SETREGS
#define PTRACE_SETREGS	0xd
#endif

#ifndef SOL_NETLINK
#define SOL_NETLINK	270
#endif

#ifndef SOL_SMC
#define SOL_SMC 286
#endif

#ifdef SOL_BLUETOOTH
// SPARC includes this in /usr/include/sparc64-linux-gnu/bits/socket.h
// but it is already in bluetooth_linux.go
#undef SOL_BLUETOOTH
#endif

// Certain constants are missing from the fs/crypto UAPI
#define FS_KEY_DESC_PREFIX              "fscrypt:"
#define FS_KEY_DESC_PREFIX_SIZE         8
#define FS_MAX_KEY_SIZE                 64

// The code generator produces -0x1 for (~0), but an unsigned value is necessary
// for the tipc_subscr timeout __u32 field.
#undef TIPC_WAIT_FOREVER
#define TIPC_WAIT_FOREVER 0xffffffff

// Copied from linux/l2tp.h
// Including linux/l2tp.h here causes conflicts between linux/in.h
// and netinet/in.h included via net/route.h above.
#define IPPROTO_L2TP		115

// Copied from linux/hid.h.
// Keep in sync with the size of the referenced fields.
#define _HIDIOCGRAWNAME_LEN	128 // sizeof_field(struct hid_device, name)
#define _HIDIOCGRAWPHYS_LEN	64  // sizeof_field(struct hid_device, phys)
#define _HIDIOCGRAWUNIQ_LEN	64  // sizeof_field(struct hid_device, uniq)

#define _HIDIOCGRAWNAME		HIDIOCGRAWNAME(_HIDIOCGRAWNAME_LEN)
#define _HIDIOCGRAWPHYS		HIDIOCGRAWPHYS(_HIDIOCGRAWPHYS_LEN)
#define _HIDIOCGRAWUNIQ		HIDIOCGRAWUNIQ(_HIDIOCGRAWUNIQ_LEN)

'

includes_NetBSD='
#include <sys/types.h>
#include <sys/param.h>
#include <sys/event.h>
#include <sys/extattr.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/sched.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/sysctl.h>
#include <sys/termios.h>
#include <sys/ttycom.h>
#include <sys/wait.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <netinet/in_systm.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>
#include <netinet/if_ether.h>

// Needed since <sys/param.h> refers to it...
#define schedppq 1
'

includes_OpenBSD='
#include <sys/types.h>
#include <sys/param.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/select.h>
#include <sys/sched.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/termios.h>
#include <sys/ttycom.h>
#include <sys/unistd.h>
#include <sys/wait.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/if_var.h>
#include <net/route.h>
#include <netinet/in.h>
#include <netinet/in_systm.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>
#include <netinet/if_ether.h>
#include <net/if_bridge.h>

// We keep some constants not supported in OpenBSD 5.5 and beyond for
// the promise of compatibility.
#define EMUL_ENABLED		0x1
#define EMUL_NATIVE		0x2
#define IPV6_FAITH		0x1d
#define IPV6_OPTIONS		0x1
#define IPV6_RTHDR_STRICT	0x1
#define IPV6_SOCKOPT_RESERVED1	0x3
#define SIOCGIFGENERIC		0xc020693a
#define SIOCSIFGENERIC		0x80206939
#define WALTSIG			0x4
'

includes_SunOS='
#include <limits.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/stat.h>
#include <sys/stream.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/mkdev.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/icmp6.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>
#include <termios.h>
'


includes='
#include <sys/types.h>
#include <sys/file.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <sys/signal.h>
#include <signal.h>
#include <sys/resource.h>
#include <time.h>
'
ccflags="$@"

# Write go tool cgo -godefs input.
(
	echo package unix
	echo
	echo '/*'
	indirect="includes_$(uname)"
	echo "${!indirect} $includes"
	echo '*/'
	echo 'import "C"'
	echo 'import "syscall"'
	echo
	echo 'const ('

	# The gcc command line prints all the #defines
	# it encounters while processing the input
	echo "${!indirect} $includes" | $CC -x c - -E -dM $ccflags |
	awk '
		$1 != "#define" || $2 ~ /\(/ || $3 == "" {next}

		$2 ~ /^E([ABCD]X|[BIS]P|[SD]I|S|FL)$/ {next}  # 386 registers
		$2 ~ /^(SIGEV_|SIGSTKSZ|SIGRT(MIN|MAX))/ {next}
		$2 ~ /^(SCM_SRCRT)$/ {next}
		$2 ~ /^(MAP_FAILED)$/ {next}
		$2 ~ /^ELF_.*$/ {next}# <asm/elf.h> contains ELF_ARCH, etc.

		$2 ~ /^EXTATTR_NAMESPACE_NAMES/ ||
		$2 ~ /^EXTATTR_NAMESPACE_[A-Z]+_STRING/ {next}

		$2 !~ /^ECCAPBITS/ &&
		$2 !~ /^ETH_/ &&
		$2 !~ /^EPROC_/ &&
		$2 !~ /^EQUIV_/ &&
		$2 !~ /^EXPR_/ &&
		$2 !~ /^EVIOC/ &&
		$2 ~ /^E[A-Z0-9_]+$/ ||
		$2 ~ /^B[0-9_]+$/ ||
		$2 ~ /^(OLD|NEW)DEV$/ ||
		$2 == "BOTHER" ||
		$2 ~ /^CI?BAUD(EX)?$/ ||
		$2 == "IBSHIFT" ||
		$2 ~ /^V[A-Z0-9]+$/ ||
		$2 ~ /^CS[A-Z0-9]/ ||
		$2 ~ /^I(SIG|CANON|CRNL|UCLC|EXTEN|MAXBEL|STRIP|UTF8)$/ ||
		$2 ~ /^IGN/ ||
		$2 ~ /^IX(ON|ANY|OFF)$/ ||
		$2 ~ /^IN(LCR|PCK)$/ ||
		$2 !~ "X86_CR3_PCID_NOFLUSH" &&
		$2 ~ /(^FLU?SH)|(FLU?SH$)/ ||
		$2 ~ /^C(LOCAL|READ|MSPAR|RTSCTS)$/ ||
		$2 == "BRKINT" ||
		$2 == "HUPCL" ||
		$2 == "PENDIN" ||
		$2 == "TOSTOP" ||
		$2 == "XCASE" ||
		$2 == "ALTWERASE" ||
		$2 == "NOKERNINFO" ||
		$2 == "NFDBITS" ||
		$2 ~ /^PAR/ ||
		$2 ~ /^SIG[^_]/ ||
		$2 ~ /^O[CNPFPL][A-Z]+[^_][A-Z]+$/ ||
		$2 ~ /^(NL|CR|TAB|BS|VT|FF)DLY$/ ||
		$2 ~ /^(NL|CR|TAB|BS|VT|FF)[0-9]$/ ||
		$2 ~ /^O?XTABS$/ ||
		$2 ~ /^TC[IO](ON|OFF)$/ ||
		$2 ~ /^IN_/ ||
		$2 ~ /^KCM/ ||
		$2 ~ /^LANDLOCK_/ ||
		$2 ~ /^LOCK_(SH|EX|NB|UN)$/ ||
		$2 ~ /^LO_(KEY|NAME)_SIZE$/ ||
		$2 ~ /^LOOP_(CLR|CTL|GET|SET)_/ ||
		$2 ~ /^(AF|SOCK|SO|SOL|IPPROTO|IP|IPV6|TCP|MCAST|EVFILT|NOTE|SHUT|PROT|MAP|MFD|T?PACKET|MSG|SCM|MCL|DT|MADV|PR|LOCAL|TCPOPT|UDP)_/ ||
		$2 ~ /^NFC_(GENL|PROTO|COMM|RF|SE|DIRECTION|LLCP|SOCKPROTO)_/ ||
		$2 ~ /^NFC_.*_(MAX)?SIZE$/ ||
		$2 ~ /^RAW_PAYLOAD_/ ||
		$2 ~ /^[US]F_/ ||
		$2 ~ /^TP_STATUS_/ ||
		$2 ~ /^FALLOC_/ ||
		$2 ~ /^ICMPV?6?_(FILTER|SEC)/ ||
		$2 == "SOMAXCONN" ||
		$2 == "NAME_MAX" ||
		$2 == "IFNAMSIZ" ||
		$2 ~ /^CTL_(HW|KERN|MAXNAME|NET|QUERY)$/ ||
		$2 ~ /^KERN_(HOSTNAME|OS(RELEASE|TYPE)|VERSION)$/ ||
		$2 ~ /^HW_MACHINE$/ ||
		$2 ~ /^SYSCTL_VERS/ ||
		$2 !~ "MNT_BITS" &&
		$2 ~ /^(MS|MNT|MOUNT|UMOUNT)_/ ||
		$2 ~ /^NS_GET_/ ||
		$2 ~ /^TUN(SET|GET|ATTACH|DETACH)/ ||
		$2 ~ /^(O|F|[ES]?FD|NAME|S|PTRACE|PT|PIOD|TFD)_/ ||
		$2 ~ /^KEXEC_/ ||
		$2 ~ /^LINUX_REBOOT_CMD_/ ||
		$2 ~ /^LINUX_REBOOT_MAGIC[12]$/ ||
		$2 ~ /^MODULE_INIT_/ ||
		$2 !~ "NLA_TYPE_MASK" &&
		$2 !~ /^RTC_VL_(ACCURACY|BACKUP|DATA)/ &&
		$2 ~ /^(NETLINK|NLM|NLMSG|NLA|IFA|IFAN|RT|RTC|RTCF|RTN|RTPROT|RTNH|ARPHRD|ETH_P|NETNSA)_/ ||
		$2 ~ /^FIORDCHK$/ ||
		$2 ~ /^SIOC/ ||
		$2 ~ /^TIOC/ ||
		$2 ~ /^TCGET/ ||
		$2 ~ /^TCSET/ ||
		$2 ~ /^TC(FLSH|SBRKP?|XONC)$/ ||
		$2 !~ "RTF_BITS" &&
		$2 ~ /^(IFF|IFT|NET_RT|RTM(GRP)?|RTF|RTV|RTA|RTAX)_/ ||
		$2 ~ /^BIOC/ ||
		$2 ~ /^DIOC/ ||
		$2 ~ /^RUSAGE_(SELF|CHILDREN|THREAD)/ ||
		$2 ~ /^RLIMIT_(AS|CORE|CPU|DATA|FSIZE|LOCKS|MEMLOCK|MSGQUEUE|NICE|NOFILE|NPROC|RSS|RTPRIO|RTTIME|SIGPENDING|STACK)|RLIM_INFINITY/ ||
		$2 ~ /^PRIO_(PROCESS|PGRP|USER)/ ||
		$2 ~ /^CLONE_[A-Z_]+/ ||
		$2 !~ /^(BPF_TIMEVAL|BPF_FIB_LOOKUP_[A-Z]+)$/ &&
		$2 ~ /^(BPF|DLT)_/ ||
		$2 ~ /^AUDIT_/ ||
		$2 ~ /^(CLOCK|TIMER)_/ ||
		$2 ~ /^CAN_/ ||
		$2 ~ /^CAP_/ ||
		$2 ~ /^CP_/ ||
		$2 ~ /^CPUSTATES$/ ||
		$2 ~ /^CTLIOCGINFO$/ ||
		$2 ~ /^ALG_/ ||
		$2 ~ /^FI(CLONE|DEDUPERANGE)/ ||
		$2 ~ /^FS_(POLICY_FLAGS|KEY_DESC|ENCRYPTION_MODE|[A-Z0-9_]+_KEY_SIZE)/ ||
		$2 ~ /^FS_IOC_.*(ENCRYPTION|VERITY|[GS]ETFLAGS)/ ||
		$2 ~ /^FS_VERITY_/ ||
		$2 ~ /^FSCRYPT_/ ||
		$2 ~ /^DM_/ ||
		$2 ~ /^GRND_/ ||
		$2 ~ /^RND/ ||
		$2 ~ /^KEY_(SPEC|REQKEY_DEFL)_/ ||
		$2 ~ /^KEYCTL_/ ||
		$2 ~ /^PERF_/ ||
		$2 ~ /^SECCOMP_MODE_/ ||
		$2 ~ /^SEEK_/ ||
		$2 ~ /^SPLICE_/ ||
		$2 ~ /^SYNC_FILE_RANGE_/ ||
		$2 !~ /IOC_MAGIC/ &&
		$2 ~ /^[A-Z][A-Z0-9_]+_MAGIC2?$/ ||
		$2 ~ /^(VM|VMADDR)_/ ||
		$2 ~ /^IOCTL_VM_SOCKETS_/ ||
		$2 ~ /^(TASKSTATS|TS)_/ ||
		$2 ~ /^CGROUPSTATS_/ ||
		$2 ~ /^GENL_/ ||
		$2 ~ /^STATX_/ ||
		$2 ~ /^RENAME/ ||
		$2 ~ /^UBI_IOC[A-Z]/ ||
		$2 ~ /^UTIME_/ ||
		$2 ~ /^XATTR_(CREATE|REPLACE|NO(DEFAULT|FOLLOW|SECURITY)|SHOWCOMPRESSION)/ ||
		$2 ~ /^ATTR_(BIT_MAP_COUNT|(CMN|VOL|FILE)_)/ ||
		$2 ~ /^FSOPT_/ ||
		$2 ~ /^WDIO[CFS]_/ ||
		$2 ~ /^NFN/ ||
		$2 ~ /^XDP_/ ||
		$2 ~ /^RWF_/ ||
		$2 ~ /^(HDIO|WIN|SMART)_/ ||
		$2 ~ /^CRYPTO_/ ||
		$2 ~ /^TIPC_/ ||
		$2 !~  "DEVLINK_RELOAD_LIMITS_VALID_MASK" &&
		$2 ~ /^DEVLINK_/ ||
		$2 ~ /^ETHTOOL_/ ||
		$2 ~ /^LWTUNNEL_IP/ ||
		$2 ~ /^ITIMER_/ ||
		$2 !~ "WMESGLEN" &&
		$2 ~ /^W[A-Z0-9]+$/ ||
		$2 ~ /^P_/ ||
		$2 ~/^PPPIOC/ ||
		$2 ~ /^FAN_|FANOTIFY_/ ||
		$2 == "HID_MAX_DESCRIPTOR_SIZE" ||
		$2 ~ /^_?HIDIOC/ ||
		$2 ~ /^BUS_(USB|HIL|BLUETOOTH|VIRTUAL)$/ ||
		$2 ~ /^MTD/ ||
		$2 ~ /^OTP/ ||
		$2 ~ /^MEM/ ||
		$2 ~ /^WG/ ||
		$2 ~ /^FIB_RULE_/ ||
		$2 ~ /^BLK[A-Z]*(GET$|SET$|BUF$|PART$|SIZE)/ {printf("\t%s = C.%s\n", $2, $2)}
		$2 ~ /^__WCOREFLAG$/ {next}
		$2 ~ /^__W[A-Z0-9]+$/ {printf("\t%s = C.%s\n", substr($2,3), $2)}

		{next}
	' | sort

	echo ')'
) >_const.go

# Pull out the error names for later.
errors=$(
	echo '#include <errno.h>' | $CC -x c - -E -dM $ccflags |
	awk '$1=="#define" && $2 ~ /^E[A-Z0-9_]+$/ { print $2 }' |
	sort
)

# Pull out the signal names for later.
signals=$(
	echo '#include <signal.h>' | $CC -x c - -E -dM $ccflags |
	awk '$1=="#define" && $2 ~ /^SIG[A-Z0-9]+$/ { print $2 }' |
	grep -v 'SIGSTKSIZE\|SIGSTKSZ\|SIGRT\|SIGMAX64' |
	sort
)

# Again, writing regexps to a file.
echo '#include <errno.h>' | $CC -x c - -E -dM $ccflags |
	awk '$1=="#define" && $2 ~ /^E[A-Z0-9_]+$/ { print "^\t" $2 "[ \t]*=" }' |
	sort >_error.grep
echo '#include <signal.h>' | $CC -x c - -E -dM $ccflags |
	awk '$1=="#define" && $2 ~ /^SIG[A-Z0-9]+$/ { print "^\t" $2 "[ \t]*=" }' |
	grep -v 'SIGSTKSIZE\|SIGSTKSZ\|SIGRT\|SIGMAX64' |
	sort >_signal.grep

echo '// mkerrors.sh' "$@"
echo '// Code generated by the command above; see README.md. DO NOT EDIT.'
echo
echo "//go:build ${GOARCH} && ${GOOS}"
echo "// +build ${GOARCH},${GOOS}"
echo
go tool cgo -godefs -- "$@" _const.go >_error.out
cat _error.out | grep -vf _error.grep | grep -vf _signal.grep
echo
echo '// Errors'
echo 'const ('
cat _error.out | grep -f _error.grep | sed 's/=\(.*\)/= syscall.Errno(\1)/'
echo ')'

echo
echo '// Signals'
echo 'const ('
cat _error.out | grep -f _signal.grep | sed 's/=\(.*\)/= syscall.Signal(\1)/'
echo ')'

# Run C program to print error and syscall strings.
(
	echo -E "
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <signal.h>

#define nelem(x) (sizeof(x)/sizeof((x)[0]))

enum { A = 'A', Z = 'Z', a = 'a', z = 'z' }; // avoid need for single quotes below

struct tuple {
	int num;
	const char *name;
};

struct tuple errors[] = {
"
	for i in $errors
	do
		echo -E '	{'$i', "'$i'" },'
	done

	echo -E "
};

struct tuple signals[] = {
"
	for i in $signals
	do
		echo -E '	{'$i', "'$i'" },'
	done

	# Use -E because on some systems bash builtin interprets \n itself.
	echo -E '
};

static int
tuplecmp(const void *a, const void *b)
{
	return ((struct tuple *)a)->num - ((struct tuple *)b)->num;
}

int
main(void)
{
	int i, e;
	char buf[1024], *p;

	printf("\n\n// Error table\n");
	printf("var errorList = [...]struct {\n");
	printf("\tnum  syscall.Errno\n");
	printf("\tname string\n");
	printf("\tdesc string\n");
	printf("} {\n");
	qsort(errors, nelem(errors), sizeof errors[0], tuplecmp);
	for(i=0; i<nelem(errors); i++) {
		e = errors[i].num;
		if(i > 0 && errors[i-1].num == e)
			continue;
		strcpy(buf, strerror(e));
		// lowercase first letter: Bad -> bad, but STREAM -> STREAM.
		if(A <= buf[0] && buf[0] <= Z && a <= buf[1] && buf[1] <= z)
			buf[0] += a - A;
		printf("\t{ %d, \"%s\", \"%s\" },\n", e, errors[i].name, buf);
	}
	printf("}\n\n");

	printf("\n\n// Signal table\n");
	printf("var signalList = [...]struct {\n");
	printf("\tnum  syscall.Signal\n");
	printf("\tname string\n");
	printf("\tdesc string\n");
	printf("} {\n");
	qsort(signals, nelem(signals), sizeof signals[0], tuplecmp);
	for(i=0; i<nelem(signals); i++) {
		e = signals[i].num;
		if(i > 0 && signals[i-1].num == e)
			continue;
		strcpy(buf, strsignal(e));
		// lowercase first letter: Bad -> bad, but STREAM -> STREAM.
		if(A <= buf[0] && buf[0] <= Z && a <= buf[1] && buf[1] <= z)
			buf[0] += a - A;
		// cut trailing : number.
		p = strrchr(buf, ":"[0]);
		if(p)
			*p = '\0';
		printf("\t{ %d, \"%s\", \"%s\" },\n", e, signals[i].name, buf);
	}
	printf("}\n\n");

	return 0;
}

'
) >_errors.c

$CC $ccflags -o _errors _errors.c && $GORUN ./_errors && rm -f _errors.c _errors _const.go _error.grep _signal.grep _error.out
