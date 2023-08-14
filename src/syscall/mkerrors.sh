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

CC=${CC:-gcc}

if [[ "$GOOS" -eq "solaris" ]]; then
	# Assumes GNU versions of utilities in PATH.
	export PATH=/usr/gnu/bin:$PATH
fi

uname=$(uname)

includes_AIX='
#include <net/if.h>
#include <net/netopt.h>
#include <netinet/ip_mroute.h>
#include <sys/mman.h>
#include <sys/protosw.h>
#include <sys/ptrace.h>
#include <sys/stropts.h>
#include <termios.h>
'

includes_Darwin='
#define _DARWIN_C_SOURCE
#define KERNEL
#define _DARWIN_USE_64_BIT_INODE
#include <sys/types.h>
#include <sys/event.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>
#include <termios.h>
'

includes_DragonFly='
#include <sys/types.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <termios.h>
#include <netinet/ip.h>
#include <net/ip_mroute/ip_mroute.h>
'

includes_FreeBSD='
#include <sys/param.h>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <termios.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>

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

#include <bits/sockaddr.h>
#include <sys/epoll.h>
#include <sys/inotify.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <linux/if.h>
#include <linux/if_arp.h>
#include <linux/if_ether.h>
#include <linux/if_tun.h>
#include <linux/if_packet.h>
#include <linux/if_addr.h>
#include <linux/filter.h>
#include <linux/netlink.h>
#include <linux/reboot.h>
#include <linux/rtnetlink.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/serial.h>
#include <linux/wait.h>
#include <linux/icmpv6.h>
#include <net/route.h>
#include <termios.h>

#ifndef MSG_FASTOPEN
#define MSG_FASTOPEN    0x20000000
#endif

#ifndef PTRACE_GETREGS
#define PTRACE_GETREGS	0xc
#endif

#ifndef PTRACE_SETREGS
#define PTRACE_SETREGS	0xd
#endif
'

includes_NetBSD='
#include <sys/types.h>
#include <sys/param.h>
#include <sys/event.h>
#include <sys/mman.h>
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
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/sysctl.h>
#include <sys/termios.h>
#include <sys/ttycom.h>
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
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <net/if_types.h>
#include <net/route.h>
#include <netinet/in.h>
#include <termios.h>
#include <netinet/ip.h>
#include <netinet/ip_mroute.h>
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
'

ccflags="$@"

# Write go tool cgo -godefs input.
(
	echo package syscall
	echo
	echo '/*'
	indirect="includes_$(uname)"
	echo "${!indirect} $includes"
	echo '*/'
	echo 'import "C"'
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
		$2 ~ /^CLONE_[A-Z_]+/ {next} # These are defined in exec_linux.go.
		$2 ~ /^ELF_.*$/ {next}	# <asm/elf.h> contains ELF_ARCH, etc.

		$2 !~ /^ETH_/ &&
		$2 !~ /^EPROC_/ &&
		$2 !~ /^EQUIV_/ &&
		$2 !~ /^EXPR_/ &&
		$2 ~ /^E[A-Z0-9_]+$/ ||
		$2 ~ /^B[0-9_]+$/ ||
		$2 ~ /^V[A-Z0-9]+$/ ||
		$2 ~ /^CS[A-Z0-9]/ ||
		$2 ~ /^I(SIG|CANON|CRNL|EXTEN|MAXBEL|STRIP|UTF8)$/ ||
		$2 ~ /^IGN/ ||
		$2 ~ /^IX(ON|ANY|OFF)$/ ||
		$2 ~ /^IN(LCR|PCK)$/ ||
		$2 ~ /(^FLU?SH)|(FLU?SH$)/ ||
		$2 ~ /^C(LOCAL|READ)$/ ||
		$2 == "BRKINT" ||
		$2 == "HUPCL" ||
		$2 == "PENDIN" ||
		$2 == "TOSTOP" ||
		$2 ~ /^PAR/ ||
		$2 ~ /^SIG[^_]/ ||
		$2 ~ /^O[CNPFP][A-Z]+[^_][A-Z]+$/ ||
		$2 ~ /^IN_/ ||
		$2 ~ /^LOCK_(SH|EX|NB|UN)$/ ||
		$2 ~ /^(AF|SOCK|SO|SOL|IPPROTO|IP|IPV6|ICMP6|TCP|EVFILT|NOTE|EV|SHUT|PROT|MAP|PACKET|MSG|SCM|MCL|DT|MADV|PR)_/ ||
		$2 == "ICMPV6_FILTER" ||
		$2 == "SOMAXCONN" ||
		$2 == "NAME_MAX" ||
		$2 == "IFNAMSIZ" ||
		$2 ~ /^CTL_(MAXNAME|NET|QUERY)$/ ||
		$2 ~ /^SYSCTL_VERS/ ||
		$2 ~ /^(MS|MNT)_/ ||
		$2 ~ /^TUN(SET|GET|ATTACH|DETACH)/ ||
		$2 ~ /^(O|F|FD|NAME|S|PTRACE|PT)_/ ||
		$2 ~ /^LINUX_REBOOT_CMD_/ ||
		$2 ~ /^LINUX_REBOOT_MAGIC[12]$/ ||
		$2 !~ "NLA_TYPE_MASK" &&
		$2 ~ /^(NETLINK|NLM|NLMSG|NLA|IFA|IFAN|RT|RTCF|RTN|RTPROT|RTNH|ARPHRD|ETH_P)_/ ||
		$2 ~ /^SIOC/ ||
		$2 ~ /^TIOC/ ||
		$2 !~ "RTF_BITS" &&
		$2 ~ /^(IFF|IFT|NET_RT|RTM|RTF|RTV|RTA|RTAX)_/ ||
		$2 ~ /^BIOC/ ||
		$2 ~ /^RUSAGE_(SELF|CHILDREN|THREAD)/ ||
		$2 ~ /^RLIMIT_(AS|CORE|CPU|DATA|FSIZE|NOFILE|STACK)|RLIM_INFINITY/ ||
		$2 ~ /^PRIO_(PROCESS|PGRP|USER)/ ||
		$2 !~ /^(BPF_TIMEVAL)$/ &&
		$2 ~ /^(BPF|DLT)_/ ||
		$2 !~ "WMESGLEN" &&
		$2 ~ /^W[A-Z0-9]+$/ {printf("\t%s = C.%s\n", $2, $2)}
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
	grep -v 'SIGSTKSIZE\|SIGSTKSZ\|SIGRT' |
	sort
)

# Again, writing regexps to a file.
echo '#include <errno.h>' | $CC -x c - -E -dM $ccflags |
	awk '$1=="#define" && $2 ~ /^E[A-Z0-9_]+$/ { print "^\t" $2 "[ \t]*=" }' |
	sort >_error.grep
echo '#include <signal.h>' | $CC -x c - -E -dM $ccflags |
	awk '$1=="#define" && $2 ~ /^SIG[A-Z0-9]+$/ { print "^\t" $2 "[ \t]*=" }' |
	grep -v 'SIGSTKSIZE\|SIGSTKSZ\|SIGRT' |
	sort >_signal.grep

echo '// mkerrors.sh' "$@"
echo '// Code generated by the command above; DO NOT EDIT.'
echo
go tool cgo -godefs -- "$@" _const.go >_error.out
cat _error.out | grep -vf _error.grep | grep -vf _signal.grep
echo
echo '// Errors'
echo 'const ('
cat _error.out | grep -f _error.grep | sed 's/=\(.*\)/= Errno(\1)/'
echo ')'

echo
echo '// Signals'
echo 'const ('
cat _error.out | grep -f _signal.grep | sed 's/=\(.*\)/= Signal(\1)/'
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

int errors[] = {
"
	for i in $errors
	do
		echo -E '	'$i,
	done

	echo -E "
};

int signals[] = {
"
	for i in $signals
	do
		echo -E '	'$i,
	done

	# Use -E because on some systems bash builtin interprets \n itself.
	echo -E '
};

static int
intcmp(const void *a, const void *b)
{
	return *(int*)a - *(int*)b;
}

int
main(void)
{
	int i, j, e;
	char buf[1024], *p;

	printf("\n\n// Error table\n");
	printf("var errors = [...]string {\n");
	qsort(errors, nelem(errors), sizeof errors[0], intcmp);
	for(i=0; i<nelem(errors); i++) {
		e = errors[i];
		if(i > 0 && errors[i-1] == e)
			continue;
		strcpy(buf, strerror(e));
		// lowercase first letter: Bad -> bad, but STREAM -> STREAM.
		if(A <= buf[0] && buf[0] <= Z && a <= buf[1] && buf[1] <= z)
			buf[0] += a - A;
		printf("\t%d: \"%s\",\n", e, buf);
	}
	printf("}\n\n");
	
	printf("\n\n// Signal table\n");
	printf("var signals = [...]string {\n");
	qsort(signals, nelem(signals), sizeof signals[0], intcmp);
	for(i=0; i<nelem(signals); i++) {
		e = signals[i];
		if(i > 0 && signals[i-1] == e)
			continue;
		strcpy(buf, strsignal(e));
		// lowercase first letter: Bad -> bad, but STREAM -> STREAM.
		if(A <= buf[0] && buf[0] <= Z && a <= buf[1] && buf[1] <= z)
			buf[0] += a - A;
		// cut trailing : number.
		p = strrchr(buf, ":"[0]);
		if(p)
			*p = '\0';
		printf("\t%d: \"%s\",\n", e, buf);
	}
	printf("}\n\n");

	return 0;
}

'
) >_errors.c

$CC $ccflags -o _errors _errors.c && $GORUN ./_errors && rm -f _errors.c _errors _const.go _error.grep _signal.grep _error.out
