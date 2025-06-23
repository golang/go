#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script runs or (given -n) prints suggested commands to generate files for
# the Architecture/OS specified by the GOARCH and GOOS environment variables.
# See README.md for more information about how the build system works.

GOOSARCH="${GOOS}_${GOARCH}"

# defaults
mksyscall="go run mksyscall.go"
mkerrors="./mkerrors.sh"
zerrors="zerrors_$GOOSARCH.go"
mksysctl=""
zsysctl="zsysctl_$GOOSARCH.go"
mksysnum=
mktypes=
mkasm=
run="sh"
cmd=""

case "$1" in
-syscalls)
	for i in zsyscall*go
	do
		# Run the command line that appears in the first line
		# of the generated file to regenerate it.
		sed 1q $i | sed 's;^// ;;' | sh > _$i && gofmt < _$i > $i
		rm _$i
	done
	exit 0
	;;
-n)
	run="cat"
	cmd="echo"
	shift
esac

case "$#" in
0)
	;;
*)
	echo 'usage: mkall.sh [-n]' 1>&2
	exit 2
esac

if [[ "$GOOS" = "linux" ]]; then
	# Use the Docker-based build system
	# Files generated through docker (use $cmd so you can Ctl-C the build or run)
	$cmd docker build --tag generate:$GOOS $GOOS
	$cmd docker run --interactive --tty --volume $(cd -- "$(dirname -- "$0")/.." && pwd):/build generate:$GOOS
	exit
fi

GOOSARCH_in=syscall_$GOOSARCH.go
case "$GOOSARCH" in
_* | *_ | _)
	echo 'undefined $GOOS_$GOARCH:' "$GOOSARCH" 1>&2
	exit 1
	;;
aix_ppc)
	mkerrors="$mkerrors -maix32"
	mksyscall="go run mksyscall_aix_ppc.go -aix"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
aix_ppc64)
	mkerrors="$mkerrors -maix64"
	mksyscall="go run mksyscall_aix_ppc64.go -aix"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
darwin_amd64)
	mkerrors="$mkerrors -m64"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	mkasm="go run mkasm.go"
	;;
darwin_arm64)
	mkerrors="$mkerrors -m64"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	mkasm="go run mkasm.go"
	;;
dragonfly_amd64)
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -dragonfly"
	mksysnum="go run mksysnum.go 'https://gitweb.dragonflybsd.org/dragonfly.git/blob_plain/HEAD:/sys/kern/syscalls.master'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
freebsd_386)
	mkerrors="$mkerrors -m32"
	mksyscall="go run mksyscall.go -l32"
	mksysnum="go run mksysnum.go 'https://cgit.freebsd.org/src/plain/sys/kern/syscalls.master?h=stable/12'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
freebsd_amd64)
	mkerrors="$mkerrors -m64"
	mksysnum="go run mksysnum.go 'https://cgit.freebsd.org/src/plain/sys/kern/syscalls.master?h=stable/12'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
freebsd_arm)
	mkerrors="$mkerrors"
	mksyscall="go run mksyscall.go -l32 -arm"
	mksysnum="go run mksysnum.go 'https://cgit.freebsd.org/src/plain/sys/kern/syscalls.master?h=stable/12'"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
freebsd_arm64)
	mkerrors="$mkerrors -m64"
	mksysnum="go run mksysnum.go 'https://cgit.freebsd.org/src/plain/sys/kern/syscalls.master?h=stable/12'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
freebsd_riscv64)
	mkerrors="$mkerrors -m64"
	mksysnum="go run mksysnum.go 'https://cgit.freebsd.org/src/plain/sys/kern/syscalls.master?h=stable/12'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
netbsd_386)
	mkerrors="$mkerrors -m32"
	mksyscall="go run mksyscall.go -l32 -netbsd"
	mksysnum="go run mksysnum.go 'http://cvsweb.netbsd.org/bsdweb.cgi/~checkout~/src/sys/kern/syscalls.master'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
netbsd_amd64)
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -netbsd"
	mksysnum="go run mksysnum.go 'http://cvsweb.netbsd.org/bsdweb.cgi/~checkout~/src/sys/kern/syscalls.master'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
netbsd_arm)
	mkerrors="$mkerrors"
	mksyscall="go run mksyscall.go -l32 -netbsd -arm"
	mksysnum="go run mksysnum.go 'http://cvsweb.netbsd.org/bsdweb.cgi/~checkout~/src/sys/kern/syscalls.master'"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
netbsd_arm64)
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -netbsd"
	mksysnum="go run mksysnum.go 'http://cvsweb.netbsd.org/bsdweb.cgi/~checkout~/src/sys/kern/syscalls.master'"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
openbsd_386)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors -m32"
	mksyscall="go run mksyscall.go -l32 -openbsd -libc"
	mksysctl="go run mksysctl_openbsd.go"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
openbsd_amd64)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -openbsd -libc"
	mksysctl="go run mksysctl_openbsd.go"
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
openbsd_arm)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors"
	mksyscall="go run mksyscall.go -l32 -openbsd -arm -libc"
	mksysctl="go run mksysctl_openbsd.go"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
openbsd_arm64)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -openbsd -libc"
	mksysctl="go run mksysctl_openbsd.go"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
openbsd_mips64)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -openbsd -libc"
	mksysctl="go run mksysctl_openbsd.go"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
openbsd_ppc64)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -openbsd -libc"
	mksysctl="go run mksysctl_openbsd.go"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
openbsd_riscv64)
	mkasm="go run mkasm.go"
	mkerrors="$mkerrors -m64"
	mksyscall="go run mksyscall.go -openbsd -libc"
	mksysctl="go run mksysctl_openbsd.go"
	# Let the type of C char be signed for making the bare syscall
	# API consistent across platforms.
	mktypes="GOARCH=$GOARCH go tool cgo -godefs -- -fsigned-char"
	;;
solaris_amd64)
	mksyscall="go run mksyscall_solaris.go"
	mkerrors="$mkerrors -m64"
	mksysnum=
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
illumos_amd64)
        mksyscall="go run mksyscall_solaris.go"
	mkerrors=
	mksysnum=
	mktypes="GOARCH=$GOARCH go tool cgo -godefs"
	;;
*)
	echo 'unrecognized $GOOS_$GOARCH: ' "$GOOSARCH" 1>&2
	exit 1
	;;
esac

(
	if [ -n "$mkerrors" ]; then echo "$mkerrors |gofmt >$zerrors"; fi
	case "$GOOS" in
	*)
		syscall_goos="syscall_$GOOS.go"
		case "$GOOS" in
		darwin | dragonfly | freebsd | netbsd | openbsd)
			syscall_goos="syscall_bsd.go $syscall_goos"
			;;
		esac
		if [ -n "$mksyscall" ]; then
			if [ "$GOOSARCH" == "aix_ppc64" ]; then
				# aix/ppc64 script generates files instead of writing to stdin.
				echo "$mksyscall -tags $GOOS,$GOARCH $syscall_goos $GOOSARCH_in && gofmt -w zsyscall_$GOOSARCH.go && gofmt -w zsyscall_"$GOOSARCH"_gccgo.go && gofmt -w zsyscall_"$GOOSARCH"_gc.go " ;
			elif [ "$GOOS" == "illumos" ]; then
			        # illumos code generation requires a --illumos switch
			        echo "$mksyscall -illumos -tags illumos,$GOARCH syscall_illumos.go |gofmt > zsyscall_illumos_$GOARCH.go";
			        # illumos implies solaris, so solaris code generation is also required
				echo "$mksyscall -tags solaris,$GOARCH syscall_solaris.go syscall_solaris_$GOARCH.go |gofmt >zsyscall_solaris_$GOARCH.go";
			else
				echo "$mksyscall -tags $GOOS,$GOARCH $syscall_goos $GOOSARCH_in |gofmt >zsyscall_$GOOSARCH.go";
			fi
		fi
	esac
	if [ -n "$mksysctl" ]; then echo "$mksysctl |gofmt >$zsysctl"; fi
	if [ -n "$mksysnum" ]; then echo "$mksysnum |gofmt >zsysnum_$GOOSARCH.go"; fi
	if [ -n "$mktypes" ]; then echo "$mktypes types_$GOOS.go | go run mkpost.go > ztypes_$GOOSARCH.go"; fi
	if [ -n "$mkasm" ]; then echo "$mkasm $GOOS $GOARCH"; fi
) | $run
