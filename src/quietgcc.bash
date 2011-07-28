#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The master for this file is $GOROOT/src/quietgcc.bash
# Changes made to $GOBIN/quietgcc will be overwritten.

# Gcc output that we don't care to see.
ignore=': error: .Each undeclared identifier'
ignore=$ignore'|: error: for each function it appears'
ignore=$ignore'|is dangerous, better use'
ignore=$ignore'|is almost always misused'
ignore=$ignore'|: In function '
ignore=$ignore'|: At top level: '
ignore=$ignore'|In file included from'
ignore=$ignore'|        from'

# Figure out which cc to run; this is set by make.bash.
gcc="@CC@"
if test "$gcc" = "@C""C@"; then
  gcc=gcc
fi

# Build 64-bit binaries on 64-bit systems, unless GOHOSTARCH=386.
case "$(uname -m -p)-$GOHOSTARCH" in
*x86_64*-386 | *amd64*-386)
	gcc="$gcc -m32"
	;;
*x86_64* | *amd64*)
	gcc="$gcc -m64"
esac

# Run gcc, save error status, redisplay output without noise, exit with gcc status.
tmp="${TMPDIR:-/tmp}/quietgcc.$$.$USER.out"
$gcc -Wall -Wno-sign-compare -Wno-missing-braces \
	-Wno-parentheses -Wno-unknown-pragmas -Wno-switch -Wno-comment \
	-Werror \
	"$@" >"$tmp" 2>&1
status=$?
egrep -v "$ignore" "$tmp" | uniq | tee "$tmp.1"

rm -f "$tmp" "$tmp.1"
exit $status
