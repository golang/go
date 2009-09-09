#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The master for this file is $GOROOT/src/quietgcc.bash
# Changes made to $HOME/bin/quietgcc will be overridden.

# Gcc output that we don't care to see.
ignore=': error: .Each undeclared identifier'
ignore=$ignore'|: error: for each function it appears'
ignore=$ignore'|is dangerous, better use'
ignore=$ignore'|is almost always misused'
ignore=$ignore'|: In function '
ignore=$ignore'|: At top level: '
ignore=$ignore'|In file included from'
ignore=$ignore'|        from'

# Figure out which cc to run.
# Can use plain cc on real 64-bit machines
# and on OS X, but have to use crosstool on
# mixed64-32 machines like thresher.
gcc=gcc
case "`uname -a`" in
*mixed64-32*)
	gcc=/usr/crosstool/v10/gcc-4.2.1-glibc-2.3.2/x86_64-unknown-linux-gnu/x86_64-unknown-linux-gnu/bin/gcc
esac

# If this is a 64-bit machine, compile 64-bit versions of
# the host tools, to match the native ptrace.
case "`uname -m -p`" in
*x86_64* | *amd64*)
	gcc="$gcc -m64"
esac


# Run gcc, save error status, redisplay output without noise, exit with gcc status.
tmp=/tmp/qcc.$$.$USER.out
$gcc -Wall -Wno-sign-compare -Wno-missing-braces \
	-Wno-parentheses -Wno-unknown-pragmas -Wno-switch -Wno-comment \
	"$@" >$tmp 2>&1
status=$?
egrep -v "$ignore" $tmp | uniq | tee $tmp.1

# Make incompatible pointer type "warnings" stop the build.
# Not quite perfect--we should remove the object file--but
# a step in the right direction.
if egrep 'incompatible pointer type' $tmp.1 >/dev/null; then
	status=1
fi
rm -f $tmp $tmp.1
exit $status
