#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This directory is intended to test the use of Go with sanitizers
# like msan, asan, etc.  See https://github.com/google/sanitizers .

set -e

# The sanitizers were originally developed with clang, so prefer it.
CC=cc
if test -x "$(type -p clang)"; then
  CC=clang
fi
export CC

if [ "$(sysctl -n vm.overcommit_memory)" = 2 ]; then
  echo "skipping msan/tsan tests: vm.overcommit_memory=2" >&2
  exit 0
fi

msan=yes

TMPDIR=${TMPDIR:-/tmp}
echo 'int main() { return 0; }' > ${TMPDIR}/testsanitizers$$.c
if $CC -fsanitize=memory -o ${TMPDIR}/testsanitizers$$ ${TMPDIR}/testsanitizers$$.c 2>&1 | grep "unrecognized" >& /dev/null; then
  echo "skipping msan tests: $CC -fsanitize=memory not supported"
  msan=no
elif ! test -x ${TMPDIR}/testsanitizers$$; then
  echo "skipping msan tests: $CC -fsanitize-memory did not generate an executable"
  msan=no
elif ! ${TMPDIR}/testsanitizers$$ >/dev/null 2>&1; then
  echo "skipping msan tests: $CC -fsanitize-memory generates broken executable"
  msan=no
fi
rm -f ${TMPDIR}/testsanitizers$$.*

tsan=yes

# The memory and thread sanitizers in versions of clang before 3.6
# don't work with Go.
if test "$msan" = "yes" && $CC --version | grep clang >& /dev/null; then
  ver=$($CC --version | sed -e 's/.* version \([0-9.-]*\).*/\1/')
  major=$(echo $ver | sed -e 's/\([0-9]*\).*/\1/')
  minor=$(echo $ver | sed -e 's/[0-9]*\.\([0-9]*\).*/\1/')
  if test "$major" -lt 3 || test "$major" -eq 3 -a "$minor" -lt 6; then
    echo "skipping msan/tsan tests: clang version $major.$minor (older than 3.6)"
    msan=no
    tsan=no
  fi

  # Clang before 3.8 does not work with Linux at or after 4.1.
  # golang.org/issue/12898.
  if test "$msan" = "yes" -a "$major" -lt 3 || test "$major" -eq 3 -a "$minor" -lt 8; then
    if test "$(uname)" = Linux; then
      linuxver=$(uname -r)
      linuxmajor=$(echo $linuxver | sed -e 's/\([0-9]*\).*/\1/')
      linuxminor=$(echo $linuxver | sed -e 's/[0-9]*\.\([0-9]*\).*/\1/')
      if test "$linuxmajor" -gt 4 || test "$linuxmajor" -eq 4 -a "$linuxminor" -ge 1; then
        echo "skipping msan/tsan tests: clang version $major.$minor (older than 3.8) incompatible with linux version $linuxmajor.$linuxminor (4.1 or newer)"
	msan=no
	tsan=no
      fi
    fi
  fi
fi

status=0

testmsanshared() {
  goos=$(go env GOOS)
  suffix="-installsuffix testsanitizers"
  libext="so"
  if [ "$goos" = "darwin" ]; then
	  libext="dylib"
  fi
  go build -msan -buildmode=c-shared $suffix -o ${TMPDIR}/libmsanshared.$libext msan_shared.go

  echo 'int main() { return 0; }' > ${TMPDIR}/testmsanshared.c
  $CC $(go env GOGCCFLAGS) -fsanitize=memory -o ${TMPDIR}/testmsanshared ${TMPDIR}/testmsanshared.c ${TMPDIR}/libmsanshared.$libext

  if ! LD_LIBRARY_PATH=. ${TMPDIR}/testmsanshared; then
    echo "FAIL: msan_shared"
    status=1
  fi
  rm -f ${TMPDIR}/{testmsanshared,testmsanshared.c,libmsanshared.$libext}
}

if test "$msan" = "yes"; then
    if ! go build -msan std; then
	echo "FAIL: build -msan std"
	status=1
    fi

    if ! go run -msan msan.go; then
	echo "FAIL: msan"
	status=1
    fi

    if ! CGO_LDFLAGS="-fsanitize=memory" CGO_CPPFLAGS="-fsanitize=memory" go run -msan -a msan2.go; then
	echo "FAIL: msan2 with -fsanitize=memory"
	status=1
    fi

    if ! go run -msan -a msan2.go; then
	echo "FAIL: msan2"
	status=1
    fi

    if ! go run -msan msan3.go; then
	echo "FAIL: msan3"
	status=1
    fi

    if ! go run -msan msan4.go; then
	echo "FAIL: msan4"
	status=1
    fi

    if ! go run -msan msan5.go; then
	echo "FAIL: msan5"
	status=1
    fi

    if go run -msan msan_fail.go 2>/dev/null; then
	echo "FAIL: msan_fail"
	status=1
    fi

    testmsanshared
fi

testtsanshared() {
  goos=$(go env GOOS)
  suffix="-installsuffix tsan"
  libext="so"
  if [ "$goos" = "darwin" ]; then
	  libext="dylib"
  fi
  go build -buildmode=c-shared $suffix -o ${TMPDIR}/libtsanshared.$libext tsan_shared.go

  echo 'int main() { return 0; }' > ${TMPDIR}/testtsanshared.c
  $CC $(go env GOGCCFLAGS) -fsanitize=thread -o ${TMPDIR}/testtsanshared ${TMPDIR}/testtsanshared.c ${TMPDIR}/libtsanshared.$libext

  if ! LD_LIBRARY_PATH=. ${TMPDIR}/testtsanshared; then
    echo "FAIL: tsan_shared"
    status=1
  fi
  rm -f ${TMPDIR}/{testtsanshared,testtsanshared.c,libtsanshared.$libext}
}

if test "$tsan" = "yes"; then
    echo 'int main() { return 0; }' > ${TMPDIR}/testsanitizers$$.c
    ok=yes
    if ! $CC -fsanitize=thread ${TMPDIR}/testsanitizers$$.c -o ${TMPDIR}/testsanitizers$$ &> ${TMPDIR}/testsanitizers$$.err; then
	ok=no
    fi
    if grep "unrecognized" ${TMPDIR}/testsanitizers$$.err >& /dev/null; then
	echo "skipping tsan tests: -fsanitize=thread not supported"
	tsan=no
    elif test "$ok" != "yes"; then
	cat ${TMPDIR}/testsanitizers$$.err
	echo "skipping tsan tests: -fsanitizer=thread build failed"
	tsan=no
    elif ! ${TMPDIR}/testsanitizers$$ 2>&1; then
	echo "skipping tsan tests: running tsan program failed"
	tsan=no
    fi
    rm -f ${TMPDIR}/testsanitizers$$*
fi

# Run a TSAN test.
# $1 test name
# $2 environment variables
# $3 go run args
testtsan() {
    err=${TMPDIR}/tsanerr$$.out
    if ! env $2 go run $3 $1 2>$err; then
	cat $err
	echo "FAIL: $1"
	status=1
    elif grep -i warning $err >/dev/null 2>&1; then
	cat $err
	echo "FAIL: $1"
	status=1
    fi
    rm -f $err
}

if test "$tsan" = "yes"; then
    testtsan tsan.go
    testtsan tsan2.go
    testtsan tsan3.go
    testtsan tsan4.go
    testtsan tsan8.go
    testtsan tsan9.go

    # These tests are only reliable using clang or GCC version 7 or later.
    # Otherwise runtime/cgo/libcgo.h can't tell whether TSAN is in use.
    ok=false
    clang=false
    if ${CC} --version | grep clang >/dev/null 2>&1; then
	ok=true
	clang=true
    else
	ver=$($CC -dumpversion)
	major=$(echo $ver | sed -e 's/\([0-9]*\).*/\1/')
	if test "$major" -lt 7; then
	    echo "skipping remaining TSAN tests: GCC version $major (older than 7)"
	else
	    ok=true
	fi
    fi

    if test "$ok" = "true"; then
	# These tests require rebuilding os/user with -fsanitize=thread.
	testtsan tsan5.go "CGO_CFLAGS=-fsanitize=thread CGO_LDFLAGS=-fsanitize=thread" "-installsuffix=tsan"
	testtsan tsan6.go "CGO_CFLAGS=-fsanitize=thread CGO_LDFLAGS=-fsanitize=thread" "-installsuffix=tsan"
	testtsan tsan7.go "CGO_CFLAGS=-fsanitize=thread CGO_LDFLAGS=-fsanitize=thread" "-installsuffix=tsan"

	# The remaining tests reportedly hang when built with GCC; issue #21196.
	if test "$clang" = "true"; then
	    testtsan tsan10.go "CGO_CFLAGS=-fsanitize=thread CGO_LDFLAGS=-fsanitize=thread" "-installsuffix=tsan"
	    testtsan tsan11.go "CGO_CFLAGS=-fsanitize=thread CGO_LDFLAGS=-fsanitize=thread" "-installsuffix=tsan"
	    testtsan tsan12.go "CGO_CFLAGS=-fsanitize=thread CGO_LDFLAGS=-fsanitize=thread" "-installsuffix=tsan"
	fi

	testtsanshared
    fi
fi

exit $status
