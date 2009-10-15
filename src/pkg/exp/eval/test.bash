#!/bin/bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run the interpreter against all the Go test programs
# that begin with the magic
#	// $G $D/$F.go && $L $F.$A && ./$A.out
# line and do not contain imports.

set -e
make
6g main.go && 6l main.6
(
for i in $(egrep -l '// \$G (\$D/)?\$F\.go \&\& \$L \$F\.\$A && \./\$A\.out' $GOROOT/test/*.go $GOROOT/test/*/*.go)
do
	if grep '^import' $i >/dev/null 2>&1
	then
		true
	else
		if $GOROOT/usr/austin/eval/6.out -f $i >/tmp/out 2>&1 && ! test -s /tmp/out
		then
			echo PASS $i
		else
			echo FAIL $i
			(
				echo '>>> ' $i
				cat /tmp/out
				echo
			) 1>&3
		fi
	fi
done | (tee /dev/fd/2 | awk '{print $1}' | sort | uniq -c) 2>&1
) 3>test.log
