#!/bin/sh
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
. $GOROOT/src/Make.$GOARCH

gc() {
	$GC $1.go; $LD $1.$O
}

gc_B() {
	$GC -B $1.go; $LD $1.$O
}

run() {
	echo -n '	'$1'	'
	$1
	shift
	(/home/r/plan9/bin/time $* 2>&1 >/dev/null) |  sed 's/r.*/r/'
}

echo 'fasta -n 25000000'
run 'gcc -O2 fasta.c' a.out 25000000
#run 'gccgo -O2 fasta.go' a.out -n 25000000	#commented out until WriteString is in bufio
run 'gc fasta' $O.out -n 25000000
run 'gc_B fasta' $O.out -n 25000000

echo
6.out -n 25000000 > x
echo 'reverse-complement < output-of-fasta-25000000'
run 'gcc -O2 reverse-complement.c' a.out 25000000 < x
run 'gccgo -O2 reverse-complement.go' a.out -n 25000000 < x
run 'gc reverse-complement' $O.out -n 25000000 < x
run 'gc_B reverse-complement' $O.out -n 25000000 < x
rm x

