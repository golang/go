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

fasta() {
	echo 'fasta -n 25000000'
	run 'gcc -O2 fasta.c' a.out 25000000
	#run 'gccgo -O2 fasta.go' a.out -n 25000000	#commented out until WriteString is in bufio
	run 'gc fasta' $O.out -n 25000000
	run 'gc_B fasta' $O.out -n 25000000
}

revcomp() {
	gcc -O2 fasta.c
	a.out 25000000 > x
	echo 'reverse-complement < output-of-fasta-25000000'
	run 'gcc -O2 reverse-complement.c' a.out < x
	run 'gccgo -O2 reverse-complement.go' a.out < x
	run 'gc reverse-complement' $O.out < x
	run 'gc_B reverse-complement' $O.out < x
	rm x
}

nbody() {
	echo 'nbody -n 50000000'
	run 'gcc -O2 nbody.c' a.out 50000000
	run 'gccgo -O2 nbody.go' a.out -n 50000000
	run 'gc nbody' $O.out -n 50000000
	run 'gc_B nbody' $O.out -n 50000000
}

binarytree() {
	echo 'binary-tree 15 # too slow to use 20'
	run 'gcc -O2 binary-tree.c -lm' a.out 15
	run 'gccgo -O2 binary-tree.go' a.out -n 15
	run 'gccgo -O2 binary-tree-freelist.go' $O.out -n 15
	run 'gc binary-tree' $O.out -n 15
	run 'gc binary-tree-freelist' $O.out -n 15
}

fannkuch() {
	echo 'fannkuch 12'
	run 'gcc -O2 fannkuch.c' a.out 12
	run 'gccgo -O2 fannkuch.go' a.out -n 12
	run 'gc fannkuch' $O.out -n 12
	run 'gc_B fannkuch' $O.out -n 12
}

regexdna() {
	gcc -O2 fasta.c
	a.out 100000 > x
	echo 'regex-dna 100000'
	run 'gcc -O2 regex-dna.c -lpcre' a.out <x
#	run 'gccgo -O2 regex-dna.go' a.out <x	# pages badly; don't run
	run 'gc regex-dna' $O.out <x
	run 'gc_B regex-dna' $O.out <x
	rm x
}

spectralnorm() {
	echo 'spectral-norm 5500'
	run 'gcc -O2 spectral-norm.c -lm' a.out 5500
	run 'gccgo -O2 spectral-norm.go' a.out -n 5500
	run 'gc spectral-norm' $O.out -n 5500
	run 'gc_B spectral-norm' $O.out -n 5500
}

knucleotide() {
	gcc -O2 fasta.c
	a.out 1000000 > x  # should be using 25000000
	echo 'k-nucleotide 1000000'
	run 'gcc -O2 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include k-nucleotide.c -lglib-2.0' a.out <x
	run 'gccgo -O2 k-nucleotide.go' a.out <x	# warning: pages badly!
	run 'gc k-nucleotide' $O.out <x
	run 'gc_B k-nucleotide' $O.out <x
	rm x
}

mandelbrot() {
	echo 'mandelbrot 16000'
	run 'gcc -O2 mandelbrot.c' a.out 16000
	run 'gccgo -O2 mandelbrot.go' a.out -n 16000
	run 'gc mandelbrot' $O.out -n 16000
	run 'gc_B mandelbrot' $O.out -n 16000
}

meteor() {
	echo 'meteor 16000'
	run 'gcc -O2 meteor-contest.c' a.out
	run 'gccgo -O2 meteor-contest.go' a.out
	run 'gc meteor-contest' $O.out
	run 'gc_B  meteor-contest' $O.out
}

pidigits() {
	echo 'pidigits 10000'
	run 'gcc -O2 pidigits.c -lgmp' a.out 10000
#	run 'gccgo -O2 pidigits.go' a.out -n 10000  # uncomment when gccgo library updated
	run 'gc pidigits' $O.out -n 10000
	run 'gc_B  pidigits' $O.out -n 10000
}

case $# in
0)
	run="fasta revcomp nbody binarytree fannkuch regexdna spectralnorm knucleotide mandelbrot meteor pidigits"
	;;
*)
	run=$*
esac

for i in $run
do
	$i
	echo
done
