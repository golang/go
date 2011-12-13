#!/usr/bin/env bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(gomake --no-print-directory -f ../../../src/Make.inc go-env)
PATH=.:$PATH

havegccgo=false
if which gccgo >/dev/null 2>&1
then
	havegccgo=true
fi

mode=run
case X"$1" in
X-test)
	mode=test
	shift
esac

gc() {
	$GC $1.go; $LD $1.$O
}

gc_B() {
	$GC -B $1.go; $LD $1.$O
}

runonly() {
	if [ $mode = run ]
	then
		"$@"
	fi
}

run() {
	if [ $mode = test ]
	then
		if echo $1 | grep -q '^gc '
		then
			$1	# compile the program
			program=$(echo $1 | sed 's/gc //')
			shift
			echo $program
			$1 <fasta-1000.out > /tmp/$$
			case $program in
			chameneosredux)
				# exact numbers may vary but non-numbers should match
				grep -v '[0-9]' /tmp/$$ > /tmp/$$x
				grep -v '[0-9]' chameneosredux.txt > /tmp/$$y
				cmp /tmp/$$x /tmp/$$y
				rm -f /tmp/$$ /tmp/$$x /tmp/$$y
				;;
			*)
				cmp /tmp/$$ $program.txt
				rm -f /tmp/$$
			esac
		fi
		return
	fi
	if ! $havegccgo && echo $1 | grep -q '^gccgo '
	then
		return
	fi
	echo -n '	'$1'	'
	$1
	shift
	
	echo $((time -p $* >/dev/null) 2>&1) | awk '{print $4 "u " $6 "s " $2 "r"}'
}

fasta() {
	runonly echo 'fasta -n 25000000'
	run 'gcc -O2 fasta.c' a.out 25000000
	run 'gccgo -O2 fasta.go' a.out -n 25000000	#commented out until WriteString is in bufio
	run 'gc fasta' $O.out -n 25000000
	run 'gc_B fasta' $O.out -n 25000000
}

revcomp() {
	runonly gcc -O2 fasta.c
	runonly a.out 25000000 > x
	runonly echo 'reverse-complement < output-of-fasta-25000000'
	run 'gcc -O2 reverse-complement.c' a.out < x
	run 'gccgo -O2 reverse-complement.go' a.out < x
	run 'gc reverse-complement' $O.out < x
	run 'gc_B reverse-complement' $O.out < x
	rm x
}

nbody() {
	runonly echo 'nbody -n 50000000'
	run 'gcc -O2 -lm nbody.c' a.out 50000000
	run 'gccgo -O2 nbody.go' a.out -n 50000000
	run 'gc nbody' $O.out -n 50000000
	run 'gc_B nbody' $O.out -n 50000000
}

binarytree() {
	runonly echo 'binary-tree 15 # too slow to use 20'
	run 'gcc -O2 binary-tree.c -lm' a.out 15
	run 'gccgo -O2 binary-tree.go' a.out -n 15
	run 'gccgo -O2 binary-tree-freelist.go' $O.out -n 15
	run 'gc binary-tree' $O.out -n 15
	run 'gc binary-tree-freelist' $O.out -n 15
}

fannkuch() {
	runonly echo 'fannkuch 12'
	run 'gcc -O2 fannkuch.c' a.out 12
	run 'gccgo -O2 fannkuch.go' a.out -n 12
	run 'gccgo -O2 fannkuch-parallel.go' a.out -n 12
	run 'gc fannkuch' $O.out -n 12
	run 'gc fannkuch-parallel' $O.out -n 12
	run 'gc_B fannkuch' $O.out -n 12
}

regexdna() {
	runonly gcc -O2 fasta.c
	runonly a.out 100000 > x
	runonly echo 'regex-dna 100000'
	run 'gcc -O2 regex-dna.c -lpcre' a.out <x
	run 'gccgo -O2 regex-dna.go' a.out <x
	run 'gccgo -O2 regex-dna-parallel.go' a.out <x
	run 'gc regex-dna' $O.out <x
	run 'gc regex-dna-parallel' $O.out <x
	run 'gc_B regex-dna' $O.out <x
	rm x
}

spectralnorm() {
	runonly echo 'spectral-norm 5500'
	run 'gcc -O2 spectral-norm.c -lm' a.out 5500
	run 'gccgo -O2 spectral-norm.go' a.out -n 5500
	run 'gc spectral-norm' $O.out -n 5500
	run 'gc_B spectral-norm' $O.out -n 5500
}

knucleotide() {
	runonly gcc -O2 fasta.c
	runonly a.out 1000000 > x  # should be using 25000000
	runonly echo 'k-nucleotide 1000000'
	run 'gcc -O2 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include k-nucleotide.c -lglib-2.0' a.out <x
	run 'gccgo -O2 k-nucleotide.go' a.out <x
	run 'gccgo -O2 k-nucleotide-parallel.go' a.out <x
	run 'gc k-nucleotide' $O.out <x
	run 'gc k-nucleotide-parallel' $O.out <x
	run 'gc_B k-nucleotide' $O.out <x
	rm x
}

mandelbrot() {
	runonly echo 'mandelbrot 16000'
	run 'gcc -O2 mandelbrot.c' a.out 16000
	run 'gccgo -O2 mandelbrot.go' a.out -n 16000
	run 'gc mandelbrot' $O.out -n 16000
	run 'gc_B mandelbrot' $O.out -n 16000
}

meteor() {
	runonly echo 'meteor 2098'
	run 'gcc -O2 meteor-contest.c' a.out 2098
	run 'gccgo -O2 meteor-contest.go' a.out -n 2098
	run 'gc meteor-contest' $O.out -n 2098
	run 'gc_B  meteor-contest' $O.out -n 2098
}

pidigits() {
	runonly echo 'pidigits 10000'
	run 'gcc -O2 pidigits.c -lgmp' a.out 10000
	run 'gccgo -O2 pidigits.go' a.out -n 10000
	run 'gc pidigits' $O.out -n 10000
	run 'gc_B  pidigits' $O.out -n 10000
}

threadring() {
	runonly echo 'threadring 50000000'
	run 'gcc -O2 threadring.c -lpthread' a.out 50000000
	run 'gccgo -O2 threadring.go' a.out -n 50000000
	run 'gc threadring' $O.out -n 50000000
}

chameneos() {
	runonly echo 'chameneos 6000000'
	run 'gcc -O2 chameneosredux.c -lpthread' a.out 6000000
	run 'gccgo -O2 chameneosredux.go' a.out 6000000
	run 'gc chameneosredux' $O.out 6000000
}

case $# in
0)
	run="fasta revcomp nbody binarytree fannkuch regexdna spectralnorm knucleotide mandelbrot meteor pidigits threadring chameneos"
	;;
*)
	run=$*
esac

for i in $run
do
	$i
	runonly echo
done
