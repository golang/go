#!/usr/bin/env bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(go tool dist env)
GC="go tool compile"
LD="go tool link"

gccm=""
case "$O" in
8)
	gccm=-m32;;
6)
	gccm=-m64;;
esac

EXE="out"
havepcre=true
haveglib=true
havegmp=true
case "$(uname)" in
*MINGW* | *WIN32* | *CYGWIN*)
	havepcre=false
	haveglib=false
	havegmp=false
	if which pkg-config >/dev/null 2>&1; then
		if pkg-config --cflags libpcre >/dev/null 2>&1
		then
			echo "havepcre"
			havepcre=true
		fi
		if pkg-config --cflags glib-2.0 >/dev/null 2>&1
		then
			haveglib=true
		fi
		if pkg-config --cflags gmp >/dev/null 2>&1
		then
			havegmp=true
		fi
	fi
	EXE=exe;;
esac

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
	$GC $1.go; $LD -o a.$EXE $1.o
}

gc_B() {
	$GC -B $1.go; $LD -o a.$EXE $1.o
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
			$1 <fasta-1000.txt > /tmp/$$
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
	run "gcc $gccm -O2 fasta.c" a.$EXE 25000000
	run 'gccgo -O2 fasta.go' a.$EXE -n 25000000	#commented out until WriteString is in bufio
	run 'gc fasta' a.$EXE -n 25000000
	run 'gc_B fasta' a.$EXE -n 25000000
}

revcomp() {
	runonly gcc -O2 fasta.c
	runonly a.$EXE 25000000 > x
	runonly echo 'reverse-complement < output-of-fasta-25000000'
	run "gcc $gccm -O2 reverse-complement.c" a.$EXE < x
	run 'gccgo -O2 reverse-complement.go' a.$EXE < x
	run 'gc reverse-complement' a.$EXE < x
	run 'gc_B reverse-complement' a.$EXE < x
	rm x
}

nbody() {
	runonly echo 'nbody -n 50000000'
	run "gcc $gccm -O2 nbody.c -lm" a.$EXE 50000000
	run 'gccgo -O2 nbody.go' a.$EXE -n 50000000
	run 'gc nbody' a.$EXE -n 50000000
	run 'gc_B nbody' a.$EXE -n 50000000
}

binarytree() {
	runonly echo 'binary-tree 15 # too slow to use 20'
	run "gcc $gccm -O2 binary-tree.c -lm" a.$EXE 15
	run 'gccgo -O2 binary-tree.go' a.$EXE -n 15
	run 'gccgo -O2 binary-tree-freelist.go' a.$EXE -n 15
	run 'gc binary-tree' a.$EXE -n 15
	run 'gc binary-tree-freelist' a.$EXE -n 15
}

fannkuch() {
	runonly echo 'fannkuch 12'
	run "gcc $gccm -O2 fannkuch.c" a.$EXE 12
	run 'gccgo -O2 fannkuch.go' a.$EXE -n 12
	run 'gccgo -O2 fannkuch-parallel.go' a.$EXE -n 12
	run 'gc fannkuch' a.$EXE -n 12
	run 'gc fannkuch-parallel' a.$EXE -n 12
	run 'gc_B fannkuch' a.$EXE -n 12
}

regexdna() {
	runonly gcc -O2 fasta.c
	runonly a.$EXE 100000 > x
	runonly echo 'regex-dna 100000'
	if  $havepcre; then
		run "gcc $gccm -O2 regex-dna.c $(pkg-config libpcre --cflags --libs)" a.$EXE <x
	fi
	run 'gccgo -O2 regex-dna.go' a.$EXE <x
	run 'gccgo -O2 regex-dna-parallel.go' a.$EXE <x
	run 'gc regex-dna' a.$EXE <x
	run 'gc regex-dna-parallel' a.$EXE <x
	run 'gc_B regex-dna' a.$EXE <x
	rm x
}

spectralnorm() {
	runonly echo 'spectral-norm 5500'
	run "gcc $gccm -O2 spectral-norm.c -lm" a.$EXE 5500
	run 'gccgo -O2 spectral-norm.go' a.$EXE -n 5500
	run 'gc spectral-norm' a.$EXE -n 5500
	run 'gc_B spectral-norm' a.$EXE -n 5500
}

knucleotide() {
	runonly gcc -O2 fasta.c
	runonly a.$EXE 1000000 > x  # should be using 25000000
	runonly echo 'k-nucleotide 1000000'
	if [ $mode = run ] && $haveglib; then
		run "gcc -O2 k-nucleotide.c $(pkg-config glib-2.0 --cflags --libs)" a.$EXE <x
	fi
	run 'gccgo -O2 k-nucleotide.go' a.$EXE <x
	run 'gccgo -O2 k-nucleotide-parallel.go' a.$EXE <x
	run 'gc k-nucleotide' a.$EXE <x
	run 'gc k-nucleotide-parallel' a.$EXE <x
	run 'gc_B k-nucleotide' a.$EXE <x
	rm x
}

mandelbrot() {
	runonly echo 'mandelbrot 16000'
	run "gcc $gccm -O2 mandelbrot.c" a.$EXE 16000
	run 'gccgo -O2 mandelbrot.go' a.$EXE -n 16000
	run 'gc mandelbrot' a.$EXE -n 16000
	run 'gc_B mandelbrot' a.$EXE -n 16000
}

meteor() {
	runonly echo 'meteor 2098'
	run "gcc $gccm -O2 meteor-contest.c" a.$EXE 2098
	run 'gccgo -O2 meteor-contest.go' a.$EXE -n 2098
	run 'gc meteor-contest' a.$EXE -n 2098
	run 'gc_B  meteor-contest' a.$EXE -n 2098
}

pidigits() {
	runonly echo 'pidigits 10000'
	if  $havegmp; then
		run "gcc $gccm -O2 pidigits.c -lgmp" a.$EXE 10000
	fi
	run 'gccgo -O2 pidigits.go' a.$EXE -n 10000
	run 'gc pidigits' a.$EXE -n 10000
	run 'gc_B  pidigits' a.$EXE -n 10000
}

threadring() {
	runonly echo 'threadring 50000000'
	run "gcc $gccm -O2 threadring.c -lpthread" a.$EXE 50000000
	run 'gccgo -O2 threadring.go' a.$EXE -n 50000000
	run 'gc threadring' a.$EXE -n 50000000
}

chameneos() {
	runonly echo 'chameneos 6000000'
	run "gcc $gccm -O2 chameneosredux.c -lpthread" a.$EXE 6000000
	run 'gccgo -O2 chameneosredux.go' a.$EXE 6000000
	run 'gc chameneosredux' a.$EXE 6000000
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

rm *.o *.$EXE # Clean up

