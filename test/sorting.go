// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import (
	"fmt";
	"rand";
	"sort";
)

func BentleyMcIlroyTests();

func main() {
	{	data := []int{74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586};
		a := sort.IntArray{&data};

		sort.Sort(&a);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/

		if !sort.IsSorted(&a) {
			panic();
		}
	}

	{	data := []float{74.3, 59.0, 238.2, -784.0, 2.3, 9845.768, -959.7485, 905, 7.8, 7.8};
		a := sort.FloatArray{&data};

		sort.Sort(&a);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/

		if !sort.IsSorted(&a) {
			panic();
		}
	}

	{	data := []string{"", "Hello", "foo", "bar", "foo", "f00", "%*&^*&^&", "***"};
		a := sort.StringArray{&data};

		sort.Sort(&a);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/

		if !sort.IsSorted(&a) {
			panic();
		}
	}

	// Same tests again, this time using the convenience wrappers

	{	data := []int{74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586};

		sort.SortInts(&data);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/

		if !sort.IntsAreSorted(&data) {
			panic();
		}
	}

	{	data := []float{74.3, 59.0, 238.2, -784.0, 2.3, 9845.768, -959.7485, 905, 7.8, 7.8};

		sort.SortFloats(&data);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/

		if !sort.FloatsAreSorted(&data) {
			panic();
		}
	}

	{	data := []string{"", "Hello", "foo", "bar", "foo", "f00", "%*&^*&^&", "***"};

		sort.SortStrings(&data);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/

		if !sort.StringsAreSorted(&data) {
			panic();
		}
	}

	{
		data := new([]int, 100000);
		for i := 0; i < len(data); i++ {
			data[i] = rand.rand() % 100;
		}
		if sort.IntsAreSorted(data) {
			panic("terrible rand.rand");
		}
		sort.SortInts(data);
		if !sort.IntsAreSorted(data) {
			panic();
		}
	}

	BentleyMcIlroyTests();
}

const (
	Sawtooth = iota;
	Rand;
	Stagger;
	Plateau;
	Shuffle;
	NDist;
)

const (
	Copy = iota;
	Reverse;
	ReverseFirstHalf;
	ReverseSecondHalf;
	Sort;
	Dither;
	NMode;
);

type TestingData struct {
	data *[]int;
	maxswap int;	// number of swaps allowed
	nswap int;
}

func (d *TestingData) len() int { return len(d.data); }
func (d *TestingData) less(i, j int) bool { return d.data[i] < d.data[j]; }
func (d *TestingData) swap(i, j int) {
	if d.nswap >= d.maxswap {
		panicln("used", d.nswap, "swaps sorting", len(d.data), "array");
	}
	d.nswap++;
	d.data[i], d.data[j] = d.data[j], d.data[i];
}

func Lg(n int) int {
	i := 0;
	for 1<<uint(i) < n {
		i++;
	}
	return i;
}

func Min(a, b int) int {
	if a < b {
		return a;
	}
	return b;
}

func SortIntsTest(mode int, data, x *[]int) {
	switch mode {
	case Copy:
		for i := 0; i < len(data); i++ {
			x[i] = data[i];
		}
	case Reverse:
		for i := 0; i < len(data); i++ {
			x[i] = data[len(data)-i-1];
		}
	case ReverseFirstHalf:
		n := len(data)/2;
		for i := 0; i < n; i++ {
			x[i] = data[n-i-1];
		}
		for i := n; i < len(data); i++ {
			x[i] = data[i];
		}
	case ReverseSecondHalf:
		n := len(data)/2;
		for i := 0; i < n; i++ {
			x[i] = data[i];
		}
		for i := n; i < len(data); i++ {
			x[i] = data[len(data)-(i-n)-1];
		}
	case Sort:
		for i := 0; i < len(data); i++ {
			x[i] = data[i];
		}
		// sort.SortInts is known to be correct
		// because mode Sort runs after mode Copy.
		sort.SortInts(x[0:len(data)]);
	case Dither:
		for i := 0; i < len(data); i++ {
			x[i] = data[i] + i%5;
		}
	}
	d := &TestingData{x[0:len(data)], len(data)*Lg(len(data))*12/10, 0};
	sort.Sort(d);

	// If we were testing C qsort, we'd have to make a copy
	// of the array and sort it ourselves and then compare
	// x against it, to ensure that qsort was only permuting
	// the data, not (for example) overwriting it with zeros.
	//
	// In go, we don't have to be so paranoid: since the only
	// mutating method sort.Sort can call is TestingData.swap,
	// it suffices here just to check that the final array is sorted.
	if !sort.IntsAreSorted(x[0:len(data)]) {
		panicln("incorrect sort");
	}
}

func BentleyMcIlroyTests() {
	sizes := []int{100, 1023, 1024, 1025};
	var x, tmp [1025]int;
	for ni := 0; ni < len(sizes); ni++ {
		n := sizes[ni];
		for m := 1; m < 2*n; m *= 2 {
			for dist := 0; dist < NDist; dist++ {
				j := 0;
				k := 1;
				for i := 0; i < n; i++ {
					switch dist {
					case Sawtooth:
						x[i] = i % m;
					case Rand:
						x[i] = rand.rand() % m;
					case Stagger:
						x[i] = (i*m + i) % n;
					case Plateau:
						x[i] = Min(i, m);
					case Shuffle:
						if rand.rand() % m != 0 {
							j += 2;
							x[i] = j;
						} else {
							k += 2;
							x[i] = k;
						}
					}
				}
				data := (&x)[0:n];
				for i := 0; i < NMode; i++ {
					SortIntsTest(i, data, &tmp);
				}
			}
		}
	}
}

