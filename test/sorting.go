// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import Sort "sort"

func main() {
	{	data := []int{74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586};
		a := Sort.IntArray{&data};
		
		Sort.Sort(&a);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/
		
		if !Sort.IsSorted(&a) {
			panic();
		}
	}

	{	data := []float{74.3, 59.0, 238.2, -784.0, 2.3, 9845.768, -959.7485, 905, 7.8, 7.8};
		a := Sort.FloatArray{&data};
		
		Sort.Sort(&a);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/
		
		if !Sort.IsSorted(&a) {
			panic();
		}
	}

	{	data := []string{"", "Hello", "foo", "bar", "foo", "f00", "%*&^*&^&", "***"};
		a := Sort.StringArray{&data};
		
		Sort.Sort(&a);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/
		
		if !Sort.IsSorted(&a) {
			panic();
		}
	}
	
	// Same tests again, this time using the convenience wrappers
	
	{	data := []int{74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586};
		
		Sort.SortInts(&data);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/
		
		if !Sort.IntsAreSorted(&data) {
			panic();
		}
	}

	{	data := []float{74.3, 59.0, 238.2, -784.0, 2.3, 9845.768, -959.7485, 905, 7.8, 7.8};
		
		Sort.SortFloats(&data);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/
		
		if !Sort.FloatsAreSorted(&data) {
			panic();
		}
	}

	{	data := []string{"", "Hello", "foo", "bar", "foo", "f00", "%*&^*&^&", "***"};
		
		Sort.SortStrings(&data);

		/*
		for i := 0; i < len(data); i++ {
			print(data[i], " ");
		}
		print("\n");
		*/
		
		if !Sort.StringsAreSorted(&data) {
			panic();
		}
	}
}
