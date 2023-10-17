// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simple operations on arrays.

package main

var b[10] float32;

func
main() {
	var a[10] float32;

	for i:=int16(5); i<10; i=i+1 {
		a[i] = float32(i);
	}

	s1 := float32(0);
	for i:=5; i<10; i=i+1 {
		s1 = s1 + a[i];
	}

	if s1 != 35 { panic(s1); }

	for i:=int16(5); i<10; i=i+1 {
		b[i] = float32(i);
	}

	s2 := float32(0);
	for i:=5; i<10; i=i+1 {
		s2 = s2 + b[i];
	}

	if s2 != 35 { panic(s2); }

	b := new([100]int);
	for i:=0; i<100; i=i+1 {
		b[i] = i;
	}

	s3 := 0;
	for i:=0; i<100; i=i+1 {
		s3 = s3+b[i];
	}

	if s3 != 4950 { panic(s3); }
}
