// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func
main()
{
	s := vlong(0);
	for i:=short(0); i<10; i=i+1 {
		s = s + vlong(i);
	}
	if s != 45 { panic s; }

	s := float(0);
	for i:=0; i<10; i=i+1 {
		s = s + float(i);
	}
	if s != 45 { panic s; }
}
