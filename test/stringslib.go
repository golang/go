// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import strings "strings"

func split(s, sep string) *[]string {
	a := strings.split(s, sep);
	b := strings.join(a, sep);
	if b != s {
		print("Split: ", s, " ", sep, " got ", len(a), "\n");
		for i := 0; i < len(a); i++ {
			print(" a[", i, "] = ", a[i], "\n")
		}
		panic("split / join "+s+" "+sep)
	}
	return a
}

func explode(s string) *[]string {
	a := strings.explode(s);
	b := strings.join(a, "");
	if b != s {
		panic("explode / join "+s)
	}
	return a
}

func itoa(i int) string {
	s := strings.itoa(i);
	n, ok := strings.atoi(s);
	if n != i {
		print("itoa: ", i, " ", s, "\n");
		panic("itoa")
	}
	return s
}

func main() {
	abcd := "abcd";
	faces := "☺☻☹";
	commas := "1,2,3,4";
	dots := "1....2....3....4";
	if strings.utflen(abcd) != 4 { panic("utflen abcd") }
	if strings.utflen(faces) != 3 { panic("utflen faces") }
	if strings.utflen(commas) != 7 { panic("utflen commas") }
	{
		a := split(abcd, "a");
		if len(a) != 2 || a[0] != "" || a[1] != "bcd" { panic("split abcd a") }
	}
	{
		a := split(abcd, "z");
		if len(a) != 1 || a[0] != "abcd" { panic("split abcd z") }
	}
	{
		a := split(abcd, "");
		if len(a) != 4 || a[0] != "a" || a[1] != "b" || a[2] != "c" || a[3] != "d" { panic("split abcd empty") }
	}
	{
		a := explode(abcd);
		if len(a) != 4 || a[0] != "a" || a[1] != "b" || a[2] != "c" || a[3] != "d" { panic("explode abcd") }
	}
	{
		a := split(commas, ",");
		if len(a) != 4 || a[0] != "1" || a[1] != "2" || a[2] != "3" || a[3] != "4" { panic("split commas") }
	}
	{
		a := split(dots, "...");
		if len(a) != 4 || a[0] != "1" || a[1] != ".2" || a[2] != ".3" || a[3] != ".4" { panic("split dots") }
	}

	{
		a := split(faces, "☹");
		if len(a) != 2 || a[0] != "☺☻" || a[1] != "" { panic("split faces 1") }
	}
	{
		a := split(faces, "~");
		if len(a) != 1 || a[0] != faces { panic("split faces ~") }
	}
	{
		a := explode(faces);
		if len(a) != 3 || a[0] != "☺" || a[1] != "☻" || a[2] != "☹" { panic("explode faces") }
	}
	{
		a := split(faces, "");
		if len(a) != 3 || a[0] != "☺" || a[1] != "☻" || a[2] != "☹" { panic("split faces empty") }
	}
	
	{
		n, ok := strings.atoi("0"); if n != 0 || !ok { panic("atoi 0") }
		n, ok = strings.atoi("-1"); if n != -1 || !ok { panic("atoi -1") }
		n, ok = strings.atoi("+345"); if n != 345 || !ok { panic("atoi +345") }
		n, ok = strings.atoi("9999"); if n != 9999 || !ok { panic("atoi 9999") }
		n, ok = strings.atoi("20ba"); if n != 0 || ok { panic("atoi 20ba") }
		n, ok = strings.atoi("hello"); if n != 0 || ok { panic("hello") }
	}
	
	if itoa(0) != "0" { panic("itoa 0") }
	if itoa(12345) != "12345" { panic("itoa 12345") }
	if itoa(-1<<31) != "-2147483648" { panic("itoa 1<<31") }
	
	// should work if int == int64: is there some way to know?
	// if itoa(-1<<63) != "-9223372036854775808" { panic("itoa 1<<63") }
}
