// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sort"

func ints() {
	data := []int(74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586);
	a := sort.IntArray(data);
	sort.Sort(a);
	if !sort.IsSorted(a) {
		panic()
	}
}

func strings() {
	data := []string("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday");
	a := sort.StringArray(data);
	sort.Sort(a);
	if !sort.IsSorted(a) {
		panic()
	}
}

type day struct {
	num        int;
	short_name string;
	long_name  string;
}

type dayArray struct {
	data []*day;
}

func (p *dayArray) Len() int            { return len(p.data); }
func (p *dayArray) Less(i, j int) bool  { return p.data[i].num < p.data[j].num; }
func (p *dayArray) Swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }

func days() {
	Sunday :=    day( 0, "SUN", "Sunday" );
	Monday :=    day( 1, "MON", "Monday" );
	Tuesday :=   day( 2, "TUE", "Tuesday" );
	Wednesday := day( 3, "WED", "Wednesday" );
	Thursday :=  day( 4, "THU", "Thursday" );
	Friday :=    day( 5, "FRI", "Friday" );
	Saturday :=  day( 6, "SAT", "Saturday" );
	data := []*day(&Tuesday, &Thursday, &Sunday, &Monday, &Friday);
	a := dayArray(data);
	sort.Sort(&a);
	if !sort.IsSorted(&a) {
		panic()
	}
	for i, d := range data {
		print(d.long_name, " ")
	}
	print("\n")
}


func main() {
	ints();
	strings();
	days();
}
