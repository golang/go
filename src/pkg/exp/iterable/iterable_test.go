// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterable

import (
	"testing";
)

type IntArray []int

func (arr IntArray) Iter() <-chan interface{} {
	ch := make(chan interface{});
	go func() {
		for _, x := range arr {
			ch <- x;
		}
		close(ch);
	}();
	return ch;
}

var oneToFive = IntArray{1, 2, 3, 4, 5}

func isNegative(n interface{}) bool {
	return n.(int) < 0;
}
func isPositive(n interface{}) bool {
	return n.(int) > 0;
}
func isAbove3(n interface{}) bool {
	return n.(int) > 3;
}
func isEven(n interface{}) bool {
	return n.(int) % 2 == 0;
}
func doubler(n interface{}) interface{} {
	return n.(int) * 2;
}
func addOne(n interface{}) interface{} {
	return n.(int) + 1;
}
func adder(acc interface{}, n interface{}) interface{} {
	return acc.(int) + n.(int);
}

// A stream of the natural numbers: 0, 1, 2, 3, ...
type integerStream struct{}

func (i integerStream) Iter() <-chan interface{} {
	ch := make(chan interface{});
	go func() {
		for i := 0; ; i++ {
			ch <- i;
		}
	}();
	return ch;
}

func TestAll(t *testing.T) {
	if !All(oneToFive, isPositive) {
		t.Error("All(oneToFive, isPositive) == false");
	}
	if All(oneToFive, isAbove3) {
		t.Error("All(oneToFive, isAbove3) == true");
	}
}

func TestAny(t *testing.T) {
	if Any(oneToFive, isNegative) {
		t.Error("Any(oneToFive, isNegative) == true");
	}
	if !Any(oneToFive, isEven) {
		t.Error("Any(oneToFive, isEven) == false");
	}
}

func assertArraysAreEqual(t *testing.T, res []interface{}, expected []int) {
	if len(res) != len(expected) {
		t.Errorf("len(res) = %v, want %v", len(res), len(expected));
		goto missing;
	}
	for i := range res {
		if v := res[i].(int); v != expected[i] {
			t.Errorf("res[%v] = %v, want %v", i, v, expected[i]);
			goto missing;
		}
	}
	return;
missing:
	t.Errorf("res = %v\nwant  %v", res, expected);
}

func TestFilter(t *testing.T) {
	ints := integerStream{};
	moreInts := Filter(ints, isAbove3).Iter();
	res := make([]interface{}, 3);
	for i := 0; i < 3; i++ {
		res[i] = <-moreInts;
	}
	assertArraysAreEqual(t, res, []int{4, 5, 6});
}

func TestFind(t *testing.T) {
	ints := integerStream{};
	first := Find(ints, isAbove3);
	if first.(int) != 4 {
		t.Errorf("Find(ints, isAbove3) = %v, want 4", first);
	}
}

func TestInject(t *testing.T) {
	res := Inject(oneToFive, 0, adder);
	if res.(int) != 15 {
		t.Errorf("Inject(oneToFive, 0, adder) = %v, want 15", res);
	}
}

func TestMap(t *testing.T) {
	res := Data(Map(Map(oneToFive, doubler), addOne));
	assertArraysAreEqual(t, res, []int{3, 5, 7, 9, 11});
}

func TestPartition(t *testing.T) {
	ti, fi := Partition(oneToFive, isEven);
	assertArraysAreEqual(t, Data(ti), []int{2, 4});
	assertArraysAreEqual(t, Data(fi), []int{1, 3, 5});
}
