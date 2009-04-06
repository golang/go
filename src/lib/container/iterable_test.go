// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterable

import (
	"iterable";
	"testing";
)

type IntArray []int;

func (arr IntArray) Iter() <-chan interface {} {
	ch := make(chan interface {});
	go func() {
		for i, x := range arr {
			ch <- x
		}
		close(ch)
	}();
	return ch
}

var oneToFive IntArray = []int{ 1, 2, 3, 4, 5 };

func isNegative(n interface {}) bool {
	return n.(int) < 0
}
func isPositive(n interface {}) bool {
	return n.(int) > 0
}
func isAbove3(n interface {}) bool {
	return n.(int) > 3
}
func isEven(n interface {}) bool {
	return n.(int) % 2 == 0
}
func doubler(n interface {}) interface {} {
	return n.(int) * 2
}
func addOne(n interface {}) interface {} {
	return n.(int) + 1
}


func TestAll(t *testing.T) {
	if !All(oneToFive, isPositive) {
		t.Error("All(oneToFive, isPositive) == false")
	}
	if All(oneToFive, isAbove3) {
		t.Error("All(oneToFive, isAbove3) == true")
	}
}


func TestAny(t *testing.T) {
	if Any(oneToFive, isNegative) {
		t.Error("Any(oneToFive, isNegative) == true")
	}
	if !Any(oneToFive, isEven) {
		t.Error("Any(oneToFive, isEven) == false")
	}
}


func TestMap(t *testing.T) {
	res := Data(Map(Map(oneToFive, doubler), addOne));
	if len(res) != len(oneToFive) {
		t.Fatal("len(res) = %v, want %v", len(res), len(oneToFive))
	}
	expected := []int{ 3, 5, 7, 9, 11 };
	for i := range res {
		if res[i].(int) != expected[i] {
			t.Errorf("res[%v] = %v, want %v", i, res[i], expected[i])
		}
	}
}
