// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterable

import (
	"container/vector"
	"testing"
)

func TestArrayTypes(t *testing.T) {
	// Test that conversion works correctly.
	bytes := ByteArray([]byte{1, 2, 3})
	if x := Data(bytes)[1].(byte); x != 2 {
		t.Error("Data(bytes)[1].(byte) = %v, want 2", x)
	}
	ints := IntArray([]int{1, 2, 3})
	if x := Data(ints)[2].(int); x != 3 {
		t.Error("Data(ints)[2].(int) = %v, want 3", x)
	}
	floats := FloatArray([]float{1, 2, 3})
	if x := Data(floats)[0].(float); x != 1 {
		t.Error("Data(floats)[0].(float) = %v, want 1", x)
	}
	strings := StringArray([]string{"a", "b", "c"})
	if x := Data(strings)[1].(string); x != "b" {
		t.Error(`Data(strings)[1].(string) = %q, want "b"`, x)
	}
}

var (
	oneToFive      = IntArray{1, 2, 3, 4, 5}
	sixToTen       = IntArray{6, 7, 8, 9, 10}
	elevenToTwenty = IntArray{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
)

func isNegative(n interface{}) bool     { return n.(int) < 0 }
func isPositive(n interface{}) bool     { return n.(int) > 0 }
func isAbove3(n interface{}) bool       { return n.(int) > 3 }
func isEven(n interface{}) bool         { return n.(int)%2 == 0 }
func doubler(n interface{}) interface{} { return n.(int) * 2 }
func addOne(n interface{}) interface{}  { return n.(int) + 1 }
func adder(acc interface{}, n interface{}) interface{} {
	return acc.(int) + n.(int)
}

// A stream of the natural numbers: 0, 1, 2, 3, ...
type integerStream struct{}

func (i integerStream) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go func() {
		for i := 0; ; i++ {
			ch <- i
		}
	}()
	return ch
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

func assertArraysAreEqual(t *testing.T, res []interface{}, expected []int) {
	if len(res) != len(expected) {
		t.Errorf("len(res) = %v, want %v", len(res), len(expected))
		goto missing
	}
	for i := range res {
		if v := res[i].(int); v != expected[i] {
			t.Errorf("res[%v] = %v, want %v", i, v, expected[i])
			goto missing
		}
	}
	return
missing:
	t.Errorf("res = %v\nwant  %v", res, expected)
}

func TestFilter(t *testing.T) {
	ints := integerStream{}
	moreInts := Filter(ints, isAbove3).Iter()
	res := make([]interface{}, 3)
	for i := 0; i < 3; i++ {
		res[i] = <-moreInts
	}
	assertArraysAreEqual(t, res, []int{4, 5, 6})
}

func TestFind(t *testing.T) {
	ints := integerStream{}
	first := Find(ints, isAbove3)
	if first.(int) != 4 {
		t.Errorf("Find(ints, isAbove3) = %v, want 4", first)
	}
}

func TestInject(t *testing.T) {
	res := Inject(oneToFive, 0, adder)
	if res.(int) != 15 {
		t.Errorf("Inject(oneToFive, 0, adder) = %v, want 15", res)
	}
}

func TestMap(t *testing.T) {
	res := Data(Map(Map(oneToFive, doubler), addOne))
	assertArraysAreEqual(t, res, []int{3, 5, 7, 9, 11})
}

func TestPartition(t *testing.T) {
	ti, fi := Partition(oneToFive, isEven)
	assertArraysAreEqual(t, Data(ti), []int{2, 4})
	assertArraysAreEqual(t, Data(fi), []int{1, 3, 5})
}

func TestTake(t *testing.T) {
	res := Take(oneToFive, 2)
	assertArraysAreEqual(t, Data(res), []int{1, 2})
	assertArraysAreEqual(t, Data(res), []int{1, 2}) // second test to ensure that .Iter() returns a new channel

	// take none
	res = Take(oneToFive, 0)
	assertArraysAreEqual(t, Data(res), []int{})

	// try to take more than available
	res = Take(oneToFive, 20)
	assertArraysAreEqual(t, Data(res), oneToFive)
}

func TestTakeWhile(t *testing.T) {
	// take some
	res := TakeWhile(oneToFive, func(v interface{}) bool { return v.(int) <= 3 })
	assertArraysAreEqual(t, Data(res), []int{1, 2, 3})
	assertArraysAreEqual(t, Data(res), []int{1, 2, 3}) // second test to ensure that .Iter() returns a new channel

	// take none
	res = TakeWhile(oneToFive, func(v interface{}) bool { return v.(int) > 3000 })
	assertArraysAreEqual(t, Data(res), []int{})

	// take all
	res = TakeWhile(oneToFive, func(v interface{}) bool { return v.(int) < 3000 })
	assertArraysAreEqual(t, Data(res), oneToFive)
}

func TestDrop(t *testing.T) {
	// drop none
	res := Drop(oneToFive, 0)
	assertArraysAreEqual(t, Data(res), oneToFive)
	assertArraysAreEqual(t, Data(res), oneToFive) // second test to ensure that .Iter() returns a new channel

	// drop some
	res = Drop(oneToFive, 2)
	assertArraysAreEqual(t, Data(res), []int{3, 4, 5})
	assertArraysAreEqual(t, Data(res), []int{3, 4, 5}) // second test to ensure that .Iter() returns a new channel

	// drop more than available
	res = Drop(oneToFive, 88)
	assertArraysAreEqual(t, Data(res), []int{})
}

func TestDropWhile(t *testing.T) {
	// drop some
	res := DropWhile(oneToFive, func(v interface{}) bool { return v.(int) < 3 })
	assertArraysAreEqual(t, Data(res), []int{3, 4, 5})
	assertArraysAreEqual(t, Data(res), []int{3, 4, 5}) // second test to ensure that .Iter() returns a new channel

	// test case where all elements are dropped
	res = DropWhile(oneToFive, func(v interface{}) bool { return v.(int) < 100 })
	assertArraysAreEqual(t, Data(res), []int{})

	// test case where none are dropped
	res = DropWhile(oneToFive, func(v interface{}) bool { return v.(int) > 1000 })
	assertArraysAreEqual(t, Data(res), oneToFive)
}

func TestCycle(t *testing.T) {
	res := Cycle(oneToFive)
	exp := []int{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4}

	// read the first nineteen values from the iterable
	out := make([]interface{}, 19)
	for i, it := 0, res.Iter(); i < 19; i++ {
		out[i] = <-it
	}
	assertArraysAreEqual(t, out, exp)

	res2 := Cycle(sixToTen)
	exp2 := []int{6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9}
	for i, it := 0, res2.Iter(); i < 19; i++ {
		out[i] = <-it
	}
	assertArraysAreEqual(t, out, exp2)

	// ensure first iterator was not harmed
	for i, it := 0, res.Iter(); i < 19; i++ {
		out[i] = <-it
	}
	assertArraysAreEqual(t, out, exp)
}

func TestChain(t *testing.T) {

	exp := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	res := Chain([]Iterable{oneToFive, sixToTen, elevenToTwenty})
	assertArraysAreEqual(t, Data(res), exp)

	// reusing the same iterator should produce the same result again
	assertArraysAreEqual(t, Data(res), exp)

	// test short read from Chain
	i := 0
	out := make([]interface{}, 4)
	for v := range res.Iter() {
		out[i] = v
		i++
		if i == len(out) {
			break
		}
	}
	assertArraysAreEqual(t, out, exp[0:4])

	// test zero length array
	res = Chain([]Iterable{})
	assertArraysAreEqual(t, Data(res), []int{})
}

func TestZipWith(t *testing.T) {
	exp := []int{7, 9, 11, 13, 15}

	// f with 2 args and 1 return value
	f := func(a, b interface{}) interface{} { return a.(int) + b.(int) }
	res := ZipWith2(f, oneToFive, sixToTen)
	assertArraysAreEqual(t, Data(res), exp)

	// test again to make sure returns new iter each time
	assertArraysAreEqual(t, Data(res), exp)

	// test a function with 3 args
	f2 := func(a, b, c interface{}) interface{} { return a.(int) + b.(int) + c.(int) }
	res = ZipWith3(f2, oneToFive, sixToTen, oneToFive)
	exp = []int{8, 11, 14, 17, 20}
	assertArraysAreEqual(t, Data(res), exp)

	// test a function with multiple values returned
	f3 := func(a, b interface{}) interface{} { return ([]interface{}{a.(int) + 1, b.(int) + 1}) }
	res = ZipWith2(f3, oneToFive, sixToTen)

	exp2 := [][]int{[]int{2, 7}, []int{3, 8}, []int{4, 9}, []int{5, 10}, []int{6, 11}}
	i := 0
	for v := range res.Iter() {
		out := v.([]interface{})
		assertArraysAreEqual(t, out, exp2[i])
		i++
	}

	// test different length iterators--should stop after shortest is exhausted
	res = ZipWith2(f, elevenToTwenty, oneToFive)
	exp = []int{12, 14, 16, 18, 20}
	assertArraysAreEqual(t, Data(res), exp)
}

func TestSlice(t *testing.T) {
	out := Data(Slice(elevenToTwenty, 2, 6))
	exp := []int{13, 14, 15, 16}
	assertArraysAreEqual(t, out, exp)

	// entire iterable
	out = Data(Slice(elevenToTwenty, 0, len(elevenToTwenty)))
	exp = []int{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	assertArraysAreEqual(t, out, exp)

	// empty slice at offset 0
	exp = []int{}
	out = Data(Slice(elevenToTwenty, 0, 0))
	assertArraysAreEqual(t, out, exp)

	// slice upper bound exceeds length of iterable
	exp = []int{1, 2, 3, 4, 5}
	out = Data(Slice(oneToFive, 0, 88))
	assertArraysAreEqual(t, out, exp)

	// slice upper bounce is lower than lower bound
	exp = []int{}
	out = Data(Slice(oneToFive, 93, 4))
	assertArraysAreEqual(t, out, exp)

	// slice lower bound is greater than len of iterable
	exp = []int{}
	out = Data(Slice(oneToFive, 93, 108))
	assertArraysAreEqual(t, out, exp)
}

func TestRepeat(t *testing.T) {
	res := Repeat(42)
	i := 0
	for v := range res.Iter() {
		if v.(int) != 42 {
			t.Fatal("Repeat returned the wrong value")
		}
		if i == 9 {
			break
		}
		i++
	}
}

func TestRepeatTimes(t *testing.T) {
	res := RepeatTimes(84, 9)
	exp := []int{84, 84, 84, 84, 84, 84, 84, 84, 84}
	assertArraysAreEqual(t, Data(res), exp)
	assertArraysAreEqual(t, Data(res), exp) // second time to ensure new iter is returned

	// 0 repeat
	res = RepeatTimes(7, 0)
	exp = []int{}
	assertArraysAreEqual(t, Data(res), exp)

	// negative repeat
	res = RepeatTimes(7, -3)
	exp = []int{}
	assertArraysAreEqual(t, Data(res), exp)
}

// a type that implements Key for ints
type intkey struct{}

func (v intkey) Key(a interface{}) interface{} {
	return a
}
func (v intkey) Equal(a, b interface{}) bool { return a.(int) == b.(int) }

func TestGroupBy(t *testing.T) {
	in := IntArray{1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5}
	exp := [][]int{[]int{1}, []int{2, 2}, []int{3, 3, 3}, []int{4, 4, 4, 4}, []int{5, 5, 5, 5, 5}}
	i := 0
	for x := range GroupBy(in, intkey{}).Iter() {
		gr := x.(Group)
		if gr.Key.(int) != i+1 {
			t.Fatal("group key wrong; expected", i+1, "but got", gr.Key.(int))
		}
		vals := Data(gr.Vals)
		assertArraysAreEqual(t, vals, exp[i])
		i++
	}
	if i != 5 {
		t.Fatal("did not return expected number of groups")
	}

	// test 0 length Iterable
	for _ = range GroupBy(IntArray([]int{}), &intkey{}).Iter() {
		t.Fatal("iterator should be empty")
	}

	// test case with only uniques
	var out vector.Vector
	for x := range GroupBy(elevenToTwenty, intkey{}).Iter() {
		out.Push(x.(Group).Key)
	}
	assertArraysAreEqual(t, out.Data(), elevenToTwenty)
}

func TestUnique(t *testing.T) {
	in := IntArray([]int{1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5})
	exp := []int{1, 2, 3, 4, 5}
	res := Unique(in, intkey{})
	assertArraysAreEqual(t, Data(res), exp)
	assertArraysAreEqual(t, Data(res), exp) // second time to ensure new iter is returned

	// test case with only uniques
	res = Unique(elevenToTwenty, intkey{})
	assertArraysAreEqual(t, Data(res), elevenToTwenty)
}
