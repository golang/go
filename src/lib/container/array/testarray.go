// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package array

import "array"
import "testing"
import "sort"

export func TestInit(t *testing.T) {
	var a array.Array;
	if a.Init(0).Len() != 0 { t.FailNow() }
	if a.Init(1).Len() != 1 { t.FailNow() }
	if a.Init(10).Len() != 10 { t.FailNow() }
}


export func TestNew(t *testing.T) {
	if array.New(0).Len() != 0 { t.FailNow() }
	if array.New(1).Len() != 1 { t.FailNow() }
	if array.New(10).Len() != 10 { t.FailNow() }
}


export func Val(i int) int {
	return i*991 - 1234
}


export func TestAccess(t *testing.T) {
	const n = 100;
	var a array.Array;
	a.Init(n);
	for i := 0; i < n; i++ {
		a.Set(i, Val(i));
	}
	for i := 0; i < n; i++ {
		if a.At(i).(int) != Val(i) { t.FailNow() }
	}
}


export func TestInsertRemoveClear(t *testing.T) {
	const n = 100;
	a := array.New(0);

	for i := 0; i < n; i++ {
		if a.Len() != i { t.FailNow() }
		a.Insert(0, Val(i));
		if a.Last().(int) != Val(0) { t.FailNow() }
	}
	for i := n-1; i >= 0; i-- {
		if a.Last().(int) != Val(0) { t.FailNow() }
		if a.Remove(0).(int) != Val(i) { t.FailNow() }
		if a.Len() != i { t.FailNow() }
	}

	if a.Len() != 0 { t.FailNow() }
	for i := 0; i < n; i++ {
		a.Push(Val(i));
		if a.Len() != i+1 { t.FailNow() }
		if a.Last().(int) != Val(i) { t.FailNow() }
	}
	a.Init(0);
	if a.Len() != 0 { t.FailNow() }

	const m = 5;
	for j := 0; j < m; j++ {
		a.Push(j);
		for i := 0; i < n; i++ {
			x := Val(i);
			a.Push(x);
			if a.Pop().(int) != x { t.FailNow() }
			if a.Len() != j+1 { t.FailNow() }
		}
	}
	if a.Len() != m { t.FailNow() }
}
