// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package array

import "array"

export func TestInit() bool {
	var a array.Array;
	if a.Init(0).Len() != 0 { return false }
	if a.Init(1).Len() != 1 { return false }
	if a.Init(10).Len() != 10 { return false }
	return true;
}


export func TestNew() bool {
	if array.New(0).Len() != 0 { return false }
	if array.New(1).Len() != 1 { return false }
	if array.New(10).Len() != 10 { return false }
	return true;
}


export func Val(i int) int {
	return i*991 - 1234
}


export func TestAccess() bool {
	const n = 100;
	var a array.Array;
	a.Init(n);
	for i := 0; i < n; i++ {
		a.Set(i, Val(i));
	}
	for i := 0; i < n; i++ {
		if a.At(i).(int) != Val(i) { return false }
	}
	return true;
}


export func TestInsertRemoveClear() bool {
	const n = 100;
	a := array.New(0);

	for i := 0; i < n; i++ {
		if a.Len() != i { return false }
		a.Insert(0, Val(i));
		if a.Last().(int) != Val(0) { return false }
	}
	for i := n-1; i >= 0; i-- {
		if a.Last().(int) != Val(0) { return false }
		if a.Remove(0).(int) != Val(i) { return false }
		if a.Len() != i { return false }
	}

	if a.Len() != 0 { return false }
	for i := 0; i < n; i++ {
		a.Push(Val(i));
		if a.Len() != i+1 { return false }
		if a.Last().(int) != Val(i) { return false }
	}
	a.Init(0);
	if a.Len() != 0 { return false }

	const m = 5;
	for j := 0; j < m; j++ {
		a.Push(j);
		for i := 0; i < n; i++ {
			x := Val(i);
			a.Push(x);
			if a.Pop().(int) != x { return false }
			if a.Len() != j+1 { return false }
		}
	}
	if a.Len() != m { return false }

	return true;
}
