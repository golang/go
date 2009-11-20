// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exception

import "testing"

func TestNoException(t *testing.T) {
	e := Try(func(throw Handler) {});
	if e != nil {
		t.Fatalf("no exception expected, found: %v", e)
	}
}


func TestNilException(t *testing.T) {
	e := Try(func(throw Handler) { throw(nil) });
	if e == nil {
		t.Fatalf("exception expected", e)
	}
	if e.Value != nil {
		t.Fatalf("nil exception expected, found: %v", e)
	}
}


func TestTry(t *testing.T) {
	s := 0;
	for i := 1; i <= 10; i++ {
		e := Try(func(throw Handler) {
			if i%3 == 0 {
				throw(i);
				panic("throw returned");
			}
		});
		if e != nil {
			s += e.Value.(int)
		}
	}
	result := 3 + 6 + 9;
	if s != result {
		t.Fatalf("expected: %d, found: %d", result, s)
	}
}


func TestCatch(t *testing.T) {
	s := 0;
	for i := 1; i <= 10; i++ {
		Try(func(throw Handler) {
			if i%3 == 0 {
				throw(i)
			}
		}).Catch(func(x interface{}) { s += x.(int) })
	}
	result := 3 + 6 + 9;
	if s != result {
		t.Fatalf("expected: %d, found: %d", result, s)
	}
}
