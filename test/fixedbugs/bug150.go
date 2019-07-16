// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug150

type T int
func (t T) M()

type M interface { M() } 

func g() (T, T)

func f() (a, b M) {
	a, b = g();
	return;
}

/*
bugs/bug150.go:13: reorder2: too many funcation calls evaluating parameters
*/
