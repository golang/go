// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T1 struct { x, y int; }
type T2 struct { z, w byte; }
type T3 T1

type MyInt int
func (MyInt) m(*T1) { }

func main() {
	{
		var i interface{} = new(T1);
		v1, ok1 := i.(*T1);
		v2, ok2 := i.(*T2);
		v3, ok3 := i.(*T3);
		if !ok1 || ok2 || ok3 {
			panicln("*T1", ok1, ok2, ok3);
		}
	}
	{
		var i interface{} = MyInt(0);
		v1, ok1 := i.(interface{ m(*T1) });
		v2, ok2 := i.(interface{ m(*T2) });
		v3, ok3 := i.(interface{ m(*T3) });
		if !ok1 || ok2 || ok3 {
			panicln("T", ok1, ok2, ok3);
		}
	}
}

