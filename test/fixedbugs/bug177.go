// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
import "reflect"
type S1 struct { i int }
type S2 struct { S1 }
func main() {
	typ := reflect.Typeof(S2{}).(*reflect.StructType);
	f := typ.Field(0);
	if f.Name != "S1" || f.Anonymous != true {
		println("BUG: ", f.Name, f.Anonymous);
		return;
	}
	f, ok := typ.FieldByName("S1");
	if !ok {
		println("BUG: missing S1");
		return;
	}
	if !f.Anonymous {
		println("BUG: S1 is not anonymous");
		return;
	}
}
