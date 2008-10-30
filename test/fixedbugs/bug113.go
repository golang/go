// $G $D/$F.go && $L $F.$A && (! ./$A.out || echo BUG: should not succeed)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
type I interface { };
func foo1(i int) int { return i }
func foo2(i int32) int32 { return i }
func main() {
  var i I;
  i = 1;
  var v1 int = i;
  if foo1(v1) != 1 { panicln(1) }
  var v2 int32 = i.(int).(int32);
  if foo2(v2) != 1 { panicln(2) }
  var v3 int32 = i; // This implicit type conversion should fail at runtime.
  if foo2(v3) != 1 { panicln(3) }
}
