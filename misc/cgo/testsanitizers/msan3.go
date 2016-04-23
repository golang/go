// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
extern int *GoFn(int *);

// Yes, you can have definitions if you use //export, as long as they are weak.
int f(void) __attribute__ ((weak));

int f() {
  int i;
  int *p = GoFn(&i);
  if (*p != 12345)
    return 0;
  return 1;
}
*/
import "C"

//export GoFn
func GoFn(p *C.int) *C.int {
	*p = C.int(12345)
	return p
}

func main() {
	if r := C.f(); r != 1 {
		panic(r)
	}
}
