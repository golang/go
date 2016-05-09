// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// golang.org/issue/6612
// Test new scheme for deciding whether C.name is an expression, type, constant.
// Clang silences some warnings when the name is a #defined macro, so test those too
// (even though we now use errors exclusively, not warnings).

package cgotest

/*
void myfunc(void) {}
int myvar = 5;
const char *mytext = "abcdef";
typedef int mytype;
enum {
	myenum = 1234,
};

#define myfunc_def myfunc
#define myvar_def myvar
#define mytext_def mytext
#define mytype_def mytype
#define myenum_def myenum
#define myint_def 12345
#define myfloat_def 1.5
#define mystring_def "hello"
*/
import "C"

import "testing"

func testNaming(t *testing.T) {
	C.myfunc()
	C.myfunc_def()
	if v := C.myvar; v != 5 {
		t.Errorf("C.myvar = %d, want 5", v)
	}
	if v := C.myvar_def; v != 5 {
		t.Errorf("C.myvar_def = %d, want 5", v)
	}
	if s := C.GoString(C.mytext); s != "abcdef" {
		t.Errorf("C.mytext = %q, want %q", s, "abcdef")
	}
	if s := C.GoString(C.mytext_def); s != "abcdef" {
		t.Errorf("C.mytext_def = %q, want %q", s, "abcdef")
	}
	if c := C.myenum; c != 1234 {
		t.Errorf("C.myenum = %v, want 1234", c)
	}
	if c := C.myenum_def; c != 1234 {
		t.Errorf("C.myenum_def = %v, want 1234", c)
	}
	{
		const c = C.myenum
		if c != 1234 {
			t.Errorf("C.myenum as const = %v, want 1234", c)
		}
	}
	{
		const c = C.myenum_def
		if c != 1234 {
			t.Errorf("C.myenum as const = %v, want 1234", c)
		}
	}
	if c := C.myint_def; c != 12345 {
		t.Errorf("C.myint_def = %v, want 12345", c)
	}
	{
		const c = C.myint_def
		if c != 12345 {
			t.Errorf("C.myint as const = %v, want 12345", c)
		}
	}

	// This would be nice, but it has never worked.
	/*
		if c := C.myfloat_def; c != 1.5 {
			t.Errorf("C.myint_def = %v, want 1.5", c)
		}
		{
			const c = C.myfloat_def
			if c != 1.5 {
			t.Errorf("C.myint as const = %v, want 1.5", c)
			}
		}
	*/

	if s := C.mystring_def; s != "hello" {
		t.Errorf("C.mystring_def = %q, want %q", s, "hello")
	}
}
