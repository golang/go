// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Introduce two pointer types which are distinct, but have the same
// base type. Make sure that both of those pointer types get resolved
// correctly. Before the fix for 26517 if one of these pointer types
// was resolved before the other one was processed, the second one
// would never be resolved.
// Before this issue was fixed this test failed on Windows,
// where va_list expands to a named char* type.

/*
#include <stdarg.h>
typedef va_list TypeOne;
typedef char *TypeTwo;
*/
import "C"

var a C.TypeOne
var b C.TypeTwo
