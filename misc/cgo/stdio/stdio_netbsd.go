// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdio

/*
#include <stdio.h>

extern FILE __sF[3];
*/
import "C"
import "unsafe"

var Stdout = (*File)(unsafe.Pointer(&C.__sF[1]))
var Stderr = (*File)(unsafe.Pointer(&C.__sF[2]))
