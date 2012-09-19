// skip

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdio

/*
#include <stdio.h>

// on mingw, stderr and stdout are defined as &_iob[FILENO]
// on netbsd, they are defined as &__sF[FILENO]
// and cgo doesn't recognize them, so write a function to get them,
// instead of depending on internals of libc implementation.
FILE *getStdout(void) { return stdout; }
FILE *getStderr(void) { return stderr; }
*/
import "C"

var Stdout = (*File)(C.getStdout())
var Stderr = (*File)(C.getStderr())
