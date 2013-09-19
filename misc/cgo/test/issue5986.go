// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>

static void output5986()
{
    int current_row = 0, row_count = 0;
    double sum_squares = 0;
    do {
        if (current_row == 10) {
            current_row = 0;
        }
        ++row_count;
    }
    while (current_row++ != 1);
    double d =  sqrt(sum_squares / row_count);
    printf("sqrt is: %g\n", d);
}
*/
import "C"
import "testing"

func test5986(t *testing.T) {
	C.output5986()
}
