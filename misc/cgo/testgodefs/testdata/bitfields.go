// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// +build ignore

package main

// This file tests that we don't generate an incorrect field location
// for a bitfield that appears aligned.

/*
struct bitfields {
    unsigned int B1     :  5;
    unsigned int B2     :  1;
    unsigned int B3     :  1;
    unsigned int B4     :  1;
    unsigned int Short1 : 16; // misaligned on 8 bit boundary
    unsigned int B5     :  1;
    unsigned int B6     :  1;
    unsigned int B7     :  1;
    unsigned int B8     :  1;
    unsigned int B9     :  1;
    unsigned int B10    :  3;
    unsigned int Short2 : 16; // alignment is OK
    unsigned int Short3 : 16; // alignment is OK
};
*/
import "C"

type bitfields C.struct_bitfields
