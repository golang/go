// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package cgotest

/*

// It's going to be hard to include a whole real JVM to test this.
// So we'll simulate a really easy JVM using just the parts we need.

// This is the relevant part of jni.h.

struct _jobject;

typedef struct _jobject *jobject;
typedef jobject jclass;
typedef jobject jthrowable;
typedef jobject jstring;
typedef jobject jarray;
typedef jarray jbooleanArray;
typedef jarray jbyteArray;
typedef jarray jcharArray;
typedef jarray jshortArray;
typedef jarray jintArray;
typedef jarray jlongArray;
typedef jarray jfloatArray;
typedef jarray jdoubleArray;
typedef jarray jobjectArray;

typedef jobject jweak;

// Note: jvalue is already a non-pointer type due to it being a C union.

*/
import "C"
import (
	"testing"
)

func test22906(t *testing.T) {
	var x1 C.jobject = 0 // Note: 0, not nil. That makes sure we use uintptr for these types.
	_ = x1
	var x2 C.jclass = 0
	_ = x2
	var x3 C.jthrowable = 0
	_ = x3
	var x4 C.jstring = 0
	_ = x4
	var x5 C.jarray = 0
	_ = x5
	var x6 C.jbooleanArray = 0
	_ = x6
	var x7 C.jbyteArray = 0
	_ = x7
	var x8 C.jcharArray = 0
	_ = x8
	var x9 C.jshortArray = 0
	_ = x9
	var x10 C.jintArray = 0
	_ = x10
	var x11 C.jlongArray = 0
	_ = x11
	var x12 C.jfloatArray = 0
	_ = x12
	var x13 C.jdoubleArray = 0
	_ = x13
	var x14 C.jobjectArray = 0
	_ = x14
	var x15 C.jweak = 0
	_ = x15
}
