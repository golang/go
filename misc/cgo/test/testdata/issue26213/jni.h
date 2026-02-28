// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// It's going to be hard to include a whole real JVM to test this.
// So we'll simulate a really easy JVM using just the parts we need.

// This is the relevant part of jni.h.

// On Android NDK16, jobject is defined like this in C and C++
typedef void* jobject;

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
