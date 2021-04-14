// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(jniFix)
}

var jniFix = fix{
	name:     "jni",
	date:     "2017-12-04",
	f:        jnifix,
	desc:     `Fixes initializers of JNI's jobject and subtypes`,
	disabled: false,
}

// Old state:
//   type jobject *_jobject
// New state:
//   type jobject uintptr
// and similar for subtypes of jobject.
// This fix finds nils initializing these types and replaces the nils with 0s.
func jnifix(f *ast.File) bool {
	return typefix(f, func(s string) bool {
		switch s {
		case "C.jobject":
			return true
		case "C.jclass":
			return true
		case "C.jthrowable":
			return true
		case "C.jstring":
			return true
		case "C.jarray":
			return true
		case "C.jbooleanArray":
			return true
		case "C.jbyteArray":
			return true
		case "C.jcharArray":
			return true
		case "C.jshortArray":
			return true
		case "C.jintArray":
			return true
		case "C.jlongArray":
			return true
		case "C.jfloatArray":
			return true
		case "C.jdoubleArray":
			return true
		case "C.jobjectArray":
			return true
		case "C.jweak":
			return true
		}
		return false
	})
}
