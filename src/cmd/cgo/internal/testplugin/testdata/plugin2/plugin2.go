// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//#include <errno.h>
//#include <string.h>
import "C"

// #include
// void cfunc() {} // uses cgo_topofstack

import (
	"reflect"
	"strings"

	"testplugin/common"
)

func init() {
	_ = strings.NewReplacer() // trigger stack unwind, Issue #18190.
	C.strerror(C.EIO)         // uses cgo_topofstack
	common.X = 2
}

type sameNameReusedInPlugins struct {
	X string
}

type sameNameHolder struct {
	F *sameNameReusedInPlugins
}

func UnexportedNameReuse() {
	h := sameNameHolder{}
	v := reflect.ValueOf(&h).Elem().Field(0)
	newval := reflect.New(v.Type().Elem())
	v.Set(newval)
}

func main() {
	panic("plugin1.main called")
}
