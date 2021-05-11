// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.8 && !go1.13
// +build go1.8,!go1.13

package sort

import "reflect"

var reflectValueOf = reflect.ValueOf
var reflectSwapper = reflect.Swapper
