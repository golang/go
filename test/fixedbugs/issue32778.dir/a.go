// Copyright 2019 The Go Authors. All rights reserved. Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package a

import "strings"

type Name string

type FullName string

func (n FullName) Name() Name {
	if i := strings.LastIndexByte(string(n), '.'); i >= 0 {
		return Name(n[i+1:])
	}
	return Name(n)
}
