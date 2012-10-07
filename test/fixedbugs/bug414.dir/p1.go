// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p1

import "fmt"

type Fer interface {
	f() string
}

type Object struct{}

func (this *Object) f() string {
	return "Object.f"
}

func PrintFer(fer Fer) {
	fmt.Sprintln(fer.f())
}
