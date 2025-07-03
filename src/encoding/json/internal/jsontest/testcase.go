// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontest

import (
	"fmt"
	"path"
	"runtime"
)

// TODO(https://go.dev/issue/52751): Replace with native testing support.

// CaseName is a case name annotated with a file and line.
type CaseName struct {
	Name  string
	Where CasePos
}

// Name annotates a case name with the file and line of the caller.
func Name(s string) (c CaseName) {
	c.Name = s
	runtime.Callers(2, c.Where.pc[:])
	return c
}

// CasePos represents a file and line number.
type CasePos struct{ pc [1]uintptr }

func (pos CasePos) String() string {
	frames := runtime.CallersFrames(pos.pc[:])
	frame, _ := frames.Next()
	return fmt.Sprintf("%s:%d", path.Base(frame.File), frame.Line)
}
