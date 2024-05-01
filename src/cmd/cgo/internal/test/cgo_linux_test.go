// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgotest

import (
	"runtime"
	"testing"
)

func TestSetgid(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("unsupported on Android")
	}
	testSetgid(t)
}

func TestSetgidStress(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("unsupported on Android")
	}
	testSetgidStress(t)
}

func Test1435(t *testing.T) { test1435(t) }
func Test6997(t *testing.T) { test6997(t) }
func Test9400(t *testing.T) { test9400(t) }

func TestBuildID(t *testing.T) { testBuildID(t) }
