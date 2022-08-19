// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"os"
	"runtime"
	"testing"
)

func TestSetgid(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("unsupported on Android")
	}
	if _, err := os.Stat("/etc/alpine-release"); err == nil {
		t.Skip("setgid is broken with musl libc - go.dev/issue/39857")
	}
	testSetgid(t)
}

func TestSetgidStress(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("unsupported on Android")
	}
	if _, err := os.Stat("/etc/alpine-release"); err == nil {
		t.Skip("setgid is broken with musl libc - go.dev/issue/39857")
	}
	testSetgidStress(t)
}

func Test1435(t *testing.T)    { test1435(t) }
func Test6997(t *testing.T)    { test6997(t) }
func TestBuildID(t *testing.T) { testBuildID(t) }
