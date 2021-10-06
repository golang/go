// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc
// +build gc

package cpu

// haveAsmFunctions reports whether the other functions in this file can
// be safely called.
func haveAsmFunctions() bool { return true }

// The following feature detection functions are defined in cpu_s390x.s.
// They are likely to be expensive to call so the results should be cached.
func stfle() facilityList
func kmQuery() queryResult
func kmcQuery() queryResult
func kmctrQuery() queryResult
func kmaQuery() queryResult
func kimdQuery() queryResult
func klmdQuery() queryResult
