// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

// Public race detection API, present iff build with -race.

package runtime

import (
	"unsafe"
)

// RaceDisable disables handling of race events in the current goroutine.
func RaceDisable()

// RaceEnable re-enables handling of race events in the current goroutine.
func RaceEnable()

func RaceAcquire(addr unsafe.Pointer)
func RaceRelease(addr unsafe.Pointer)
func RaceReleaseMerge(addr unsafe.Pointer)

func RaceRead(addr unsafe.Pointer)
func RaceWrite(addr unsafe.Pointer)

func RaceSemacquire(s *uint32)
func RaceSemrelease(s *uint32)
