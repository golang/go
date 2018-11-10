// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context_test

import (
	. "context"
	"testing"
)

func TestBackground(t *testing.T)                      { XTestBackground(t) }
func TestTODO(t *testing.T)                            { XTestTODO(t) }
func TestWithCancel(t *testing.T)                      { XTestWithCancel(t) }
func TestParentFinishesChild(t *testing.T)             { XTestParentFinishesChild(t) }
func TestChildFinishesFirst(t *testing.T)              { XTestChildFinishesFirst(t) }
func TestDeadline(t *testing.T)                        { XTestDeadline(t) }
func TestTimeout(t *testing.T)                         { XTestTimeout(t) }
func TestCanceledTimeout(t *testing.T)                 { XTestCanceledTimeout(t) }
func TestValues(t *testing.T)                          { XTestValues(t) }
func TestAllocs(t *testing.T)                          { XTestAllocs(t, testing.Short, testing.AllocsPerRun) }
func TestSimultaneousCancels(t *testing.T)             { XTestSimultaneousCancels(t) }
func TestInterlockedCancels(t *testing.T)              { XTestInterlockedCancels(t) }
func TestLayersCancel(t *testing.T)                    { XTestLayersCancel(t) }
func TestLayersTimeout(t *testing.T)                   { XTestLayersTimeout(t) }
func TestCancelRemoves(t *testing.T)                   { XTestCancelRemoves(t) }
func TestWithCancelCanceledParent(t *testing.T)        { XTestWithCancelCanceledParent(t) }
func TestWithValueChecksKey(t *testing.T)              { XTestWithValueChecksKey(t) }
func TestDeadlineExceededSupportsTimeout(t *testing.T) { XTestDeadlineExceededSupportsTimeout(t) }
