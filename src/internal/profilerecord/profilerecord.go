// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package profilerecord holds internal types used to represent profiling
// records with deep stack traces.
//
// TODO: Consider moving this to internal/runtime, see golang.org/issue/65355.
package profilerecord

import "unsafe"

type StackRecord struct {
	Stack []uintptr
}

func (r StackRecord) GetStack() []uintptr           { return r.Stack }
func (r StackRecord) GetLabels() unsafe.Pointer     { return nil }
func (r StackRecord) GetGoroutine() GoroutineRecord { return GoroutineRecord{} }

type MemProfileRecord struct {
	AllocBytes, FreeBytes     int64
	AllocObjects, FreeObjects int64
	Stack                     []uintptr
}

func (r *MemProfileRecord) InUseBytes() int64   { return r.AllocBytes - r.FreeBytes }
func (r *MemProfileRecord) InUseObjects() int64 { return r.AllocObjects - r.FreeObjects }

type BlockProfileRecord struct {
	Count  int64
	Cycles int64
	Stack  []uintptr
}

type GoroutineRecord struct {
	Labels unsafe.Pointer
	Stack  []uintptr
}

func (r GoroutineRecord) GetStack() []uintptr           { return r.Stack }
func (r GoroutineRecord) GetLabels() unsafe.Pointer     { return r.Labels }
func (r GoroutineRecord) GetGoroutine() GoroutineRecord { return r }
