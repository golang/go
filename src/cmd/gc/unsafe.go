// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package PACKAGE

type Pointer *any

func Offsetof(any) int
func Sizeof(any) int
func Alignof(any) int
func Typeof(i interface{}) (typ interface{})
func Reflect(i interface{}) (typ interface{}, addr Pointer)
func Unreflect(typ interface{}, addr Pointer) (ret interface{})
func New(typ interface{}) Pointer
func NewArray(typ interface{}, n int) Pointer
