// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the useany checker.

package a

type Any interface{}

func _[T interface{}]()                    {} // want "could use \"any\" for this empty interface"
func _[X any, T interface{}]()             {} // want "could use \"any\" for this empty interface"
func _[any interface{}]()                  {} // want "could use \"any\" for this empty interface"
func _[T Any]()                            {} // want "could use \"any\" for this empty interface"
func _[T interface{ int | interface{} }]() {} // want "could use \"any\" for this empty interface"
func _[T interface{ int | Any }]()         {} // want "could use \"any\" for this empty interface"
func _[T any]()                            {}

type _[T interface{}] int                    // want "could use \"any\" for this empty interface"
type _[X any, T interface{}] int             // want "could use \"any\" for this empty interface"
type _[any interface{}] int                  // want "could use \"any\" for this empty interface"
type _[T Any] int                            // want "could use \"any\" for this empty interface"
type _[T interface{ int | interface{} }] int // want "could use \"any\" for this empty interface"
type _[T interface{ int | Any }] int         // want "could use \"any\" for this empty interface"
type _[T any] int
