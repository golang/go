// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows
// +build !go1.3

package svc

// from go12.c
func getServiceMain(r *uintptr)
