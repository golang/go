// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for cgo.

package cgotest

/*
// issue 1222
typedef union {
	long align;
} xxpthread_mutex_t;

struct ibv_async_event {
	union {
		int x;
	} element;
};

struct ibv_context {
	xxpthread_mutex_t mutex;
};
*/
import "C"

type AsyncEvent struct {
	event C.struct_ibv_async_event
}
