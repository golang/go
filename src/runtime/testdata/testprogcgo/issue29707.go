// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

// This is for issue #29707

package main

/*
#include <pthread.h>

extern void* callback(void*);
typedef void* (*cb)(void*);

static void testCallback(cb cb) {
	pthread_t thread_id;
	pthread_create(&thread_id, NULL, cb, NULL);
	pthread_join(thread_id, NULL);
}
*/
import "C"

import (
	"bytes"
	"fmt"
	traceparser "internal/trace"
	"runtime/trace"
	"time"
	"unsafe"
)

func init() {
	register("CgoTraceParser", CgoTraceParser)
}

//export callback
func callback(unsafe.Pointer) unsafe.Pointer {
	time.Sleep(time.Millisecond)
	return nil
}

func CgoTraceParser() {
	buf := new(bytes.Buffer)

	trace.Start(buf)
	C.testCallback(C.cb(C.callback))
	trace.Stop()

	_, err := traceparser.Parse(buf, "")
	if err != nil {
		fmt.Println("Parse error: ", err)
	} else {
		fmt.Println("OK")
	}
}
