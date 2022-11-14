// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

// This is for issue #29707

package main

/*
#include <pthread.h>

extern void* callbackTraceParser(void*);
typedef void* (*cbTraceParser)(void*);

static void testCallbackTraceParser(cbTraceParser cb) {
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

//export callbackTraceParser
func callbackTraceParser(unsafe.Pointer) unsafe.Pointer {
	time.Sleep(time.Millisecond)
	return nil
}

func CgoTraceParser() {
	buf := new(bytes.Buffer)

	trace.Start(buf)
	C.testCallbackTraceParser(C.cbTraceParser(C.callbackTraceParser))
	trace.Stop()

	_, err := traceparser.Parse(buf, "")
	if err == traceparser.ErrTimeOrder {
		fmt.Println("ErrTimeOrder")
	} else if err != nil {
		fmt.Println("Parse error: ", err)
	} else {
		fmt.Println("OK")
	}
}
