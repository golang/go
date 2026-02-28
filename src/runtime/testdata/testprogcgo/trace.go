// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
// Defined in trace_*.c.
void cCalledFromGo(void);
*/
import "C"
import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime/trace"
)

func init() {
	register("Trace", Trace)
}

// Trace is used by TestTraceUnwindCGO.
func Trace() {
	file, err := os.CreateTemp("", "testprogcgo_trace")
	if err != nil {
		log.Fatalf("failed to create temp file: %s", err)
	}
	defer file.Close()

	if err := trace.Start(file); err != nil {
		log.Fatal(err)
	}
	defer trace.Stop()

	goCalledFromGo()
	<-goCalledFromCThreadChan

	fmt.Printf("trace path:%s", file.Name())
}

// goCalledFromGo calls cCalledFromGo which calls back into goCalledFromC and
// goCalledFromCThread.
func goCalledFromGo() {
	C.cCalledFromGo()
}

//export goCalledFromC
func goCalledFromC() {
	trace.Log(context.Background(), "goCalledFromC", "")
}

var goCalledFromCThreadChan = make(chan struct{})

//export goCalledFromCThread
func goCalledFromCThread() {
	trace.Log(context.Background(), "goCalledFromCThread", "")
	close(goCalledFromCThreadChan)
}
