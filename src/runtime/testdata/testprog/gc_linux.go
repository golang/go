// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"syscall"
	"unsafe"
)

func init() {
	register("GCSpanQueueOutOfMemory", GCSpanQueueOutOfMemory)
}

type spanQueueOOMNode struct {
	p *spanQueueOOMNode
	_ [120]byte
}

var spanQueueOOMRoots []*spanQueueOOMNode

// This is a regression test for https://go.dev/issue/82667, where the GC
// crashed instead of throwing when it could not allocate a span queue.
func GCSpanQueueOutOfMemory() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)

	runtime.GC()

	// Point each oblet of the roots into every span, so that scanning it
	// queues more spans than the first span queue ring holds.
	const spans = 2048
	perSpan := int(8192 / unsafe.Sizeof(spanQueueOOMNode{}))
	nodes := make([]*spanQueueOOMNode, spans*perSpan)
	for i := range nodes {
		nodes[i] = new(spanQueueOOMNode)
	}
	spanQueueOOMRoots = make([]*spanQueueOOMNode, len(nodes))
	for i := range spanQueueOOMRoots {
		spanQueueOOMRoots[i] = nodes[(i%spans)*perSpan+i/spans]
	}

	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		println("reading /proc/self/status:", err.Error())
		return
	}
	var vmData uint64
	for _, line := range bytes.Split(status, []byte{'\n'}) {
		if bytes.HasPrefix(line, []byte("VmData:")) {
			vmData, _ = strconv.ParseUint(string(bytes.Fields(line)[1]), 10, 64)
			break
		}
	}
	if vmData == 0 {
		println("VmData not found")
		return
	}
	lim := &syscall.Rlimit{Cur: vmData << 10, Max: vmData << 10}
	if err := syscall.Setrlimit(syscall.RLIMIT_DATA, lim); err != nil {
		println("setrlimit:", err.Error())
		return
	}

	runtime.GC()
	println("OK")
}
