// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows

package main

// Regression test for https://go.dev/issue/72870. Go code called from C should
// never be reported as external code.

/*
#include <pthread.h>

void go_callback1();
void go_callback2();

static void *callback_pprof_thread(void *arg) {
    go_callback1();
    return 0;
}

static void c_callback(void) {
    go_callback2();
}

static void start_callback_pprof_thread() {
    pthread_t th;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_create(&th, &attr, callback_pprof_thread, 0);
    // Don't join, caller will watch pprof.
}
*/
import "C"

import (
	"bytes"
	"fmt"
	"internal/profile"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("CgoCallbackPprof", CgoCallbackPprof)
}

func CgoCallbackPprof() {
	C.start_callback_pprof_thread()

	var buf bytes.Buffer
	if err := pprof.StartCPUProfile(&buf); err != nil {
		fmt.Printf("Error starting CPU profile: %v\n", err)
		os.Exit(1)
	}
	time.Sleep(1 * time.Second)
	pprof.StopCPUProfile()

	p, err := profile.Parse(&buf)
	if err != nil {
		fmt.Printf("Error parsing profile: %v\n", err)
		os.Exit(1)
	}

	foundCallee := false
	for _, s := range p.Sample {
		funcs := flattenFrames(s)
		if len(funcs) == 0 {
			continue
		}

		leaf := funcs[0]
		if leaf.Name != "main.go_callback1_callee" {
			continue
		}
		foundCallee = true

		if len(funcs) < 2 {
			fmt.Printf("Profile: %s\n", p)
			frames := make([]string, len(funcs))
			for i := range funcs {
				frames[i] = funcs[i].Name
			}
			fmt.Printf("FAIL: main.go_callback1_callee sample missing caller in frames %v\n", frames)
			os.Exit(1)
		}

		if funcs[1].Name != "main.go_callback1" {
			// In https://go.dev/issue/72870, this will be runtime._ExternalCode.
			fmt.Printf("Profile: %s\n", p)
			frames := make([]string, len(funcs))
			for i := range funcs {
				frames[i] = funcs[i].Name
			}
			fmt.Printf("FAIL: main.go_callback1_callee sample caller got %s want main.go_callback1 in frames %v\n", funcs[1].Name, frames)
			os.Exit(1)
		}
	}

	if !foundCallee {
		fmt.Printf("Missing main.go_callback1_callee sample in profile %s\n", p)
		os.Exit(1)
	}

	fmt.Printf("OK\n")
}

// Return the frame functions in s, regardless of inlining.
func flattenFrames(s *profile.Sample) []*profile.Function {
	ret := make([]*profile.Function, 0, len(s.Location))
	for _, loc := range s.Location {
		for _, line := range loc.Line {
			ret = append(ret, line.Function)
		}
	}
	return ret
}

//export go_callback1
func go_callback1() {
	// This is a separate function just to ensure we have another Go
	// function as the caller in the profile.
	go_callback1_callee()
}

func go_callback1_callee() {
	C.c_callback()

	// Spin for CPU samples.
	for {
	}
}

//export go_callback2
func go_callback2() {
}
