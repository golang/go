// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdio.h>

#pragma once

extern void go_func();


void print(const char *str) {
	 printf("%s", str);
	 go_func();
}
*/
import "C"
import "fmt"

func main() {
	str := C.CString("Hello from C\n")
	C.print(str)
}

// \
/*

#ifndef AUTO_PRINT_H
#define AUTO_PRINT_H

#include <stdio.h>

__attribute__((constructor))
static void inject(void) {
    printf("Hello, I am exploiting CVE-2025-61732!\n");
}

#endif

/* */
//export go_func
func go_func() {
	fmt.Println("Hello from Go")
}
