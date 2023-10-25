// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <windows.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        return 1;
    }
    // Allocate more than 64 TLS indices
    // so the Go runtime doesn't find
    // enough space in the TEB TLS slots.
    for (int i = 0; i < 65; i++) {
        TlsAlloc();
    }
    HMODULE hlib = LoadLibrary(argv[1]);
    if (hlib == NULL) {
        return 2;
    }
    FARPROC proc = GetProcAddress(hlib, argv[2]);
    if (proc == NULL) {
        return 3;
    }
    if (proc() != 42) {
        return 4;
    }
    return 0;
}