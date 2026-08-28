// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <windows.h>
#include <winternl.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        return 1;
    }
    // Exhaust the inline slots used by TlsAlloc. Loader-provided
    // static TLS uses a separate index space and must still work.
    for (int i = 0; i < 65; i++) {
        TlsAlloc();
    }

    PNT_TIB tib = (PNT_TIB)NtCurrentTeb();
    void *arbitrary_user_pointer = &arbitrary_user_pointer;
    tib->ArbitraryUserPointer = arbitrary_user_pointer;

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
    if (tib->ArbitraryUserPointer != arbitrary_user_pointer) {
        return 5;
    }
    return 0;
}