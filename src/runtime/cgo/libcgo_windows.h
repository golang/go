// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Call _beginthread, aborting on failure.
void _cgo_beginthread(unsigned long (__stdcall *func)(void*), void* arg);
