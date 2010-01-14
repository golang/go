// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The following function allows one to dynamically
// resolve DLL function names.
// The arguments are strings.
void *get_proc_addr(void *library, void *name);

// Call a Windows function with stdcall conventions.
void *stdcall(void *fn, ...);
void *stdcall_raw(void *fn, ...);

extern void *VirtualAlloc;

#define goargs mingw_goargs
void mingw_goargs(void);

// Get start address of symbol data in memory.
void *get_symdat_addr(void);
