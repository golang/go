// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "textflag.h"

#define AT_NULL		0
#define AT_RANDOM	25
#define AT_SYSINFO	32
extern uint32 runtime·_vdso;

#pragma textflag NOSPLIT
void
runtime·linux_setup_vdso(int32 argc, byte **argv)
{
	byte **envp;
	uint32 *auxv;

	// skip envp to get to ELF auxiliary vector.
	for(envp = &argv[argc+1]; *envp != nil; envp++)
		;
	envp++;
	
	for(auxv=(uint32*)envp; auxv[0] != AT_NULL; auxv += 2) {
		if(auxv[0] == AT_SYSINFO) {
			runtime·_vdso = auxv[1];
			continue;
		}
		if(auxv[0] == AT_RANDOM) {
			runtime·startup_random_data = (byte*)auxv[1];
			runtime·startup_random_data_len = 16;
			continue;
		}
	}
}
