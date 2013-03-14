// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

#define AT_NULL		0
#define AT_PLATFORM	15 // introduced in at least 2.6.11
#define AT_HWCAP	16 // introduced in at least 2.6.11
#define AT_RANDOM	25 // introduced in 2.6.29
#define HWCAP_VFP	(1 << 6) // introduced in at least 2.6.11
#define HWCAP_VFPv3	(1 << 13) // introduced in 2.6.30
static uint32 runtime·randomNumber;
uint8  runtime·armArch = 6;	// we default to ARMv6
uint32 runtime·hwcap;	// set by setup_auxv
uint8  runtime·goarm;	// set by 5l

void
runtime·checkgoarm(void)
{
	if(runtime·goarm > 5 && !(runtime·hwcap & HWCAP_VFP)) {
		runtime·printf("runtime: this CPU has no floating point hardware, so it cannot run\n");
		runtime·printf("this GOARM=%d binary. Recompile using GOARM=5.\n", runtime·goarm);
		runtime·exit(1);
	}
	if(runtime·goarm > 6 && !(runtime·hwcap & HWCAP_VFPv3)) {
		runtime·printf("runtime: this CPU has no VFPv3 floating point hardware, so it cannot run\n");
		runtime·printf("this GOARM=%d binary. Recompile using GOARM=6.\n", runtime·goarm);
		runtime·exit(1);
	}
}

#pragma textflag 7
void
runtime·setup_auxv(int32 argc, void *argv_list)
{
	byte **argv;
	byte **envp;
	byte *rnd;
	uint32 *auxv;
	uint32 t;

	argv = &argv_list;

	// skip envp to get to ELF auxiliary vector.
	for(envp = &argv[argc+1]; *envp != nil; envp++)
		;
	envp++;
	
	for(auxv=(uint32*)envp; auxv[0] != AT_NULL; auxv += 2) {
		switch(auxv[0]) {
		case AT_RANDOM: // kernel provided 16-byte worth of random data
			if(auxv[1]) {
				rnd = (byte*)auxv[1];
				runtime·randomNumber = rnd[4] | rnd[5]<<8 | rnd[6]<<16 | rnd[7]<<24;
			}
			break;
		case AT_PLATFORM: // v5l, v6l, v7l
			if(auxv[1]) {
				t = *(uint8*)(auxv[1]+1);
				if(t >= '5' && t <= '7')
					runtime·armArch = t - '0';
			}
			break;
		case AT_HWCAP: // CPU capability bit flags
			runtime·hwcap = auxv[1];
			break;
		}
	}
}

#pragma textflag 7
int64
runtime·cputicks(void)
{
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand1().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// runtime·randomNumber provides better seeding of fastrand1.
	return runtime·nanotime() + runtime·randomNumber;
}
