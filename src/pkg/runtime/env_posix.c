// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

Slice syscall·envs;

byte*
runtime·getenv(int8 *s)
{
	int32 i, j;
	intgo len;
	byte *v, *bs;
	String* envv;
	int32 envc;

	bs = (byte*)s;
	len = runtime·findnull(bs);
	envv = (String*)syscall·envs.array;
	envc = syscall·envs.len;
	for(i=0; i<envc; i++){
		if(envv[i].len <= len)
			continue;
		v = envv[i].str;
		for(j=0; j<len; j++)
			if(bs[j] != v[j])
				goto nomatch;
		if(v[len] != '=')
			goto nomatch;
		return v+len+1;
	nomatch:;
	}
	return nil;
}

void (*_cgo_setenv)(byte**);

// Update the C environment if cgo is loaded.
// Called from syscall.Setenv.
void
syscall·setenv_c(String k, String v)
{
	byte *arg[2];
	uintptr len;

	if(_cgo_setenv == nil)
		return;

	// Objects that are explicitly freed must be at least 16 bytes in size,
	// so that they are not allocated using tiny alloc.
	len = k.len + 1;
	if(len < TinySize)
		len = TinySize;
	arg[0] = runtime·malloc(len);
	runtime·memmove(arg[0], k.str, k.len);
	arg[0][k.len] = 0;

	len = v.len + 1;
	if(len < TinySize)
		len = TinySize;
	arg[1] = runtime·malloc(len);
	runtime·memmove(arg[1], v.str, v.len);
	arg[1][v.len] = 0;

	runtime·asmcgocall((void*)_cgo_setenv, arg);
	runtime·free(arg[0]);
	runtime·free(arg[1]);
}
