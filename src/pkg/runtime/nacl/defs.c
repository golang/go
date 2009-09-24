// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Input to godefs.

godefs -f-m32 -f-I/home/rsc/pub/nacl/native_client/src/third_party/nacl_sdk/linux/sdk/nacl-sdk/nacl/include -f-I/home/rsc/pub/nacl/native_client defs.c >386/defs.h
*/

#define __native_client__ 1

#define suseconds_t nacl_suseconds_t_1
#include <sys/types.h>
#undef suseconds_t

#include <sys/mman.h>

enum {
	$PROT_NONE = PROT_NONE,
	$PROT_READ = PROT_READ,
	$PROT_WRITE = PROT_WRITE,
	$PROT_EXEC = PROT_EXEC,

	$MAP_ANON = MAP_ANONYMOUS,
	$MAP_PRIVATE = MAP_PRIVATE,
};
