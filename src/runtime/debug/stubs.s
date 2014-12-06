// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#ifdef GOARCH_arm
#define JMP B
#endif
#ifdef GOARCH_ppc64
#define JMP BR
#endif
#ifdef GOARCH_ppc64le
#define JMP BR
#endif

TEXT ·setMaxStack(SB),NOSPLIT,$0-0
  JMP runtime·setMaxStack(SB)

TEXT ·setGCPercent(SB),NOSPLIT,$0-0
  JMP runtime·setGCPercent(SB)

TEXT ·setPanicOnFault(SB),NOSPLIT,$0-0
  JMP runtime·setPanicOnFault(SB)

TEXT ·setMaxThreads(SB),NOSPLIT,$0-0
  JMP runtime·setMaxThreads(SB)
