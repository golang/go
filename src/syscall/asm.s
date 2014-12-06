// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#ifdef GOARCH_ppc64
#define RET RETURN
#endif
#ifdef GOARCH_ppc64le
#define RET RETURN
#endif

TEXT ·use(SB),NOSPLIT,$0
	RET
