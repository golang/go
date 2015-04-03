// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef GOOS_linux
#define TPIDR TPIDR_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040
#endif

// Define something that will break the build if
// the GOOS is unknown.
#ifndef TPIDR
#define MRS_TPIDR_R0 TPIDR_UNKNOWN
#endif
