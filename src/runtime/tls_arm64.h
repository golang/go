// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef GOOS_android
#define TLS_linux
#define TLSG_IS_VARIABLE
#endif
#ifdef GOOS_linux
#define TLS_linux
#endif
#ifdef TLS_linux
#define TPIDR TPIDR_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040 // MRS TPIDR_EL0, R0
#endif

#ifdef GOOS_darwin
#define TPIDR TPIDRRO_EL0
#define TLSG_IS_VARIABLE
#define MRS_TPIDR_R0 WORD $0xd53bd060 // MRS TPIDRRO_EL0, R0
#endif

#ifdef GOOS_netbsd
#define TPIDR TPIDRRO_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040 // MRS TPIDRRO_EL0, R0
#endif

#ifdef GOOS_openbsd
#define TPIDR TPIDR_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040 // MRS TPIDR_EL0, R0
#endif

// Define something that will break the build if
// the GOOS is unknown.
#ifndef TPIDR
#define MRS_TPIDR_R0 TPIDR_UNKNOWN
#endif
