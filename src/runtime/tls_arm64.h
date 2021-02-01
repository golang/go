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
#define TLS_darwin
#endif
#ifdef GOOS_ios
#define TLS_darwin
#endif
#ifdef TLS_darwin
#define TPIDR TPIDRRO_EL0
#define TLSG_IS_VARIABLE
#define MRS_TPIDR_R0 WORD $0xd53bd060 // MRS TPIDRRO_EL0, R0
#endif

#ifdef GOOS_freebsd
#define TPIDR TPIDR_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040 // MRS TPIDR_EL0, R0
#endif

#ifdef GOOS_netbsd
#define TPIDR TPIDRRO_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040 // MRS TPIDRRO_EL0, R0
#endif

#ifdef GOOS_openbsd
#define TPIDR TPIDR_EL0
#define MRS_TPIDR_R0 WORD $0xd53bd040 // MRS TPIDR_EL0, R0
#endif

#ifdef GOOS_windows
#define TLS_windows
#endif
#ifdef TLS_windows
#define TLSG_IS_VARIABLE
#define MRS_TPIDR_R0 MOVD R18_PLATFORM, R0
#endif

// Define something that will break the build if
// the GOOS is unknown.
#ifndef MRS_TPIDR_R0
#define MRS_TPIDR_R0 unknown_TLS_implementation_in_tls_arm64_h
#endif
