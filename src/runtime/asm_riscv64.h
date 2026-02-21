// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Define features that are guaranteed to be supported by setting the GORISCV64 variable.
// If a feature is supported, there's no need to check it at runtime every time.

#ifdef GORISCV64_rva22u64
#define hasZba
#define hasZbb
#define hasZbs
#endif

#ifdef GORISCV64_rva23u64
#define hasV
#define hasZba
#define hasZbb
#define hasZbs
#define hasZfa
#define hasZicond
#define hasZihintpause
#endif
