// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>

// Issue 432 - enum fields in struct can cause misaligned struct fields
typedef enum {
	a
} T1;

struct T2 {
	uint8_t a;
	T1 b;
	T1 c;
	uint16_t d;
};

typedef struct T2 T2;
typedef T2 $T2;

// Issue 1162 - structs with fields named Pad[0-9]+ conflict with field
// names used by godefs for padding
struct T3 {
	uint8_t a;
	int Pad0;
};

typedef struct T3 $T3;

// Issue 1466 - forward references to types in stabs debug info were
// always treated as enums
struct T4 {};

struct T5 {
	struct T4 *a;
};

typedef struct T5 T5;
typedef struct T4 $T4;
typedef T5 $T5;