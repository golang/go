// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef CGOOPT_H
#define CGOOPT_H

#include <errno.h>
#include <stdint.h>

typedef struct { uint64_t whole; double fraction; } opt_value;

uint64_t opt_scalar(uint64_t value);
opt_value opt_aggregate(uint64_t value);
int opt_errno(int value);
int *opt_retain(int *value);
int opt_retained(void);
void opt_forget(void);
int opt_callback(int value);
int opt_wait(void);
int opt_entered(void);
void opt_release(void);

#endif
