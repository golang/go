// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linux-specific system calls
int32	futex(uint32*, int32, uint32, Timespec*, uint32*, uint32);
int32	clone(int32, void*, M*, G*, void(*)(void));

struct Sigaction;
void	rt_sigaction(int64, struct Sigaction*, void*, uint64);
