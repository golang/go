// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

int32 thread_create(void(*fn)(void), void *stk, void *tls, int32 tlssize);
void close(int32);
int32 mutex_create(void);
int32 mutex_lock(int32);
int32 mutex_unlock(int32);
