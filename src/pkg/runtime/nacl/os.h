// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

int32 runtime·thread_create(void(*fn)(void), void *stk, void *tls, int32 tlssize);
void runtime·close(int32);
int32 runtime·mutex_create(void);
int32 runtime·mutex_lock(int32);
int32 runtime·mutex_unlock(int32);
