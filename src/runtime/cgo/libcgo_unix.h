// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Call pthread_create, retrying on EAGAIN.
 */
int _cgo_try_pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
