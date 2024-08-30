// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern int weaksym __attribute__((__weak__));
int weaksym = 42;

int foo1()
{
	return weaksym;
}
