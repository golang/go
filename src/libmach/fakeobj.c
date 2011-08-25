// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * obj.c
 * routines universal to all object files
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <ar.h>
#include <mach.h>
#include "obj.h"

int _is2(char* x) { USED(x); return 0; }
int _is7(char* x) { USED(x); return 0; }
int _is9(char* x) { USED(x); return 0; }
int _isk(char* x) { USED(x); return 0; }
int _isq(char* x) { USED(x); return 0; }
int _isv(char* x) { USED(x); return 0; }
int _isu(char* x) { USED(x); return 0; }
int _read2(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
int _read7(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
int _read9(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
int _readk(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
int _readq(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
int _readv(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
int _readu(Biobuf* b, Prog* p) { USED(b); USED(p); return 0; }
