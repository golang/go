/* Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

/* A trivial example of wrapping a C library using SWIG.  */

%{
#include <stdio.h>
#include <stdlib.h>
%}

%typemap(gotype) const char * "string"
%typemap(in) const char * %{
	$1 = malloc($input.n + 1);
	memcpy($1, $input.p, $input.n);
	$1[$input.n] = '\0';
%}
%typemap(freearg) const char * %{
	free($1);
%}

FILE *fopen(const char *name, const char *mode);
int fclose(FILE *);
int fgetc(FILE *);
