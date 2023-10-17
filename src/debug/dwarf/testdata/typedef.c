// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Linux ELF:
gcc -gdwarf-2 -m64 -c typedef.c && gcc -gdwarf-2 -m64 -o typedef.elf typedef.o

OS X Mach-O:
gcc -gdwarf-2 -m64 -c typedef.c -o typedef.macho
gcc -gdwarf-4 -m64 -c typedef.c -o typedef.macho4
*/
#include <complex.h>

typedef volatile int* t_ptr_volatile_int;
typedef const char *t_ptr_const_char;
typedef long t_long;
typedef unsigned short t_ushort;
typedef int t_func_int_of_float_double(float, double);
typedef int (*t_ptr_func_int_of_float_double)(float, double);
typedef int (*t_ptr_func_int_of_float_complex)(float complex);
typedef int (*t_ptr_func_int_of_double_complex)(double complex);
typedef int (*t_ptr_func_int_of_long_double_complex)(long double complex);
typedef int *t_func_ptr_int_of_char_schar_uchar(char, signed char, unsigned char);
typedef void t_func_void_of_char(char);
typedef void t_func_void_of_void(void);
typedef void t_func_void_of_ptr_char_dots(char*, ...);
typedef struct my_struct {
	volatile int vi;
	char x : 1;
	int y : 4;
	int z[0];
	long long array[40];
	int zz[0];
} t_my_struct;
typedef struct my_struct1 {
	int zz [1];
} t_my_struct1;
typedef union my_union {
	volatile int vi;
	char x : 1;
	int y : 4;
	long long array[40];
} t_my_union;
typedef enum my_enum {
	e1 = 1,
	e2 = 2,
	e3 = -5,
	e4 = 1000000000000000LL,
} t_my_enum;

typedef struct list t_my_list;
struct list {
	short val;
	t_my_list *next;
};

typedef struct tree {
	struct tree *left, *right;
	unsigned long long val;
} t_my_tree;

t_ptr_volatile_int *a2;
t_ptr_const_char **a3a;
t_long *a4;
t_ushort *a5;
t_func_int_of_float_double *a6;
t_ptr_func_int_of_float_double *a7;
t_func_ptr_int_of_char_schar_uchar *a8;
t_func_void_of_char *a9;
t_func_void_of_void *a10;
t_func_void_of_ptr_char_dots *a11;
t_my_struct *a12;
t_my_struct1 *a12a;
t_my_union *a12b;
t_my_enum *a13;
t_my_list *a14;
t_my_tree *a15;
t_ptr_func_int_of_float_complex *a16;
t_ptr_func_int_of_double_complex *a17;
t_ptr_func_int_of_long_double_complex *a18;

int main()
{
	return 0;
}
