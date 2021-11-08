// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for cgo.
// Both the import "C" prologue and the main file are sorted by issue number.
// This file contains C definitions (not just declarations)
// and so it must NOT contain any //export directives on Go functions.
// See testx.go for exports.

package cgotest

/*
#include <complex.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#cgo LDFLAGS: -lm

#ifndef WIN32
#include <pthread.h>
#include <signal.h>
#endif

// alignment tests

typedef unsigned char Uint8;
typedef unsigned short Uint16;

typedef enum {
 MOD1 = 0x0000,
 MODX = 0x8000
} SDLMod;

typedef enum {
 A1 = 1,
 B1 = 322,
 SDLK_LAST
} SDLKey;

typedef struct SDL_keysym {
	Uint8 scancode;
	SDLKey sym;
	SDLMod mod;
	Uint16 unicode;
} SDL_keysym;

typedef struct SDL_KeyboardEvent {
	Uint8 typ;
	Uint8 which;
	Uint8 state;
	SDL_keysym keysym;
} SDL_KeyboardEvent;

void makeEvent(SDL_KeyboardEvent *event) {
 unsigned char *p;
 int i;

 p = (unsigned char*)event;
 for (i=0; i<sizeof *event; i++) {
   p[i] = i;
 }
}

int same(SDL_KeyboardEvent* e, Uint8 typ, Uint8 which, Uint8 state, Uint8 scan, SDLKey sym, SDLMod mod, Uint16 uni) {
  return e->typ == typ && e->which == which && e->state == state && e->keysym.scancode == scan && e->keysym.sym == sym && e->keysym.mod == mod && e->keysym.unicode == uni;
}

void cTest(SDL_KeyboardEvent *event) {
 printf("C: %#x %#x %#x %#x %#x %#x %#x\n", event->typ, event->which, event->state,
   event->keysym.scancode, event->keysym.sym, event->keysym.mod, event->keysym.unicode);
 fflush(stdout);
}

// api

const char *greeting = "hello, world";

// basic test cases

#define SHIFT(x, y)  ((x)<<(y))
#define KILO SHIFT(1, 10)
#define UINT32VAL 0xc008427bU

enum E {
	Enum1 = 1,
	Enum2 = 2,
};

typedef unsigned char cgo_uuid_t[20];

void uuid_generate(cgo_uuid_t x) {
	x[0] = 0;
}

struct S {
	int x;
};

const char *cstr = "abcefghijklmnopqrstuvwxyzABCEFGHIJKLMNOPQRSTUVWXYZ1234567890";

extern enum E myConstFunc(struct S* const ctx, int const id, struct S **const filter);

enum E myConstFunc(struct S *const ctx, int const id, struct S **const filter) { return 0; }

int add(int x, int y) {
	return x+y;
};

// Following mimicks vulkan complex definitions for benchmarking cgocheck overhead.

typedef uint32_t VkFlags;
typedef VkFlags  VkDeviceQueueCreateFlags;
typedef uint32_t VkStructureType;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkDeviceQueueCreateFlags    flags;
    uint32_t                    queueFamilyIndex;
    uint32_t                    queueCount;
    const float*                pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkPhysicalDeviceFeatures {
    uint32_t bools[56];
} VkPhysicalDeviceFeatures;

typedef struct VkDeviceCreateInfo {
    VkStructureType                    sType;
    const void*                        pNext;
    VkFlags                            flags;
    uint32_t                           queueCreateInfoCount;
    const VkDeviceQueueCreateInfo*     pQueueCreateInfos;
    uint32_t                           enabledLayerCount;
    const char* const*                 ppEnabledLayerNames;
    uint32_t                           enabledExtensionCount;
    const char* const*                 ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures*    pEnabledFeatures;
} VkDeviceCreateInfo;

void handleComplexPointer(VkDeviceCreateInfo *a0) {}
void handleComplexPointer8(
	VkDeviceCreateInfo *a0, VkDeviceCreateInfo *a1, VkDeviceCreateInfo *a2, VkDeviceCreateInfo *a3,
	VkDeviceCreateInfo *a4, VkDeviceCreateInfo *a5, VkDeviceCreateInfo *a6, VkDeviceCreateInfo *a7
) {}

// complex alignment

struct {
	float x;
	_Complex float y;
} cplxAlign = { 3.14, 2.17 };

// constants and pointer checking

#define CheckConstVal 0

typedef struct {
	int *p;
} CheckConstStruct;

static void CheckConstFunc(CheckConstStruct *p, int e) {}

// duplicate symbol

int base_symbol = 0;
#define alias_one base_symbol
#define alias_two base_symbol

// function pointer variables

typedef int (*intFunc) ();

int
bridge_int_func(intFunc f)
{
	return f();
}

int fortytwo()
{
	return 42;
}

// issue 1222
typedef union {
	long align;
} xxpthread_mutex_t;
struct ibv_async_event {
	union {
		int x;
	} element;
};
struct ibv_context {
	xxpthread_mutex_t mutex;
};

// issue 1635
// Mac OS X's gcc will generate scattered relocation 2/1 for
// this function on Darwin/386, and 8l couldn't handle it.
// this example is in issue 1635
void scatter() {
	void *p = scatter;
	printf("scatter = %p\n", p);
}

// Adding this explicit extern declaration makes this a test for
// https://gcc.gnu.org/PR68072 aka https://golang.org/issue/13344 .
// It used to cause a cgo error when building with GCC 6.
extern int hola;

// this example is in issue 3253
int hola = 0;
int testHola() { return hola; }

// issue 3250
#ifdef WIN32
void testSendSIG() {}
#else
static void *thread(void *p) {
	const int M = 100;
	int i;
	(void)p;
	for (i = 0; i < M; i++) {
		pthread_kill(pthread_self(), SIGCHLD);
		usleep(rand() % 20 + 5);
	}
	return NULL;
}
void testSendSIG() {
	const int N = 20;
	int i;
	pthread_t tid[N];
	for (i = 0; i < N; i++) {
		usleep(rand() % 200 + 100);
		pthread_create(&tid[i], 0, thread, NULL);
	}
	for (i = 0; i < N; i++)
		pthread_join(tid[i], 0);
}
#endif

// issue 3261
// libgcc on ARM might be compiled as thumb code, but our 5l
// can't handle that, so we have to disable this test on arm.
#ifdef __ARMEL__
int vabs(int x) {
	puts("testLibgcc is disabled on ARM because 5l cannot handle thumb library.");
	return (x < 0) ? -x : x;
}
#elif defined(__arm64__) && defined(__clang__)
int vabs(int x) {
	puts("testLibgcc is disabled on ARM64 with clang due to lack of libgcc.");
	return (x < 0) ? -x : x;
}
#else
int __absvsi2(int); // dummy prototype for libgcc function
// we shouldn't name the function abs, as gcc might use
// the builtin one.
int vabs(int x) { return __absvsi2(x); }
#endif


// issue 3729
// access errno from void C function
const char _expA = 0x42;
const float _expB = 3.14159;
const short _expC = 0x55aa;
const int _expD = 0xdeadbeef;

#ifdef WIN32
void g(void) {}
void g2(int x, char a, float b, short c, int d) {}
#else

void g(void) {
	errno = E2BIG;
}

// try to pass some non-trivial arguments to function g2
void g2(int x, char a, float b, short c, int d) {
	if (a == _expA && b == _expB && c == _expC && d == _expD)
		errno = x;
	else
		errno = -1;
}
#endif

// issue 3945
// Test that cgo reserves enough stack space during cgo call.
// See https://golang.org/issue/3945 for details.
void say() {
	printf("%s from C\n", "hello");
}

// issue 4054 part 1 - other half in testx.go

typedef enum {
	A = 0,
	B,
	C,
	D,
	E,
	F,
	G,
	H,
	II,
	J,
} issue4054a;

// issue 4339
// We've historically permitted #include <>, so test it here.  Issue 29333.
// Also see issue 41059.
#include <issue4339.h>

// issue 4417
// cmd/cgo: bool alignment/padding issue.
// bool alignment is wrong and causing wrong arguments when calling functions.
static int c_bool(bool a, bool b, int c, bool d, bool e)  {
   return c;
}

// issue 4857
#cgo CFLAGS: -Werror
const struct { int a; } *issue4857() { return (void *)0; }

// issue 5224
// Test that the #cgo CFLAGS directive works,
// with and without platform filters.
#cgo CFLAGS: -DCOMMON_VALUE=123
#cgo windows CFLAGS: -DIS_WINDOWS=1
#cgo !windows CFLAGS: -DIS_WINDOWS=0
int common = COMMON_VALUE;
int is_windows = IS_WINDOWS;

// issue 5227
// linker incorrectly treats common symbols and
// leaves them undefined.

typedef struct {
        int Count;
} Fontinfo;

Fontinfo SansTypeface;

extern void init();

Fontinfo loadfont() {
        Fontinfo f = {0};
        return f;
}

void init() {
        SansTypeface = loadfont();
}

// issue 5242
// Cgo incorrectly computed the alignment of structs
// with no Go accessible fields as 0, and then panicked on
// modulo-by-zero computations.
typedef struct {
} foo;

typedef struct {
	int x : 1;
} bar;

int issue5242(foo f, bar b) {
	return 5242;
}

// issue 5337
// Verify that we can withstand SIGPROF received on foreign threads

#ifdef WIN32
void test5337() {}
#else
static void *thread1(void *p) {
	(void)p;
	pthread_kill(pthread_self(), SIGPROF);
	return NULL;
}
void test5337() {
	pthread_t tid;
	pthread_create(&tid, 0, thread1, NULL);
	pthread_join(tid, 0);
}
#endif

// issue 5603

const long long issue5603exp = 0x12345678;
long long issue5603foo0() { return issue5603exp; }
long long issue5603foo1(void *p) { return issue5603exp; }
long long issue5603foo2(void *p, void *q) { return issue5603exp; }
long long issue5603foo3(void *p, void *q, void *r) { return issue5603exp; }
long long issue5603foo4(void *p, void *q, void *r, void *s) { return issue5603exp; }

// issue 5740

int test5740a(void), test5740b(void);

// issue 5986
static void output5986()
{
    int current_row = 0, row_count = 0;
    double sum_squares = 0;
    double d;
    do {
        if (current_row == 10) {
            current_row = 0;
        }
        ++row_count;
    }
    while (current_row++ != 1);
    d =  sqrt(sum_squares / row_count);
    printf("sqrt is: %g\n", d);
}

// issue 6128
// Test handling of #defined names in clang.
// NOTE: Must use hex, or else a shortcut for decimals
// in cgo avoids trying to pass this to clang.
#define X 0x1

// issue 6472
typedef struct
{
        struct
        {
            int x;
        } y[16];
} z;

// issue 6612
// Test new scheme for deciding whether C.name is an expression, type, constant.
// Clang silences some warnings when the name is a #defined macro, so test those too
// (even though we now use errors exclusively, not warnings).

void myfunc(void) {}
int myvar = 5;
const char *mytext = "abcdef";
typedef int mytype;
enum {
	myenum = 1234,
};

#define myfunc_def myfunc
#define myvar_def myvar
#define mytext_def mytext
#define mytype_def mytype
#define myenum_def myenum
#define myint_def 12345
#define myfloat_def 1.5
#define mystring_def "hello"

// issue 6907
char* Issue6907CopyString(_GoString_ s) {
	size_t n;
	const char *p;
	char *r;

	n = _GoStringLen(s);
	p = _GoStringPtr(s);
	r = malloc(n + 1);
	memmove(r, p, n);
	r[n] = '\0';
	return r;
}

// issue 7560
typedef struct {
	char x;
	long y;
} __attribute__((__packed__)) misaligned;

int
offset7560(void)
{
	return (uintptr_t)&((misaligned*)0)->y;
}

// issue 7786
// No runtime test, just make sure that typedef and struct/union/class are interchangeable at compile time.

struct test7786;
typedef struct test7786 typedef_test7786;
void f7786(struct test7786 *ctx) {}
void g7786(typedef_test7786 *ctx) {}

typedef struct body7786 typedef_body7786;
struct body7786 { int x; };
void b7786(struct body7786 *ctx) {}
void c7786(typedef_body7786 *ctx) {}

typedef union union7786 typedef_union7786;
void u7786(union union7786 *ctx) {}
void v7786(typedef_union7786 *ctx) {}

// issue 8092
// Test that linker defined symbols (e.g., text, data) don't
// conflict with C symbols.
char text[] = "text";
char data[] = "data";
char *ctext(void) { return text; }
char *cdata(void) { return data; }

// issue 8428
// Cgo inconsistently translated zero size arrays.

struct issue8428one {
	char b;
	char rest[];
};

struct issue8428two {
	void *p;
	char b;
	char rest[0];
	char pad;
};

struct issue8428three {
	char w[1][2][3][0];
	char x[2][3][0][1];
	char y[3][0][1][2];
	char z[0][1][2][3];
};

// issue 8331 part 1 - part 2 in testx.go
// A typedef of an unnamed struct is the same struct when
// #include'd twice.  No runtime test; just make sure it compiles.
#include "issue8331.h"

// issue 8368 and 8441
// Recursive struct definitions didn't work.
// No runtime test; just make sure it compiles.
typedef struct one one;
typedef struct two two;
struct one {
	two *x;
};
struct two {
	one *x;
};

// issue 8811

extern int issue8811Initialized;
extern void issue8811Init();

void issue8811Execute() {
	if(!issue8811Initialized)
		issue8811Init();
}

// issue 8945

typedef void (*PFunc8945)();
PFunc8945 func8945;

// issue 9557

struct issue9557_t {
  int a;
} test9557bar = { 42 };
struct issue9557_t *issue9557foo = &test9557bar;

// issue 10303
// Pointers passed to C were not marked as escaping (bug in cgo).

typedef int *intptr;

void setintstar(int *x) {
	*x = 1;
}

void setintptr(intptr x) {
	*x = 1;
}

void setvoidptr(void *x) {
	*(int*)x = 1;
}

typedef struct Struct Struct;
struct Struct {
	int *P;
};

void setstruct(Struct s) {
	*s.P = 1;
}

// issue 11925
// Structs with zero-length trailing fields are now padded by the Go compiler.

struct a11925 {
	int i;
	char a[0];
	char b[0];
};

struct b11925 {
	int i;
	char a[0];
	char b[];
};

// issue 12030
void issue12030conv(char *buf, double x) {
	sprintf(buf, "d=%g", x);
}

// issue 14838

int check_cbytes(char *b, size_t l) {
	int i;
	for (i = 0; i < l; i++) {
		if (b[i] != i) {
			return 0;
		}
	}
	return 1;
}

// issue 17065
// Test that C symbols larger than a page play nicely with the race detector.
int ii[65537];

// issue 17537
// The void* cast introduced by cgo to avoid problems
// with const/volatile qualifiers breaks C preprocessor macros that
// emulate functions.

typedef struct {
	int i;
} S17537;

int I17537(S17537 *p);

#define I17537(p) ((p)->i)

// Calling this function used to fail without the cast.
const int F17537(const char **p) {
	return **p;
}

// issue 17723
// API compatibility checks

typedef char *cstring_pointer;
static void cstring_pointer_fun(cstring_pointer dummy) { }
const char *api_hello = "hello!";

// Calling this function used to trigger an error from the C compiler
// (issue 18298).
void F18298(const void *const *p) {
}

// Test that conversions between typedefs work as they used to.
typedef const void *T18298_1;
struct S18298 { int i; };
typedef const struct S18298 *T18298_2;
void G18298(T18298_1 t) {
}

// issue 18126
// cgo check of void function returning errno.
void Issue18126C(void **p) {}

// issue 18720

#define HELLO "hello"
#define WORLD "world"
#define HELLO_WORLD HELLO "\000" WORLD

struct foo { char c; };
#define SIZE_OF(x) sizeof(x)
#define SIZE_OF_FOO SIZE_OF(struct foo)
#define VAR1 VAR
#define VAR var
int var = 5;

#define ADDR &var

#define CALL fn()
int fn(void) {
	return ++var;
}

// issue 20129

int issue20129 = 0;
typedef void issue20129Void;
issue20129Void issue20129Foo() {
	issue20129 = 1;
}
typedef issue20129Void issue20129Void2;
issue20129Void2 issue20129Bar() {
	issue20129 = 2;
}

// issue 20369
#define XUINT64_MAX        18446744073709551615ULL

// issue 21668
// Fail to guess the kind of the constant "x".
// No runtime test; just make sure it compiles.
const int x21668 = 42;

// issue 21708
#define CAST_TO_INT64 (int64_t)(-1)

// issue 21809
// Compile C `typedef` to go type aliases.

typedef long MySigned_t;
// tests alias-to-alias
typedef MySigned_t MySigned2_t;
long takes_long(long x) { return x * x; }
MySigned_t takes_typedef(MySigned_t x) { return x * x; }

// issue 22906

// It's going to be hard to include a whole real JVM to test this.
// So we'll simulate a really easy JVM using just the parts we need.
// This is the relevant part of jni.h.

struct _jobject;

typedef struct _jobject *jobject;
typedef jobject jclass;
typedef jobject jthrowable;
typedef jobject jstring;
typedef jobject jarray;
typedef jarray jbooleanArray;
typedef jarray jbyteArray;
typedef jarray jcharArray;
typedef jarray jshortArray;
typedef jarray jintArray;
typedef jarray jlongArray;
typedef jarray jfloatArray;
typedef jarray jdoubleArray;
typedef jarray jobjectArray;

typedef jobject jweak;

// Note: jvalue is already a non-pointer type due to it being a C union.

// issue 22958

typedef struct {
	unsigned long long f8  : 8;
	unsigned long long f16 : 16;
	unsigned long long f24 : 24;
	unsigned long long f32 : 32;
	unsigned long long f40 : 40;
	unsigned long long f48 : 48;
	unsigned long long f56 : 56;
	unsigned long long f64 : 64;
} issue22958Type;

// issue 23356
int a(void) { return 5; };
int r(void) { return 3; };

// issue 23720
typedef int *issue23720A;
typedef const int *issue23720B;
void issue23720F(issue23720B a) {}

// issue 24206
#if defined(__linux__) && defined(__x86_64__)
#include <sys/mman.h>
// Returns string with null byte at the last valid address
char* dangerousString1() {
	int pageSize = 4096;
	char *data = mmap(0, 2 * pageSize, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	mprotect(data + pageSize,pageSize,PROT_NONE);
	int start = pageSize - 123 - 1; // last 123 bytes of first page + 1 null byte
	int i = start;
	for (; i < pageSize; i++) {
	data[i] = 'x';
	}
	data[pageSize -1 ] = 0;
	return data+start;
}

char* dangerousString2() {
	int pageSize = 4096;
	char *data = mmap(0, 3 * pageSize, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	mprotect(data + 2 * pageSize,pageSize,PROT_NONE);
	int start = pageSize - 123 - 1; // last 123 bytes of first page + 1 null byte
	int i = start;
	for (; i < 2 * pageSize; i++) {
	data[i] = 'x';
	}
	data[2*pageSize -1 ] = 0;
	return data+start;
}
#else
char *dangerousString1() { return NULL; }
char *dangerousString2() { return NULL; }
#endif

// issue 26066
const unsigned long long int issue26066 = (const unsigned long long) -1;

// issue 26517
// Introduce two pointer types which are distinct, but have the same
// base type. Make sure that both of those pointer types get resolved
// correctly. Before the fix for 26517 if one of these pointer types
// was resolved before the other one was processed, the second one
// would never be resolved.
// Before this issue was fixed this test failed on Windows,
// where va_list expands to a named char* type.
typedef va_list TypeOne;
typedef char *TypeTwo;

// issue 28540

static void twoargs1(void *p, int n) {}
static void *twoargs2() { return 0; }
static int twoargs3(void * p) { return 0; }

// issue 28545
// Failed to add type conversion for negative constant.

static void issue28545F(char **p, int n, complex double a) {}

// issue 28772 part 1 - part 2 in testx.go
// Failed to add type conversion for Go constant set to C constant.
// No runtime test; just make sure it compiles.

#define issue28772Constant 1

// issue 28896
// cgo was incorrectly adding padding after a packed struct.
typedef struct {
	void *f1;
	uint32_t f2;
} __attribute__((__packed__)) innerPacked;

typedef struct {
	innerPacked g1;
	uint64_t g2;
} outerPacked;

typedef struct {
	void *f1;
	uint32_t f2;
} innerUnpacked;

typedef struct {
	innerUnpacked g1;
	uint64_t g2;
} outerUnpacked;

size_t offset(int x) {
	switch (x) {
	case 0:
		return offsetof(innerPacked, f2);
	case 1:
		return offsetof(outerPacked, g2);
	case 2:
		return offsetof(innerUnpacked, f2);
	case 3:
		return offsetof(outerUnpacked, g2);
	default:
		abort();
	}
}

// issue 29748

typedef struct { char **p; } S29748;
static int f29748(S29748 *p) { return 0; }

// issue 29781
// Error with newline inserted into constant expression.
// Compilation test only, nothing to run.

static void issue29781F(char **p, int n) {}
#define ISSUE29781C 0

// issue 31093
static uint16_t issue31093F(uint16_t v) { return v; }

// issue 32579
typedef struct S32579 { unsigned char data[1]; } S32579;

// issue 37033, cgo.Handle
extern void GoFunc37033(uintptr_t handle);
void cFunc37033(uintptr_t handle) { GoFunc37033(handle); }

// issue 38649
// Test that #define'd type aliases work.
#define netbsd_gid unsigned int

// issue 40494
// Inconsistent handling of tagged enum and union types.
enum Enum40494 { X_40494 };
union Union40494 { int x; };
void issue40494(enum Enum40494 e, union Union40494* up) {}

// Issue 45451, bad handling of go:notinheap types.
typedef struct issue45451Undefined issue45451;
*/
import "C"

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"reflect"
	"runtime"
	"runtime/cgo"
	"sync"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// alignment

func testAlign(t *testing.T) {
	var evt C.SDL_KeyboardEvent
	C.makeEvent(&evt)
	if C.same(&evt, evt.typ, evt.which, evt.state, evt.keysym.scancode, evt.keysym.sym, evt.keysym.mod, evt.keysym.unicode) == 0 {
		t.Error("*** bad alignment")
		C.cTest(&evt)
		t.Errorf("Go: %#x %#x %#x %#x %#x %#x %#x\n",
			evt.typ, evt.which, evt.state, evt.keysym.scancode,
			evt.keysym.sym, evt.keysym.mod, evt.keysym.unicode)
		t.Error(evt)
	}
}

// api

const greeting = "hello, world"

type testPair struct {
	Name      string
	Got, Want interface{}
}

var testPairs = []testPair{
	{"GoString", C.GoString(C.greeting), greeting},
	{"GoStringN", C.GoStringN(C.greeting, 5), greeting[:5]},
	{"GoBytes", C.GoBytes(unsafe.Pointer(C.greeting), 5), []byte(greeting[:5])},
}

func testHelpers(t *testing.T) {
	for _, pair := range testPairs {
		if !reflect.DeepEqual(pair.Got, pair.Want) {
			t.Errorf("%s: got %#v, want %#v", pair.Name, pair.Got, pair.Want)
		}
	}
}

// basic test cases

const EINVAL = C.EINVAL /* test #define */

var KILO = C.KILO

func uuidgen() {
	var uuid C.cgo_uuid_t
	C.uuid_generate(&uuid[0])
}

func Strtol(s string, base int) (int, error) {
	p := C.CString(s)
	n, err := C.strtol(p, nil, C.int(base))
	C.free(unsafe.Pointer(p))
	return int(n), err
}

func Atol(s string) int {
	p := C.CString(s)
	n := C.atol(p)
	C.free(unsafe.Pointer(p))
	return int(n)
}

func testConst(t *testing.T) {
	C.myConstFunc(nil, 0, nil)
}

func testEnum(t *testing.T) {
	if C.Enum1 != 1 || C.Enum2 != 2 {
		t.Error("bad enum", C.Enum1, C.Enum2)
	}
}

func testNamedEnum(t *testing.T) {
	e := new(C.enum_E)

	*e = C.Enum1
	if *e != 1 {
		t.Error("bad enum", C.Enum1)
	}

	*e = C.Enum2
	if *e != 2 {
		t.Error("bad enum", C.Enum2)
	}
}

func testCastToEnum(t *testing.T) {
	e := C.enum_E(C.Enum1)
	if e != 1 {
		t.Error("bad enum", C.Enum1)
	}

	e = C.enum_E(C.Enum2)
	if e != 2 {
		t.Error("bad enum", C.Enum2)
	}
}

func testAtol(t *testing.T) {
	l := Atol("123")
	if l != 123 {
		t.Error("Atol 123: ", l)
	}
}

func testErrno(t *testing.T) {
	p := C.CString("no-such-file")
	m := C.CString("r")
	f, err := C.fopen(p, m)
	C.free(unsafe.Pointer(p))
	C.free(unsafe.Pointer(m))
	if err == nil {
		C.fclose(f)
		t.Fatalf("C.fopen: should fail")
	}
	if err != syscall.ENOENT {
		t.Fatalf("C.fopen: unexpected error: %v", err)
	}
}

func testMultipleAssign(t *testing.T) {
	p := C.CString("234")
	n, m := C.strtol(p, nil, 345), C.strtol(p, nil, 10)
	if runtime.GOOS == "openbsd" {
		// Bug in OpenBSD strtol(3) - base > 36 succeeds.
		if (n != 0 && n != 239089) || m != 234 {
			t.Fatal("Strtol x2: ", n, m)
		}
	} else if n != 0 || m != 234 {
		t.Fatal("Strtol x2: ", n, m)
	}
	C.free(unsafe.Pointer(p))
}

var (
	cuint  = (C.uint)(0)
	culong C.ulong
	cchar  C.char
)

type Context struct {
	ctx *C.struct_ibv_context
}

func benchCgoCall(b *testing.B) {
	b.Run("add-int", func(b *testing.B) {
		const x = C.int(2)
		const y = C.int(3)

		for i := 0; i < b.N; i++ {
			C.add(x, y)
		}
	})

	b.Run("one-pointer", func(b *testing.B) {
		var a0 C.VkDeviceCreateInfo
		for i := 0; i < b.N; i++ {
			C.handleComplexPointer(&a0)
		}
	})
	b.Run("eight-pointers", func(b *testing.B) {
		var a0, a1, a2, a3, a4, a5, a6, a7 C.VkDeviceCreateInfo
		for i := 0; i < b.N; i++ {
			C.handleComplexPointer8(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7)
		}
	})
	b.Run("eight-pointers-nil", func(b *testing.B) {
		var a0, a1, a2, a3, a4, a5, a6, a7 *C.VkDeviceCreateInfo
		for i := 0; i < b.N; i++ {
			C.handleComplexPointer8(a0, a1, a2, a3, a4, a5, a6, a7)
		}
	})
	b.Run("eight-pointers-array", func(b *testing.B) {
		var a [8]C.VkDeviceCreateInfo
		for i := 0; i < b.N; i++ {
			C.handleComplexPointer8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7])
		}
	})
	b.Run("eight-pointers-slice", func(b *testing.B) {
		a := make([]C.VkDeviceCreateInfo, 8)
		for i := 0; i < b.N; i++ {
			C.handleComplexPointer8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7])
		}
	})
}

// Benchmark measuring overhead from Go to C and back to Go (via a callback)
func benchCallback(b *testing.B) {
	var x = false
	for i := 0; i < b.N; i++ {
		nestedCall(func() { x = true })
	}
	if !x {
		b.Fatal("nestedCall was not invoked")
	}
}

var sinkString string

func benchGoString(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkString = C.GoString(C.cstr)
	}
	const want = "abcefghijklmnopqrstuvwxyzABCEFGHIJKLMNOPQRSTUVWXYZ1234567890"
	if sinkString != want {
		b.Fatalf("%q != %q", sinkString, want)
	}
}

// Static (build-time) test that syntax traversal visits all operands of s[i:j:k].
func sliceOperands(array [2000]int) {
	_ = array[C.KILO:C.KILO:C.KILO] // no type error
}

// set in cgo_thread_lock.go init
var testThreadLockFunc = func(*testing.T) {}

// complex alignment

func TestComplexAlign(t *testing.T) {
	if C.cplxAlign.x != 3.14 {
		t.Errorf("got %v, expected 3.14", C.cplxAlign.x)
	}
	if C.cplxAlign.y != 2.17 {
		t.Errorf("got %v, expected 2.17", C.cplxAlign.y)
	}
}

// constants and pointer checking

func testCheckConst(t *testing.T) {
	// The test is that this compiles successfully.
	p := C.malloc(C.size_t(unsafe.Sizeof(C.int(0))))
	defer C.free(p)
	C.CheckConstFunc(&C.CheckConstStruct{(*C.int)(p)}, C.CheckConstVal)
}

// duplicate symbol

func duplicateSymbols() {
	fmt.Printf("%v %v %v\n", C.base_symbol, C.alias_one, C.alias_two)
}

// environment

// This is really an os package test but here for convenience.
func testSetEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		// Go uses SetEnvironmentVariable on windows. However,
		// C runtime takes a *copy* at process startup of the
		// OS environment, and stores it in environ/envp.
		// It is this copy that	getenv/putenv manipulate.
		t.Logf("skipping test")
		return
	}
	const key = "CGO_OS_TEST_KEY"
	const val = "CGO_OS_TEST_VALUE"
	os.Setenv(key, val)
	keyc := C.CString(key)
	defer C.free(unsafe.Pointer(keyc))
	v := C.getenv(keyc)
	if uintptr(unsafe.Pointer(v)) == 0 {
		t.Fatal("getenv returned NULL")
	}
	vs := C.GoString(v)
	if vs != val {
		t.Fatalf("getenv() = %q; want %q", vs, val)
	}
}

// function pointer variables

func callBridge(f C.intFunc) int {
	return int(C.bridge_int_func(f))
}

func callCBridge(f C.intFunc) C.int {
	return C.bridge_int_func(f)
}

func testFpVar(t *testing.T) {
	const expected = 42
	f := C.intFunc(C.fortytwo)
	res1 := C.bridge_int_func(f)
	if r1 := int(res1); r1 != expected {
		t.Errorf("got %d, want %d", r1, expected)
	}
	res2 := callCBridge(f)
	if r2 := int(res2); r2 != expected {
		t.Errorf("got %d, want %d", r2, expected)
	}
	r3 := callBridge(f)
	if r3 != expected {
		t.Errorf("got %d, want %d", r3, expected)
	}
}

// issue 1222
type AsyncEvent struct {
	event C.struct_ibv_async_event
}

// issue 1635

func test1635(t *testing.T) {
	C.scatter()
	if v := C.hola; v != 0 {
		t.Fatalf("C.hola is %d, should be 0", v)
	}
	if v := C.testHola(); v != 0 {
		t.Fatalf("C.testHola() is %d, should be 0", v)
	}
}

// issue 2470

func testUnsignedInt(t *testing.T) {
	a := (int64)(C.UINT32VAL)
	b := (int64)(0xc008427b)
	if a != b {
		t.Errorf("Incorrect unsigned int - got %x, want %x", a, b)
	}
}

// issue 3250

func test3250(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("not applicable on windows")
	}

	t.Skip("skipped, see golang.org/issue/5885")
	var (
		thres = 1
		sig   = syscall_dot_SIGCHLD
	)
	type result struct {
		n   int
		sig os.Signal
	}
	var (
		sigCh     = make(chan os.Signal, 10)
		waitStart = make(chan struct{})
		waitDone  = make(chan result)
	)

	signal.Notify(sigCh, sig)

	go func() {
		n := 0
		alarm := time.After(time.Second * 3)
		for {
			select {
			case <-waitStart:
				waitStart = nil
			case v := <-sigCh:
				n++
				if v != sig || n > thres {
					waitDone <- result{n, v}
					return
				}
			case <-alarm:
				waitDone <- result{n, sig}
				return
			}
		}
	}()

	waitStart <- struct{}{}
	C.testSendSIG()
	r := <-waitDone
	if r.sig != sig {
		t.Fatalf("received signal %v, but want %v", r.sig, sig)
	}
	t.Logf("got %d signals\n", r.n)
	if r.n <= thres {
		t.Fatalf("expected more than %d", thres)
	}
}

// issue 3261

func testLibgcc(t *testing.T) {
	var table = []struct {
		in, out C.int
	}{
		{0, 0},
		{1, 1},
		{-42, 42},
		{1000300, 1000300},
		{1 - 1<<31, 1<<31 - 1},
	}
	for _, v := range table {
		if o := C.vabs(v.in); o != v.out {
			t.Fatalf("abs(%d) got %d, should be %d", v.in, o, v.out)
			return
		}
	}
}

// issue 3729

func test3729(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping on windows")
	}

	_, e := C.g()
	if e != syscall.E2BIG {
		t.Errorf("got %q, expect %q", e, syscall.E2BIG)
	}
	_, e = C.g2(C.EINVAL, C._expA, C._expB, C._expC, C._expD)
	if e != syscall.EINVAL {
		t.Errorf("got %q, expect %q", e, syscall.EINVAL)
	}
}

// issue 3945

func testPrintf(t *testing.T) {
	C.say()
}

// issue 4054

var issue4054a = []int{C.A, C.B, C.C, C.D, C.E, C.F, C.G, C.H, C.I, C.J}

// issue 4339

func test4339(t *testing.T) {
	C.handle4339(&C.exported4339)
}

// issue 4417

func testBoolAlign(t *testing.T) {
	b := C.c_bool(true, true, 10, true, false)
	if b != 10 {
		t.Fatalf("found %d expected 10\n", b)
	}
	b = C.c_bool(true, true, 5, true, true)
	if b != 5 {
		t.Fatalf("found %d expected 5\n", b)
	}
	b = C.c_bool(true, true, 3, true, false)
	if b != 3 {
		t.Fatalf("found %d expected 3\n", b)
	}
	b = C.c_bool(false, false, 1, true, false)
	if b != 1 {
		t.Fatalf("found %d expected 1\n", b)
	}
	b = C.c_bool(false, true, 200, true, false)
	if b != 200 {
		t.Fatalf("found %d expected 200\n", b)
	}
}

// issue 4857

func test4857() {
	_ = C.issue4857()
}

// issue 5224

func testCflags(t *testing.T) {
	is_windows := C.is_windows == 1
	if is_windows != (runtime.GOOS == "windows") {
		t.Errorf("is_windows: %v, runtime.GOOS: %s", is_windows, runtime.GOOS)
	}
	if C.common != 123 {
		t.Errorf("common: %v (expected 123)", C.common)
	}
}

// issue 5227

func test5227(t *testing.T) {
	C.init()
}

func selectfont() C.Fontinfo {
	return C.SansTypeface
}

// issue 5242

func test5242(t *testing.T) {
	if got := C.issue5242(C.foo{}, C.bar{}); got != 5242 {
		t.Errorf("got %v", got)
	}
}

func test5603(t *testing.T) {
	var x [5]int64
	exp := int64(C.issue5603exp)
	x[0] = int64(C.issue5603foo0())
	x[1] = int64(C.issue5603foo1(nil))
	x[2] = int64(C.issue5603foo2(nil, nil))
	x[3] = int64(C.issue5603foo3(nil, nil, nil))
	x[4] = int64(C.issue5603foo4(nil, nil, nil, nil))
	for i, v := range x {
		if v != exp {
			t.Errorf("issue5603foo%d() returns %v, expected %v", i, v, exp)
		}
	}
}

// issue 5337

func test5337(t *testing.T) {
	C.test5337()
}

// issue 5740

func test5740(t *testing.T) {
	if v := C.test5740a() + C.test5740b(); v != 5 {
		t.Errorf("expected 5, got %v", v)
	}
}

// issue 5986

func test5986(t *testing.T) {
	C.output5986()
}

// issue 6128

func test6128() {
	// nothing to run, just make sure this compiles.
	_ = C.X
}

// issue 6390

func test6390(t *testing.T) {
	p1 := C.malloc(1024)
	if p1 == nil {
		t.Fatalf("C.malloc(1024) returned nil")
	}
	p2 := C.malloc(0)
	if p2 == nil {
		t.Fatalf("C.malloc(0) returned nil")
	}
	C.free(p1)
	C.free(p2)
}

func test6472() {
	// nothing to run, just make sure this compiles
	s := new(C.z)
	println(s.y[0].x)
}

// issue 6506

func test6506() {
	// nothing to run, just make sure this compiles
	var x C.size_t

	C.calloc(x, x)
	C.malloc(x)
	C.realloc(nil, x)
	C.memcpy(nil, nil, x)
	C.memcmp(nil, nil, x)
	C.memmove(nil, nil, x)
	C.strncpy(nil, nil, x)
	C.strncmp(nil, nil, x)
	C.strncat(nil, nil, x)
	x = C.strxfrm(nil, nil, x)
	C.memchr(nil, 0, x)
	x = C.strcspn(nil, nil)
	x = C.strspn(nil, nil)
	C.memset(nil, 0, x)
	x = C.strlen(nil)
	_ = x
}

// issue 6612

func testNaming(t *testing.T) {
	C.myfunc()
	C.myfunc_def()
	if v := C.myvar; v != 5 {
		t.Errorf("C.myvar = %d, want 5", v)
	}
	if v := C.myvar_def; v != 5 {
		t.Errorf("C.myvar_def = %d, want 5", v)
	}
	if s := C.GoString(C.mytext); s != "abcdef" {
		t.Errorf("C.mytext = %q, want %q", s, "abcdef")
	}
	if s := C.GoString(C.mytext_def); s != "abcdef" {
		t.Errorf("C.mytext_def = %q, want %q", s, "abcdef")
	}
	if c := C.myenum; c != 1234 {
		t.Errorf("C.myenum = %v, want 1234", c)
	}
	if c := C.myenum_def; c != 1234 {
		t.Errorf("C.myenum_def = %v, want 1234", c)
	}
	{
		const c = C.myenum
		if c != 1234 {
			t.Errorf("C.myenum as const = %v, want 1234", c)
		}
	}
	{
		const c = C.myenum_def
		if c != 1234 {
			t.Errorf("C.myenum as const = %v, want 1234", c)
		}
	}
	if c := C.myint_def; c != 12345 {
		t.Errorf("C.myint_def = %v, want 12345", c)
	}
	{
		const c = C.myint_def
		if c != 12345 {
			t.Errorf("C.myint as const = %v, want 12345", c)
		}
	}

	if c := C.myfloat_def; c != 1.5 {
		t.Errorf("C.myint_def = %v, want 1.5", c)
	}
	{
		const c = C.myfloat_def
		if c != 1.5 {
			t.Errorf("C.myint as const = %v, want 1.5", c)
		}
	}

	if s := C.mystring_def; s != "hello" {
		t.Errorf("C.mystring_def = %q, want %q", s, "hello")
	}
}

// issue 6907

func test6907(t *testing.T) {
	want := "yarn"
	if got := C.GoString(C.Issue6907CopyString(want)); got != want {
		t.Errorf("C.GoString(C.Issue6907CopyString(%q)) == %q, want %q", want, got, want)
	}
}

// issue 7560

func test7560(t *testing.T) {
	// some mingw don't implement __packed__ correctly.
	if C.offset7560() != 1 {
		t.Skip("C compiler did not pack struct")
	}

	// C.misaligned should have x but then a padding field to get to the end of the struct.
	// There should not be a field named 'y'.
	var v C.misaligned
	rt := reflect.TypeOf(&v).Elem()
	if rt.NumField() != 2 || rt.Field(0).Name != "x" || rt.Field(1).Name != "_" {
		t.Errorf("unexpected fields in C.misaligned:\n")
		for i := 0; i < rt.NumField(); i++ {
			t.Logf("%+v\n", rt.Field(i))
		}
	}
}

// issue 7786

func f() {
	var x1 *C.typedef_test7786
	var x2 *C.struct_test7786
	x1 = x2
	x2 = x1
	C.f7786(x1)
	C.f7786(x2)
	C.g7786(x1)
	C.g7786(x2)

	var b1 *C.typedef_body7786
	var b2 *C.struct_body7786
	b1 = b2
	b2 = b1
	C.b7786(b1)
	C.b7786(b2)
	C.c7786(b1)
	C.c7786(b2)

	var u1 *C.typedef_union7786
	var u2 *C.union_union7786
	u1 = u2
	u2 = u1
	C.u7786(u1)
	C.u7786(u2)
	C.v7786(u1)
	C.v7786(u2)
}

// issue 8092

func test8092(t *testing.T) {
	tests := []struct {
		s    string
		a, b *C.char
	}{
		{"text", &C.text[0], C.ctext()},
		{"data", &C.data[0], C.cdata()},
	}
	for _, test := range tests {
		if test.a != test.b {
			t.Errorf("%s: pointer mismatch: %v != %v", test.s, test.a, test.b)
		}
		if got := C.GoString(test.a); got != test.s {
			t.Errorf("%s: points at %#v, want %#v", test.s, got, test.s)
		}
	}
}

// issues 8368 and 8441

func issue8368(one *C.struct_one, two *C.struct_two) {
}

func issue8441(one *C.one, two *C.two) {
	issue8441(two.x, one.x)
}

// issue 8428

var _ = C.struct_issue8428one{
	b: C.char(0),
	// The trailing rest field is not available in cgo.
	// See issue 11925.
	// rest: [0]C.char{},
}

var _ = C.struct_issue8428two{
	p:    unsafe.Pointer(nil),
	b:    C.char(0),
	rest: [0]C.char{},
}

var _ = C.struct_issue8428three{
	w: [1][2][3][0]C.char{},
	x: [2][3][0][1]C.char{},
	y: [3][0][1][2]C.char{},
	z: [0][1][2][3]C.char{},
}

// issue 8811

func test8811(t *testing.T) {
	C.issue8811Execute()
}

// issue 9557

func test9557(t *testing.T) {
	// implicitly dereference a Go variable
	foo := C.issue9557foo
	if v := foo.a; v != 42 {
		t.Fatalf("foo.a expected 42, but got %d", v)
	}

	// explicitly dereference a C variable
	if v := (*C.issue9557foo).a; v != 42 {
		t.Fatalf("(*C.issue9557foo).a expected 42, but is %d", v)
	}

	// implicitly dereference a C variable
	if v := C.issue9557foo.a; v != 42 {
		t.Fatalf("C.issue9557foo.a expected 42, but is %d", v)
	}
}

// issue 8331 part 1

func issue8331a() C.issue8331 {
	return issue8331Var
}

// issue 10303

func test10303(t *testing.T, n int) {
	if runtime.Compiler == "gccgo" {
		t.Skip("gccgo permits C pointers on the stack")
	}

	// Run at a few different stack depths just to avoid an unlucky pass
	// due to variables ending up on different pages.
	if n > 0 {
		test10303(t, n-1)
	}
	if t.Failed() {
		return
	}
	var x, y, z, v, si C.int
	var s C.Struct
	C.setintstar(&x)
	C.setintptr(&y)
	C.setvoidptr(unsafe.Pointer(&v))
	s.P = &si
	C.setstruct(s)

	if uintptr(unsafe.Pointer(&x))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C int* argument on stack")
	}
	if uintptr(unsafe.Pointer(&y))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C intptr argument on stack")
	}
	if uintptr(unsafe.Pointer(&v))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C void* argument on stack")
	}
	if uintptr(unsafe.Pointer(&si))&^0xfff == uintptr(unsafe.Pointer(&z))&^0xfff {
		t.Error("C struct field pointer on stack")
	}
}

// issue 11925

func test11925(t *testing.T) {
	if C.sizeof_struct_a11925 != unsafe.Sizeof(C.struct_a11925{}) {
		t.Errorf("size of a changed: C %d, Go %d", C.sizeof_struct_a11925, unsafe.Sizeof(C.struct_a11925{}))
	}
	if C.sizeof_struct_b11925 != unsafe.Sizeof(C.struct_b11925{}) {
		t.Errorf("size of b changed: C %d, Go %d", C.sizeof_struct_b11925, unsafe.Sizeof(C.struct_b11925{}))
	}
}

// issue 12030

func test12030(t *testing.T) {
	buf := (*C.char)(C.malloc(256))
	defer C.free(unsafe.Pointer(buf))
	for _, f := range []float64{1.0, 2.0, 3.14} {
		C.issue12030conv(buf, C.double(f))
		got := C.GoString(buf)
		if want := fmt.Sprintf("d=%g", f); got != want {
			t.Fatalf("C.sprintf failed for %g: %q != %q", f, got, want)
		}
	}
}

// issue 13402

var _ C.complexfloat
var _ C.complexdouble

// issue 13930
// Test that cgo's multiple-value special form for
// C function calls works in variable declaration statements.

var _, _ = C.abs(0)

// issue 14838

func test14838(t *testing.T) {
	data := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	cData := C.CBytes(data)
	defer C.free(cData)

	if C.check_cbytes((*C.char)(cData), C.size_t(len(data))) == 0 {
		t.Fatalf("mismatched data: expected %v, got %v", data, (*(*[10]byte)(unsafe.Pointer(cData)))[:])
	}
}

// issue 17065

var sink C.int

func test17065(t *testing.T) {
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		t.Skip("broken on darwin; issue 17065")
	}
	for i := range C.ii {
		sink = C.ii[i]
	}
}

// issue 17537

func test17537(t *testing.T) {
	v := C.S17537{i: 17537}
	if got, want := C.I17537(&v), C.int(17537); got != want {
		t.Errorf("got %d, want %d", got, want)
	}

	p := (*C.char)(C.malloc(1))
	*p = 17
	if got, want := C.F17537(&p), C.int(17); got != want {
		t.Errorf("got %d, want %d", got, want)
	}

	C.F18298(nil)
	var v18298 C.T18298_2
	C.G18298(C.T18298_1(v18298))
}

// issue 17723

func testAPI() {
	var cs *C.char
	cs = C.CString("hello")
	defer C.free(unsafe.Pointer(cs))
	var s string
	s = C.GoString((*C.char)(C.api_hello))
	s = C.GoStringN((*C.char)(C.api_hello), C.int(6))
	var b []byte
	b = C.GoBytes(unsafe.Pointer(C.api_hello), C.int(6))
	_, _ = s, b
	C.cstring_pointer_fun(nil)
}

// issue 18126

func test18126(t *testing.T) {
	p := C.malloc(1)
	_, err := C.Issue18126C(&p)
	C.free(p)
	_ = err
}

// issue 18720

func test18720(t *testing.T) {
	if got, want := C.HELLO_WORLD, "hello\000world"; got != want {
		t.Errorf("C.HELLO_WORLD == %q, expected %q", got, want)
	}

	if got, want := C.VAR1, C.int(5); got != want {
		t.Errorf("C.VAR1 == %v, expected %v", got, want)
	}

	if got, want := *C.ADDR, C.int(5); got != want {
		t.Errorf("*C.ADDR == %v, expected %v", got, want)
	}

	if got, want := C.CALL, C.int(6); got != want {
		t.Errorf("C.CALL == %v, expected %v", got, want)
	}

	if got, want := C.CALL, C.int(7); got != want {
		t.Errorf("C.CALL == %v, expected %v", got, want)
	}

	// Issue 20125.
	if got, want := C.SIZE_OF_FOO, 1; got != want {
		t.Errorf("C.SIZE_OF_FOO == %v, expected %v", got, want)
	}
}

// issue 20129

func test20129(t *testing.T) {
	if C.issue20129 != 0 {
		t.Fatal("test is broken")
	}
	C.issue20129Foo()
	if C.issue20129 != 1 {
		t.Errorf("got %v but expected %v", C.issue20129, 1)
	}
	C.issue20129Bar()
	if C.issue20129 != 2 {
		t.Errorf("got %v but expected %v", C.issue20129, 2)
	}
}

// issue 20369

func test20369(t *testing.T) {
	if C.XUINT64_MAX != math.MaxUint64 {
		t.Fatalf("got %v, want %v", uint64(C.XUINT64_MAX), uint64(math.MaxUint64))
	}
}

// issue 21668

var issue21668_X = C.x21668

// issue 21708

func test21708(t *testing.T) {
	if got, want := C.CAST_TO_INT64, -1; got != want {
		t.Errorf("C.CAST_TO_INT64 == %v, expected %v", got, want)
	}
}

// issue 21809

func test21809(t *testing.T) {
	longVar := C.long(3)
	typedefVar := C.MySigned_t(4)
	typedefTypedefVar := C.MySigned2_t(5)

	// all three should be considered identical to `long`
	if ret := C.takes_long(longVar); ret != 9 {
		t.Errorf("got %v but expected %v", ret, 9)
	}
	if ret := C.takes_long(typedefVar); ret != 16 {
		t.Errorf("got %v but expected %v", ret, 16)
	}
	if ret := C.takes_long(typedefTypedefVar); ret != 25 {
		t.Errorf("got %v but expected %v", ret, 25)
	}

	// They should also be identical to the typedef'd type
	if ret := C.takes_typedef(longVar); ret != 9 {
		t.Errorf("got %v but expected %v", ret, 9)
	}
	if ret := C.takes_typedef(typedefVar); ret != 16 {
		t.Errorf("got %v but expected %v", ret, 16)
	}
	if ret := C.takes_typedef(typedefTypedefVar); ret != 25 {
		t.Errorf("got %v but expected %v", ret, 25)
	}
}

// issue 22906

func test22906(t *testing.T) {
	var x1 C.jobject = 0 // Note: 0, not nil. That makes sure we use uintptr for these types.
	_ = x1
	var x2 C.jclass = 0
	_ = x2
	var x3 C.jthrowable = 0
	_ = x3
	var x4 C.jstring = 0
	_ = x4
	var x5 C.jarray = 0
	_ = x5
	var x6 C.jbooleanArray = 0
	_ = x6
	var x7 C.jbyteArray = 0
	_ = x7
	var x8 C.jcharArray = 0
	_ = x8
	var x9 C.jshortArray = 0
	_ = x9
	var x10 C.jintArray = 0
	_ = x10
	var x11 C.jlongArray = 0
	_ = x11
	var x12 C.jfloatArray = 0
	_ = x12
	var x13 C.jdoubleArray = 0
	_ = x13
	var x14 C.jobjectArray = 0
	_ = x14
	var x15 C.jweak = 0
	_ = x15
}

// issue 22958
// Nothing to run, just make sure this compiles.
var Vissue22958 C.issue22958Type

func test23356(t *testing.T) {
	if got, want := C.a(), C.int(5); got != want {
		t.Errorf("C.a() == %v, expected %v", got, want)
	}
	if got, want := C.r(), C.int(3); got != want {
		t.Errorf("C.r() == %v, expected %v", got, want)
	}
}

// issue 23720

func Issue23720F() {
	var x C.issue23720A
	C.issue23720F(x)
}

// issue 24206

func test24206(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("skipping on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	if l := len(C.GoString(C.dangerousString1())); l != 123 {
		t.Errorf("Incorrect string length - got %d, want 123", l)
	}
	if l := len(C.GoString(C.dangerousString2())); l != 4096+123 {
		t.Errorf("Incorrect string length - got %d, want %d", l, 4096+123)
	}
}

// issue 25143

func issue25143sum(ns ...C.int) C.int {
	total := C.int(0)
	for _, n := range ns {
		total += n
	}
	return total
}

func test25143(t *testing.T) {
	if got, want := issue25143sum(1, 2, 3), C.int(6); got != want {
		t.Errorf("issue25143sum(1, 2, 3) == %v, expected %v", got, want)
	}
}

// issue 26066
// Wrong type of constant with GCC 8 and newer.

func test26066(t *testing.T) {
	var i = int64(C.issue26066)
	if i != -1 {
		t.Errorf("got %d, want -1", i)
	}
}

// issue 26517
var a C.TypeOne
var b C.TypeTwo

// issue 27660
// Stress the interaction between the race detector and cgo in an
// attempt to reproduce the memory corruption described in #27660.
// The bug was very timing sensitive; at the time of writing this
// test would only trigger the bug about once out of every five runs.

func test27660(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ints := make([]int, 100)
	locks := make([]sync.Mutex, 100)
	// Slowly create threads so that ThreadSanitizer is forced to
	// frequently resize its SyncClocks.
	for i := 0; i < 100; i++ {
		go func() {
			for ctx.Err() == nil {
				// Sleep in C for long enough that it is likely that the runtime
				// will retake this goroutine's currently wired P.
				C.usleep(1000 /* 1ms */)
				runtime.Gosched() // avoid starvation (see #28701)
			}
		}()
		go func() {
			// Trigger lots of synchronization and memory reads/writes to
			// increase the likelihood that the race described in #27660
			// results in corruption of ThreadSanitizer's internal state
			// and thus an assertion failure or segfault.
			i := 0
			for ctx.Err() == nil {
				j := rand.Intn(100)
				locks[j].Lock()
				ints[j]++
				locks[j].Unlock()
				// needed for gccgo, to avoid creation of an
				// unpreemptible "fast path" in this loop. Choice
				// of (1<<24) is somewhat arbitrary.
				if i%(1<<24) == 0 {
					runtime.Gosched()
				}
				i++

			}
		}()
		time.Sleep(time.Millisecond)
	}
}

// issue 28540

func twoargsF() {
	v := []string{}
	C.twoargs1(C.twoargs2(), C.twoargs3(unsafe.Pointer(&v)))
}

// issue 28545

func issue28545G(p **C.char) {
	C.issue28545F(p, -1, (0))
	C.issue28545F(p, 2+3, complex(1, 1))
	C.issue28545F(p, issue28772Constant, issue28772Constant2)
}

// issue 28772 part 1 - part 2 in testx.go

const issue28772Constant = C.issue28772Constant

// issue 28896

func offset(i int) uintptr {
	var pi C.innerPacked
	var po C.outerPacked
	var ui C.innerUnpacked
	var uo C.outerUnpacked
	switch i {
	case 0:
		return unsafe.Offsetof(pi.f2)
	case 1:
		return unsafe.Offsetof(po.g2)
	case 2:
		return unsafe.Offsetof(ui.f2)
	case 3:
		return unsafe.Offsetof(uo.g2)
	default:
		panic("can't happen")
	}
}

func test28896(t *testing.T) {
	for i := 0; i < 4; i++ {
		c := uintptr(C.offset(C.int(i)))
		g := offset(i)
		if c != g {
			t.Errorf("%d: C: %d != Go %d", i, c, g)
		}
	}
}

// issue 29383
// cgo's /*line*/ comments failed when inserted after '/',
// because the result looked like a "//" comment.
// No runtime test; just make sure it compiles.

func Issue29383(n, size uint) int {
	if ^C.size_t(0)/C.size_t(n) < C.size_t(size) {
		return 0
	}
	return 0
}

// issue 29748
// Error handling a struct initializer that requires pointer checking.
// Compilation test only, nothing to run.

var Vissue29748 = C.f29748(&C.S29748{
	nil,
})

func Fissue299748() {
	C.f29748(&C.S29748{
		nil,
	})
}

// issue 29781

var issue29781X struct{ X int }

func issue29781F(...int) int { return 0 }

func issue29781G() {
	var p *C.char
	C.issue29781F(&p, C.ISSUE29781C+1)
	C.issue29781F(nil, (C.int)(
		0))
	C.issue29781F(&p, (C.int)(0))
	C.issue29781F(&p, (C.int)(
		0))
	C.issue29781F(&p, (C.int)(issue29781X.
		X))
}

// issue 30065

func test30065(t *testing.T) {
	var a [256]byte
	b := []byte("a")
	C.memcpy(unsafe.Pointer(&a), unsafe.Pointer(&b[0]), 1)
	if a[0] != 'a' {
		t.Errorf("&a failed: got %c, want %c", a[0], 'a')
	}

	b = []byte("b")
	C.memcpy(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), 1)
	if a[0] != 'b' {
		t.Errorf("&a[0] failed: got %c, want %c", a[0], 'b')
	}

	d := make([]byte, 256)
	b = []byte("c")
	C.memcpy(unsafe.Pointer(&d[0]), unsafe.Pointer(&b[0]), 1)
	if d[0] != 'c' {
		t.Errorf("&d[0] failed: got %c, want %c", d[0], 'c')
	}
}

// issue 31093
// No runtime test; just make sure it compiles.

func Issue31093() {
	C.issue31093F(C.ushort(0))
}

// issue 32579

func test32579(t *testing.T) {
	var s [1]C.struct_S32579
	C.memset(unsafe.Pointer(&s[0].data[0]), 1, 1)
	if s[0].data[0] != 1 {
		t.Errorf("&s[0].data[0] failed: got %d, want %d", s[0].data[0], 1)
	}
}

// issue 37033, check if cgo.Handle works properly

func testHandle(t *testing.T) {
	ch := make(chan int)

	for i := 0; i < 42; i++ {
		h := cgo.NewHandle(ch)
		go func() {
			C.cFunc37033(C.uintptr_t(h))
		}()
		if v := <-ch; issue37033 != v {
			t.Fatalf("unexpected receiving value: got %d, want %d", v, issue37033)
		}
		h.Delete()
	}
}

// issue 38649

var issue38649 C.netbsd_gid = 42

// issue 39877

var issue39877 *C.void = nil

// issue 40494
// No runtime test; just make sure it compiles.

func Issue40494() {
	C.issue40494(C.enum_Enum40494(C.X_40494), (*C.union_Union40494)(nil))
}

// Issue 45451.
func test45451(t *testing.T) {
	var u *C.issue45451
	typ := reflect.ValueOf(u).Type().Elem()

	// The type is undefined in C so allocating it should panic.
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()

	_ = reflect.New(typ)
	t.Errorf("reflect.New(%v) should have panicked", typ)
}
