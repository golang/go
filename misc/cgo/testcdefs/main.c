// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cdefstest.h"

void runtime·printf(int8*, ...);

// From cdefstest.go.
typedef struct CdefsOrig CdefsOrig;
struct CdefsOrig {
	int8 array1[20];
	int8 array2[20][20];
	int8 *array3[20];
	int8 *array4[20][20];
	int8 **array5[20][20];
};

// Packed structs are no longer supported for -cdefs.
/*
typedef struct PackedOrig PackedOrig;
#pragma pack on
struct PackedOrig {
	int8 first;
	int32 second;
	int64 third;
};
#pragma pack off
*/

void
main·test(int32 ret)
{
	CdefsOrig o;
	CdefsTest t;
	// PackedOrig po;
	// PackedTest pt;
	
	ret = 0;
	if(sizeof(t.array1) != sizeof(o.array1) || offsetof(CdefsTest, array1[0]) != offsetof(CdefsOrig, array1[0])) {
		runtime·printf("array1: size, offset = %d, %d, want %d, %d\n", sizeof(t.array1), offsetof(CdefsTest, array1[0]), sizeof(o.array1), offsetof(CdefsOrig, array1[0]));
		ret = 1;
	}
	if(sizeof(t.array2) != sizeof(o.array2) || offsetof(CdefsTest, array2[0][0]) != offsetof(CdefsOrig, array2[0][0])) {
		runtime·printf("array2: size, offset = %d, %d, want %d, %d\n", sizeof(t.array2), offsetof(CdefsTest, array2[0][0]), sizeof(o.array2), offsetof(CdefsOrig, array2[0][0]));
		ret = 1;
	}
	if(sizeof(t.array3) != sizeof(o.array3) || offsetof(CdefsTest, array3[0]) != offsetof(CdefsOrig, array3[0])) {
		runtime·printf("array3: size, offset = %d, %d, want %d, %d\n", sizeof(t.array3), offsetof(CdefsTest, array3[0]), sizeof(o.array3), offsetof(CdefsOrig, array3[0]));
		ret = 1;
	}
	if(sizeof(t.array4) != sizeof(o.array4) || offsetof(CdefsTest, array4[0][0]) != offsetof(CdefsOrig, array4[0][0])) {
		runtime·printf("array4: size, offset = %d, %d, want %d, %d\n", sizeof(t.array4), offsetof(CdefsTest, array4[0][0]), sizeof(o.array4), offsetof(CdefsOrig, array4[0][0]));
		ret = 1;
	}
	if(sizeof(t.array5) != sizeof(o.array5) || offsetof(CdefsTest, array5[0][0]) != offsetof(CdefsOrig, array5[0][0])) {
		runtime·printf("array5: size, offset = %d, %d, want %d, %d\n", sizeof(t.array5), offsetof(CdefsTest, array5[0][0]), sizeof(o.array5), offsetof(CdefsOrig, array5[0][0]));
		ret = 1;
	}
/*
	if(sizeof(pt.first) != sizeof(po.first) || offsetof(PackedTest, first) != offsetof(PackedOrig, first)) {
		runtime·printf("first: size, offset = %d, %d, want %d, %d\n", sizeof(pt.first), offsetof(PackedTest, first), sizeof(po.first), offsetof(PackedOrig, first));
		ret = 1;
	}
	if(sizeof(pt.second) != sizeof(po.second) || offsetof(PackedTest, second) != offsetof(PackedOrig, second)) {
		runtime·printf("second: size, offset = %d, %d, want %d, %d\n", sizeof(pt.second), offsetof(PackedTest, second), sizeof(po.second), offsetof(PackedOrig, second));
		ret = 1;
	}
	if(sizeof(pt.third) != sizeof(po.third) || offsetof(PackedTest, third) != offsetof(PackedOrig, third)) {
		runtime·printf("third: size, offset = %d, %d, want %d, %d\n", sizeof(pt.third), offsetof(PackedTest, third), sizeof(po.third), offsetof(PackedOrig, third));
		ret = 1;
	}
*/
	FLUSH(&ret); // flush return value
}
