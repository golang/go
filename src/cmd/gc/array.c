// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

enum {
	DEFAULTCAPACITY = 16,
};

struct Array
{
	int32	length;  // number of elements
	int32	size;  // element size
	int32	capacity;  // size of data in elements
	char	*data;  // element storage
};

Array*
arraynew(int32 capacity, int32 size)
{
	Array *result;

	if(capacity < 0)
		fatal("arraynew: capacity %d is not positive", capacity);
	if(size < 0)
		fatal("arraynew: size %d is not positive\n", size);
	result = malloc(sizeof(*result));
	if(result == nil)
		fatal("arraynew: malloc failed\n");
	result->length = 0;
	result->size = size;
	result->capacity = capacity == 0 ? DEFAULTCAPACITY : capacity;
	result->data = malloc(result->capacity * result->size);
	if(result->data == nil)
		fatal("arraynew: malloc failed\n");
	return result;
}

void
arrayfree(Array *array)
{
	if(array == nil)
		return;
	free(array->data);
	free(array);
}

int32
arraylength(Array *array)
{
	return array->length;
}

void*
arrayget(Array *array, int32 index)
{
	if(array == nil)
		fatal("arrayget: array is nil\n");
	if(index < 0 || index >= array->length)
		fatal("arrayget: index %d is out of bounds for length %d\n", index, array->length);
	return array->data + index * array->size;
}

void
arrayset(Array *array, int32 index, void *element)
{
	if(array == nil)
		fatal("arrayset: array is nil\n");
	if(element == nil)
		fatal("arrayset: element is nil\n");
	if(index < 0 || index >= array->length)
		fatal("arrayget: index %d is out of bounds for length %d\n", index, array->length);
	memmove(array->data + index * array->size, element, array->size);
}

static void
ensurecapacity(Array *array, int32 capacity)
{
	int32 newcapacity;
	char *newdata;

	if(array == nil)
		fatal("ensurecapacity: array is nil\n");
	if(capacity < 0)
		fatal("ensurecapacity: capacity %d is not positive", capacity);
	if(capacity >= array->capacity) {
		newcapacity = capacity + (capacity >> 1);
		newdata = realloc(array->data, newcapacity * array->size);
		if(newdata == nil)
			fatal("ensurecapacity: realloc failed\n");
		array->capacity = newcapacity;
		array->data = newdata;
	}
}

void
arrayadd(Array *array, void *element)
{
	if(array == nil)
		fatal("arrayset: array is nil\n");
	if(element == nil)
		fatal("arrayset: element is nil\n");
	ensurecapacity(array, array->length + 1);
	array->length++;
	arrayset(array, array->length - 1, element);
}

void
arraysort(Array *array, int (*cmp)(const void*, const void*))
{
	qsort(array->data, array->length, array->size, cmp);
}
