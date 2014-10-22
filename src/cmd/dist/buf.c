// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Byte buffers and string vectors.

#include "a.h"

// binit prepares an uninitialized buffer for use.
void
binit(Buf *b)
{
	b->p = nil;
	b->len = 0;
	b->cap = 0;
}

// breset truncates the buffer back to zero length.
void
breset(Buf *b)
{
	b->len = 0;
}

// bfree frees the storage associated with a buffer.
void
bfree(Buf *b)
{
	xfree(b->p);
	binit(b);
}

// bgrow ensures that the buffer has at least n more bytes
// between its len and cap.
void
bgrow(Buf *b, int n)
{
	int want;
	
	want = b->len+n;
	if(want > b->cap) {
		b->cap = 2*want;
		if(b->cap < 64)
			b->cap = 64;
		b->p = xrealloc(b->p, b->cap);
	}
}

// bwrite appends the n bytes at v to the buffer.
void
bwrite(Buf *b, void *v, int n)
{
	bgrow(b, n);
	xmemmove(b->p+b->len, v, n);
	b->len += n;
}

// bwritestr appends the string p to the buffer.
void
bwritestr(Buf *b, char *p)
{
	bwrite(b, p, xstrlen(p));
}

// bstr returns a pointer to a NUL-terminated string of the
// buffer contents.  The pointer points into the buffer.
char*
bstr(Buf *b)
{
	bgrow(b, 1);
	b->p[b->len] = '\0';
	return b->p;
}

// btake takes ownership of the string form of the buffer.
// After this call, the buffer has zero length and does not
// refer to the memory that btake returned.
char*
btake(Buf *b)
{
	char *p;
	
	p = bstr(b);
	binit(b);
	return p;
}

// bwriteb appends the src buffer to the dst buffer.
void
bwriteb(Buf *dst, Buf *src)
{
	bwrite(dst, src->p, src->len);
}

// bequal reports whether the buffers have the same content.
bool
bequal(Buf *s, Buf *t)
{
	return s->len == t->len && xmemcmp(s->p, t->p, s->len) == 0;
}

// bsubst rewites b to replace all occurrences of x with y.
void
bsubst(Buf *b, char *x, char *y)
{
	char *p;
	int nx, ny, pos;

	nx = xstrlen(x);
	ny = xstrlen(y);

	pos = 0;
	for(;;) {
		p = xstrstr(bstr(b)+pos, x);
		if(p == nil)
			break;
		if(nx != ny) {
			if(nx < ny) {
				pos = p - b->p;
				bgrow(b, ny-nx);
				p = b->p + pos;
			}
			xmemmove(p+ny, p+nx, (b->p+b->len)-(p+nx));
		}
		xmemmove(p, y, ny);
		pos = p+ny - b->p;
		b->len += ny - nx;
	}
}

// The invariant with the vectors is that v->p[0:v->len] is allocated
// strings that are owned by the vector.  The data beyond v->len may
// be garbage.

// vinit prepares an uninitialized vector for use.
void
vinit(Vec *v)
{
	v->p = nil;
	v->len = 0;
	v->cap = 0;
}

// vreset truncates the vector back to zero length.
void
vreset(Vec *v)
{
	int i;
	
	for(i=0; i<v->len; i++) {
		xfree(v->p[i]);
		v->p[i] = nil;
	}
	v->len = 0;
}

// vfree frees the storage associated with the vector.
void
vfree(Vec *v)
{
	vreset(v);
	xfree(v->p);
	vinit(v);
}


// vgrow ensures that the vector has room for at least 
// n more entries between len and cap.
void
vgrow(Vec *v, int n)
{
	int want;
	
	want = v->len+n;
	if(want > v->cap) {
		v->cap = 2*want;
		if(v->cap < 64)
			v->cap = 64;
		v->p = xrealloc(v->p, v->cap*sizeof v->p[0]);
	}
}

// vcopy copies the srclen strings at src into the vector.
void
vcopy(Vec *dst, char **src, int srclen)
{
	int i;
	
	// use vadd, to make copies of strings
	for(i=0; i<srclen; i++)
		vadd(dst, src[i]);
}

// vadd adds a copy of the string p to the vector.
void
vadd(Vec *v, char *p)
{
	vgrow(v, 1);
	if(p != nil)
		p = xstrdup(p);
	v->p[v->len++] = p;
}

// vaddn adds a string consisting of the n bytes at p to the vector.
static void
vaddn(Vec *v, char *p, int n)
{
	char *q;

	vgrow(v, 1);
	q = xmalloc(n+1);
	xmemmove(q, p, n);
	q[n] = '\0';
	v->p[v->len++] = q;
}

static int
strpcmp(const void *a, const void *b)
{
	return xstrcmp(*(char**)a, *(char**)b);
}

// vuniq sorts the vector and then discards duplicates,
// in the manner of sort | uniq.
void
vuniq(Vec *v)
{
	int i, n;

	xqsort(v->p, v->len, sizeof(v->p[0]), strpcmp);
	n = 0;
	for(i=0; i<v->len; i++) {
		if(n>0 && streq(v->p[i], v->p[n-1]))
			xfree(v->p[i]);
		else
			v->p[n++] = v->p[i];
	}
	v->len = n;
}

// splitlines replaces the vector v with the result of splitting
// the input p after each \n.
void
splitlines(Vec *v, char *p)
{
	int i;
	char *start;
	
	vreset(v);
	start = p;
	for(i=0; p[i]; i++) {
		if(p[i] == '\n') {
			vaddn(v, start, (p+i+1)-start);
			start = p+i+1;
		}
	}
	if(*start != '\0')
		vadd(v, start);
}

// splitfields replaces the vector v with the result of splitting
// the input p into non-empty fields containing no spaces.
void
splitfields(Vec *v, char *p)
{
	char *start;

	vreset(v);
	for(;;) {
		while(*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
			p++;
		if(*p == '\0')
			break;
		start = p;
		while(*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n' && *p != '\0')
			p++;
		vaddn(v, start, p-start);
	}
}
