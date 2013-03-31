/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 1998-2002 by Lucent Technologies.
 *              Portions Copyright (c) 2009 The Go Authors.  All rights reserved.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

#ifndef _UTFH_
#define _UTFH_ 1

typedef unsigned int Rune;	/* Code-point values in Unicode 4.0 are 21 bits wide.*/

enum
{
  UTFmax	= 4,		/* maximum bytes per rune */
  Runesync	= 0x80,		/* cannot represent part of a UTF sequence (<) */
  Runeself	= 0x80,		/* rune and UTF sequences are the same (<) */
  Runeerror	= 0xFFFD,	/* decoding error in UTF */
  Runemax	= 0x10FFFF,	/* maximum rune value */
};

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * rune routines
 */

/*
 * These routines were written by Rob Pike and Ken Thompson
 * and first appeared in Plan 9.
 * SEE ALSO
 * utf (7)
 * tcs (1)
*/

// runetochar copies (encodes) one rune, pointed to by r, to at most
// UTFmax bytes starting at s and returns the number of bytes generated.

int runetochar(char* s, const Rune* r);


// chartorune copies (decodes) at most UTFmax bytes starting at s to
// one rune, pointed to by r, and returns the number of bytes consumed.
// If the input is not exactly in UTF format, chartorune will set *r
// to Runeerror and return 1.
//
// Note: There is no special case for a "null-terminated" string. A
// string whose first byte has the value 0 is the UTF8 encoding of the
// Unicode value 0 (i.e., ASCII NULL). A byte value of 0 is illegal
// anywhere else in a UTF sequence.

int chartorune(Rune* r, const char* s);


// charntorune is like chartorune, except that it will access at most
// n bytes of s.  If the UTF sequence is incomplete within n bytes,
// charntorune will set *r to Runeerror and return 0. If it is complete
// but not in UTF format, it will set *r to Runeerror and return 1.
//
// Added 2004-09-24 by Wei-Hwa Huang

int charntorune(Rune* r, const char* s, int n);

// isvalidcharntorune(str, n, r, consumed)
// is a convenience function that calls "*consumed = charntorune(r, str, n)"
// and returns an int (logically boolean) indicating whether the first
// n bytes of str was a valid and complete UTF sequence.

int isvalidcharntorune(const char* str, int n, Rune* r, int* consumed);

// runelen returns the number of bytes required to convert r into UTF.

int runelen(Rune r);


// runenlen returns the number of bytes required to convert the n
// runes pointed to by r into UTF.

int runenlen(const Rune* r, int n);


// fullrune returns 1 if the string s of length n is long enough to be
// decoded by chartorune, and 0 otherwise. This does not guarantee
// that the string contains a legal UTF encoding. This routine is used
// by programs that obtain input one byte at a time and need to know
// when a full rune has arrived.

int fullrune(const char* s, int n);

// The following routines are analogous to the corresponding string
// routines with "utf" substituted for "str", and "rune" substituted
// for "chr".

// utflen returns the number of runes that are represented by the UTF
// string s. (cf. strlen)

int utflen(const char* s);


// utfnlen returns the number of complete runes that are represented
// by the first n bytes of the UTF string s. If the last few bytes of
// the string contain an incompletely coded rune, utfnlen will not
// count them; in this way, it differs from utflen, which includes
// every byte of the string. (cf. strnlen)

int utfnlen(const char* s, long n);


// utfrune returns a pointer to the first occurrence of rune r in the
// UTF string s, or 0 if r does not occur in the string.  The NULL
// byte terminating a string is considered to be part of the string s.
// (cf. strchr)

/*const*/ char* utfrune(const char* s, Rune r);


// utfrrune returns a pointer to the last occurrence of rune r in the
// UTF string s, or 0 if r does not occur in the string.  The NULL
// byte terminating a string is considered to be part of the string s.
// (cf. strrchr)

/*const*/ char* utfrrune(const char* s, Rune r);


// utfutf returns a pointer to the first occurrence of the UTF string
// s2 as a UTF substring of s1, or 0 if there is none. If s2 is the
// null string, utfutf returns s1. (cf. strstr)

const char* utfutf(const char* s1, const char* s2);


// utfecpy copies UTF sequences until a null sequence has been copied,
// but writes no sequences beyond es1.  If any sequences are copied,
// s1 is terminated by a null sequence, and a pointer to that sequence
// is returned.  Otherwise, the original s1 is returned. (cf. strecpy)

char* utfecpy(char *s1, char *es1, const char *s2);



// These functions are rune-string analogues of the corresponding
// functions in strcat (3).
//
// These routines first appeared in Plan 9.
// SEE ALSO
// memmove (3)
// rune (3)
// strcat (2)
//
// BUGS: The outcome of overlapping moves varies among implementations.

Rune* runestrcat(Rune* s1, const Rune* s2);
Rune* runestrncat(Rune* s1, const Rune* s2, long n);

const Rune* runestrchr(const Rune* s, Rune c);

int runestrcmp(const Rune* s1, const Rune* s2);
int runestrncmp(const Rune* s1, const Rune* s2, long n);

Rune* runestrcpy(Rune* s1, const Rune* s2);
Rune* runestrncpy(Rune* s1, const Rune* s2, long n);
Rune* runestrecpy(Rune* s1, Rune* es1, const Rune* s2);

Rune* runestrdup(const Rune* s);

const Rune* runestrrchr(const Rune* s, Rune c);
long runestrlen(const Rune* s);
const Rune* runestrstr(const Rune* s1, const Rune* s2);



// The following routines test types and modify cases for Unicode
// characters.  Unicode defines some characters as letters and
// specifies three cases: upper, lower, and title.  Mappings among the
// cases are also defined, although they are not exhaustive: some
// upper case letters have no lower case mapping, and so on.  Unicode
// also defines several character properties, a subset of which are
// checked by these routines.  These routines are based on Unicode
// version 3.0.0.
//
// NOTE: The routines are implemented in C, so the boolean functions
// (e.g., isupperrune) return 0 for false and 1 for true.
//
//
// toupperrune, tolowerrune, and totitlerune are the Unicode case
// mappings. These routines return the character unchanged if it has
// no defined mapping.

Rune toupperrune(Rune r);
Rune tolowerrune(Rune r);
Rune totitlerune(Rune r);


// isupperrune tests for upper case characters, including Unicode
// upper case letters and targets of the toupper mapping. islowerrune
// and istitlerune are defined analogously.

int isupperrune(Rune r);
int islowerrune(Rune r);
int istitlerune(Rune r);


// isalpharune tests for Unicode letters; this includes ideographs in
// addition to alphabetic characters.

int isalpharune(Rune r);


// isdigitrune tests for digits. Non-digit numbers, such as Roman
// numerals, are not included.

int isdigitrune(Rune r);


// isspacerune tests for whitespace characters, including "C" locale
// whitespace, Unicode defined whitespace, and the "zero-width
// non-break space" character.

int isspacerune(Rune r);


// (The comments in this file were copied from the manpage files rune.3,
// isalpharune.3, and runestrcat.3. Some formatting changes were also made
// to conform to Google style. /JRM 11/11/05)

#ifdef	__cplusplus
}
#endif

#endif
