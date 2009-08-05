/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of "The Computer Language Benchmarks Game" nor the
    name of "The Computer Language Shootout Benchmarks" nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*
** The Computer Language Shootout
** http://shootout.alioth.debian.org/
** contributed by Mike Pall
**
** regex-dna benchmark using PCRE
**
** compile with:
**   gcc -O3 -fomit-frame-pointer -o regexdna regexdna.c -lpcre
*/

#define __USE_STRING_INLINES
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pcre.h>

typedef struct fbuf {
  char *buf;
  size_t size, len;
} fbuf_t;

static void fb_init(fbuf_t *b)
{
  b->buf = NULL;
  b->len = b->size = 0;
}

static char *fb_need(fbuf_t *b, size_t need)
{
  need += b->len;
  if (need > b->size) {
    if (b->size == 0) b->size = need;
    else while (need > b->size) b->size += b->size;
    if (!(b->buf = realloc(b->buf, b->size))) exit(1);
  }
  return b->buf+b->len;
}

#define FB_MINREAD	(3<<16)

/* Read all of a stdio stream into dst buffer. */
static size_t fb_readall(fbuf_t *dst, FILE *fp)
{
  char *dp;
  int n;
  for (dp = fb_need(dst, FB_MINREAD);
       (n = fread(dp, 1, dst->size-dst->len, fp)) > 0;
       dp = fb_need(dst, FB_MINREAD)) dst->len += n;
  if (ferror(fp)) exit(1);
  return dst->len;
}

/* Substitute pattern p with replacement r, copying from src to dst buffer. */
static size_t fb_subst(fbuf_t *dst, fbuf_t *src, const char *p, const char *r)
{
  pcre *re;
  pcre_extra *re_ex;
  const char *re_e;
  char *dp;
  int re_eo, m[3], pos, rlen, clen;
  if (!(re = pcre_compile(p, PCRE_CASELESS, &re_e, &re_eo, NULL))) exit(1);
  re_ex = pcre_study(re, 0, &re_e);
  for (dst->len = 0, rlen = strlen(r), pos = 0;
       pcre_exec(re, re_ex, src->buf, src->len, pos, 0, m, 3) >= 0;
       pos = m[1]) {
    clen = m[0]-pos;
    dp = fb_need(dst, clen+rlen);
    dst->len += clen+rlen;
    memcpy(dp, src->buf+pos, clen);
    memcpy(dp+clen, r, rlen);
  }
  clen = src->len-pos;
  dp = fb_need(dst, clen);
  dst->len += clen;
  memcpy(dp, src->buf+pos, clen);
  return dst->len;
}

/* Count all matches with pattern p in src buffer. */
static int fb_countmatches(fbuf_t *src, const char *p)
{
  pcre *re;
  pcre_extra *re_ex;
  const char *re_e;
  int re_eo, m[3], pos, count;
  if (!(re = pcre_compile(p, PCRE_CASELESS, &re_e, &re_eo, NULL))) exit(1);
  re_ex = pcre_study(re, 0, &re_e);
  for (count = 0, pos = 0;
       pcre_exec(re, re_ex, src->buf, src->len, pos, 0, m, 3) >= 0;
       pos = m[1]) count++;
  return count;
}

static const char *variants[] = {
  "agggtaaa|tttaccct",         "[cgt]gggtaaa|tttaccc[acg]",
  "a[act]ggtaaa|tttacc[agt]t", "ag[act]gtaaa|tttac[agt]ct",
  "agg[act]taaa|ttta[agt]cct", "aggg[acg]aaa|ttt[cgt]ccct",
  "agggt[cgt]aa|tt[acg]accct", "agggta[cgt]a|t[acg]taccct",
  "agggtaa[cgt]|[acg]ttaccct", NULL
};

static const char *subst[] = {
  "B", "(c|g|t)", "D", "(a|g|t)",   "H", "(a|c|t)", "K", "(g|t)",
  "M", "(a|c)",   "N", "(a|c|g|t)", "R", "(a|g)",   "S", "(c|g)",
  "V", "(a|c|g)", "W", "(a|t)",     "Y", "(c|t)",   NULL
};

int main(int argc, char **argv)
{
  fbuf_t seq[2];
  const char **pp;
  size_t ilen, clen, slen;
  int flip;
  fb_init(&seq[0]);
  fb_init(&seq[1]);
  ilen = fb_readall(&seq[0], stdin);
  clen = fb_subst(&seq[1], &seq[0], ">.*|\n", "");
  for (pp = variants; *pp; pp++)
    printf("%s %d\n", *pp, fb_countmatches(&seq[1], *pp));
  for (slen = 0, flip = 1, pp = subst; *pp; pp += 2, flip = 1-flip)
    slen = fb_subst(&seq[1-flip], &seq[flip], *pp, pp[1]);
  printf("\n%zu\n%zu\n%zu\n", ilen, clen, slen);
  return 0;
}
