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

/* The Computer Language Benchmarks Game
  http://shootout.alioth.debian.org/

  contributed by Paolo Bonzini & Sean Bartlett
  modified by Michael Mellor
*/

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

static mpz_t numer, accum, denom, tmp1, tmp2;

static int extract_digit()
{
  if (mpz_cmp(numer, accum) > 0)
    return -1;

  /* Compute (numer * 3 + accum) / denom */
  mpz_mul_2exp(tmp1, numer, 1);
  mpz_add(tmp1, tmp1, numer);
  mpz_add(tmp1, tmp1, accum);
  mpz_fdiv_qr(tmp1, tmp2, tmp1, denom);

  /* Now, if (numer * 4 + accum) % denom... */
  mpz_add(tmp2, tmp2, numer);

  /* ... is normalized, then the two divisions have the same result.  */
  if (mpz_cmp(tmp2, denom) >= 0)
    return -1;

  return mpz_get_ui(tmp1);
}

static void next_term(unsigned int k)
{
  unsigned int y2 = k*2 + 1;

  mpz_mul_2exp(tmp1, numer, 1);
  mpz_add(accum, accum, tmp1);
  mpz_mul_ui(accum, accum, y2);
  mpz_mul_ui(numer, numer, k);
  mpz_mul_ui(denom, denom, y2);
}

static void eliminate_digit(unsigned int d)
{
  mpz_submul_ui(accum, denom, d);
  mpz_mul_ui(accum, accum, 10);
  mpz_mul_ui(numer, numer, 10);
}

static void pidigits(unsigned int n)
{
  int d;
  unsigned int i = 0, k = 0, m;
  mpz_init(tmp1);
  mpz_init(tmp2);
  mpz_init_set_ui(numer, 1);
  mpz_init_set_ui(accum, 0);
  mpz_init_set_ui(denom, 1);

  for(;;)
  {
    do {
      k++;
      next_term(k);
      d = extract_digit();
    } while(d == -1);

    putchar(d + '0');

    i++;
    m = i%10;
    if(m == 0)
      printf("\t:%d\n", i);
    if(i >= n)
      break;
    eliminate_digit(d);
  }

  if(m) {
    m = 10 - m;
    while(m--)
      putchar(' ');
    printf("\t:%d\n", n);
  }
}

int main(int argc, char **argv)
{
  pidigits(argc > 1 ? atoi(argv[1]) : 27);
  return 0;
}
