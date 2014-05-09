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
* The Computer Language Benchmarks Game
* http://shootout.alioth.debian.org/

* contributed by Premysl Hruby
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <limits.h>

// PTHREAD_STACK_MIN undeclared on mingw
#ifndef PTHREAD_STACK_MIN
#define PTHREAD_STACK_MIN 65535
#endif

#define THREADS (503)

struct stack {
   char x[PTHREAD_STACK_MIN];
};


/* staticaly initialize mutex[0] mutex */
static pthread_mutex_t mutex[THREADS];
static int data[THREADS];
static struct stack stacks[THREADS];
/* stacks must be defined staticaly, or my i386 box run of virtual memory for this
 * process while creating thread +- #400 */

static void* thread(void *num)
{
   int l = (int)(uintptr_t)num;
   int r = (l+1) % THREADS;
   int token;

   while(1) {
      pthread_mutex_lock(mutex + l);
      token = data[l];
      if (token) {
         data[r] = token - 1;
         pthread_mutex_unlock(mutex + r);
      }
      else {
         printf("%i\n", l+1);
         exit(0);
      }
   }
}



int main(int argc, char **argv)
{
   int i;
   pthread_t cthread;
   pthread_attr_t stack_attr;

   if (argc != 2)
      exit(255);
   data[0] = atoi(argv[1]);

   pthread_attr_init(&stack_attr);

   for (i = 0; i < THREADS; i++) {
      pthread_mutex_init(mutex + i, NULL);
      pthread_mutex_lock(mutex + i);

#if defined(__MINGW32__) || defined(__MINGW64__)
      pthread_attr_setstackaddr(&stack_attr, &stacks[i]);
      pthread_attr_setstacksize(&stack_attr, sizeof(struct stack));
#else
      pthread_attr_setstack(&stack_attr, &stacks[i], sizeof(struct stack));
#endif

      pthread_create(&cthread, &stack_attr, thread, (void*)(uintptr_t)i);
   }

   pthread_mutex_unlock(mutex + 0);
   pthread_join(cthread, NULL);
}
