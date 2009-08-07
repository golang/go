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

   contributed by Michael Barker
   based on a Java contribution by Luzius Meisser

   convert to C by dualamd
*/

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>


enum Colour
{
   blue      = 0,
   red      = 1,
   yellow   = 2,
   Invalid   = 3
};

const char* ColourName[] = {"blue", "red", "yellow"};
const int STACK_SIZE   = 32*1024;

typedef unsigned int BOOL;
const BOOL TRUE = 1;
const BOOL FALSE = 0;

int CreatureID = 0;


enum Colour doCompliment(enum Colour c1, enum Colour c2)
{
   switch (c1)
   {
   case blue:
      switch (c2)
      {
      case blue:
         return blue;
      case red:
         return yellow;
      case yellow:
         return red;
      default:
         goto errlb;
      }
   case red:
      switch (c2)
      {
      case blue:
         return yellow;
      case red:
         return red;
      case yellow:
         return blue;
      default:
         goto errlb;
      }
   case yellow:
      switch (c2)
      {
      case blue:
         return red;
      case red:
         return blue;
      case yellow:
         return yellow;
      default:
         goto errlb;
      }
   default:
      break;
   }

errlb:
   printf("Invalid colour\n");
   exit( 1 );
}

/* convert integer to number string: 1234 -> "one two three four" */
char* formatNumber(int n, char* outbuf)
{
   int ochar = 0, ichar = 0;
   int i;
   char tmp[64];

   const char* NUMBERS[] =
   {
      "zero", "one", "two", "three", "four", "five",
      "six", "seven", "eight", "nine"
   };

   ichar = sprintf(tmp, "%d", n);

   for (i = 0; i < ichar; i++)
      ochar += sprintf( outbuf + ochar, " %s", NUMBERS[ tmp[i] - '0' ] );

   return outbuf;
}


struct MeetingPlace
{
   pthread_mutex_t   mutex;
   int             meetingsLeft;
   struct Creature*   firstCreature;
};

struct Creature
{
   pthread_t         ht;
   pthread_attr_t      stack_att;

   struct MeetingPlace* place;
   int         count;
   int         sameCount;

   enum Colour   colour;
   int          id;

   BOOL      two_met;
   BOOL      sameid;
};


void MeetingPlace_Init(struct MeetingPlace* m, int meetings )
{
   pthread_mutex_init( &m->mutex, 0 );
   m->meetingsLeft = meetings;
   m->firstCreature = 0;
}


BOOL Meet( struct Creature* cr)
{
   BOOL retval = TRUE;

   struct MeetingPlace* mp = cr->place;
   pthread_mutex_lock( &(mp->mutex) );

   if ( mp->meetingsLeft > 0 )
   {
      if ( mp->firstCreature == 0 )
      {
         cr->two_met = FALSE;
         mp->firstCreature = cr;
      }
      else
      {
         struct Creature* first;
         enum Colour newColour;

         first = mp->firstCreature;
         newColour = doCompliment( cr->colour, first->colour );

         cr->sameid = cr->id == first->id;
         cr->colour = newColour;
         cr->two_met = TRUE;

         first->sameid = cr->sameid;
         first->colour = newColour;
         first->two_met = TRUE;

         mp->firstCreature = 0;
         mp->meetingsLeft--;
      }
   }
   else
      retval = FALSE;

   pthread_mutex_unlock( &(mp->mutex) );
   return retval;
}


void* CreatureThreadRun(void* param)
{
   struct Creature* cr = (struct Creature*)param;

   while (TRUE)
   {
      if ( Meet(cr) )
      {
         while (cr->two_met == FALSE)
            sched_yield();

         if (cr->sameid)
            cr->sameCount++;
         cr->count++;
      }
      else
         break;
   }

   return 0;
}

void Creature_Init( struct Creature *cr, struct MeetingPlace* place, enum Colour colour )
{
   cr->place = place;
   cr->count = cr->sameCount = 0;

   cr->id = ++CreatureID;
   cr->colour = colour;
   cr->two_met = FALSE;

   pthread_attr_init( &cr->stack_att );
   pthread_attr_setstacksize( &cr->stack_att, STACK_SIZE );
   pthread_create( &cr->ht, &cr->stack_att, &CreatureThreadRun, (void*)(cr) );
}

/* format meeting times of each creature to string */
char* Creature_getResult(struct Creature* cr, char* str)
{
   char numstr[256];
   formatNumber(cr->sameCount, numstr);

   sprintf( str, "%u%s", cr->count, numstr );
   return str;
}


void runGame( int n_meeting, int ncolor, const enum Colour* colours )
{
   int i;
   int total = 0;
   char str[256];

   struct MeetingPlace place;
   struct Creature *creatures = (struct Creature*) calloc( ncolor, sizeof(struct Creature) );

   MeetingPlace_Init( &place, n_meeting );

   /* print initial color of each creature */
   for (i = 0; i < ncolor; i++)
   {
      printf( "%s ", ColourName[ colours[i] ] );
      Creature_Init( &(creatures[i]), &place, colours[i] );
   }
   printf("\n");

   /* wait for them to meet */
   for (i = 0; i < ncolor; i++)
      pthread_join( creatures[i].ht, 0 );

   /* print meeting times of each creature */
   for (i = 0; i < ncolor; i++)
   {
      printf( "%s\n", Creature_getResult(&(creatures[i]), str) );
      total += creatures[i].count;
   }

   /* print total meeting times, should equal n_meeting */
   printf( "%s\n\n", formatNumber(total, str) );

   /* cleaup & quit */
   pthread_mutex_destroy( &place.mutex );
   free( creatures );
}


void printColours( enum Colour c1, enum Colour c2 )
{
   printf( "%s + %s -> %s\n",
      ColourName[c1],
      ColourName[c2],
      ColourName[doCompliment(c1, c2)]   );
}

void printColoursTable(void)
{
   printColours(blue, blue);
   printColours(blue, red);
   printColours(blue, yellow);
   printColours(red, blue);
   printColours(red, red);
   printColours(red, yellow);
   printColours(yellow, blue);
   printColours(yellow, red);
   printColours(yellow, yellow);
}

int main(int argc, char** argv)
{
   int n = (argc == 2) ? atoi(argv[1]) : 600;

   printColoursTable();
   printf("\n");

   const enum Colour r1[] = {   blue, red, yellow   };
   const enum Colour r2[] = {   blue, red, yellow,
               red, yellow, blue,
               red, yellow, red, blue   };

   runGame( n, sizeof(r1) / sizeof(r1[0]), r1 );
   runGame( n, sizeof(r2) / sizeof(r2[0]), r2 );

   return 0;
}
