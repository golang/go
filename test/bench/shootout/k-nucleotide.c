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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <glib.h>

typedef struct stat_s stat_t;
struct stat_s
{
   const gchar *key;
   long stat;
};

#define MAX_ELM (8192 / sizeof (stat_t))

static int
generate_frequencies (int fl, char *buffer, long buflen,
		      GHashTable *ht, GTrashStack **ts, GPtrArray *roots, GStringChunk *sc)
{
   gchar *key;
   long i;

   if (fl > buflen) return 0;
   if (fl == 0) return 0;

   for (i = 0; i < buflen - fl + 1; ++i)
     {
	char nulled;
	stat_t *stat;

	nulled = buffer[i + fl];
	buffer[i + fl] = '\0';

	key = g_string_chunk_insert_const(sc, buffer + i);

	stat = g_hash_table_lookup(ht, key);
	if (!stat)
	  {
	     stat = g_trash_stack_pop(ts);
	     if (!stat)
	       {
		  int j;

		  stat = malloc(sizeof (stat_t) * MAX_ELM);
		  g_ptr_array_add(roots, stat);

		  for (j = 1; j < MAX_ELM; ++j)
		    g_trash_stack_push(ts, stat + j);
	       }
	     stat->stat = 1;
	     stat->key = key;

	     g_hash_table_insert(ht, key, stat);
	  }
	else
	  stat->stat++;

	buffer[i + fl] = nulled;
     }

   return buflen - fl + 1;
}

static int
cmp_func(gconstpointer a, gconstpointer b)
{
   const stat_t *left = a;
   const stat_t *right = b;

   return right->stat - left->stat;
}

static void
sorted_list(gpointer key, gpointer value, gpointer user_data)
{
   stat_t *data = value;
   GList **lst = user_data;

   *lst = g_list_insert_sorted(*lst, data, cmp_func);
}

static void
display_stat(gpointer data, gpointer user_data)
{
   long *total = user_data;
   stat_t *st = data;

   printf("%s %.3f\n", st->key, 100 * (float) st->stat / *total);
}

void
write_frequencies (int fl, char *buffer, long buflen, GTrashStack **ts, GPtrArray *roots)
{
   GStringChunk *sc;
   GHashTable *ht;
   GList *lst;
   long total;

   ht = g_hash_table_new_full(g_str_hash, g_str_equal, NULL /* free key */, NULL /* free value */);
   sc = g_string_chunk_new(buflen);
   lst = NULL;

   total = generate_frequencies (fl, buffer, buflen, ht, ts, roots, sc);

   if (!total) goto on_error;

   g_hash_table_foreach(ht, sorted_list, &lst);
   g_list_foreach(lst, display_stat, &total);
   g_list_free(lst);

 on_error:
   g_hash_table_destroy(ht);
   g_string_chunk_free(sc);
}

void
write_count (char *searchFor, char *buffer, long buflen, GTrashStack **ts, GPtrArray *roots)
{
   GStringChunk *sc;
   GHashTable *ht;
   stat_t *result;
   GList *lst;
   long total;
   long fl;

   fl = strlen(searchFor);

   ht = g_hash_table_new_full(g_str_hash, g_str_equal, NULL /* free key */, NULL /* free value */);
   sc = g_string_chunk_new(buflen);
   lst = NULL;
   result = NULL;

   total = generate_frequencies (fl, buffer, buflen, ht, ts, roots, sc);

   if (!total) goto on_error;

   result = g_hash_table_lookup(ht, searchFor);

 on_error:
   printf("%ld\t%s\n", result ? result->stat : 0, searchFor);

   g_hash_table_destroy(ht);
   g_string_chunk_free(sc);
}

int
main ()
{
   char buffer[4096];
   GTrashStack *ts;
   GPtrArray *roots;
   GString *stuff;
   gchar *s;
   int len;

   roots = g_ptr_array_new();
   ts = NULL;

   while (fgets(buffer, sizeof (buffer), stdin))
     if (strncmp(buffer, ">THREE", 6) == 0)
       break;

   stuff = g_string_new(NULL);

   while (fgets(buffer, sizeof (buffer), stdin))
     {
	size_t sz;

	if (buffer[0] == '>')
	  break;

	sz = strlen(buffer);
	if (buffer[sz - 1] == '\n')
	  --sz;

	stuff = g_string_append_len(stuff, buffer, sz);
     }

   stuff = g_string_ascii_up(stuff);
   len = stuff->len;
   s = g_string_free(stuff, FALSE);

   write_frequencies(1, s, len, &ts, roots);
   printf("\n");
   write_frequencies(2, s, len, &ts, roots);
   printf("\n");
   write_count("GGT", s, len, &ts, roots);
   write_count("GGTA", s, len, &ts, roots);
   write_count("GGTATT", s, len, &ts, roots);
   write_count("GGTATTTTAATT", s, len, &ts, roots);
   write_count("GGTATTTTAATTTATAGT", s, len, &ts, roots);

   free(s);

   g_ptr_array_foreach(roots, (GFunc)free, NULL);
   g_ptr_array_free(roots, TRUE);

   return 0;
}
