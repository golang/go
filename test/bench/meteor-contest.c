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
 * http://shootout.alioth.debian.org/
 *
 * contributed by Christian Vosteen
 */

#include <stdlib.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0

/* The board is a 50 cell hexagonal pattern.  For    . . . . .
 * maximum speed the board will be implemented as     . . . . .
 * 50 bits, which will fit into a 64 bit long long   . . . . .
 * int.                                               . . . . .
 *                                                   . . . . .
 * I will represent 0's as empty cells and 1's        . . . . .
 * as full cells.                                    . . . . .
 *                                                    . . . . .
 *                                                   . . . . .
 *                                                    . . . . .
 */

unsigned long long board = 0xFFFC000000000000ULL;

/* The puzzle pieces must be specified by the path followed
 * from one end to the other along 12 hexagonal directions.
 *
 *   Piece 0   Piece 1   Piece 2   Piece 3   Piece 4
 *
 *  O O O O    O   O O   O O O     O O O     O   O
 *         O    O O           O       O       O O
 *                           O         O         O
 *
 *   Piece 5   Piece 6   Piece 7   Piece 8   Piece 9
 *
 *    O O O     O O       O O     O O        O O O O
 *       O O       O O       O       O O O        O
 *                  O       O O
 *
 * I had to make it 12 directions because I wanted all of the
 * piece definitions to fit into the same size arrays.  It is
 * not possible to define piece 4 in terms of the 6 cardinal
 * directions in 4 moves.
 */

#define E     0
#define ESE   1
#define SE    2
#define S     3
#define SW    4
#define WSW   5
#define W     6
#define WNW   7
#define NW    8
#define N     9
#define NE    10
#define ENE   11
#define PIVOT 12

char piece_def[10][4] = {
   {  E,  E,  E, SE},
   { SE,  E, NE,  E},
   {  E,  E, SE, SW},
   {  E,  E, SW, SE},
   { SE,  E, NE,  S},
   {  E,  E, SW,  E},
   {  E, SE, SE, NE},
   {  E, SE, SE,  W},
   {  E, SE,  E,  E},
   {  E,  E,  E, SW}
};


/* To minimize the amount of work done in the recursive solve function below,
 * I'm going to allocate enough space for all legal rotations of each piece
 * at each position on the board. That's 10 pieces x 50 board positions x
 * 12 rotations.  However, not all 12 rotations will fit on every cell, so
 * I'll have to keep count of the actual number that do.
 * The pieces are going to be unsigned long long ints just like the board so
 * they can be bitwise-anded with the board to determine if they fit.
 * I'm also going to record the next possible open cell for each piece and
 * location to reduce the burden on the solve function.
 */
unsigned long long pieces[10][50][12];
int piece_counts[10][50];
char next_cell[10][50][12];

/* Returns the direction rotated 60 degrees clockwise */
char rotate(char dir) {
   return (dir + 2) % PIVOT;
}

/* Returns the direction flipped on the horizontal axis */
char flip(char dir) {
   return (PIVOT - dir) % PIVOT;
}


/* Returns the new cell index from the specified cell in the
 * specified direction.  The index is only valid if the
 * starting cell and direction have been checked by the
 * out_of_bounds function first.
 */
char shift(char cell, char dir) {
   switch(dir) {
      case E:
         return cell + 1;
      case ESE:
         if((cell / 5) % 2)
            return cell + 7;
         else
            return cell + 6;
      case SE:
         if((cell / 5) % 2)
            return cell + 6;
         else
            return cell + 5;
      case S:
         return cell + 10;
      case SW:
         if((cell / 5) % 2)
            return cell + 5;
         else
            return cell + 4;
      case WSW:
         if((cell / 5) % 2)
            return cell + 4;
         else
            return cell + 3;
      case W:
         return cell - 1;
      case WNW:
         if((cell / 5) % 2)
            return cell - 6;
         else
            return cell - 7;
      case NW:
         if((cell / 5) % 2)
            return cell - 5;
         else
            return cell - 6;
      case N:
         return cell - 10;
      case NE:
         if((cell / 5) % 2)
            return cell - 4;
         else
            return cell - 5;
      case ENE:
         if((cell / 5) % 2)
            return cell - 3;
         else
            return cell - 4;
      default:
         return cell;
   }
}

/* Returns wether the specified cell and direction will land outside
 * of the board.  Used to determine if a piece is at a legal board
 * location or not.
 */
char out_of_bounds(char cell, char dir) {
   char i;
   switch(dir) {
      case E:
         return cell % 5 == 4;
      case ESE:
         i = cell % 10;
         return i == 4 || i == 8 || i == 9 || cell >= 45;
      case SE:
         return cell % 10 == 9 || cell >= 45;
      case S:
         return cell >= 40;
      case SW:
         return cell % 10 == 0 || cell >= 45;
      case WSW:
         i = cell % 10;
         return i == 0 || i == 1 || i == 5 || cell >= 45;
      case W:
         return cell % 5 == 0;
      case WNW:
         i = cell % 10;
         return i == 0 || i == 1 || i == 5 || cell < 5;
      case NW:
         return cell % 10 == 0 || cell < 5;
      case N:
         return cell < 10;
      case NE:
         return cell % 10 == 9 || cell < 5;
      case ENE:
         i = cell % 10;
         return i == 4 || i == 8 || i == 9 || cell < 5;
      default:
         return FALSE;
   }
}

/* Rotate a piece 60 degrees clockwise */
void rotate_piece(int piece) {
   int i;
   for(i = 0; i < 4; i++)
      piece_def[piece][i] = rotate(piece_def[piece][i]);
}

/* Flip a piece along the horizontal axis */
void flip_piece(int piece) {
   int i;
   for(i = 0; i < 4; i++)
      piece_def[piece][i] = flip(piece_def[piece][i]);
}

/* Convenience function to quickly calculate all of the indices for a piece */
void calc_cell_indices(char *cell, int piece, char index) {
   cell[0] = index;
   cell[1] = shift(cell[0], piece_def[piece][0]);
   cell[2] = shift(cell[1], piece_def[piece][1]);
   cell[3] = shift(cell[2], piece_def[piece][2]);
   cell[4] = shift(cell[3], piece_def[piece][3]);
}

/* Convenience function to quickly calculate if a piece fits on the board */
int cells_fit_on_board(char *cell, int piece) {
   return (!out_of_bounds(cell[0], piece_def[piece][0]) &&
         !out_of_bounds(cell[1], piece_def[piece][1]) &&
         !out_of_bounds(cell[2], piece_def[piece][2]) &&
         !out_of_bounds(cell[3], piece_def[piece][3]));
}

/* Returns the lowest index of the cells of a piece.
 * I use the lowest index that a piece occupies as the index for looking up
 * the piece in the solve function.
 */
char minimum_of_cells(char *cell) {
   char minimum = cell[0];
   minimum = cell[1] < minimum ? cell[1] : minimum;
   minimum = cell[2] < minimum ? cell[2] : minimum;
   minimum = cell[3] < minimum ? cell[3] : minimum;
   minimum = cell[4] < minimum ? cell[4] : minimum;
   return minimum;
}

/* Calculate the lowest possible open cell if the piece is placed on the board.
 * Used to later reduce the amount of time searching for open cells in the
 * solve function.
 */
char first_empty_cell(char *cell, char minimum) {
   char first_empty = minimum;
   while(first_empty == cell[0] || first_empty == cell[1] ||
         first_empty == cell[2] || first_empty == cell[3] ||
         first_empty == cell[4])
      first_empty++;
   return first_empty;
}

/* Generate the unsigned long long int that will later be anded with the
 * board to determine if it fits.
 */
unsigned long long bitmask_from_cells(char *cell) {
   unsigned long long piece_mask = 0ULL;
   int i;
   for(i = 0; i < 5; i++)
      piece_mask |= 1ULL << cell[i];
   return piece_mask;
}

/* Record the piece and other important information in arrays that will
 * later be used by the solve function.
 */
void record_piece(int piece, int minimum, char first_empty,
      unsigned long long piece_mask) {
   pieces[piece][minimum][piece_counts[piece][minimum]] = piece_mask;
   next_cell[piece][minimum][piece_counts[piece][minimum]] = first_empty;
   piece_counts[piece][minimum]++;
}


/* Fill the entire board going cell by cell.  If any cells are "trapped"
 * they will be left alone.
 */
void fill_contiguous_space(char *board, int index) {
   if(board[index] == 1)
      return;
   board[index] = 1;
   if(!out_of_bounds(index, E))
      fill_contiguous_space(board, shift(index, E));
   if(!out_of_bounds(index, SE))
      fill_contiguous_space(board, shift(index, SE));
   if(!out_of_bounds(index, SW))
      fill_contiguous_space(board, shift(index, SW));
   if(!out_of_bounds(index, W))
      fill_contiguous_space(board, shift(index, W));
   if(!out_of_bounds(index, NW))
      fill_contiguous_space(board, shift(index, NW));
   if(!out_of_bounds(index, NE))
      fill_contiguous_space(board, shift(index, NE));
}


/* To thin the number of pieces, I calculate if any of them trap any empty
 * cells at the edges.  There are only a handful of exceptions where the
 * the board can be solved with the trapped cells.  For example:  piece 8 can
 * trap 5 cells in the corner, but piece 3 can fit in those cells, or piece 0
 * can split the board in half where both halves are viable.
 */
int has_island(char *cell, int piece) {
   char temp_board[50];
   char c;
   int i;
   for(i = 0; i < 50; i++)
      temp_board[i] = 0;
   for(i = 0; i < 5; i++)
      temp_board[((int)cell[i])] = 1;
   i = 49;
   while(temp_board[i] == 1)
      i--;
   fill_contiguous_space(temp_board, i);
   c = 0;
   for(i = 0; i < 50; i++)
      if(temp_board[i] == 0)
         c++;
   if(c == 0 || (c == 5 && piece == 8) || (c == 40 && piece == 8) ||
         (c % 5 == 0 && piece == 0))
      return FALSE;
   else
      return TRUE;
}


/* Calculate all six rotations of the specified piece at the specified index.
 * We calculate only half of piece 3's rotations.  This is because any solution
 * found has an identical solution rotated 180 degrees.  Thus we can reduce the
 * number of attempted pieces in the solve algorithm by not including the 180-
 * degree-rotated pieces of ONE of the pieces.  I chose piece 3 because it gave
 * me the best time ;)
 */
 void calc_six_rotations(char piece, char index) {
   char rotation, cell[5];
   char minimum, first_empty;
   unsigned long long piece_mask;

   for(rotation = 0; rotation < 6; rotation++) {
      if(piece != 3 || rotation < 3) {
         calc_cell_indices(cell, piece, index);
         if(cells_fit_on_board(cell, piece) && !has_island(cell, piece)) {
            minimum = minimum_of_cells(cell);
            first_empty = first_empty_cell(cell, minimum);
            piece_mask = bitmask_from_cells(cell);
            record_piece(piece, minimum, first_empty, piece_mask);
         }
      }
      rotate_piece(piece);
   }
}

/* Calculate every legal rotation for each piece at each board location. */
void calc_pieces(void) {
   char piece, index;

   for(piece = 0; piece < 10; piece++) {
      for(index = 0; index < 50; index++) {
         calc_six_rotations(piece, index);
         flip_piece(piece);
         calc_six_rotations(piece, index);
      }
   }
}



/* Calculate all 32 possible states for a 5-bit row and all rows that will
 * create islands that follow any of the 32 possible rows.  These pre-
 * calculated 5-bit rows will be used to find islands in a partially solved
 * board in the solve function.
 */
#define ROW_MASK 0x1F
#define TRIPLE_MASK 0x7FFF
char all_rows[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
int bad_even_rows[32][32];
int bad_odd_rows[32][32];
int bad_even_triple[32768];
int bad_odd_triple[32768];

int rows_bad(char row1, char row2, int even) {
   /* even is referring to row1 */
   int i, in_zeroes, group_okay;
   char block, row2_shift;
   /* Test for blockages at same index and shifted index */
   if(even)
      row2_shift = ((row2 << 1) & ROW_MASK) | 0x01;
   else
      row2_shift = (row2 >> 1) | 0x10;
   block = ((row1 ^ row2) & row2) & ((row1 ^ row2_shift) & row2_shift);
   /* Test for groups of 0's */
   in_zeroes = FALSE;
   group_okay = FALSE;
   for(i = 0; i < 5; i++) {
      if(row1 & (1 << i)) {
         if(in_zeroes) {
            if(!group_okay)
               return TRUE;
            in_zeroes = FALSE;
            group_okay = FALSE;
         }
      } else {
         if(!in_zeroes)
            in_zeroes = TRUE;
         if(!(block & (1 << i)))
            group_okay = TRUE;
      }
   }
   if(in_zeroes)
      return !group_okay;
   else
      return FALSE;
}

/* Check for cases where three rows checked sequentially cause a false
 * positive.  One scenario is when 5 cells may be surrounded where piece 5
 * or 7 can fit.  The other scenario is when piece 2 creates a hook shape.
 */
int triple_is_okay(char row1, char row2, char row3, int even) {
   if(even) {
      /* There are four cases:
       * row1: 00011  00001  11001  10101
       * row2: 01011  00101  10001  10001
       * row3: 011??  00110  ?????  ?????
       */
      return ((row1 == 0x03) && (row2 == 0x0B) && ((row3 & 0x1C) == 0x0C)) ||
            ((row1 == 0x01) && (row2 == 0x05) && (row3 == 0x06)) ||
            ((row1 == 0x19) && (row2 == 0x11)) ||
            ((row1 == 0x15) && (row2 == 0x11));
   } else {
      /* There are two cases:
       * row1: 10011  10101
       * row2: 10001  10001
       * row3: ?????  ?????
       */
      return ((row1 == 0x13) && (row2 == 0x11)) ||
            ((row1 == 0x15) && (row2 == 0x11));
   }
}


void calc_rows(void) {
   int row1, row2, row3;
   int result1, result2;
   for(row1 = 0; row1 < 32; row1++) {
      for(row2 = 0; row2 < 32; row2++) {
         bad_even_rows[row1][row2] = rows_bad(row1, row2, TRUE);
         bad_odd_rows[row1][row2] = rows_bad(row1, row2, FALSE);
      }
   }
   for(row1 = 0; row1 < 32; row1++) {
      for(row2 = 0; row2 < 32; row2++) {
         for(row3 = 0; row3 < 32; row3++) {
            result1 = bad_even_rows[row1][row2];
            result2 = bad_odd_rows[row2][row3];
            if(result1 == FALSE && result2 == TRUE
                  && triple_is_okay(row1, row2, row3, TRUE))
               bad_even_triple[row1+(row2*32)+(row3*1024)] = FALSE;
            else
               bad_even_triple[row1+(row2*32)+(row3*1024)] = result1 || result2;

            result1 = bad_odd_rows[row1][row2];
            result2 = bad_even_rows[row2][row3];
            if(result1 == FALSE && result2 == TRUE
                  && triple_is_okay(row1, row2, row3, FALSE))
               bad_odd_triple[row1+(row2*32)+(row3*1024)] = FALSE;
            else
               bad_odd_triple[row1+(row2*32)+(row3*1024)] = result1 || result2;
         }
      }
   }
}



/* Calculate islands while solving the board.
 */
int boardHasIslands(char cell) {
   /* Too low on board, don't bother checking */
   if(cell >= 40)
      return FALSE;
   int current_triple = (board >> ((cell / 5) * 5)) & TRIPLE_MASK;
   if((cell / 5) % 2)
      return bad_odd_triple[current_triple];
   else
      return bad_even_triple[current_triple];
}


/* The recursive solve algorithm.  Try to place each permutation in the upper-
 * leftmost empty cell.  Mark off available pieces as it goes along.
 * Because the board is a bit mask, the piece number and bit mask must be saved
 * at each successful piece placement.  This data is used to create a 50 char
 * array if a solution is found.
 */
short avail = 0x03FF;
char sol_nums[10];
unsigned long long sol_masks[10];
signed char solutions[2100][50];
int solution_count = 0;
int max_solutions = 2100;

void record_solution(void) {
   int sol_no, index;
   unsigned long long sol_mask;
   for(sol_no = 0; sol_no < 10; sol_no++) {
      sol_mask = sol_masks[sol_no];
      for(index = 0; index < 50; index++) {
         if(sol_mask & 1ULL) {
            solutions[solution_count][index] = sol_nums[sol_no];
            /* Board rotated 180 degrees is a solution too! */
            solutions[solution_count+1][49-index] = sol_nums[sol_no];
         }
         sol_mask = sol_mask >> 1;
      }
   }
   solution_count += 2;
}

void solve(int depth, int cell) {
   int piece, rotation, max_rots;
   unsigned long long *piece_mask;
   short piece_no_mask;

   if(solution_count >= max_solutions)
      return;

   while(board & (1ULL << cell))
      cell++;

   for(piece = 0; piece < 10; piece++) {
      piece_no_mask = 1 << piece;
      if(!(avail & piece_no_mask))
         continue;
      avail ^= piece_no_mask;
      max_rots = piece_counts[piece][cell];
      piece_mask = pieces[piece][cell];
      for(rotation = 0; rotation < max_rots; rotation++) {
         if(!(board & *(piece_mask + rotation))) {
            sol_nums[depth] = piece;
            sol_masks[depth] = *(piece_mask + rotation);
            if(depth == 9) {
               /* Solution found!!!!!11!!ONE! */
               record_solution();
               avail ^= piece_no_mask;
               return;
            }
            board |= *(piece_mask + rotation);
            if(!boardHasIslands(next_cell[piece][cell][rotation]))
               solve(depth + 1, next_cell[piece][cell][rotation]);
            board ^= *(piece_mask + rotation);
         }
      }
      avail ^= piece_no_mask;
   }
}


/* qsort comparator - used to find first and last solutions */
int solution_sort(const void *elem1, const void *elem2) {
   signed char *char1 = (signed char *) elem1;
   signed char *char2 = (signed char *) elem2;
   int i = 0;
   while(i < 50 && char1[i] == char2[i])
      i++;
   return char1[i] - char2[i];
}


/* pretty print a board in the specified hexagonal format */
void pretty(signed char *b) {
   int i;
   for(i = 0; i < 50; i += 10) {
      printf("%c %c %c %c %c \n %c %c %c %c %c \n", b[i]+'0', b[i+1]+'0',
            b[i+2]+'0', b[i+3]+'0', b[i+4]+'0', b[i+5]+'0', b[i+6]+'0',
            b[i+7]+'0', b[i+8]+'0', b[i+9]+'0');
   }
   printf("\n");
}

int main(int argc, char **argv) {
   if(argc > 1)
      max_solutions = atoi(argv[1]);
   calc_pieces();
   calc_rows();
   solve(0, 0);
   printf("%d solutions found\n\n", solution_count);
   qsort(solutions, solution_count, 50 * sizeof(signed char), solution_sort);
   pretty(solutions[0]);
   pretty(solutions[solution_count-1]);
   return 0;
}
