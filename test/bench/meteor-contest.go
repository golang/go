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
 * contributed by The Go Authors.
 * based on meteor-contest.c by Christian Vosteen
 */

package main

import (
	"flag"
	"fmt"
)

var max_solutions = flag.Int("n", 2100, "maximum number of solutions")


func boolInt(b bool) int8 {
	if b {
		return 1
	}
	return 0
}

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

var board uint64 = 0xFFFC000000000000

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

const (
	E = iota
	ESE
	SE
	S
	SW
	WSW
	W
	WNW
	NW
	N
	NE
	ENE
	PIVOT
)

var piece_def = [10][4]int8{
	[4]int8{E, E, E, SE},
	[4]int8{SE, E, NE, E},
	[4]int8{E, E, SE, SW},
	[4]int8{E, E, SW, SE},
	[4]int8{SE, E, NE, S},
	[4]int8{E, E, SW, E},
	[4]int8{E, SE, SE, NE},
	[4]int8{E, SE, SE, W},
	[4]int8{E, SE, E, E},
	[4]int8{E, E, E, SW},
}


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
var (
	pieces       [10][50][12]uint64
	piece_counts [10][50]int
	next_cell    [10][50][12]int8
)

/* Returns the direction rotated 60 degrees clockwise */
func rotate(dir int8) int8 { return (dir + 2) % PIVOT }

/* Returns the direction flipped on the horizontal axis */
func flip(dir int8) int8 { return (PIVOT - dir) % PIVOT }


/* Returns the new cell index from the specified cell in the
 * specified direction.  The index is only valid if the
 * starting cell and direction have been checked by the
 * out_of_bounds function first.
 */
func shift(cell, dir int8) int8 {
	switch dir {
	case E:
		return cell + 1
	case ESE:
		if ((cell / 5) % 2) != 0 {
			return cell + 7
		} else {
			return cell + 6
		}
	case SE:
		if ((cell / 5) % 2) != 0 {
			return cell + 6
		} else {
			return cell + 5
		}
	case S:
		return cell + 10
	case SW:
		if ((cell / 5) % 2) != 0 {
			return cell + 5
		} else {
			return cell + 4
		}
	case WSW:
		if ((cell / 5) % 2) != 0 {
			return cell + 4
		} else {
			return cell + 3
		}
	case W:
		return cell - 1
	case WNW:
		if ((cell / 5) % 2) != 0 {
			return cell - 6
		} else {
			return cell - 7
		}
	case NW:
		if ((cell / 5) % 2) != 0 {
			return cell - 5
		} else {
			return cell - 6
		}
	case N:
		return cell - 10
	case NE:
		if ((cell / 5) % 2) != 0 {
			return cell - 4
		} else {
			return cell - 5
		}
	case ENE:
		if ((cell / 5) % 2) != 0 {
			return cell - 3
		} else {
			return cell - 4
		}
	}
	return cell
}

/* Returns wether the specified cell and direction will land outside
 * of the board.  Used to determine if a piece is at a legal board
 * location or not.
 */
func out_of_bounds(cell, dir int8) bool {
	switch dir {
	case E:
		return cell%5 == 4
	case ESE:
		i := cell % 10
		return i == 4 || i == 8 || i == 9 || cell >= 45
	case SE:
		return cell%10 == 9 || cell >= 45
	case S:
		return cell >= 40
	case SW:
		return cell%10 == 0 || cell >= 45
	case WSW:
		i := cell % 10
		return i == 0 || i == 1 || i == 5 || cell >= 45
	case W:
		return cell%5 == 0
	case WNW:
		i := cell % 10
		return i == 0 || i == 1 || i == 5 || cell < 5
	case NW:
		return cell%10 == 0 || cell < 5
	case N:
		return cell < 10
	case NE:
		return cell%10 == 9 || cell < 5
	case ENE:
		i := cell % 10
		return i == 4 || i == 8 || i == 9 || cell < 5
	}
	return false
}

/* Rotate a piece 60 degrees clockwise */
func rotate_piece(piece int) {
	for i := 0; i < 4; i++ {
		piece_def[piece][i] = rotate(piece_def[piece][i])
	}
}

/* Flip a piece along the horizontal axis */
func flip_piece(piece int) {
	for i := 0; i < 4; i++ {
		piece_def[piece][i] = flip(piece_def[piece][i])
	}
}

/* Convenience function to quickly calculate all of the indices for a piece */
func calc_cell_indices(cell []int8, piece int, index int8) {
	cell[0] = index
	for i := 1; i < 5; i++ {
		cell[i] = shift(cell[i-1], piece_def[piece][i-1])
	}
}

/* Convenience function to quickly calculate if a piece fits on the board */
func cells_fit_on_board(cell []int8, piece int) bool {
	return !out_of_bounds(cell[0], piece_def[piece][0]) &&
		!out_of_bounds(cell[1], piece_def[piece][1]) &&
		!out_of_bounds(cell[2], piece_def[piece][2]) &&
		!out_of_bounds(cell[3], piece_def[piece][3])
}

/* Returns the lowest index of the cells of a piece.
 * I use the lowest index that a piece occupies as the index for looking up
 * the piece in the solve function.
 */
func minimum_of_cells(cell []int8) int8 {
	minimum := cell[0]
	for i := 1; i < 5; i++ {
		if cell[i] < minimum {
			minimum = cell[i]
		}
	}
	return minimum
}

/* Calculate the lowest possible open cell if the piece is placed on the board.
 * Used to later reduce the amount of time searching for open cells in the
 * solve function.
 */
func first_empty_cell(cell []int8, minimum int8) int8 {
	first_empty := minimum
	for first_empty == cell[0] || first_empty == cell[1] ||
		first_empty == cell[2] || first_empty == cell[3] ||
		first_empty == cell[4] {
		first_empty++
	}
	return first_empty
}

/* Generate the unsigned long long int that will later be anded with the
 * board to determine if it fits.
 */
func bitmask_from_cells(cell []int8) uint64 {
	var piece_mask uint64
	for i := 0; i < 5; i++ {
		piece_mask |= 1 << uint(cell[i])
	}
	return piece_mask
}

/* Record the piece and other important information in arrays that will
 * later be used by the solve function.
 */
func record_piece(piece int, minimum int8, first_empty int8, piece_mask uint64) {
	pieces[piece][minimum][piece_counts[piece][minimum]] = piece_mask
	next_cell[piece][minimum][piece_counts[piece][minimum]] = first_empty
	piece_counts[piece][minimum]++
}


/* Fill the entire board going cell by cell.  If any cells are "trapped"
 * they will be left alone.
 */
func fill_contiguous_space(board []int8, index int8) {
	if board[index] == 1 {
		return
	}
	board[index] = 1
	if !out_of_bounds(index, E) {
		fill_contiguous_space(board, shift(index, E))
	}
	if !out_of_bounds(index, SE) {
		fill_contiguous_space(board, shift(index, SE))
	}
	if !out_of_bounds(index, SW) {
		fill_contiguous_space(board, shift(index, SW))
	}
	if !out_of_bounds(index, W) {
		fill_contiguous_space(board, shift(index, W))
	}
	if !out_of_bounds(index, NW) {
		fill_contiguous_space(board, shift(index, NW))
	}
	if !out_of_bounds(index, NE) {
		fill_contiguous_space(board, shift(index, NE))
	}
}


/* To thin the number of pieces, I calculate if any of them trap any empty
 * cells at the edges.  There are only a handful of exceptions where the
 * the board can be solved with the trapped cells.  For example:  piece 8 can
 * trap 5 cells in the corner, but piece 3 can fit in those cells, or piece 0
 * can split the board in half where both halves are viable.
 */
func has_island(cell []int8, piece int) bool {
	temp_board := make([]int8, 50)
	var i int
	for i = 0; i < 5; i++ {
		temp_board[cell[i]] = 1
	}
	i = 49
	for temp_board[i] == 1 {
		i--
	}
	fill_contiguous_space(temp_board, int8(i))
	c := 0
	for i = 0; i < 50; i++ {
		if temp_board[i] == 0 {
			c++
		}
	}
	if c == 0 || (c == 5 && piece == 8) || (c == 40 && piece == 8) ||
		(c%5 == 0 && piece == 0) {
		return false
	}
	return true
}


/* Calculate all six rotations of the specified piece at the specified index.
 * We calculate only half of piece 3's rotations.  This is because any solution
 * found has an identical solution rotated 180 degrees.  Thus we can reduce the
 * number of attempted pieces in the solve algorithm by not including the 180-
 * degree-rotated pieces of ONE of the pieces.  I chose piece 3 because it gave
 * me the best time ;)
 */
func calc_six_rotations(piece, index int) {
	cell := make([]int8, 5)
	for rotation := 0; rotation < 6; rotation++ {
		if piece != 3 || rotation < 3 {
			calc_cell_indices(cell, piece, int8(index))
			if cells_fit_on_board(cell, piece) && !has_island(cell, piece) {
				minimum := minimum_of_cells(cell)
				first_empty := first_empty_cell(cell, minimum)
				piece_mask := bitmask_from_cells(cell)
				record_piece(piece, minimum, first_empty, piece_mask)
			}
		}
		rotate_piece(piece)
	}
}

/* Calculate every legal rotation for each piece at each board location. */
func calc_pieces() {
	for piece := 0; piece < 10; piece++ {
		for index := 0; index < 50; index++ {
			calc_six_rotations(piece, index)
			flip_piece(piece)
			calc_six_rotations(piece, index)
		}
	}
}


/* Calculate all 32 possible states for a 5-bit row and all rows that will
 * create islands that follow any of the 32 possible rows.  These pre-
 * calculated 5-bit rows will be used to find islands in a partially solved
 * board in the solve function.
 */
const (
	ROW_MASK    = 0x1F
	TRIPLE_MASK = 0x7FFF
)

var (
	all_rows = [32]int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	}
	bad_even_rows   [32][32]int8
	bad_odd_rows    [32][32]int8
	bad_even_triple [32768]int8
	bad_odd_triple  [32768]int8
)

func rows_bad(row1, row2 int8, even bool) int8 {
	/* even is referring to row1 */
	var row2_shift int8
	/* Test for blockages at same index and shifted index */
	if even {
		row2_shift = ((row2 << 1) & ROW_MASK) | 0x01
	} else {
		row2_shift = (row2 >> 1) | 0x10
	}
	block := ((row1 ^ row2) & row2) & ((row1 ^ row2_shift) & row2_shift)
	/* Test for groups of 0's */
	in_zeroes := false
	group_okay := false
	for i := uint8(0); i < 5; i++ {
		if row1&(1<<i) != 0 {
			if in_zeroes {
				if !group_okay {
					return 1
				}
				in_zeroes = false
				group_okay = false
			}
		} else {
			if !in_zeroes {
				in_zeroes = true
			}
			if (block & (1 << i)) == 0 {
				group_okay = true
			}
		}
	}
	if in_zeroes {
		return boolInt(!group_okay)
	}
	return 0
}

/* Check for cases where three rows checked sequentially cause a false
 * positive.  One scenario is when 5 cells may be surrounded where piece 5
 * or 7 can fit.  The other scenario is when piece 2 creates a hook shape.
 */
func triple_is_okay(row1, row2, row3 int, even bool) bool {
	if even {
		/* There are four cases:
		 * row1: 00011  00001  11001  10101
		 * row2: 01011  00101  10001  10001
		 * row3: 011??  00110  ?????  ?????
		 */
		return ((row1 == 0x03) && (row2 == 0x0B) && ((row3 & 0x1C) == 0x0C)) ||
			((row1 == 0x01) && (row2 == 0x05) && (row3 == 0x06)) ||
			((row1 == 0x19) && (row2 == 0x11)) ||
			((row1 == 0x15) && (row2 == 0x11))
	}
	/* There are two cases:
	 * row1: 10011  10101
	 * row2: 10001  10001
	 * row3: ?????  ?????
	 */
	return ((row1 == 0x13) && (row2 == 0x11)) ||
		((row1 == 0x15) && (row2 == 0x11))
}

func calc_rows() {
	for row1 := int8(0); row1 < 32; row1++ {
		for row2 := int8(0); row2 < 32; row2++ {
			bad_even_rows[row1][row2] = rows_bad(row1, row2, true)
			bad_odd_rows[row1][row2] = rows_bad(row1, row2, false)
		}
	}
	for row1 := 0; row1 < 32; row1++ {
		for row2 := 0; row2 < 32; row2++ {
			for row3 := 0; row3 < 32; row3++ {
				result1 := bad_even_rows[row1][row2]
				result2 := bad_odd_rows[row2][row3]
				if result1 == 0 && result2 != 0 && triple_is_okay(row1, row2, row3, true) {
					bad_even_triple[row1+(row2*32)+(row3*1024)] = 0
				} else {
					bad_even_triple[row1+(row2*32)+(row3*1024)] = boolInt(result1 != 0 || result2 != 0)
				}

				result1 = bad_odd_rows[row1][row2]
				result2 = bad_even_rows[row2][row3]
				if result1 == 0 && result2 != 0 && triple_is_okay(row1, row2, row3, false) {
					bad_odd_triple[row1+(row2*32)+(row3*1024)] = 0
				} else {
					bad_odd_triple[row1+(row2*32)+(row3*1024)] = boolInt(result1 != 0 || result2 != 0)
				}
			}
		}
	}
}


/* Calculate islands while solving the board.
 */
func boardHasIslands(cell int8) int8 {
	/* Too low on board, don't bother checking */
	if cell >= 40 {
		return 0
	}
	current_triple := (board >> uint((cell/5)*5)) & TRIPLE_MASK
	if (cell/5)%2 != 0 {
		return bad_odd_triple[current_triple]
	}
	return bad_even_triple[current_triple]
}


/* The recursive solve algorithm.  Try to place each permutation in the upper-
 * leftmost empty cell.  Mark off available pieces as it goes along.
 * Because the board is a bit mask, the piece number and bit mask must be saved
 * at each successful piece placement.  This data is used to create a 50 char
 * array if a solution is found.
 */
var (
	avail          uint16 = 0x03FF
	sol_nums       [10]int8
	sol_masks      [10]uint64
	solutions      [2100][50]int8
	solution_count = 0
)

func record_solution() {
	for sol_no := 0; sol_no < 10; sol_no++ {
		sol_mask := sol_masks[sol_no]
		for index := 0; index < 50; index++ {
			if sol_mask&1 == 1 {
				solutions[solution_count][index] = sol_nums[sol_no]
				/* Board rotated 180 degrees is a solution too! */
				solutions[solution_count+1][49-index] = sol_nums[sol_no]
			}
			sol_mask = sol_mask >> 1
		}
	}
	solution_count += 2
}

func solve(depth, cell int8) {
	if solution_count >= *max_solutions {
		return
	}

	for board&(1<<uint(cell)) != 0 {
		cell++
	}

	for piece := int8(0); piece < 10; piece++ {
		var piece_no_mask uint16 = 1 << uint(piece)
		if avail&piece_no_mask == 0 {
			continue
		}
		avail ^= piece_no_mask
		max_rots := piece_counts[piece][cell]
		piece_mask := pieces[piece][cell]
		for rotation := 0; rotation < max_rots; rotation++ {
			if board&piece_mask[rotation] == 0 {
				sol_nums[depth] = piece
				sol_masks[depth] = piece_mask[rotation]
				if depth == 9 {
					/* Solution found!!!!!11!!ONE! */
					record_solution()
					avail ^= piece_no_mask
					return
				}
				board |= piece_mask[rotation]
				if boardHasIslands(next_cell[piece][cell][rotation]) == 0 {
					solve(depth+1, next_cell[piece][cell][rotation])
				}
				board ^= piece_mask[rotation]
			}
		}
		avail ^= piece_no_mask
	}
}

/* pretty print a board in the specified hexagonal format */
func pretty(b *[50]int8) {
	for i := 0; i < 50; i += 10 {
		fmt.Printf("%c %c %c %c %c \n %c %c %c %c %c \n", b[i]+'0', b[i+1]+'0',
			b[i+2]+'0', b[i+3]+'0', b[i+4]+'0', b[i+5]+'0', b[i+6]+'0',
			b[i+7]+'0', b[i+8]+'0', b[i+9]+'0')
	}
	fmt.Printf("\n")
}

/* Find smallest and largest solutions */
func smallest_largest() (smallest, largest *[50]int8) {
	smallest = &solutions[0]
	largest = &solutions[0]
	for i := 1; i < solution_count; i++ {
		candidate := &solutions[i]
		for j, s := range *smallest {
			c := candidate[j]
			if c == s {
				continue
			}
			if c < s {
				smallest = candidate
			}
			break
		}
		for j, s := range *largest {
			c := candidate[j]
			if c == s {
				continue
			}
			if c > s {
				largest = candidate
			}
			break
		}
	}
	return
}

func main() {
	flag.Parse()
	calc_pieces()
	calc_rows()
	solve(0, 0)
	fmt.Printf("%d solutions found\n\n", solution_count)
	smallest, largest := smallest_largest()
	pretty(smallest)
	pretty(largest)
}
