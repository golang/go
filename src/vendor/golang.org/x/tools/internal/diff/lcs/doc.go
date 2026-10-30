// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package lcs contains code to find longest-common-subsequences
// (and diffs)
package lcs

/*
Compute longest-common-subsequences of two slices A, B using
algorithms from Myers' paper. A longest-common-subsequence
(LCS from now on) of A and B is a maximal set of lexically increasing
pairs of subscripts (x,y) with A[x]==B[y]. There may be many LCS, but
they all have the same length. An LCS determines a sequence of edits
that changes A into B.

The key concept is the edit graph of A and B.
If A has length N and B has length M, then the edit graph has
vertices v[i][j] for 0 <= i <= N, 0 <= j <= M. There is a
horizontal edge from v[i][j] to v[i+1][j] whenever both are in
the graph, and a vertical edge from v[i][j] to f[i][j+1] similarly.
When A[i] == B[j] there is a diagonal edge from v[i][j] to v[i+1][j+1].

A path between in the graph between (0,0) and (N,M) determines a sequence
of edits converting A into B: each horizontal edge corresponds to removing
an element of A, and each vertical edge corresponds to inserting an
element of B.

A vertex (x,y) is on (forward) diagonal k if x-y=k. A path in the graph
is of length D if it has D non-diagonal edges. The algorithms generate
forward paths (in which at least one of x,y increases at each edge),
or backward paths (in which at least one of x,y decreases at each edge),
or a combination. (Note that the orientation is the traditional mathematical one,
with the origin in the lower-left corner.)

Here is the edit graph for A:"aabbaa", B:"aacaba". (I know the diagonals look weird.)
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   b      |             |             |   ___/‾‾‾   |   ___/‾‾‾   |             |             |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   c      |             |             |             |             |             |             |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
                 a             a             b             b             a             a


The algorithm labels a vertex (x,y) with D,k if it is on diagonal k and at
the end of a maximal path of length D. (Because x-y=k it suffices to remember
only the x coordinate of the vertex.)

The forward algorithm: Find the longest diagonal starting at (0,0) and
label its end with D=0,k=0. From that vertex take a vertical step and
then follow the longest diagonal (up and to the right), and label that vertex
with D=1,k=-1. From the D=0,k=0 point take a horizontal step and the follow
the longest diagonal (up and to the right) and label that vertex
D=1,k=1. In the same way, having labelled all the D vertices,
from a vertex labelled D,k find two vertices
tentatively labelled D+1,k-1 and D+1,k+1. There may be two on the same
diagonal, in which case take the one with the larger x.

Eventually the path gets to (N,M), and the diagonals on it are the LCS.

Here is the edit graph with the ends of D-paths labelled. (So, for instance,
0/2,2 indicates that x=2,y=2 is labelled with 0, as it should be, since the first
step is to go up the longest diagonal from (0,0).)
A:"aabbaa", B:"aacaba"
          ⊙   -------   ⊙   -------   ⊙   -------(3/3,6)-------   ⊙   -------(3/5,6)-------(4/6,6)
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------(2/3,5)-------   ⊙   -------   ⊙   -------   ⊙
   b      |             |             |   ___/‾‾‾   |   ___/‾‾‾   |             |             |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------(3/5,4)-------   ⊙
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------(1/2,3)-------(2/3,3)-------   ⊙   -------   ⊙   -------   ⊙
   c      |             |             |             |             |             |             |
          ⊙   -------   ⊙   -------(0/2,2)-------(1/3,2)-------(2/4,2)-------(3/5,2)-------(4/6,2)
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
   a      |   ___/‾‾‾   |   ___/‾‾‾   |             |             |   ___/‾‾‾   |   ___/‾‾‾   |
          ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙   -------   ⊙
                 a             a             b             b             a             a

The 4-path is reconstructed starting at (4/6,6), horizontal to (3/5,6), diagonal to (3,4), vertical
to (2/3,3), horizontal to (1/2,3), vertical to (0/2,2), and diagonal to (0,0). As expected,
there are 4 non-diagonal steps, and the diagonals form an LCS.

There is a symmetric backward algorithm, which gives (backwards labels are prefixed with a colon):
A:"aabbaa", B:"aacaba"
            ⊙   --------    ⊙   --------    ⊙   --------    ⊙   --------    ⊙   --------    ⊙   --------    ⊙
    a       |   ____/‾‾‾    |   ____/‾‾‾    |               |               |   ____/‾‾‾    |   ____/‾‾‾    |
            ⊙   --------    ⊙   --------    ⊙   --------    ⊙   --------    ⊙   --------(:0/5,5)--------    ⊙
    b       |               |               |   ____/‾‾‾    |   ____/‾‾‾    |               |               |
            ⊙   --------    ⊙   --------    ⊙   --------(:1/3,4)--------    ⊙   --------    ⊙   --------    ⊙
    a       |   ____/‾‾‾    |   ____/‾‾‾    |               |               |   ____/‾‾‾    |   ____/‾‾‾    |
        (:3/0,3)--------(:2/1,3)--------    ⊙   --------(:2/3,3)--------(:1/4,3)--------    ⊙   --------    ⊙
    c       |               |               |               |               |               |               |
            ⊙   --------    ⊙   --------    ⊙   --------(:3/3,2)--------(:2/4,2)--------    ⊙   --------    ⊙
    a       |   ____/‾‾‾    |   ____/‾‾‾    |               |               |   ____/‾‾‾    |   ____/‾‾‾    |
        (:3/0,1)--------    ⊙   --------    ⊙   --------    ⊙   --------(:3/4,1)--------    ⊙   --------    ⊙
    a       |   ____/‾‾‾    |   ____/‾‾‾    |               |               |   ____/‾‾‾    |   ____/‾‾‾    |
        (:4/0,0)--------    ⊙   --------    ⊙   --------    ⊙   --------(:4/4,0)--------    ⊙   --------    ⊙
                    a               a               b               b               a               a

Neither of these is ideal for use in an editor, where it is undesirable to send very long diffs to the
front end. It's tricky to decide exactly what 'very long diffs' means, as "replace A by B" is very short.
We want to control how big D can be, by stopping when it gets too large. The forward algorithm then
privileges common prefixes, and the backward algorithm privileges common suffixes. Either is an undesirable
asymmetry.

Fortunately there is a two-sided algorithm, implied by results in Myers' paper. Here's what the labels in
the edit graph look like.
A:"aabbaa", B:"aacaba"
             ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙
    a        |    ____/‾‾‾‾    |    ____/‾‾‾‾    |                 |                 |    ____/‾‾‾‾    |    ____/‾‾‾‾    |
             ⊙    ---------    ⊙    ---------    ⊙    --------- (2/3,5) ---------    ⊙    --------- (:0/5,5)---------    ⊙
    b        |                 |                 |    ____/‾‾‾‾    |    ____/‾‾‾‾    |                 |                 |
             ⊙    ---------    ⊙    ---------    ⊙    --------- (:1/3,4)---------    ⊙    ---------    ⊙    ---------    ⊙
    a        |    ____/‾‾‾‾    |    ____/‾‾‾‾    |                 |                 |    ____/‾‾‾‾    |    ____/‾‾‾‾    |
             ⊙    --------- (:2/1,3)--------- (1/2,3) ---------(2:2/3,3)--------- (:1/4,3)---------    ⊙    ---------    ⊙
    c        |                 |                 |                 |                 |                 |                 |
             ⊙    ---------    ⊙    --------- (0/2,2) --------- (1/3,2) ---------(2:2/4,2)---------    ⊙    ---------    ⊙
    a        |    ____/‾‾‾‾    |    ____/‾‾‾‾    |                 |                 |    ____/‾‾‾‾    |    ____/‾‾‾‾    |
             ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙
    a        |    ____/‾‾‾‾    |    ____/‾‾‾‾    |                 |                 |    ____/‾‾‾‾    |    ____/‾‾‾‾    |
             ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙    ---------    ⊙
                      a                 a                 b                 b                 a                 a

The algorithm stopped when it saw the backwards 2-path ending at (1,3) and the forwards 2-path ending at (3,5). The criterion
is a backwards path ending at (u,v) and a forward path ending at (x,y), where u <= x and the two points are on the same
diagonal. (Here the edgegraph has a diagonal, but the criterion is x-y=u-v.) Myers proves there is a forward
2-path from (0,0) to (1,3), and that together with the backwards 2-path ending at (1,3) gives the expected 4-path.
Unfortunately the forward path has to be constructed by another run of the forward algorithm; it can't be found from the
computed labels. That is the worst case. Had the code noticed (x,y)=(u,v)=(3,3) the whole path could be reconstructed
from the edgegraph. The implementation looks for a number of special cases to try to avoid computing an extra forward path.

If the two-sided algorithm has stop early (because D has become too large) it will have found a forward LCS and a
backwards LCS. Ideally these go with disjoint prefixes and suffixes of A and B, but disjointedness may fail and the two
computed LCS may conflict. (An easy example is where A is a suffix of B, and shares a short prefix. The backwards LCS
is all of A, and the forward LCS is a prefix of A.) The algorithm combines the two
to form a best-effort LCS. In the worst case the forward partial LCS may have to
be recomputed.
*/

/* Eugene Myers paper is titled
"An O(ND) Difference Algorithm and Its Variations"
and can be found at
http://www.xmailserver.org/diff2.pdf

(There is a generic implementation of the algorithm the repository with git hash
b9ad7e4ade3a686d608e44475390ad428e60e7fc)
*/
