// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// comment 0
package a

//BUG(uid): bug0

//TODO(uid): todo0

// A note with some spaces after it, should be ignored (watch out for
// emacs modes that remove trailing whitespace).
//NOTE(uid):

// SECBUG(uid): sec hole 0
// need to fix asap

// Multiple notes may be in the same comment group and should be
// recognized individually. Notes may start in the middle of a
// comment group as long as they start at the beginning of an
// individual comment.
//
// NOTE(foo): 1 of 4 - this is the first line of note 1
// - note 1 continues on this 2nd line
// - note 1 continues on this 3rd line
// NOTE(foo): 2 of 4
// NOTE(bar): 3 of 4
/* NOTE(bar): 4 of 4 */
// - this is the last line of note 4
//
//

// NOTE(bam): This note which contains a (parenthesized) subphrase
//            must appear in its entirety.

// NOTE(xxx) The ':' after the marker and uid is optional.

// NOTE(): NO uid - should not show up.
// NOTE()  NO uid - should not show up.
