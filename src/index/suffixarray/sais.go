// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Suffix array construction by induced sorting (SAIS).
// See Ge Nong, Sen Zhang, and Wai Hong Chen,
// "Two Efficient Algorithms for Linear Time Suffix Array Construction",
// especially section 3 (https://ieeexplore.ieee.org/document/5582081).
// See also http://zork.net/~st/jottings/sais.html.
//
// With optimizations inspired by Yuta Mori's sais-lite
// (https://sites.google.com/site/yuta256/sais).
//
// And with other new optimizations.

// Many of these functions are parameterized by the sizes of
// the types they operate on. The generator gen.go makes
// copies of these functions for use with other sizes.
// Specifically:
//
// - A function with a name ending in _8_32 takes []byte and []int32 arguments
//   and is duplicated into _32_32, _8_64, and _64_64 forms.
//   The _32_32 and _64_64_ suffixes are shortened to plain _32 and _64.
//   Any lines in the function body that contain the text "byte-only" or "256"
//   are stripped when creating _32_32 and _64_64 forms.
//   (Those lines are typically 8-bit-specific optimizations.)
//
// - A function with a name ending only in _32 operates on []int32
//   and is duplicated into a _64 form. (Note that it may still take a []byte,
//   but there is no need for a version of the function in which the []byte
//   is widened to a full integer array.)

// The overall runtime of this code is linear in the input size:
// it runs a sequence of linear passes to reduce the problem to
// a subproblem at most half as big, invokes itself recursively,
// and then runs a sequence of linear passes to turn the answer
// for the subproblem into the answer for the original problem.
// This gives T(N) = O(N) + T(N/2) = O(N) + O(N/2) + O(N/4) + ... = O(N).
//
// The outline of the code, with the forward and backward scans
// through O(N)-sized arrays called out, is:
//
// sais_I_N
//	placeLMS_I_B
//		bucketMax_I_B
//			freq_I_B
//				<scan +text> (1)
//			<scan +freq> (2)
//		<scan -text, random bucket> (3)
//	induceSubL_I_B
//		bucketMin_I_B
//			freq_I_B
//				<scan +text, often optimized away> (4)
//			<scan +freq> (5)
//		<scan +sa, random text, random bucket> (6)
//	induceSubS_I_B
//		bucketMax_I_B
//			freq_I_B
//				<scan +text, often optimized away> (7)
//			<scan +freq> (8)
//		<scan -sa, random text, random bucket> (9)
//	assignID_I_B
//		<scan +sa, random text substrings> (10)
//	map_B
//		<scan -sa> (11)
//	recurse_B
//		(recursive call to sais_B_B for a subproblem of size at most 1/2 input, often much smaller)
//	unmap_I_B
//		<scan -text> (12)
//		<scan +sa> (13)
//	expand_I_B
//		bucketMax_I_B
//			freq_I_B
//				<scan +text, often optimized away> (14)
//			<scan +freq> (15)
//		<scan -sa, random text, random bucket> (16)
//	induceL_I_B
//		bucketMin_I_B
//			freq_I_B
//				<scan +text, often optimized away> (17)
//			<scan +freq> (18)
//		<scan +sa, random text, random bucket> (19)
//	induceS_I_B
//		bucketMax_I_B
//			freq_I_B
//				<scan +text, often optimized away> (20)
//			<scan +freq> (21)
//		<scan -sa, random text, random bucket> (22)
//
// Here, _B indicates the suffix array size (_32 or _64) and _I the input size (_8 or _B).
//
// The outline shows there are in general 22 scans through
// O(N)-sized arrays for a given level of the recursion.
// In the top level, operating on 8-bit input text,
// the six freq scans are fixed size (256) instead of potentially
// input-sized. Also, the frequency is counted once and cached
// whenever there is room to do so (there is nearly always room in general,
// and always room at the top level), which eliminates all but
// the first freq_I_B text scans (that is, 5 of the 6).
// So the top level of the recursion only does 22 - 6 - 5 = 11
// input-sized scans and a typical level does 16 scans.
//
// The linear scans do not cost anywhere near as much as
// the random accesses to the text made during a few of
// the scans (specifically #6, #9, #16, #19, #22 marked above).
// In real texts, there is not much but some locality to
// the accesses, due to the repetitive structure of the text
// (the same reason Burrows-Wheeler compression is so effective).
// For random inputs, there is no locality, which makes those
// accesses even more expensive, especially once the text
// no longer fits in cache.
// For example, running on 50 MB of Go source code, induceSubL_8_32
// (which runs only once, at the top level of the recursion)
// takes 0.44s, while on 50 MB of random input, it takes 2.55s.
// Nearly all the relative slowdown is explained by the text access:
//
//		c0, c1 := text[k-1], text[k]
//
// That line runs for 0.23s on the Go text and 2.02s on random text.

//go:generate go run gen.go

package suffixarray

// text_32 returns the suffix array for the input text.
// It requires that len(text) fit in an int32
// and that the caller zero sa.
func text_32(text []byte, sa []int32) {
	if int(int32(len(text))) != len(text) || len(text) != len(sa) {
		panic("suffixarray: misuse of text_32")
	}
	sais_8_32(text, 256, sa, make([]int32, 2*256))
}

// sais_8_32 computes the suffix array of text.
// The text must contain only values in [0, textMax).
// The suffix array is stored in sa, which the caller
// must ensure is already zeroed.
// The caller must also provide temporary space tmp
// with len(tmp) ≥ textMax. If len(tmp) ≥ 2*textMax
// then the algorithm runs a little faster.
// If sais_8_32 modifies tmp, it sets tmp[0] = -1 on return.
func sais_8_32(text []byte, textMax int, sa, tmp []int32) {
	if len(sa) != len(text) || len(tmp) < int(textMax) {
		panic("suffixarray: misuse of sais_8_32")
	}

	// Trivial base cases. Sorting 0 or 1 things is easy.
	if len(text) == 0 {
		return
	}
	if len(text) == 1 {
		sa[0] = 0
		return
	}

	// Establish slices indexed by text character
	// holding character frequency and bucket-sort offsets.
	// If there's only enough tmp for one slice,
	// we make it the bucket offsets and recompute
	// the character frequency each time we need it.
	var freq, bucket []int32
	if len(tmp) >= 2*textMax {
		freq, bucket = tmp[:textMax], tmp[textMax:2*textMax]
		freq[0] = -1 // mark as uninitialized
	} else {
		freq, bucket = nil, tmp[:textMax]
	}

	// The SAIS algorithm.
	// Each of these calls makes one scan through sa.
	// See the individual functions for documentation
	// about each's role in the algorithm.
	numLMS := placeLMS_8_32(text, sa, freq, bucket)
	if numLMS <= 1 {
		// 0 or 1 items are already sorted. Do nothing.
	} else {
		induceSubL_8_32(text, sa, freq, bucket)
		induceSubS_8_32(text, sa, freq, bucket)
		length_8_32(text, sa, numLMS)
		maxID := assignID_8_32(text, sa, numLMS)
		if maxID < numLMS {
			map_32(sa, numLMS)
			recurse_32(sa, tmp, numLMS, maxID)
			unmap_8_32(text, sa, numLMS)
		} else {
			// If maxID == numLMS, then each LMS-substring
			// is unique, so the relative ordering of two LMS-suffixes
			// is determined by just the leading LMS-substring.
			// That is, the LMS-suffix sort order matches the
			// (simpler) LMS-substring sort order.
			// Copy the original LMS-substring order into the
			// suffix array destination.
			copy(sa, sa[len(sa)-numLMS:])
		}
		expand_8_32(text, freq, bucket, sa, numLMS)
	}
	induceL_8_32(text, sa, freq, bucket)
	induceS_8_32(text, sa, freq, bucket)

	// Mark for caller that we overwrote tmp.
	tmp[0] = -1
}

// freq_8_32 returns the character frequencies
// for text, as a slice indexed by character value.
// If freq is nil, freq_8_32 uses and returns bucket.
// If freq is non-nil, freq_8_32 assumes that freq[0] >= 0
// means the frequencies are already computed.
// If the frequency data is overwritten or uninitialized,
// the caller must set freq[0] = -1 to force recomputation
// the next time it is needed.
func freq_8_32(text []byte, freq, bucket []int32) []int32 {
	if freq != nil && freq[0] >= 0 {
		return freq // already computed
	}
	if freq == nil {
		freq = bucket
	}

	freq = freq[:256] // eliminate bounds check for freq[c] below
	for i := range freq {
		freq[i] = 0
	}
	for _, c := range text {
		freq[c]++
	}
	return freq
}

// bucketMin_8_32 stores into bucket[c] the minimum index
// in the bucket for character c in a bucket-sort of text.
func bucketMin_8_32(text []byte, freq, bucket []int32) {
	freq = freq_8_32(text, freq, bucket)
	freq = freq[:256]     // establish len(freq) = 256, so 0 ≤ i < 256 below
	bucket = bucket[:256] // eliminate bounds check for bucket[i] below
	total := int32(0)
	for i, n := range freq {
		bucket[i] = total
		total += n
	}
}

// bucketMax_8_32 stores into bucket[c] the maximum index
// in the bucket for character c in a bucket-sort of text.
// The bucket indexes for c are [min, max).
// That is, max is one past the final index in that bucket.
func bucketMax_8_32(text []byte, freq, bucket []int32) {
	freq = freq_8_32(text, freq, bucket)
	freq = freq[:256]     // establish len(freq) = 256, so 0 ≤ i < 256 below
	bucket = bucket[:256] // eliminate bounds check for bucket[i] below
	total := int32(0)
	for i, n := range freq {
		total += n
		bucket[i] = total
	}
}

// The SAIS algorithm proceeds in a sequence of scans through sa.
// Each of the following functions implements one scan,
// and the functions appear here in the order they execute in the algorithm.

// placeLMS_8_32 places into sa the indexes of the
// final characters of the LMS substrings of text,
// sorted into the rightmost ends of their correct buckets
// in the suffix array.
//
// The imaginary sentinel character at the end of the text
// is the final character of the final LMS substring, but there
// is no bucket for the imaginary sentinel character,
// which has a smaller value than any real character.
// The caller must therefore pretend that sa[-1] == len(text).
//
// The text indexes of LMS-substring characters are always ≥ 1
// (the first LMS-substring must be preceded by one or more L-type
// characters that are not part of any LMS-substring),
// so using 0 as a “not present” suffix array entry is safe,
// both in this function and in most later functions
// (until induceL_8_32 below).
func placeLMS_8_32(text []byte, sa, freq, bucket []int32) int {
	bucketMax_8_32(text, freq, bucket)

	numLMS := 0
	lastB := int32(-1)
	bucket = bucket[:256] // eliminate bounds check for bucket[c1] below

	// The next stanza of code (until the blank line) loop backward
	// over text, stopping to execute a code body at each position i
	// such that text[i] is an L-character and text[i+1] is an S-character.
	// That is, i+1 is the position of the start of an LMS-substring.
	// These could be hoisted out into a function with a callback,
	// but at a significant speed cost. Instead, we just write these
	// seven lines a few times in this source file. The copies below
	// refer back to the pattern established by this original as the
	// "LMS-substring iterator".
	//
	// In every scan through the text, c0, c1 are successive characters of text.
	// In this backward scan, c0 == text[i] and c1 == text[i+1].
	// By scanning backward, we can keep track of whether the current
	// position is type-S or type-L according to the usual definition:
	//
	//	- position len(text) is type S with text[len(text)] == -1 (the sentinel)
	//	- position i is type S if text[i] < text[i+1], or if text[i] == text[i+1] && i+1 is type S.
	//	- position i is type L if text[i] > text[i+1], or if text[i] == text[i+1] && i+1 is type L.
	//
	// The backward scan lets us maintain the current type,
	// update it when we see c0 != c1, and otherwise leave it alone.
	// We want to identify all S positions with a preceding L.
	// Position len(text) is one such position by definition, but we have
	// nowhere to write it down, so we eliminate it by untruthfully
	// setting isTypeS = false at the start of the loop.
	c0, c1, isTypeS := byte(0), byte(0), false
	for i := len(text) - 1; i >= 0; i-- {
		c0, c1 = text[i], c0
		if c0 < c1 {
			isTypeS = true
		} else if c0 > c1 && isTypeS {
			isTypeS = false

			// Bucket the index i+1 for the start of an LMS-substring.
			b := bucket[c1] - 1
			bucket[c1] = b
			sa[b] = int32(i + 1)
			lastB = b
			numLMS++
		}
	}

	// We recorded the LMS-substring starts but really want the ends.
	// Luckily, with two differences, the start indexes and the end indexes are the same.
	// The first difference is that the rightmost LMS-substring's end index is len(text),
	// so the caller must pretend that sa[-1] == len(text), as noted above.
	// The second difference is that the first leftmost LMS-substring start index
	// does not end an earlier LMS-substring, so as an optimization we can omit
	// that leftmost LMS-substring start index (the last one we wrote).
	//
	// Exception: if numLMS <= 1, the caller is not going to bother with
	// the recursion at all and will treat the result as containing LMS-substring starts.
	// In that case, we don't remove the final entry.
	if numLMS > 1 {
		sa[lastB] = 0
	}
	return numLMS
}

// induceSubL_8_32 inserts the L-type text indexes of LMS-substrings
// into sa, assuming that the final characters of the LMS-substrings
// are already inserted into sa, sorted by final character, and at the
// right (not left) end of the corresponding character bucket.
// Each LMS-substring has the form (as a regexp) /S+L+S/:
// one or more S-type, one or more L-type, final S-type.
// induceSubL_8_32 leaves behind only the leftmost L-type text
// index for each LMS-substring. That is, it removes the final S-type
// indexes that are present on entry, and it inserts but then removes
// the interior L-type indexes too.
// (Only the leftmost L-type index is needed by induceSubS_8_32.)
func induceSubL_8_32(text []byte, sa, freq, bucket []int32) {
	// Initialize positions for left side of character buckets.
	bucketMin_8_32(text, freq, bucket)
	bucket = bucket[:256] // eliminate bounds check for bucket[cB] below

	// As we scan the array left-to-right, each sa[i] = j > 0 is a correctly
	// sorted suffix array entry (for text[j:]) for which we know that j-1 is type L.
	// Because j-1 is type L, inserting it into sa now will sort it correctly.
	// But we want to distinguish a j-1 with j-2 of type L from type S.
	// We can process the former but want to leave the latter for the caller.
	// We record the difference by negating j-1 if it is preceded by type S.
	// Either way, the insertion (into the text[j-1] bucket) is guaranteed to
	// happen at sa[i´] for some i´ > i, that is, in the portion of sa we have
	// yet to scan. A single pass therefore sees indexes j, j-1, j-2, j-3,
	// and so on, in sorted but not necessarily adjacent order, until it finds
	// one preceded by an index of type S, at which point it must stop.
	//
	// As we scan through the array, we clear the worked entries (sa[i] > 0) to zero,
	// and we flip sa[i] < 0 to -sa[i], so that the loop finishes with sa containing
	// only the indexes of the leftmost L-type indexes for each LMS-substring.
	//
	// The suffix array sa therefore serves simultaneously as input, output,
	// and a miraculously well-tailored work queue.

	// placeLMS_8_32 left out the implicit entry sa[-1] == len(text),
	// corresponding to the identified type-L index len(text)-1.
	// Process it before the left-to-right scan of sa proper.
	// See body in loop for commentary.
	k := len(text) - 1
	c0, c1 := text[k-1], text[k]
	if c0 < c1 {
		k = -k
	}

	// Cache recently used bucket index:
	// we're processing suffixes in sorted order
	// and accessing buckets indexed by the
	// byte before the sorted order, which still
	// has very good locality.
	// Invariant: b is cached, possibly dirty copy of bucket[cB].
	cB := c1
	b := bucket[cB]
	sa[b] = int32(k)
	b++

	for i := 0; i < len(sa); i++ {
		j := int(sa[i])
		if j == 0 {
			// Skip empty entry.
			continue
		}
		if j < 0 {
			// Leave discovered type-S index for caller.
			sa[i] = int32(-j)
			continue
		}
		sa[i] = 0

		// Index j was on work queue, meaning k := j-1 is L-type,
		// so we can now place k correctly into sa.
		// If k-1 is L-type, queue k for processing later in this loop.
		// If k-1 is S-type (text[k-1] < text[k]), queue -k to save for the caller.
		k := j - 1
		c0, c1 := text[k-1], text[k]
		if c0 < c1 {
			k = -k
		}

		if cB != c1 {
			bucket[cB] = b
			cB = c1
			b = bucket[cB]
		}
		sa[b] = int32(k)
		b++
	}
}

// induceSubS_8_32 inserts the S-type text indexes of LMS-substrings
// into sa, assuming that the leftmost L-type text indexes are already
// inserted into sa, sorted by LMS-substring suffix, and at the
// left end of the corresponding character bucket.
// Each LMS-substring has the form (as a regexp) /S+L+S/:
// one or more S-type, one or more L-type, final S-type.
// induceSubS_8_32 leaves behind only the leftmost S-type text
// index for each LMS-substring, in sorted order, at the right end of sa.
// That is, it removes the L-type indexes that are present on entry,
// and it inserts but then removes the interior S-type indexes too,
// leaving the LMS-substring start indexes packed into sa[len(sa)-numLMS:].
// (Only the LMS-substring start indexes are processed by the recursion.)
func induceSubS_8_32(text []byte, sa, freq, bucket []int32) {
	// Initialize positions for right side of character buckets.
	bucketMax_8_32(text, freq, bucket)
	bucket = bucket[:256] // eliminate bounds check for bucket[cB] below

	// Analogous to induceSubL_8_32 above,
	// as we scan the array right-to-left, each sa[i] = j > 0 is a correctly
	// sorted suffix array entry (for text[j:]) for which we know that j-1 is type S.
	// Because j-1 is type S, inserting it into sa now will sort it correctly.
	// But we want to distinguish a j-1 with j-2 of type S from type L.
	// We can process the former but want to leave the latter for the caller.
	// We record the difference by negating j-1 if it is preceded by type L.
	// Either way, the insertion (into the text[j-1] bucket) is guaranteed to
	// happen at sa[i´] for some i´ < i, that is, in the portion of sa we have
	// yet to scan. A single pass therefore sees indexes j, j-1, j-2, j-3,
	// and so on, in sorted but not necessarily adjacent order, until it finds
	// one preceded by an index of type L, at which point it must stop.
	// That index (preceded by one of type L) is an LMS-substring start.
	//
	// As we scan through the array, we clear the worked entries (sa[i] > 0) to zero,
	// and we flip sa[i] < 0 to -sa[i] and compact into the top of sa,
	// so that the loop finishes with the top of sa containing exactly
	// the LMS-substring start indexes, sorted by LMS-substring.

	// Cache recently used bucket index:
	cB := byte(0)
	b := bucket[cB]

	top := len(sa)
	for i := len(sa) - 1; i >= 0; i-- {
		j := int(sa[i])
		if j == 0 {
			// Skip empty entry.
			continue
		}
		sa[i] = 0
		if j < 0 {
			// Leave discovered LMS-substring start index for caller.
			top--
			sa[top] = int32(-j)
			continue
		}

		// Index j was on work queue, meaning k := j-1 is S-type,
		// so we can now place k correctly into sa.
		// If k-1 is S-type, queue k for processing later in this loop.
		// If k-1 is L-type (text[k-1] > text[k]), queue -k to save for the caller.
		k := j - 1
		c1 := text[k]
		c0 := text[k-1]
		if c0 > c1 {
			k = -k
		}

		if cB != c1 {
			bucket[cB] = b
			cB = c1
			b = bucket[cB]
		}
		b--
		sa[b] = int32(k)
	}
}

// length_8_32 computes and records the length of each LMS-substring in text.
// The length of the LMS-substring at index j is stored at sa[j/2],
// avoiding the LMS-substring indexes already stored in the top half of sa.
// (If index j is an LMS-substring start, then index j-1 is type L and cannot be.)
// There are two exceptions, made for optimizations in name_8_32 below.
//
// First, the final LMS-substring is recorded as having length 0, which is otherwise
// impossible, instead of giving it a length that includes the implicit sentinel.
// This ensures the final LMS-substring has length unequal to all others
// and therefore can be detected as different without text comparison
// (it is unequal because it is the only one that ends in the implicit sentinel,
// and the text comparison would be problematic since the implicit sentinel
// is not actually present at text[len(text)]).
//
// Second, to avoid text comparison entirely, if an LMS-substring is very short,
// sa[j/2] records its actual text instead of its length, so that if two such
// substrings have matching “length,” the text need not be read at all.
// The definition of “very short” is that the text bytes must pack into an uint32,
// and the unsigned encoding e must be ≥ len(text), so that it can be
// distinguished from a valid length.
func length_8_32(text []byte, sa []int32, numLMS int) {
	end := 0 // index of current LMS-substring end (0 indicates final LMS-substring)

	// The encoding of N text bytes into a “length” word
	// adds 1 to each byte, packs them into the bottom
	// N*8 bits of a word, and then bitwise inverts the result.
	// That is, the text sequence A B C (hex 41 42 43)
	// encodes as ^uint32(0x42_43_44).
	// LMS-substrings can never start or end with 0xFF.
	// Adding 1 ensures the encoded byte sequence never
	// starts or ends with 0x00, so that present bytes can be
	// distinguished from zero-padding in the top bits,
	// so the length need not be separately encoded.
	// Inverting the bytes increases the chance that a
	// 4-byte encoding will still be ≥ len(text).
	// In particular, if the first byte is ASCII (<= 0x7E, so +1 <= 0x7F)
	// then the high bit of the inversion will be set,
	// making it clearly not a valid length (it would be a negative one).
	//
	// cx holds the pre-inverted encoding (the packed incremented bytes).
	cx := uint32(0) // byte-only

	// This stanza (until the blank line) is the "LMS-substring iterator",
	// described in placeLMS_8_32 above, with one line added to maintain cx.
	c0, c1, isTypeS := byte(0), byte(0), false
	for i := len(text) - 1; i >= 0; i-- {
		c0, c1 = text[i], c0
		cx = cx<<8 | uint32(c1+1) // byte-only
		if c0 < c1 {
			isTypeS = true
		} else if c0 > c1 && isTypeS {
			isTypeS = false

			// Index j = i+1 is the start of an LMS-substring.
			// Compute length or encoded text to store in sa[j/2].
			j := i + 1
			var code int32
			if end == 0 {
				code = 0
			} else {
				code = int32(end - j)
				if code <= 32/8 && ^cx >= uint32(len(text)) { // byte-only
					code = int32(^cx) // byte-only
				} // byte-only
			}
			sa[j>>1] = code
			end = j + 1
			cx = uint32(c1 + 1) // byte-only
		}
	}
}

// assignID_8_32 assigns a dense ID numbering to the
// set of LMS-substrings respecting string ordering and equality,
// returning the maximum assigned ID.
// For example given the input "ababab", the LMS-substrings
// are "aba", "aba", and "ab", renumbered as 2 2 1.
// sa[len(sa)-numLMS:] holds the LMS-substring indexes
// sorted in string order, so to assign numbers we can
// consider each in turn, removing adjacent duplicates.
// The new ID for the LMS-substring at index j is written to sa[j/2],
// overwriting the length previously stored there (by length_8_32 above).
func assignID_8_32(text []byte, sa []int32, numLMS int) int {
	id := 0
	lastLen := int32(-1) // impossible
	lastPos := int32(0)
	for _, j := range sa[len(sa)-numLMS:] {
		// Is the LMS-substring at index j new, or is it the same as the last one we saw?
		n := sa[j/2]
		if n != lastLen {
			goto New
		}
		if uint32(n) >= uint32(len(text)) {
			// “Length” is really encoded full text, and they match.
			goto Same
		}
		{
			// Compare actual texts.
			n := int(n)
			this := text[j:][:n]
			last := text[lastPos:][:n]
			for i := 0; i < n; i++ {
				if this[i] != last[i] {
					goto New
				}
			}
			goto Same
		}
	New:
		id++
		lastPos = j
		lastLen = n
	Same:
		sa[j/2] = int32(id)
	}
	return id
}

// map_32 maps the LMS-substrings in text to their new IDs,
// producing the subproblem for the recursion.
// The mapping itself was mostly applied by assignID_8_32:
// sa[i] is either 0, the ID for the LMS-substring at index 2*i,
// or the ID for the LMS-substring at index 2*i+1.
// To produce the subproblem we need only remove the zeros
// and change ID into ID-1 (our IDs start at 1, but text chars start at 0).
//
// map_32 packs the result, which is the input to the recursion,
// into the top of sa, so that the recursion result can be stored
// in the bottom of sa, which sets up for expand_8_32 well.
func map_32(sa []int32, numLMS int) {
	w := len(sa)
	for i := len(sa) / 2; i >= 0; i-- {
		j := sa[i]
		if j > 0 {
			w--
			sa[w] = j - 1
		}
	}
}

// recurse_32 calls sais_32 recursively to solve the subproblem we've built.
// The subproblem is at the right end of sa, the suffix array result will be
// written at the left end of sa, and the middle of sa is available for use as
// temporary frequency and bucket storage.
func recurse_32(sa, oldTmp []int32, numLMS, maxID int) {
	dst, saTmp, text := sa[:numLMS], sa[numLMS:len(sa)-numLMS], sa[len(sa)-numLMS:]

	// Set up temporary space for recursive call.
	// We must pass sais_32 a tmp buffer with at least maxID entries.
	//
	// The subproblem is guaranteed to have length at most len(sa)/2,
	// so that sa can hold both the subproblem and its suffix array.
	// Nearly all the time, however, the subproblem has length < len(sa)/3,
	// in which case there is a subproblem-sized middle of sa that
	// we can reuse for temporary space (saTmp).
	// When recurse_32 is called from sais_8_32, oldTmp is length 512
	// (from text_32), and saTmp will typically be much larger, so we'll use saTmp.
	// When deeper recursions come back to recurse_32, now oldTmp is
	// the saTmp from the top-most recursion, it is typically larger than
	// the current saTmp (because the current sa gets smaller and smaller
	// as the recursion gets deeper), and we keep reusing that top-most
	// large saTmp instead of the offered smaller ones.
	//
	// Why is the subproblem length so often just under len(sa)/3?
	// See Nong, Zhang, and Chen, section 3.6 for a plausible explanation.
	// In brief, the len(sa)/2 case would correspond to an SLSLSLSLSLSL pattern
	// in the input, perfect alternation of larger and smaller input bytes.
	// Real text doesn't do that. If each L-type index is randomly followed
	// by either an L-type or S-type index, then half the substrings will
	// be of the form SLS, but the other half will be longer. Of that half,
	// half (a quarter overall) will be SLLS; an eighth will be SLLLS, and so on.
	// Not counting the final S in each (which overlaps the first S in the next),
	// This works out to an average length 2×½ + 3×¼ + 4×⅛ + ... = 3.
	// The space we need is further reduced by the fact that many of the
	// short patterns like SLS will often be the same character sequences
	// repeated throughout the text, reducing maxID relative to numLMS.
	//
	// For short inputs, the averages may not run in our favor, but then we
	// can often fall back to using the length-512 tmp available in the
	// top-most call. (Also a short allocation would not be a big deal.)
	//
	// For pathological inputs, we fall back to allocating a new tmp of length
	// max(maxID, numLMS/2). This level of the recursion needs maxID,
	// and all deeper levels of the recursion will need no more than numLMS/2,
	// so this one allocation is guaranteed to suffice for the entire stack
	// of recursive calls.
	tmp := oldTmp
	if len(tmp) < len(saTmp) {
		tmp = saTmp
	}
	if len(tmp) < numLMS {
		// TestSAIS/forcealloc reaches this code.
		n := maxID
		if n < numLMS/2 {
			n = numLMS / 2
		}
		tmp = make([]int32, n)
	}

	// sais_32 requires that the caller arrange to clear dst,
	// because in general the caller may know dst is
	// freshly-allocated and already cleared. But this one is not.
	for i := range dst {
		dst[i] = 0
	}
	sais_32(text, maxID, dst, tmp)
}

// unmap_8_32 unmaps the subproblem back to the original.
// sa[:numLMS] is the LMS-substring numbers, which don't matter much anymore.
// sa[len(sa)-numLMS:] is the sorted list of those LMS-substring numbers.
// The key part is that if the list says K that means the K'th substring.
// We can replace sa[:numLMS] with the indexes of the LMS-substrings.
// Then if the list says K it really means sa[K].
// Having mapped the list back to LMS-substring indexes,
// we can place those into the right buckets.
func unmap_8_32(text []byte, sa []int32, numLMS int) {
	unmap := sa[len(sa)-numLMS:]
	j := len(unmap)

	// "LMS-substring iterator" (see placeLMS_8_32 above).
	c0, c1, isTypeS := byte(0), byte(0), false
	for i := len(text) - 1; i >= 0; i-- {
		c0, c1 = text[i], c0
		if c0 < c1 {
			isTypeS = true
		} else if c0 > c1 && isTypeS {
			isTypeS = false

			// Populate inverse map.
			j--
			unmap[j] = int32(i + 1)
		}
	}

	// Apply inverse map to subproblem suffix array.
	sa = sa[:numLMS]
	for i := 0; i < len(sa); i++ {
		sa[i] = unmap[sa[i]]
	}
}

// expand_8_32 distributes the compacted, sorted LMS-suffix indexes
// from sa[:numLMS] into the tops of the appropriate buckets in sa,
// preserving the sorted order and making room for the L-type indexes
// to be slotted into the sorted sequence by induceL_8_32.
func expand_8_32(text []byte, freq, bucket, sa []int32, numLMS int) {
	bucketMax_8_32(text, freq, bucket)
	bucket = bucket[:256] // eliminate bound check for bucket[c] below

	// Loop backward through sa, always tracking
	// the next index to populate from sa[:numLMS].
	// When we get to one, populate it.
	// Zero the rest of the slots; they have dead values in them.
	x := numLMS - 1
	saX := sa[x]
	c := text[saX]
	b := bucket[c] - 1
	bucket[c] = b

	for i := len(sa) - 1; i >= 0; i-- {
		if i != int(b) {
			sa[i] = 0
			continue
		}
		sa[i] = saX

		// Load next entry to put down (if any).
		if x > 0 {
			x--
			saX = sa[x] // TODO bounds check
			c = text[saX]
			b = bucket[c] - 1
			bucket[c] = b
		}
	}
}

// induceL_8_32 inserts L-type text indexes into sa,
// assuming that the leftmost S-type indexes are inserted
// into sa, in sorted order, in the right bucket halves.
// It leaves all the L-type indexes in sa, but the
// leftmost L-type indexes are negated, to mark them
// for processing by induceS_8_32.
func induceL_8_32(text []byte, sa, freq, bucket []int32) {
	// Initialize positions for left side of character buckets.
	bucketMin_8_32(text, freq, bucket)
	bucket = bucket[:256] // eliminate bounds check for bucket[cB] below

	// This scan is similar to the one in induceSubL_8_32 above.
	// That one arranges to clear all but the leftmost L-type indexes.
	// This scan leaves all the L-type indexes and the original S-type
	// indexes, but it negates the positive leftmost L-type indexes
	// (the ones that induceS_8_32 needs to process).

	// expand_8_32 left out the implicit entry sa[-1] == len(text),
	// corresponding to the identified type-L index len(text)-1.
	// Process it before the left-to-right scan of sa proper.
	// See body in loop for commentary.
	k := len(text) - 1
	c0, c1 := text[k-1], text[k]
	if c0 < c1 {
		k = -k
	}

	// Cache recently used bucket index.
	cB := c1
	b := bucket[cB]
	sa[b] = int32(k)
	b++

	for i := 0; i < len(sa); i++ {
		j := int(sa[i])
		if j <= 0 {
			// Skip empty or negated entry (including negated zero).
			continue
		}

		// Index j was on work queue, meaning k := j-1 is L-type,
		// so we can now place k correctly into sa.
		// If k-1 is L-type, queue k for processing later in this loop.
		// If k-1 is S-type (text[k-1] < text[k]), queue -k to save for the caller.
		// If k is zero, k-1 doesn't exist, so we only need to leave it
		// for the caller. The caller can't tell the difference between
		// an empty slot and a non-empty zero, but there's no need
		// to distinguish them anyway: the final suffix array will end up
		// with one zero somewhere, and that will be a real zero.
		k := j - 1
		c1 := text[k]
		if k > 0 {
			if c0 := text[k-1]; c0 < c1 {
				k = -k
			}
		}

		if cB != c1 {
			bucket[cB] = b
			cB = c1
			b = bucket[cB]
		}
		sa[b] = int32(k)
		b++
	}
}

func induceS_8_32(text []byte, sa, freq, bucket []int32) {
	// Initialize positions for right side of character buckets.
	bucketMax_8_32(text, freq, bucket)
	bucket = bucket[:256] // eliminate bounds check for bucket[cB] below

	cB := byte(0)
	b := bucket[cB]

	for i := len(sa) - 1; i >= 0; i-- {
		j := int(sa[i])
		if j >= 0 {
			// Skip non-flagged entry.
			// (This loop can't see an empty entry; 0 means the real zero index.)
			continue
		}

		// Negative j is a work queue entry; rewrite to positive j for final suffix array.
		j = -j
		sa[i] = int32(j)

		// Index j was on work queue (encoded as -j but now decoded),
		// meaning k := j-1 is L-type,
		// so we can now place k correctly into sa.
		// If k-1 is S-type, queue -k for processing later in this loop.
		// If k-1 is L-type (text[k-1] > text[k]), queue k to save for the caller.
		// If k is zero, k-1 doesn't exist, so we only need to leave it
		// for the caller.
		k := j - 1
		c1 := text[k]
		if k > 0 {
			if c0 := text[k-1]; c0 <= c1 {
				k = -k
			}
		}

		if cB != c1 {
			bucket[cB] = b
			cB = c1
			b = bucket[cB]
		}
		b--
		sa[b] = int32(k)
	}
}
