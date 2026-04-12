// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maphash

// A Hasher defines the interface between a hash-based container and its elements.
// It provides a hash function and an equivalence relation over values
// of type T, enabling those values to be inserted in hash tables
// and similar data structures.
//
// Of course, comparable types can already be used as keys of Go's
// built-in map type, but a Hasher enables non-comparable types to be
// used as keys of a suitable hash table too.
// Hashers may be useful even for comparable types, to define an
// equivalence relation that differs from the usual one (==), such as a
// field-based comparison for a pointer-to-struct type, or a
// case-insensitive comparison for strings, as in this example:
//
//	// CaseInsensitive is a Hasher[string] whose
//	// equivalence relation ignores letter case.
//	type CaseInsensitive struct{}
//
//	func (CaseInsensitive) Hash(h *Hash, s string) {
//		h.WriteString(strings.ToLower(s))
//	}
//
//	func (CaseInsensitive) Equal(x, y string) bool {
//		// (We avoid strings.EqualFold as it is not
//		// consistent with ToLower for all values.)
//		return strings.ToLower(x) == strings.ToLower(y)
//	}
//
// A Hasher also permits values to be used with other hash-based data
// structures such as a Bloom filter.
// The [ComparableHasher] type makes it convenient to enable comparable
// types to be used in such data structures under their usual (==)
// equivalence relation.
//
// # Hash invariants
//
// If two values are equal as defined by Equal(x, y), then they must
// have the same hash as defined by the effects of Hash(h, x) on h.
//
// Hashers must be logically stateless: the behavior of the Hash and
// Equal methods depends only on the arguments.
//
// # Writing a good function
//
// When defining a hash function and equivalence relation for a data
// type, it may help to first define a canonical encoding for values
// of that type as a sequence of elements, each being a number,
// string, boolean, or pointer.
// An encoding is canonical if two values that are logically equal
// have the same encoding, even if they are represented differently.
// For example, a canonical case-insensitive encoding of a string is
// [strings.ToLower].
//
// Once you have defined the encoding, the Hasher's Hash method should
// encode a value into the [Hash] using a sequence of calls to
// [Hash.Write] for byte slices, [Hash.WriteString] for strings,
// [Hash.WriteByte] for bytes, and [WriteComparable] for elements of
// other types. The Hasher's Equal method should compute the
// encodings of two values, then compare their corresponding
// elements, returning false at the first mismatch.
//
// A Hash method may discard information so long as it remains
// consistent with the Equal method as defined above.
// For example, valid implementations of CaseInsensitive.Hash might inspect
// only the first letter of the string, or even use a constant value.
// However, the lossier the hash function, the more frequent
// the hash collisions and the slower the hash table.
//
// Some data types, such as sets, are inherently unordered: the set
// {a, b, c} is equal to the set {c, b, a}.
// In some cases it is possible to define a canonical encoding for a
// set by sorting the elements into some order.
// In other cases this may inefficient, since it may require allocating
// memory, or infeasible, as when there is no convenient order.
// Another way to hash an unordered set is to compute the hash
// for each element separately, then combine all the element hashes
// using a commutative (order-independent) operator such as + or ^.
//
// The Hash method below, for a hypothetical Set type, illustrates
// this approach:
//
//	type Set[T comparable] struct{ ... }
//
//	type setHasher[T comparable] struct{}
//
//	func (setHasher[T]) Hash(hash *maphash.Hash, set *Set[T]) {
//		var accum uint64
//		for elem := range set.Elements() {
//			// Initialize a hasher for the element,
//			// using same seed as the outer hash.
//			var sub maphash.Hash
//			sub.SetSeed(hash.Seed())
//
//			// Hash the element.
//			maphash.WriteComparable(&sub, elem)
//
//			// Mix the element's hash into the set's hash.
//			accum ^= sub.Sum64()
//		}
//		maphash.WriteComparable(hash, accum)
//	}
//
// In many languages, a data type's hash operation simply returns an
// integer value.
// However, that makes it possible for an adversary to systematically
// construct a large number of values that all have the same hash,
// degrading the asymptotic performance of hash tables in a
// denial-of-service attack known as "hash flooding".
// By contrast, computing hashes as a sequence of values emitted into
// a [Hash] with an unpredictable [Seed] that varies from one hash
// table to another mitigates this attack.
//
// In effect, the Seed chooses one of 2⁶⁴ different hash functions.
// The code example above calls SetSeed on the element's sub-Hasher
// so that it uses the same hash function as for the Set itself, and
// not a random one.
type Hasher[T any] interface {
	Hash(*Hash, T)
	Equal(x, y T) bool
}

// ComparableHasher is an implementation of [Hasher] whose
// Equal(x, y) method is consistent with x == y.
//
// ComparableHasher is defined only for comparable types.
// The type system will not prevent you from instantiating a type
// such as ComparableHasher[any]; nonetheless you must not pass
// non-comparable argument values to its Hash or Equal methods.
type ComparableHasher[T comparable] struct {
	_ [0]func(T) // disallow comparison, and conversion between ComparableHasher[X] and ComparableHasher[Y]
}

func (ComparableHasher[T]) Hash(h *Hash, v T) { WriteComparable(h, v) }
func (ComparableHasher[T]) Equal(x, y T) bool { return x == y }
