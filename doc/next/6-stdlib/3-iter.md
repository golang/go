### Iterators

The new [`iter` package](/pkg/iter/) provides the basic definitions for
working with user-defined iterators.

The [`slices` package](/pkg/slices/) adds several functions that work
with iterators:
- [All](/pkg/slices#All) returns an iterator over slice indexes and values.
- [Values](/pkg/slices#Values) returns an iterator over slice elements.
- [Backward](/pkg/slices#Backward) returns an iterator that loops over
  a slice backward.
- [Collect](/pkg/slices#Collect) collects values from an iterator into
  a new slice.
- [AppendSeq](/pkg/slices#AppendSeq) appends values from an iterator to
  an existing slice.
- [Sorted](/pkg/slices#Sorted) collects values from an iterator into a
  new slice, and then sorts the slice.
- [SortedFunc](/pkg/slices#SortedFunc) is like `Sorted` but with a
  comparison function.
- [SortedStableFunc](/pkg/slices#SortedStableFunc) is like `SortFunc`
  but uses a stable sort algorithm.
- [Chunk](/pkg/slices#Chunk) returns an iterator over consecutive
  sub-slices of up to n elements of a slice.
