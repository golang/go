// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique_test

import (
	"fmt"
	"unique"
)

// Example demonstrates basic usage of unique handles for string interning.
func Example() {
	// Create handles for the same string value
	h1 := unique.Make("hello")
	h2 := unique.Make("hello")
	h3 := unique.Make("world")

	// Handles for equal values are equal
	fmt.Printf("h1 == h2: %v\n", h1 == h2)
	fmt.Printf("h1 == h3: %v\n", h1 == h3)

	// Retrieve the original value
	fmt.Printf("h1.Value(): %s\n", h1.Value())

	// Output:
	// h1 == h2: true
	// h1 == h3: false
	// h1.Value(): hello
}

// ExampleMake demonstrates creating unique handles for various types.
func ExampleMake() {
	// String handles
	s1 := unique.Make("interned")
	s2 := unique.Make("interned")
	fmt.Printf("String handles equal: %v\n", s1 == s2)

	// Integer handles
	i1 := unique.Make(42)
	i2 := unique.Make(42)
	fmt.Printf("Integer handles equal: %v\n", i1 == i2)

	// Struct handles
	type Point struct{ x, y int }
	p1 := unique.Make(Point{1, 2})
	p2 := unique.Make(Point{1, 2})
	fmt.Printf("Struct handles equal: %v\n", p1 == p2)

	// Output:
	// String handles equal: true
	// Integer handles equal: true
	// Struct handles equal: true
}

// ExampleHandle_Value demonstrates retrieving values from handles.
func ExampleHandle_Value() {
	h := unique.Make("canonical")
	value := h.Value()
	fmt.Printf("Value: %s\n", value)

	// The value is a copy, modifying it doesn't affect the handle
	type Data struct{ count int }
	hd := unique.Make(Data{count: 10})
	data := hd.Value()
	data.count = 20
	fmt.Printf("Original: %d, Modified: %d\n", hd.Value().count, data.count)

	// Output:
	// Value: canonical
	// Original: 10, Modified: 20
}

// Example_stringInterning demonstrates using unique for string interning
// to reduce memory usage when dealing with many duplicate strings.
func Example_stringInterning() {
	// Simulate reading duplicate strings from a data source
	strings := []string{"apple", "banana", "apple", "cherry", "banana", "apple"}

	// Intern the strings using unique handles
	handles := make([]unique.Handle[string], len(strings))
	for i, s := range strings {
		handles[i] = unique.Make(s)
	}

	// Count unique strings by comparing handles
	seen := make(map[unique.Handle[string]]bool)
	for _, h := range handles {
		seen[h] = true
	}

	fmt.Printf("Total strings: %d\n", len(strings))
	fmt.Printf("Unique strings: %d\n", len(seen))

	// Output:
	// Total strings: 6
	// Unique strings: 3
}

// Example_canonicalization demonstrates using unique handles to ensure
// only one instance of equal values exists in memory.
func Example_canonicalization() {
	type Config struct {
		host string
		port int
	}

	// Create multiple configs with the same values
	c1 := unique.Make(Config{"localhost", 8080})
	c2 := unique.Make(Config{"localhost", 8080})
	c3 := unique.Make(Config{"localhost", 9090})

	// Handles for equal configs are identical
	fmt.Printf("c1 == c2: %v\n", c1 == c2)
	fmt.Printf("c1 == c3: %v\n", c1 == c3)

	// Use handles as map keys for efficient lookups
	connections := make(map[unique.Handle[Config]]int)
	connections[c1] = 5
	connections[c2] += 3 // Same as c1
	connections[c3] = 2

	fmt.Printf("Connections for c1: %d\n", connections[c1])
	fmt.Printf("Connections for c3: %d\n", connections[c3])

	// Output:
	// c1 == c2: true
	// c1 == c3: false
	// Connections for c1: 8
	// Connections for c3: 2
}

// Example_deduplication demonstrates using unique handles to deduplicate
// data structures efficiently.
func Example_deduplication() {
	type Record struct {
		id   int
		name string
	}

	// Simulate records with duplicate names
	records := []Record{
		{1, "Alice"},
		{2, "Bob"},
		{3, "Alice"},
		{4, "Charlie"},
		{5, "Bob"},
	}

	// Create handles for names
	nameHandles := make(map[unique.Handle[string]][]int)
	for _, r := range records {
		h := unique.Make(r.name)
		nameHandles[h] = append(nameHandles[h], r.id)
	}

	// Print deduplicated names and their record IDs
	for h, ids := range nameHandles {
		fmt.Printf("%s: %v\n", h.Value(), ids)
	}

	// Output varies due to map iteration order, but will contain:
	// Alice: [1 3]
	// Bob: [2 5]
	// Charlie: [4]
}

// Example_comparison demonstrates efficient comparison using handles.
func Example_comparison() {
	// Create handles for complex values
	type LargeStruct struct {
		data [100]byte
		id   int
	}

	s1 := LargeStruct{id: 1}
	s1.data[0] = 'A'

	s2 := LargeStruct{id: 1}
	s2.data[0] = 'A'

	// Create handles
	h1 := unique.Make(s1)
	h2 := unique.Make(s2)

	// Handle comparison is much faster than struct comparison
	// for large structs
	fmt.Printf("Handles equal: %v\n", h1 == h2)
	fmt.Printf("Values equal: %v\n", h1.Value() == h2.Value())

	// Output:
	// Handles equal: true
	// Values equal: true
}

// Example_caching demonstrates using unique handles as cache keys.
func Example_caching() {
	type Query struct {
		table  string
		filter string
	}

	// Cache using handles as keys
	cache := make(map[unique.Handle[Query]]string)

	// Store results
	q1 := unique.Make(Query{"users", "active=true"})
	cache[q1] = "result1"

	// Retrieve with equivalent query
	q2 := unique.Make(Query{"users", "active=true"})
	if result, ok := cache[q2]; ok {
		fmt.Printf("Cache hit: %s\n", result)
	}

	// Different query
	q3 := unique.Make(Query{"users", "active=false"})
	if _, ok := cache[q3]; !ok {
		fmt.Println("Cache miss")
	}

	// Output:
	// Cache hit: result1
	// Cache miss
}

// Example_mapKey demonstrates using handles as efficient map keys.
func Example_mapKey() {
	type Coordinate struct {
		x, y, z float64
	}

	// Map using handles as keys
	locations := make(map[unique.Handle[Coordinate]]string)

	// Add locations
	home := unique.Make(Coordinate{0, 0, 0})
	work := unique.Make(Coordinate{10.5, 20.3, 0})

	locations[home] = "Home"
	locations[work] = "Work"

	// Lookup with equivalent coordinates
	lookup := unique.Make(Coordinate{0, 0, 0})
	fmt.Printf("Location: %s\n", locations[lookup])

	// Output:
	// Location: Home
}

// Example_zeroValue demonstrates that zero values work correctly.
func Example_zeroValue() {
	// Zero value handles
	h1 := unique.Make("")
	h2 := unique.Make("")
	fmt.Printf("Empty string handles equal: %v\n", h1 == h2)

	h3 := unique.Make(0)
	h4 := unique.Make(0)
	fmt.Printf("Zero int handles equal: %v\n", h3 == h4)

	type Empty struct{}
	h5 := unique.Make(Empty{})
	h6 := unique.Make(Empty{})
	fmt.Printf("Empty struct handles equal: %v\n", h5 == h6)

	// Output:
	// Empty string handles equal: true
	// Zero int handles equal: true
	// Empty struct handles equal: true
}
