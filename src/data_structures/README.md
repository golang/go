# Data Structures in Golang

## Installation

`go get github.com/beyrakIn/data_structures`

## Using 
```go
package main

import (
	ds "github.com/beyrakIn/data_structures/linkedlist"
)

func main() {
	// Declare new LinkedList
	ll := ds.NewLinkedList()
	
	ll.Add(0)
	ll.Add(2)
	ll.Add(5)

	ll.PrintAll() // 2 3 5
}
    
```