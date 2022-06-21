# Data Structures in Go

## Linked List
```go
- NewLinkedList()
  - Add(value)
  - Remove(value)
  - RemoveAll(value)
  - RemoveFirst()
  - RemoveLast()
  - RemoveAt(index)
  - Contains(value)
  - Length()
  - String()
  - Reverse()
  - PrintAll()
  - IndexOf(value)
  - Clear()

```
  

```go
// Initialize a new Linked List
ll := NewLinkedList()
```

```go
// Add(value) adds a new node with the given value to the end of the list
ll.Add(1)
ll.Add(2)
ll.Add(3)
// 1 -> 2 -> 3
``` 

```go
// Remove(value) removes the first node with the given value from the list
// 1 -> 2 -> 3
ll.Remove(2)
// 1 -> 3
```

```go
// RemoveAll() removes all given nodes from the linked list
// 1 -> 2- > 2 -> 3 -> 2
ll.RemoveAll(2)
// 1 -> 3
```

```go
// RemoveFirst() removes the first node from the linked list
// 1 -> 3
ll.RemoveFirst()
// 3
```

```go
//RemoveLast() removes the last node from the linked list
// 1 -> 2 -> 3
ll.RemoveLast()
// 1 -> 2
```

```go
// RemoveAt(index) removes the node at the given index from the linked list
// 1 -> 2 -> 3
ll.RemoveAt(1)
// 1 -> 3
```

```go
// Contains a value in the linked list
// 1 -> 2 -> 3
ll.Contains(2)
// true
ll.Contains(4)
// false
```

```go
// Length() returns the length of the linked list
// 1 -> 2 -> 3
ll.Length()
// 3
ll.Add(7)
ll.Length()
// 4
```

```go
// String() returns the linked list as a string
// 1 -> 2 -> 3
ll.String()
// "1 -> 2 -> 3"
```

```go
// Reverse() reverses the linked list
// 1 -> 2 -> 3
ll.Reverse()
// 3 -> 2 -> 1
```

```go
// PrintAll() prints all the values in the linked list
// 3 -> 2 -> 3
ll.PrintAll()
// 3 -> 2 -> 3
```

```go
// IndexOf() returns the index of the given value
// 1 -> 2 -> 3
ll.IndexOf(2)
// 1
ll.IndexOf(4)
// -1
```

```go
// Clear() removes all nodes from the linked list
// 1 -> 2 -> 3
ll.Clear()
// 0
```
