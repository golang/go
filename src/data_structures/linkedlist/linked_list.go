package linkedlist

import "fmt"

//LinkedList is a struct
type LinkedList struct {
	head *Node
	tail *Node
}

//Node is a struct
type Node struct {
	value interface{}
	next  *Node
}

//NewLinkedList is a function
func NewLinkedList() *LinkedList {
	return &LinkedList{}
}

// Add is a function
func (ll *LinkedList) Add(value interface{}) {
	node := &Node{value: value}
	if ll.head == nil {
		ll.head = node
		ll.tail = node
	} else {
		ll.tail.next = node
		ll.tail = node
	}
}

// Remove is a function
func (ll *LinkedList) Remove(value interface{}) {
	node := ll.head
	if node.value == value {
		ll.head = node.next
		return
	}
	for node.next != nil {
		if node.next.value == value {
			node.next = node.next.next
			return
		}
		node = node.next
	}
}

// RemoveFirst is a function
func (ll *LinkedList) RemoveFirst() {
	ll.head = ll.head.next
}

// RemoveLast is a function
func (ll *LinkedList) RemoveLast() {
	node := ll.head
	for node.next != nil {
		node = node.next
	}
	node.next = nil
	ll.tail = node
}

// RemoveAt is a function
func (ll *LinkedList) RemoveAt(index int) {
	node := ll.head
	for index > 0 {
		node = node.next
		index--
	}
	ll.head = node.next
}

// RemoveAll is a function
func (ll *LinkedList) RemoveAll(value interface{}) {
	node := ll.head
	for node != nil {
		if node.value == value {
			node = node.next
			ll.head = node
		}
		node = node.next
	}
}

// Contains is a function
func (ll *LinkedList) Contains(value interface{}) bool {
	node := ll.head
	for node != nil {
		if node.value == value {
			return true
		}
		node = node.next
	}
	return false
}

// Length is a function
func (ll *LinkedList) Length() int {
	length := 0
	node := ll.head
	for node != nil {
		length++
		node = node.next
	}
	return length
}

// String is a function
func (ll *LinkedList) String() string {
	str := ""
	node := ll.head
	for node != nil {
		str += fmt.Sprintf("%v ", node.value)
		node = node.next
	}
	return str
}

// PrintAll is a function
func (ll *LinkedList) PrintAll() {
	node := ll.head
	for node != nil {
		fmt.Printf("%v ", node.value)
		node = node.next
	}
	fmt.Println()
}

// Reverse is a function
func (ll *LinkedList) Reverse() {
	node := ll.head
	var prev *Node
	for node != nil {
		temp := node.next
		node.next = prev
		prev = node
		node = temp
	}
	ll.head = prev
}

// IndexOf is a function
func (ll *LinkedList) IndexOf(value interface{}) int {
	index := 0
	node := ll.head
	for node != nil {
		if node.value == value {
			return index
		}
		index++
		node = node.next
	}
	return -1
}

// Clear is a function
func (ll *LinkedList) Clear() {
	ll.head = nil
	ll.tail = nil
}
