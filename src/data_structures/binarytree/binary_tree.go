package binarytree

import "fmt"

// BinaryTree struct
type BinaryTree struct {
	Root *Node
}

// Node struct
type Node struct {
	Value int
	Left  *Node
	Right *Node
}

// Insert function
func (bt *BinaryTree) Insert(value int) {
	if bt.Root == nil {
		bt.Root = &Node{Value: value}
		return
	} else {
		bt.Root.Insert(value)
	}
}

// Insert node to node
func (node *Node) Insert(value int) {
	if node == nil {
		return
	}

	if value < node.Value {
		if node.Left == nil {
			node.Left = &Node{Value: value}
		} else {
			node.Left.Insert(value)
		}
	} else {
		if node.Right == nil {
			node.Right = &Node{Value: value}
		} else {
			node.Right.Insert(value)
		}
	}
}

// PrintWithParent tree
func (node *Node) PrintWithParent() {
	if node == nil {
		return
	}

	node.Print()

	if node.Left != nil {
		fmt.Printf("%d L -> %d\n", node.Value, node.Left.Value)
		node.Left.PrintWithParent()
	}

	if node.Right != nil {
		fmt.Printf("%d R -> %d\n", node.Value, node.Right.Value)
		node.Right.PrintWithParent()
	}
}

// Print node
func (node *Node) Print() {
	if node == nil {
		return
	}

	fmt.Printf("%d\t", node.Value)
}
