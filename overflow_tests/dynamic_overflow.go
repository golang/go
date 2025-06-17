package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: go run overflow.go <num1> <op> <num2>")
		fmt.Println("Ops: +, -, *, /, ^")
		os.Exit(1)
	}

	// Parse numbers as int32
	num1 := int32(parseInt(os.Args[1]))
	op := os.Args[2]
	num2 := int32(parseInt(os.Args[3]))

	var result int32
	switch op {
	case "+":
		result = num1 + num2
	case "-":
		result = num1 - num2
	case "*":
		result = num1 * num2
	case "/":
		if num2 == 0 {
			fmt.Println("Division by zero")
			os.Exit(1)
		}
		result = num1 / num2
	case "^":
		result = 1
		for i := int32(0); i < num2; i++ {
			result *= num1
		}
	default:
		fmt.Printf("Unknown op: %s\n", op)
		os.Exit(1)
	}

	fmt.Printf("%d %s %d = %d\n", num1, op, num2, result)
}

func parseInt(s string) int64 {
	n, err := strconv.ParseInt(s, 10, 32)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	return n
}