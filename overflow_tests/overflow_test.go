package main

import (
	"testing"
)

func TestSafeOperations(t *testing.T) {
	// These should work fine
	var a int8 = 50
	var b int8 = 30
	result := a + b
	if result != 80 {
		t.Errorf("Expected 80, got %d", result)
	}
	
	var c int16 = 1000
	var d int16 = 500
	result16 := c + d
	if result16 != 1500 {
		t.Errorf("Expected 1500, got %d", result16)
	}
}

func TestOverflowPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for overflow, but didn't panic")
		}
	}()
	
	// This should panic
	var a int8 = 127
	var b int8 = 1
	_ = a + b  // Should panic with overflow
}

func TestSubtractionUnderflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for underflow, but didn't panic")
		}
	}()
	
	// This should panic
	var a int8 = -128
	var b int8 = 1
	_ = a - b  // Should panic with underflow
}

func TestMultiplicationOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for multiplication overflow, but didn't panic")
		}
	}()
	
	// This should panic
	var a int8 = 127
	var b int8 = 2
	_ = a * b  // Should panic with overflow
}