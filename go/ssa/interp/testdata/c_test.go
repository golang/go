package c_test

import (
	"os"
	"testing"
)

func TestC(t *testing.T) {
	println("TestC")
}

func TestMain(m *testing.M) {
	println("TestMain start")
	code := m.Run()
	println("TestMain end")
	os.Exit(code)
}
