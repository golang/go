package main

import (
	"strings" // really ../vendor/strings
	"testing"
)

func TestMsgInternal(t *testing.T) {
	if strings.Msg != "hello, world" {
		t.Fatal("unexpected msg: %v", strings.Msg)
	}
}
