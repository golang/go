package main_test

import (
	"strings" // really ../vendor/strings
	"testing"
)

func TestMsgExternal(t *testing.T) {
	if strings.Msg != "hello, world" {
		t.Fatalf("unexpected msg: %v", strings.Msg)
	}
}
