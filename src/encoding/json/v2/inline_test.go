// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// Whether a function is inlinable is dependent on the Go compiler version
// and also relies on the presence of the Go toolchain itself being installed.
// This test is disabled by default and explicitly enabled with an
// environment variable that is specified in our integration tests,
// which have fine control over exactly which Go version is being tested.
var testInline = os.Getenv("TEST_INLINE") != ""

func TestInline(t *testing.T) {
	if !testInline {
		t.SkipNow()
	}

	pkgs := map[string]map[string]bool{
		".": {
			"hash64":   true,
			"foldName": true, // thin wrapper over appendFoldedName
		},
		"./internal/jsonwire": {
			"ConsumeWhitespace":    true,
			"ConsumeNull":          true,
			"ConsumeFalse":         true,
			"ConsumeTrue":          true,
			"ConsumeSimpleString":  true,
			"ConsumeString":        true, // thin wrapper over consumeStringResumable
			"ConsumeSimpleNumber":  true,
			"ConsumeNumber":        true, // thin wrapper over consumeNumberResumable
			"UnquoteMayCopy":       true, // thin wrapper over unescapeString
			"HasSuffixByte":        true,
			"TrimSuffixByte":       true,
			"TrimSuffixString":     true,
			"TrimSuffixWhitespace": true,
		},
		"./jsontext": {
			"encoderState.NeedFlush":                  true,
			"Decoder.ReadToken":                       true, // thin wrapper over decoderState.ReadToken
			"Decoder.ReadValue":                       true, // thin wrapper over decoderState.ReadValue
			"Encoder.WriteToken":                      true, // thin wrapper over encoderState.WriteToken
			"Encoder.WriteValue":                      true, // thin wrapper over encoderState.WriteValue
			"decodeBuffer.needMore":                   true,
			"stateMachine.appendLiteral":              true,
			"stateMachine.appendNumber":               true,
			"stateMachine.appendString":               true,
			"stateMachine.Depth":                      true,
			"stateMachine.reset":                      true,
			"stateMachine.MayAppendDelim":             true,
			"stateMachine.needDelim":                  true,
			"stateMachine.popArray":                   true,
			"stateMachine.popObject":                  true,
			"stateMachine.pushArray":                  true,
			"stateMachine.pushObject":                 true,
			"stateEntry.Increment":                    true,
			"stateEntry.decrement":                    true,
			"stateEntry.isArray":                      true,
			"stateEntry.isObject":                     true,
			"stateEntry.Length":                       true,
			"stateEntry.needImplicitColon":            true,
			"stateEntry.needImplicitComma":            true,
			"stateEntry.NeedObjectName":               true,
			"stateEntry.needObjectValue":              true,
			"objectNameStack.reset":                   true,
			"objectNameStack.length":                  true,
			"objectNameStack.getUnquoted":             true,
			"objectNameStack.push":                    true,
			"objectNameStack.ReplaceLastQuotedOffset": true,
			"objectNameStack.replaceLastUnquotedName": true,
			"objectNameStack.pop":                     true,
			"objectNameStack.ensureCopiedBuffer":      true,
			"objectNamespace.insertQuoted":            true, // thin wrapper over objectNamespace.insert
			"objectNamespace.InsertUnquoted":          true, // thin wrapper over objectNamespace.insert
			"Token.String":                            true, // thin wrapper over Token.string
		},
	}

	for pkg, fncs := range pkgs {
		cmd := exec.Command("go", "build", "-gcflags=-m", pkg)
		b, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("exec.Command error: %v\n\n%s", err, b)
		}
		for _, line := range strings.Split(string(b), "\n") {
			const phrase = ": can inline "
			if i := strings.Index(line, phrase); i >= 0 {
				fnc := line[i+len(phrase):]
				fnc = strings.ReplaceAll(fnc, "(", "")
				fnc = strings.ReplaceAll(fnc, "*", "")
				fnc = strings.ReplaceAll(fnc, ")", "")
				delete(fncs, fnc)
			}
		}
		for fnc := range fncs {
			t.Errorf("%v is not inlinable, expected it to be", fnc)
		}
	}
}
