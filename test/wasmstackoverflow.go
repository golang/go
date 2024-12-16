// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test ensure stackoverflow was caught correctly

package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
)

const errorCode = `
package main

func growStack(n int64) {
	if n > 0 {
		growStack(n - 1)
	}
}

func main() {
	growStack(998244353)
}
`

func main() {
	tmpDir, err := os.MkdirTemp("", "wasm-stackoverflow")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmpDir)

	if err := os.WriteFile(tmpDir+"/test.go", []byte(errorCode), 0666); err != nil {
		panic(err)
	}

	cmd := exec.Command("go", "build", "-o", tmpDir+"/test.wasm", tmpDir+"/test.go")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOOS=js", "GOARCH=wasm")
	if err := cmd.Run(); err != nil {
		panic(err)
	}

	//input, err := os.ReadFile(tmpDir + "/test.wasm")
	//if err != nil {
	//	panic(err)
	//}
	//
	//err = os.WriteFile("T:/wasm/test.wasm", input, 0666)
	//if err != nil {
	//	panic(err)
	//}

	node, err := exec.LookPath("node")
	if err != nil {
		// skip wasm stackoverflow test because node is not found
		return
	}

	// ensure ../lib/wasm/wasm_exec_node.js exist
	_, err = os.Stat("../lib/wasm/wasm_exec_node.js")
	if err != nil {
		panic(err)
	}
	exec_node := "../lib/wasm/wasm_exec_node.js"
	exec_node, err = filepath.Abs(exec_node)
	if err != nil {
		panic(err)
	}

	out := bytes.NewBuffer(nil)
	cmd = exec.Command(node, exec_node, tmpDir+"/test.wasm")
	cmd.Stdout = out
	cmd.Stderr = out
	if err := cmd.Run(); err == nil {
		panic("expected error, got nil")
	}

	if !bytes.Contains(out.Bytes(), []byte("Maximum call stack size exceeded")) {
		panic("expected stackoverflow error")
	}
}
