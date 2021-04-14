// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"
)

func ExampleLookPath() {
	path, err := exec.LookPath("fortune")
	if err != nil {
		log.Fatal("installing fortune is in your future")
	}
	fmt.Printf("fortune is available at %s\n", path)
}

func ExampleCommand() {
	cmd := exec.Command("tr", "a-z", "A-Z")
	cmd.Stdin = strings.NewReader("some input")
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("in all caps: %q\n", out.String())
}

func ExampleCommand_environment() {
	cmd := exec.Command("prog")
	cmd.Env = append(os.Environ(),
		"FOO=duplicate_value", // ignored
		"FOO=actual_value",    // this value is used
	)
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}
}

func ExampleCmd_Output() {
	out, err := exec.Command("date").Output()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("The date is %s\n", out)
}

func ExampleCmd_Run() {
	cmd := exec.Command("sleep", "1")
	log.Printf("Running command and waiting for it to finish...")
	err := cmd.Run()
	log.Printf("Command finished with error: %v", err)
}

func ExampleCmd_Start() {
	cmd := exec.Command("sleep", "5")
	err := cmd.Start()
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Waiting for command to finish...")
	err = cmd.Wait()
	log.Printf("Command finished with error: %v", err)
}

func ExampleCmd_StdoutPipe() {
	cmd := exec.Command("echo", "-n", `{"Name": "Bob", "Age": 32}`)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	var person struct {
		Name string
		Age  int
	}
	if err := json.NewDecoder(stdout).Decode(&person); err != nil {
		log.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s is %d years old\n", person.Name, person.Age)
}

func ExampleCmd_StdinPipe() {
	cmd := exec.Command("cat")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		defer stdin.Close()
		io.WriteString(stdin, "values written to stdin are passed to cmd's standard input")
	}()

	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s\n", out)
}

func ExampleCmd_StderrPipe() {
	cmd := exec.Command("sh", "-c", "echo stdout; echo 1>&2 stderr")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}

	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}

	slurp, _ := io.ReadAll(stderr)
	fmt.Printf("%s\n", slurp)

	if err := cmd.Wait(); err != nil {
		log.Fatal(err)
	}
}

func ExampleCmd_CombinedOutput() {
	cmd := exec.Command("sh", "-c", "echo stdout; echo 1>&2 stderr")
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", stdoutStderr)
}

func ExampleCommandContext() {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	if err := exec.CommandContext(ctx, "sleep", "5").Run(); err != nil {
		// This will fail after 100 milliseconds. The 5 second sleep
		// will be interrupted.
	}
}
