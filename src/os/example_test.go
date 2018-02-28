// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"fmt"
	"log"
	"os"
	"time"
)

func ExampleOpenFile() {
	f, err := os.OpenFile("notes.txt", os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

func ExampleOpenFile_append() {
	// If the file doesn't exist, create it, or append to the file
	f, err := os.OpenFile("access.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	if _, err := f.Write([]byte("appended some data\n")); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

func ExampleChmod() {
	if err := os.Chmod("some-filename", 0644); err != nil {
		log.Fatal(err)
	}
}

func ExampleChtimes() {
	mtime := time.Date(2006, time.February, 1, 3, 4, 5, 0, time.UTC)
	atime := time.Date(2007, time.March, 2, 4, 5, 6, 0, time.UTC)
	if err := os.Chtimes("some-filename", atime, mtime); err != nil {
		log.Fatal(err)
	}
}

func ExampleFileMode() {
	fi, err := os.Lstat("some-filename")
	if err != nil {
		log.Fatal(err)
	}

	switch mode := fi.Mode(); {
	case mode.IsRegular():
		fmt.Println("regular file")
	case mode.IsDir():
		fmt.Println("directory")
	case mode&os.ModeSymlink != 0:
		fmt.Println("symbolic link")
	case mode&os.ModeNamedPipe != 0:
		fmt.Println("named pipe")
	}
}

func ExampleIsNotExist() {
	filename := "a-nonexistent-file"
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Printf("file does not exist")
	}
	// Output:
	// file does not exist
}

func init() {
	os.Setenv("USER", "gopher")
	os.Setenv("HOME", "/usr/gopher")
	os.Unsetenv("GOPATH")
}

func ExampleExpandEnv() {
	fmt.Println(os.ExpandEnv("$USER lives in ${HOME}."))

	// Output:
	// gopher lives in /usr/gopher.
}

func ExampleLookupEnv() {
	show := func(key string) {
		val, ok := os.LookupEnv(key)
		if !ok {
			fmt.Printf("%s not set\n", key)
		} else {
			fmt.Printf("%s=%s\n", key, val)
		}
	}

	show("USER")
	show("GOPATH")

	// Output:
	// USER=gopher
	// GOPATH not set
}

func ExampleGetenv() {
	fmt.Printf("%s lives in %s.\n", os.Getenv("USER"), os.Getenv("HOME"))

	// Output:
	// gopher lives in /usr/gopher.
}

func ExampleUnsetenv() {
	os.Setenv("TMPDIR", "/my/tmp")
	defer os.Unsetenv("TMPDIR")
}
