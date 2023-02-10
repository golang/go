# The Go Programming Language

Go is an open source programming language that makes it easy to build simple,
reliable, and efficient software.

![Gopher image](https://golang.org/doc/gopher/fiveyears.jpg)
*Gopher image by [Renee French][rf], licensed under [Creative Commons 4.0 Attributions license][cc4-by].*

Our canonical Git repository is located at https://go.googlesource.com/go.
There is a mirror of the repository at https://github.com/golang/go.

Unless otherwise noted, the Go source files are distributed under the
BSD-style license found in the LICENSE file.

### Introduction
Go is a compiled high-level programming language designed by Google. It's syntactically similar to C, but with memory safety, garbage collection, structural typing, and CSP-style concurrency. It's easy for users who know programming before to learn and use Go language to build software. By using Go language, you could create many wonderful software including the following but not limited to:
* Simple web server
* CRUD API
* AWS Lambda
* CRM Fiber
* Website

### Download and Install

#### Binary Distributions

Official binary distributions are available at https://go.dev/dl/.

After downloading a binary release, visit https://go.dev/doc/install
for installation instructions.

#### Install From Source

If a binary distribution is not available for your combination of
operating system and architecture, visit
https://go.dev/doc/install/source
for source installation instructions.

### Getting started
This simple example will bring you to go through the basic usage of Go if you are new to use Go.
#### Install Go
From the link provided above, you could choose the one that best fits your computer and environment.
After the download is complete, you'd be able to write some codes.
#### Hello World Message
1. create a empty folder on you computer and enter into this folder.
2. In this example, please run the following code to enable dependency tracking. For more detailed tracking methods, please visit https://go.dev/doc/modules/managing-dependencies#naming_module
```
go mod init example/hello
```
3. Open your own text editor and create a new file named <hello.go> in which to write your code.
4. Paste the following code to your recently created file and save it.
```
package main
import "fmt"
func main() {
    fmt.Println("Hello, World!")
}
```
5. Run the code and see the terminal.
```
go run .
```

#### Create a simple web server
1. Do the same previous 3 actions of hello world message.
2. import "fmt", "log", and "net/http" packages.
3. Paste the following code to your file.
```
func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hi there, I love %s!", r.URL.Path[1:])
}

func main() {
    http.HandleFunc("/", handler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```
4. Run the code by using the same run command with the hello world message.
5. Open a new broswer with the link http://localhost:8080, you will se the messages there.

### Contributing

Go is the work of thousands of contributors. We appreciate your help!

To contribute, please read the contribution guidelines at https://go.dev/doc/contribute.

Note that the Go project uses the issue tracker for bug reports and
proposals only. See https://go.dev/wiki/Questions for a list of
places to ask questions about the Go language.

[rf]: https://reneefrench.blogspot.com/
[cc4-by]: https://creativecommons.org/licenses/by/4.0/
