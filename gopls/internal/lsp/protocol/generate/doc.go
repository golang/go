// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

/*
GenLSP  generates the files tsprotocol.go, tsclient.go,
tsserver.go, tsjson.go that support the language server protocol
for gopls.

Usage:

	go run . [flags]

The flags are:

	-d <directory name>
		The directory containing the vscode-languageserver-node repository.
		(git clone https://github.com/microsoft/vscode-languageserver-node.git).
		If not specified, the default is $HOME/vscode-languageserver-node.

	-o <directory name>
		The directory to write the generated files to. It must exist.
		The default is "gen".

	-c <directory name>
		Compare the generated files to the files in the specified directory.
		If this flag is not specified, no comparison is done.
*/
package main
